"""Contiguous sub-epoch 1D-CNN + BiGRU waveform sleep stager and resumable trainer."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn import functional as F

from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET
from physionet_sleep.experiments.waveform_clock import active_clock
from physionet_sleep.experiments.waveform_data import CHANNELS, EPOCH_SAMPLES
from physionet_sleep.experiments.waveform_training import available_memory_bytes


N_CLASSES = 5
DECIMATION = 2
STEM_STRIDE = 5
BLOCK1_STRIDE = 4
BLOCK2_STRIDE = 3
TOTAL_CONV_STRIDE = DECIMATION * STEM_STRIDE * BLOCK1_STRIDE * BLOCK2_STRIDE  # 120 samples -> 50 steps/30s
TOKENS_PER_EPOCH = 5  # 6-second tokens per 30s epoch
WIDTH = 64

MAX_RSS_BYTES = int(2.2 * 2**30)
MIN_AVAILABLE_BYTES = int(1.0 * 2**30)
CHECKPOINT_INTERVAL_STEPS = 250
CHECKPOINT_INTERVAL_SECONDS = 300
VALIDATION_INTERVAL = 500
MEMORY_SAMPLE_INTERVAL_SECONDS = 5


class _ResidualBlock(nn.Module):
    """Pre-act-free 1D residual conv block with GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(4, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.layers(x) + self.skip(x))


class SubEpochWaveformNet(nn.Module):
    """Multi-scale 100 Hz conv stem (5x 6s tokens/epoch) + 2-layer BiGRU stager."""

    def __init__(self, n_channels: int = len(CHANNELS), width: int = WIDTH, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_channels = n_channels
        self.width = width
        self.n_classes = n_classes

        # Multi-scale stride-5 stem operating on 100 Hz anti-aliased waveforms (3,000 -> 600 steps).
        self.stem_fast = nn.Conv1d(n_channels, 16, kernel_size=7, stride=STEM_STRIDE, padding=3, bias=False)
        self.stem_slow = nn.Conv1d(n_channels, 16, kernel_size=25, stride=STEM_STRIDE, padding=12, bias=False)
        self.lead_embedding = nn.Linear(n_channels, 32, bias=False)
        self.stem_norm = nn.GroupNorm(4, 32)

        self.encoder = nn.Sequential(
            _ResidualBlock(32, 48, stride=BLOCK1_STRIDE, kernel_size=5),
            _ResidualBlock(48, width, stride=BLOCK2_STRIDE, kernel_size=5),
            _ResidualBlock(width, width, stride=1, kernel_size=5),
        )

        self.gru = nn.GRU(
            input_size=width,
            hidden_size=width,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        gru_dim = width * 2
        self.subepoch_attn = nn.Linear(gru_dim, 1)
        self.stem_residual = nn.Linear(width, gru_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Dropout(0.1),
            nn.Linear(gru_dim, n_classes),
        )
        self.local_classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, n_classes),
        )

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_subepochs(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        channels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `(B, T, C, L)` waveforms into `(B, T, 5, width)` 6-second tokens and validity mask."""
        batch, n_epochs, n_ch, n_samples = x.shape
        unit = TOTAL_CONV_STRIDE * TOKENS_PER_EPOCH
        if n_samples < unit or n_samples % unit != 0:
            raise ValueError(f"Expected epoch length divisible by {unit}, got {n_samples}")

        flat_n = batch * n_epochs
        x_flat = x.reshape(flat_n, n_ch, n_samples)
        valid_flat = valid.reshape(flat_n, n_samples)
        channels_flat = channels.reshape(flat_n, n_ch)

        visible = valid_flat[:, None, :] & channels_flat[:, :, None]
        x_flat = x_flat.masked_fill(~visible, 0.0)

        # 2x anti-aliased decimation (200 Hz -> 100 Hz)
        x_100 = F.avg_pool1d(x_flat, kernel_size=DECIMATION, stride=DECIMATION)

        stem = torch.cat((self.stem_fast(x_100), self.stem_slow(x_100)), dim=1)
        stem = stem + self.lead_embedding(channels_flat.to(dtype=stem.dtype))[:, :, None]
        stem = F.gelu(self.stem_norm(stem))
        feat = self.encoder(stem)

        n_steps = n_samples // TOTAL_CONV_STRIDE
        steps_per_token = n_steps // TOKENS_PER_EPOCH
        step_valid = valid_flat.reshape(flat_n, n_steps, TOTAL_CONV_STRIDE).any(dim=-1)
        weights = step_valid.reshape(flat_n, 1, TOKENS_PER_EPOCH, steps_per_token).to(dtype=feat.dtype)
        grouped = feat.reshape(flat_n, self.width, TOKENS_PER_EPOCH, steps_per_token)
        tokens = (grouped * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
        token_valid = (weights.sum(dim=-1).squeeze(1) > 0) & channels_flat.any(dim=-1, keepdim=True)
        tokens = tokens.transpose(1, 2) * token_valid[:, :, None].to(dtype=tokens.dtype)
        return (
            tokens.reshape(batch, n_epochs, TOKENS_PER_EPOCH, self.width),
            token_valid.reshape(batch, n_epochs, TOKENS_PER_EPOCH),
        )

    def classify_subepochs(
        self,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BiGRU over `(B, T * 5, width)` tokens and return `(logits, local_logits)` of shape `(B, T, 5)`."""
        batch, n_epochs, n_tok, width = tokens.shape
        tv_f = token_valid.to(dtype=tokens.dtype)
        epoch_conv = (tokens * tv_f[..., None]).sum(dim=2) / tv_f.sum(dim=2, keepdim=True).clamp(min=1.0)

        seq = tokens.reshape(batch, n_epochs * n_tok, width)
        hidden, _ = self.gru(seq)
        hidden = hidden.reshape(batch, n_epochs, n_tok, width * 2)

        attn_scores = self.subepoch_attn(hidden).squeeze(-1).masked_fill(~token_valid, -1e9)
        attn_weights = F.softmax(attn_scores, dim=2) * tv_f
        attn_weights = attn_weights / attn_weights.sum(dim=2, keepdim=True).clamp(min=1e-6)

        h_attn = (hidden * attn_weights[..., None]).sum(dim=2)
        h_mean = (hidden * tv_f[..., None]).sum(dim=2) / tv_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        fused = h_mean + h_attn + self.stem_residual(epoch_conv)
        return self.classifier(fused), self.local_classifier(epoch_conv)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        channels: torch.Tensor,
        *,
        return_local: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens, token_valid = self.encode_subepochs(x, valid, channels)
        logits, local_logits = self.classify_subepochs(tokens, token_valid)
        return (logits, local_logits) if return_local else logits


@dataclass(frozen=True)
class _SubjectNight:
    subject_id: str
    min_epoch: int
    dense_labels: np.ndarray
    scored_offsets: np.ndarray
    row_indices: np.ndarray
    row_epochs: np.ndarray


@dataclass
class ContiguousCohort:
    """Validated per-subject contiguous epoch index with fingerprint for safe resuming."""

    nights: tuple[_SubjectNight, ...]
    n_rows: int
    labels: np.ndarray
    sample_weights: np.ndarray
    fingerprint: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> ContiguousCohort:
        data = frame[[SUBJ, "epoch_index", TARGET]].reset_index(drop=True)
        if data.empty or data.isna().any().any() or data.duplicated([SUBJ, "epoch_index"]).any():
            raise ValueError("Waveform splits require unique, complete patient/epoch rows")
        epochs = data["epoch_index"].to_numpy(dtype=np.int64)
        labels = data[TARGET].to_numpy(dtype=np.int64)
        if (
            (epochs < 0).any()
            or not np.array_equal(epochs, data["epoch_index"])
            or not np.array_equal(labels, data[TARGET])
            or not np.isin(labels, range(N_CLASSES)).all()
        ):
            raise ValueError("Waveform splits require nonnegative epochs and valid integer stage codes")

        subjects = data[SUBJ].astype(str)
        hashed = pd.util.hash_pandas_object(data, index=False).to_numpy()
        fingerprint = hashlib.sha256(hashed.tobytes()).hexdigest()

        nights: list[_SubjectNight] = []
        for subject, rows in subjects.groupby(subjects, sort=False).indices.items():
            sub_epochs = epochs[rows]
            sub_labels = labels[rows]
            min_ep = int(sub_epochs.min())
            max_ep = int(sub_epochs.max())
            span = max_ep - min_ep + 1
            dense = np.full(span, -1, dtype=np.int64)
            offsets = sub_epochs - min_ep
            dense[offsets] = sub_labels
            nights.append(
                _SubjectNight(
                    subject_id=str(subject),
                    min_epoch=min_ep,
                    dense_labels=dense,
                    scored_offsets=np.sort(offsets),
                    row_indices=rows,
                    row_epochs=sub_epochs,
                )
            )
        counts = np.asarray([len(n.scored_offsets) for n in nights], dtype=np.float64)
        return cls(
            nights=tuple(nights),
            n_rows=len(data),
            labels=labels,
            sample_weights=counts / counts.sum(),
            fingerprint=fingerprint,
        )

    @property
    def subject_ids(self) -> set[str]:
        return {n.subject_id for n in self.nights}


def process_memory_bytes() -> tuple[int, int]:
    """Return `(current_rss_bytes, peak_rss_bytes)` for the running process."""
    if sys.platform == "win32":
        import ctypes.wintypes as w

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("size", w.DWORD),
                ("fault_count", w.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("remaining", ctypes.c_size_t * 6),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = w.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [w.HANDLE, ctypes.POINTER(_Counters), w.DWORD]
        query.restype = w.BOOL
        counters = _Counters()
        counters.size = ctypes.sizeof(counters)
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.working_set), int(counters.peak_working_set)
    if sys.platform.startswith("linux"):
        rss = 0
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return rss, peak
    raise OSError(f"Process RSS measurement unavailable on {sys.platform}")


def tempered_class_weights(labels: np.ndarray, *, power: float = -0.50) -> tuple[np.ndarray, torch.Tensor]:
    """Compute stage counts and normalized tempered weights `counts ** power`."""
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    if (counts <= 0).any():
        raise ValueError(f"Every stage 0..{N_CLASSES - 1} must appear in the training split")
    weights = np.power(counts, power)
    weights = (weights / weights.mean()).astype(np.float32)
    return counts.astype(np.int64), torch.from_numpy(weights)


def sample_contiguous_batch(
    store,
    cohort: ContiguousCohort,
    rng: np.random.Generator,
    *,
    batch_size: int,
    chunk_epochs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `batch_size` contiguous `chunk_epochs` blocks, each containing >= 1 scored epoch."""
    chosen = rng.choice(len(cohort.nights), size=batch_size, replace=True, p=cohort.sample_weights)
    x_batch = np.zeros((batch_size, chunk_epochs, len(CHANNELS), EPOCH_SAMPLES), dtype=np.float32)
    v_batch = np.zeros((batch_size, chunk_epochs, EPOCH_SAMPLES), dtype=bool)
    c_batch = np.zeros((batch_size, chunk_epochs, len(CHANNELS)), dtype=bool)
    y_batch = np.full((batch_size, chunk_epochs), -1, dtype=np.int64)

    for b, night_idx in enumerate(chosen):
        night = cohort.nights[int(night_idx)]
        span = len(night.dense_labels)
        anchor = int(night.scored_offsets[int(rng.integers(len(night.scored_offsets)))])
        if span <= chunk_epochs:
            start_offset = 0
            length = span
        else:
            lo = max(0, anchor - (chunk_epochs - 1))
            hi = min(anchor, span - chunk_epochs)
            start_offset = int(rng.integers(lo, hi + 1))
            length = chunk_epochs
        epochs_range = range(night.min_epoch + start_offset, night.min_epoch + start_offset + length)
        x_n, v_n, c_n = store.epochs(night.subject_id, epochs_range)
        x_batch[b, :length] = x_n
        v_batch[b, :length] = v_n
        c_batch[b, :length] = c_n
        y_batch[b, :length] = night.dense_labels[start_offset:start_offset + length]

    return (
        torch.from_numpy(x_batch).to(device),
        torch.from_numpy(v_batch).to(device),
        torch.from_numpy(c_batch).to(device),
        torch.from_numpy(y_batch).to(device),
    )


@torch.inference_mode()
def predict_subepoch(
    store,
    model: SubEpochWaveformNet,
    cohort: ContiguousCohort,
    *,
    batch_epochs: int = 64,
    device: str | torch.device = "cpu",
    stop=lambda: False,
) -> np.ndarray | None:
    """Encode each night in non-overlapping `batch_epochs` chunks, then run BiGRU over the full night."""
    model.eval()
    device = torch.device(device)
    probabilities = np.full((cohort.n_rows, N_CLASSES), np.nan, dtype=np.float32)

    for night in cohort.nights:
        if stop():
            return None
        span = len(night.dense_labels)
        token_chunks: list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []
        for start in range(0, span, batch_epochs):
            if stop():
                return None
            end = min(start + batch_epochs, span)
            ep_range = range(night.min_epoch + start, night.min_epoch + end)
            x_n, v_n, c_n = store.epochs(night.subject_id, ep_range)
            x_t = torch.from_numpy(x_n).unsqueeze(0).to(device)
            v_t = torch.from_numpy(v_n).unsqueeze(0).to(device)
            c_t = torch.from_numpy(c_n).unsqueeze(0).to(device)
            toks, tok_v = model.encode_subepochs(x_t, v_t, c_t)
            token_chunks.append(toks)
            valid_chunks.append(tok_v)
        night_tokens = torch.cat(token_chunks, dim=1)
        night_valid = torch.cat(valid_chunks, dim=1)
        logits, _ = model.classify_subepochs(night_tokens, night_valid)
        probs = logits.squeeze(0).softmax(dim=-1).cpu().numpy()
        probabilities[night.row_indices] = probs[night.row_epochs - night.min_epoch]

    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite predictions in predict_subepoch")
    return probabilities


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def fit_subepoch_fold(
    store,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    output_dir: Path,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    chunk_epochs: int = 32,
    eval_batch_epochs: int = 64,
    max_steps: int = 4000,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-3,
    local_loss_weight: float = 0.3,
    validation_interval: int = VALIDATION_INTERVAL,
    checkpoint_interval_steps: int = CHECKPOINT_INTERVAL_STEPS,
    seed: int = 7,
    time_limit_seconds: float | None = None,
    pause_file: Path | None = None,
    smoke_pause_after: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Train or resume one fold; returns `(test_probs, val_probs, runtime)` or `None` if paused."""
    started = active_clock()
    training = ContiguousCohort.from_frame(train)
    tuning = ContiguousCohort.from_frame(validation)
    scoring = ContiguousCohort.from_frame(test)
    partitions = [training.subject_ids, tuning.subject_ids, scoring.subject_ids]
    if len(set.union(*partitions)) != sum(map(len, partitions)):
        raise ValueError("Training, validation, and scoring patients must be disjoint")
    if batch_size < 1 or chunk_epochs < 1 or max_steps < 1:
        raise ValueError("Batch size, chunk_epochs, and max_steps must be positive")

    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.pt"
    runtime_path = output_dir / "runtime.json"

    config = {
        "version": 1,
        "device": str(device),
        "batch_size": batch_size,
        "chunk_epochs": chunk_epochs,
        "eval_batch_epochs": eval_batch_epochs,
        "max_steps": max_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "local_loss_weight": local_loss_weight,
        "validation_interval": validation_interval,
        "seed": seed,
        "splits": [training.fingerprint, tuning.fingerprint, scoring.fingerprint],
    }

    torch.manual_seed(seed)
    model = SubEpochWaveformNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)
    stage_counts, weights_cpu = tempered_class_weights(training.labels, power=-0.50)
    weights = weights_cpu.to(device)

    rss_now, peak_rss = process_memory_bytes()
    state: dict = {
        "config": config,
        "phase": "train",
        "step": 0,
        "last_validation": 0,
        "best_score": -1.0,
        "best_step": 0,
        "best_model": None,
        "history": [],
        "probabilities": None,
        "val_probabilities": None,
        "active_seconds": 0.0,
        "scoring_seconds": 0.0,
        "peak_rss_bytes": peak_rss,
        "minimum_available_bytes": available_memory_bytes(),
    }

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError(f"Checkpoint data or training configuration changed: {ckpt_path}")
        if state["phase"] == "complete":
            runtime = state["runtime"]
            _atomic_json(runtime_path, runtime)
            return state["probabilities"].numpy(), state["val_probabilities"].numpy(), runtime
        model.load_state_dict(state.pop("model"))
        optimizer.load_state_dict(state.pop("optimizer"))
        scheduler.load_state_dict(state.pop("scheduler"))
        torch.set_rng_state(state.pop("torch_rng"))

    prior_seconds = state["active_seconds"]
    prior_scoring_seconds = state["scoring_seconds"]
    scoring_started: float | None = None
    last_checkpoint = started
    last_mem_sample = started
    stop_reason: str | None = None

    def log(msg: str) -> None:
        print(f"subepoch {output_dir.name}: {msg}", flush=True)

    def stop() -> bool:
        nonlocal last_mem_sample, stop_reason
        if stop_reason is not None:
            return True
        now = active_clock()
        if pause_file is not None and pause_file.exists():
            stop_reason = "pause requested"
        elif time_limit_seconds is not None and now - started >= time_limit_seconds:
            stop_reason = "active time limit reached"
        elif now - last_mem_sample >= MEMORY_SAMPLE_INTERVAL_SECONDS:
            last_mem_sample = now
            rss, peak = process_memory_bytes()
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], peak)
            avail = available_memory_bytes()
            state["minimum_available_bytes"] = min(state["minimum_available_bytes"], avail)
            if rss > MAX_RSS_BYTES:
                stop_reason = f"process RSS {rss / 2**30:.2f} GiB exceeded {MAX_RSS_BYTES / 2**30:.2f} GiB limit"
            elif avail < MIN_AVAILABLE_BYTES:
                stop_reason = f"available RAM {avail / 2**30:.2f} GiB fell below {MIN_AVAILABLE_BYTES / 2**30:.2f} GiB"
        return stop_reason is not None

    def checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint
        now = active_clock()
        if not force and now - last_checkpoint < CHECKPOINT_INTERVAL_SECONDS:
            return
        rss, peak = process_memory_bytes()
        state["peak_rss_bytes"] = max(state["peak_rss_bytes"], peak)
        state["minimum_available_bytes"] = min(state["minimum_available_bytes"], available_memory_bytes())
        state["active_seconds"] = prior_seconds + now - started
        if scoring_started is not None:
            state["scoring_seconds"] = prior_scoring_seconds + now - scoring_started
        payload = {
            **state,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
        }
        _atomic_save(payload, ckpt_path)
        last_checkpoint = now
        log(f"checkpoint saved at {state['phase']} step {state['step']}")

    log(f"{state['phase']} step {state['step']}/{max_steps} on {device} (rss={rss_now / 2**20:.1f} MiB)")

    while state["phase"] == "train" and not stop():
        step = state["step"]
        needs_val = step > state["last_validation"] and (
            step % validation_interval == 0 or step == max_steps
        )
        if needs_val:
            val_probs = predict_subepoch(
                store, model, tuning, batch_epochs=eval_batch_epochs, device=device, stop=stop,
            )
            if val_probs is None:
                break
            val_f1 = float(
                f1_score(
                    tuning.labels,
                    val_probs.argmax(axis=1),
                    labels=range(N_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            )
            state["history"].append({"step": step, "val_macro_f1": val_f1})
            if val_f1 > state["best_score"]:
                state["best_score"] = val_f1
                state["best_step"] = step
                state["best_model"] = {
                    k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                }
            state["last_validation"] = step
            gc.collect()
            log(f"step {step}/{max_steps}: val macro-F1 {val_f1:.4f} (best {state['best_score']:.4f} @ {state['best_step']})")
            if step == max_steps:
                state["phase"] = "scoring"
                checkpoint(force=True)
                break
            checkpoint(force=True)
            continue

        if step == max_steps:
            state["phase"] = "scoring"
            checkpoint(force=True)
            break

        model.train()
        rng = np.random.default_rng([seed, step])
        x, valid, channels, y = sample_contiguous_batch(
            store, training, rng, batch_size=batch_size, chunk_epochs=chunk_epochs, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, local_logits = model(x, valid, channels, return_local=True)
        loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), y.reshape(-1), weight=weights, ignore_index=-1)
        if local_loss_weight > 0:
            loss = loss + local_loss_weight * F.cross_entropy(
                local_logits.reshape(-1, N_CLASSES), y.reshape(-1), weight=weights, ignore_index=-1,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()

        state["step"] = step + 1
        if smoke_pause_after is not None and state["step"] == smoke_pause_after:
            checkpoint(force=True)
            log(f"smoke pause reached at step {state['step']}")
            return None
        if state["step"] % checkpoint_interval_steps == 0:
            checkpoint(force=True)
        else:
            checkpoint(force=False)

    if state["phase"] == "scoring" and not stop():
        scoring_started = active_clock()
        if state["best_model"] is None:
            state["best_model"] = {
                k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
            }
            state["best_step"] = state["step"]
        model.load_state_dict(state["best_model"])
        val_probs = predict_subepoch(
            store, model, tuning, batch_epochs=eval_batch_epochs, device=device, stop=stop,
        )
        test_probs = (
            predict_subepoch(store, model, scoring, batch_epochs=eval_batch_epochs, device=device, stop=stop)
            if val_probs is not None
            else None
        )
        if val_probs is not None and test_probs is not None:
            now = active_clock()
            state["active_seconds"] = prior_seconds + now - started
            state["scoring_seconds"] = prior_scoring_seconds + now - scoring_started
            _, peak = process_memory_bytes()
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], peak)
            state["minimum_available_bytes"] = min(state["minimum_available_bytes"], available_memory_bytes())
            runtime = {
                "config": config,
                "fit_patients": len(training.nights),
                "validation_patients": len(tuning.nights),
                "evaluation_patients": len(scoring.nights),
                "fit_epochs": training.n_rows,
                "validation_epochs": tuning.n_rows,
                "evaluation_epochs": scoring.n_rows,
                "parameter_count": model.parameter_count,
                "stage_counts": stage_counts.tolist(),
                "class_weights": weights_cpu.tolist(),
                "best_step": state["best_step"],
                "best_validation_macro_f1": state["best_score"],
                "active_seconds": state["active_seconds"],
                "scoring_seconds": state["scoring_seconds"],
                "ms_per_step": 1000.0 * (state["active_seconds"] - state["scoring_seconds"]) / max_steps,
                "peak_rss_bytes": state["peak_rss_bytes"],
                "minimum_available_bytes": state["minimum_available_bytes"],
                "history": state["history"],
            }
            state["probabilities"] = torch.from_numpy(test_probs)
            state["val_probabilities"] = torch.from_numpy(val_probs)
            state["runtime"] = runtime
            state["phase"] = "complete"
            checkpoint(force=True)
            _atomic_json(runtime_path, runtime)
            gc.collect()
            return test_probs, val_probs, runtime

    checkpoint(force=True)
    if stop_reason:
        log(stop_reason)
    return None


def calibrate_prior_power(probabilities: np.ndarray, priors: np.ndarray, alpha: float) -> np.ndarray:
    """Apply `p * (prior ** alpha)` and renormalize rows to sum to 1."""
    if alpha == 0.0:
        out = probabilities.astype(np.float64, copy=True)
        out /= out.sum(axis=1, keepdims=True)
        return out.astype(np.float32)
    factor = np.power(np.clip(priors, 1e-12, None), alpha)
    scaled = probabilities.astype(np.float64) * factor
    scaled /= scaled.sum(axis=1, keepdims=True)
    return scaled.astype(np.float32)


def oof_prior_power_calibration(
    y: np.ndarray,
    folds: np.ndarray,
    probabilities: np.ndarray,
    fold_train_counts: dict[int, np.ndarray],
    alpha_grid: np.ndarray | None = None,
) -> dict:
    """Leak-free OOF prior-power calibration (`alpha=0.5` and nested-OOF grid selection)."""
    if alpha_grid is None:
        alpha_grid = np.round(np.linspace(0.0, 1.0, 21), 4)
    fold_ids = sorted(fold_train_counts)
    priors = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for f in fold_ids:
        counts = np.asarray(fold_train_counts[f], dtype=np.float64)
        priors[folds == f] = counts / counts.sum()

    bayes_half = calibrate_prior_power(probabilities, priors, 0.5)

    nested = np.empty_like(probabilities, dtype=np.float32)
    fold_alphas: dict[str, float] = {}
    for f in fold_ids:
        train_mask = folds != f
        eval_mask = folds == f
        best_alpha = 0.0
        best_f1 = -1.0
        for alpha in alpha_grid:
            cand = calibrate_prior_power(probabilities[train_mask], priors[train_mask], float(alpha))
            score = float(f1_score(y[train_mask], cand.argmax(axis=1), labels=range(N_CLASSES), average="macro", zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_alpha = float(alpha)
        fold_alphas[str(f)] = best_alpha
        nested[eval_mask] = calibrate_prior_power(probabilities[eval_mask], priors[eval_mask], best_alpha)

    return {
        "bayes_half_probabilities": bayes_half,
        "nested_probabilities": nested,
        "fold_alphas": fold_alphas,
        "mean_alpha": float(np.mean(list(fold_alphas.values()))),
    }
