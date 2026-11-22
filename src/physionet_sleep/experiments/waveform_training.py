"""Bounded, resumable training of a development-fold waveform classifier."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F

from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET
from physionet_sleep.experiments.waveform_clock import active_clock
from physionet_sleep.experiments.waveform_data import CONTEXT_EPOCHS as RADIUS, SAMPLE_RATE
from physionet_sleep.labels.cinc2018 import EPOCH_SECONDS


N_CLASSES = 5
PATCHES_PER_EPOCH = EPOCH_SECONDS
CONTEXT_EPOCHS = 2 * RADIUS + 1
VALIDATION_INTERVAL = 250
CHECKPOINT_INTERVAL_SECONDS = 300
MEMORY_SAMPLE_INTERVAL_SECONDS = 5
MINIMUM_AVAILABLE_BYTES = 3 * 2**29


@dataclass
class _Examples:
    subjects: np.ndarray
    epochs: np.ndarray
    labels: np.ndarray
    by_subject: dict[str, np.ndarray]
    fingerprint: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> _Examples:
        data = frame[[SUBJ, "epoch_index", TARGET]].reset_index(drop=True)
        if data.empty or data.isna().any().any() or data.duplicated([SUBJ, "epoch_index"]).any():
            raise ValueError("Waveform splits require unique, complete patient/epoch rows")
        epochs = data["epoch_index"].to_numpy(dtype=np.int64)
        labels = data[TARGET].to_numpy(dtype=np.int64)
        if (
            (epochs < 0).any() or not np.array_equal(epochs, data["epoch_index"])
            or not np.array_equal(labels, data[TARGET]) or not np.isin(labels, range(N_CLASSES)).all()
        ):
            raise ValueError("Waveform splits require nonnegative epochs and valid integer stage codes")
        subjects = data[SUBJ].astype(str)
        hashed = pd.util.hash_pandas_object(data, index=False).to_numpy()
        return cls(
            subjects.to_numpy(), epochs, labels, subjects.groupby(subjects, sort=False).indices,
            hashlib.sha256(hashed.tobytes()).hexdigest(),
        )


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "xpu":
        torch.xpu.manual_seed(seed)


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)


def _weights(examples: _Examples, device: torch.device) -> torch.Tensor:
    counts = np.bincount(examples.labels, minlength=N_CLASSES).clip(min=1)
    return torch.from_numpy((counts.mean() / counts).astype(np.float32)).to(device)


def _batch(store, examples: _Examples, rng: np.random.Generator, size: int, device: torch.device):
    indices = rng.integers(len(examples.epochs), size=size)
    windows = [store.window(examples.subjects[i], int(examples.epochs[i])) for i in indices]
    arrays = tuple(torch.as_tensor(np.stack([item[key] for item in windows]), device=device)
                   for key in ("x", "valid", "channels"))
    return (*arrays, torch.as_tensor(examples.labels[indices], device=device))


def _update(model, optimizer, batch, weights, rng, pretrain: bool) -> None:
    x, valid, channels, y = batch
    optimizer.zero_grad(set_to_none=True)
    if pretrain:
        # One contiguous 15-second gap per 30-second epoch. Exclude padding and missing leads.
        width = PATCHES_PER_EPOCH // 2
        begin = rng.integers(PATCHES_PER_EPOCH - width + 1, size=x.shape[:2])
        patches = np.arange(PATCHES_PER_EPOCH)
        mask = torch.as_tensor((begin[..., None] <= patches) & (patches < begin[..., None] + width), device=x.device)
        sample_mask = mask.repeat_interleave(SAMPLE_RATE, dim=-1) & valid
        observed = sample_mask.unsqueeze(-2) & channels.unsqueeze(-1)
        reconstruction = model.reconstruct(x, valid, channels, mask)
        loss = ((reconstruction - x).square() * observed).sum() / observed.sum().clamp_min(1)
    else:
        loss = F.cross_entropy(model(x, valid, channels), y, weight=weights)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite waveform training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()


def _epoch_embeddings(store, model, subject, epochs, batch_size, device, stop):
    embeddings, present = [], []
    for start in range(0, len(epochs), batch_size):
        if stop():
            return None
        x, valid, channels = (torch.as_tensor(value, device=device)
                              for value in store.epochs(subject, epochs[start:start + batch_size]))
        embeddings.append(model.encode_epochs(x, valid, channels))
        present.append(valid.any(dim=-1) & channels.any(dim=-1))
    return torch.cat(embeddings), torch.cat(present)


@torch.inference_mode()
def _predict(store, model, examples, probabilities, batch_size, device, stop, checkpoint) -> bool:
    """Compute one local embedding per distinct epoch and publish complete patients in source row order."""
    model.eval()
    offsets = np.arange(-RADIUS, RADIUS + 1)
    for subject, rows in examples.by_subject.items():
        if torch.isfinite(probabilities[rows, 0]).all():
            continue
        contexts = examples.epochs[rows, None] + offsets
        unique, mapping = np.unique(contexts, return_inverse=True)
        encoded = _epoch_embeddings(store, model, subject, unique, batch_size * CONTEXT_EPOCHS, device, stop)
        if encoded is None:
            return False
        vectors, present = encoded
        mapping = torch.as_tensor(mapping.reshape(contexts.shape), device=device)
        for start in range(0, len(rows), batch_size):
            if stop():
                probabilities[rows] = float("nan")
                return False
            positions = mapping[start:start + batch_size]
            logits = model.classify_epochs(vectors[positions], present[positions])
            probabilities[rows[start:start + batch_size]] = logits.softmax(-1).cpu()
        checkpoint()
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Non-finite waveform predictions")
    return True


def _save(state: dict, path: Path) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as destination:
        torch.save(state, destination)
        destination.flush()
        os.fsync(destination.fileno())
    temp.replace(path)


def _save_runtime(state: dict, path: Path) -> None:
    runtime = {name: state[name] for name in (
        "config", "active_seconds", "scoring_seconds", "minimum_available_bytes",
        "staging_parameter_count", "total_parameter_count", "best_step", "best_score",
    )}
    n_epochs = len(state["probabilities"])
    runtime["scored_epochs"] = n_epochs
    runtime["scoring_seconds_per_epoch"] = state["scoring_seconds"] / n_epochs
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(runtime, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fit_waveform_fold(
    store, model_factory, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, *,
    pretrained: bool, output_dir: Path, device: str | torch.device, batch_size: int, seed: int,
    pretrain_steps: int = 1000, supervised_steps: int = 1000,
    time_limit_seconds: float | None = None, pause_file: Path | None = None,
) -> np.ndarray | None:
    """Fit or resume a single fold; return ordered probabilities, or None after pausing.

    ``store`` supplies aligned waveform windows/epochs. Only the caller's training rows
    are sampled for updates. The supplied validation patients pick the best checkpoint,
    and the supplied test patients are scored only after that checkpoint is locked.
    """
    started = active_clock()
    training, tuning, scoring = (_Examples.from_frame(frame) for frame in (train, validation, test))
    partitions = [set(examples.by_subject) for examples in (training, tuning, scoring)]
    if len(set.union(*partitions)) != sum(map(len, partitions)):
        raise ValueError("Training, validation and scoring patients must be disjoint")
    if batch_size < 1 or supervised_steps < 1 or (pretrained and pretrain_steps < 1):
        raise ValueError("Waveform batch size and applicable optimization budgets must be positive")
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "checkpoint.pt"
    config = {
        "version": 2, "device": str(device), "batch_size": batch_size, "seed": seed,
        "pretrained": pretrained, "pretrain_steps": pretrain_steps, "supervised_steps": supervised_steps,
        "splits": [examples.fingerprint for examples in (training, tuning, scoring)],
    }
    _seed(seed, device)
    model = model_factory().to(device)
    optimizer = _optimizer(model)
    weights = _weights(training, device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    state = {
        "config": config, "phase": "pretrain" if pretrained else "supervised", "step": 0,
        "last_validation": 0, "best_score": -1.0, "best_step": 0, "best_model": None,
        "probabilities": None, "active_seconds": 0.0, "scoring_seconds": 0.0,
        "minimum_available_bytes": available_memory_bytes(),
        "staging_parameter_count": getattr(model, "staging_parameter_count", total_parameters),
        "total_parameter_count": total_parameters,
    }
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError(f"Checkpoint data or training configuration changed: {path}")
        if state["phase"] == "complete":
            _save_runtime(state, output_dir / "runtime.json")
            return state["probabilities"].numpy()
        model.load_state_dict(state.pop("model"))
        optimizer.load_state_dict(state.pop("optimizer"))
        torch.set_rng_state(state.pop("torch_rng"))
        if device.type == "xpu":
            torch.xpu.set_rng_state(state.pop("xpu_rng"), device=device)

    prior_seconds = state["active_seconds"]
    prior_scoring_seconds = state["scoring_seconds"]
    scoring_started = None
    last_checkpoint = started
    last_memory_sample = started
    low_memory_seen = False
    stop_reason = None

    def log(message: str) -> None:
        print(f"waveform {output_dir.name}: {message}", flush=True)

    log(f"{state['phase']} step {state['step']} on {device}")

    def stop() -> bool:
        nonlocal last_memory_sample, low_memory_seen, stop_reason
        if stop_reason is not None:
            return True
        now = active_clock()
        if pause_file is not None and pause_file.exists():
            stop_reason = "pause requested"
        elif time_limit_seconds is not None and now - started >= time_limit_seconds:
            stop_reason = "active time limit reached"
        elif now - last_memory_sample >= MEMORY_SAMPLE_INTERVAL_SECONDS:
            available = available_memory_bytes()
            last_memory_sample = now
            state["minimum_available_bytes"] = min(state["minimum_available_bytes"], available)
            low = available < MINIMUM_AVAILABLE_BYTES
            if low and low_memory_seen:
                stop_reason = "available RAM stayed below 1.5 GiB"
            low_memory_seen = low
        return stop_reason is not None

    def checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint
        now = active_clock()
        if force or now - last_checkpoint >= CHECKPOINT_INTERVAL_SECONDS:
            state["active_seconds"] = prior_seconds + now - started
            if scoring_started is not None:
                state["scoring_seconds"] = prior_scoring_seconds + now - scoring_started
            payload = {
                **state, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
            }
            if device.type == "xpu":
                payload["xpu_rng"] = torch.xpu.get_rng_state(device=device)
            _save(payload, path)
            last_checkpoint = now
            log(f"checkpoint saved at {state['phase']} step {state['step']}")

    def train_step(pretrain: bool) -> None:
        model.train()
        # Deriving sampling from the committed update index makes resuming exact.
        rng = np.random.default_rng([seed, int(pretrain), state["step"]])
        _update(model, optimizer, _batch(store, training, rng, batch_size, device), weights, rng, pretrain)
        state["step"] += 1
        if state["step"] % VALIDATION_INTERVAL == 0:
            total = pretrain_steps if pretrain else supervised_steps
            log(f"{state['phase']} {state['step']}/{total} updates")
        checkpoint()

    while state["phase"] == "pretrain" and not stop():
        if state["step"] == pretrain_steps:
            state.update(phase="supervised", step=0)
            _seed(seed, device)
            optimizer = _optimizer(model)
            checkpoint(force=True)
            break
        train_step(pretrain=True)

    while state["phase"] == "supervised" and not stop():
        step = state["step"]
        needs_validation = step > state["last_validation"] and (
            step % VALIDATION_INTERVAL == 0 or step == supervised_steps
        )
        if needs_validation:
            if state["probabilities"] is None:
                log(f"starting inner validation after {step} updates")
                state["probabilities"] = torch.full((len(tuning.epochs), N_CLASSES), float("nan"))
            probabilities = state["probabilities"]
            if not _predict(store, model, tuning, probabilities, batch_size, device, stop, checkpoint):
                break
            score = f1_score(tuning.labels, probabilities.argmax(1).numpy(), labels=range(N_CLASSES),
                             average="macro", zero_division=0)
            if score > state["best_score"]:
                state.update(best_score=score, best_step=step,
                             best_model={name: value.detach().to("cpu", copy=True)
                                         for name, value in model.state_dict().items()})
            state.update(last_validation=step, probabilities=None)
            log(f"inner validation macro-F1 {score:.4f} at {step} updates; best step {state['best_step']}")
            if step == supervised_steps:
                model.load_state_dict(state["best_model"])
                state.update(phase="scoring", probabilities=torch.full((len(scoring.epochs), N_CLASSES), float("nan")))
                checkpoint(force=True)
                break
        else:
            train_step(pretrain=False)

    if state["phase"] == "scoring" and not stop():
        scoring_started = active_clock()
        if _predict(store, model, scoring, state["probabilities"], batch_size, device, stop, checkpoint):
            state["phase"] = "complete"
    checkpoint(force=True)
    if state["phase"] == "complete":
        _save_runtime(state, output_dir / "runtime.json")
        return state["probabilities"].numpy()
    if stop_reason:
        log(stop_reason)
    return None


def available_memory_bytes() -> int:
    """Return available host RAM for the laptop profiling gate."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "total_page", "available_page",
                    "total_virtual", "available_virtual", "available_extended",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Could not measure available physical memory")
        return status.available_physical
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise OSError(f"Host memory measurement is unavailable on {sys.platform}")


def profile_waveform(store, model_factory, train: pd.DataFrame, *, device: str | torch.device,
                     batch_size: int, seed: int = 7, repeats: int = 5) -> dict[str, float | int]:
    """Measure supervised, masked and inference throughput using training patients only."""
    device = torch.device(device)
    examples = _Examples.from_frame(train)
    _seed(seed, device)
    model = model_factory().to(device)
    optimizer = _optimizer(model)
    weights = _weights(examples, device)
    rng = np.random.default_rng(seed)
    stop_sampling = threading.Event()
    minimum = available_memory_bytes()

    def sample_memory() -> None:
        nonlocal minimum
        while not stop_sampling.wait(0.1):
            minimum = min(minimum, available_memory_bytes())

    def synchronize() -> None:
        if device.type == "xpu":
            torch.xpu.synchronize(device)

    def profile_steps(pretrain: bool) -> float:
        _update(model, optimizer, _batch(store, examples, rng, batch_size, device), weights, rng, pretrain)
        synchronize()
        started = active_clock()
        for _ in range(repeats):
            _update(model, optimizer, _batch(store, examples, rng, batch_size, device), weights, rng, pretrain)
        synchronize()
        return (active_clock() - started) / repeats

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        pretraining_seconds = profile_steps(pretrain=True)
        supervised_seconds = profile_steps(pretrain=False)
        first_night = next(iter(examples.by_subject.values()))
        source = train.iloc[first_night[:batch_size * repeats]]
        sample = _Examples.from_frame(source)
        probabilities = torch.full((len(sample.epochs), N_CLASSES), float("nan"))
        started = active_clock()
        _predict(store, model, sample, probabilities, batch_size, device, lambda: False, lambda: None)
        synchronize()
        inference_seconds = (active_clock() - started) / len(sample.epochs)
    finally:
        stop_sampling.set()
        sampler.join()

    return {
        "pretraining_step_seconds": pretraining_seconds, "supervised_step_seconds": supervised_seconds,
        "inference_epoch_seconds": inference_seconds, "min_available_bytes": minimum,
    }
