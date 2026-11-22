"""Train and evaluate the unified whole-night BiGRU staging model (#75)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_all_additional_training as additional  # noqa: E402
import run_staging_complementary_model as expanded  # noqa: E402
import run_staging_gated_temporal as gated  # noqa: E402
from build_metric_intervals import (  # noqa: E402
    cv_fold_draws,
    cv_scores,
    interval,
    patient_confusions,
)
from physionet_sleep.analysis.epoch_windows import (  # noqa: E402
    DERIVED_FEATURES,
    EPOCH_SECONDS,
    WindowEncoder,
    dense_signals,
    spindle_burst_counts,
)
from physionet_sleep.experiments.bigru_staging import (  # noqa: E402
    BASE_STEP_TRACKS,
    DEFAULT_EPOCH_DIM,
    DEFAULT_STEP_DIM,
    EPOCH_FEATURE_COLUMNS,
    NIGHT_HISTORY_SOURCE_COLUMNS,
    STEP_PER_EPOCH,
    STEP_TRACKS,
    UnifiedBiGRUStagingNet,
    block_epoch_mean,
    compute_night_history,
    compute_sigma_ratio,
    compute_split_support,
    extract_occipital_theta,
    fit_prior_power_oof,
)
from physionet_sleep.experiments.gated_staging import stage_loss  # noqa: E402
from physionet_sleep.experiments.waveform_clock import active_clock  # noqa: E402
from physionet_sleep.experiments.waveform_data import WaveformStore  # noqa: E402
from run_staging_feature_tuning import save_frame  # noqa: E402
from run_waveform_development import (  # noqa: E402
    BASELINE,
    BUDGET_SECONDS,
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    RAW,
    SEED,
    STAGES,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
    pin_manifest,
    write_predictions,
)


OUT = EXPLORATION / "cinc-bigru-unified"
OCCIPITAL_CACHE = OUT / "occipital_cache"
DEV_MARKER_CACHE = expanded.OUT / "feature_cache"
ADDED_MARKER_CACHE = additional.OUT / "feature_cache"
ADDED_WINDOW_CACHE = additional.ADDED_WINDOW_CACHE
REFERENCE_73 = additional.OUT / "predictions.parquet"
SEQUENCE = gated.sequence
CHECKPOINT_SECONDS = 300
CLASSES = np.arange(len(STAGES))


@dataclass(frozen=True)
class TrainConfig:
    width: int = 96
    dropout: float = 0.15
    num_layers: int = 2
    weight_power: float = 0.45
    local_loss: float = 0.25
    lr: float = 0.0025
    eta_min: float = 0.00025
    weight_decay: float = 0.001
    max_passes: int = 18
    min_passes: int = 6
    patience: int = 5
    batch_size: int = 8


def assert_holdout_disjoint(subjects: set[str]) -> set[str]:
    """Ensure the 200 reserved CinC holdout patients never enter feature prep or training."""
    reserved = set((RAW / "selected_records_holdout.txt").read_text(encoding="utf-8").split())
    ledger = set(
        pd.read_csv(BASELINE / "holdout_subjects.csv", usecols=["record_id"]).record_id
    )
    if len(reserved) != 200 or reserved != ledger:
        raise ValueError("Reserved Boston holdout manifest changed unexpectedly")
    overlap = subjects & reserved
    if overlap:
        raise ValueError(f"Holdout leakage detected for {sorted(overlap)[:5]}")
    return reserved


def all_eligible_subjects() -> list[str]:
    """Return the 794 fitting+development subjects (320 dev + 474 added), asserting holdout disjointness."""
    dev = (BASELINE / "dev_subjects.txt").read_text(encoding="utf-8").split()
    added = additional.SUBJECTS.read_text(encoding="utf-8").split()
    if len(dev) != 320 or len(added) != 474 or set(dev) & set(added):
        raise ValueError("Unexpected Boston dev/added cohort sizes or overlap")
    combined = sorted(set(dev) | set(added))
    assert_holdout_disjoint(set(combined))
    return combined


def _valid_occipital_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        cols = pd.read_parquet(path, columns=["second", "o_rel_theta"])
    except Exception:
        return False
    return not cols.empty and list(cols.columns) == ["second", "o_rel_theta"]


def build_occipital_cache(
    subjects: list[str] | None = None,
    *,
    limit: int | None = None,
    cache_dir: Path = OCCIPITAL_CACHE,
    raw_dir: Path = RAW,
) -> dict:
    """Resumable extraction of 1 Hz occipital relative theta via WaveformStore."""
    eligible = all_eligible_subjects() if subjects is None else sorted(subjects)
    assert_holdout_disjoint(set(eligible))
    if limit is not None:
        eligible = eligible[:limit]
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = WaveformStore(raw_dir, permitted_subjects=eligible, max_open_nights=4)

    started = perf_counter()
    extracted = skipped = 0
    for subject in eligible:
        target = cache_dir / f"{subject}.parquet"
        if _valid_occipital_file(target):
            skipped += 1
            continue
        o1 = o2 = None
        try:
            o1 = store.channel_uv(subject, "O1-M2")
        except ValueError:
            o1 = None
        try:
            o2 = store.channel_uv(subject, "O2-M1")
        except ValueError:
            o2 = None
        df = extract_occipital_theta(o1, o2)
        tmp = target.with_suffix(".part")
        df.to_parquet(tmp, index=False)
        tmp.replace(target)
        extracted += 1
        if extracted % 25 == 0:
            log(f"occipital cache: extracted {extracted} (+{skipped} cached), {perf_counter() - started:.1f}s")

    elapsed = perf_counter() - started
    summary = {
        "requested": len(eligible),
        "extracted": extracted,
        "skipped": skipped,
        "seconds": elapsed,
        "seconds_per_extracted": elapsed / extracted if extracted else 0.0,
    }
    log(
        f"occipital cache done: requested={len(eligible)}, extracted={extracted}, "
        f"skipped={skipped}, {elapsed:.2f}s"
    )
    return summary


def _load_occipital_track(
    subject: str,
    n_seconds: int,
    *,
    occipital_dir: Path | None,
    allow_missing: bool,
) -> np.ndarray:
    if occipital_dir is None:
        if not allow_missing:
            raise FileNotFoundError(f"Occipital cache directory not provided for {subject}")
        return np.full(n_seconds, np.nan, dtype=np.float32)
    path = occipital_dir / f"{subject}.parquet"
    if not path.is_file():
        if not allow_missing:
            raise FileNotFoundError(
                f"Missing occipital cache for {subject}: run --build-cache first"
            )
        return np.full(n_seconds, np.nan, dtype=np.float32)
    frame = pd.read_parquet(path, columns=["second", "o_rel_theta"])
    values = frame["o_rel_theta"].to_numpy(dtype=np.float32)
    out = np.full(n_seconds, np.nan, dtype=np.float32)
    usable = min(n_seconds, len(values))
    out[:usable] = values[:usable]
    return out


def _load_complementary_panel(subject: str, is_added: bool) -> np.ndarray:
    folder = ADDED_MARKER_CACHE if is_added else DEV_MARKER_CACHE
    panel = pd.read_parquet(folder / f"{subject}.parquet")
    if (
        panel.empty
        or not panel.columns.is_unique
        or set(panel.columns) != set(expanded.FEATURES)
        or panel.index.name != "epoch_index"
        or not panel.index.equals(pd.RangeIndex(len(panel)))
    ):
        raise ValueError(f"{subject}: marker panel differs from #60/#61 contract")
    values = panel.loc[:, list(expanded.FEATURES)].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _subject_raw_signals(
    subject: str,
    *,
    added: set[str],
    occipital_dir: Path | None,
    allow_missing_occipital: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load 1 Hz step dense array (per-subject robust scaled) and raw 30 s epoch array."""
    is_added = subject in added
    cache = ADDED_WINDOW_CACHE if is_added else CACHE
    track_cols = list(dict.fromkeys(["second", *BASE_STEP_TRACKS, "c_rel_sigma"]))
    track_frame = pd.read_parquet(cache / "tracks1s" / f"{subject}.parquet", columns=track_cols)
    n_seconds = len(track_frame)

    raw_row_features = [f for f in SEQUENCE.FEATURES if f not in DERIVED_FEATURES]
    row_cols = list(dict.fromkeys(["slot", *raw_row_features, *NIGHT_HISTORY_SOURCE_COLUMNS]))
    row_frame = pd.read_parquet(cache / "rows6" / f"{subject}.parquet", columns=row_cols)

    o_rel_theta = _load_occipital_track(
        subject,
        n_seconds,
        occipital_dir=occipital_dir,
        allow_missing=allow_missing_occipital,
    )
    aug_tracks = track_frame.copy()
    aug_tracks["sigma_ratio"] = compute_sigma_ratio(track_frame)
    aug_tracks["o_rel_theta"] = o_rel_theta
    dense = dense_signals(
        aug_tracks,
        row_frame,
        list(STEP_TRACKS),
        list(SEQUENCE.FEATURES),
    )

    n_epochs = -(-n_seconds // EPOCH_SECONDS)
    epoch_raw = np.full((n_epochs, len(EPOCH_FEATURE_COLUMNS)), np.nan, dtype=np.float32)

    panel = _load_complementary_panel(subject, is_added)
    m = min(len(panel), n_epochs)
    epoch_raw[:m, 0:5] = panel[:m]

    spindle_all = spindle_burst_counts(track_frame)
    recent_k, recent_spindle = compute_split_support(panel[:m, 3], spindle_all[:m])
    epoch_raw[:m, 5] = recent_k
    epoch_raw[:m, 6] = recent_spindle

    epoch_raw[:, 7:14] = compute_night_history(row_frame, n_seconds)
    epoch_raw[:, 14] = block_epoch_mean(o_rel_theta, n_epochs)
    return dense, epoch_raw


def _fit_moments(
    arrays: list[np.ndarray] | None,
    width: int,
    *,
    allow_all_nan_cols: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.zeros(width, dtype=np.int64)
    total = np.zeros(width, dtype=np.float64)
    squared = np.zeros(width, dtype=np.float64)
    for arr in arrays or ():
        finite = np.isfinite(arr)
        count += finite.sum(axis=0)
        total += np.nansum(arr, axis=0, dtype=np.float64)
        squared += np.nansum(np.square(arr, dtype=np.float64), axis=0)
    return _finalize_moments(count, total, squared, allow_all_nan_cols=allow_all_nan_cols)


def _finalize_moments(
    count: np.ndarray,
    total: np.ndarray,
    squared: np.ndarray,
    *,
    allow_all_nan_cols: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allow = allow_all_nan_cols or set()
    missing = [i for i, n in enumerate(count) if n == 0 and i not in allow]
    if missing:
        raise ValueError(f"Fitting subjects lack observed values for columns {missing}")
    safe_count = np.maximum(count, 1)
    mean = np.where(count > 0, total / safe_count, 0.0)
    var = np.where(count > 0, np.maximum(0.0, squared / safe_count - mean**2), 1.0)
    scale = np.where(count > 0, np.sqrt(var) + 1e-6, 1.0)
    return mean, scale, count


def prepare_recordings(
    frame: pd.DataFrame,
    fit: set[str],
    check,
    *,
    added: set[str] | None = None,
    occipital_dir: Path | None = OCCIPITAL_CACHE,
    allow_missing_occipital: bool = False,
) -> tuple[dict[str, SEQUENCE.Recording], dict]:
    """Stream per-subject features with strictly fitting-only normalization."""
    added = added or set()
    all_subjects = set(frame.subject_id.astype(str).unique())
    assert_holdout_disjoint(all_subjects | fit)

    step_signal_names = [*STEP_TRACKS, *SEQUENCE.FEATURES]
    n_step = len(step_signal_names)
    n_epoch = len(EPOCH_FEATURE_COLUMNS)
    if n_step * STEP_PER_EPOCH != DEFAULT_STEP_DIM or n_epoch != DEFAULT_EPOCH_DIM:
        raise ValueError("Feature dimensions disagree with model contract")

    step_count = np.zeros(n_step, dtype=np.int64)
    step_total = np.zeros(n_step, dtype=np.float64)
    step_squared = np.zeros(n_step, dtype=np.float64)

    ep_count = np.zeros(n_epoch, dtype=np.int64)
    ep_total = np.zeros(n_epoch, dtype=np.float64)
    ep_squared = np.zeros(n_epoch, dtype=np.float64)

    fit_seconds = 0
    fit_epochs = 0
    for subject in sorted(fit):
        check()
        dense, ep = _subject_raw_signals(
            subject,
            added=added,
            occipital_dir=occipital_dir,
            allow_missing_occipital=allow_missing_occipital,
        )
        fit_seconds += len(dense)
        fit_epochs += len(ep)
        step_count += np.isfinite(dense).sum(axis=0)
        step_total += np.nansum(dense, axis=0, dtype=np.float64)
        step_squared += np.nansum(np.square(dense, dtype=np.float64), axis=0)

        ep_count += np.isfinite(ep).sum(axis=0)
        ep_total += np.nansum(ep, axis=0, dtype=np.float64)
        ep_squared += np.nansum(np.square(ep, dtype=np.float64), axis=0)

    step_allow = {step_signal_names.index("o_rel_theta")} if allow_missing_occipital else set()
    ep_allow = {EPOCH_FEATURE_COLUMNS.index("o_theta_30s")} if allow_missing_occipital else set()
    step_mean, step_scale, step_count = _finalize_moments(
        step_count, step_total, step_squared, allow_all_nan_cols=step_allow
    )
    ep_mean, ep_scale, ep_count = _finalize_moments(
        ep_count, ep_total, ep_squared, allow_all_nan_cols=ep_allow
    )

    encoder = WindowEncoder(step_signal_names, context=0, mean=step_mean, sd=step_scale)
    pad = encoder.steps_per_epoch
    recordings: dict[str, SEQUENCE.Recording] = {}

    for subject, night in frame.groupby("subject_id", sort=False):
        check()
        subject_id = str(subject)
        dense, ep = _subject_raw_signals(
            subject_id,
            added=added,
            occipital_dir=occipital_dir,
            allow_missing_occipital=allow_missing_occipital,
        )
        steps = encoder._steps(dense)[pad:-pad]
        steps = np.pad(steps, ((0, -len(steps) % pad), (0, 0)))
        step_values = steps.reshape(-1, DEFAULT_STEP_DIM).astype(np.float32)

        if len(ep) != len(step_values):
            raise ValueError(
                f"{subject_id}: step epochs ({len(step_values)}) != epoch features ({len(ep)})"
            )
        ep_norm = np.clip((ep - ep_mean) / ep_scale, -10.0, 10.0).astype(np.float32)
        ep_norm = np.where(np.isfinite(ep_norm), ep_norm, 0.0).astype(np.float32)
        combined = np.hstack((step_values, ep_norm)).astype(np.float32)
        if not np.isfinite(combined).all():
            raise ValueError(f"{subject_id}: nonfinite model features after standardization")

        labels = np.full(len(combined), -1, dtype=np.int64)
        at = night["epoch_index"].to_numpy()
        if at.max() >= len(combined):
            raise ValueError(f"{subject_id}: scored epoch extends beyond feature cache")
        labels[at] = night[LABEL].to_numpy()
        recordings[subject_id] = SEQUENCE.Recording(
            torch.from_numpy(combined), torch.from_numpy(labels)
        )

    return recordings, {
        "step_signals": step_signal_names,
        "epoch_features": list(EPOCH_FEATURE_COLUMNS),
        "total_fit_seconds": fit_seconds,
        "total_fit_epochs": fit_epochs,
        "step_finite_counts": step_count.tolist(),
        "step_mean": step_mean.tolist(),
        "step_scale": step_scale.tolist(),
        "epoch_finite_counts": ep_count.tolist(),
        "epoch_mean": ep_mean.tolist(),
        "epoch_scale": ep_scale.tolist(),
    }


def iter_batches(
    subjects: list[str], batch_size: int, rng: np.random.Generator | None = None
):
    order = subjects if rng is None else rng.permutation(subjects)
    for start in range(0, len(order), batch_size):
        yield list(order[start : start + batch_size])


def stopped(state: dict, config: TrainConfig) -> bool:
    return state["next_pass"] >= config.max_passes or (
        state["next_pass"] > config.min_passes
        and state["next_pass"] - state["best_pass"] >= config.patience
    )


def fit_bigru(
    recordings: dict[str, SEQUENCE.Recording],
    fit: list[str],
    validation: list[str],
    seed: int,
    directory: Path,
    pause_path: Path,
    deadline: float,
    config: TrainConfig = TrainConfig(),
    account=lambda: None,
    *,
    smoke_pause_updates: int | None = None,
) -> tuple[UnifiedBiGRUStagingNet | None, dict]:
    """Train UnifiedBiGRUStagingNet on all unique fitting patients per pass with exact resume."""
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    fit_subjects = sorted(fit)

    torch.manual_seed(seed)
    model = UnifiedBiGRUStagingNet(
        step_dim=DEFAULT_STEP_DIM,
        epoch_dim=DEFAULT_EPOCH_DIM,
        width=config.width,
        dropout=config.dropout,
        num_layers=config.num_layers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_passes, eta_min=config.eta_min
    )
    labels = torch.cat([recordings[s].scored_labels for s in fit_subjects])
    counts = torch.bincount(labels[labels >= 0], minlength=len(STAGES)).float()
    if (counts == 0).any():
        raise ValueError("Every stage must occur in the fitting pool")
    weights = counts.pow(-config.weight_power)

    state = {
        "next_pass": 0,
        "next_batch": 0,
        "updates": 0,
        "best": -1.0,
        "best_pass": 0,
        "best_weights": None,
        "history": [],
        "pass_seconds": 0.0,
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])

    def save() -> float:
        state.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=torch.get_rng_state(),
        )
        SEQUENCE.checkpoint(checkpoint, state)
        account()
        return active_clock()

    last_checkpoint = active_clock()
    while not stopped(state, config):
        if pause_path.exists() or active_clock() >= deadline:
            return None, state
        step = state["next_pass"]
        batches = list(
            iter_batches(fit_subjects, config.batch_size, np.random.default_rng(seed + step))
        )
        began = active_clock()
        model.train()
        for i, batch in enumerate(batches):
            if i < state["next_batch"]:
                continue
            if pause_path.exists() or active_clock() >= deadline:
                state["pass_seconds"] += active_clock() - began
                save()
                return None, state
            x, y, recorded = SEQUENCE.collate(recordings, batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, recorded, return_components=True)
            loss = (
                stage_loss(output.logits, y, weights)
                + config.local_loss * stage_loss(output.local, y, weights)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss in UnifiedBiGRUStagingNet")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            state["updates"] += 1
            state["next_batch"] = i + 1
            now = active_clock()
            smoke_stop = state["updates"] == smoke_pause_updates
            stopping = pause_path.exists() or now >= deadline or smoke_stop
            if stopping or now - last_checkpoint >= CHECKPOINT_SECONDS:
                state["pass_seconds"] += now - began
                last_checkpoint = began = save()
            if stopping:
                return None, state

        scheduler.step()
        score = SEQUENCE.validation_score(model, recordings, validation)
        pass_number = step + 1
        if score > state["best"]:
            state.update(
                best=score,
                best_pass=pass_number,
                best_weights={
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                },
            )
        elapsed = state["pass_seconds"] + active_clock() - began
        state["history"].append(
            {
                "pass": pass_number,
                "val_macro_f1": score,
                "lr": float(scheduler.get_last_lr()[0]),
                "updates": state["updates"],
                "seconds": elapsed,
            }
        )
        state.update(next_pass=pass_number, next_batch=0, pass_seconds=0.0)
        last_checkpoint = save()
        log(
            f"{directory.name} pass {pass_number}/{config.max_passes}: "
            f"val macro-F1 {score:.4f} (best {state['best']:.4f} @ {state['best_pass']}), {elapsed:.1f}s"
        )

    if state["best_weights"] is None:
        raise RuntimeError("Training completed without recording any validation checkpoint")
    model.load_state_dict(state["best_weights"])
    return model, {
        "unique_patients": len(fit_subjects),
        "validation_patients": len(validation),
        "passes": state["next_pass"],
        "optimizer_updates": state["updates"],
        "best_pass": state["best_pass"],
        "best_validation_macro_f1": state["best"],
        "stage_counts": counts.int().tolist(),
        "class_weights": weights.tolist(),
        "weight_power": config.weight_power,
        "width": config.width,
        "history": state["history"],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def verify_model_causality(recordings: dict[str, SEQUENCE.Recording], subjects: list[str]) -> None:
    """Verify strict past causality, future anti-causality, and batch-padding invariance."""
    model = UnifiedBiGRUStagingNet(
        step_dim=DEFAULT_STEP_DIM, epoch_dim=DEFAULT_EPOCH_DIM, width=32, dropout=0.0
    ).eval()
    sub_a, sub_b = subjects[0], subjects[1]
    x_single, _, rec_single = SEQUENCE.collate(recordings, [sub_a])
    x_batch, _, rec_batch = SEQUENCE.collate(recordings, [sub_a, sub_b])
    len_a = int(rec_single[0].sum().item())
    with torch.inference_mode():
        terms_single = model(x_single, rec_single, return_components=True)
        terms_batch = model(x_batch, rec_batch, return_components=True)
        for field in terms_single._fields:
            torch.testing.assert_close(
                getattr(terms_single, field)[0, :len_a],
                getattr(terms_batch, field)[0, :len_a],
                rtol=0,
                atol=1e-5,
            )
        pivot = min(10, len_a // 2)
        perturbed_future = x_single.clone()
        perturbed_future[:, pivot + 1 :] += 1.5
        out_future = model(perturbed_future, rec_single, return_components=True)
        torch.testing.assert_close(
            out_future.local[:, : pivot + 1],
            terms_single.local[:, : pivot + 1],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            out_future.past[:, : pivot + 1],
            terms_single.past[:, : pivot + 1],
            rtol=0,
            atol=1e-6,
        )
        perturbed_past = x_single.clone()
        perturbed_past[:, :pivot] -= 1.5
        out_past = model(perturbed_past, rec_single, return_components=True)
        torch.testing.assert_close(
            out_past.local[:, pivot:],
            terms_single.local[:, pivot:],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            out_past.future[:, pivot:],
            terms_single.future[:, pivot:],
            rtol=0,
            atol=1e-6,
        )


def smoke(partition, config: TrainConfig) -> None:
    started = active_clock()
    train, validation, _ = partition
    fit_ids = sorted(train.subject_id.unique())[:12]
    val_ids = sorted(validation.subject_id.unique())[:2]
    frame = pd.concat(
        [train[train.subject_id.isin(fit_ids)], validation[validation.subject_id.isin(val_ids)]]
    )
    recordings, scaling = prepare_recordings(
        frame,
        set(fit_ids),
        lambda: None,
        allow_missing_occipital=True,
    )
    verify_model_causality(recordings, fit_ids[:2])

    smoke_cfg = TrainConfig(
        width=config.width,
        dropout=config.dropout,
        num_layers=config.num_layers,
        weight_power=config.weight_power,
        local_loss=config.local_loss,
        lr=config.lr,
        eta_min=config.eta_min,
        weight_decay=config.weight_decay,
        max_passes=2,
        min_passes=6,
        patience=5,
        batch_size=8,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT, prefix="smoke-") as directory:
        path = Path(directory)
        pause = path / "PAUSE"
        paused, state = fit_bigru(
            recordings,
            fit_ids,
            val_ids,
            SEED,
            path / "resume",
            pause,
            float("inf"),
            smoke_cfg,
            smoke_pause_updates=1,
        )
        if paused is not None or state["next_pass"] != 0 or state["next_batch"] != 1:
            raise ValueError("Smoke fit failed to pause mid-pass after 1 update")
        resumed, timing = fit_bigru(
            recordings,
            fit_ids,
            val_ids,
            SEED,
            path / "resume",
            pause,
            float("inf"),
            smoke_cfg,
        )
        uninterrupted, _ = fit_bigru(
            recordings,
            fit_ids,
            val_ids,
            SEED,
            path / "uninterrupted",
            pause,
            float("inf"),
            smoke_cfg,
        )
        for name, parameter in resumed.state_dict().items():
            torch.testing.assert_close(parameter, uninterrupted.state_dict()[name], rtol=0, atol=0)
        if timing["optimizer_updates"] != 4:
            raise ValueError(f"Expected 4 optimizer updates in smoke, got {timing['optimizer_updates']}")

        val_sample = validation[validation.subject_id.isin(val_ids)].iloc[:25]
        probs, terms = gated.score_components(resumed, recordings, val_sample)
        if probs.shape != (len(val_sample), len(STAGES)) or terms.shape != (
            len(val_sample),
            len(gated.TERM_COLUMNS),
        ):
            raise ValueError("Unexpected score_components output shape")
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        rng = np.random.default_rng(SEED)
        fake_p = rng.dirichlet(np.ones(len(STAGES)), size=100)
        fake_y = rng.integers(0, len(STAGES), size=100)
        fake_folds = np.repeat(np.arange(5), 20)
        cal_p, cal_info = fit_prior_power_oof(fake_p, fake_y, fake_folds)
        np.testing.assert_allclose(cal_p.sum(axis=1), 1.0, atol=1e-5)

    atomic_json(
        OUT / "smoke.json",
        {
            "passed": True,
            "exact_resume": True,
            "causality_and_padding_invariant": True,
            "outer_evaluation_subjects": 0,
            "fitting_patients": len(fit_ids),
            "validation_patients": len(val_ids),
            "input_dim": DEFAULT_STEP_DIM + DEFAULT_EPOCH_DIM,
            "step_dim": DEFAULT_STEP_DIM,
            "epoch_dim": DEFAULT_EPOCH_DIM,
            "parameters": timing["parameters"],
            "active_seconds": active_clock() - started,
            "process_peak_rss_bytes": additional.peak_memory_bytes(),
            "pass_metrics": timing["history"],
            "calibration_smoke_folds": len(cal_info),
            "occipital_finite_fit_seconds": scaling["step_finite_counts"][
                scaling["step_signals"].index("o_rel_theta")
            ],
        },
    )
    log("Smoke passed: causality, padding invariance, shapes, and bit-exact mid-pass resume verified")


def _stage_metrics(y: np.ndarray, probs: np.ndarray, folds: list[np.ndarray]) -> dict:
    pred = probs.argmax(axis=1)
    fold_f1 = [
        float(f1_score(y[idx], pred[idx], labels=CLASSES, average="macro", zero_division=0))
        for idx in folds
    ]
    stage_f1 = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    truth_eye = np.eye(len(STAGES))[y]
    q = probs[np.arange(len(y)), y]
    return {
        "mean_fold_macro_f1": float(np.mean(fold_f1)),
        "fold_macro_f1": fold_f1,
        "pooled_macro_f1": float(
            f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y, pred)),
        "log_loss": float(-np.log(q.clip(1e-12)).mean()),
        "brier": float(np.square(probs - truth_eye).sum(axis=1).mean()),
        "stage_f1": {stage: float(stage_f1[k]) for k, stage in enumerate(STAGES)},
    }


def report(frame: pd.DataFrame, partitions: list, config: TrainConfig) -> None:
    expected = pd.concat(
        [test[[*KEYS, LABEL]].assign(fold=f) for f, (_, _, test) in enumerate(partitions)],
        ignore_index=True,
    )
    raw_folds = pd.concat(
        [pd.read_parquet(OUT / f"fold-{f}" / "predictions.parquet") for f in range(len(partitions))],
        ignore_index=True,
    )
    aligned = expected.merge(
        raw_folds, on=[*KEYS, LABEL, "fold"], validate="one_to_one", how="left", indicator=True
    )
    if len(aligned) != len(expected) or not aligned["_merge"].eq("both").all():
        raise ValueError("Fold predictions do not match the 320 development epochs")
    aligned = aligned.drop(columns="_merge")
    save_frame(OUT / "predictions.parquet", aligned)

    y = aligned[LABEL].to_numpy()
    fold_arr = aligned["fold"].to_numpy()
    fold_indices = [np.flatnonzero(fold_arr == f) for f in range(len(partitions))]
    raw_probs = aligned[PROBABILITIES].to_numpy(dtype=np.float32)

    cal_probs, cal_details = fit_prior_power_oof(raw_probs, y, fold_arr)
    cal_frame = aligned[[*KEYS, LABEL, "fold"]].copy()
    cal_frame[PROBABILITIES] = cal_probs
    save_frame(OUT / "calibrated_predictions.parquet", cal_frame)

    summary: dict = {
        "experiment": 75,
        "model": "UnifiedBiGRUStagingNet",
        "config": config.__dict__,
        "development_patients": int(aligned["subject_id"].nunique()),
        "development_scored_epochs": int(len(aligned)),
        "uncalibrated": _stage_metrics(y, raw_probs, fold_indices),
        "calibrated": {
            **_stage_metrics(y, cal_probs, fold_indices),
            "oof_prior_power_folds": cal_details,
        },
        "folds": [
            json.loads((OUT / f"fold-{f}" / "runtime.json").read_text(encoding="utf-8"))
            for f in range(len(partitions))
        ],
    }

    if REFERENCE_73.is_file():
        ref = expected.merge(
            pd.read_parquet(REFERENCE_73),
            on=[*KEYS, LABEL, "fold"],
            validate="one_to_one",
            how="left",
            indicator=True,
        )
        if len(ref) == len(expected) and ref["_merge"].eq("both").all():
            ref_probs = ref[PROBABILITIES].to_numpy(dtype=np.float32)
            subjects = aligned["subject_id"].to_numpy()
            cm_ref, patients = patient_confusions(subjects, y, ref_probs.argmax(axis=1))
            membership = aligned.groupby("subject_id")["fold"].first().loc[patients].to_numpy()
            groups = [np.flatnonzero(membership == f) for f in range(len(partitions))]
            draws = cv_fold_draws(groups, 2000, np.random.default_rng(SEED))
            comparisons = {}
            for tag, cand_probs in (("uncalibrated", raw_probs), ("calibrated", cal_probs)):
                cm_cand, _ = patient_confusions(subjects, y, cand_probs.argmax(axis=1))
                delta_f1 = (
                    cv_scores(cm_cand, groups, draws)["macro_f1"]
                    - cv_scores(cm_ref, groups, draws)["macro_f1"]
                )
                delta_k = (
                    cv_scores(cm_cand, groups, draws)["kappa"]
                    - cv_scores(cm_ref, groups, draws)["kappa"]
                )
                comparisons[f"{tag}_vs_73"] = {
                    "macro_f1_delta": summary[tag]["mean_fold_macro_f1"]
                    - _stage_metrics(y, ref_probs, fold_indices)["mean_fold_macro_f1"],
                    "paired_macro_f1_ci": interval(delta_f1),
                    "paired_kappa_ci": interval(delta_k),
                }
            summary["reference_73_comparison"] = comparisons

    atomic_json(OUT / "summary.json", summary)
    log(
        f"#75 completed: uncalibrated macro-F1 {summary['uncalibrated']['mean_fold_macro_f1']:.4f}, "
        f"calibrated {summary['calibrated']['mean_fold_macro_f1']:.4f}"
    )


def execute(args: argparse.Namespace) -> int:
    if args.build_cache:
        build_occipital_cache(limit=args.limit)
        return 0

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare Experiment 75 in FINDINGS.md before running")

    config = TrainConfig(
        width=args.width,
        dropout=args.dropout,
        num_layers=args.num_layers,
        weight_power=args.weight_power,
        local_loss=args.local_loss,
        lr=args.lr,
        eta_min=args.eta_min,
        weight_decay=args.weight_decay,
        max_passes=args.max_passes,
        min_passes=args.min_passes,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    frame, splits = development_frame()
    partitions = [fold_frames(frame, split) for split in splits]
    if args.smoke:
        smoke(partitions[0], config)
        return 0

    smoke_path = OUT / "smoke.json"
    if not smoke_path.is_file() or not json.loads(smoke_path.read_text(encoding="utf-8"))["passed"]:
        raise ValueError("Run --smoke and verify it passes before full 5-fold training")

    epochs, added = additional.training_cohort(set(frame.subject_id))
    cohort = pd.concat([frame[[*KEYS, LABEL]], epochs], ignore_index=True)
    all_subs = set(cohort.subject_id.astype(str).unique())
    assert_holdout_disjoint(all_subs)

    missing_occ = [
        s for s in sorted(all_subs) if not _valid_occipital_file(OCCIPITAL_CACHE / f"{s}.parquet")
    ]
    if missing_occ:
        raise FileNotFoundError(
            f"Missing {len(missing_occ)} occipital cache files (e.g. {missing_occ[0]}). "
            "Run --build-cache first."
        )

    pin_manifest(
        OUT / "training_manifest.json",
        {
            "experiment": 75,
            "seed": SEED,
            "torch": torch.__version__,
            "config": config.__dict__,
            "development_subjects": len(set(frame.subject_id)),
            "added_subjects": len(added),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )

    ledger = OUT / "active_budget.json"
    used = json.loads(ledger.read_text(encoding="utf-8"))["seconds"] if ledger.exists() else 0.0
    start = last_account = active_clock()
    deadline = start + BUDGET_SECONDS - used
    pause = OUT / "PAUSE"

    def account() -> None:
        nonlocal last_account
        last_account = active_clock()
        atomic_json(ledger, {"seconds": used + last_account - start})

    def check() -> None:
        now = active_clock()
        if now - last_account >= CHECKPOINT_SECONDS:
            account()
        if pause.exists() or now >= deadline:
            raise additional.Paused

    selected_folds = (
        [int(f) for f in args.folds.split(",")] if args.folds else list(range(len(partitions)))
    )
    try:
        for f in selected_folds:
            train, val, test = partitions[f]
            fold_dir = OUT / f"fold-{f}"
            if (fold_dir / "predictions.parquet").is_file():
                continue
            check()
            fit_subjects = sorted(set(train.subject_id) | added)
            val_subjects = sorted(val.subject_id.unique())
            log(f"Fold {f + 1}/5 preparing features for {len(fit_subjects)} fitting patients")
            began = active_clock()
            recordings, scaling = prepare_recordings(
                cohort,
                set(fit_subjects),
                check,
                added=added,
                occipital_dir=OCCIPITAL_CACHE,
                allow_missing_occipital=False,
            )
            prep_seconds = active_clock() - began
            model, runtime = fit_bigru(
                recordings,
                fit_subjects,
                val_subjects,
                SEED + f,
                fold_dir,
                pause,
                deadline,
                config,
                account,
            )
            if model is None:
                raise additional.Paused
            probs, terms = gated.score_components(model, recordings, test)
            keys = test[[*KEYS, LABEL]].assign(fold=f).reset_index(drop=True)
            save_frame(
                fold_dir / "components.parquet",
                keys.join(pd.DataFrame(terms, columns=gated.TERM_COLUMNS)),
            )
            atomic_json(
                fold_dir / "runtime.json",
                {
                    "fold": f,
                    "training": runtime,
                    "scaling": scaling,
                    "preparation_seconds": prep_seconds,
                    "process_peak_rss_bytes": additional.peak_memory_bytes(),
                    "scored_patients": int(test.subject_id.nunique()),
                    "scored_epochs": int(len(test)),
                },
            )
            write_predictions(fold_dir / "predictions.parquet", test, probs, f)
            log(
                f"Fold {f + 1}/5 saved: best val macro-F1 {runtime['best_validation_macro_f1']:.4f} "
                f"at pass {runtime['best_pass']}"
            )
            del recordings
            gc.collect()

        if all((OUT / f"fold-{f}" / "predictions.parquet").is_file() for f in range(len(partitions))):
            report(frame, partitions, config)
    except additional.Paused:
        log("#75 stopped cleanly at a saved checkpoint")
        return 75
    finally:
        account()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cache", action="store_true", help="Extract 1 Hz occipital theta cache")
    parser.add_argument("--limit", type=int, default=None, help="Optional subject limit for --build-cache")
    parser.add_argument("--smoke", action="store_true", help="Verify causality, shapes, and exact mid-pass resume")
    parser.add_argument("--folds", type=str, default=None, help="Optional comma-separated fold indices (e.g. 0,1)")
    parser.add_argument("--weight-power", type=float, default=0.45)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--local-loss", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--eta-min", type=float, default=0.00025)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-passes", type=int, default=18)
    parser.add_argument("--min-passes", type=int, default=6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.build_cache:
        raise SystemExit(execute(parsed))
    with local_training_slot():
        raise SystemExit(execute(parsed))
