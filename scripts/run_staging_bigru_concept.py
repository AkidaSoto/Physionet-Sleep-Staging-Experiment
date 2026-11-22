"""Experiment 79: Concept-additive slim BiGRU with EMA, label smoothing, and conformal sets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score, f1_score, log_loss
import torch
from torch.nn.utils import clip_grad_norm_

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_all_additional_training as additional
import run_staging_bigru_unified as unified
import run_staging_feature_context as sequence
import run_staging_gated_temporal as gated
from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from physionet_sleep.analysis.epoch_windows import WindowEncoder
from physionet_sleep.experiments.concept_bigru_staging import (
    CONCEPT_EPOCH_SIGNALS,
    CONCEPT_NAMES,
    CONCEPT_SLICES,
    CONCEPT_STEP_SIGNALS,
    ConceptBiGRUStagingNet,
    ConceptTerms,
    DEFAULT_CONCEPT_DIMS,
    EMAWeights,
    FEATURE_NAMES_79,
    STEP_PER_EPOCH,
    TOTAL_CONCEPT_DIM,
    evaluate_conformal_oof,
    fit_prior_power_oof,
    smoothed_stage_loss,
)
from physionet_sleep.experiments.waveform_clock import active_clock
from run_waveform_development import (
    BUDGET_SECONDS,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
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


OUT = EXPLORATION / "cinc-bigru-concept"
OCCIPITAL_CACHE = unified.OCCIPITAL_CACHE
REFERENCE_73 = unified.REFERENCE_73
REFERENCE_75 = EXPLORATION / "cinc-bigru-unified/calibrated_predictions.parquet"
REFERENCE_75_RAW = EXPLORATION / "cinc-bigru-unified/predictions.parquet"
CHECKPOINT_SECONDS = 300.0
CLASSES = np.arange(len(STAGES))
CONCEPT_TERM_COLUMNS = [
    f"concept_{concept}_{stage}" for concept in CONCEPT_NAMES for stage in STAGES
]

_ALL_RAW_STEP_NAMES = [*unified.STEP_TRACKS, *sequence.FEATURES]
_ALL_RAW_EPOCH_NAMES = list(unified.EPOCH_FEATURE_COLUMNS)
_STEP_IDX = {name: i for i, name in enumerate(_ALL_RAW_STEP_NAMES)}
_EPOCH_IDX = {name: i for i, name in enumerate(_ALL_RAW_EPOCH_NAMES)}


@dataclass(frozen=True)
class TrainConfig:
    width_per_concept: int = 16
    num_layers: int = 1
    dropout: float = 0.15
    weight_power: float = 0.45
    local_loss: float = 0.25
    label_smoothing: float = 0.03
    ema_decay: float = 0.995
    lr: float = 0.0025
    eta_min: float = 0.00025
    weight_decay: float = 0.001
    max_passes: int = 18
    min_passes: int = 6
    patience: int = 5
    batch_size: int = 8
    grad_clip: float = 5.0


def _subject_concept_raw(
    subject: str,
    *,
    added: set[str],
    occipital_dir: Path | None,
    allow_missing_occipital: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pruned per-subject step (6-s eligible 1s tracks) and unique 30-s epoch signals."""
    dense, ep = unified._subject_raw_signals(
        subject,
        added=added,
        occipital_dir=occipital_dir,
        allow_missing_occipital=allow_missing_occipital,
    )
    step_cols: list[str] = []
    for concept in CONCEPT_NAMES:
        step_cols.extend(CONCEPT_STEP_SIGNALS[concept])
    dense_step = dense[:, [_STEP_IDX[c] for c in step_cols]].astype(np.float32)

    n_ep = len(ep)
    epoch_parts: list[np.ndarray] = []
    for concept in CONCEPT_NAMES:
        for col in CONCEPT_EPOCH_SIGNALS[concept]:
            if col in _EPOCH_IDX:
                epoch_parts.append(ep[:, _EPOCH_IDX[col]])
            elif col in _STEP_IDX:
                # 30s constant rows6 / derived signal: downsample 1 Hz -> 1/epoch cleanly
                raw_1hz = dense[: n_ep * 30, _STEP_IDX[col]]
                if len(raw_1hz) < n_ep * 30:
                    padded = np.full(n_ep * 30, np.nan, dtype=np.float32)
                    padded[: len(raw_1hz)] = raw_1hz
                    raw_1hz = padded
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    epoch_mean = np.nanmean(raw_1hz.reshape(n_ep, 30), axis=1).astype(
                        np.float32
                    )
                epoch_parts.append(epoch_mean)
            else:
                raise KeyError(f"Unknown concept epoch signal: {col}")
    ep_unique = np.column_stack(epoch_parts).astype(np.float32)
    return dense_step, ep_unique


def _concept_feature_order_indices() -> tuple[list[str], list[str], np.ndarray]:
    """Map standardized (n_ep, 5*n_step + n_ep_unique) into concept-contiguous columns."""
    step_cols: list[str] = []
    ep_cols: list[str] = []
    for concept in CONCEPT_NAMES:
        step_cols.extend(CONCEPT_STEP_SIGNALS[concept])
        ep_cols.extend(CONCEPT_EPOCH_SIGNALS[concept])
    n_step = len(step_cols)

    step_pos = {name: i for i, name in enumerate(step_cols)}
    ep_pos = {name: i for i, name in enumerate(ep_cols)}

    gather_idx: list[int] = []
    for concept in CONCEPT_NAMES:
        for sig in CONCEPT_STEP_SIGNALS[concept]:
            sig_i = step_pos[sig]
            for step_idx in range(STEP_PER_EPOCH):
                gather_idx.append(step_idx * n_step + sig_i)
        for sig in CONCEPT_EPOCH_SIGNALS[concept]:
            gather_idx.append(STEP_PER_EPOCH * n_step + ep_pos[sig])
    return step_cols, ep_cols, np.asarray(gather_idx, dtype=np.int64)


def prepare_concept_recordings(
    frame: pd.DataFrame,
    fit: set[str],
    check,
    *,
    added: set[str] | None = None,
    occipital_dir: Path | None = OCCIPITAL_CACHE,
    allow_missing_occipital: bool = False,
) -> tuple[dict[str, sequence.Recording], dict]:
    """Stream pruned non-redundant features organized into 4 physiological concept slices."""
    added = added or set()
    all_subjects = set(frame.subject_id.astype(str).unique())
    unified.assert_holdout_disjoint(all_subjects | fit)

    step_cols, ep_cols, gather_idx = _concept_feature_order_indices()
    n_step = len(step_cols)
    n_ep_cols = len(ep_cols)
    if len(gather_idx) != TOTAL_CONCEPT_DIM:
        raise ValueError(
            f"Concept dimension mismatch: {len(gather_idx)} vs {TOTAL_CONCEPT_DIM}"
        )

    step_count = np.zeros(n_step, dtype=np.int64)
    step_total = np.zeros(n_step, dtype=np.float64)
    step_squared = np.zeros(n_step, dtype=np.float64)

    ep_count = np.zeros(n_ep_cols, dtype=np.int64)
    ep_total = np.zeros(n_ep_cols, dtype=np.float64)
    ep_squared = np.zeros(n_ep_cols, dtype=np.float64)

    fit_seconds = 0
    fit_epochs = 0
    for subject in sorted(fit):
        check()
        dense_step, ep_unique = _subject_concept_raw(
            subject,
            added=added,
            occipital_dir=occipital_dir,
            allow_missing_occipital=allow_missing_occipital,
        )
        fit_seconds += len(dense_step)
        fit_epochs += len(ep_unique)
        step_count += np.isfinite(dense_step).sum(axis=0)
        step_total += np.nansum(dense_step, axis=0, dtype=np.float64)
        step_squared += np.nansum(np.square(dense_step, dtype=np.float64), axis=0)

        ep_count += np.isfinite(ep_unique).sum(axis=0)
        ep_total += np.nansum(ep_unique, axis=0, dtype=np.float64)
        ep_squared += np.nansum(np.square(ep_unique, dtype=np.float64), axis=0)

    step_allow = (
        {step_cols.index("o_rel_theta")}
        if allow_missing_occipital and "o_rel_theta" in step_cols
        else set()
    )
    step_mean, step_scale, step_count = unified._finalize_moments(
        step_count, step_total, step_squared, allow_all_nan_cols=step_allow
    )
    ep_mean, ep_scale, ep_count = unified._finalize_moments(
        ep_count, ep_total, ep_squared
    )

    encoder = WindowEncoder(step_cols, context=0, mean=step_mean, sd=step_scale)
    pad = encoder.steps_per_epoch
    recordings: dict[str, sequence.Recording] = {}

    for subject, night in frame.groupby("subject_id", sort=False):
        check()
        subject_id = str(subject)
        dense_step, ep_unique = _subject_concept_raw(
            subject_id,
            added=added,
            occipital_dir=occipital_dir,
            allow_missing_occipital=allow_missing_occipital,
        )
        steps = encoder._steps(dense_step)[pad:-pad]
        steps = np.pad(steps, ((0, -len(steps) % pad), (0, 0)))
        step_values = steps.reshape(-1, STEP_PER_EPOCH * n_step).astype(np.float32)

        if len(ep_unique) != len(step_values):
            raise ValueError(
                f"{subject_id}: step epochs ({len(step_values)}) != epoch features ({len(ep_unique)})"
            )
        ep_norm = np.clip((ep_unique - ep_mean) / ep_scale, -10.0, 10.0).astype(
            np.float32
        )
        ep_norm = np.where(np.isfinite(ep_norm), ep_norm, 0.0).astype(np.float32)
        flat = np.hstack((step_values, ep_norm)).astype(np.float32)
        concept_x = np.ascontiguousarray(flat[:, gather_idx], dtype=np.float32)
        if not np.isfinite(concept_x).all():
            raise ValueError(f"{subject_id}: nonfinite features after standardization")

        labels = np.full(len(concept_x), -1, dtype=np.int64)
        at = night["epoch_index"].to_numpy()
        if at.max() >= len(concept_x):
            raise ValueError(f"{subject_id}: scored epoch extends beyond feature cache")
        labels[at] = night[LABEL].to_numpy()
        recordings[subject_id] = sequence.Recording(
            torch.from_numpy(concept_x), torch.from_numpy(labels)
        )

    return recordings, {
        "feature_names": list(FEATURE_NAMES_79),
        "concept_slices": {
            k: [int(s.start), int(s.stop)] for k, s in CONCEPT_SLICES.items()
        },
        "total_fit_seconds": fit_seconds,
        "total_fit_epochs": fit_epochs,
        "step_signals": step_cols,
        "epoch_signals": ep_cols,
        "step_finite_counts": step_count.tolist(),
        "ep_finite_counts": ep_count.tolist(),
    }


def _stopped(state: dict, config: TrainConfig) -> bool:
    return state["next_pass"] >= config.max_passes or (
        state["next_pass"] > config.min_passes
        and state["next_pass"] - state["best_pass"] >= config.patience
    )


def fit_concept_bigru(
    recordings: dict[str, sequence.Recording],
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
    concept_dims: tuple[int, ...] = DEFAULT_CONCEPT_DIMS,
    concept_names: tuple[str, ...] = CONCEPT_NAMES,
    width_per_concept: int | tuple[int, ...] | None = None,
) -> tuple[ConceptBiGRUStagingNet | None, dict]:
    """Train ConceptBiGRUStagingNet with EMA validation selection & exact mid-pass resume."""
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    fit_subjects = sorted(fit)

    torch.manual_seed(seed)
    model = ConceptBiGRUStagingNet(
        concept_dims=concept_dims,
        concept_names=concept_names,
        width_per_concept=(
            config.width_per_concept if width_per_concept is None else width_per_concept
        ),
        dropout=config.dropout,
        num_layers=config.num_layers,
    )
    ema = EMAWeights(model, decay=config.ema_decay)
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
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])

    def _save_state(next_pass: int, next_batch: int) -> None:
        state["next_pass"] = next_pass
        state["next_batch"] = next_batch
        state["model"] = model.state_dict()
        state["ema"] = ema.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict()
        state["rng"] = torch.get_rng_state()
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(checkpoint)

    last_save = active_clock()
    while not _stopped(state, config):
        if pause_path.exists() or active_clock() >= deadline:
            return None, state

        pass_idx = state["next_pass"]
        start_batch = state["next_batch"]
        pass_rng = np.random.default_rng(seed + pass_idx)
        batches = list(unified.iter_batches(fit_subjects, config.batch_size, pass_rng))
        if start_batch >= len(batches):
            start_batch = 0

        model.train()
        total_loss = 0.0
        total_epochs = 0
        pass_start = perf_counter()
        for batch_idx in range(start_batch, len(batches)):
            account()
            x, y, recorded = sequence.collate(recordings, batches[batch_idx])
            terms = model(x, recorded, return_components=True)
            loss = smoothed_stage_loss(
                terms.logits, y, weights, label_smoothing=config.label_smoothing
            ) + config.local_loss * smoothed_stage_loss(
                terms.local, y, weights, label_smoothing=config.label_smoothing
            )
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            ema.update(model)

            epochs_n = int((y >= 0).sum().item())
            total_loss += float(loss.item()) * epochs_n
            total_epochs += epochs_n
            state["updates"] += 1

            if (
                smoke_pause_updates is not None
                and state["updates"] >= smoke_pause_updates
            ):
                _save_state(pass_idx, batch_idx + 1)
                return None, state

            if (
                active_clock() - last_save >= CHECKPOINT_SECONDS
                and batch_idx + 1 < len(batches)
            ):
                _save_state(pass_idx, batch_idx + 1)
                last_save = active_clock()
            if pause_path.exists() or active_clock() >= deadline:
                _save_state(pass_idx, batch_idx + 1)
                return None, state

        scheduler.step()
        state["pass_seconds"] += perf_counter() - pass_start

        # Evaluate using EMA shadow weights (plus track raw validation for comparison)
        raw_val = sequence.validation_score(model, recordings, validation)
        backup = ema.copy_to(model)
        ema_val = sequence.validation_score(model, recordings, validation)
        model.load_state_dict(backup)

        avg_loss = total_loss / max(1, total_epochs)
        state["history"].append(
            {
                "pass": pass_idx + 1,
                "train_loss": avg_loss,
                "raw_val_macro_f1": float(raw_val),
                "ema_val_macro_f1": float(ema_val),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if ema_val > state["best"]:
            backup_best = ema.copy_to(model)
            state["best"] = float(ema_val)
            state["best_pass"] = pass_idx + 1
            state["best_weights"] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            model.load_state_dict(backup_best)

        log(
            f"  [concept-bigru pass {pass_idx + 1:02d}] loss={avg_loss:.4f} "
            f"val_raw={raw_val:.5f} val_ema={ema_val:.5f} "
            f"best={state['best']:.5f}@{state['best_pass']}"
        )
        _save_state(pass_idx + 1, 0)
        last_save = active_clock()

    if state["best_weights"] is not None:
        model.load_state_dict(state["best_weights"])
    model.eval()
    return model, state


@torch.inference_mode()
def score_fold_with_concepts(
    model: ConceptBiGRUStagingNet,
    recordings: dict[str, sequence.Recording],
    frame: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score recordings and extract temporal + per-concept additive logit contributions."""
    model.eval()
    pred_parts: list[pd.DataFrame] = []
    comp_parts: list[pd.DataFrame] = []
    concept_parts: list[pd.DataFrame] = []

    for subject, night in frame.groupby("subject_id", sort=False):
        rec = recordings[str(subject)]
        x = rec.features.unsqueeze(0)
        recorded = torch.ones((1, len(rec.features)), dtype=torch.bool)
        terms: ConceptTerms = model(x, recorded, return_components=True)

        at = night["epoch_index"].to_numpy()
        logits_np = terms.logits[0, at].cpu().numpy()
        probs_np = softmax(logits_np, axis=1)

        pred_df = night[[*KEYS, LABEL]].copy()
        pred_df["fold"] = int(fold)
        pred_df["prediction"] = probs_np.argmax(axis=1).astype(np.int64)
        for idx, col in enumerate(PROBABILITIES):
            pred_df[col] = probs_np[:, idx].astype(np.float32)
        pred_parts.append(pred_df)

        comp_df = night[[*KEYS, LABEL]].copy()
        comp_df["fold"] = int(fold)
        for term_name in ("local", "past", "future", "past_gate", "future_gate"):
            arr = getattr(terms, term_name)[0, at].cpu().numpy()
            for idx, stage in enumerate(STAGES):
                comp_df[f"{term_name}_{stage}"] = arr[:, idx].astype(np.float32)
        comp_parts.append(comp_df)

        c_df = night[[*KEYS, LABEL]].copy()
        c_df["fold"] = int(fold)
        c_arr = terms.concept_logits[0, at].cpu().numpy()  # (n_scored, K, 5)
        for c_idx, concept in enumerate(model.concept_names):
            for s_idx, stage in enumerate(STAGES):
                c_df[f"concept_{concept}_{stage}"] = c_arr[:, c_idx, s_idx].astype(
                    np.float32
                )
        for s_idx, stage in enumerate(STAGES):
            c_df[f"local_bias_{stage}"] = np.float32(terms.local_bias[s_idx].item())
        concept_parts.append(c_df)

    return (
        pd.concat(pred_parts, ignore_index=True),
        pd.concat(comp_parts, ignore_index=True),
        pd.concat(concept_parts, ignore_index=True),
    )


def verify_concept_model(model: ConceptBiGRUStagingNet) -> None:
    """Assert exact concept additivity, zero local cross-talk, causality, and padding invariance."""
    model.eval()
    torch.manual_seed(21)
    seq_a = torch.randn(1, 9, model.input_dim)
    rec_a = torch.ones((1, 9), dtype=torch.bool)
    seq_b = torch.randn(1, 14, model.input_dim)
    x_batch = torch.zeros(2, 14, model.input_dim)
    x_batch[0, :9] = seq_a[0]
    x_batch[1] = seq_b[0]
    rec_batch = torch.zeros(2, 14, dtype=torch.bool)
    rec_batch[0, :9] = True
    rec_batch[1] = True

    with torch.inference_mode():
        terms_a: ConceptTerms = model(seq_a, rec_a, return_components=True)
        terms_b: ConceptTerms = model(x_batch, rec_batch, return_components=True)
        for field in ("logits", "local", "past", "future", "past_gate", "future_gate", "concept_logits"):
            torch.testing.assert_close(
                getattr(terms_a, field)[0, :9],
                getattr(terms_b, field)[0, :9],
                rtol=0,
                atol=1e-5,
            )
        # Exact concept additivity: local == sum_k concept_logits_k + local_bias
        reconstructed = terms_a.concept_logits.sum(dim=-2) + terms_a.local_bias.view(1, 1, 5)
        torch.testing.assert_close(terms_a.local, reconstructed, rtol=0, atol=1e-5)

        # Concept isolation: perturbing only concept 0 leaves concepts 1..K-1 unchanged
        seq_c0 = seq_a.clone()
        s0 = slice(0, model.concept_dims[0])
        seq_c0[:, :, s0] += 3.5
        terms_c0 = model(seq_c0, rec_a, return_components=True)
        torch.testing.assert_close(
            terms_a.concept_logits[:, :, 1:],
            terms_c0.concept_logits[:, :, 1:],
            rtol=0,
            atol=1e-6,
        )

        # Causality & anti-causality: perturbing epoch t=4 cannot affect past[s<4] or future[s>4]
        seq_edit = seq_a.clone()
        seq_edit[:, 4, :] += 4.0
        terms_edit = model(seq_edit, rec_a, return_components=True)
        torch.testing.assert_close(
            terms_a.past[:, :4], terms_edit.past[:, :4], rtol=0, atol=1e-5
        )
        torch.testing.assert_close(
            terms_a.future[:, 5:], terms_edit.future[:, 5:], rtol=0, atol=1e-5
        )
        # Furthermore past_rnn output at t=4 uses only hidden[:, :4] (unchanged); only gates(hidden[:,4]) varies.
        past_raw_a = terms_a.past[:, 4] / terms_a.past_gate[:, 4].clamp_min(1e-6)
        past_raw_e = terms_edit.past[:, 4] / terms_edit.past_gate[:, 4].clamp_min(1e-6)
        torch.testing.assert_close(past_raw_a, past_raw_e, rtol=0, atol=1e-5)


def smoke() -> None:
    """Verify concept feature prep, isolation/causality, conformal math, and exact resume."""
    started = perf_counter()
    dev_all, splits = development_frame()
    fit, val, _ = fold_frames(dev_all, splits[0])
    subjects = sorted(fit.subject_id.astype(str).unique())[:6]
    val_subjects = sorted(val.subject_id.astype(str).unique())[:3]
    mini_frame = pd.concat(
        [
            fit[fit.subject_id.astype(str).isin(subjects)],
            val[val.subject_id.astype(str).isin(val_subjects)],
        ],
        ignore_index=True,
    )
    recs, norm_meta = prepare_concept_recordings(
        mini_frame, set(subjects), lambda: None
    )
    assert norm_meta["total_fit_epochs"] > 0
    assert recs[subjects[0]].features.shape[1] == TOTAL_CONCEPT_DIM

    cfg = TrainConfig(max_passes=2, min_passes=1, batch_size=3, patience=5)
    with TemporaryDirectory() as tmp:
        d1 = Path(tmp) / "uninterrupted"
        d2 = Path(tmp) / "resumed"
        m1, _ = fit_concept_bigru(
            recs, subjects, val_subjects, SEED, d1, d1 / "PAUSE", active_clock() + 600, cfg
        )
        assert m1 is not None
        paused_m, _ = fit_concept_bigru(
            recs,
            subjects,
            val_subjects,
            SEED,
            d2,
            d2 / "PAUSE",
            active_clock() + 600,
            cfg,
            smoke_pause_updates=1,
        )
        assert paused_m is None
        m2, _ = fit_concept_bigru(
            recs, subjects, val_subjects, SEED, d2, d2 / "PAUSE", active_clock() + 600, cfg
        )
        assert m2 is not None
        for k, v in m1.state_dict().items():
            torch.testing.assert_close(v, m2.state_dict()[k], rtol=0, atol=1e-6)
        verify_concept_model(m1)
        param_count = sum(p.numel() for p in m1.parameters() if p.requires_grad)

    log(
        f"SMOKE PASSED: dims={TOTAL_CONCEPT_DIM} ({DEFAULT_CONCEPT_DIMS}), "
        f"params={param_count}, rss={additional.peak_memory_bytes() / 1024**2:.1f} MiB, "
        f"elapsed={perf_counter() - started:.2f}s"
    )


def _compute_concept_diagnostics(
    comp_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    concept_names: tuple[str, ...] = CONCEPT_NAMES,
) -> dict:
    """Measure additive concept attribution & leave-one-concept-out local ablations."""
    y = comp_df[LABEL].to_numpy()
    full_logits = np.column_stack(
        [
            comp_df[f"local_{s}"].to_numpy()
            + comp_df[f"past_{s}"].to_numpy()
            + comp_df[f"future_{s}"].to_numpy()
            for s in STAGES
        ]
    )
    local_logits = np.column_stack([comp_df[f"local_{s}"].to_numpy() for s in STAGES])
    local_f1 = float(
        f1_score(y, local_logits.argmax(axis=1), labels=CLASSES, average="macro")
    )
    full_f1 = float(
        f1_score(y, full_logits.argmax(axis=1), labels=CLASSES, average="macro")
    )

    concept_ablation: dict[str, dict] = {}
    stage_mean_abs: dict[str, dict[str, float]] = {}
    for concept in concept_names:
        c_mat = np.column_stack(
            [concept_df[f"concept_{concept}_{s}"].to_numpy() for s in STAGES]
        )
        ablated_local = local_logits - c_mat
        ablated_f1 = float(
            f1_score(y, ablated_local.argmax(axis=1), labels=CLASSES, average="macro")
        )
        # Drop from full context logits too (holding temporal term fixed to isolate direct drop)
        ablated_full = full_logits - c_mat
        ablated_full_f1 = float(
            f1_score(y, ablated_full.argmax(axis=1), labels=CLASSES, average="macro")
        )
        per_stage_delta = {}
        for idx, stage in enumerate(STAGES):
            mask = y == idx
            base_s = float(
                f1_score(
                    y == idx,
                    local_logits.argmax(axis=1) == idx,
                    zero_division=0,
                )
            )
            drop_s = float(
                f1_score(
                    y == idx,
                    ablated_local.argmax(axis=1) == idx,
                    zero_division=0,
                )
            )
            per_stage_delta[stage] = round(drop_s - base_s, 5)
            stage_mean_abs.setdefault(concept, {})[stage] = round(
                float(np.abs(c_mat[mask, idx]).mean()), 4
            )
        concept_ablation[concept] = {
            "local_macro_f1_without": ablated_f1,
            "local_macro_f1_drop": round(ablated_f1 - local_f1, 5),
            "full_direct_drop": round(ablated_full_f1 - full_f1, 5),
            "per_stage_local_f1_delta": per_stage_delta,
            "mean_abs_true_stage_logit": stage_mean_abs[concept],
        }

    gate_means = {}
    for idx, stage in enumerate(STAGES):
        mask = y == idx
        gate_means[stage] = {
            "past_gate_mean": round(
                float(comp_df.loc[mask, f"past_gate_{stage}"].mean()), 4
            ),
            "future_gate_mean": round(
                float(comp_df.loc[mask, f"future_gate_{stage}"].mean()), 4
            ),
        }

    return {
        "local_only_macro_f1": local_f1,
        "full_macro_f1": full_f1,
        "temporal_context_gain": round(full_f1 - local_f1, 5),
        "concept_ablation": concept_ablation,
        "stage_gates": gate_means,
    }


def _transition_flags(frame: pd.DataFrame) -> np.ndarray:
    """Flag epochs adjacent to a scored stage change within the same night."""
    ordered = frame.sort_values([*KEYS]).reset_index()
    subj = ordered["subject_id"].to_numpy()
    ei = ordered["epoch_index"].to_numpy()
    y = ordered[LABEL].to_numpy()
    trans = np.zeros(len(ordered), dtype=bool)
    prev_same = (subj[1:] == subj[:-1]) & (ei[1:] == ei[:-1] + 1) & (y[1:] != y[:-1])
    trans[1:] |= prev_same
    trans[:-1] |= prev_same
    out = np.zeros(len(frame), dtype=bool)
    out[ordered["index"].to_numpy()] = trans
    return out


def report(
    oof: pd.DataFrame,
    val_oof: pd.DataFrame,
    comp_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    fold_histories: dict[int, dict],
    norm_summaries: dict[int, dict],
    param_count: int,
    config: TrainConfig,
    budget_seconds: float,
) -> dict:
    """Evaluate raw & OOF-calibrated accuracy, conformal sets, concepts, and #75/#73 deltas."""
    y = oof[LABEL].to_numpy()
    probs = oof[PROBABILITIES].to_numpy()
    folds_col = oof["fold"].to_numpy()
    subjects = oof["subject_id"].astype(str).to_numpy()

    cal_probs, cal_details = fit_prior_power_oof(probs, y, folds_col)
    cal_df = oof.copy()
    cal_df["prediction"] = cal_probs.argmax(axis=1).astype(np.int64)
    for idx, col in enumerate(PROBABILITIES):
        cal_df[col] = cal_probs[:, idx].astype(np.float32)

    unified.save_frame(OUT / "predictions.parquet", oof)
    unified.save_frame(OUT / "calibrated_predictions.parquet", cal_df)
    if not val_oof.empty:
        unified.save_frame(OUT / "val_predictions.parquet", val_oof)

    fold_masks = [folds_col == f for f in sorted(np.unique(folds_col).tolist())]
    raw_metrics = unified._stage_metrics(y, probs, fold_masks)
    cal_metrics = unified._stage_metrics(y, cal_probs, fold_masks)

    is_trans = _transition_flags(oof)
    conformal = {
        "raw": evaluate_conformal_oof(
            probs, y, folds_col, is_trans, stages=tuple(STAGES)
        ),
        "calibrated": evaluate_conformal_oof(
            cal_probs, y, folds_col, is_trans, stages=tuple(STAGES)
        ),
    }
    atomic_json(OUT / "conformal_summary.json", conformal)

    concept_diag = _compute_concept_diagnostics(comp_df, concept_df)

    # Paired bootstrap comparison vs #75 and #73
    rng = np.random.default_rng(SEED)
    cm_new_raw, pats_raw = patient_confusions(subjects, y, probs.argmax(axis=1))
    cm_new_cal, pats_cal = patient_confusions(subjects, y, cal_probs.argmax(axis=1))
    assert (pats_raw == pats_cal).all()
    pat_fold = np.array([folds_col[subjects == pt][0] for pt in pats_raw])
    f_idx = [np.flatnonzero(pat_fold == f) for f in sorted(np.unique(folds_col).tolist())]
    draws = cv_fold_draws(f_idx, 2000, rng)
    sc_new_raw = cv_scores(cm_new_raw, f_idx, draws)
    sc_new_cal = cv_scores(cm_new_cal, f_idx, draws)
    comparisons = {}
    for ref_name, ref_path in (
        ("vs_75_calibrated", REFERENCE_75),
        ("vs_75_raw", REFERENCE_75_RAW),
        ("vs_73", REFERENCE_73),
    ):
        if not ref_path.exists():
            continue
        ref = pd.read_parquet(ref_path)
        if "prediction" not in ref.columns:
            ref["prediction"] = ref[PROBABILITIES].to_numpy().argmax(axis=1)
        merged = oof[[*KEYS, LABEL, "fold"]].merge(
            ref[[*KEYS, "prediction"]], on=[*KEYS], how="inner"
        )
        if len(merged) != len(oof):
            continue
        cm_ref, pats_ref = patient_confusions(
            merged["subject_id"].astype(str).to_numpy(),
            merged[LABEL].to_numpy(),
            merged["prediction"].to_numpy(),
        )
        if not (pats_ref == pats_raw).all():
            continue
        sc_ref = cv_scores(cm_ref, f_idx, draws)
        d_raw = sc_new_raw["macro_f1"] - sc_ref["macro_f1"]
        d_cal = sc_new_cal["macro_f1"] - sc_ref["macro_f1"]
        comparisons[ref_name] = {
            "raw_macro_f1_delta": {"mean": float(d_raw.mean()), "ci95": interval(d_raw)},
            "calibrated_macro_f1_delta": {"mean": float(d_cal.mean()), "ci95": interval(d_cal)},
        }

    summary = {
        "experiment": 79,
        "name": "cinc-bigru-concept",
        "config": asdict(config),
        "model_params": int(param_count),
        "input_dim": int(TOTAL_CONCEPT_DIM),
        "concept_dims": {k: int(v) for k, v in zip(CONCEPT_NAMES, DEFAULT_CONCEPT_DIMS)},
        "feature_names": list(FEATURE_NAMES_79),
        "raw": raw_metrics,
        "calibrated": cal_metrics,
        "calibration_oof_details": cal_details,
        "conformal": conformal,
        "concept_diagnostics": concept_diag,
        "comparisons": comparisons,
        "fold_histories": {str(k): v for k, v in fold_histories.items()},
        "normalization_counts": {
            str(k): {
                "total_fit_seconds": v["total_fit_seconds"],
                "total_fit_epochs": v["total_fit_epochs"],
            }
            for k, v in norm_summaries.items()
        },
        "peak_rss_mib": round(additional.peak_memory_bytes() / 1024**2, 1),
        "budget_seconds": budget_seconds,
    }
    atomic_json(OUT / "summary.json", summary)
    return summary


def execute(config: TrainConfig, budget_seconds: float, pause_path: Path) -> int:
    """Run all 5 folds with resumable checkpoints and strictly isolated folds."""
    OUT.mkdir(parents=True, exist_ok=True)
    if pause_path.exists():
        log(f"PAUSE marker present at {pause_path}; exiting cleanly.")
        return 75

    dev_all, splits = development_frame()
    added_frame, eligible_added = additional.training_cohort(
        set(dev_all.subject_id.astype(str))
    )
    added_subjects = {str(s) for s in eligible_added}
    unified.assert_holdout_disjoint(
        set(dev_all.subject_id.astype(str).unique()) | added_subjects
    )

    deadline = active_clock() + budget_seconds
    fold_preds: list[pd.DataFrame] = []
    fold_vals: list[pd.DataFrame] = []
    fold_comps: list[pd.DataFrame] = []
    fold_concepts: list[pd.DataFrame] = []
    fold_histories: dict[int, dict] = {}
    norm_summaries: dict[int, dict] = {}
    param_count = 0

    for fold, split in enumerate(splits):
        train, val, test = fold_frames(dev_all, split)
        fold_dir = OUT / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_file = fold_dir / "predictions.parquet"
        val_file = fold_dir / "val_predictions.parquet"
        comp_file = fold_dir / "components.parquet"
        concept_file = fold_dir / "concept_logits.parquet"
        norm_file = fold_dir / "norm_meta.json"
        hist_file = fold_dir / "history.json"

        if (
            pred_file.exists()
            and comp_file.exists()
            and concept_file.exists()
            and hist_file.exists()
            and norm_file.exists()
        ):
            log(f"Fold {fold}: already complete, loading cached outputs.")
            fold_preds.append(pd.read_parquet(pred_file))
            if val_file.exists():
                fold_vals.append(pd.read_parquet(val_file))
            fold_comps.append(pd.read_parquet(comp_file))
            fold_concepts.append(pd.read_parquet(concept_file))
            norm_summaries[fold] = json.loads(norm_file.read_text(encoding="utf-8"))
            fold_histories[fold] = json.loads(hist_file.read_text(encoding="utf-8"))
            param_count = fold_histories[fold].get("param_count", param_count)
            continue

        train_added = added_frame.copy()
        full_fit = pd.concat([train, train_added], ignore_index=True)
        fit_subjects = sorted(full_fit.subject_id.astype(str).unique())
        val_subjects = sorted(val.subject_id.astype(str).unique())
        fold_frame = pd.concat([full_fit, val, test], ignore_index=True)

        log(
            f"Fold {fold}: preparing {len(fit_subjects)} fit + {len(val_subjects)} val + "
            f"{test.subject_id.nunique()} test recordings..."
        )
        recordings, norm_meta = prepare_concept_recordings(
            fold_frame,
            set(fit_subjects),
            lambda: None,
            added=added_subjects,
            occipital_dir=OCCIPITAL_CACHE,
        )
        atomic_json(norm_file, norm_meta)
        norm_summaries[fold] = norm_meta

        model, state = fit_concept_bigru(
            recordings,
            fit_subjects,
            val_subjects,
            SEED + fold,
            fold_dir,
            pause_path,
            deadline,
            config,
        )
        if model is None:
            log(f"Fold {fold}: paused/timed out cleanly at pass {state['next_pass']} batch {state['next_batch']}.")
            return 75

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        verify_concept_model(model)
        pred_df, comp_df, concept_df = score_fold_with_concepts(
            model, recordings, test, fold
        )
        val_df, _, _ = score_fold_with_concepts(model, recordings, val, fold)

        unified.save_frame(pred_file, pred_df)
        unified.save_frame(val_file, val_df)
        unified.save_frame(comp_file, comp_df)
        unified.save_frame(concept_file, concept_df)
        hist_payload = {
            "param_count": int(param_count),
            "best": float(state["best"]),
            "best_pass": int(state["best_pass"]),
            "pass_seconds": float(state["pass_seconds"]),
            "history": state["history"],
        }
        atomic_json(hist_file, hist_payload)

        fold_preds.append(pred_df)
        fold_vals.append(val_df)
        fold_comps.append(comp_df)
        fold_concepts.append(concept_df)
        fold_histories[fold] = hist_payload

        del recordings, model
        gc.collect()

    oof = pd.concat(fold_preds, ignore_index=True)
    val_oof = pd.concat(fold_vals, ignore_index=True) if fold_vals else pd.DataFrame()
    comp_df = pd.concat(fold_comps, ignore_index=True)
    concept_df = pd.concat(fold_concepts, ignore_index=True)
    summary = report(
        oof,
        val_oof,
        comp_df,
        concept_df,
        fold_histories,
        norm_summaries,
        param_count,
        config,
        budget_seconds,
    )
    log(
        f"EXPERIMENT 79 COMPLETE: params={param_count}, "
        f"raw macro-F1={summary['raw']['mean_fold_macro_f1']:.5f}, "
        f"calibrated macro-F1={summary['calibrated']['mean_fold_macro_f1']:.5f}, "
        f"kappa={summary['calibrated']['kappa']:.5f}, "
        f"local-only={summary['concept_diagnostics']['local_only_macro_f1']:.5f}, "
        f"rss={summary['peak_rss_mib']:.1f} MiB"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=BUDGET_SECONDS)
    parser.add_argument("--max-passes", type=int, default=18)
    parser.add_argument("--width-per-concept", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--weight-power", type=float, default=0.45)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=0.0025)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        with local_training_slot():
            smoke()
        return 0
    cfg = TrainConfig(
        width_per_concept=args.width_per_concept,
        num_layers=args.num_layers,
        weight_power=args.weight_power,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        lr=args.lr,
        max_passes=args.max_passes,
    )
    with local_training_slot():
        return execute(cfg, args.budget_seconds, OUT / "PAUSE")


if __name__ == "__main__":
    raise SystemExit(main())
