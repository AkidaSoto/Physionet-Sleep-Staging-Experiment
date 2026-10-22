"""Epoch frames and fine-resolution context windows for the window staging model.

The window model scores each 30-s epoch from what happens around it: 14 one-second tracks (algorithms.eeg_tracks)
plus the 6-s staging features, over the epoch and two neighbours on each side (150 s). Seconds are averaged into
6-s steps, so every epoch is described by 25 steps x 33 signals = 825 numbers that keep their meaning
("sigma share 12 s before the epoch starts").

Normalisation, in order:
  1. per subject: tracks and features scaled by the subject's median and IQR over the whole night
     (ratio and topography features stay raw, as in tabular_specs.FEATURE_NORM_OVERRIDES_BY_PREFIX);
  2. per training fold: z-score with the training subjects' mean and SD, clip to +-10, missing -> 0 (the mean).
Seconds outside the recording are padded with 0 (the training mean).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SUBJ, TIME, TARGET = "subject_id", "time_seconds", "label.stage_30s"
NO_NORM_PREFIXES = ("epoch_ratio_features.", "topography_features.")
EPOCH_SECONDS = 30
ROW_SECONDS = 6

BASE16 = [
    "spindle_features.aperiodic_exponent", "spindle_features.aperiodic_intercept", "spindle_features.delta_power",
    "spindle_features.theta_residual_power", "spindle_features.alpha_residual_power",
    "spindle_features.beta_residual_power", "spindle_features.gamma_residual_power",
    "epoch_ratio_features.rel_theta", "epoch_ratio_features.rel_beta",
    "eye_movement_activity.eye_movement_activity", "eye_movement_activity.eye_movement_baseline",
    "epoch_ratio_features.eog_rem_vs_noise_db", "epoch_ratio_features.eog_rem_vs_sem_db",
    "epoch_ratio_features.emg_floor_db", "topography_features.frontal_slow_wave_fraction",
    "topography_features.occipital_rel_alpha",
]
CARDIO3 = [
    "hrv_features.hr_slow_sd_300s",
    "cardiorespiratory_coupling.hr_resp_coherence_120s",
    "breath_regularity.breath_interval_cv_60s",
]
WINDOW_FEATURES = BASE16 + CARDIO3
SPINDLE_BURSTS = "spindle_bursts"


def spindle_burst_counts(tracks: pd.DataFrame) -> np.ndarray:
    """Seconds per 30-s epoch (epoch k = seconds 30k..30k+29) where the central spindle-band ratio
    c_rel_sigma / (c_rel_alpha + c_rel_beta) exceeds the night's median by more than 2 robust SDs (1.4826 x MAD).
    Validated in staging_window_model/cinc2018/feature_reduction (N2 vs Wake d = 1.24)."""
    ratio = (tracks["c_rel_sigma"] / (tracks["c_rel_alpha"] + tracks["c_rel_beta"] + 1e-6)).to_numpy(np.float32)
    med = np.nanmedian(ratio)
    burst = (ratio > med + 2 * 1.4826 * np.nanmedian(np.abs(ratio - med))).astype(np.float32)
    n = -(-len(burst) // EPOCH_SECONDS)
    return np.r_[burst, np.zeros(n * EPOCH_SECONDS - len(burst), np.float32)].reshape(n, EPOCH_SECONDS).sum(1)


# Epoch-level features computed from the 1-s tracks rather than read from the 6-s rows: name -> per-epoch values.
DERIVED_FEATURES = {SPINDLE_BURSTS: spindle_burst_counts}


def _normalised(col: str) -> bool:
    return not col.startswith(NO_NORM_PREFIXES)


def robust_scale(X: np.ndarray, columns: list[str]) -> np.ndarray:
    """Median/IQR scaling of one subject's array, column by column; ratio and topography columns untouched."""
    X = np.asarray(X, np.float32).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(X, 0)
        iqr = np.nanpercentile(X, 75, 0) - np.nanpercentile(X, 25, 0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
    for j, col in enumerate(columns):
        if _normalised(col):
            X[:, j] = (X[:, j] - med[j]) / iqr[j]
    return X


# ---------------------------------------------------------------- 6-s rows and epochs
def subject_rows6(dataset: str, subject_id: str, columns: list[str], cache_dir: Path) -> pd.DataFrame:
    """6-s feature rows of one subject with their slot (time / 6 s) and per-row stage label, cached as parquet."""
    path = Path(cache_dir) / f"{subject_id}.parquet"
    if path.exists():
        rows = pd.read_parquet(path)
        if all(c in rows.columns for c in columns):
            return rows
    from physionet_sleep.analysis.datasets import load_reference_dataset

    t = load_reference_dataset(dataset, subject_ids=[subject_id])
    t = t.sort_values(TIME, kind="stable").reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    rows = pd.DataFrame({SUBJ: str(subject_id), TIME: t[TIME].to_numpy(float)})
    rows["slot"] = np.rint(rows[TIME].to_numpy() / ROW_SECONDS).astype(int)
    for c in columns:
        rows[c] = t[c].to_numpy(float) if c in t.columns else np.nan
    rows["stage_sec"] = pd.to_numeric(t["label.stage_seconds"], errors="coerce").to_numpy(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    return rows


def subject_epochs(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Labelled 30-s epochs: five consecutive, aligned 6-s rows with a majority stage; feature columns averaged."""
    per = EPOCH_SECONDS // ROW_SECONDS
    nb = len(rows) // per
    rows = rows.iloc[: nb * per]
    slots = rows["slot"].to_numpy().reshape(nb, per)
    ok = (slots[:, 0] % per == 0) & np.all(slots == slots[:, :1] + np.arange(per), axis=1)
    lab = rows["stage_sec"].to_numpy().reshape(nb, per)
    labels = np.full(nb, -1)
    for i in range(nb):
        v = lab[i][np.isfinite(lab[i]) & (lab[i] >= 0)]
        if v.size:
            u, n = np.unique(np.rint(v).astype(int), return_counts=True)
            labels[i] = u[np.argmax(n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means = np.nanmean(rows[columns].to_numpy(float).reshape(nb, per, len(columns)), axis=1)
    ep = pd.DataFrame(means, columns=columns)
    ep.insert(0, "epoch_index", slots[:, 0] // per)
    ep.insert(0, TIME, rows[TIME].to_numpy().reshape(nb, per)[:, 0])
    ep.insert(0, SUBJ, str(rows[SUBJ].iloc[0]) if len(rows) else "")
    ep[TARGET] = labels
    return ep[ok & (labels >= 0)].reset_index(drop=True)


def epoch_frame(rows_by_subject: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """All subjects' labelled epochs, with epoch-mean features scaled per subject (the epoch-only reference model)."""
    frames = [subject_epochs(r, columns) for r in rows_by_subject.values()]
    epochs = pd.concat(frames, ignore_index=True)
    norm = [c for c in columns if _normalised(c)]
    g = epochs.groupby(SUBJ)[norm]
    iqr = g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25)).replace(0, np.nan).fillna(1.0)
    epochs[norm] = (epochs[norm] - g.transform("median")) / iqr
    epochs[TARGET] = epochs[TARGET].astype(int)
    return epochs


# ---------------------------------------------------------------- dense 1-s signals and windows
def dense_signals(tracks: pd.DataFrame, rows: pd.DataFrame, track_cols: list[str], feature_cols: list[str]) -> np.ndarray:
    """(n_seconds, tracks + features) for one subject: subject-scaled tracks, then 6-s features held for 6 s.

    Features in DERIVED_FEATURES are computed from the tracks, held across their 30-s epoch and subject-scaled over
    the night's seconds; they must come after the 6-s features in feature_cols."""
    X = robust_scale(tracks[track_cols].to_numpy(np.float32), track_cols)
    row_cols = [c for c in feature_cols if c not in DERIVED_FEATURES]
    derived = [c for c in feature_cols if c in DERIVED_FEATURES]
    if feature_cols != row_cols + derived:
        raise ValueError("derived features must follow the 6-s features")
    parts = [X]
    if row_cols:
        n6 = int(rows["slot"].max()) + 1
        F = np.full((n6, len(row_cols)), np.nan, np.float32)
        F[rows["slot"].to_numpy()] = robust_scale(rows[row_cols].to_numpy(np.float32), row_cols)
        held = np.repeat(F, ROW_SECONDS, axis=0)[: len(X)]
        if len(held) < len(X):
            held = np.vstack([held, np.full((len(X) - len(held), len(row_cols)), np.nan, np.float32)])
        parts.append(held)
    for c in derived:
        held = np.repeat(DERIVED_FEATURES[c](tracks), EPOCH_SECONDS)[: len(X)]
        parts.append(robust_scale(held[:, None], [c]))
    return np.hstack(parts)


@dataclass
class WindowEncoder:
    """Turns dense per-subject signals into (epochs, steps, signals) windows of `context` epochs on each side."""

    signal_names: list[str]
    context: int = 2
    step_seconds: int = ROW_SECONDS
    mean: np.ndarray | None = None
    sd: np.ndarray | None = None

    @property
    def steps_per_epoch(self) -> int:
        return EPOCH_SECONDS // self.step_seconds

    @property
    def n_steps(self) -> int:
        return (2 * self.context + 1) * self.steps_per_epoch

    def fit(self, dense: dict[str, np.ndarray], subjects: Iterable[str]) -> "WindowEncoder":
        allX = np.vstack([dense[s] for s in subjects])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mean = np.nanmean(allX, 0)
            self.sd = np.nanstd(allX, 0) + 1e-6
        return self

    def _steps(self, X: np.ndarray) -> np.ndarray:
        """Standardise, clip, fill, and average one subject's seconds into steps; padded by `context` epochs."""
        Z = np.nan_to_num(np.clip((X - self.mean) / self.sd, -10, 10)).astype(np.float32)
        n = -(-len(Z) // self.step_seconds)  # a final partial step is averaged with zero padding
        Z = np.vstack([Z, np.zeros((n * self.step_seconds - len(Z), Z.shape[1]), np.float32)])
        S = Z.reshape(n, self.step_seconds, Z.shape[1]).mean(1)
        pad = np.zeros(((self.context + 1) * self.steps_per_epoch, Z.shape[1]), np.float32)
        return np.vstack([pad, S, pad])

    def transform(self, dense: dict[str, np.ndarray], epochs: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(epochs), self.n_steps, len(self.signal_names)), np.float32)
        subj = epochs[SUBJ].astype(str).to_numpy()
        ei = epochs["epoch_index"].to_numpy()
        pad = (self.context + 1) * self.steps_per_epoch
        for s in pd.unique(subj):
            idx = np.flatnonzero(subj == s)
            S = self._steps(dense[s])
            start = pad + (ei[idx] - self.context) * self.steps_per_epoch
            gather = start[:, None] + np.arange(self.n_steps)[None, :]
            gather = np.clip(gather, 0, len(S) - 1)
            out[idx] = S[gather]
        return out

    def column_names(self) -> list[str]:
        """Flattened names, step-major: '<signal>@<offset s>', offset from the start of the scored epoch."""
        offsets = [(t - self.context * self.steps_per_epoch) * self.step_seconds for t in range(self.n_steps)]
        return [f"{name}@{o:+d}s" for o in offsets for name in self.signal_names]
