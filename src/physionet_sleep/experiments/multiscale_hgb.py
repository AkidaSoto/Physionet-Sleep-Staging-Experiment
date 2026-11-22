"""Streaming multi-scale tempered HistGradientBoosting pipeline for Experiment 76 (cinc-hgb-704)."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from physionet_sleep.analysis.epoch_windows import (
    DERIVED_FEATURES,
    EPOCH_SECONDS,
    ROW_SECONDS,
    SPINDLE_BURSTS,
    dense_signals,
)
from physionet_sleep.experiments.complementary_n2_events import N2_EVENT_COLUMNS
from physionet_sleep.experiments.complementary_slow_waves import SLOW_WAVE_PANEL_COLUMNS


SEQUENCE_TRACKS = (
    "c_rel_theta",
    "c_rel_alpha",
    "c_rel_beta",
    "c_log_power",
    "o_rel_alpha",
    "eog_high",
)
EXTRA_TRACKS = ("c_rel_sigma", "sigma_ratio")
FAST_TRACKS = (
    "c_rel_theta",
    "c_rel_alpha",
    "c_rel_sigma",
    "c_rel_beta",
    "c_log_power",
    "o_rel_alpha",
    "eog_high",
    "sigma_ratio",
)
SEQUENCE_FEATURES = (
    "spindle_features.aperiodic_exponent",
    "spindle_features.delta_power",
    "epoch_ratio_features.rel_theta",
    "epoch_ratio_features.rel_beta",
    "eye_movement_activity.eye_movement_activity",
    "eye_movement_activity.eye_movement_baseline",
    "epoch_ratio_features.eog_rem_vs_noise_db",
    "epoch_ratio_features.eog_rem_vs_sem_db",
    "epoch_ratio_features.emg_floor_db",
    "topography_features.frontal_slow_wave_fraction",
    "topography_features.occipital_rel_alpha",
    SPINDLE_BURSTS,
)
BASE_SIGNALS = (*SEQUENCE_TRACKS, *EXTRA_TRACKS, *SEQUENCE_FEATURES)
NON_FAST_SIGNALS = SEQUENCE_FEATURES
TOP10_SIGNALS = (
    *FAST_TRACKS,
    "topography_features.frontal_slow_wave_fraction",
    "epoch_ratio_features.emg_floor_db",
)
BLOCK_WINDOWS = ((-5, -3), (3, 5), (-10, -6), (6, 10), (-20, -11), (11, 20))
MARKER_COLUMNS = (*SLOW_WAVE_PANEL_COLUMNS, *N2_EVENT_COLUMNS)
NIGHT_HISTORY_COLUMNS = (
    "elapsed_hours",
    "elapsed_squared",
    "fraction_elapsed",
    "slow_burden_vs_clock",
    "slow_coverage_prior_30min",
    "atonia_fraction_10min",
    "nearby_eye_atonia",
)
OCCIPITAL_COLUMN = "o_theta_30s"

SLOW_COL = "topography_features.frontal_slow_wave_fraction"
CHIN_COL = "epoch_ratio_features.emg_floor_db"
EYE_COL = "epoch_ratio_features.eog_rem_vs_noise_db"

STEPS_PER_EPOCH = EPOCH_SECONDS // ROW_SECONDS
FAST_INDICES = np.array([BASE_SIGNALS.index(s) for s in FAST_TRACKS], dtype=np.int64)
NON_FAST_INDICES = np.array([BASE_SIGNALS.index(s) for s in NON_FAST_SIGNALS], dtype=np.int64)
TOP10_INDICES = np.array([BASE_SIGNALS.index(s) for s in TOP10_SIGNALS], dtype=np.int64)
MAX_BLOCK_CONTEXT = 20
DEFAULT_WEIGHT_POWER = 0.50


@dataclass(frozen=True)
class CacheLayout:
    """Resolve per-subject parquet locations across development and added training caches."""

    dev_window_cache: Path
    added_window_cache: Path
    dev_marker_cache: Path
    added_marker_cache: Path
    added_subjects: frozenset[str]
    occipital_cache: Path | None = None

    def window_cache(self, subject: str) -> Path:
        return self.added_window_cache if subject in self.added_subjects else self.dev_window_cache

    def marker_path(self, subject: str) -> Path:
        root = self.added_marker_cache if subject in self.added_subjects else self.dev_marker_cache
        return root / f"{subject}.parquet"

    def occipital_path(self, subject: str) -> Path | None:
        if self.occipital_cache is None:
            return None
        return self.occipital_cache / f"{subject}.parquet"


@dataclass
class FoldMoments:
    """Per-fold standardization statistics fitted strictly on fitting subjects."""

    signal_mean: np.ndarray
    signal_sd: np.ndarray
    marker_mean: np.ndarray
    marker_sd: np.ndarray
    occipital_mean: float | None
    occipital_sd: float | None
    include_occipital: bool
    total_fit_seconds: int
    finite_fit_seconds: list[int]
    finite_fit_marker_epochs: list[int]


def _short_name(signal: str) -> str:
    return signal.split(".")[-1]


def feature_column_names(include_occipital: bool = False) -> list[str]:
    """Return ordered, unique feature column names matching encode_subject_features."""
    names: list[str] = []
    # 5 x 20 = 100 center epoch 6-s steps
    for step in range(STEPS_PER_EPOCH):
        offset = step * ROW_SECONDS
        for sig in BASE_SIGNALS:
            names.append(f"{_short_name(sig)}@{offset:+d}s")
    # 5 x 8 = 40 steps at -1 epoch and 5 x 8 = 40 steps at +1 epoch
    for ep_offset in (-1, 1):
        for step in range(STEPS_PER_EPOCH):
            offset = (ep_offset * STEPS_PER_EPOCH + step) * ROW_SECONDS
            for sig in FAST_TRACKS:
                names.append(f"{_short_name(sig)}@{offset:+d}s")
    # 2 x 12 = 24 epoch means at -1 and +1 for non-fast signals
    for ep_offset in (-1, 1):
        for sig in NON_FAST_SIGNALS:
            names.append(f"{_short_name(sig)}@ep{ep_offset:+d}")
    # 2 x 10 = 20 epoch means at -2 and +2 for top 10 signals
    for ep_offset in (-2, 2):
        for sig in TOP10_SIGNALS:
            names.append(f"{_short_name(sig)}@ep{ep_offset:+d}")
    # 6 x 10 = 60 multi-minute block means for top 10 signals
    for start, end in BLOCK_WINDOWS:
        for sig in TOP10_SIGNALS:
            names.append(f"{_short_name(sig)}@ep[{start:+d}..{end:+d}]")
    # 3 x 5 = 15 complementary markers at -1, 0, +1
    for ep_offset in (-1, 0, 1):
        for marker in MARKER_COLUMNS:
            names.append(f"{marker}@ep{ep_offset:+d}")
    # 7 night-clock and label-free physiology cues
    names.extend(NIGHT_HISTORY_COLUMNS)
    # Optional 3 occipital theta columns
    if include_occipital:
        for ep_offset in (-1, 0, 1):
            names.append(f"{OCCIPITAL_COLUMN}@ep{ep_offset:+d}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature column names generated")
    return names


def has_occipital_cache(occipital_cache: Path | None, subjects: Iterable[str]) -> bool:
    """Return True only when occipital_cache exists and has parquet files for every subject."""
    if occipital_cache is None or not occipital_cache.is_dir():
        return False
    return all((occipital_cache / f"{s}.parquet").is_file() for s in subjects)


def compute_night_history(row_frame: pd.DataFrame, n_seconds: int) -> np.ndarray:
    """Compute #51's 7 night-clock and label-free physiology cues from 6-s rows."""
    n = max(1, -(-int(n_seconds) // EPOCH_SECONDS))
    observed = np.full((n * STEPS_PER_EPOCH, 3), np.nan, dtype=np.float64)
    if len(row_frame):
        slots = row_frame["slot"].to_numpy()
        mask = (slots >= 0) & (slots < len(observed))
        if mask.any():
            observed[slots[mask]] = row_frame[[SLOW_COL, CHIN_COL, EYE_COL]].to_numpy(dtype=np.float64)[mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        slow, chin, eye = np.nanmean(observed.reshape(n, STEPS_PER_EPOCH, 3), axis=1).T
    t = np.arange(n)
    valid_chin = np.isfinite(chin)
    valid_eye = np.isfinite(eye)
    q_chin = np.nanquantile(chin, 0.25) if valid_chin.any() else -np.inf
    q_eye = np.nanquantile(eye, 0.75) if valid_eye.any() else np.inf
    atonia = valid_chin & (chin <= q_chin)
    anchor = np.flatnonzero(atonia & valid_eye & (eye > q_eye))
    end_before = np.maximum(0, t - 2)
    start_after = np.minimum(n, t + 3)
    slow_sum = np.r_[0.0, np.nancumsum(np.clip(slow, 0.0, 1.0))]
    slow_n = np.r_[0, np.cumsum(np.isfinite(slow))]
    since = np.maximum(0, end_before - 60)
    count = slow_n[end_before] - slow_n[since]
    prior_slow = np.divide(slow_sum[end_before] - slow_sum[since], count, out=np.zeros(n), where=count > 0)
    burden = slow_sum[end_before] / slow_sum[-1] - end_before / n if slow_sum[-1] > 0 else np.zeros(n)
    chin_end, chin_start = np.minimum(n, t + 11), np.maximum(0, t - 10)
    low_n = np.r_[0, np.cumsum(atonia)]
    total_n = np.r_[0, np.cumsum(valid_chin)]
    left = low_n[end_before] - low_n[chin_start]
    right = low_n[chin_end] - low_n[start_after]
    count_chin = total_n[end_before] - total_n[chin_start] + total_n[chin_end] - total_n[start_after]
    persistence = np.divide(left + right, count_chin, out=np.zeros(n), where=count_chin > 0)
    prev = np.r_[-10 * n, anchor]
    nex = np.r_[anchor, 10 * n]
    left_gap = t - prev[np.searchsorted(anchor, end_before, side="left")]
    right_gap = nex[np.searchsorted(anchor, start_after, side="left")] - t
    close_anchor = 1.0 - np.minimum(np.minimum(left_gap, right_gap), 60) / 60.0
    hours = t / 120.0
    out = np.column_stack([hours, hours**2, t / max(1, n - 1), burden, prior_slow, persistence, close_anchor])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _read_markers(path: Path, subject: str) -> np.ndarray:
    panel = pd.read_parquet(path)
    if (
        panel.empty
        or not panel.columns.is_unique
        or set(panel.columns) != set(MARKER_COLUMNS)
        or panel.index.name != "epoch_index"
        or not panel.index.equals(pd.RangeIndex(len(panel)))
    ):
        raise ValueError(f"{subject}: marker panel differs from the #60/#61 feature contract")
    values = panel.loc[:, list(MARKER_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _read_occipital(path: Path, subject: str) -> np.ndarray:
    frame = pd.read_parquet(path)
    if OCCIPITAL_COLUMN in frame.columns:
        if "epoch_index" in frame.columns:
            frame = frame.sort_values("epoch_index", kind="stable").set_index("epoch_index")
        if not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError(f"{subject}: occipital cache requires contiguous epoch_index 0..N-1")
        values = frame[OCCIPITAL_COLUMN].to_numpy(dtype=np.float32, copy=True)
    elif "o_rel_theta" in frame.columns:
        sec_vals = frame["o_rel_theta"].to_numpy(dtype=np.float32)
        n_epochs = -(-len(sec_vals) // EPOCH_SECONDS)
        padded = np.pad(sec_vals, (0, n_epochs * EPOCH_SECONDS - len(sec_vals)), constant_values=np.nan)
        blocks = padded.reshape(n_epochs, EPOCH_SECONDS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.nanmean(blocks, axis=1).astype(np.float32)
    else:
        raise ValueError(f"{subject}: occipital cache missing {OCCIPITAL_COLUMN} or o_rel_theta")
    values[~np.isfinite(values)] = np.nan
    return values


def load_subject_night(
    layout: CacheLayout,
    subject: str,
    *,
    include_occipital: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load one night's 20 dense 1-Hz signals, #60 markers, #51 history, and optional occipital theta."""
    wcache = layout.window_cache(subject)
    raw_tracks = [*SEQUENCE_TRACKS, "c_rel_sigma"]
    track_cols = list(dict.fromkeys([*raw_tracks, "c_rel_alpha", "c_rel_beta"]))
    row_cols = ["slot", *(name for name in SEQUENCE_FEATURES if name not in DERIVED_FEATURES)]
    track_frame = pd.read_parquet(wcache / "tracks1s" / f"{subject}.parquet", columns=track_cols)
    row_frame = pd.read_parquet(wcache / "rows6" / f"{subject}.parquet", columns=row_cols)
    track_frame["sigma_ratio"] = (
        track_frame["c_rel_sigma"] / (track_frame["c_rel_alpha"] + track_frame["c_rel_beta"] + 1e-6)
    ).astype(np.float32)

    tracks_for_dense = [*SEQUENCE_TRACKS, *EXTRA_TRACKS]
    dense = dense_signals(track_frame, row_frame, tracks_for_dense, list(SEQUENCE_FEATURES))
    markers = _read_markers(layout.marker_path(subject), subject)
    history = compute_night_history(row_frame, len(dense))
    occipital = None
    if include_occipital:
        occ_path = layout.occipital_path(subject)
        if occ_path is None or not occ_path.is_file():
            raise FileNotFoundError(f"{subject}: missing occipital cache at {occ_path}")
        occipital = _read_occipital(occ_path, subject)
    return dense, markers, history, occipital


def fit_fold_moments(
    layout: CacheLayout,
    fit_subjects: Iterable[str],
    *,
    include_occipital: bool = False,
    check_fn: Callable[[], None] | None = None,
) -> FoldMoments:
    """Streaming Pass 1: estimate fold mean/SD using float64 accumulators, one night at a time."""
    n_sig = len(BASE_SIGNALS)
    n_mrk = len(MARKER_COLUMNS)
    sig_count = np.zeros(n_sig, dtype=np.int64)
    sig_sum = np.zeros(n_sig, dtype=np.float64)
    sig_sq = np.zeros(n_sig, dtype=np.float64)
    mrk_count = np.zeros(n_mrk, dtype=np.int64)
    mrk_sum = np.zeros(n_mrk, dtype=np.float64)
    mrk_sq = np.zeros(n_mrk, dtype=np.float64)
    occ_count = 0
    occ_sum = 0.0
    occ_sq = 0.0
    fit_seconds = 0

    for subject in sorted(fit_subjects):
        if check_fn is not None:
            check_fn()
        dense, markers, _, occipital = load_subject_night(layout, subject, include_occipital=include_occipital)
        fit_seconds += len(dense)
        sig_count += np.isfinite(dense).sum(axis=0)
        sig_sum += np.nansum(dense, axis=0, dtype=np.float64)
        sig_sq += np.nansum(np.square(dense, dtype=np.float64), axis=0)

        n_epochs = -(-len(dense) // EPOCH_SECONDS)
        m_valid = markers[:n_epochs]
        mrk_count += np.isfinite(m_valid).sum(axis=0)
        mrk_sum += np.nansum(m_valid, axis=0, dtype=np.float64)
        mrk_sq += np.nansum(np.square(m_valid, dtype=np.float64), axis=0)

        if include_occipital and occipital is not None:
            o_valid = occipital[:n_epochs]
            finite = np.isfinite(o_valid)
            occ_count += int(finite.sum())
            occ_sum += float(np.nansum(o_valid, dtype=np.float64))
            occ_sq += float(np.nansum(np.square(o_valid, dtype=np.float64)))
        del dense, markers, occipital

    if not np.all(sig_count > 0):
        raise ValueError("Every base signal must have observed values in the fitting patients")
    if not np.all(mrk_count > 0):
        raise ValueError("Every complementary marker must have observed values in the fitting patients")

    sig_mean = sig_sum / sig_count
    sig_sd = np.sqrt(np.maximum(0.0, sig_sq / sig_count - sig_mean**2)) + 1e-6
    mrk_mean = mrk_sum / mrk_count
    mrk_sd = np.sqrt(np.maximum(0.0, mrk_sq / mrk_count - mrk_mean**2)) + 1e-6

    occ_mean = occ_sd = None
    if include_occipital:
        if occ_count == 0:
            raise ValueError("No observed values for optional o_theta_30s in the fitting patients")
        occ_mean = occ_sum / occ_count
        occ_sd = float(np.sqrt(max(0.0, occ_sq / occ_count - occ_mean**2)) + 1e-6)

    return FoldMoments(
        signal_mean=sig_mean.astype(np.float32),
        signal_sd=sig_sd.astype(np.float32),
        marker_mean=mrk_mean.astype(np.float32),
        marker_sd=mrk_sd.astype(np.float32),
        occipital_mean=occ_mean,
        occipital_sd=occ_sd,
        include_occipital=include_occipital,
        total_fit_seconds=fit_seconds,
        finite_fit_seconds=sig_count.tolist(),
        finite_fit_marker_epochs=mrk_count.tolist(),
    )


def encode_subject_features(
    dense: np.ndarray,
    markers: np.ndarray,
    history: np.ndarray,
    occipital: np.ndarray | None,
    epoch_indices: np.ndarray,
    moments: FoldMoments,
) -> np.ndarray:
    """Build multi-scale float32 features for one subject's scored epochs."""
    idx = np.asarray(epoch_indices, dtype=np.int64)
    n_sig = len(BASE_SIGNALS)
    n_epochs = -(-len(dense) // EPOCH_SECONDS)
    if len(idx) == 0:
        return np.empty((0, len(feature_column_names(moments.include_occipital))), dtype=np.float32)
    if idx.min() < 0 or idx.max() >= n_epochs:
        raise ValueError("Scored epoch index extends outside the recording length")
    if idx.max() >= len(markers):
        raise ValueError("Scored epoch index extends beyond complementary marker stream")
    if idx.max() >= len(history):
        raise ValueError("Scored epoch index extends beyond night history stream")

    z_sec = np.nan_to_num(
        np.clip((dense - moments.signal_mean) / moments.signal_sd, -10.0, 10.0),
        nan=0.0,
    ).astype(np.float32)
    pad_sec = n_epochs * EPOCH_SECONDS - len(z_sec)
    if pad_sec:
        z_sec = np.vstack([z_sec, np.zeros((pad_sec, n_sig), dtype=np.float32)])
    steps = z_sec.reshape(n_epochs, STEPS_PER_EPOCH, ROW_SECONDS, n_sig).mean(axis=2, dtype=np.float32)
    del z_sec

    m = len(idx)
    n_cols = len(feature_column_names(moments.include_occipital))
    out = np.empty((m, n_cols), dtype=np.float32)
    col = 0

    steps_pad = np.pad(steps, ((1, 1), (0, 0), (0, 0)), mode="constant", constant_values=0.0)
    # 1) Center 6-s steps (100 cols)
    out[:, col : col + 100] = steps_pad[idx + 1].reshape(m, 100)
    col += 100
    # 2) +/-1 epoch 6-s steps for 8 fast tracks (80 cols)
    out[:, col : col + 40] = steps_pad[idx][:, :, FAST_INDICES].reshape(m, 40)
    col += 40
    out[:, col : col + 40] = steps_pad[idx + 2][:, :, FAST_INDICES].reshape(m, 40)
    col += 40
    del steps_pad

    # 3) Epoch means & multi-scale block means
    ep_means = steps.mean(axis=1, dtype=np.float32)
    del steps
    ep_pad = np.pad(
        ep_means,
        ((MAX_BLOCK_CONTEXT, MAX_BLOCK_CONTEXT), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    del ep_means
    p = idx + MAX_BLOCK_CONTEXT
    for ep_offset in (-1, 1):
        out[:, col : col + len(NON_FAST_INDICES)] = ep_pad[p + ep_offset][:, NON_FAST_INDICES]
        col += len(NON_FAST_INDICES)
    for ep_offset in (-2, 2):
        out[:, col : col + len(TOP10_INDICES)] = ep_pad[p + ep_offset][:, TOP10_INDICES]
        col += len(TOP10_INDICES)

    cs = np.vstack([
        np.zeros((1, len(TOP10_INDICES)), dtype=np.float64),
        np.cumsum(ep_pad[:, TOP10_INDICES], axis=0, dtype=np.float64),
    ])
    del ep_pad
    for start, end in BLOCK_WINDOWS:
        width = float(end - start + 1)
        out[:, col : col + len(TOP10_INDICES)] = ((cs[p + end + 1] - cs[p + start]) / width).astype(np.float32)
        col += len(TOP10_INDICES)
    del cs

    # 4) #60 complementary markers at -1, 0, +1 (15 cols)
    mrk_norm = np.nan_to_num(
        np.clip((markers[:n_epochs] - moments.marker_mean) / moments.marker_sd, -10.0, 10.0),
        nan=0.0,
    ).astype(np.float32)
    if len(mrk_norm) < n_epochs:
        mrk_norm = np.vstack([mrk_norm, np.zeros((n_epochs - len(mrk_norm), len(MARKER_COLUMNS)), dtype=np.float32)])
    mrk_pad = np.pad(mrk_norm, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
    for ep_offset in (-1, 0, 1):
        out[:, col : col + len(MARKER_COLUMNS)] = mrk_pad[idx + 1 + ep_offset]
        col += len(MARKER_COLUMNS)
    del mrk_norm, mrk_pad

    # 5) #51 night history (7 cols)
    out[:, col : col + len(NIGHT_HISTORY_COLUMNS)] = history[idx]
    col += len(NIGHT_HISTORY_COLUMNS)

    # 6) Optional occipital theta at -1, 0, +1 (3 cols)
    if moments.include_occipital:
        if occipital is None or idx.max() >= len(occipital):
            raise ValueError("Missing or truncated optional occipital cache")
        occ_norm = np.nan_to_num(
            np.clip((occipital[:n_epochs] - moments.occipital_mean) / moments.occipital_sd, -10.0, 10.0),
            nan=0.0,
        ).astype(np.float32)
        if len(occ_norm) < n_epochs:
            occ_norm = np.concatenate([occ_norm, np.zeros(n_epochs - len(occ_norm), dtype=np.float32)])
        occ_pad = np.pad(occ_norm, (1, 1), mode="constant", constant_values=0.0)
        for ep_offset in (-1, 0, 1):
            out[:, col] = occ_pad[idx + 1 + ep_offset]
            col += 1

    if col != n_cols:
        raise RuntimeError(f"Column count mismatch: wrote {col}, expected {n_cols}")
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite features produced by encode_subject_features")
    return out


def build_fit_matrix(
    layout: CacheLayout,
    fit_frame: pd.DataFrame,
    moments: FoldMoments,
    label_col: str = "label.stage_30s",
    *,
    check_fn: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming Pass 2: pre-allocate float32 X_fit and fill one night at a time."""
    n_rows = len(fit_frame)
    n_cols = len(feature_column_names(moments.include_occipital))
    X_fit = np.empty((n_rows, n_cols), dtype=np.float32)
    y_fit = np.empty(n_rows, dtype=np.int64)
    for subject, positions in fit_frame.groupby("subject_id", sort=False).indices.items():
        if check_fn is not None:
            check_fn()
        sub_frame = fit_frame.iloc[positions]
        dense, markers, history, occipital = load_subject_night(
            layout, str(subject), include_occipital=moments.include_occipital
        )
        X_fit[positions] = encode_subject_features(
            dense, markers, history, occipital, sub_frame["epoch_index"].to_numpy(), moments
        )
        y_fit[positions] = sub_frame[label_col].to_numpy(dtype=np.int64)
        del dense, markers, history, occipital
    return X_fit, y_fit


def compute_tempered_weights(
    y_fit: np.ndarray,
    n_classes: int = 5,
    weight_power: float = DEFAULT_WEIGHT_POWER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized sample weights proportional to counts[y] ** (-weight_power)."""
    counts = np.bincount(y_fit, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"Every stage must occur in the fitting pool; got counts={counts.tolist()}")
    class_weights = counts.astype(np.float64) ** (-float(weight_power))
    sample_weight = class_weights[y_fit]
    sample_weight = sample_weight / sample_weight.mean()
    return sample_weight.astype(np.float64), counts, class_weights


def fit_tempered_hgb(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
    max_iter: int = 350,
    memmap_threshold: int = 100_000,
) -> HistGradientBoostingClassifier:
    """Fit HistGradientBoostingClassifier with bounded RAM via explicit validation split + memmap for large cohorts."""
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=seed,
    )
    if len(y_fit) <= memmap_threshold:
        clf.fit(X_fit, y_fit, sample_weight=sample_weight)
        return clf

    # Split 90/10 outside fit() and stream X_train to a float64 memmap so scikit-learn
    # does not simultaneously hold float32 + float64 + train_test_split copies (~4.3 GiB).
    indices = np.arange(len(y_fit))
    tr_idx, va_idx = train_test_split(indices, test_size=0.1, stratify=y_fit, random_state=seed)
    X_va = X_fit[va_idx].astype(np.float64)
    y_va = y_fit[va_idx]
    sw_va = sample_weight[va_idx]
    y_tr = y_fit[tr_idx]
    sw_tr = sample_weight[tr_idx]

    with TemporaryDirectory(prefix="hgb-memmap-") as tmp_dir:
        mm_path = Path(tmp_dir) / "x_train.dat"
        X_tr = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(len(tr_idx), X_fit.shape[1]))
        chunk = 50_000
        for start in range(0, len(tr_idx), chunk):
            end = min(len(tr_idx), start + chunk)
            X_tr[start:end] = X_fit[tr_idx[start:end]]
        X_tr.flush()
        del X_fit
        gc.collect()
        clf.fit(X_tr, y_tr, sample_weight=sw_tr, X_val=X_va, y_val=y_va, sample_weight_val=sw_va)
        del X_tr
    return clf


def score_subjects_streaming(
    clf: HistGradientBoostingClassifier,
    layout: CacheLayout,
    eval_frame: pd.DataFrame,
    moments: FoldMoments,
    *,
    n_classes: int = 5,
    check_fn: Callable[[], None] | None = None,
) -> np.ndarray:
    """Score evaluation patients one night at a time in float32, preserving input row order."""
    probs = np.zeros((len(eval_frame), n_classes), dtype=np.float32)
    classes = np.asarray(clf.classes_, dtype=np.int64)
    for subject, positions in eval_frame.groupby("subject_id", sort=False).indices.items():
        if check_fn is not None:
            check_fn()
        sub_frame = eval_frame.iloc[positions]
        dense, markers, history, occipital = load_subject_night(
            layout, str(subject), include_occipital=moments.include_occipital
        )
        X_night = encode_subject_features(
            dense, markers, history, occipital, sub_frame["epoch_index"].to_numpy(), moments
        )
        probs[np.ix_(positions, classes)] = clf.predict_proba(X_night).astype(np.float32)
        del dense, markers, history, occipital, X_night
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid probabilities produced during streaming inference")
    return probs


def apply_prior_power(
    probabilities: np.ndarray,
    fold_indices: np.ndarray,
    fold_priors: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Rescale probabilities by fold_priors[fold] ** alpha and renormalize."""
    log_p = np.log(np.clip(probabilities, 1e-12, 1.0))
    log_prior = np.log(np.clip(fold_priors[fold_indices], 1e-12, 1.0))
    scaled = softmax(log_p + float(alpha) * log_prior, axis=1)
    return scaled.astype(np.float32)


def calibrate_oof_prior_power(
    probabilities: np.ndarray,
    labels: np.ndarray,
    fold_indices: np.ndarray,
    fold_priors: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Leave-one-fold-out optimization of scalar prior-power exponent alpha for macro-F1."""
    grid = np.linspace(-0.5, 1.5, 201) if alphas is None else np.asarray(alphas, dtype=np.float64)
    unique_folds = sorted(int(f) for f in np.unique(fold_indices))
    log_p = np.log(np.clip(probabilities, 1e-12, 1.0))
    log_priors = np.log(np.clip(fold_priors, 1e-12, 1.0))
    row_log_prior = log_priors[fold_indices]

    def best_alpha(mask: np.ndarray, folds_subset: list[int]) -> tuple[float, float]:
        best_a = 0.0
        best_score = -1.0
        fold_masks = [mask & (fold_indices == f) for f in folds_subset]
        for a in grid:
            scores = []
            for fm in fold_masks:
                f_preds = (log_p[fm] + a * row_log_prior[fm]).argmax(axis=1)
                scores.append(f1_score(labels[fm], f_preds, average="macro", zero_division=0))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_a = float(a)
        return best_a, best_score

    calibrated = np.empty_like(probabilities, dtype=np.float32)
    lofo_alphas: dict[int, float] = {}
    if len(unique_folds) >= 2:
        for f in unique_folds:
            train_mask = fold_indices != f
            other_folds = [k for k in unique_folds if k != f]
            alpha_f, _ = best_alpha(train_mask, other_folds)
            lofo_alphas[f] = alpha_f
            eval_mask = fold_indices == f
            calibrated[eval_mask] = softmax(
                log_p[eval_mask] + alpha_f * row_log_prior[eval_mask], axis=1
            ).astype(np.float32)
    else:
        f = unique_folds[0]
        alpha_f, _ = best_alpha(np.ones(len(labels), dtype=bool), unique_folds)
        lofo_alphas[f] = alpha_f
        calibrated[:] = softmax(log_p + alpha_f * row_log_prior, axis=1).astype(np.float32)

    pooled_alpha, pooled_score = best_alpha(np.ones(len(labels), dtype=bool), unique_folds)
    raw_macro = float(
        np.mean([
            f1_score(labels[fold_indices == f], probabilities[fold_indices == f].argmax(axis=1), average="macro", zero_division=0)
            for f in unique_folds
        ])
    )
    cal_macro = float(
        np.mean([
            f1_score(labels[fold_indices == f], calibrated[fold_indices == f].argmax(axis=1), average="macro", zero_division=0)
            for f in unique_folds
        ])
    )
    return calibrated, {
        "lofo_alphas": {str(k): v for k, v in lofo_alphas.items()},
        "pooled_best_alpha": pooled_alpha,
        "pooled_grid_macro_f1": pooled_score,
        "raw_mean_fold_macro_f1": raw_macro,
        "calibrated_mean_fold_macro_f1": cal_macro,
    }
