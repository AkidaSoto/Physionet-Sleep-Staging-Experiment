"""Complementary frontal slow-wave measurements on the Boston epoch grid."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from physionet_sleep.experiments.staging_feature_waveform import _frontal_quality
from physionet_sleep.experiments.waveform_data import EPOCH_SAMPLES, SAMPLE_RATE

SLOW_WAVE_PANEL_COLUMNS = (
    "slow_wave_75_max_clean",
    "slow_wave_75_bilateral_overlap",
    "slow_wave_60_to_75_max_clean",
)


@lru_cache(maxsize=1)
def _slow_wave_filter() -> np.ndarray:
    return butter(2, [0.5, 2.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")


def _wave_intervals(
    lead_uv: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve event timing from the calibrated slow-wave coverage detector."""
    values = np.nan_to_num(lead_uv, nan=0.0, posinf=0.0, neginf=0.0)
    filtered = sosfiltfilt(_slow_wave_filter(), values)
    crossings = np.flatnonzero((filtered[:-1] < 0) & (filtered[1:] >= 0)) + 1
    amplitudes = (
        np.maximum.reduceat(filtered, crossings)
        - np.minimum.reduceat(filtered, crossings)
    )[:-1]
    start, end = crossings[:-1], crossings[1:]
    duration = end - start
    valid = (
        (duration >= 0.5 * SAMPLE_RATE)
        & (duration <= 2.0 * SAMPLE_RATE)
        & (start < limit)
    )
    return start[valid], np.minimum(end[valid], limit), amplitudes[valid]


def _epoch_coverage(start: np.ndarray, end: np.ndarray, n_epochs: int) -> np.ndarray:
    epoch = start // EPOCH_SAMPLES
    before_boundary = np.minimum(end, (epoch + 1) * EPOCH_SAMPLES)
    covered = np.bincount(epoch, weights=before_boundary - start, minlength=n_epochs)
    # Each interval is at most two seconds, so it can cross at most one epoch boundary.
    crossed = end > before_boundary
    covered += np.bincount(
        epoch[crossed] + 1, weights=(end - before_boundary)[crossed], minlength=n_epochs
    )
    return covered / EPOCH_SAMPLES


def _overlap_coverage(
    f3: tuple[np.ndarray, np.ndarray], f4: tuple[np.ndarray, np.ndarray], n_epochs: int
) -> np.ndarray:
    starts = np.concatenate((f3[0], f4[0]))
    ends = np.concatenate((f3[1], f4[1]))
    edges, positions = np.unique(np.concatenate((starts, ends)), return_inverse=True)
    delta = np.bincount(positions[: len(starts)], minlength=len(edges))
    delta -= np.bincount(positions[len(starts) :], minlength=len(edges))
    both = np.cumsum(delta)[:-1] == 2
    return _epoch_coverage(edges[:-1][both], edges[1:][both], n_epochs)


def slow_wave_panel(
    f3_uv: np.ndarray, f4_uv: np.ndarray, quality: pd.DataFrame
) -> pd.DataFrame:
    """Return high-amplitude, coincident, and moderate-amplitude occupancy.

    Per-lead coverage requires the existing clean-epoch rule and finite samples.
    A single usable lead supplies either amplitude-band maximum. Simultaneous
    coverage requires both leads. An unavailable measurement is represented by NaN.
    """
    f3, f4 = np.asarray(f3_uv), np.asarray(f4_uv)
    if f3.ndim != 1 or f3.shape != f4.shape:
        raise ValueError(
            "Frontal leads must be one-dimensional and share an aligned sample clock"
        )
    n_epochs = len(f3) // EPOCH_SAMPLES
    if n_epochs == 0:
        raise ValueError("Expected at least one complete waveform epoch")
    limit = n_epochs * EPOCH_SAMPLES
    clean = _frontal_quality(quality, n_epochs)
    high_intervals = []
    high_coverage = []
    moderate_coverage = []
    available = []
    for lead, clean_epochs in zip((f3, f4), clean, strict=True):
        usable = clean_epochs & np.isfinite(lead[:limit]).reshape(
            n_epochs, EPOCH_SAMPLES
        ).all(axis=1)
        start, end, p2p = _wave_intervals(lead, limit)
        high = p2p >= 75.0
        moderate = (p2p >= 60.0) & (p2p < 75.0)
        intervals = (start[high], end[high])
        high_intervals.append(intervals)
        high_coverage.append(
            np.where(usable, _epoch_coverage(*intervals, n_epochs), np.nan)
        )
        moderate_coverage.append(
            np.where(
                usable,
                _epoch_coverage(start[moderate], end[moderate], n_epochs),
                np.nan,
            )
        )
        available.append(usable)
    both = np.where(
        available[0] & available[1],
        _overlap_coverage(*high_intervals, n_epochs),
        np.nan,
    )
    values = np.column_stack(
        (np.fmax(*high_coverage), both, np.fmax(*moderate_coverage))
    )
    return pd.DataFrame(
        values.astype(np.float32),
        index=pd.RangeIndex(n_epochs, name="epoch_index"),
        columns=SLOW_WAVE_PANEL_COLUMNS,
    )
