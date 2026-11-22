"""Frontal K-complex-like morphology and recent N2 event support."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from physionet_sleep.algorithms.eeg_quality import QUALITY_RULES, bad_seconds
from physionet_sleep.experiments.waveform_data import EPOCH_SAMPLES, SAMPLE_RATE
from physionet_sleep.labels.cinc2018 import EPOCH_SECONDS

N2_EVENT_COLUMNS = ("kc_shape_events", "n2_recent_event_support")
DETECTOR_HZ = 50
DETECTOR_EPOCH = EPOCH_SECONDS * DETECTOR_HZ
MAX_LAG = 4
KC_FILTER = butter(2, [0.3, 4.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")


def _quality_bad(quality: pd.DataFrame, lead: str) -> np.ndarray:
    columns = [f"{lead}_{metric}" for metric in QUALITY_RULES]
    return bad_seconds(quality, lead) | ~np.isfinite(quality[columns].to_numpy()).all(
        axis=1
    )


def _clean_epochs(wave: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter in bounded 30-second batches to keep later epochs out of earlier event counts."""
    n = len(wave)
    usable = np.empty(n, dtype=bool)
    filtered = np.empty((n, DETECTOR_EPOCH), dtype=np.float32)
    bad = bad.copy()
    for start in range(0, n, 64):
        end = min(n, start + 64)
        block = wave[start:end]
        finite = np.isfinite(block)
        bad[start:end] |= ~finite.reshape(-1, EPOCH_SECONDS, SAMPLE_RATE).all(axis=-1)
        usable[start:end] = bad[start:end].mean(axis=1) <= 0.20
        filled = np.where(finite, block, 0)
        filtered[start:end] = sosfiltfilt(KC_FILTER, filled, axis=-1)[
            :, :: SAMPLE_RATE // DETECTOR_HZ
        ]
    return filtered.ravel(), bad.ravel(), usable


def _rhythmic_neighbours(troughs: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    """Flag pairs of similar deep troughs separated by a slow-oscillation period."""
    distance = np.diff(troughs) / DETECTOR_HZ
    amplitude = np.abs(voltage[troughs])
    similar = np.minimum(amplitude[:-1], amplitude[1:]) >= 0.6 * np.maximum(
        amplitude[:-1], amplitude[1:]
    )
    pairs = (
        (distance >= 0.45)
        & (distance <= 1.8)
        & similar
        & (troughs[:-1] // DETECTOR_EPOCH == troughs[1:] // DETECTOR_EPOCH)
    )
    return np.r_[pairs, False] | np.r_[False, pairs]


def _event_troughs(
    voltage: np.ndarray, bad: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    troughs, _ = find_peaks(
        -voltage,
        height=30,
        prominence=25,
        distance=0.24 * DETECTOR_HZ,
        width=(0.12 * DETECTOR_HZ, 0.6 * DETECTOR_HZ),
        wlen=2 * DETECTOR_HZ + 1,
    )
    if not len(troughs):
        return troughs
    edge = DETECTOR_HZ // 2
    post = np.arange(round(0.16 * DETECTOR_HZ), round(0.65 * DETECTOR_HZ) + 1)
    epoch = troughs // DETECTOR_EPOCH
    position = troughs % DETECTOR_EPOCH
    keep = (
        usable[epoch]
        & ~_rhythmic_neighbours(troughs, voltage)
        & (position >= edge)
        & (position + post[-1] < DETECTOR_EPOCH - edge)
    )
    troughs = troughs[keep]
    if not len(troughs):
        return troughs
    rebound_view = voltage[troughs[:, None] + post]
    rebound = troughs + post[rebound_view.argmax(axis=1)]
    rebound_uv = voltage[rebound]
    keep = (rebound_uv >= 12) & (rebound_uv - voltage[troughs] >= 50)
    troughs = troughs[keep]
    rebound = rebound[keep]
    downward = np.flatnonzero((voltage[:-1] >= 0) & (voltage[1:] < 0)) + 1
    start_index = np.searchsorted(downward, troughs, side="right") - 1
    end_index = np.searchsorted(downward, rebound, side="right")
    keep = (start_index >= 0) & (end_index < len(downward))
    troughs = troughs[keep]
    begin, end = downward[start_index[keep]], downward[end_index[keep]]
    duration_s = (end - begin) / DETECTOR_HZ
    quality_counts = np.r_[0, np.cumsum(bad)]
    clean = (
        quality_counts[(end - 1) // DETECTOR_HZ + 1]
        == quality_counts[begin // DETECTOR_HZ]
    )
    keep = (
        (duration_s >= 0.5)
        & (duration_s <= 1.7)
        & (begin // DETECTOR_EPOCH == troughs // DETECTOR_EPOCH)
        & (end // DETECTOR_EPOCH == troughs // DETECTOR_EPOCH)
        & (begin % DETECTOR_EPOCH >= edge)
        & (end % DETECTOR_EPOCH < DETECTOR_EPOCH - edge)
        & clean
    )
    return troughs[keep]


def _recent_support(events: np.ndarray, spindle_count: np.ndarray) -> np.ndarray:
    positive = (events > 0) | (spindle_count >= 2)
    known = positive | (np.isfinite(events) & np.isfinite(spindle_count))
    recent = np.zeros(len(events), dtype=np.float32)
    observed = np.zeros(len(events), dtype=bool)
    for lag in range(1, min(MAX_LAG, len(events) - 1) + 1):
        past = positive[:-lag] * 2 ** (-lag / 2)
        recent[lag:] = np.maximum(recent[lag:], past)
        observed[lag:] |= known[:-lag]
    return np.where(observed, recent, np.nan)


def n2_event_panel(
    f3_uv: np.ndarray,
    f4_uv: np.ndarray,
    quality: pd.DataFrame,
    spindle_count: np.ndarray,
) -> pd.DataFrame:
    """Return a count of isolated frontal biphasic events and past event support.

    The waveform criteria approximate candidate K-complex morphology. The
    recent-support feature is the largest exponentially decayed indicator
    of K-like or spindle-proxy activity across the four preceding epochs.
    """
    f3_uv, f4_uv, spindle_count = (np.asarray(x) for x in (f3_uv, f4_uv, spindle_count))
    if f3_uv.ndim != 1 or f4_uv.shape != f3_uv.shape:
        raise ValueError("Frontal leads must share a one-dimensional sample clock")
    n = len(f3_uv) // EPOCH_SAMPLES
    if n == 0 or spindle_count.shape != (n,):
        raise ValueError(
            "Expected one spindle-proxy count per complete 30-second epoch"
        )
    seconds = n * EPOCH_SECONDS
    if len(quality) < seconds or not np.array_equal(
        quality["second"].to_numpy()[:seconds], np.arange(seconds)
    ):
        raise ValueError("Frontal quality must be aligned to complete waveform seconds")
    quality = quality.iloc[:seconds]

    detections, usable = [], []
    for lead, wave in (("F3", f3_uv), ("F4", f4_uv)):
        epoch_wave = wave[: n * EPOCH_SAMPLES].reshape(n, EPOCH_SAMPLES)
        bad = _quality_bad(quality, lead).reshape(n, EPOCH_SECONDS)
        volts, quality_bad, clean = _clean_epochs(epoch_wave, bad)
        detections.append(_event_troughs(volts, quality_bad, clean))
        usable.append(clean)

    merged = np.sort(np.concatenate(detections))
    if len(merged):
        merged = merged[np.r_[True, np.diff(merged) > round(0.20 * DETECTOR_HZ)]]
    counts = np.bincount(merged // DETECTOR_EPOCH, minlength=n).astype(np.float32)
    counts[~(usable[0] | usable[1])] = np.nan
    result = np.column_stack((counts, _recent_support(counts, spindle_count)))
    return pd.DataFrame(
        result.astype(np.float32),
        index=pd.RangeIndex(n, name="epoch_index"),
        columns=N2_EVENT_COLUMNS,
    )
