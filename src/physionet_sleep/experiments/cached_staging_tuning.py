"""Theta and sigma candidates from existing per-lead CinC spectral caches."""

from __future__ import annotations

import numpy as np
import pandas as pd

from physionet_sleep.algorithms.eeg_quality import QUALITY_RULES, bad_seconds
from physionet_sleep.algorithms.eeg_tracks import EPS
from physionet_sleep.analysis.epoch_windows import EPOCH_SECONDS, ROW_SECONDS

TRACK_COLUMNS = ("second", "c_rel_theta")
CACHE_COLUMNS = (
    "second", "O1_theta", "O2_theta", "O1_total", "O2_total",
    *(f"{lead}_{band}" for lead in ("C3", "C4") for band in ("sigma", "total", *QUALITY_RULES)),
)
THETA_CANDIDATES = ("c_theta_6s", "c_theta_30s", "o_theta_6s", "o_theta_30s")
SIGMA_CANDIDATES = tuple(
    f"sigma_{threshold}mad_{persistence}s_{pool}"
    for threshold in (1, 2)
    for persistence in (1, 2)
    for pool in ("mean", "max")
)


def _block_mean(values: np.ndarray, block_seconds: int) -> np.ndarray:
    """Average non-overlapping blocks, including the final partial block."""
    padded = np.pad(values, (0, -len(values) % block_seconds), constant_values=np.nan)
    blocks = padded.reshape(-1, block_seconds)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=1)
    sums = np.where(finite, blocks, 0).sum(axis=1)
    return np.divide(sums, counts, out=np.full(len(blocks), np.nan), where=counts > 0)


def _local_reference(values: np.ndarray, window_seconds: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate medians and robust SDs from overlapping whole-recording windows."""
    half = window_seconds // 2
    centers = np.unique(np.r_[np.arange(0, len(values), half), len(values) - 1])
    median = np.full(len(centers), np.nan)
    sd = median.copy()
    for i, center in enumerate(centers):
        segment = values[max(0, center - half):center + half]
        valid = segment[np.isfinite(segment)]
        if valid.size >= 30:
            median[i] = np.median(valid)
            sd[i] = 1.4826 * np.median(np.abs(valid - median[i]))
    usable = np.isfinite(median)
    if not usable.any():
        return np.full(len(values), np.nan), np.full(len(values), np.nan)
    seconds = np.arange(len(values))
    return np.interp(seconds, centers[usable], median[usable]), np.interp(seconds, centers[usable], sd[usable])


def _sigma_candidates(quality: pd.DataFrame, n_slots: int) -> dict[str, np.ndarray]:
    scores, available = [], []
    for lead in ("C3", "C4"):
        total = quality[f"{lead}_total"].to_numpy(dtype=np.float64)
        sigma = quality[f"{lead}_sigma"].to_numpy(dtype=np.float64)
        share = sigma / (total + EPS)
        share[bad_seconds(quality, lead) | (total <= 0) | ~np.isfinite(share)] = np.nan
        median, sd = _local_reference(share)
        score = (share - median) / np.maximum(sd, np.finfo(np.float32).eps)
        scores.append(score)
        available.append(_block_mean(np.isfinite(score).astype(float), EPOCH_SECONDS) >= 0.8)

    available = np.stack(available)
    counts_available = available.sum(axis=0)
    candidates = {}
    for threshold in (1, 2):
        elevated = np.stack([score > threshold for score in scores])
        leading = np.pad(elevated[:, :-1], ((0, 0), (1, 0)))
        trailing = np.pad(elevated[:, 1:], ((0, 0), (0, 1)))
        for persistence, qualified in [(1, elevated), (2, elevated & (leading | trailing))]:
            blocks = np.pad(qualified, ((0, 0), (0, -len(quality) % EPOCH_SECONDS)))
            counts = blocks.reshape(2, -1, EPOCH_SECONDS).sum(axis=-1)
            counts = np.where(available, counts, 0)
            mean = np.divide(counts.sum(axis=0), counts_available,
                             out=np.full(counts.shape[1], np.nan), where=counts_available > 0)
            strongest = np.where(counts_available > 0, counts.max(axis=0), np.nan)
            for pool, values in [("mean", mean), ("max", strongest)]:
                name = f"sigma_{threshold}mad_{persistence}s_{pool}"
                candidates[name] = np.repeat(values, EPOCH_SECONDS // ROW_SECONDS)[:n_slots]
    return candidates


def cached_staging_candidates(tracks: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Label-free candidates at existing 6-second slots. No raw recordings are read.

    Theta is the 4–8 Hz share at central or occipital leads, averaged over
    6 or 30 seconds. Sigma counts elevated 11–16 Hz power-share bins by
    central lead in each scoring epoch. The 2-second, overlapping spectral
    frames measure persistence; adjacent bins cannot establish event duration.
    The retained alpha+beta spindle feature is computed elsewhere.
    """
    seconds = np.arange(len(tracks))
    if not len(seconds) or not np.array_equal(tracks["second"], seconds):
        raise ValueError("tracks must contain the uninterrupted 1-second grid from recording start")
    if not np.array_equal(quality["second"], seconds):
        raise ValueError("quality and track caches must share the same 1-second grid")
    central = tracks["c_rel_theta"].to_numpy(dtype=np.float64)
    total = (quality["O1_total"] + quality["O2_total"]).to_numpy(dtype=np.float64)
    theta = (quality["O1_theta"] + quality["O2_theta"]).to_numpy(dtype=np.float64)
    # Two-lead power pooling uses the same convention as the central track.
    occipital = theta / (total + 2 * EPS)
    occipital[total <= 0] = np.nan
    n_slots = -(-len(tracks) // ROW_SECONDS)
    columns = {}
    for prefix, values in [("c", central), ("o", occipital)]:
        columns[f"{prefix}_theta_6s"] = _block_mean(values, ROW_SECONDS)
        columns[f"{prefix}_theta_30s"] = np.repeat(
            _block_mean(values, EPOCH_SECONDS), EPOCH_SECONDS // ROW_SECONDS
        )[:n_slots]
    columns.update(_sigma_candidates(quality, n_slots))
    return pd.DataFrame(columns, index=pd.RangeIndex(n_slots, name="slot"), dtype=np.float32)
