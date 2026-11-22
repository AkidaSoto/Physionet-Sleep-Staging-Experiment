"""Cached, label-free slow-wave and short-window eye-movement variants for CinC development."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from physionet_sleep.algorithms.eeg_quality import bad_seconds
from physionet_sleep.experiments.waveform_data import EPOCH_SAMPLES, SAMPLE_RATE, WaveformStore
from physionet_sleep.labels.cinc2018 import EPOCH_SECONDS

AMPLITUDES_UV = (60, 75, 90)
WAVE_POOLING = ("mean", "best_clean")
EOG_WINDOWS = (4, 6)
EOG_QUANTILES = (75, 90)
EOG_GUARDS = ("unguarded", "frontal2")
SLOW_WAVE_VARIANTS = tuple(
    f"slow_wave_{amplitude}_{pooling}" for amplitude in AMPLITUDES_UV for pooling in WAVE_POOLING
)
EOG_VARIANTS = tuple(
    f"eog_short_{seconds}s_q{quantile}_{guard}"
    for seconds in EOG_WINDOWS
    for quantile in EOG_QUANTILES
    for guard in EOG_GUARDS
)
RAW_VARIANTS = SLOW_WAVE_VARIANTS + EOG_VARIANTS


def _slow_wave_fractions(lead_uv: np.ndarray, n_epochs: int) -> dict[int, np.ndarray]:
    """Use the production 0.5–2 Hz filter and 0.5–2 s crossing-to-crossing duration."""
    sos = butter(2, [0.5, 2.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = sosfiltfilt(sos, np.nan_to_num(lead_uv))
    crossings = np.flatnonzero((filtered[:-1] < 0) & (filtered[1:] >= 0)) + 1
    if len(crossings) < 2:
        return {threshold: np.zeros(n_epochs) for threshold in AMPLITUDES_UV}
    bounds = n_epochs * EPOCH_SAMPLES
    start, end = crossings[:-1], crossings[1:]
    lengths = end - start
    p2p = np.maximum.reduceat(filtered, crossings)[:-1] - np.minimum.reduceat(filtered, crossings)[:-1]
    duration_ok = (lengths >= 0.5 * SAMPLE_RATE) & (lengths <= 2.0 * SAMPLE_RATE)
    usable = duration_ok & (start < bounds)
    start, end, p2p = start[usable], np.minimum(end[usable], bounds), p2p[usable]
    epoch = start // EPOCH_SAMPLES
    boundary = (epoch + 1) * EPOCH_SAMPLES
    start_overlap = np.minimum(end, boundary) - start
    next_overlap = end - np.minimum(end, boundary)
    fractions = {}
    for threshold in AMPLITUDES_UV:
        chosen = p2p >= threshold
        covered = np.bincount(epoch[chosen], weights=start_overlap[chosen], minlength=n_epochs)
        # At most two scoring epochs intersect a wave (maximum wave length is 2 s).
        cross = chosen & (next_overlap > 0)
        covered += np.bincount(epoch[cross] + 1, weights=next_overlap[cross], minlength=n_epochs)
        fractions[threshold] = covered / EPOCH_SAMPLES
    return fractions


def _frontal_quality(quality: pd.DataFrame, n_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing label-free bad-second rule with its 20% per-epoch allowance."""
    n_seconds = n_epochs * EPOCH_SECONDS
    if len(quality) < n_seconds or not np.array_equal(
        quality["second"].to_numpy()[:n_seconds], np.arange(n_seconds)
    ):
        raise ValueError("frontal quality must contain aligned full-epoch seconds")
    return tuple(
        bad_seconds(quality, lead)[:n_seconds].reshape(n_epochs, EPOCH_SECONDS).mean(axis=1) <= 0.2
        for lead in ("F3", "F4")
    )


def _pool_frontal(
    f3: np.ndarray, f4: np.ndarray, f3_clean: np.ndarray, f4_clean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = (f3 + f4) / 2
    best_clean = np.where(
        f3_clean & f4_clean,
        np.maximum(f3, f4),
        np.where(f3_clean, f3, np.where(f4_clean, f4, mean)),
    )
    return mean, best_clean


def _eye_window_scores(
    eog: np.ndarray, frontal: np.ndarray, n_epochs: int, window_seconds: int
) -> dict[str, np.ndarray]:
    """Score eye-band versus high-band magnitude, with an optional same-band frontal penalty.

    The guard raises the denominator to at least twice F3's 0.5–5 Hz
    magnitude, discounting short windows with comparable activity on F3.
    Each window stays inside its 30 s scoring epoch. This is a spectral
    activity summary from one EOG, not a count of annotated REM events.
    """
    win = window_seconds * SAMPLE_RATE
    hop = 2 * SAMPLE_RATE
    freqs = rfftfreq(win, d=1 / SAMPLE_RATE)
    rem = (freqs >= 0.5) & (freqs <= 5)
    noise = (freqs >= 5) & (freqs <= 30)
    taper = tukey(win, 0.25, sym=False)
    floor = 1e-12
    by_guard = {name: np.full((n_epochs, 1 + (EPOCH_SAMPLES - win) // hop), np.nan) for name in EOG_GUARDS}

    for first in range(0, n_epochs, 64):
        last = min(n_epochs, first + 64)
        wing = slice(first * EPOCH_SAMPLES, last * EPOCH_SAMPLES)
        eye_windows = np.lib.stride_tricks.sliding_window_view(
            eog[wing].reshape(-1, EPOCH_SAMPLES), win, axis=1
        )[:, ::hop]
        frontal_windows = np.lib.stride_tricks.sliding_window_view(
            frontal[wing].reshape(-1, EPOCH_SAMPLES), win, axis=1
        )[:, ::hop]
        eye_spectrum = np.abs(rfft(eye_windows * taper, axis=-1))
        frontal_spectrum = np.abs(rfft(frontal_windows * taper, axis=-1))
        eye_rem = eye_spectrum[..., rem].mean(axis=-1)
        eye_noise = eye_spectrum[..., noise].mean(axis=-1)
        frontal_rem = frontal_spectrum[..., rem].mean(axis=-1)
        by_guard["unguarded"][first:last] = 10 * np.log10(
            np.maximum(eye_rem, floor) / np.maximum(eye_noise, floor)
        )
        by_guard["frontal2"][first:last] = 10 * np.log10(
            np.maximum(eye_rem, floor) / np.maximum.reduce(
                [eye_noise, 2 * frontal_rem, np.full_like(eye_noise, floor)]
            )
        )
        no_eye_signal = np.ptp(eye_windows, axis=-1) <= 0
        for variant in by_guard.values():
            variant[first:last][no_eye_signal] = np.nan
    return by_guard


def extract_raw_variants(
    f3_uv: np.ndarray, f4_uv: np.ndarray, eog_uv: np.ndarray, quality: pd.DataFrame
) -> pd.DataFrame:
    """Build one row per complete 30 s epoch from the calibrated waveform leads."""
    if not len(f3_uv) == len(f4_uv) == len(eog_uv):
        raise ValueError("frontal and EOG leads must share a sample clock")
    n_epochs = len(eog_uv) // EPOCH_SAMPLES
    if not n_epochs:
        raise ValueError("expected at least one complete waveform epoch")
    f3_clean, f4_clean = _frontal_quality(quality, n_epochs)
    f3 = _slow_wave_fractions(f3_uv, n_epochs)
    f4 = _slow_wave_fractions(f4_uv, n_epochs)
    values = {}
    for threshold in AMPLITUDES_UV:
        mean, best_clean = _pool_frontal(f3[threshold], f4[threshold], f3_clean, f4_clean)
        values[f"slow_wave_{threshold}_mean"] = mean
        values[f"slow_wave_{threshold}_best_clean"] = best_clean

    for seconds in EOG_WINDOWS:
        for guard, scores in _eye_window_scores(eog_uv, f3_uv, n_epochs, seconds).items():
            finite = np.isfinite(scores).any(axis=1)
            pooled = np.full((len(EOG_QUANTILES), n_epochs), np.nan)
            pooled[:, finite] = np.nanpercentile(scores[finite], EOG_QUANTILES, axis=1)
            for q, value in zip(EOG_QUANTILES, pooled, strict=True):
                values[f"eog_short_{seconds}s_q{q}_{guard}"] = value
    epoch_index = pd.RangeIndex(n_epochs, name="epoch_index")
    return pd.DataFrame(values, index=epoch_index).loc[:, RAW_VARIANTS].astype(np.float32)


class RawStagingFeatureCache:
    """Keep one open waveform and persist each night independently for resumable extraction."""

    def __init__(
        self,
        raw_root: Path,
        quality_root: Path,
        cache_root: Path,
        *,
        permitted_subjects: Iterable[str],
    ):
        self.waveforms = WaveformStore(raw_root, permitted_subjects=permitted_subjects, max_open_nights=1)
        self.quality_root = Path(quality_root)
        self.cache_root = Path(cache_root)

    def read(self, subject_id: str) -> pd.DataFrame:
        if subject_id not in self.waveforms.permitted_subjects:
            raise ValueError(f"{subject_id!r} is outside the permitted waveform manifest")
        path = self.cache_root / f"{subject_id}.parquet"
        if path.exists():
            cached = pd.read_parquet(path)
            if tuple(cached.columns) != RAW_VARIANTS or not cached.index.equals(
                pd.RangeIndex(len(cached), name="epoch_index")
            ):
                raise ValueError(f"{path}: cached feature columns or epoch grid do not match")
            return cached
        metrics = ("flat", "line", "ecg_corr", "ptp")
        quality = pd.read_parquet(
            self.quality_root / f"{subject_id}.parquet",
            columns=["second", *(f"{lead}_{metric}" for lead in ("F3", "F4") for metric in metrics)],
        )
        features = extract_raw_variants(
            self.waveforms.channel_uv(subject_id, "F3-M2"),
            self.waveforms.channel_uv(subject_id, "F4-M1"),
            self.waveforms.channel_uv(subject_id, "E1-M2"),
            quality,
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=self.cache_root, prefix=f"{subject_id}-", suffix=".parquet", delete=False) as temp:
            pending = Path(temp.name)
        try:
            features.to_parquet(pending)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return features
