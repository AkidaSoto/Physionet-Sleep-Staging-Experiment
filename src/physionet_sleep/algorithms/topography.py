"""Features that need calibrated microvolts or frontal/occipital derivations (not available in UCDDB).

frontal_slow_wave_fraction: AASM N3 rule. Fraction of each 30-s epoch covered by 0.5-2 Hz waves with peak-to-peak
    amplitude >= 75 uV on the frontal derivations (F3-M2, F4-M1), averaged over channels. N3 is scored when this is
    >= 0.2.
occipital_rel_alpha: AASM wake rule. 8-12 Hz share of 1-30 Hz power on O1-M2/O2-M1 (Welch, 4-s segments).

Both are emitted on the 6-s row grid (sample i starts at i * row_seconds and carries its epoch's value).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, SignalRecord, SignalSeries


def slow_wave_coverage(signal_uv: np.ndarray, fs: float, *, min_p2p_uv: float = 75.0) -> np.ndarray:
    """Boolean per sample: inside a 0.5-2 Hz wave whose peak-to-peak amplitude is >= min_p2p_uv."""
    x = np.nan_to_num(np.asarray(signal_uv, dtype=float))
    sos = butter(2, [0.5, 2.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    # Full waves run between consecutive negative-to-positive zero crossings.
    ups = np.flatnonzero((filtered[:-1] < 0) & (filtered[1:] >= 0)) + 1
    covered = np.zeros(x.size, dtype=bool)
    for start, end in zip(ups[:-1], ups[1:]):
        duration = (end - start) / fs
        if not 0.5 <= duration <= 2.0:
            continue
        segment = filtered[start:end]
        if segment.max() - segment.min() >= min_p2p_uv:
            covered[start:end] = True
    return covered


def _epoch_view(values: np.ndarray, fs: float, k: int, epoch_seconds: float) -> np.ndarray | None:
    a, b = int(round(k * epoch_seconds * fs)), int(round((k + 1) * epoch_seconds * fs))
    return values[a:b] if b <= values.shape[0] else None


class CalibratedTopographyGenerator(BaseAlgorithm):
    def __init__(
        self,
        *,
        frontal_key: str = "eeg_frontal",
        occipital_key: str = "eeg_occipital",
        epoch_seconds: float = 30.0,
        row_seconds: float = 6.0,
        min_p2p_uv: float = 75.0,
    ) -> None:
        super().__init__(name="topography_features")
        self.frontal_key = frontal_key
        self.occipital_key = occipital_key
        self.epoch_seconds = float(epoch_seconds)
        self.row_seconds = float(row_seconds)
        self.min_p2p_uv = float(min_p2p_uv)
        self.rows_per_epoch = int(round(self.epoch_seconds / self.row_seconds))

    def required_signals(self) -> tuple[str, ...]:
        return (self.frontal_key, self.occipital_key)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        frontal, occipital = record.require(self.frontal_key, self.occipital_key)
        for series in (frontal, occipital):
            units = str(series.units or "").lower()
            if units not in ("uv", "µv"):
                raise ValueError(f"{self.name} needs EEG in microvolts, got units '{series.units}'")
        f_values = np.asarray(frontal.values, dtype=float).reshape(len(frontal.values), -1)
        o_values = np.asarray(occipital.values, dtype=float).reshape(len(occipital.values), -1)
        f_fs, o_fs = float(frontal.sample_rate), float(occipital.sample_rate)
        n_epochs = int(min(f_values.shape[0] / f_fs, o_values.shape[0] / o_fs) // self.epoch_seconds)

        coverage = [slow_wave_coverage(f_values[:, ch], f_fs, min_p2p_uv=self.min_p2p_uv) for ch in range(f_values.shape[1])]
        sw_fraction = np.full(n_epochs, np.nan)
        rel_alpha = np.full(n_epochs, np.nan)
        nperseg = int(round(4.0 * o_fs))
        for k in range(n_epochs):
            fractions = [float(_epoch_view(c, f_fs, k, self.epoch_seconds).mean()) for c in coverage]
            sw_fraction[k] = float(np.mean(fractions))
            shares = []
            for ch in range(o_values.shape[1]):
                segment = _epoch_view(o_values[:, ch], o_fs, k, self.epoch_seconds)
                if segment is None or not np.all(np.isfinite(segment)) or np.std(segment) == 0:
                    continue
                freqs, psd = welch(segment, fs=o_fs, nperseg=nperseg)
                total = psd[(freqs >= 1.0) & (freqs < 30.0)].sum()
                if total > 0:
                    shares.append(psd[(freqs >= 8.0) & (freqs < 12.0)].sum() / total)
            rel_alpha[k] = float(np.mean(shares)) if shares else np.nan

        metadata = {"start_seconds": 0.0, "epoch_seconds": self.epoch_seconds, "row_seconds": self.row_seconds}
        signals = {
            "frontal_slow_wave_fraction": SignalSeries(np.repeat(sw_fraction, self.rows_per_epoch), sample_rate=1.0 / self.row_seconds, units="fraction", metadata=dict(metadata)),
            "occipital_rel_alpha": SignalSeries(np.repeat(rel_alpha, self.rows_per_epoch), sample_rate=1.0 / self.row_seconds, units="ratio", metadata=dict(metadata)),
        }
        return AlgorithmResult(
            name=self.name,
            signals=signals,
            features={"min_p2p_uv": self.min_p2p_uv, "n_epochs": n_epochs},
        )
