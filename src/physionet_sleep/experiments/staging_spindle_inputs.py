"""Blockwise extraction of the two spindle measurements used by the gated stager."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from physionet_sleep.algorithms.staging import (
    SpindleFeatureGenerator,
    _as_2d_signal,
    _band_median,
    _spectrogram_abs,
    _time_series_rate,
)
from physionet_sleep.core.types import AlgorithmResult, SignalRecord, SignalSeries


def _aperiodic_exponent(spectrum: np.ndarray, hz: np.ndarray) -> np.ndarray:
    fit = ((hz >= 2.0) & (hz <= 7.0)) | ((hz >= 14.0) & (hz <= 24.0))
    x = np.log10(hz[fit])
    values = spectrum[fit].T
    valid = np.isfinite(values) & (values > 0)
    counts = valid.sum(axis=1)
    result = np.full(len(values), np.nan)
    if len(x) < 5:
        return result

    # Reuse polyfit's degree-one design and retain its single-response LAPACK
    # solve. Batched right-hand sides change rounding in the fitted slope.
    design = np.vander(x, 2)
    scale = np.sqrt((design * design).sum(axis=0))
    design /= scale
    rcond = len(x) * np.finfo(x.dtype).eps
    for i in np.flatnonzero(counts == len(x)):
        coefficients, *_ = np.linalg.lstsq(design, np.log10(values[i]), rcond=rcond)
        result[i] = -coefficients[0] / scale[0]
    for i in np.flatnonzero((counts >= 5) & (counts < len(x))):
        keep = valid[i]
        result[i] = -np.polyfit(x[keep], np.log10(values[i, keep]), 1)[0]
    return result


class StagingSpindleInputs(SpindleFeatureGenerator):
    """Keep the archived FFT, spectral fit, band reduction and recording-wide clock."""

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        series = record.get(self.signal_key)
        eeg = _as_2d_signal(series.values)
        fs = float(series.sample_rate)
        window_seconds, overlap = 2.0, 0.9
        window = max(int(round(fs * window_seconds)), 8)
        stride = window - min(int(round(window * overlap)), window - 1)
        if len(eeg) < window:
            return super()._run(record, prior=prior)
        starts = np.arange(0, len(eeg) - window + 1, stride)
        timestamps = (starts + window / 2) / fs
        power = np.empty((len(starts), eeg.shape[1]))
        exponent = np.empty_like(power)
        block_frames = 4096
        for column in range(eeg.shape[1]):
            for first in range(0, len(starts), block_frames):
                stop = min(first + block_frames, len(starts))
                values = eeg[starts[first] : starts[stop - 1] + window, column]
                hz, _, spectrum = _spectrogram_abs(
                    values, fs, window_seconds=window_seconds, overlap_fraction=overlap
                )
                power[first:stop, column] = _band_median(spectrum, hz, 1.0, 4.0)
                exponent[first:stop, column] = _aperiodic_exponent(spectrum, hz)

        rate = _time_series_rate(timestamps)
        clock = {"start_seconds": float(timestamps[0])}
        return AlgorithmResult(
            name=self.name,
            signals={
                "delta_power": SignalSeries(np.nanmean(power, axis=1), sample_rate=rate, metadata=clock),
                "aperiodic_exponent": SignalSeries(np.nanmean(exponent, axis=1), sample_rate=rate, metadata=clock),
            },
        )
