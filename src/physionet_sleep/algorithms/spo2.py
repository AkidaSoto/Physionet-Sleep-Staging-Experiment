from __future__ import annotations

from typing import Mapping

import numpy as np

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, Event, SignalRecord, SignalSeries
from physionet_sleep.utils.overlap import has_interval_overlap
from physionet_sleep.utils.runs import count_true_runs, expand_runs
from physionet_sleep.utils.signal import future_rolling_min


class SpO2DropDetector(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "spo2") -> None:
        super().__init__(
            name="spo2_drop",
            references=("external/st_vincent_analysis/apnea/algo_sleep_spo2_drop.m",),
        )
        self.signal_key = signal_key

    def required_signals(self) -> tuple[str, ...]:
        return (self.signal_key,)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        series = record.get(self.signal_key)
        spo2 = np.asarray(series.values, dtype=float).squeeze()
        fs = float(series.sample_rate)

        bad_mask = _compute_bad_spo2_mask(spo2, fs)
        bad_starts, bad_lengths = count_true_runs(bad_mask)

        seg3_mask, seg4_mask = _segment_drop_masks(spo2)
        jag3_mask = _jagged_drop_mask(spo2, fs, threshold=2.8)
        jag4_mask = _jagged_drop_mask(spo2, fs, threshold=3.8)

        drop3_mask = seg3_mask | jag3_mask
        drop4_mask = seg4_mask | jag4_mask

        d3_starts, d3_lengths = count_true_runs(drop3_mask)
        d4_starts, d4_lengths = count_true_runs(drop4_mask)
        if bad_starts.size:
            d3_keep = ~has_interval_overlap(d3_starts, d3_lengths, bad_starts, bad_lengths)
            d4_keep = ~has_interval_overlap(d4_starts, d4_lengths, bad_starts, bad_lengths)
            d3_starts, d3_lengths = d3_starts[d3_keep], d3_lengths[d3_keep]
            d4_starts, d4_lengths = d4_starts[d4_keep], d4_lengths[d4_keep]

        clean_drop3 = expand_runs(d3_starts, d3_lengths, spo2.size)
        clean_drop4 = expand_runs(d4_starts, d4_lengths, spo2.size)

        return AlgorithmResult(
            name=self.name,
            signals={
                "spo2_bad_mask": SignalSeries(bad_mask.astype(float), sample_rate=fs, units="binary"),
                "spo2_drop3_mask": SignalSeries(clean_drop3.astype(float), sample_rate=fs, units="binary"),
                "spo2_drop4_mask": SignalSeries(clean_drop4.astype(float), sample_rate=fs, units="binary"),
            },
            events={
                "bad": _build_period_events(bad_starts, bad_lengths, "bad_spo2", fs),
                "drop_3": _build_period_events(d3_starts, d3_lengths, "spo2_drop_3", fs),
                "drop_4": _build_period_events(d4_starts, d4_lengths, "spo2_drop_4", fs),
            },
            features={
                "input_signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "num_bad_events": int(bad_starts.size),
                "num_drop3_events": int(d3_starts.size),
                "num_drop4_events": int(d4_starts.size),
            },
            metadata={"status": "ported_feature_extraction"},
        )


def _compute_bad_spo2_mask(spo2: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    bad_buffer = max(int(round(60.0 * sample_rate_hz)), 1)
    starts, lengths = count_true_runs(spo2 <= 0)
    if starts.size == 0:
        return np.zeros(spo2.size, dtype=bool)
    expanded_start = np.maximum(starts - bad_buffer, 0)
    expanded_end = np.minimum(expanded_start + lengths + 2 * bad_buffer, spo2.size)
    mask = np.zeros(spo2.size, dtype=bool)
    for start, end in zip(expanded_start, expanded_end, strict=False):
        mask[start:end] = True
    return mask


def _segment_drop_masks(spo2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff_spo2 = np.diff(spo2)
    rise_starts, _ = count_true_runs(diff_spo2 > 0.5)
    bounds = np.unique(np.r_[0, rise_starts + 1, spo2.size])
    drop3 = np.zeros(spo2.size, dtype=bool)
    drop4 = np.zeros(spo2.size, dtype=bool)

    for start, end in zip(bounds[:-1], bounds[1:], strict=False):
        if end - start < 2:
            continue
        segment = spo2[start:end]
        neg = np.flatnonzero(np.diff(segment) < -0.3)
        if neg.size == 0:
            first = start
            last = end - 1
        else:
            first = start + int(neg[0])
            last = start + int(neg[-1]) + 1
        if np.ptp(segment) >= 2.8:
            drop3[first : last + 1] = True
        if np.ptp(segment) >= 3.8:
            drop4[first : last + 1] = True
    return drop3, drop4


def _jagged_drop_mask(spo2: np.ndarray, sample_rate_hz: float, *, threshold: float) -> np.ndarray:
    window = max(int(round(30.0 * sample_rate_hz)), 1)
    future_min = future_rolling_min(spo2, window)
    candidate = (spo2 - future_min) > threshold
    starts, lengths = count_true_runs(candidate)
    if starts.size == 0:
        return np.zeros(spo2.size, dtype=bool)

    out = np.zeros(spo2.size, dtype=bool)
    for start, length in zip(starts, lengths, strict=False):
        run_end = int(start + length - 1)
        end = min(run_end + window, spo2.size)
        local = spo2[run_end:end]
        if local.size == 0:
            continue
        min_offset = int(np.argmin(local))
        out[run_end : run_end + min_offset + 1] = True
    return out


def _build_period_events(
    starts: np.ndarray,
    lengths: np.ndarray,
    label: str,
    sample_rate_hz: float,
) -> list[Event]:
    return [
        Event(
            start=int(start),
            end=int(start + length - 1),
            label=label,
            kind="period",
            metadata={
                "sample_rate_hz": sample_rate_hz,
                "start_sec": float(start / sample_rate_hz),
                "end_sec": float((start + length - 1) / sample_rate_hz),
            },
        )
        for start, length in zip(starts, lengths, strict=False)
    ]
