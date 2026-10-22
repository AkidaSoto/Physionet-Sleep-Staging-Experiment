from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, SignalRecord, SignalSeries

# Longest analysis window used by any upstream track (30-s EOG spectrogram); bounds edge padding.
_EDGE_FILL_SECONDS = 30.0


class FeatureTableCompiler(BaseAlgorithm):
    def __init__(
        self,
        *,
        include_algorithms: tuple[str, ...] | None = None,
        window_seconds: float = 6.0,
        include_feature_arrays: bool = False,
        exclude_signal_units: tuple[str, ...] = ("binary", "class"),
        exclude_signal_name_tokens: tuple[str, ...] = ("mask", "vector"),
        name: str = "feature_table",
    ) -> None:
        super().__init__(name=name)
        self.include_algorithms = include_algorithms
        self.window_seconds = float(window_seconds)
        self.include_feature_arrays = include_feature_arrays
        self.exclude_signal_units = tuple(value.lower() for value in exclude_signal_units)
        self.exclude_signal_name_tokens = tuple(value.lower() for value in exclude_signal_name_tokens)

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        target_length = _infer_target_length(record, self.window_seconds)
        target_sample_rate = 1.0 / self.window_seconds if self.window_seconds > 0 else 0.0
        columns: dict[str, np.ndarray] = {}
        scalar_features: dict[str, float] = {}

        for algo_name, result in prior.items():
            if self.include_algorithms is not None and algo_name not in self.include_algorithms:
                continue

            for signal_name, signal in result.signals.items():
                if _should_skip_signal(
                    signal_name,
                    signal,
                    excluded_units=self.exclude_signal_units,
                    excluded_name_tokens=self.exclude_signal_name_tokens,
                ):
                    continue
                values = _coerce_signal(signal, target_length, self.window_seconds)
                if values is not None:
                    columns[f"{algo_name}.{signal_name}"] = values

            for feature_name, value in result.features.items():
                if self.include_feature_arrays:
                    series = _coerce_feature(feature_name, value, target_length)
                    if series is not None:
                        columns[f"{algo_name}.{feature_name}"] = series
                        continue
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    scalar_features[f"{algo_name}.{feature_name}"] = float(value)

        if columns:
            table = pd.DataFrame(columns)
            if self.window_seconds > 0:
                table.insert(0, "time_seconds", np.arange(target_length, dtype=float) * self.window_seconds)
        else:
            table = pd.DataFrame()

        return AlgorithmResult(
            name=self.name,
            artifacts={"feature_table": table},
            features={
                "column_names": list(table.columns),
                "row_count": int(len(table)),
                "target_sample_rate": float(target_sample_rate),
                "window_seconds": float(self.window_seconds),
                "scalar_features": scalar_features,
            },
            metadata={"status": "compiled_features"},
        )


def _infer_target_length(record: SignalRecord, window_seconds: float) -> int:
    if not record.channels or window_seconds <= 0:
        return 0
    max_duration = max(series.duration_seconds for series in record.channels.values())
    return max(int(np.ceil(max_duration / window_seconds)), 1)


def _coerce_signal(series: SignalSeries, target_length: int, window_seconds: float) -> np.ndarray | None:
    values = np.asarray(series.values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        return None
    mode = "max" if series.units in {"binary", "class"} else "mean"
    return _window_reduce_1d(
        values.astype(float, copy=False),
        sample_rate=series.sample_rate,
        window_seconds=window_seconds,
        target_length=target_length,
        mode=mode,
        sample_times=_signal_sample_times(series, values.shape[0]),
    )


def _signal_sample_times(series: SignalSeries, n_samples: int) -> np.ndarray | None:
    """Time in seconds of each sample, from metadata when the producer knows it.

    `metadata["time_seconds"]` gives explicit per-sample times (e.g. spectrogram frame centres);
    `metadata["start_seconds"]` gives the time of sample 0 for an otherwise regular series.
    Without either, sample k sits at k / sample_rate.
    """
    explicit = series.metadata.get("time_seconds") if series.metadata else None
    if explicit is not None:
        times = np.asarray(explicit, dtype=float)
        if times.shape == (n_samples,):
            return times
    if series.sample_rate <= 0:
        return None
    start = float(series.metadata.get("start_seconds", 0.0)) if series.metadata else 0.0
    return start + np.arange(n_samples, dtype=float) / float(series.sample_rate)


def _should_skip_signal(
    signal_name: str,
    signal: SignalSeries,
    *,
    excluded_units: tuple[str, ...],
    excluded_name_tokens: tuple[str, ...],
) -> bool:
    units = (signal.units or "").lower()
    if units in excluded_units:
        return True
    lowered_name = signal_name.lower()
    return any(token in lowered_name for token in excluded_name_tokens)


def _coerce_feature(name: str, value, target_length: int) -> np.ndarray | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return None
    lowered = name.lower()
    if any(token in lowered for token in ("indices", "width", "time_idx", "interval", "order", "agreement")):
        return None
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size < 2 or not np.issubdtype(arr.dtype, np.number):
        return None
    return _resample_1d(arr.astype(float, copy=False), target_length)


def _window_reduce_1d(
    values: np.ndarray,
    *,
    sample_rate: float,
    window_seconds: float,
    target_length: int,
    mode: str,
    sample_times: np.ndarray | None = None,
) -> np.ndarray | None:
    """Reduce a track to rows of `window_seconds` by sample TIME, not by sample index.

    Row i holds the samples whose time falls in [i * window_seconds, (i + 1) * window_seconds).
    Index-based binning (round(rate * window) samples per row) drifted for spectrogram tracks whose
    frame hop does not divide the row width, and ignored the frame-centre offset of the first frame.
    Empty rows at the start or end of the night that lie within _EDGE_FILL_SECONDS of the track's first
    or last sample take the nearest row's value (a long analysis window centres its first frame late).
    """
    if target_length <= 0 or values.size == 0 or sample_rate <= 0 or window_seconds <= 0:
        return None
    if sample_times is None:
        sample_times = np.arange(values.size, dtype=float) / float(sample_rate)
    # The small epsilon keeps exact multiples (k / fs == i * window) on row i despite float error.
    rows = np.floor(np.asarray(sample_times, dtype=float) / float(window_seconds) + 1e-9).astype(np.int64)
    in_range = (rows >= 0) & (rows < target_length)
    finite = in_range & np.isfinite(values)
    out = np.full(target_length, np.nan, dtype=float)
    if mode == "max":
        if np.any(finite):
            np.fmax.at(out, rows[finite], values[finite])
        return out

    sums = np.bincount(rows[finite], weights=values[finite], minlength=target_length)
    counts = np.bincount(rows[finite], minlength=target_length)
    np.divide(sums, counts, out=out, where=counts > 0)

    covered = np.flatnonzero(np.bincount(rows[in_range], minlength=target_length) > 0)
    if covered.size:
        fill_rows = int(np.floor(_EDGE_FILL_SECONDS / float(window_seconds)))
        first, last = int(covered[0]), int(covered[-1])
        lead = np.arange(max(first - fill_rows, 0), first)
        trail = np.arange(last + 1, min(last + 1 + fill_rows, target_length))
        out[lead] = out[first]
        out[trail] = out[last]
    return out


def _resample_1d(values: np.ndarray, target_length: int) -> np.ndarray | None:
    if target_length <= 0 or values.size < 2:
        return None
    if values.size == target_length:
        return values
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(x_new, x_old, values)
