from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, SignalRecord, SignalSeries


class FeatureTableCompiler(BaseAlgorithm):
    def __init__(
        self,
        *,
        include_algorithms: tuple[str, ...] | None = None,
        window_seconds: float = 6.0,
        include_feature_arrays: bool = False,
        name: str = "feature_table",
    ) -> None:
        super().__init__(name=name)
        self.include_algorithms = include_algorithms
        self.window_seconds = float(window_seconds)
        self.include_feature_arrays = include_feature_arrays

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
    )


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
) -> np.ndarray | None:
    if target_length <= 0 or values.size == 0 or sample_rate <= 0 or window_seconds <= 0:
        return None
    samples_per_window = max(int(round(sample_rate * window_seconds)), 1)
    out = np.full(target_length, np.nan, dtype=float)
    for idx in range(target_length):
        start = idx * samples_per_window
        end = min(start + samples_per_window, values.size)
        if end <= start:
            break
        chunk = values[start:end]
        finite = chunk[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        if mode == "max":
            out[idx] = float(np.max(finite))
        else:
            out[idx] = float(np.mean(finite))
    return out


def _resample_1d(values: np.ndarray, target_length: int) -> np.ndarray | None:
    if target_length <= 0 or values.size < 2:
        return None
    if values.size == target_length:
        return values
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(x_new, x_old, values)
