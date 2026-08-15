from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from physionet_sleep.core.types import DatasetRun, SignalSeries


@dataclass(slots=True)
class HarvestedSeries:
    name: str
    values: np.ndarray
    source_kind: str
    algorithm: str


def iter_harvested_series(
    dataset_run: DatasetRun,
    *,
    target_track: str = "stage_seconds",
) -> Iterator[HarvestedSeries]:
    labels = dataset_run.alignment.tracks.get(target_track)
    if labels is None:
        return
    target_len = int(np.asarray(labels).size)
    for algo_name, result in dataset_run.results.items():
        for signal_name, signal in result.signals.items():
            values = _coerce_signal_series(signal, target_len)
            if values is not None:
                yield HarvestedSeries(
                    name=f"{algo_name}.{signal_name}",
                    values=values,
                    source_kind="signal",
                    algorithm=algo_name,
                )
        for feature_name, raw_value in result.features.items():
            values = _coerce_array(raw_value, target_len)
            if values is not None:
                yield HarvestedSeries(
                    name=f"{algo_name}.{feature_name}",
                    values=values,
                    source_kind="feature",
                    algorithm=algo_name,
                )


def _coerce_signal_series(series: SignalSeries, target_len: int) -> np.ndarray | None:
    values = np.asarray(series.values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        return None
    return _resample_like_matched_length(values.astype(float, copy=False), target_len)


def _coerce_array(value, target_len: int) -> np.ndarray | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return None
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return None
    return _resample_like_matched_length(arr.astype(float, copy=False), target_len)


def _resample_like_matched_length(values: np.ndarray, target_len: int) -> np.ndarray | None:
    if values.size < 2 or target_len < 2:
        return None
    if values.size == target_len:
        return values
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(x_new, x_old, values)
