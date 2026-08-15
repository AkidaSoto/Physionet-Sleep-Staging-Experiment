from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from physionet_sleep.core.types import DatasetRun, Event


def enrich_feature_tables(
    dataset_run: DatasetRun,
    *,
    context_lags: tuple[int, ...] = (1,),
) -> None:
    for result in dataset_run.results.values():
        table = result.artifacts.get("feature_table")
        if not isinstance(table, pd.DataFrame):
            continue

        enriched = table.copy()
        target_sample_rate = float(result.features.get("target_sample_rate", 1.0) or 1.0)
        window_seconds = float(result.features.get("window_seconds", 1.0 / target_sample_rate if target_sample_rate > 0 else 1.0))
        n_rows = len(enriched)

        enriched = _attach_alignment_tracks(enriched, dataset_run, n_rows, window_seconds=window_seconds)
        enriched = _attach_event_columns(enriched, dataset_run, n_rows, target_sample_rate)
        enriched = _attach_context_columns(enriched, context_lags=context_lags)

        result.artifacts["feature_table"] = enriched
        result.features["column_names"] = list(enriched.columns)
        result.features["row_count"] = int(len(enriched))
        result.features["context_lags"] = list(context_lags)
        result.metadata["labels_attached"] = True


def _attach_alignment_tracks(
    table: pd.DataFrame,
    dataset_run: DatasetRun,
    n_rows: int,
    *,
    window_seconds: float,
) -> pd.DataFrame:
    out = table.copy()
    for track_name, values in dataset_run.alignment.tracks.items():
        label_track = dataset_run.labels.tracks.get(track_name)
        is_discrete = bool(label_track is not None and label_track.class_map) or track_name == "stage_seconds"
        mapped = _downsample_track(np.asarray(values), n_rows, window_seconds=window_seconds, discrete=is_discrete)
        out[f"label.{track_name}"] = mapped

        if label_track is not None and label_track.class_map:
            out[f"label.{track_name}_name"] = [
                label_track.class_map.get(int(v), "unknown") if np.isfinite(v) else None for v in mapped
            ]
        elif track_name == "stage_seconds" and "stage_seconds_collapsed" in dataset_run.labels.tracks:
            class_map = dataset_run.labels.tracks["stage_seconds_collapsed"].class_map
            out["label.stage_seconds_name"] = [
                class_map.get(int(v), "unknown") if np.isfinite(v) else None for v in mapped
            ]
    return out


def _attach_event_columns(
    table: pd.DataFrame,
    dataset_run: DatasetRun,
    n_rows: int,
    target_sample_rate: float,
) -> pd.DataFrame:
    out = table.copy()
    out = _attach_event_group(
        out,
        prefix="truth",
        event_groups=dataset_run.labels.events,
        n_rows=n_rows,
        target_sample_rate=target_sample_rate,
    )
    out = _attach_event_group(
        out,
        prefix="aligned",
        event_groups=dataset_run.alignment.events,
        n_rows=n_rows,
        target_sample_rate=target_sample_rate,
    )
    return out


def _attach_event_group(
    table: pd.DataFrame,
    *,
    prefix: str,
    event_groups: dict[str, list[Event]],
    n_rows: int,
    target_sample_rate: float,
) -> pd.DataFrame:
    out = table.copy()
    for key, events in event_groups.items():
        if not events:
            continue
        out[f"{prefix}.{key}.any"] = _event_mask(events, n_rows=n_rows, target_sample_rate=target_sample_rate)
        labels = sorted({event.label for event in events if event.label})
        for label in labels:
            label_events = [event for event in events if event.label == label]
            safe_label = label.replace(" ", "_")
            out[f"{prefix}.{key}.{safe_label}"] = _event_mask(
                label_events,
                n_rows=n_rows,
                target_sample_rate=target_sample_rate,
            )
    return out


def _attach_context_columns(table: pd.DataFrame, *, context_lags: Iterable[int]) -> pd.DataFrame:
    out = table.copy()
    protected_prefixes = ("label.", "truth.", "aligned.")
    numeric_cols = [
        col
        for col in out.columns
        if col != "time_seconds"
        and not col.startswith(protected_prefixes)
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    for lag in context_lags:
        for col in numeric_cols:
            out[f"{col}.prev{lag}"] = out[col].shift(lag)
    return out


def _downsample_track(values: np.ndarray, n_rows: int, *, window_seconds: float, discrete: bool) -> np.ndarray:
    if values.size == 0:
        return np.full(n_rows, np.nan, dtype=float)
    if n_rows <= 0:
        return np.array([], dtype=float)
    if window_seconds <= 0:
        return _resample_track(values, n_rows)
    source = np.asarray(values)
    samples_per_window = max(int(round(window_seconds)), 1)
    out = np.full(n_rows, np.nan, dtype=float)
    for idx in range(n_rows):
        start = idx * samples_per_window
        end = min(start + samples_per_window, source.size)
        if end <= start:
            break
        chunk = source[start:end]
        finite = chunk[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        if discrete:
            rounded = np.rint(finite).astype(int)
            vals, counts = np.unique(rounded, return_counts=True)
            out[idx] = float(vals[int(np.argmax(counts))])
        else:
            out[idx] = float(np.mean(finite))
    return out


def _resample_track(values: np.ndarray, n_rows: int) -> np.ndarray:
    if values.size == 0:
        return np.full(n_rows, np.nan, dtype=float)
    if values.size == n_rows:
        return values
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=n_rows, endpoint=True)
    if np.issubdtype(values.dtype, np.number):
        return np.interp(x_new, x_old, values.astype(float, copy=False))
    idx = np.clip(np.round(x_new * (values.size - 1)).astype(int), 0, values.size - 1)
    return values[idx]


def _event_mask(events: list[Event], *, n_rows: int, target_sample_rate: float) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=int)
    if n_rows == 0 or target_sample_rate <= 0:
        return mask
    for event in events:
        start_sec, end_sec = _event_seconds(event)
        start_idx = max(int(np.floor(start_sec * target_sample_rate)), 0)
        end_idx = min(int(np.floor(end_sec * target_sample_rate)), n_rows - 1)
        if end_idx < start_idx:
            end_idx = start_idx
        mask[start_idx : end_idx + 1] = 1
    return mask


def _event_seconds(event: Event) -> tuple[float, float]:
    meta = event.metadata or {}
    if "start_sec" in meta:
        start_sec = float(meta.get("start_sec", 0.0))
        end_sec = float(meta.get("end_sec", start_sec))
        return start_sec, end_sec
    return float(event.start), float(event.end)
