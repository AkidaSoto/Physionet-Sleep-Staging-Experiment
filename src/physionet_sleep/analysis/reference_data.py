from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physionet_sleep.core.types import RecordLabels
from physionet_sleep.io.ucddb import discover_ucddb_records, resolve_ucddb_record
from physionet_sleep.labels.ucddb import load_ucddb_labels
from physionet_sleep.recipe.algo_store import ParquetAlgoStore

RESP_APNEA_CLASS_MAP = {
    0: "none",
    1: "hypopnea",
    2: "apnea",
}

RESP_OBSTRUCTION_CLASS_MAP = {
    0: "none",
    1: "central",
    2: "obstructive",
    3: "mixed",
}


def load_ucddb_reference_dataset(
    *,
    cache_dir: str | Path = "artifacts/ucddb_recipe_cache",
    root_dir: str | Path = "data/raw/ucddb",
    subject_ids: list[str] | None = None,
    feature_table_name: str = "final__feature_table",
    window_seconds: float = 6.0,
) -> pd.DataFrame:
    cache_root = Path(cache_dir)
    root = Path(root_dir)
    store = ParquetAlgoStore()
    subject_order = subject_ids or [paths.record_id for paths in discover_ucddb_records(root)]

    tables: list[pd.DataFrame] = []
    for subject_id in subject_order:
        subject_cache_dir = cache_root / subject_id
        if not subject_cache_dir.exists():
            continue
        loaded = store.load_one(str(subject_cache_dir), feature_table_name)
        if loaded is None:
            continue
        table = loaded.artifacts.get("feature_table")
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        labels = load_ucddb_labels(resolve_ucddb_record(root, subject_id))
        enriched = attach_ucddb_labels_to_feature_table(
            table,
            labels=labels,
            subject_id=subject_id,
            window_seconds=window_seconds,
        )
        tables.append(enriched)

    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, axis=0, ignore_index=True)
    drop_cols = [col for col in combined.columns if "__eeg_" in col]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)
    return combined


def attach_ucddb_labels_to_feature_table(
    table: pd.DataFrame,
    *,
    labels: RecordLabels,
    subject_id: str,
    window_seconds: float = 6.0,
) -> pd.DataFrame:
    out = table.copy()
    if "time_seconds" not in out.columns:
        out.insert(0, "time_seconds", np.arange(len(out), dtype=float) * window_seconds)
    out.insert(0, "subject_id", subject_id)

    time_seconds = out["time_seconds"].to_numpy(dtype=float, copy=False)
    window_seconds_int = max(int(round(window_seconds)), 1)

    if "stage_seconds_collapsed" in labels.tracks:
        stage_track = np.asarray(labels.tracks["stage_seconds_collapsed"].values, dtype=float)
        stage_values = _window_reduce_track(
            stage_track,
            time_seconds=time_seconds,
            window_seconds=window_seconds_int,
            mode="mode",
        )
        out["label.stage_seconds"] = stage_values
        class_map = labels.tracks["stage_seconds_collapsed"].class_map
        out["label.stage_seconds_name"] = [
            class_map.get(int(value), "unknown") if np.isfinite(value) else None for value in stage_values
        ]

    resp_mask = _respiratory_truth_mask(labels, time_seconds, window_seconds_int)
    out["label.respiratory_event_truth"] = resp_mask
    out["label.respiratory_event_fraction"] = _respiratory_truth_fraction(labels, time_seconds, window_seconds_int)
    resp_apnea_track, resp_obstruction_track = _respiratory_truth_tracks(labels, time_seconds, window_seconds_int)
    out["label.respiratory_apnea"] = resp_apnea_track
    out["label.respiratory_apnea_name"] = [
        RESP_APNEA_CLASS_MAP.get(int(value), "unknown") if np.isfinite(value) else None for value in resp_apnea_track
    ]
    out["label.respiratory_ge_hypopnea"] = (resp_apnea_track >= 1).astype(int)
    out["label.respiratory_ge_apnea"] = (resp_apnea_track >= 2).astype(int)
    out["label.respiratory_obstruction"] = resp_obstruction_track
    out["label.respiratory_obstruction_name"] = [
        RESP_OBSTRUCTION_CLASS_MAP.get(int(value), "unknown") if np.isfinite(value) else None
        for value in resp_obstruction_track
    ]

    if "apnea_events.apnea_event_vector" in out.columns:
        out["pred.apnea_event_prediction"] = (out["apnea_events.apnea_event_vector"].to_numpy(dtype=float) > 0).astype(int)

    return out


def _window_reduce_track(
    track: np.ndarray,
    *,
    time_seconds: np.ndarray,
    window_seconds: int,
    mode: str,
) -> np.ndarray:
    values = np.full(time_seconds.shape[0], np.nan, dtype=float)
    if track.size == 0:
        return values
    for idx, start_sec in enumerate(time_seconds):
        start = max(int(np.floor(start_sec)), 0)
        end = min(start + window_seconds, track.size)
        if end <= start:
            continue
        chunk = track[start:end]
        finite = chunk[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        if mode == "mode":
            rounded = np.rint(finite).astype(int)
            unique, counts = np.unique(rounded, return_counts=True)
            values[idx] = float(unique[int(np.argmax(counts))])
        else:
            values[idx] = float(np.mean(finite))
    return values


def _respiratory_truth_mask(
    labels: RecordLabels,
    time_seconds: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    fraction = _respiratory_truth_fraction(labels, time_seconds, window_seconds)
    return (fraction > 0).astype(int)


def _respiratory_truth_fraction(
    labels: RecordLabels,
    time_seconds: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    total_length = _infer_total_seconds(labels, time_seconds, window_seconds)
    truth = np.zeros(total_length, dtype=float)
    for event in labels.events.get("respiratory", []):
        start = max(int(event.start), 0)
        end = min(int(event.end) + 1, total_length)
        if end > start:
            truth[start:end] = 1.0

    frac = np.zeros(time_seconds.shape[0], dtype=float)
    for idx, start_sec in enumerate(time_seconds):
        start = max(int(np.floor(start_sec)), 0)
        end = min(start + window_seconds, total_length)
        if end <= start:
            continue
        frac[idx] = float(np.mean(truth[start:end]))
    return frac


def _respiratory_truth_tracks(
    labels: RecordLabels,
    time_seconds: np.ndarray,
    window_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_length = _infer_total_seconds(labels, time_seconds, window_seconds)
    apnea_truth = np.zeros(total_length, dtype=float)
    obstruction_truth = np.zeros(total_length, dtype=float)
    for event in labels.events.get("respiratory", []):
        start = max(int(event.start), 0)
        end = min(int(event.end) + 1, total_length)
        if end <= start:
            continue
        type_raw = str(event.metadata.get("type_raw", "")).upper()
        apnea_code, obstruction_code = _parse_respiratory_type(type_raw)
        apnea_truth[start:end] = float(apnea_code)
        obstruction_truth[start:end] = float(obstruction_code)
    apnea_values = _window_reduce_track(
        apnea_truth,
        time_seconds=time_seconds,
        window_seconds=window_seconds,
        mode="mode",
    )
    obstruction_values = _window_reduce_track(
        obstruction_truth,
        time_seconds=time_seconds,
        window_seconds=window_seconds,
        mode="mode",
    )
    apnea_values = np.nan_to_num(apnea_values, nan=0.0)
    obstruction_values = np.nan_to_num(obstruction_values, nan=0.0)
    return apnea_values.astype(int), obstruction_values.astype(int)


def _parse_respiratory_type(type_raw: str) -> tuple[int, int]:
    if type_raw.startswith("HYP"):
        apnea_code = 1
    elif type_raw.startswith("APNEA"):
        apnea_code = 2
    else:
        apnea_code = 0

    if type_raw.endswith("-C"):
        obstruction_code = 1
    elif type_raw.endswith("-O"):
        obstruction_code = 2
    elif type_raw.endswith("-M"):
        obstruction_code = 3
    else:
        obstruction_code = 0
    return apnea_code, obstruction_code


def _infer_total_seconds(labels: RecordLabels, time_seconds: np.ndarray, window_seconds: int) -> int:
    candidates = [int(np.ceil(np.nanmax(time_seconds) + window_seconds)) if time_seconds.size else 0]
    if "stage_seconds_collapsed" in labels.tracks:
        candidates.append(int(np.asarray(labels.tracks["stage_seconds_collapsed"].values).size))
    if "respiratory" in labels.events and labels.events["respiratory"]:
        candidates.append(max(int(event.end) + 1 for event in labels.events["respiratory"]))
    return max(candidates + [0])
