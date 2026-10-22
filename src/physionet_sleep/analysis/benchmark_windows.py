from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

PAPER_APNEA_DROPPED_SUBJECTS = ("ucddb008", "ucddb011", "ucddb013", "ucddb018")


def build_staging_benchmark_table(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    base_window_seconds: int = 6,
    target_window_seconds: int = 30,
) -> pd.DataFrame:
    return _aggregate_nonoverlapping(
        table,
        feature_columns=feature_columns,
        subject_column=subject_column,
        time_column=time_column,
        base_window_seconds=base_window_seconds,
        target_window_seconds=target_window_seconds,
        target_builder=_build_staging_target_row,
    )


def build_apnea_benchmark_table(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    base_window_seconds: int = 1,
    target_window_seconds: int = 11,
    stride_seconds: int = 1,
    anchor_offset_seconds: int = 1,
    dropped_subject_ids: Iterable[str] = PAPER_APNEA_DROPPED_SUBJECTS,
) -> pd.DataFrame:
    drop_set = {str(value) for value in dropped_subject_ids}
    filtered = table.loc[~table[subject_column].astype(str).isin(drop_set)].copy()
    return _build_apnea_benchmark_sliding(
        filtered,
        feature_columns=feature_columns,
        subject_column=subject_column,
        time_column=time_column,
        base_window_seconds=base_window_seconds,
        target_window_seconds=target_window_seconds,
        stride_seconds=stride_seconds,
        anchor_offset_seconds=anchor_offset_seconds,
    )


def _aggregate_nonoverlapping(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    subject_column: str,
    time_column: str,
    base_window_seconds: int,
    target_window_seconds: int,
    target_builder,
) -> pd.DataFrame:
    rows_per_target = int(round(target_window_seconds / base_window_seconds))
    if rows_per_target <= 0:
        raise ValueError("rows_per_target must be positive")

    required = [subject_column, time_column, *feature_columns]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise KeyError(f"Missing benchmark-window columns: {missing}")

    out_rows: list[dict[str, object]] = []
    for subject_id, subject_df in table.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        slot_index = np.rint(ordered[time_column].to_numpy(dtype=float, copy=False) / float(base_window_seconds)).astype(int)
        n_rows = len(ordered)
        for start in range(0, n_rows - rows_per_target + 1, rows_per_target):
            block = ordered.iloc[start : start + rows_per_target]
            block_slots = slot_index[start : start + rows_per_target]
            if block_slots.size != rows_per_target:
                continue
            if block_slots[0] % rows_per_target != 0:
                continue
            if not np.array_equal(block_slots, np.arange(block_slots[0], block_slots[0] + rows_per_target)):
                continue
            target_fields = target_builder(block, base_window_seconds)
            if target_fields is None:
                continue
            feature_values = {
                feature: _finite_mean(block[feature].to_numpy(dtype=float, copy=False))
                for feature in feature_columns
            }
            row = {
                subject_column: str(subject_id),
                time_column: float(block[time_column].iloc[0]),
                "window_seconds": float(target_window_seconds),
                "window_end_seconds": float(block[time_column].iloc[-1] + base_window_seconds),
            }
            row.update(feature_values)
            row.update(target_fields)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def _aggregate_sliding(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    subject_column: str,
    time_column: str,
    base_window_seconds: int,
    target_window_seconds: int,
    stride_seconds: int,
    target_builder,
) -> pd.DataFrame:
    rows_per_target = int(round(target_window_seconds / base_window_seconds))
    rows_per_stride = int(round(stride_seconds / base_window_seconds))
    if rows_per_target <= 0:
        raise ValueError("rows_per_target must be positive")
    if rows_per_stride <= 0:
        raise ValueError("rows_per_stride must be positive")

    required = [subject_column, time_column, *feature_columns]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise KeyError(f"Missing benchmark-window columns: {missing}")

    out_rows: list[dict[str, object]] = []
    for subject_id, subject_df in table.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        slot_index = np.rint(ordered[time_column].to_numpy(dtype=float, copy=False) / float(base_window_seconds)).astype(int)
        n_rows = len(ordered)
        for start in range(0, n_rows - rows_per_target + 1, rows_per_stride):
            stop = start + rows_per_target
            block = ordered.iloc[start:stop]
            block_slots = slot_index[start:stop]
            if block_slots.size != rows_per_target:
                continue
            if not np.array_equal(block_slots, np.arange(block_slots[0], block_slots[0] + rows_per_target)):
                continue
            target_fields = target_builder(block, base_window_seconds)
            if target_fields is None:
                continue
            feature_values = {
                feature: _finite_mean(block[feature].to_numpy(dtype=float, copy=False))
                for feature in feature_columns
            }
            row = {
                subject_column: str(subject_id),
                time_column: float(block[time_column].iloc[0]),
                "window_seconds": float(target_window_seconds),
                "window_end_seconds": float(block[time_column].iloc[-1] + base_window_seconds),
                "window_stride_seconds": float(stride_seconds),
            }
            row.update(feature_values)
            row.update(target_fields)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def _build_apnea_benchmark_sliding(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    subject_column: str,
    time_column: str,
    base_window_seconds: int,
    target_window_seconds: int,
    stride_seconds: int,
    anchor_offset_seconds: int,
) -> pd.DataFrame:
    rows_per_target = int(round(target_window_seconds / base_window_seconds))
    rows_per_stride = int(round(stride_seconds / base_window_seconds))
    if rows_per_target <= 0:
        raise ValueError("rows_per_target must be positive")
    if rows_per_stride <= 0:
        raise ValueError("rows_per_stride must be positive")

    required = [
        subject_column,
        time_column,
        *feature_columns,
        "label.respiratory_event_fraction",
        "label.respiratory_event_truth",
        "label.respiratory_apnea",
        "label.respiratory_obstruction",
    ]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise KeyError(f"Missing benchmark-window columns: {missing}")

    frames: list[pd.DataFrame] = []
    anchor_row_offset = int(round(anchor_offset_seconds / base_window_seconds))
    for subject_id, subject_df in table.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        slot_index = np.rint(ordered[time_column].to_numpy(dtype=float, copy=False) / float(base_window_seconds)).astype(int)
        breaks = np.flatnonzero(np.diff(slot_index) != 1) + 1
        bounds = np.r_[0, breaks, len(ordered)]
        for start_idx, end_idx in zip(bounds[:-1], bounds[1:], strict=False):
            segment = ordered.iloc[start_idx:end_idx].reset_index(drop=True)
            if len(segment) < rows_per_target:
                continue
            segment_frame = _build_apnea_benchmark_segment(
                segment,
                feature_columns=feature_columns,
                subject_id=str(subject_id),
                subject_column=subject_column,
                time_column=time_column,
                base_window_seconds=base_window_seconds,
                target_window_seconds=target_window_seconds,
                rows_per_target=rows_per_target,
                rows_per_stride=rows_per_stride,
                anchor_row_offset=anchor_row_offset,
            )
            if not segment_frame.empty:
                frames.append(segment_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _build_apnea_benchmark_segment(
    segment: pd.DataFrame,
    *,
    feature_columns: list[str],
    subject_id: str,
    subject_column: str,
    time_column: str,
    base_window_seconds: int,
    target_window_seconds: int,
    rows_per_target: int,
    rows_per_stride: int,
    anchor_row_offset: int,
) -> pd.DataFrame:
    n_rows = len(segment)
    window_starts = np.arange(0, n_rows - rows_per_target + 1, rows_per_stride, dtype=int)
    if window_starts.size == 0:
        return pd.DataFrame()
    window_ends = window_starts + rows_per_target

    out: dict[str, object] = {
        subject_column: np.repeat(subject_id, window_starts.size),
        time_column: segment[time_column].to_numpy(dtype=float, copy=False)[window_starts],
        "window_seconds": np.repeat(float(target_window_seconds), window_starts.size),
        "window_end_seconds": segment[time_column].to_numpy(dtype=float, copy=False)[window_ends - 1] + float(base_window_seconds),
        "window_stride_seconds": np.repeat(float(rows_per_stride * base_window_seconds), window_starts.size),
    }

    feature_matrix = segment[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
    feature_means = _window_nanmean_matrix(feature_matrix, rows_per_target, rows_per_stride)
    for idx, feature_name in enumerate(feature_columns):
        out[feature_name] = feature_means[:, idx]

    frac = pd.to_numeric(segment["label.respiratory_event_fraction"], errors="coerce").to_numpy(dtype=float, copy=False)
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
    event_seconds = _window_sum_1d(frac, rows_per_target, rows_per_stride) * float(base_window_seconds)
    out["label.apnea_11s_event_seconds"] = event_seconds
    out["label.apnea_11s_event_fraction"] = event_seconds / float(rows_per_target * base_window_seconds)

    anchor_indices = np.clip(window_starts + anchor_row_offset, 0, n_rows - 1)
    truth = pd.to_numeric(segment["label.respiratory_event_truth"], errors="coerce").to_numpy(dtype=float, copy=False)
    apnea = pd.to_numeric(segment["label.respiratory_apnea"], errors="coerce").to_numpy(dtype=float, copy=False)
    obstruction = pd.to_numeric(segment["label.respiratory_obstruction"], errors="coerce").to_numpy(dtype=float, copy=False)
    out["label.apnea_11s_anchor1_event"] = _take_int_or_zero(truth, anchor_indices)
    out["label.apnea_11s_anchor1_apnea"] = _take_int_or_zero(apnea, anchor_indices)
    out["label.apnea_11s_anchor1_obstruction"] = _take_int_or_zero(obstruction, anchor_indices)

    return pd.DataFrame(out)


def _build_staging_target_row(block: pd.DataFrame, seconds_per_row: int) -> dict[str, object] | None:
    if "label.stage_seconds" not in block.columns:
        return None
    values = pd.to_numeric(block["label.stage_seconds"], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    finite = finite[finite >= 0]
    if finite.size == 0:
        return None
    rounded = np.rint(finite).astype(int)
    unique, counts = np.unique(rounded, return_counts=True)
    stage_value = int(unique[int(np.argmax(counts))])
    return {"label.stage_30s": stage_value}


def _build_apnea_target_row(
    block: pd.DataFrame,
    *,
    seconds_per_row: int,
    anchor_offset_seconds: int,
) -> dict[str, object] | None:
    if "label.respiratory_event_fraction" not in block.columns:
        return None
    frac = pd.to_numeric(block["label.respiratory_event_fraction"], errors="coerce").to_numpy(dtype=float)
    if frac.size == 0:
        return None
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
    event_seconds = float(np.sum(frac) * float(seconds_per_row))
    anchor_index = min(max(int(round(anchor_offset_seconds / max(seconds_per_row, 1))), 0), len(block) - 1)
    anchor_truth = _row_int(block.get("label.respiratory_event_truth"), anchor_index)
    severity = _row_int(block.get("label.respiratory_apnea"), anchor_index)
    obstruction = _row_int(block.get("label.respiratory_obstruction"), anchor_index)
    return {
        "label.apnea_11s_anchor1_event": anchor_truth,
        "label.apnea_11s_event_seconds": event_seconds,
        "label.apnea_11s_event_fraction": event_seconds / float(len(block) * seconds_per_row),
        "label.apnea_11s_anchor1_apnea": severity,
        "label.apnea_11s_anchor1_obstruction": obstruction,
    }


def _mode_int(series: pd.Series | None) -> int:
    if series is None:
        return 0
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    rounded = np.rint(finite).astype(int)
    unique, counts = np.unique(rounded, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def _row_int(series: pd.Series | None, index: int) -> int:
    if series is None or len(series) <= index:
        return 0
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if values.size <= index or not np.isfinite(values[index]):
        return 0
    return int(np.rint(values[index]))


def _window_sum_1d(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
    starts = np.arange(0, values.shape[0] - window + 1, stride, dtype=int)
    ends = starts + window
    return csum[ends] - csum[starts]


def _window_nanmean_matrix(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    csum = np.vstack([np.zeros((1, values.shape[1]), dtype=float), np.cumsum(safe_values, axis=0, dtype=float)])
    ccount = np.vstack([np.zeros((1, values.shape[1]), dtype=float), np.cumsum(finite.astype(float), axis=0, dtype=float)])
    starts = np.arange(0, values.shape[0] - window + 1, stride, dtype=int)
    ends = starts + window
    sums = csum[ends] - csum[starts]
    counts = ccount[ends] - ccount[starts]
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def _take_int_or_zero(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    picked = values[indices]
    out = np.zeros(indices.shape[0], dtype=int)
    finite = np.isfinite(picked)
    out[finite] = np.rint(picked[finite]).astype(int)
    return out


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))
