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
    base_window_seconds: int = 6,
    target_window_seconds: int = 60,
    positive_threshold_seconds: float = 5.0,
    dropped_subject_ids: Iterable[str] = PAPER_APNEA_DROPPED_SUBJECTS,
) -> pd.DataFrame:
    drop_set = {str(value) for value in dropped_subject_ids}
    filtered = table.loc[~table[subject_column].astype(str).isin(drop_set)].copy()
    return _aggregate_nonoverlapping(
        filtered,
        feature_columns=feature_columns,
        subject_column=subject_column,
        time_column=time_column,
        base_window_seconds=base_window_seconds,
        target_window_seconds=target_window_seconds,
        target_builder=lambda block, seconds_per_row: _build_apnea_target_row(
            block,
            seconds_per_row=seconds_per_row,
            positive_threshold_seconds=positive_threshold_seconds,
        ),
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
    positive_threshold_seconds: float,
) -> dict[str, object] | None:
    if "label.respiratory_event_fraction" not in block.columns:
        return None
    frac = pd.to_numeric(block["label.respiratory_event_fraction"], errors="coerce").to_numpy(dtype=float)
    if frac.size == 0:
        return None
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
    event_seconds = float(np.sum(frac) * float(seconds_per_row))
    target_value = int(event_seconds > float(positive_threshold_seconds))

    severity = _mode_int(block.get("label.respiratory_apnea"))
    obstruction = _mode_int(block.get("label.respiratory_obstruction"))
    return {
        "label.apnea_60s_gt5s": target_value,
        "label.apnea_60s_event_seconds": event_seconds,
        "label.apnea_60s_event_fraction": event_seconds / float(len(block) * seconds_per_row),
        "label.apnea_60s_apnea_mode": severity,
        "label.apnea_60s_obstruction_mode": obstruction,
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


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))
