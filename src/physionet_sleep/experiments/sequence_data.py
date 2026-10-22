from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]
    context_radius: int
    window_length: int


def build_epoch_aligned_sequence_dataset(
    feature_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    history_steps: int,
    current_steps: int,
    future_steps: int,
    expected_step_seconds: float = 6.0,
    step_tolerance_seconds: float = 0.25,
) -> SequenceDataset:
    required = [subject_column, time_column, target_column, *feature_columns]
    missing = [col for col in required if col not in feature_table.columns]
    if missing:
        raise KeyError(f"Missing columns for sequence dataset: {missing}")

    history_steps = int(history_steps)
    current_steps = int(current_steps)
    future_steps = int(future_steps)
    if history_steps < 0 or current_steps <= 0 or future_steps < 0:
        raise ValueError("history_steps/future_steps must be non-negative and current_steps must be positive")

    window_length = history_steps + current_steps + future_steps
    frame = feature_table[required].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[subject_column, time_column, target_column])

    seq_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, object]] = []

    for subject_id, subject_df in frame.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        times = ordered[time_column].to_numpy(dtype=float, copy=False)
        y = ordered[target_column].to_numpy(copy=False)
        X = ordered[feature_columns].to_numpy(dtype=float, copy=False)
        slot_index = np.rint(times / float(expected_step_seconds)).astype(int)

        if len(ordered) < window_length:
            continue

        max_anchor_start = len(ordered) - current_steps - future_steps
        for anchor_start in range(history_steps, max_anchor_start + 1):
            current_slot = slot_index[anchor_start]
            if current_slot % current_steps != 0:
                continue
            seq_start = anchor_start - history_steps
            seq_stop = anchor_start + current_steps + future_steps
            seq_times = times[seq_start:seq_stop]
            if seq_times.size != window_length:
                continue
            if not _is_contiguous_sequence(
                seq_times,
                expected_step_seconds=expected_step_seconds,
                step_tolerance_seconds=step_tolerance_seconds,
            ):
                continue

            target_values = pd.to_numeric(pd.Series(y[anchor_start : anchor_start + current_steps]), errors="coerce").to_numpy(dtype=float)
            finite_target = target_values[np.isfinite(target_values)]
            finite_target = finite_target[finite_target >= 0]
            if finite_target.size == 0:
                continue
            rounded = np.rint(finite_target).astype(int)
            unique, counts = np.unique(rounded, return_counts=True)
            target_value = float(unique[int(np.argmax(counts))])

            seq_list.append(X[seq_start:seq_stop])
            y_list.append(target_value)
            meta_rows.append(
                {
                    subject_column: str(subject_id),
                    time_column: float(times[anchor_start]),
                    "sequence_start_seconds": float(seq_times[0]),
                    "sequence_end_seconds": float(seq_times[-1]),
                    "target_start_seconds": float(times[anchor_start]),
                    "target_end_seconds": float(times[anchor_start + current_steps - 1] + expected_step_seconds),
                }
            )

    if not seq_list:
        return SequenceDataset(
            X=np.empty((0, window_length, len(feature_columns)), dtype=float),
            y=np.empty((0,), dtype=float),
            metadata=pd.DataFrame(
                columns=[
                    subject_column,
                    time_column,
                    "sequence_start_seconds",
                    "sequence_end_seconds",
                    "target_start_seconds",
                    "target_end_seconds",
                ]
            ),
            feature_columns=list(feature_columns),
            context_radius=history_steps,
            window_length=window_length,
        )

    return SequenceDataset(
        X=np.stack(seq_list, axis=0),
        y=np.asarray(y_list),
        metadata=pd.DataFrame(meta_rows),
        feature_columns=list(feature_columns),
        context_radius=history_steps,
        window_length=window_length,
    )


def build_hierarchical_epoch_sequence_dataset(
    feature_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    history_epochs: int,
    current_epochs: int,
    future_epochs: int,
    epoch_steps: int,
    expected_step_seconds: float = 6.0,
    step_tolerance_seconds: float = 0.25,
) -> SequenceDataset:
    required = [subject_column, time_column, target_column, *feature_columns]
    missing = [col for col in required if col not in feature_table.columns]
    if missing:
        raise KeyError(f"Missing columns for sequence dataset: {missing}")

    history_epochs = int(history_epochs)
    current_epochs = int(current_epochs)
    future_epochs = int(future_epochs)
    epoch_steps = int(epoch_steps)
    if history_epochs < 0 or current_epochs <= 0 or future_epochs < 0:
        raise ValueError("history_epochs/future_epochs must be non-negative and current_epochs must be positive")
    if epoch_steps <= 0:
        raise ValueError("epoch_steps must be positive")

    total_epochs = history_epochs + current_epochs + future_epochs
    total_steps = total_epochs * epoch_steps
    frame = feature_table[required].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[subject_column, time_column, target_column])

    seq_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, object]] = []

    history_step_count = history_epochs * epoch_steps
    current_step_count = current_epochs * epoch_steps
    future_step_count = future_epochs * epoch_steps

    for subject_id, subject_df in frame.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        times = ordered[time_column].to_numpy(dtype=float, copy=False)
        y = ordered[target_column].to_numpy(copy=False)
        X = ordered[feature_columns].to_numpy(dtype=float, copy=False)
        slot_index = np.rint(times / float(expected_step_seconds)).astype(int)

        if len(ordered) < total_steps:
            continue

        max_anchor_start = len(ordered) - current_step_count - future_step_count
        for anchor_start in range(history_step_count, max_anchor_start + 1):
            current_slot = slot_index[anchor_start]
            if current_slot % epoch_steps != 0:
                continue
            seq_start = anchor_start - history_step_count
            seq_stop = anchor_start + current_step_count + future_step_count
            seq_times = times[seq_start:seq_stop]
            if seq_times.size != total_steps:
                continue
            if not _is_contiguous_sequence(
                seq_times,
                expected_step_seconds=expected_step_seconds,
                step_tolerance_seconds=step_tolerance_seconds,
            ):
                continue

            target_values = pd.to_numeric(
                pd.Series(y[anchor_start : anchor_start + current_step_count]),
                errors="coerce",
            ).to_numpy(dtype=float)
            finite_target = target_values[np.isfinite(target_values)]
            finite_target = finite_target[finite_target >= 0]
            if finite_target.size == 0:
                continue
            rounded = np.rint(finite_target).astype(int)
            unique, counts = np.unique(rounded, return_counts=True)
            target_value = float(unique[int(np.argmax(counts))])

            epoch_tensor = X[seq_start:seq_stop].reshape(total_epochs, epoch_steps, len(feature_columns))
            seq_list.append(epoch_tensor)
            y_list.append(target_value)
            meta_rows.append(
                {
                    subject_column: str(subject_id),
                    time_column: float(times[anchor_start]),
                    "sequence_start_seconds": float(seq_times[0]),
                    "sequence_end_seconds": float(seq_times[-1]),
                    "target_start_seconds": float(times[anchor_start]),
                    "target_end_seconds": float(times[anchor_start + current_step_count - 1] + expected_step_seconds),
                }
            )

    if not seq_list:
        return SequenceDataset(
            X=np.empty((0, total_epochs, epoch_steps, len(feature_columns)), dtype=float),
            y=np.empty((0,), dtype=float),
            metadata=pd.DataFrame(
                columns=[
                    subject_column,
                    time_column,
                    "sequence_start_seconds",
                    "sequence_end_seconds",
                    "target_start_seconds",
                    "target_end_seconds",
                ]
            ),
            feature_columns=list(feature_columns),
            context_radius=history_epochs,
            window_length=total_epochs,
        )

    return SequenceDataset(
        X=np.stack(seq_list, axis=0),
        y=np.asarray(y_list),
        metadata=pd.DataFrame(meta_rows),
        feature_columns=list(feature_columns),
        context_radius=history_epochs,
        window_length=total_epochs,
    )


def build_flat_context_frame(
    feature_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    context_radius: int = 2,
    expected_step_seconds: float = 6.0,
    step_tolerance_seconds: float = 0.25,
) -> pd.DataFrame:
    dataset = build_centered_sequence_dataset(
        feature_table,
        feature_columns=feature_columns,
        target_column=target_column,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=context_radius,
        expected_step_seconds=expected_step_seconds,
        step_tolerance_seconds=step_tolerance_seconds,
    )
    if dataset.X.shape[0] == 0:
        return dataset.metadata.assign(**{target_column: pd.Series(dtype=float)})

    offsets = list(range(-context_radius, context_radius + 1))
    flat_columns: list[str] = []
    flat_blocks: list[np.ndarray] = []
    for offset_idx, offset in enumerate(offsets):
        label = _offset_label(offset)
        flat_blocks.append(dataset.X[:, offset_idx, :])
        flat_columns.extend([f"{label}.{feature}" for feature in dataset.feature_columns])

    flat_values = np.concatenate(flat_blocks, axis=1)
    flat_frame = pd.DataFrame(flat_values, columns=flat_columns)
    flat_frame.insert(0, target_column, dataset.y)
    return pd.concat([dataset.metadata.reset_index(drop=True), flat_frame], axis=1)


def build_mean_context_frame(
    feature_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    context_radius: int = 2,
    expected_step_seconds: float = 6.0,
    step_tolerance_seconds: float = 0.25,
    suffix: str = "__ctxmean",
) -> pd.DataFrame:
    dataset = build_centered_sequence_dataset(
        feature_table,
        feature_columns=feature_columns,
        target_column=target_column,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=context_radius,
        expected_step_seconds=expected_step_seconds,
        step_tolerance_seconds=step_tolerance_seconds,
    )
    if dataset.X.shape[0] == 0:
        out = dataset.metadata.copy()
        out[target_column] = pd.Series(dtype=float)
        return out

    finite_mask = np.isfinite(dataset.X)
    pooled_sum = np.where(finite_mask, dataset.X, 0.0).sum(axis=1)
    pooled_count = finite_mask.sum(axis=1)
    pooled = np.divide(
        pooled_sum,
        pooled_count,
        out=np.full_like(pooled_sum, np.nan, dtype=float),
        where=pooled_count > 0,
    )
    pooled_columns = [f"{feature}{suffix}" for feature in dataset.feature_columns]
    pooled_frame = pd.DataFrame(pooled, columns=pooled_columns)
    pooled_frame.insert(0, target_column, dataset.y)
    return pd.concat([dataset.metadata.reset_index(drop=True), pooled_frame], axis=1)


def build_centered_sequence_dataset(
    feature_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    context_radius: int = 2,
    expected_step_seconds: float = 6.0,
    step_tolerance_seconds: float = 0.25,
) -> SequenceDataset:
    required = [subject_column, time_column, target_column, *feature_columns]
    missing = [col for col in required if col not in feature_table.columns]
    if missing:
        raise KeyError(f"Missing columns for sequence dataset: {missing}")

    window_length = (2 * int(context_radius)) + 1
    if window_length < 1:
        raise ValueError("window_length must be positive")

    frame = feature_table[required].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[subject_column, time_column, target_column])

    seq_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, object]] = []

    for subject_id, subject_df in frame.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        times = ordered[time_column].to_numpy(dtype=float, copy=False)
        y = ordered[target_column].to_numpy(copy=False)
        X = ordered[feature_columns].to_numpy(dtype=float, copy=False)

        if len(ordered) < window_length:
            continue

        for center in range(context_radius, len(ordered) - context_radius):
            start = center - context_radius
            stop = center + context_radius + 1
            seq_times = times[start:stop]
            if not _is_contiguous_sequence(
                seq_times,
                expected_step_seconds=expected_step_seconds,
                step_tolerance_seconds=step_tolerance_seconds,
            ):
                continue
            seq_list.append(X[start:stop])
            y_list.append(y[center])
            meta_rows.append(
                {
                    subject_column: str(subject_id),
                    time_column: float(times[center]),
                    "sequence_start_seconds": float(seq_times[0]),
                    "sequence_end_seconds": float(seq_times[-1]),
                }
            )

    if not seq_list:
        return SequenceDataset(
            X=np.empty((0, window_length, len(feature_columns)), dtype=float),
            y=np.empty((0,), dtype=float),
            metadata=pd.DataFrame(columns=[subject_column, time_column, "sequence_start_seconds", "sequence_end_seconds"]),
            feature_columns=list(feature_columns),
            context_radius=int(context_radius),
            window_length=window_length,
        )

    return SequenceDataset(
        X=np.stack(seq_list, axis=0),
        y=np.asarray(y_list),
        metadata=pd.DataFrame(meta_rows),
        feature_columns=list(feature_columns),
        context_radius=int(context_radius),
        window_length=window_length,
    )


def flatten_sequence_tensor(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D sequence tensor, got shape {X.shape}")
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _is_contiguous_sequence(
    seq_times: np.ndarray,
    *,
    expected_step_seconds: float,
    step_tolerance_seconds: float,
) -> bool:
    if seq_times.size < 2:
        return True
    diffs = np.diff(seq_times.astype(float, copy=False))
    return bool(np.all(np.abs(diffs - expected_step_seconds) <= step_tolerance_seconds))
