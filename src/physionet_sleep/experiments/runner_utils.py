from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physionet_sleep.experiments.normalization import apply_normalization


def build_split_tag(
    *,
    outer_validation: str,
    outer_folds: int,
    inner_validation: str | None = None,
) -> str:
    outer_part = f"{outer_validation}{outer_folds}" if outer_validation == "kfold" else outer_validation
    if inner_validation is None:
        return outer_part
    return f"{outer_part}__{inner_validation}"


def load_selected_norm_map(path: Path, *, suffix: str = "") -> dict[str, str]:
    from physionet_sleep.experiments.tabular_specs import FEATURE_NORM_OVERRIDES_BY_PREFIX

    frame = pd.read_parquet(path)
    norm_map = {} if frame.empty else {
        f"{str(row['feature_name'])}{suffix}": str(row["selected_norm_name"])
        for _, row in frame.iterrows()
        if pd.notna(row.get("feature_name")) and pd.notna(row.get("selected_norm_name"))
    }
    return _NormMapWithOverrides(norm_map, FEATURE_NORM_OVERRIDES_BY_PREFIX)


class _NormMapWithOverrides(dict):
    """Selected norms, with prefix overrides taking precedence (also for features absent from the selection file)."""

    def __init__(self, base: dict[str, str], overrides: dict[str, str]) -> None:
        super().__init__(base)
        self.overrides = dict(overrides)

    def get(self, key, default=None):
        for prefix, norm in self.overrides.items():
            if str(key).startswith(prefix):
                return norm
        return super().get(key, default)

    def __getitem__(self, key):
        for prefix, norm in self.overrides.items():
            if str(key).startswith(prefix):
                return norm
        return super().__getitem__(key)


def apply_featurewise_norm_map(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_map: dict[str, str],
    subject_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    grouped: dict[str, list[str]] = {}
    for feature_name in feature_columns:
        norm_name = norm_map.get(feature_name, "robust_by_subject")
        grouped.setdefault(norm_name, []).append(feature_name)
    for norm_name, cols in grouped.items():
        train, test, _ = apply_normalization(
            train,
            test,
            feature_columns=cols,
            norm_name=norm_name,
            group_column=subject_column,
        )
    return train, test


def prepare_model_frame(
    table: pd.DataFrame,
    *,
    subject_column: str,
    target_column: str,
    feature_columns: list[str],
    time_column: str = "time_seconds",
    drop_negative_targets: bool = False,
) -> pd.DataFrame:
    required = [subject_column, target_column, *feature_columns]
    extra_columns = [time_column] if time_column in table.columns else []
    frame = table[required + extra_columns].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[subject_column, target_column]).copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column]).copy()
    frame[target_column] = frame[target_column].astype(int)
    if drop_negative_targets:
        frame = frame.loc[frame[target_column] >= 0].copy()
    return frame


def ordered_prediction_sequences(
    pred_frame: pd.DataFrame,
    *,
    subject_column: str,
    time_column: str = "time_seconds",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_by_subject: list[np.ndarray] = []
    pred_by_subject: list[np.ndarray] = []
    for _, group in pred_frame.groupby(subject_column, sort=False):
        ordered = group.sort_values(time_column, kind="stable") if time_column in group.columns else group
        true_by_subject.append(ordered["y_true"].to_numpy())
        pred_by_subject.append(ordered["y_pred"].to_numpy())
    return true_by_subject, pred_by_subject


def summarize_numeric_folds(fold_df: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    numeric_cols = [col for col in fold_df.columns if pd.api.types.is_numeric_dtype(fold_df[col]) and col != "fold_index"]
    row: dict[str, object] = {
        "eval_source": eval_source,
        "n_folds": int(len(fold_df)),
    }
    for col in numeric_cols:
        row[f"{col}_mean"] = float(fold_df[col].mean())
        row[f"{col}_median"] = float(fold_df[col].median())
    return pd.DataFrame([row])


def align_probability_matrix(
    proba: np.ndarray | None,
    *,
    estimator_classes: np.ndarray,
    class_labels: list[int],
) -> np.ndarray:
    n_rows = int(proba.shape[0]) if proba is not None else 0
    aligned = np.zeros((n_rows, len(class_labels)), dtype=float)
    if proba is None:
        return aligned
    class_to_idx = {int(value): idx for idx, value in enumerate(class_labels)}
    for src_idx, label in enumerate(estimator_classes.tolist()):
        dst_idx = class_to_idx.get(int(label))
        if dst_idx is not None:
            aligned[:, dst_idx] = proba[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    nonzero = row_sum.squeeze(1) > 0
    aligned[nonzero] = aligned[nonzero] / row_sum[nonzero]
    return aligned


def attach_aligned_probabilities(
    frame: pd.DataFrame,
    *,
    class_labels: list[int],
    aligned_proba: np.ndarray,
    prefix: str = "proba_",
) -> pd.DataFrame:
    out = frame.copy()
    for idx, label in enumerate(class_labels):
        out[f"{prefix}{label}"] = aligned_proba[:, idx]
    return out


def offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def safe_logit(value: float, *, eps: float = 1.0e-4) -> float:
    clipped = float(np.clip(value, eps, 1.0 - eps))
    return float(np.log(clipped / (1.0 - clipped)))


def confidence_summary(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    probs = np.asarray(values, dtype=float)
    top_sorted = np.sort(probs)[::-1]
    top1 = float(top_sorted[0])
    top2 = float(top_sorted[1]) if top_sorted.size > 1 else 0.0
    margin = top1 - top2
    entropy = float(-(probs * np.log(np.clip(probs, 1.0e-12, 1.0))).sum())
    return top1, margin, entropy


def build_probability_context_frame(
    pred_frame: pd.DataFrame,
    *,
    class_labels: list[int],
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    context_radius: int = 2,
    expected_step_seconds: float = 30.0,
    step_tolerance_seconds: float = 0.25,
    include_probability_logits: bool = False,
    include_confidence: bool = False,
    include_deltas: bool = False,
    passthrough_prefixes: tuple[str, ...] = (),
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    proba_cols = [f"proba_{label}" for label in class_labels]
    passthrough_cols = (
        sorted(
            [col for col in pred_frame.columns if col.startswith(passthrough_prefixes)],
            key=lambda value: (value.split("_", 1)[0], value),
        )
        if passthrough_prefixes
        else []
    )
    window_length = (2 * int(context_radius)) + 1

    for subject_id, subject_df in pred_frame.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        if len(ordered) < window_length:
            continue
        times = ordered[time_column].to_numpy(dtype=float, copy=False)
        for center in range(context_radius, len(ordered) - context_radius):
            start = center - context_radius
            stop = center + context_radius + 1
            seq_times = times[start:stop]
            diffs = np.diff(seq_times)
            if diffs.size and not np.all(np.abs(diffs - expected_step_seconds) <= step_tolerance_seconds):
                continue
            block = ordered.iloc[start:stop]
            row: dict[str, object] = {
                subject_column: str(subject_id),
                time_column: float(times[center]),
                "y_true": int(ordered["y_true"].iloc[center]),
                "y_pred_stage1": int(ordered["y_pred"].iloc[center]),
            }
            center_values: dict[int, float] = {}
            prev_values: dict[int, float] = {}
            next_values: dict[int, float] = {}
            for offset in range(-context_radius, context_radius + 1):
                offset_tag = offset_label(offset)
                source = block.iloc[offset + context_radius]
                values = np.asarray([float(source[f"proba_{label}"]) for label in class_labels], dtype=float)
                for proba_col in proba_cols:
                    value = float(source[proba_col])
                    row[f"{offset_tag}.{proba_col}"] = value
                    if include_probability_logits:
                        row[f"{offset_tag}.{proba_col}_logit"] = safe_logit(value)
                    label = int(proba_col.removeprefix("proba_"))
                    if offset == 0:
                        center_values[label] = value
                    elif offset == -1:
                        prev_values[label] = value
                    elif offset == 1:
                        next_values[label] = value
                for passthrough_col in passthrough_cols:
                    row[f"{offset_tag}.{passthrough_col}"] = float(source[passthrough_col])
                if include_confidence:
                    top1, margin, entropy = confidence_summary(values)
                    row[f"{offset_tag}.top1_proba"] = top1
                    row[f"{offset_tag}.margin"] = margin
                    row[f"{offset_tag}.entropy"] = entropy
            if include_deltas:
                for label in class_labels:
                    center_value = center_values.get(int(label), 0.0)
                    row[f"delta_prev.proba_{label}"] = center_value - prev_values.get(int(label), center_value)
                    row[f"delta_next.proba_{label}"] = next_values.get(int(label), center_value) - center_value
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def apply_prediction_updates(
    raw_pred_frame: pd.DataFrame,
    update_frame: pd.DataFrame,
    *,
    y_pred_values: np.ndarray,
    subject_column: str,
    time_column: str,
    stage1_column: str = "y_pred_stage1",
    variant_name: str | None = None,
    variant_column: str = "variant",
) -> pd.DataFrame:
    out = raw_pred_frame.copy()
    out[stage1_column] = out["y_pred"].to_numpy()
    updates = update_frame[[subject_column, time_column]].copy()
    updates["y_pred"] = np.asarray(y_pred_values)
    if variant_name is not None:
        updates[variant_column] = variant_name
    suffix = "__updated"
    out = out.merge(
        updates,
        on=[subject_column, time_column],
        how="left",
        suffixes=("", suffix),
    )
    updated_col = f"y_pred{suffix}"
    mask = out[updated_col].notna()
    out.loc[mask, "y_pred"] = out.loc[mask, updated_col].astype(int)
    out = out.drop(columns=[updated_col])
    if variant_name is not None:
        out[variant_column] = variant_name
    return out
