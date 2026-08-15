from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class NormalizationSpec:
    name: str
    enabled: bool
    scope: str
    center: str
    scale: str


_NORMALIZATION_ALIASES = {
    "raw": "none",
    "none": "none",
    "off": "none",
    "zscore": "standard_global",
    "standard": "standard_global",
    "standard_global": "standard_global",
    "medianiqr": "robust_by_subject",
    "robust_by_subject": "robust_by_subject",
    "robust_global": "robust_global",
}


def resolve_normalization(name: str) -> NormalizationSpec:
    canonical = _NORMALIZATION_ALIASES.get(name.lower(), name.lower())
    if canonical == "none":
        return NormalizationSpec(name="none", enabled=False, scope="global", center="none", scale="none")
    if canonical == "standard_global":
        return NormalizationSpec(name="standard_global", enabled=True, scope="global", center="mean", scale="std")
    if canonical == "robust_global":
        return NormalizationSpec(name="robust_global", enabled=True, scope="global", center="median", scale="iqr")
    if canonical == "robust_by_subject":
        return NormalizationSpec(name="robust_by_subject", enabled=True, scope="by_subject", center="median", scale="iqr")
    raise ValueError(f"Unsupported normalization: {name}")


def apply_normalization(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_name: str,
    group_column: str = "subject_id",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec = resolve_normalization(norm_name)
    train = train_df.copy()
    val = val_df.copy()
    if not spec.enabled or not feature_columns:
        return train, val, {"name": spec.name, "scope": spec.scope, "center": [], "scale": []}

    numeric_cols = [col for col in feature_columns if pd.api.types.is_numeric_dtype(train[col])]
    if not numeric_cols:
        return train, val, {"name": spec.name, "scope": spec.scope, "center": [], "scale": []}

    if spec.scope == "global":
        center_v, scale_v = _compute_matrix_stats(train[numeric_cols], center=spec.center, scale=spec.scale)
        train.loc[:, numeric_cols] = _apply_matrix_stats(train[numeric_cols], center_v, scale_v)
        val.loc[:, numeric_cols] = _apply_matrix_stats(val[numeric_cols], center_v, scale_v)
        return train, val, {
            "name": spec.name,
            "scope": spec.scope,
            "columns": numeric_cols,
            "center": center_v.tolist(),
            "scale": scale_v.tolist(),
        }

    if spec.scope == "by_subject":
        train = _normalize_by_group(train, numeric_cols, group_column=group_column, center=spec.center, scale=spec.scale)
        val = _normalize_by_group(val, numeric_cols, group_column=group_column, center=spec.center, scale=spec.scale)
        return train, val, {
            "name": spec.name,
            "scope": spec.scope,
            "columns": numeric_cols,
            "group_column": group_column,
        }

    raise ValueError(f"Unsupported normalization scope: {spec.scope}")


def apply_normalization_single(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_name: str,
    group_column: str = "subject_id",
) -> pd.DataFrame:
    empty = df.iloc[0:0].copy()
    normalized, _, _ = apply_normalization(
        df,
        empty,
        feature_columns=feature_columns,
        norm_name=norm_name,
        group_column=group_column,
    )
    return normalized


def _normalize_by_group(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    group_column: str,
    center: str,
    scale: str,
) -> pd.DataFrame:
    out = df.copy()
    for _, idx in out.groupby(group_column, sort=False).groups.items():
        chunk = out.loc[idx, feature_columns]
        center_v, scale_v = _compute_matrix_stats(chunk, center=center, scale=scale)
        out.loc[idx, feature_columns] = _apply_matrix_stats(chunk, center_v, scale_v)
    return out


def _compute_matrix_stats(frame: pd.DataFrame, *, center: str, scale: str) -> tuple[np.ndarray, np.ndarray]:
    arr = frame.to_numpy(dtype=float, copy=False)
    center_v = np.zeros(arr.shape[1], dtype=float)
    scale_v = np.ones(arr.shape[1], dtype=float)
    for col_idx in range(arr.shape[1]):
        column = arr[:, col_idx]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        if center == "mean":
            center_v[col_idx] = float(np.mean(finite))
        elif center == "median":
            center_v[col_idx] = float(np.median(finite))
        if scale == "std":
            scale_v[col_idx] = float(np.std(finite, ddof=0))
        elif scale == "iqr":
            q75, q25 = np.percentile(finite, [75, 25])
            scale_v[col_idx] = float(q75 - q25)
        if not np.isfinite(scale_v[col_idx]) or scale_v[col_idx] == 0:
            scale_v[col_idx] = 1.0
    return center_v, scale_v


def _apply_matrix_stats(frame: pd.DataFrame, center_v: np.ndarray, scale_v: np.ndarray) -> pd.DataFrame:
    arr = frame.to_numpy(dtype=float, copy=True)
    safe_scale = np.where(np.isfinite(scale_v) & (scale_v != 0), scale_v, 1.0)
    arr = (arr - center_v.reshape(1, -1)) / safe_scale.reshape(1, -1)
    return pd.DataFrame(arr, index=frame.index, columns=frame.columns)
