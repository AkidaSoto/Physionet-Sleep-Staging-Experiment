from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from physionet_sleep.experiments.normalization import apply_normalization
from physionet_sleep.experiments.tabular_specs import TabularExperimentSpec


@dataclass(slots=True)
class TabularRunResult:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any]


def run_tabular_experiments(
    feature_table: pd.DataFrame,
    *,
    specs: dict[str, TabularExperimentSpec],
    subject_column: str = "subject_id",
    output_dir: str | Path | None = None,
    validation: str = "kfold",
    random_state: int = 7,
) -> dict[str, TabularRunResult]:
    results: dict[str, TabularRunResult] = {}
    save_root = Path(output_dir) if output_dir is not None else None

    for spec_name, spec in specs.items():
        prepared = _prepare_experiment_frame(feature_table, spec=spec, subject_column=subject_column)
        if prepared.empty:
            continue
        result = _run_single_experiment(
            prepared,
            spec=spec,
            subject_column=subject_column,
            validation=validation,
            random_state=random_state,
        )
        results[spec_name] = result
        if save_root is not None:
            exp_dir = save_root / spec_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            result.summary.to_parquet(exp_dir / "summary.parquet", index=False)
            result.fold_metrics.to_parquet(exp_dir / "fold_metrics.parquet", index=False)
            result.predictions.to_parquet(exp_dir / "predictions.parquet", index=False)
            pd.DataFrame([result.metadata]).to_json(exp_dir / "metadata.json", orient="records", indent=2)
    return results


def _run_single_experiment(
    frame: pd.DataFrame,
    *,
    spec: TabularExperimentSpec,
    subject_column: str,
    validation: str,
    random_state: int,
) -> TabularRunResult:
    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[spec.target_column].to_numpy()

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for norm_name in spec.normalization_candidates:
        for model_family in spec.model_families:
            split_iter = list(
                build_validation_splits(
                    groups=groups,
                    y=y,
                    folds=spec.cv_folds,
                    validation=validation,
                    random_state=random_state,
                )
            )
            fold_metric_rows: list[dict[str, Any]] = []
            family_predictions: list[pd.DataFrame] = []
            for fold_index, (train_idx, val_idx) in enumerate(split_iter, start=1):
                train_df = frame.iloc[train_idx].copy()
                val_df = frame.iloc[val_idx].copy()
                train_df, val_df, norm_stats = apply_normalization(
                    train_df,
                    val_df,
                    feature_columns=spec.feature_columns,
                    norm_name=norm_name,
                    group_column=subject_column,
                )
                usable_columns = [col for col in spec.feature_columns if train_df[col].notna().any()]
                if not usable_columns:
                    continue
                estimator = build_model(model_family=model_family, task=spec.task, random_state=random_state)
                X_train = train_df[usable_columns]
                X_val = val_df[usable_columns]
                y_train = train_df[spec.target_column]
                y_val = val_df[spec.target_column]
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_val)

                proba = None
                if hasattr(estimator, "predict_proba"):
                    try:
                        proba = estimator.predict_proba(X_val)
                    except Exception:
                        proba = None

                metric_row = {
                    "experiment": spec.name,
                    "norm_name": norm_name,
                    "model_family": model_family,
                    "fold_index": fold_index,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_features_used": int(len(usable_columns)),
                    "n_subjects_train": int(train_df[subject_column].nunique()),
                    "n_subjects_val": int(val_df[subject_column].nunique()),
                }
                metric_row.update(score_predictions(y_true=y_val.to_numpy(), y_pred=np.asarray(y_pred), proba=proba))
                fold_metric_rows.append(metric_row | {"norm_scope": norm_stats.get("scope", "")})

                pred_frame = pd.DataFrame(
                    {
                        "experiment": spec.name,
                        "norm_name": norm_name,
                        "model_family": model_family,
                        "fold_index": fold_index,
                        subject_column: val_df[subject_column].to_numpy(),
                        "time_seconds": val_df["time_seconds"].to_numpy() if "time_seconds" in val_df.columns else np.nan,
                        "y_true": y_val.to_numpy(),
                        "y_pred": np.asarray(y_pred),
                    }
                )
                family_predictions.append(pred_frame)

            fold_df = pd.DataFrame(fold_metric_rows)
            family_pred_df = pd.concat(family_predictions, axis=0, ignore_index=True)
            summary_rows.append(
                summarize_fold_metrics(
                    fold_df,
                    experiment_name=spec.name,
                    norm_name=norm_name,
                    model_family=model_family,
                )
            )
            fold_rows.extend(fold_df.to_dict(orient="records"))
            prediction_rows.append(family_pred_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["macro_f1_mean", "cohen_kappa_mean", "balanced_accuracy_mean"],
        ascending=False,
        na_position="last",
    )
    fold_metrics_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_rows, axis=0, ignore_index=True) if prediction_rows else pd.DataFrame()
    metadata = {
        "spec": asdict(spec),
        "validation": validation,
        "subjects": sorted(frame[subject_column].astype(str).unique().tolist()),
        "rows": int(len(frame)),
    }
    return TabularRunResult(
        summary=summary_df,
        fold_metrics=fold_metrics_df,
        predictions=predictions_df,
        metadata=metadata,
    )


def _prepare_experiment_frame(
    feature_table: pd.DataFrame,
    *,
    spec: TabularExperimentSpec,
    subject_column: str,
) -> pd.DataFrame:
    required = [subject_column, spec.target_column, *spec.feature_columns]
    missing = [col for col in required if col not in feature_table.columns]
    if missing:
        return pd.DataFrame()
    frame = feature_table[required + (["time_seconds"] if "time_seconds" in feature_table.columns else [])].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[spec.target_column, subject_column])
    if spec.target_column == "label.stage_seconds":
        frame = frame.loc[pd.to_numeric(frame[spec.target_column], errors="coerce") >= 0].copy()
    return frame


def build_validation_splits(
    *,
    groups: np.ndarray,
    y: np.ndarray,
    folds: int,
    validation: str,
    random_state: int,
):
    unique_groups = np.unique(groups)
    if validation == "loso":
        splitter = LeaveOneGroupOut()
        yield from splitter.split(np.zeros_like(y), y, groups)
        return

    effective_folds = max(2, min(folds, unique_groups.size))
    try:
        splitter = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        yield from splitter.split(np.zeros_like(y), y, groups)
        return
    except Exception:
        splitter = GroupKFold(n_splits=effective_folds)
        yield from splitter.split(np.zeros_like(y), y, groups)


def build_model(
    *,
    model_family: str,
    task: str,
    random_state: int,
    class_weight: str | dict[int, float] | None = None,
) -> Pipeline:
    if task != "classification":
        raise ValueError(f"Unsupported task: {task}")
    if model_family == "linear":
        estimator = LogisticRegression(
            max_iter=1500,
            random_state=random_state,
            solver="saga",
            class_weight=class_weight,
        )
    elif model_family == "tree":
        estimator = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state,
            class_weight=class_weight,
        )
    elif model_family == "ensemble":
        estimator = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_leaf_nodes=21,
                min_samples_leaf=10,
                random_state=random_state,
                class_weight=class_weight,
            ),
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def score_predictions(*, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    unique_classes = np.unique(y_true)
    if unique_classes.size == 2:
        positive = unique_classes.max()
        binary_true = (y_true == positive).astype(int)
        binary_pred = (y_pred == positive).astype(int)
        metrics.update(
            {
                "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
                "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
                "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
                "specificity": float(_specificity(binary_true, binary_pred)),
            }
        )
        if proba is not None and getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(binary_true, proba[:, -1]))
            except Exception:
                metrics["roc_auc"] = float("nan")
    return metrics


def summarize_fold_metrics(
    fold_df: pd.DataFrame,
    *,
    experiment_name: str,
    norm_name: str,
    model_family: str,
) -> dict[str, Any]:
    metric_cols = [col for col in fold_df.columns if col not in {"experiment", "norm_name", "norm_scope", "model_family", "fold_index"} and pd.api.types.is_numeric_dtype(fold_df[col])]
    row: dict[str, Any] = {
        "experiment": experiment_name,
        "norm_name": norm_name,
        "model_family": model_family,
        "n_folds": int(len(fold_df)),
    }
    for col in metric_cols:
        row[f"{col}_mean"] = float(fold_df[col].mean())
        row[f"{col}_std"] = float(fold_df[col].std(ddof=0))
    return row


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")
