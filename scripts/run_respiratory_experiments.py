from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RespiratoryTaskSpec:
    name: str
    target_column: str
    model_family: str = "linear"
    norm_name: str = "robust_by_subject"
    event_only: bool = False


def _prepare_frame(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str,
    event_only: bool,
) -> pd.DataFrame:
    required = [subject_column, target_column, *feature_columns]
    frame = table[required + (["time_seconds"] if "time_seconds" in table.columns else [])].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[target_column, subject_column])
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column]).copy()
    frame[target_column] = frame[target_column].astype(int)
    if event_only:
        frame = frame.loc[frame[target_column] > 0].copy()
    return frame


def _evaluate_task(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str,
    model_family: str,
    norm_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from physionet_sleep.experiments.normalization import apply_normalization
    from physionet_sleep.experiments.tabular_runner import build_model, build_validation_splits, score_predictions

    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[target_column].to_numpy()
    split_iter = list(
        build_validation_splits(
            groups=groups,
            y=y,
            folds=5,
            validation="loso",
            random_state=7,
        )
    )

    fold_rows: list[dict[str, object]] = []
    pred_rows: list[pd.DataFrame] = []
    for fold_index, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_df = frame.iloc[train_idx].copy()
        val_df = frame.iloc[val_idx].copy()
        train_df, val_df, _ = apply_normalization(
            train_df,
            val_df,
            feature_columns=feature_columns,
            norm_name=norm_name,
            group_column=subject_column,
        )
        usable_columns = [col for col in feature_columns if train_df[col].notna().any()]
        if not usable_columns:
            continue
        estimator = build_model(model_family=model_family, task="classification", random_state=7)
        estimator.fit(train_df[usable_columns], train_df[target_column])
        y_pred = estimator.predict(val_df[usable_columns])
        proba = None
        if hasattr(estimator, "predict_proba"):
            try:
                proba = estimator.predict_proba(val_df[usable_columns])
            except Exception:
                proba = None
        metrics = score_predictions(
            y_true=val_df[target_column].to_numpy(),
            y_pred=np.asarray(y_pred),
            proba=proba,
        )
        metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_features_used": int(len(usable_columns)),
                "n_subjects_train": int(train_df[subject_column].nunique()),
                "n_subjects_val": int(val_df[subject_column].nunique()),
            }
        )
        fold_rows.append(metrics)
        pred_rows.append(
            pd.DataFrame(
                {
                    subject_column: val_df[subject_column].to_numpy(),
                    "time_seconds": val_df["time_seconds"].to_numpy() if "time_seconds" in val_df.columns else np.nan,
                    "y_true": val_df[target_column].to_numpy(),
                    "y_pred": np.asarray(y_pred),
                    "fold_index": fold_index,
                }
            )
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True) if pred_rows else pd.DataFrame()
    if fold_df.empty:
        return pd.DataFrame(), fold_df, pred_df

    numeric_cols = [col for col in fold_df.columns if pd.api.types.is_numeric_dtype(fold_df[col]) and col != "fold_index"]
    summary = {
        "target_column": target_column,
        "model_family": model_family,
        "norm_name": norm_name,
        "n_folds": int(len(fold_df)),
    }
    for col in numeric_cols:
        summary[f"{col}_mean"] = float(fold_df[col].mean())
        summary[f"{col}_median"] = float(fold_df[col].median())
    return pd.DataFrame([summary]), fold_df, pred_df


def _run_drop_column(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str,
    model_family: str,
    norm_name: str,
) -> pd.DataFrame:
    baseline_summary, _, _ = _evaluate_task(
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        subject_column=subject_column,
        model_family=model_family,
        norm_name=norm_name,
    )
    if baseline_summary.empty:
        return pd.DataFrame()
    baseline_row = baseline_summary.iloc[0]
    baseline_macro_f1 = float(baseline_row.get("macro_f1_mean", np.nan))
    baseline_bal_acc = float(baseline_row.get("balanced_accuracy_mean", np.nan))
    baseline_kappa = float(baseline_row.get("cohen_kappa_mean", np.nan))
    baseline_auc = float(baseline_row.get("roc_auc_mean", np.nan))

    rows: list[dict[str, object]] = []
    for idx, feature_name in enumerate(feature_columns, start=1):
        print(f"drop_column_start feature={feature_name} index={idx}/{len(feature_columns)} target={target_column}", flush=True)
        reduced = [col for col in feature_columns if col != feature_name]
        summary, _, _ = _evaluate_task(
            frame,
            feature_columns=reduced,
            target_column=target_column,
            subject_column=subject_column,
            model_family=model_family,
            norm_name=norm_name,
        )
        if summary.empty:
            continue
        row = summary.iloc[0]
        macro_f1 = float(row.get("macro_f1_mean", np.nan))
        bal_acc = float(row.get("balanced_accuracy_mean", np.nan))
        kappa = float(row.get("cohen_kappa_mean", np.nan))
        auc = float(row.get("roc_auc_mean", np.nan))
        rows.append(
            {
                "feature_name": feature_name,
                "baseline_macro_f1": baseline_macro_f1,
                "macro_f1_without": macro_f1,
                "delta_macro_f1": baseline_macro_f1 - macro_f1 if np.isfinite(baseline_macro_f1) and np.isfinite(macro_f1) else np.nan,
                "baseline_balanced_accuracy": baseline_bal_acc,
                "balanced_accuracy_without": bal_acc,
                "delta_balanced_accuracy": baseline_bal_acc - bal_acc if np.isfinite(baseline_bal_acc) and np.isfinite(bal_acc) else np.nan,
                "baseline_kappa": baseline_kappa,
                "kappa_without": kappa,
                "delta_kappa": baseline_kappa - kappa if np.isfinite(baseline_kappa) and np.isfinite(kappa) else np.nan,
                "baseline_roc_auc": baseline_auc,
                "roc_auc_without": auc,
                "delta_roc_auc": baseline_auc - auc if np.isfinite(baseline_auc) and np.isfinite(auc) else np.nan,
            }
        )
        print(
            f"drop_column_done feature={feature_name} index={idx}/{len(feature_columns)} "
            f"delta_macro_f1={rows[-1]['delta_macro_f1']}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("delta_macro_f1", ascending=False, na_position="last").reset_index(drop=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        print("no_rows", flush=True)
        return 1

    apnea_features = build_default_experiment_specs(table)["apnea_plus_shared"].feature_columns
    tasks = [
        RespiratoryTaskSpec(name="severity_ge_hypopnea", target_column="label.respiratory_ge_hypopnea"),
        RespiratoryTaskSpec(name="severity_ge_apnea", target_column="label.respiratory_ge_apnea"),
        RespiratoryTaskSpec(name="obstruction", target_column="label.respiratory_obstruction", event_only=True),
    ]

    out_root = repo_root / "artifacts" / "reference_experiments_by_task" / "respiratory"
    out_root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"task_start name={task.name} target={task.target_column}", flush=True)
        frame = _prepare_frame(
            table,
            feature_columns=apnea_features,
            target_column=task.target_column,
            subject_column="subject_id",
            event_only=task.event_only,
        )
        summary, fold_metrics, predictions = _evaluate_task(
            frame,
            feature_columns=apnea_features,
            target_column=task.target_column,
            subject_column="subject_id",
            model_family=task.model_family,
            norm_name=task.norm_name,
        )
        if not summary.empty:
            print(
                f"baseline_done name={task.name} macro_f1={summary.iloc[0].get('macro_f1_mean')} "
                f"balanced_accuracy={summary.iloc[0].get('balanced_accuracy_mean')} roc_auc={summary.iloc[0].get('roc_auc_mean')}",
                flush=True,
            )
        drop_column = _run_drop_column(
            frame,
            feature_columns=apnea_features,
            target_column=task.target_column,
            subject_column="subject_id",
            model_family=task.model_family,
            norm_name=task.norm_name,
        )
        task_dir = out_root / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(task_dir / "summary.parquet", index=False)
        fold_metrics.to_parquet(task_dir / "fold_metrics.parquet", index=False)
        predictions.to_parquet(task_dir / "predictions.parquet", index=False)
        drop_column.to_parquet(task_dir / "drop_column_importance.parquet", index=False)
        metadata = {
            "target_column": task.target_column,
            "event_only": task.event_only,
            "model_family": task.model_family,
            "norm_name": task.norm_name,
            "feature_count": len(apnea_features),
            "row_count": int(len(frame)),
            "class_counts": {str(k): int(v) for k, v in frame[task.target_column].value_counts().sort_index().items()},
        }
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(
            f"task_done name={task.name} rows={len(frame)} macro_f1={summary.iloc[0]['macro_f1_mean'] if not summary.empty else 'nan'}",
            flush=True,
        )

    print(f"saved={out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
