from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _log(message: str, *, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_norm_map(path: Path) -> dict[str, str]:
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    return {
        str(row["feature_name"]): str(row["selected_norm_name"])
        for _, row in frame.iterrows()
        if pd.notna(row.get("feature_name")) and pd.notna(row.get("selected_norm_name"))
    }


def _apply_featurewise_best_norm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_map: dict[str, str],
    subject_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from physionet_sleep.experiments.normalization import apply_normalization

    train = train_df.copy()
    val = val_df.copy()
    grouped: dict[str, list[str]] = {}
    for feature_name in feature_columns:
        norm_name = norm_map.get(feature_name, "robust_by_subject")
        grouped.setdefault(norm_name, []).append(feature_name)
    for norm_name, cols in grouped.items():
        train, val, _ = apply_normalization(
            train,
            val,
            feature_columns=cols,
            norm_name=norm_name,
            group_column=subject_column,
        )
    return train, val


def _prepare_frame(
    table: pd.DataFrame,
    *,
    subject_column: str,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    required = [subject_column, target_column, *feature_columns]
    frame = table[required + (["time_seconds"] if "time_seconds" in table.columns else [])].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[target_column, subject_column]).copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column]).copy()
    frame[target_column] = frame[target_column].astype(int)
    if target_column == "label.stage_seconds":
        frame = frame.loc[frame[target_column] >= 0].copy()
    return frame


def _sequence_arrays(pred_frame: pd.DataFrame, *, subject_column: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true_by_subject: list[np.ndarray] = []
    pred_by_subject: list[np.ndarray] = []
    for _, group in pred_frame.groupby(subject_column, sort=False):
        ordered = group.sort_values("time_seconds", kind="stable") if "time_seconds" in group.columns else group
        true_by_subject.append(ordered["y_true"].to_numpy())
        pred_by_subject.append(ordered["y_pred"].to_numpy())
    return true_by_subject, pred_by_subject


def _summarize_fold_metrics(fold_df: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
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


def _run_task(
    frame: pd.DataFrame,
    *,
    task_name: str,
    subject_column: str,
    target_column: str,
    feature_columns: list[str],
    norm_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from physionet_sleep.decoders.hsmm import apply_hsmm_viterbi, learn_hsmm_params
    from physionet_sleep.experiments.tabular_runner import build_model, build_validation_splits, score_predictions

    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[target_column].to_numpy()
    splits = list(build_validation_splits(groups=groups, y=y, folds=5, validation="kfold", random_state=7))

    raw_folds: list[dict[str, object]] = []
    hsmm_folds: list[dict[str, object]] = []
    raw_preds: list[pd.DataFrame] = []
    hsmm_preds: list[pd.DataFrame] = []

    for fold_index, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = frame.iloc[train_idx].copy()
        val_df = frame.iloc[val_idx].copy()
        train_df, val_df = _apply_featurewise_best_norm(
            train_df,
            val_df,
            feature_columns=feature_columns,
            norm_map=norm_map,
            subject_column=subject_column,
        )
        usable = [col for col in feature_columns if train_df[col].notna().any()]
        estimator = build_model(model_family="ensemble", task="classification", random_state=7)
        estimator.fit(train_df[usable], train_df[target_column])

        train_pred = np.asarray(estimator.predict(train_df[usable]))
        val_pred = np.asarray(estimator.predict(val_df[usable]))
        val_proba = None
        if hasattr(estimator, "predict_proba"):
            try:
                val_proba = estimator.predict_proba(val_df[usable])
            except Exception:
                val_proba = None

        raw_metrics = score_predictions(
            y_true=val_df[target_column].to_numpy(),
            y_pred=val_pred,
            proba=val_proba,
        )
        raw_metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_features_used": int(len(usable)),
                "n_subjects_train": int(train_df[subject_column].nunique()),
                "n_subjects_val": int(val_df[subject_column].nunique()),
            }
        )
        raw_folds.append(raw_metrics)

        raw_pred_frame = val_df[[subject_column] + (["time_seconds"] if "time_seconds" in val_df.columns else [])].copy()
        raw_pred_frame["y_true"] = val_df[target_column].to_numpy()
        raw_pred_frame["y_pred"] = val_pred
        raw_pred_frame["fold_index"] = fold_index
        raw_preds.append(raw_pred_frame)

        train_seq_true, train_seq_pred = _sequence_arrays(
            train_df.assign(y_true=train_df[target_column].to_numpy(), y_pred=train_pred),
            subject_column=subject_column,
        )
        val_seq_true, val_seq_pred = _sequence_arrays(
            raw_pred_frame,
            subject_column=subject_column,
        )
        hsmm_params = learn_hsmm_params(train_seq_true, train_seq_pred)
        val_seq_hsmm = apply_hsmm_viterbi(val_seq_pred, hsmm_params)
        hsmm_flat = np.concatenate([np.asarray(seq) for seq in val_seq_hsmm if len(seq)], axis=0) if any(len(seq) for seq in val_seq_hsmm) else np.asarray([])
        true_flat = np.concatenate([np.asarray(seq) for seq in val_seq_true if len(seq)], axis=0) if any(len(seq) for seq in val_seq_true) else np.asarray([])
        hsmm_metrics = score_predictions(
            y_true=true_flat,
            y_pred=hsmm_flat,
            proba=None,
        )
        hsmm_metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_features_used": int(len(usable)),
                "n_subjects_train": int(train_df[subject_column].nunique()),
                "n_subjects_val": int(val_df[subject_column].nunique()),
                "dur_d_max": int(hsmm_params.dur_d_max),
            }
        )
        hsmm_folds.append(hsmm_metrics)

        hsmm_pred_frame = raw_pred_frame.copy()
        if hsmm_flat.size == len(hsmm_pred_frame):
            hsmm_pred_frame["y_pred"] = hsmm_flat
        else:
            rebuilt: list[pd.DataFrame] = []
            cursor = 0
            for _, group in raw_pred_frame.groupby(subject_column, sort=False):
                ordered = group.sort_values("time_seconds", kind="stable") if "time_seconds" in group.columns else group
                n = len(ordered)
                seq = np.asarray(val_seq_hsmm[len(rebuilt)])
                tmp = ordered.copy()
                tmp["y_pred"] = seq[:n]
                rebuilt.append(tmp)
                cursor += n
            hsmm_pred_frame = pd.concat(rebuilt, axis=0, ignore_index=True)
        hsmm_pred_frame["fold_index"] = fold_index
        hsmm_preds.append(hsmm_pred_frame)

        print(
            f"{task_name} fold={fold_index}/{len(splits)} raw_macro_f1={raw_metrics.get('macro_f1')} "
            f"hsmm_macro_f1={hsmm_metrics.get('macro_f1')}",
            flush=True,
        )

    raw_fold_df = pd.DataFrame(raw_folds)
    hsmm_fold_df = pd.DataFrame(hsmm_folds)
    raw_summary = _summarize_fold_metrics(raw_fold_df, eval_source="raw")
    hsmm_summary = _summarize_fold_metrics(hsmm_fold_df, eval_source="hsmm")
    raw_pred_df = pd.concat(raw_preds, axis=0, ignore_index=True) if raw_preds else pd.DataFrame()
    hsmm_pred_df = pd.concat(hsmm_preds, axis=0, ignore_index=True) if hsmm_preds else pd.DataFrame()
    return raw_summary, hsmm_summary, raw_fold_df, hsmm_fold_df, raw_pred_df, hsmm_pred_df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    log_dir = repo_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "full_ensemble_5fold_hsmm.log"
    status_path = log_dir / "full_ensemble_5fold_hsmm_status.json"

    _log("run_start", log_path=log_path)
    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        status_path.write_text(json.dumps({"status": "failed", "reason": "no_rows"}, indent=2), encoding="utf-8")
        _log("no_rows", log_path=log_path)
        return 1

    specs = build_default_experiment_specs(table)
    tasks = {
        "staging_plus_shared": {
            "target_column": specs["staging_plus_shared"].target_column,
            "feature_columns": specs["staging_plus_shared"].feature_columns,
            "norm_map": _load_norm_map(repo_root / "artifacts" / "reference_diagnostics_by_task" / "staging" / "best_feature_norms.parquet"),
        },
        "apnea_plus_shared": {
            "target_column": specs["apnea_plus_shared"].target_column,
            "feature_columns": specs["apnea_plus_shared"].feature_columns,
            "norm_map": _load_norm_map(repo_root / "artifacts" / "reference_diagnostics_by_task" / "apnea" / "best_feature_norms.parquet"),
        },
    }

    out_root = repo_root / "artifacts" / "reference_experiments_full_ensemble_5fold_hsmm"
    out_root.mkdir(parents=True, exist_ok=True)
    overall_status: dict[str, object] = {"status": "running", "tasks": {}}
    status_path.write_text(json.dumps(overall_status, indent=2), encoding="utf-8")

    for task_name, cfg in tasks.items():
        _log(
            f"task_start name={task_name} target={cfg['target_column']} features={len(cfg['feature_columns'])}",
            log_path=log_path,
        )
        frame = _prepare_frame(
            table,
            subject_column="subject_id",
            target_column=str(cfg["target_column"]),
            feature_columns=list(cfg["feature_columns"]),
        )
        raw_summary, hsmm_summary, raw_folds, hsmm_folds, raw_preds, hsmm_preds = _run_task(
            frame,
            task_name=task_name,
            subject_column="subject_id",
            target_column=str(cfg["target_column"]),
            feature_columns=list(cfg["feature_columns"]),
            norm_map=dict(cfg["norm_map"]),
        )
        task_dir = out_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        raw_summary.to_parquet(task_dir / "summary_raw.parquet", index=False)
        hsmm_summary.to_parquet(task_dir / "summary_hsmm.parquet", index=False)
        raw_folds.to_parquet(task_dir / "fold_metrics_raw.parquet", index=False)
        hsmm_folds.to_parquet(task_dir / "fold_metrics_hsmm.parquet", index=False)
        raw_preds.to_parquet(task_dir / "predictions_raw.parquet", index=False)
        hsmm_preds.to_parquet(task_dir / "predictions_hsmm.parquet", index=False)
        meta = {
            "target_column": cfg["target_column"],
            "feature_count": len(cfg["feature_columns"]),
            "row_count": int(len(frame)),
            "n_subjects": int(frame["subject_id"].nunique()),
            "folds": 5,
            "validation": "grouped_kfold",
            "model_family": "ensemble",
            "sequence_postprocessing": "hsmm",
            "norm_source": "best_feature_norms",
        }
        (task_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        overall_status["tasks"][task_name] = {
            "raw_summary": raw_summary.iloc[0].to_dict() if not raw_summary.empty else {},
            "hsmm_summary": hsmm_summary.iloc[0].to_dict() if not hsmm_summary.empty else {},
        }
        status_path.write_text(json.dumps(overall_status, indent=2, default=str), encoding="utf-8")
        _log(
            f"task_done name={task_name} raw_macro_f1={raw_summary.iloc[0].get('macro_f1_mean') if not raw_summary.empty else 'nan'} "
            f"hsmm_macro_f1={hsmm_summary.iloc[0].get('macro_f1_mean') if not hsmm_summary.empty else 'nan'}",
            log_path=log_path,
        )

    overall_status["status"] = "completed"
    overall_status["output_dir"] = str(out_root)
    status_path.write_text(json.dumps(overall_status, indent=2, default=str), encoding="utf-8")
    _log(f"run_done output_dir={out_root}", log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
