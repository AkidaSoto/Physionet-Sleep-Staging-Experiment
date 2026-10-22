from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.experiments.runner_utils import (
    apply_featurewise_norm_map,
    build_split_tag,
    load_selected_norm_map,
    ordered_prediction_sequences,
    prepare_model_frame,
    summarize_numeric_folds,
)


def _log(message: str, *, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_norm_map(path: Path) -> dict[str, str]:
    return load_selected_norm_map(path)


def _apply_featurewise_best_norm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_map: dict[str, str],
    subject_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_featurewise_norm_map(
        train_df,
        val_df,
        feature_columns=feature_columns,
        norm_map=norm_map,
        subject_column=subject_column,
    )


def _prepare_frame(
    table: pd.DataFrame,
    *,
    subject_column: str,
    target_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    return prepare_model_frame(
        table,
        subject_column=subject_column,
        target_column=target_column,
        feature_columns=feature_columns,
        drop_negative_targets=(target_column == "label.stage_30s"),
    )


def _sequence_arrays(pred_frame: pd.DataFrame, *, subject_column: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return ordered_prediction_sequences(pred_frame, subject_column=subject_column)


def _summarize_fold_metrics(fold_df: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    return summarize_numeric_folds(fold_df, eval_source=eval_source)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-validation", choices=("none", "single_subject"), default="none")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("staging_30s", "apnea_11s_anchor1_drop4"),
        default=("staging_30s", "apnea_11s_anchor1_drop4"),
    )
    return parser.parse_args()


def _split_tag(*, outer_validation: str, outer_folds: int, inner_validation: str) -> str:
    return build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
    )


def _run_task(
    frame: pd.DataFrame,
    *,
    task_name: str,
    subject_column: str,
    target_column: str,
    feature_columns: list[str],
    norm_map: dict[str, str],
    use_hsmm: bool,
    outer_validation: str,
    outer_folds: int,
    inner_validation: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from physionet_sleep.decoders.hsmm import apply_hsmm_viterbi, learn_hsmm_params
    from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits, score_predictions

    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[target_column].to_numpy()
    splits = list(
        build_subject_eval_splits(
            groups=groups,
            y=y,
            outer_validation=outer_validation,
            outer_folds=outer_folds,
            inner_validation=inner_validation,
            random_state=random_state,
        )
    )

    raw_folds: list[dict[str, object]] = []
    hsmm_folds: list[dict[str, object]] = []
    raw_preds: list[pd.DataFrame] = []
    hsmm_preds: list[pd.DataFrame] = []

    for split in splits:
        fold_index = split.fold_index
        train_df = frame.iloc[split.train_idx].copy()
        test_df = frame.iloc[split.test_idx].copy()
        calib_df = frame.iloc[split.validation_idx].copy() if split.validation_idx is not None else None
        train_df, test_df = _apply_featurewise_best_norm(
            train_df,
            test_df,
            feature_columns=feature_columns,
            norm_map=norm_map,
            subject_column=subject_column,
        )
        if calib_df is not None:
            train_df, calib_df = _apply_featurewise_best_norm(
                train_df,
                calib_df,
                feature_columns=feature_columns,
                norm_map=norm_map,
                subject_column=subject_column,
            )
        usable = [col for col in feature_columns if train_df[col].notna().any()]
        estimator = build_model(
            model_family="ensemble",
            task="classification",
            random_state=random_state,
            class_weight="balanced",
        )
        estimator.fit(train_df[usable], train_df[target_column])

        train_pred = np.asarray(estimator.predict(train_df[usable]))
        test_pred = np.asarray(estimator.predict(test_df[usable]))
        test_proba = None
        if hasattr(estimator, "predict_proba"):
            try:
                test_proba = estimator.predict_proba(test_df[usable])
            except Exception:
                test_proba = None

        raw_metrics = score_predictions(
            y_true=test_df[target_column].to_numpy(),
            y_pred=test_pred,
            proba=test_proba,
        )
        raw_metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_features_used": int(len(usable)),
                "n_subjects_train": int(train_df[subject_column].nunique()),
                "n_subjects_test": int(test_df[subject_column].nunique()),
                "n_subjects_validation": int(calib_df[subject_column].nunique()) if calib_df is not None else 0,
            }
        )
        raw_folds.append(raw_metrics)

        raw_pred_frame = test_df[[subject_column] + (["time_seconds"] if "time_seconds" in test_df.columns else [])].copy()
        raw_pred_frame["y_true"] = test_df[target_column].to_numpy()
        raw_pred_frame["y_pred"] = test_pred
        raw_pred_frame["fold_index"] = fold_index
        raw_preds.append(raw_pred_frame)

        if use_hsmm:
            calib_source = train_df
            calib_pred = train_pred
            if calib_df is not None:
                calib_source = calib_df
                calib_pred = np.asarray(estimator.predict(calib_df[usable]))
            calib_seq_true, calib_seq_pred = _sequence_arrays(
                calib_source.assign(y_true=calib_source[target_column].to_numpy(), y_pred=calib_pred),
                subject_column=subject_column,
            )
            test_seq_true, test_seq_pred = _sequence_arrays(
                raw_pred_frame,
                subject_column=subject_column,
            )
            hsmm_params = learn_hsmm_params(calib_seq_true, calib_seq_pred)
            test_seq_hsmm = apply_hsmm_viterbi(test_seq_pred, hsmm_params)
            hsmm_flat = np.concatenate([np.asarray(seq) for seq in test_seq_hsmm if len(seq)], axis=0) if any(len(seq) for seq in test_seq_hsmm) else np.asarray([])
            true_flat = np.concatenate([np.asarray(seq) for seq in test_seq_true if len(seq)], axis=0) if any(len(seq) for seq in test_seq_true) else np.asarray([])
            hsmm_metrics = score_predictions(
                y_true=true_flat,
                y_pred=hsmm_flat,
                proba=None,
            )
            hsmm_metrics.update(
                {
                    "fold_index": fold_index,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_features_used": int(len(usable)),
                    "n_subjects_train": int(train_df[subject_column].nunique()),
                    "n_subjects_test": int(test_df[subject_column].nunique()),
                    "n_subjects_validation": int(calib_df[subject_column].nunique()) if calib_df is not None else 0,
                    "dur_d_max": int(hsmm_params.dur_d_max),
                }
            )
            hsmm_folds.append(hsmm_metrics)

            hsmm_pred_frame = raw_pred_frame.copy()
            rebuilt: list[pd.DataFrame] = []
            for seq_index, (_, group) in enumerate(raw_pred_frame.groupby(subject_column, sort=False)):
                ordered = group.sort_values("time_seconds", kind="stable") if "time_seconds" in group.columns else group
                tmp = ordered.copy()
                tmp["y_pred"] = np.asarray(test_seq_hsmm[seq_index])[: len(ordered)]
                rebuilt.append(tmp)
            hsmm_pred_frame = pd.concat(rebuilt, axis=0, ignore_index=True)
            hsmm_pred_frame["fold_index"] = fold_index
            hsmm_preds.append(hsmm_pred_frame)
        else:
            hsmm_metrics = {}

        print(
            f"{task_name} fold={fold_index}/{len(splits)} raw_macro_f1={raw_metrics.get('macro_f1')} "
            f"{'hsmm_macro_f1=' + str(hsmm_metrics.get('macro_f1')) if use_hsmm else 'hsmm=off'}",
            flush=True,
        )

    raw_fold_df = pd.DataFrame(raw_folds)
    hsmm_fold_df = pd.DataFrame(hsmm_folds)
    raw_summary = _summarize_fold_metrics(raw_fold_df, eval_source="raw")
    hsmm_summary = _summarize_fold_metrics(hsmm_fold_df, eval_source="hsmm") if use_hsmm else pd.DataFrame()
    raw_pred_df = pd.concat(raw_preds, axis=0, ignore_index=True) if raw_preds else pd.DataFrame()
    hsmm_pred_df = pd.concat(hsmm_preds, axis=0, ignore_index=True) if hsmm_preds else pd.DataFrame()
    return raw_summary, hsmm_summary, raw_fold_df, hsmm_fold_df, raw_pred_df, hsmm_pred_df


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import (
        PAPER_APNEA_DROPPED_SUBJECTS,
        build_apnea_benchmark_table,
        build_staging_benchmark_table,
        load_ucddb_reference_dataset,
    )
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    log_dir = repo_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    split_tag = _split_tag(
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
    )
    log_path = log_dir / f"paper_matched_benchmarks_hsmm__{split_tag}.log"
    status_path = log_dir / f"paper_matched_benchmarks_hsmm__{split_tag}_status.json"

    _log("run_start", log_path=log_path)
    staging_source = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    apnea_source = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
        feature_table_name="final__feature_table_apnea_1s",
    )
    if staging_source.empty or apnea_source.empty:
        status_path.write_text(json.dumps({"status": "failed", "reason": "no_rows"}, indent=2), encoding="utf-8")
        _log("no_rows", log_path=log_path)
        return 1

    staging_specs = build_default_experiment_specs(staging_source)
    apnea_specs = build_default_experiment_specs(apnea_source)
    staging_table = build_staging_benchmark_table(
        staging_source,
        feature_columns=staging_specs["staging_plus_shared"].feature_columns,
    )
    apnea_table = build_apnea_benchmark_table(
        apnea_source,
        feature_columns=apnea_specs["apnea_plus_shared"].feature_columns,
        dropped_subject_ids=PAPER_APNEA_DROPPED_SUBJECTS,
    )

    tasks = {
        "staging_30s": {
            "table": staging_table,
            "target_column": "label.stage_30s",
            "feature_columns": staging_specs["staging_plus_shared"].feature_columns,
            "norm_map": _load_norm_map(repo_root / "artifacts" / "reference_diagnostics_by_task" / "staging" / "best_feature_norms.parquet"),
            "use_hsmm": True,
            "dropped_subject_ids": [],
            "feature_table_name": "final__feature_table",
        },
        "apnea_11s_anchor1_drop4": {
            "table": apnea_table,
            "target_column": "label.apnea_11s_anchor1_event",
            "feature_columns": apnea_specs["apnea_plus_shared"].feature_columns,
            "norm_map": _load_norm_map(repo_root / "artifacts" / "reference_diagnostics_by_task" / "apnea" / "best_feature_norms.parquet"),
            "use_hsmm": True,
            "dropped_subject_ids": list(PAPER_APNEA_DROPPED_SUBJECTS),
            "feature_table_name": "final__feature_table_apnea_1s",
        },
    }
    tasks = {name: tasks[name] for name in args.tasks if name in tasks}
    if not tasks:
        status_path.write_text(json.dumps({"status": "failed", "reason": "no_tasks_selected"}, indent=2), encoding="utf-8")
        _log("no_tasks_selected", log_path=log_path)
        return 1

    out_root = repo_root / "artifacts" / "paper_matched_benchmarks_hsmm" / split_tag
    out_root.mkdir(parents=True, exist_ok=True)
    overall_status: dict[str, object] = {"status": "running", "tasks": {}}
    status_path.write_text(json.dumps(overall_status, indent=2), encoding="utf-8")

    for task_name, cfg in tasks.items():
        _log(
            f"task_start name={task_name} target={cfg['target_column']} features={len(cfg['feature_columns'])}",
            log_path=log_path,
        )
        frame = _prepare_frame(
            cfg["table"],
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
            use_hsmm=bool(cfg["use_hsmm"]),
            outer_validation=args.outer_validation,
            outer_folds=args.outer_folds,
            inner_validation=args.inner_validation,
            random_state=args.random_state,
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
            "feature_table_name": str(cfg["feature_table_name"]),
            "outer_validation": args.outer_validation,
            "outer_folds": args.outer_folds,
            "inner_validation": args.inner_validation,
            "model_family": "ensemble",
            "class_weight": "balanced",
            "sequence_postprocessing": "hsmm" if cfg["use_hsmm"] else "off",
            "norm_source": "best_feature_norms",
            "dropped_subject_ids": list(cfg["dropped_subject_ids"]),
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
