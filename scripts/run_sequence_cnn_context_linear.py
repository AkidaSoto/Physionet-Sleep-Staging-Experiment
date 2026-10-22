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
    align_probability_matrix,
    apply_prediction_updates,
    attach_aligned_probabilities,
    build_probability_context_frame,
    build_split_tag,
    prepare_model_frame,
    summarize_numeric_folds,
)


def _log(message: str, *, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-validation", choices=("none", "single_subject"), default="none")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--context-radius", type=int, default=2)
    return parser.parse_args()


def _split_tag(*, outer_validation: str, outer_folds: int, inner_validation: str) -> str:
    return build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
    )


def _context_tag(context_radius: int) -> str:
    return f"ctx{(2 * int(context_radius)) + 1}"


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


def _build_context_prob_frame(
    pred_frame: pd.DataFrame,
    *,
    class_labels: list[int],
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    context_radius: int = 2,
    expected_step_seconds: float = 30.0,
    step_tolerance_seconds: float = 0.25,
) -> pd.DataFrame:
    return build_probability_context_frame(
        pred_frame,
        class_labels=class_labels,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=context_radius,
        expected_step_seconds=expected_step_seconds,
        step_tolerance_seconds=step_tolerance_seconds,
    )


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _context_linear_columns(class_labels: list[int], *, context_radius: int) -> list[str]:
    out: list[str] = []
    for offset in range(-context_radius, context_radius + 1):
        label = _offset_label(offset)
        out.extend([f"{label}.proba_{stage}" for stage in class_labels])
    return out


def _attach_probabilities(
    frame: pd.DataFrame,
    *,
    class_labels: list[int],
    encoder_classes: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    aligned = align_probability_matrix(
        proba,
        estimator_classes=encoder_classes,
        class_labels=class_labels,
    )
    return attach_aligned_probabilities(
        frame,
        class_labels=class_labels,
        aligned_proba=aligned,
    )


def _apply_stage2_predictions(
    raw_pred_frame: pd.DataFrame,
    context_test: pd.DataFrame,
    *,
    y_pred_context: np.ndarray,
    subject_column: str,
    time_column: str,
) -> pd.DataFrame:
    return apply_prediction_updates(
        raw_pred_frame,
        context_test,
        y_pred_values=np.asarray(y_pred_context),
        subject_column=subject_column,
        time_column=time_column,
    )


def _summarize_fold_metrics(fold_df: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    return summarize_numeric_folds(fold_df, eval_source=eval_source)


def main() -> int:
    args = _parse_args()
    repo_root = REPO_ROOT
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import build_staging_benchmark_table, load_ucddb_reference_dataset
    from physionet_sleep.experiments.normalization import apply_normalization
    from physionet_sleep.experiments.sequence_cnn import (
        SequenceExperimentSpec,
        _impute_features,
        _predict_cnn_classifier,
        _split_inner_validation_subjects,
        _train_cnn_classifier,
    )
    from physionet_sleep.experiments.sequence_data import build_centered_sequence_dataset
    from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits, score_predictions
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    split_tag = _split_tag(
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
    )
    run_tag = f"{split_tag}__{_context_tag(args.context_radius)}"
    log_dir = repo_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sequence_cnn_context_linear__{run_tag}.log"
    status_path = log_dir / f"sequence_cnn_context_linear__{run_tag}_status.json"

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
    base_spec = specs["staging_plus_shared"]
    staging_table = build_staging_benchmark_table(
        table,
        feature_columns=base_spec.feature_columns,
    )
    frame = _prepare_frame(
        staging_table,
        subject_column="subject_id",
        target_column="label.stage_30s",
        feature_columns=list(base_spec.feature_columns),
    )
    output_root = repo_root / "artifacts" / "sequence_cnn_context_linear" / run_tag
    output_root.mkdir(parents=True, exist_ok=True)

    spec = SequenceExperimentSpec(
        name=f"staging_plus_shared_cnn_{_context_tag(args.context_radius)}",
        target_column="label.stage_30s",
        feature_columns=base_spec.feature_columns,
        context_radius=args.context_radius,
        normalization="medianiqr",
        cv_folds=args.outer_folds,
        epochs=60,
        batch_size=256,
        learning_rate=1e-3,
        hidden_channels=64,
        kernel_size=3,
        dropout=0.15,
        min_epochs=12,
        patience=10,
        internal_val_subjects=1,
        weighted_sampling=True,
        expected_step_seconds=30.0,
        step_tolerance_seconds=0.25,
    )

    groups = frame["subject_id"].astype(str).to_numpy()
    y = frame["label.stage_30s"].to_numpy()
    class_labels = sorted(int(value) for value in np.unique(y))
    stage2_columns = _context_linear_columns(class_labels, context_radius=spec.context_radius)
    outer_splits = list(
        build_subject_eval_splits(
            groups=groups,
            y=y,
            outer_validation=args.outer_validation,
            outer_folds=args.outer_folds,
            inner_validation=args.inner_validation,
            random_state=args.random_state,
        )
    )

    raw_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    raw_pred_frames: list[pd.DataFrame] = []
    context_pred_frames: list[pd.DataFrame] = []

    for split in outer_splits:
        fold_index = split.fold_index
        train_df_raw = frame.iloc[split.train_idx].copy()
        test_df_raw = frame.iloc[split.test_idx].copy()
        fit_train_raw, fit_val_raw = _split_inner_validation_subjects(
            train_df_raw,
            subject_column="subject_id",
            fold_index=fold_index,
            n_val_subjects=spec.internal_val_subjects,
            random_state=args.random_state,
        )

        fit_train_df, fit_val_df, norm_stats = apply_normalization(
            fit_train_raw,
            fit_val_raw,
            feature_columns=spec.feature_columns,
            norm_name=spec.normalization,
            group_column="subject_id",
        )
        _, test_df = apply_normalization(
            fit_train_raw,
            test_df_raw,
            feature_columns=spec.feature_columns,
            norm_name=spec.normalization,
            group_column="subject_id",
        )[:2]
        fit_train_df, fit_val_df = _impute_features(
            fit_train_df,
            fit_val_df,
            feature_columns=spec.feature_columns,
        )
        fit_train_for_eval, test_df = _impute_features(
            fit_train_df,
            test_df,
            feature_columns=spec.feature_columns,
        )

        train_dataset = build_centered_sequence_dataset(
            fit_train_df,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column="subject_id",
            time_column="time_seconds",
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
        fit_val_dataset = build_centered_sequence_dataset(
            fit_val_df,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column="subject_id",
            time_column="time_seconds",
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
        train_eval_dataset = build_centered_sequence_dataset(
            fit_train_for_eval,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column="subject_id",
            time_column="time_seconds",
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
        test_dataset = build_centered_sequence_dataset(
            test_df,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column="subject_id",
            time_column="time_seconds",
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
        if train_dataset.X.shape[0] == 0 or fit_val_dataset.X.shape[0] == 0 or test_dataset.X.shape[0] == 0:
            continue

        model, encoder, training_info = _train_cnn_classifier(
            train_dataset.X,
            train_dataset.y,
            X_early_val=fit_val_dataset.X,
            y_early_val=fit_val_dataset.y,
            epochs=spec.epochs,
            min_epochs=spec.min_epochs,
            patience=spec.patience,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            hidden_channels=spec.hidden_channels,
            kernel_size=spec.kernel_size,
            dropout=spec.dropout,
            weighted_sampling=spec.weighted_sampling,
            random_state=args.random_state,
        )

        y_train_pred, train_proba = _predict_cnn_classifier(model, encoder, train_eval_dataset.X)
        y_val_pred, val_proba = _predict_cnn_classifier(model, encoder, fit_val_dataset.X)
        y_test_pred, test_proba = _predict_cnn_classifier(model, encoder, test_dataset.X)

        train_metrics = score_predictions(y_true=train_eval_dataset.y, y_pred=y_train_pred, proba=train_proba)
        raw_metrics = score_predictions(y_true=test_dataset.y, y_pred=y_test_pred, proba=test_proba)
        raw_metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(train_dataset.X.shape[0]),
                "n_test": int(test_dataset.X.shape[0]),
                "n_early_val": int(fit_val_dataset.X.shape[0]),
                "n_subjects_train": int(fit_train_df["subject_id"].nunique()),
                "n_subjects_early_val": int(fit_val_df["subject_id"].nunique()),
                "n_subjects_test": int(test_df["subject_id"].nunique()),
                "best_epoch": int(training_info["best_epoch"]),
                "early_val_macro_f1": float(training_info["best_val_macro_f1"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }
        )
        raw_rows.append(raw_metrics)

        val_pred_frame = fit_val_dataset.metadata.copy()
        val_pred_frame["fold_index"] = fold_index
        val_pred_frame["y_true"] = fit_val_dataset.y
        val_pred_frame["y_pred"] = y_val_pred
        val_pred_frame = _attach_probabilities(
            val_pred_frame,
            class_labels=class_labels,
            encoder_classes=encoder.classes_,
            proba=val_proba,
        )

        test_pred_frame = test_dataset.metadata.copy()
        test_pred_frame["fold_index"] = fold_index
        test_pred_frame["y_true"] = test_dataset.y
        test_pred_frame["y_pred"] = y_test_pred
        test_pred_frame = _attach_probabilities(
            test_pred_frame,
            class_labels=class_labels,
            encoder_classes=encoder.classes_,
            proba=test_proba,
        )
        raw_pred_frames.append(test_pred_frame.copy())

        context_train = _build_context_prob_frame(
            val_pred_frame,
            class_labels=class_labels,
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
        context_test = _build_context_prob_frame(
            test_pred_frame,
            class_labels=class_labels,
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )

        smoother = build_model(
            model_family="linear",
            task="classification",
            random_state=args.random_state,
            class_weight="balanced",
        )
        smoother.fit(context_train[stage2_columns], context_train["y_true"])
        context_test_pred = np.asarray(smoother.predict(context_test[stage2_columns]))
        refined_pred_frame = _apply_stage2_predictions(
            test_pred_frame,
            context_test,
            y_pred_context=context_test_pred,
            subject_column="subject_id",
            time_column="time_seconds",
        )
        context_pred_frames.append(refined_pred_frame)
        context_metrics = score_predictions(
            y_true=refined_pred_frame["y_true"].to_numpy(),
            y_pred=refined_pred_frame["y_pred"].to_numpy(),
            proba=None,
        )
        context_metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(train_dataset.X.shape[0]),
                "n_test": int(test_dataset.X.shape[0]),
                "n_early_val": int(fit_val_dataset.X.shape[0]),
                "n_stage2_rows_train": int(len(context_train)),
                "n_stage2_rows_test": int(len(context_test)),
                "n_stage2_features": int(len(stage2_columns)),
                "n_subjects_train": int(fit_train_df["subject_id"].nunique()),
                "n_subjects_early_val": int(fit_val_df["subject_id"].nunique()),
                "n_subjects_test": int(test_df["subject_id"].nunique()),
                "best_epoch": int(training_info["best_epoch"]),
                "early_val_macro_f1": float(training_info["best_val_macro_f1"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }
        )
        context_rows.append(context_metrics)

        _log(
            f"fold={fold_index}/{len(outer_splits)} raw_macro_f1={raw_metrics.get('macro_f1')} "
            f"context_linear_macro_f1={context_metrics.get('macro_f1')}",
            log_path=log_path,
        )

    raw_fold_df = pd.DataFrame(raw_rows)
    context_fold_df = pd.DataFrame(context_rows)
    raw_summary = _summarize_fold_metrics(raw_fold_df, eval_source="cnn_raw")
    context_summary = _summarize_fold_metrics(context_fold_df, eval_source="cnn_context_linear")
    raw_pred_df = pd.concat(raw_pred_frames, axis=0, ignore_index=True) if raw_pred_frames else pd.DataFrame()
    context_pred_df = pd.concat(context_pred_frames, axis=0, ignore_index=True) if context_pred_frames else pd.DataFrame()

    raw_summary.to_parquet(output_root / "summary_raw.parquet", index=False)
    context_summary.to_parquet(output_root / "summary_context_linear.parquet", index=False)
    raw_fold_df.to_parquet(output_root / "fold_metrics_raw.parquet", index=False)
    context_fold_df.to_parquet(output_root / "fold_metrics_context_linear.parquet", index=False)
    raw_pred_df.to_parquet(output_root / "predictions_raw.parquet", index=False)
    context_pred_df.to_parquet(output_root / "predictions_context_linear.parquet", index=False)

    metadata = {
        "outer_validation": args.outer_validation,
        "outer_folds": args.outer_folds,
        "inner_validation": args.inner_validation,
        "stage1_model": "cnn",
        "stage2_model": "context_linear",
        "target_column": spec.target_column,
        "feature_count": len(spec.feature_columns),
        "context_radius": spec.context_radius,
        "expected_step_seconds": spec.expected_step_seconds,
        "rows": int(len(frame)),
        "subjects": sorted(frame["subject_id"].astype(str).unique().tolist()),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    status = {
        "status": "completed",
        "raw": raw_summary.iloc[0].to_dict() if not raw_summary.empty else {},
        "context_linear": context_summary.iloc[0].to_dict() if not context_summary.empty else {},
        "output_dir": str(output_root),
    }
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    _log(f"run_done output_dir={output_root}", log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
