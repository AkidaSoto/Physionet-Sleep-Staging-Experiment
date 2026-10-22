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
    parser.add_argument("--history-epochs", type=int, default=2)
    parser.add_argument("--current-epochs", type=int, default=1)
    parser.add_argument("--future-epochs", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--tcn-levels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--internal-val-subjects", type=int, default=3)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucddb"],
        help="Datasets from physionet_sleep.analysis.datasets to use; non-default runs get their own output folder.",
    )
    parser.add_argument("--stage3-context-radius", type=int, default=2)
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def _split_tag(*, outer_validation: str, outer_folds: int) -> str:
    return build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
    )


def _shape_tag(*, history_epochs: int, current_epochs: int, future_epochs: int, epoch_steps: int) -> str:
    return f"eh{history_epochs}_ec{current_epochs}_ef{future_epochs}_s{epoch_steps}"


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
        drop_negative_targets=True,
    )


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _safe_logit(value: float, *, eps: float = 1.0e-4) -> float:
    clipped = float(np.clip(value, eps, 1.0 - eps))
    return float(np.log(clipped / (1.0 - clipped)))


def _confidence_summary(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    probs = np.asarray(values, dtype=float)
    top_sorted = np.sort(probs)[::-1]
    top1 = float(top_sorted[0])
    top2 = float(top_sorted[1]) if top_sorted.size > 1 else 0.0
    margin = top1 - top2
    entropy = float(-(probs * np.log(np.clip(probs, 1.0e-12, 1.0))).sum())
    return top1, margin, entropy


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
        include_confidence=True,
        include_deltas=True,
        passthrough_prefixes=("logit_", "embed_"),
    )


def _stage3_feature_sets(class_labels: list[int], *, context_radius: int) -> dict[str, list[str]]:
    center_cols = [f"ctx_0.proba_{label}" for label in class_labels]
    context_cols: list[str] = []
    context_logit_cols: list[str] = []
    confidence_cols: list[str] = []
    embed_cols = [f"embed_{idx}" for idx in range(256)]
    context_embed_cols: list[str] = []
    for offset in range(-context_radius, context_radius + 1):
        offset_tag = _offset_label(offset)
        context_cols.extend([f"{offset_tag}.proba_{stage}" for stage in class_labels])
        context_logit_cols.extend([f"{offset_tag}.logit_{stage}" for stage in class_labels])
        context_embed_cols.extend([f"{offset_tag}.{col}" for col in embed_cols])
        confidence_cols.extend(
            [
                f"{offset_tag}.top1_proba",
                f"{offset_tag}.margin",
                f"{offset_tag}.entropy",
            ]
        )
    delta_cols = [f"delta_prev.proba_{label}" for label in class_labels] + [
        f"delta_next.proba_{label}" for label in class_labels
    ]
    return {
        "center_linear": center_cols,
        "context_linear": context_cols,
        "context_true_logit": context_logit_cols,
        "context_true_logit_conf": [*context_logit_cols, *confidence_cols, *delta_cols],
        "context_embed_linear": context_embed_cols,
        "context_embed_logit_conf": [*context_logit_cols, *context_embed_cols, *confidence_cols, *delta_cols],
    }


def _available_stage3_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in frame.columns]


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


def _attach_logits_and_embeddings(
    frame: pd.DataFrame,
    *,
    class_labels: list[int],
    encoder_classes: np.ndarray,
    logits: np.ndarray,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    out = frame.copy()
    aligned = np.zeros((len(out), len(class_labels)), dtype=float)
    class_to_idx = {int(value): idx for idx, value in enumerate(class_labels)}
    for src_idx, label in enumerate(encoder_classes.tolist()):
        dst_idx = class_to_idx.get(int(label))
        if dst_idx is not None:
            aligned[:, dst_idx] = logits[:, src_idx]
    logit_frame = pd.DataFrame(
        {f"logit_{label}": aligned[:, idx] for idx, label in enumerate(class_labels)}
    )
    embed_frame = pd.DataFrame(
        embeddings,
        columns=[f"embed_{idx}" for idx in range(int(embeddings.shape[1]))],
    )
    return pd.concat([out.reset_index(drop=True), logit_frame, embed_frame], axis=1)


def _build_stage3_linear_estimator():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _apply_stage3_predictions(
    raw_pred_frame: pd.DataFrame,
    context_test: pd.DataFrame,
    *,
    variant_name: str,
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
        variant_name=variant_name,
    )


def _summarize_fold_metrics(fold_df: pd.DataFrame, *, eval_source: str) -> pd.DataFrame:
    return summarize_numeric_folds(fold_df, eval_source=eval_source)


def main() -> int:
    args = _parse_args()
    repo_root = REPO_ROOT
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis.reference_data import load_ucddb_reference_dataset
    from physionet_sleep.experiments.normalization import apply_normalization
    from physionet_sleep.experiments.sequence_cnn import (
        SequenceExperimentSpec,
        _build_sequence_dataset,
        _impute_features,
        _predict_cnn_classifier,
        _predict_cnn_classifier_detailed,
        _split_inner_validation_subjects,
        _train_cnn_classifier,
    )
    from physionet_sleep.experiments.tabular_runner import build_validation_splits, score_predictions
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    split_tag = _split_tag(
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
    )
    shape_tag = _shape_tag(
        history_epochs=args.history_epochs,
        current_epochs=args.current_epochs,
        future_epochs=args.future_epochs,
        epoch_steps=args.epoch_steps,
    )
    run_tag = f"{split_tag}__{shape_tag}__hierarchical_epoch"
    if args.datasets != ["ucddb"]:
        run_tag = f"{run_tag}__{'+'.join(args.datasets)}"
    log_dir = repo_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hierarchical_epoch_staging__{run_tag}.log"
    status_path = log_dir / f"hierarchical_epoch_staging__{run_tag}_status.json"

    _log("run_start", log_path=log_path)
    if args.datasets == ["ucddb"]:
        table = load_ucddb_reference_dataset(
            cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
            root_dir=repo_root / "data" / "raw" / "ucddb",
        )
    else:
        from physionet_sleep.analysis.datasets import load_reference_dataset

        table = load_reference_dataset(args.datasets)
    if table.empty:
        status_path.write_text(json.dumps({"status": "failed", "reason": "no_rows"}, indent=2), encoding="utf-8")
        _log("no_rows", log_path=log_path)
        return 1

    specs = build_default_experiment_specs(table)
    base_spec = specs["staging_plus_shared"]
    frame = _prepare_frame(
        table,
        subject_column="subject_id",
        target_column="label.stage_seconds",
        feature_columns=list(base_spec.feature_columns),
    )
    output_root = repo_root / "artifacts" / "hierarchical_epoch_staging" / run_tag
    output_root.mkdir(parents=True, exist_ok=True)

    spec = SequenceExperimentSpec(
        name=f"staging_plus_shared_hierarchical_{shape_tag}",
        target_column="label.stage_seconds",
        feature_columns=list(base_spec.feature_columns),
        model_arch="hierarchical_epoch",
        sequence_mode="hierarchical_epochs",
        history_steps=args.history_epochs,
        current_steps=args.current_epochs,
        future_steps=args.future_epochs,
        epoch_steps=args.epoch_steps,
        normalization="medianiqr",
        cv_folds=args.outer_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        tcn_levels=args.tcn_levels,
        min_epochs=args.min_epochs,
        patience=args.patience,
        internal_val_subjects=args.internal_val_subjects,
        weighted_sampling=True,
        expected_step_seconds=6.0,
        step_tolerance_seconds=0.25,
    )

    groups = frame["subject_id"].astype(str).to_numpy()
    y = frame["label.stage_seconds"].to_numpy()
    outer_splits = list(
        build_validation_splits(
            groups=groups,
            y=y,
            folds=args.outer_folds,
            validation=args.outer_validation,
            random_state=args.random_state,
        )
    )
    class_labels = sorted(int(value) for value in np.unique(y[y >= 0]))
    stage3_feature_sets = _stage3_feature_sets(class_labels, context_radius=args.stage3_context_radius)

    raw_rows: list[dict[str, object]] = []
    raw_pred_frames: list[pd.DataFrame] = []
    stage3_rows_by_variant: dict[str, list[dict[str, object]]] = {name: [] for name in stage3_feature_sets}
    stage3_pred_frames_by_variant: dict[str, list[pd.DataFrame]] = {name: [] for name in stage3_feature_sets}

    for fold_index, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train_df_raw = frame.iloc[train_idx].copy()
        test_df_raw = frame.iloc[test_idx].copy()
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
        _, test_df, _ = apply_normalization(
            fit_train_raw,
            test_df_raw,
            feature_columns=spec.feature_columns,
            norm_name=spec.normalization,
            group_column="subject_id",
        )
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

        train_dataset = _build_sequence_dataset(
            fit_train_df,
            spec=spec,
            subject_column="subject_id",
            time_column="time_seconds",
        )
        fit_val_dataset = _build_sequence_dataset(
            fit_val_df,
            spec=spec,
            subject_column="subject_id",
            time_column="time_seconds",
        )
        train_eval_dataset = _build_sequence_dataset(
            fit_train_for_eval,
            spec=spec,
            subject_column="subject_id",
            time_column="time_seconds",
        )
        test_dataset = _build_sequence_dataset(
            test_df,
            spec=spec,
            subject_column="subject_id",
            time_column="time_seconds",
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
            model_arch=spec.model_arch,
            history_steps=spec.history_steps,
            current_steps=spec.current_steps,
            future_steps=spec.future_steps,
            tcn_levels=spec.tcn_levels,
            weighted_sampling=spec.weighted_sampling,
            random_state=args.random_state,
        )

        y_train_pred, train_proba = _predict_cnn_classifier(model, encoder, train_eval_dataset.X)
        y_val_pred, val_proba, val_logits, val_embeddings = _predict_cnn_classifier_detailed(model, encoder, fit_val_dataset.X)
        y_test_pred, test_proba, test_logits, test_embeddings = _predict_cnn_classifier_detailed(model, encoder, test_dataset.X)

        train_metrics = score_predictions(y_true=train_eval_dataset.y, y_pred=y_train_pred, proba=train_proba)
        raw_metrics = score_predictions(y_true=test_dataset.y, y_pred=y_test_pred, proba=test_proba)
        raw_metrics.update(
            {
                "fold_index": fold_index,
                "n_train": int(train_dataset.X.shape[0]),
                "n_test": int(test_dataset.X.shape[0]),
                "n_early_val": int(fit_val_dataset.X.shape[0]),
                "n_features_used": int(len(spec.feature_columns)),
                "epoch_window_length": int(train_dataset.window_length),
                "epoch_steps": int(spec.epoch_steps),
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
        val_pred_frame["y_true"] = fit_val_dataset.y.astype(int)
        val_pred_frame["y_pred"] = y_val_pred.astype(int)
        val_pred_frame = _attach_probabilities(
            val_pred_frame,
            class_labels=class_labels,
            encoder_classes=encoder.classes_,
            proba=val_proba,
        )
        val_pred_frame = _attach_logits_and_embeddings(
            val_pred_frame,
            class_labels=class_labels,
            encoder_classes=encoder.classes_,
            logits=val_logits,
            embeddings=val_embeddings,
        )

        test_pred_frame = test_dataset.metadata.copy()
        test_pred_frame["fold_index"] = fold_index
        test_pred_frame["y_true"] = test_dataset.y.astype(int)
        test_pred_frame["y_pred"] = y_test_pred.astype(int)
        test_pred_frame = _attach_probabilities(
            test_pred_frame,
            class_labels=class_labels,
            encoder_classes=encoder.classes_,
            proba=test_proba,
        )
        test_pred_frame = _attach_logits_and_embeddings(
            test_pred_frame,
            class_labels=class_labels,
            encoder_classes=encoder.classes_,
            logits=test_logits,
            embeddings=test_embeddings,
        )
        raw_pred_frames.append(test_pred_frame.copy())

        context_train = pd.DataFrame()
        context_test = pd.DataFrame()
        if not args.skip_stage3:
            context_train = _build_context_prob_frame(
                val_pred_frame,
                class_labels=class_labels,
                context_radius=args.stage3_context_radius,
                expected_step_seconds=30.0,
                step_tolerance_seconds=0.25,
            )
            context_test = _build_context_prob_frame(
                test_pred_frame,
                class_labels=class_labels,
                context_radius=args.stage3_context_radius,
                expected_step_seconds=30.0,
                step_tolerance_seconds=0.25,
            )

        for variant_name, stage3_columns in stage3_feature_sets.items():
            if args.skip_stage3 or context_train.empty or context_test.empty:
                continue
            usable_stage3_columns = _available_stage3_columns(context_train, stage3_columns)
            usable_stage3_columns = [col for col in usable_stage3_columns if col in context_test.columns]
            if not usable_stage3_columns:
                continue
            smoother = _build_stage3_linear_estimator()
            smoother.fit(context_train[usable_stage3_columns], context_train["y_true"])
            context_test_pred = np.asarray(smoother.predict(context_test[usable_stage3_columns]))
            refined_pred_frame = _apply_stage3_predictions(
                test_pred_frame,
                context_test,
                variant_name=variant_name,
                y_pred_context=context_test_pred,
                subject_column="subject_id",
                time_column="time_seconds",
            )
            stage3_pred_frames_by_variant[variant_name].append(refined_pred_frame)
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
                    "n_features_used": int(len(spec.feature_columns)),
                    "epoch_window_length": int(train_dataset.window_length),
                    "epoch_steps": int(spec.epoch_steps),
                    "n_stage3_rows_train": int(len(context_train)),
                    "n_stage3_rows_test": int(len(context_test)),
                    "n_stage3_features": int(len(usable_stage3_columns)),
                    "n_subjects_train": int(fit_train_df["subject_id"].nunique()),
                    "n_subjects_early_val": int(fit_val_df["subject_id"].nunique()),
                    "n_subjects_test": int(test_df["subject_id"].nunique()),
                    "best_epoch": int(training_info["best_epoch"]),
                    "early_val_macro_f1": float(training_info["best_val_macro_f1"]),
                    "train_macro_f1": float(train_metrics["macro_f1"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                }
            )
            stage3_rows_by_variant[variant_name].append(context_metrics)

        if args.skip_stage3:
            _log(
                f"fold={fold_index}/{len(outer_splits)} raw_macro_f1={raw_metrics.get('macro_f1')}",
                log_path=log_path,
            )
        else:
            best_stage3 = stage3_rows_by_variant["context_embed_logit_conf"][-1]["macro_f1"] if stage3_rows_by_variant["context_embed_logit_conf"] else None
            _log(
                f"fold={fold_index}/{len(outer_splits)} raw_macro_f1={raw_metrics.get('macro_f1')} "
                f"stage3_embed_logit_conf_macro_f1={best_stage3}",
                log_path=log_path,
            )

    raw_fold_df = pd.DataFrame(raw_rows)
    raw_summary = _summarize_fold_metrics(raw_fold_df, eval_source="hierarchical_raw")
    raw_pred_df = pd.concat(raw_pred_frames, axis=0, ignore_index=True) if raw_pred_frames else pd.DataFrame()
    raw_summary.to_parquet(output_root / "summary_raw.parquet", index=False)
    raw_fold_df.to_parquet(output_root / "fold_metrics_raw.parquet", index=False)
    raw_pred_df.to_parquet(output_root / "predictions_raw.parquet", index=False)

    all_summary_rows: list[dict[str, object]] = []
    all_fold_rows: list[pd.DataFrame] = []
    if not raw_summary.empty:
        raw_row = raw_summary.iloc[0].to_dict()
        raw_row["variant"] = "raw"
        raw_row["stage3_context_radius"] = int(args.stage3_context_radius)
        all_summary_rows.append(raw_row)
    if not raw_fold_df.empty:
        raw_fold_out = raw_fold_df.copy()
        raw_fold_out["variant"] = "raw"
        raw_fold_out["stage3_context_radius"] = int(args.stage3_context_radius)
        all_fold_rows.append(raw_fold_out)

    status: dict[str, object] = {
        "status": "completed",
        "raw": raw_summary.iloc[0].to_dict() if not raw_summary.empty else {},
        "stage3": {},
        "output_dir": str(output_root),
    }
    for variant_name, rows in stage3_rows_by_variant.items():
        fold_df = pd.DataFrame(rows)
        pred_df = pd.concat(stage3_pred_frames_by_variant[variant_name], axis=0, ignore_index=True) if stage3_pred_frames_by_variant[variant_name] else pd.DataFrame()
        summary_df = _summarize_fold_metrics(fold_df, eval_source=variant_name)
        summary_df.to_parquet(output_root / f"summary_{variant_name}.parquet", index=False)
        fold_df.to_parquet(output_root / f"fold_metrics_{variant_name}.parquet", index=False)
        pred_df.to_parquet(output_root / f"predictions_{variant_name}.parquet", index=False)
        if not summary_df.empty:
            summary_row = summary_df.iloc[0].to_dict()
            summary_row["variant"] = variant_name
            summary_row["stage3_context_radius"] = int(args.stage3_context_radius)
            all_summary_rows.append(summary_row)
            status["stage3"][variant_name] = summary_row
        if not fold_df.empty:
            fold_out = fold_df.copy()
            fold_out["variant"] = variant_name
            fold_out["stage3_context_radius"] = int(args.stage3_context_radius)
            all_fold_rows.append(fold_out)

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_parquet(output_root / "summary_all.parquet", index=False)
    if all_fold_rows:
        pd.concat(all_fold_rows, axis=0, ignore_index=True).to_parquet(output_root / "fold_metrics_all.parquet", index=False)

    metadata = {
        "outer_validation": args.outer_validation,
        "outer_folds": args.outer_folds,
        "history_epochs": args.history_epochs,
        "current_epochs": args.current_epochs,
        "future_epochs": args.future_epochs,
        "epoch_steps": args.epoch_steps,
        "model_arch": "hierarchical_epoch",
        "target_column": spec.target_column,
        "feature_count": len(spec.feature_columns),
        "stage3_context_radius": args.stage3_context_radius,
        "rows": int(len(frame)),
        "subjects": sorted(frame["subject_id"].astype(str).unique().tolist()),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    _log(f"run_done output_dir={output_root}", log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
