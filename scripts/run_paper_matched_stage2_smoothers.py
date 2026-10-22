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
    apply_featurewise_norm_map,
    apply_prediction_updates,
    attach_aligned_probabilities,
    build_probability_context_frame,
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
    parser.add_argument("--context-radii", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--stage1-model-family",
        choices=("ensemble", "hgb", "random_forest", "extra_trees", "tree", "linear"),
        default="ensemble",
    )
    parser.add_argument(
        "--stage2-train-probs",
        choices=("auto", "insample", "oof"),
        default="auto",
        help="Stage-1 probabilities the stage-2 smoothers and HSMM learn from when inner validation is none. "
        "auto = oof for hgb (it fits its training rows almost perfectly), insample otherwise.",
    )
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucddb"],
        help="Datasets from physionet_sleep.analysis.datasets to pool (subject-grouped folds over all of them).",
    )
    return parser.parse_args()


def resolve_stage2_train_probs(stage2_train_probs: str, stage1_model_family: str) -> str:
    if stage2_train_probs != "auto":
        return stage2_train_probs
    return "oof" if stage1_model_family == "hgb" else "insample"


def out_of_fold_probabilities(
    train_df: pd.DataFrame,
    *,
    usable: list[str],
    target_column: str,
    subject_column: str,
    class_labels: list[int],
    model_family: str,
    random_state: int,
    n_folds: int = 3,
) -> np.ndarray:
    """Stage-1 probabilities for every training row from models that never saw that row's subject."""
    from sklearn.model_selection import GroupKFold

    from physionet_sleep.experiments.tabular_runner import build_model

    groups = train_df[subject_column].astype(str).to_numpy()
    n_splits = min(int(n_folds), int(np.unique(groups).size))
    if n_splits < 2:
        raise ValueError("out-of-fold stage-1 probabilities need at least two training subjects")
    oof = np.zeros((len(train_df), len(class_labels)), dtype=float)
    for fit_idx, pred_idx in GroupKFold(n_splits=n_splits).split(train_df, groups=groups):
        estimator = build_model(
            model_family=model_family,
            task="classification",
            random_state=random_state,
            class_weight="balanced",
        )
        fit_rows = train_df.iloc[fit_idx]
        estimator.fit(fit_rows[usable], fit_rows[target_column])
        oof[pred_idx] = align_probability_matrix(
            estimator.predict_proba(train_df.iloc[pred_idx][usable]),
            estimator_classes=np.asarray(estimator.classes_, dtype=int),
            class_labels=class_labels,
        )
    return oof


def _split_tag(*, outer_validation: str, outer_folds: int, inner_validation: str) -> str:
    return build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
    )


def _aligned_proba(proba: np.ndarray | None, *, estimator_classes: np.ndarray, class_labels: list[int]) -> np.ndarray:
    return align_probability_matrix(
        proba,
        estimator_classes=estimator_classes,
        class_labels=class_labels,
    )


def _attach_probabilities(
    frame: pd.DataFrame,
    *,
    class_labels: list[int],
    aligned_proba: np.ndarray,
) -> pd.DataFrame:
    return attach_aligned_probabilities(
        frame,
        class_labels=class_labels,
        aligned_proba=aligned_proba,
    )


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
        include_probability_logits=True,
        include_confidence=True,
        include_deltas=True,
    )


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _stage2_feature_sets(class_labels: list[int], *, context_radius: int) -> dict[str, list[str]]:
    center_cols = [f"ctx_0.proba_{label}" for label in class_labels]
    context_cols: list[str] = []
    context_logit_cols: list[str] = []
    confidence_cols: list[str] = []
    for offset in range(-context_radius, context_radius + 1):
        label = _offset_label(offset)
        context_cols.extend([f"{label}.proba_{stage}" for stage in class_labels])
        context_logit_cols.extend([f"{label}.proba_{stage}_logit" for stage in class_labels])
        confidence_cols.extend(
            [
                f"{label}.top1_proba",
                f"{label}.margin",
                f"{label}.entropy",
            ]
        )
    delta_cols = [f"delta_prev.proba_{label}" for label in class_labels] + [
        f"delta_next.proba_{label}" for label in class_labels
    ]
    return {
        "center_linear": center_cols,
        "context_linear": context_cols,
        "context_ensemble": context_cols,
        "context_linear_logit": context_logit_cols,
        "context_linear_logit_conf": [*context_logit_cols, *confidence_cols, *delta_cols],
    }


def _apply_stage2_predictions(
    raw_pred_frame: pd.DataFrame,
    context_val: pd.DataFrame,
    *,
    variant_name: str,
    y_pred_context: np.ndarray,
    subject_column: str,
    time_column: str,
) -> pd.DataFrame:
    return apply_prediction_updates(
        raw_pred_frame,
        context_val,
        y_pred_values=np.asarray(y_pred_context),
        subject_column=subject_column,
        time_column=time_column,
        variant_name=variant_name,
    )


def _fit_stage1_cache(
    frame: pd.DataFrame,
    *,
    task_name: str,
    subject_column: str,
    target_column: str,
    feature_columns: list[str],
    norm_map: dict[str, str],
    context_radius: int,
    outer_validation: str,
    outer_folds: int,
    inner_validation: str,
    random_state: int,
    log_path: Path | None = None,
    stage1_model_family: str = "ensemble",
    stage2_train_probs: str = "insample",
    oof_folds: int = 3,
) -> dict[str, object]:
    from physionet_sleep.decoders.hsmm import apply_hsmm_viterbi, learn_hsmm_params
    from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits, score_predictions

    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[target_column].to_numpy()
    class_labels = sorted(int(value) for value in np.unique(y))
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

    raw_metrics_rows: list[dict[str, object]] = []
    hsmm_metrics_rows: list[dict[str, object]] = []
    raw_pred_frames: list[pd.DataFrame] = []
    hsmm_pred_frames: list[pd.DataFrame] = []
    stage1_folds: list[dict[str, object]] = []

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
            model_family=stage1_model_family,
            task="classification",
            random_state=random_state,
            class_weight="balanced",
        )
        estimator.fit(train_df[usable], train_df[target_column])

        train_pred = np.asarray(estimator.predict(train_df[usable]))
        test_pred = np.asarray(estimator.predict(test_df[usable]))
        train_proba_raw = estimator.predict_proba(train_df[usable]) if hasattr(estimator, "predict_proba") else None
        test_proba_raw = estimator.predict_proba(test_df[usable]) if hasattr(estimator, "predict_proba") else None
        estimator_classes = np.asarray(getattr(estimator, "classes_", class_labels), dtype=int)
        train_aligned = _aligned_proba(train_proba_raw, estimator_classes=estimator_classes, class_labels=class_labels)
        test_aligned = _aligned_proba(test_proba_raw, estimator_classes=estimator_classes, class_labels=class_labels)

        raw_metrics = score_predictions(
            y_true=test_df[target_column].to_numpy(),
            y_pred=test_pred,
            proba=test_aligned,
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
        raw_metrics_rows.append(raw_metrics)

        train_pred_frame = _attach_probabilities(
            train_df[[subject_column] + (["time_seconds"] if "time_seconds" in train_df.columns else [])].assign(
                y_true=train_df[target_column].to_numpy(),
                y_pred=train_pred,
                fold_index=fold_index,
            ),
            class_labels=class_labels,
            aligned_proba=train_aligned,
        )
        test_pred_frame = _attach_probabilities(
            test_df[[subject_column] + (["time_seconds"] if "time_seconds" in test_df.columns else [])].assign(
                y_true=test_df[target_column].to_numpy(),
                y_pred=test_pred,
                fold_index=fold_index,
            ),
            class_labels=class_labels,
            aligned_proba=test_aligned,
        )
        raw_pred_frames.append(test_pred_frame.copy())

        calib_pred_frame = train_pred_frame
        if calib_df is None and stage2_train_probs == "oof":
            oof_aligned = out_of_fold_probabilities(
                train_df,
                usable=usable,
                target_column=target_column,
                subject_column=subject_column,
                class_labels=class_labels,
                model_family=stage1_model_family,
                random_state=random_state,
                n_folds=oof_folds,
            )
            calib_pred_frame = _attach_probabilities(
                train_df[[subject_column] + (["time_seconds"] if "time_seconds" in train_df.columns else [])].assign(
                    y_true=train_df[target_column].to_numpy(),
                    y_pred=np.asarray(class_labels, dtype=int)[oof_aligned.argmax(axis=1)],
                    fold_index=fold_index,
                ),
                class_labels=class_labels,
                aligned_proba=oof_aligned,
            )
        if calib_df is not None:
            calib_pred = np.asarray(estimator.predict(calib_df[usable]))
            calib_proba_raw = estimator.predict_proba(calib_df[usable]) if hasattr(estimator, "predict_proba") else None
            calib_aligned = _aligned_proba(calib_proba_raw, estimator_classes=estimator_classes, class_labels=class_labels)
            calib_pred_frame = _attach_probabilities(
                calib_df[[subject_column] + (["time_seconds"] if "time_seconds" in calib_df.columns else [])].assign(
                    y_true=calib_df[target_column].to_numpy(),
                    y_pred=calib_pred,
                    fold_index=fold_index,
                ),
                class_labels=class_labels,
                aligned_proba=calib_aligned,
            )

        calib_seq_true, calib_seq_pred = _sequence_arrays(
            calib_pred_frame,
            subject_column=subject_column,
        )
        test_seq_true, test_seq_pred = _sequence_arrays(
            test_pred_frame,
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
        hsmm_metrics_rows.append(hsmm_metrics)

        hsmm_pred_frame = test_pred_frame.copy()
        rebuilt: list[pd.DataFrame] = []
        for seq_index, (_, group) in enumerate(test_pred_frame.groupby(subject_column, sort=False)):
            ordered = group.sort_values("time_seconds", kind="stable") if "time_seconds" in group.columns else group
            tmp = ordered.copy()
            tmp["y_pred"] = np.asarray(test_seq_hsmm[seq_index])[: len(ordered)]
            rebuilt.append(tmp)
        hsmm_pred_frame = pd.concat(rebuilt, axis=0, ignore_index=True)
        hsmm_pred_frame["fold_index"] = fold_index
        hsmm_pred_frames.append(hsmm_pred_frame)

        stage1_folds.append(
            {
                "fold_index": fold_index,
                "raw_metrics": raw_metrics.copy(),
                "hsmm_metrics": hsmm_metrics.copy(),
                "raw_pred_frame": test_pred_frame.copy(),
                "hsmm_pred_frame": hsmm_pred_frame.copy(),
                "calib_pred_frame": calib_pred_frame.copy(),
                "test_pred_frame": test_pred_frame.copy(),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_features_used": int(len(usable)),
                "n_subjects_train": int(train_df[subject_column].nunique()),
                "n_subjects_test": int(test_df[subject_column].nunique()),
                "n_subjects_validation": int(calib_df[subject_column].nunique()) if calib_df is not None else 0,
            }
        )
        if log_path is not None:
            _log(
                f"{task_name} stage1_fold_done fold={fold_index}/{len(splits)} "
                f"raw_macro_f1={raw_metrics.get('macro_f1')} hsmm_macro_f1={hsmm_metrics.get('macro_f1')}",
                log_path=log_path,
            )

    raw_fold_df = pd.DataFrame(raw_metrics_rows)
    hsmm_fold_df = pd.DataFrame(hsmm_metrics_rows)
    return {
        "class_labels": class_labels,
        "stage1_folds": stage1_folds,
        "summary_by_variant": {
            "raw": _summarize_fold_metrics(raw_fold_df, eval_source="raw"),
            "hsmm": _summarize_fold_metrics(hsmm_fold_df, eval_source="hsmm"),
        },
        "detail_by_variant": {
            "raw": (
                raw_fold_df,
                pd.concat(raw_pred_frames, axis=0, ignore_index=True) if raw_pred_frames else pd.DataFrame(),
            ),
            "hsmm": (
                hsmm_fold_df,
                pd.concat(hsmm_pred_frames, axis=0, ignore_index=True) if hsmm_pred_frames else pd.DataFrame(),
            ),
        },
    }


def _run_task_from_stage1_cache(
    stage1_cache: dict[str, object],
    *,
    task_name: str,
    subject_column: str,
    context_radius: int,
    random_state: int,
    log_path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    from physionet_sleep.experiments.tabular_runner import build_model, score_predictions

    class_labels = [int(value) for value in stage1_cache["class_labels"]]
    feature_sets = _stage2_feature_sets(class_labels, context_radius=context_radius)
    stage1_folds = list(stage1_cache["stage1_folds"])

    fold_metrics_by_variant: dict[str, list[dict[str, object]]] = {
        "raw": [dict(row) for row in stage1_cache["detail_by_variant"]["raw"][0].to_dict("records")],
        "hsmm": [dict(row) for row in stage1_cache["detail_by_variant"]["hsmm"][0].to_dict("records")],
        **{name: [] for name in feature_sets},
    }
    pred_frames_by_variant: dict[str, list[pd.DataFrame]] = {
        "raw": [stage1_cache["detail_by_variant"]["raw"][1].copy()],
        "hsmm": [stage1_cache["detail_by_variant"]["hsmm"][1].copy()],
        **{name: [] for name in feature_sets},
    }

    for fold_payload in stage1_folds:
        fold_index = int(fold_payload["fold_index"])
        calib_pred_frame = fold_payload["calib_pred_frame"].copy()
        test_pred_frame = fold_payload["test_pred_frame"].copy()
        context_train = _build_context_prob_frame(
            calib_pred_frame,
            class_labels=class_labels,
            subject_column=subject_column,
            time_column="time_seconds",
            context_radius=context_radius,
        )
        context_test = _build_context_prob_frame(
            test_pred_frame,
            class_labels=class_labels,
            subject_column=subject_column,
            time_column="time_seconds",
            context_radius=context_radius,
        )

        for variant_name, stage2_columns in feature_sets.items():
            model_family = "linear" if "linear" in variant_name else "ensemble"
            stage2 = build_model(
                model_family=model_family,
                task="classification",
                random_state=random_state,
                class_weight="balanced",
            )
            stage2.fit(context_train[stage2_columns], context_train["y_true"])
            context_test_pred = np.asarray(stage2.predict(context_test[stage2_columns]))
            refined_pred_frame = _apply_stage2_predictions(
                test_pred_frame,
                context_test,
                variant_name=variant_name,
                y_pred_context=context_test_pred,
                subject_column=subject_column,
                time_column="time_seconds",
            )
            pred_frames_by_variant[variant_name].append(refined_pred_frame)
            metrics = score_predictions(
                y_true=refined_pred_frame["y_true"].to_numpy(),
                y_pred=refined_pred_frame["y_pred"].to_numpy(),
                proba=None,
            )
            metrics.update(
                {
                    "fold_index": fold_index,
                    "n_train": int(fold_payload["n_train"]),
                    "n_test": int(fold_payload["n_test"]),
                    "n_features_used": int(fold_payload["n_features_used"]),
                    "n_subjects_train": int(fold_payload["n_subjects_train"]),
                    "n_subjects_test": int(fold_payload["n_subjects_test"]),
                    "n_subjects_validation": int(fold_payload["n_subjects_validation"]),
                    "n_stage2_rows_train": int(len(context_train)),
                    "n_stage2_rows_test": int(len(context_test)),
                    "n_stage2_features": int(len(stage2_columns)),
                }
            )
            fold_metrics_by_variant[variant_name].append(metrics)

        if log_path is not None:
            context_macro = fold_metrics_by_variant["context_linear"][-1].get("macro_f1")
            _log(
                f"{task_name} radius={context_radius} fold={fold_index}/{len(stage1_folds)} "
                f"context_linear_macro_f1={context_macro}",
                log_path=log_path,
            )

    summary_by_variant = {
        name: _summarize_fold_metrics(pd.DataFrame(rows), eval_source=name)
        for name, rows in fold_metrics_by_variant.items()
    }
    preds_by_variant = {
        name: pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in pred_frames_by_variant.items()
    }
    folds_by_variant = {
        name: pd.DataFrame(rows)
        for name, rows in fold_metrics_by_variant.items()
    }
    return summary_by_variant, {name: (folds_by_variant[name], preds_by_variant[name]) for name in folds_by_variant}


def main() -> int:
    args = _parse_args()
    repo_root = REPO_ROOT
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import (
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
    stage2_train_probs = resolve_stage2_train_probs(args.stage2_train_probs, args.stage1_model_family)
    if args.inner_validation != "none":
        stage2_train_probs = "inner_validation"
    # The default (bagged trees, in-sample) keeps the existing output path; other setups get their own folder.
    if (args.stage1_model_family, stage2_train_probs) != ("ensemble", "insample"):
        split_tag = f"{split_tag}__{args.stage1_model_family}_{stage2_train_probs}"
    if args.datasets != ["ucddb"]:
        split_tag = f"{split_tag}__{'+'.join(args.datasets)}"
    log_path = log_dir / f"paper_matched_stage2_smoothers__{split_tag}.log"
    status_path = log_dir / f"paper_matched_stage2_smoothers__{split_tag}_status.json"

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
    staging_table = build_staging_benchmark_table(
        table,
        feature_columns=specs["staging_plus_shared"].feature_columns,
    )
    frame = _prepare_frame(
        staging_table,
        subject_column="subject_id",
        target_column="label.stage_30s",
        feature_columns=list(specs["staging_plus_shared"].feature_columns),
    )
    norm_map = _load_norm_map(
        repo_root / "artifacts" / "reference_diagnostics_by_task" / "staging" / "best_feature_norms.parquet"
    )
    out_root = repo_root / "artifacts" / "paper_matched_stage2_smoothers" / split_tag
    out_root.mkdir(parents=True, exist_ok=True)
    overall_status: dict[str, object] = {"status": "running", "tasks": {}}
    status_path.write_text(json.dumps(overall_status, indent=2), encoding="utf-8")

    task_dir = out_root / "staging_30s"
    task_dir.mkdir(parents=True, exist_ok=True)
    all_summary_rows: list[dict[str, object]] = []
    all_fold_rows: list[pd.DataFrame] = []
    overall_status["tasks"]["staging_30s"] = {}

    _log("stage1_cache_start", log_path=log_path)
    stage1_cache = _fit_stage1_cache(
        frame,
        task_name="staging_30s",
        subject_column="subject_id",
        target_column="label.stage_30s",
        feature_columns=list(specs["staging_plus_shared"].feature_columns),
        norm_map=norm_map,
        context_radius=min(sorted(set(int(value) for value in args.context_radii))),
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
        random_state=args.random_state,
        log_path=log_path,
        stage1_model_family=args.stage1_model_family,
        stage2_train_probs=stage2_train_probs,
        oof_folds=args.oof_folds,
    )
    _log("stage1_cache_done", log_path=log_path)

    for context_radius in sorted(set(int(value) for value in args.context_radii)):
        radius_tag = f"ctx{(2 * context_radius) + 1}"
        _log(f"run_context_radius_start radius={context_radius} tag={radius_tag}", log_path=log_path)
        summary_by_variant, detail_by_variant = _run_task_from_stage1_cache(
            stage1_cache,
            task_name=f"staging_30s_{radius_tag}",
            subject_column="subject_id",
            context_radius=context_radius,
            random_state=args.random_state,
            log_path=log_path,
        )
        radius_dir = task_dir / radius_tag
        radius_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "target_column": "label.stage_30s",
            "feature_count": len(specs["staging_plus_shared"].feature_columns),
            "row_count": int(len(frame)),
            "n_subjects": int(frame["subject_id"].nunique()),
            "outer_validation": args.outer_validation,
            "outer_folds": args.outer_folds,
            "inner_validation": args.inner_validation,
            "stage1_model_family": args.stage1_model_family,
            "stage1_class_weight": "balanced",
            "stage2_train_probs": stage2_train_probs,
            "oof_folds": int(args.oof_folds) if stage2_train_probs == "oof" else None,
            "context_radius": context_radius,
            "variants": list(summary_by_variant.keys()),
        }
        (radius_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        for variant_name, summary_df in summary_by_variant.items():
            summary_df.to_parquet(radius_dir / f"summary_{variant_name}.parquet", index=False)
            fold_df, pred_df = detail_by_variant[variant_name]
            fold_df.to_parquet(radius_dir / f"fold_metrics_{variant_name}.parquet", index=False)
            pred_df.to_parquet(radius_dir / f"predictions_{variant_name}.parquet", index=False)
            if not fold_df.empty:
                fold_out = fold_df.copy()
                fold_out["variant"] = variant_name
                fold_out["context_radius"] = int(context_radius)
                all_fold_rows.append(fold_out)
            if not summary_df.empty:
                summary_row = summary_df.iloc[0].to_dict()
                summary_row["variant"] = variant_name
                summary_row["context_radius"] = int(context_radius)
                all_summary_rows.append(summary_row)
                overall_status["tasks"]["staging_30s"][f"{radius_tag}.{variant_name}"] = summary_row

        _log(f"run_context_radius_done radius={context_radius} tag={radius_tag}", log_path=log_path)

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_parquet(task_dir / "summary_all.parquet", index=False)
    if all_fold_rows:
        pd.concat(all_fold_rows, axis=0, ignore_index=True).to_parquet(task_dir / "fold_metrics_all.parquet", index=False)

    overall_status["status"] = "completed"
    overall_status["output_dir"] = str(out_root)
    status_path.write_text(json.dumps(overall_status, indent=2, default=str), encoding="utf-8")
    _log(f"run_done output_dir={out_root}", log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
