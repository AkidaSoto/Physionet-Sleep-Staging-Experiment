from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score


STAGE_NAME_MAP = {
    0: "wake",
    1: "n1",
    2: "n2",
    3: "n3",
    4: "rem",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        default="artifacts/paper_matched_stage2_smoothers/staging_30s",
    )
    parser.add_argument("--output-dir", default="artifacts/staging_diagnostics")
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-validation", choices=("none", "single_subject"), default="none")
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def _split_tag(*, outer_validation: str, outer_folds: int, inner_validation: str) -> str:
    outer_part = f"{outer_validation}{outer_folds}" if outer_validation == "kfold" else outer_validation
    return f"{outer_part}__{inner_validation}"


def _find_predictions_dir(repo_root: Path, requested: str) -> Path:
    direct = repo_root / requested
    if direct.exists():
        return direct
    split_candidates = sorted((repo_root / "artifacts" / "paper_matched_stage2_smoothers").glob("*/staging_30s"))
    if split_candidates:
        return split_candidates[-1]
    raise FileNotFoundError(f"Predictions dir not found: {direct}")


def _variant_name_from_path(path: Path) -> str:
    return path.stem.removeprefix("predictions_")


def _compute_overall_rows(pred_frame: pd.DataFrame, *, variant_name: str) -> pd.DataFrame:
    report = classification_report(
        pred_frame["y_true"].to_numpy(),
        pred_frame["y_pred"].to_numpy(),
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, object]] = []
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        class_id = int(float(label))
        rows.append(
            {
                "variant": variant_name,
                "stage_id": class_id,
                "stage_name": STAGE_NAME_MAP.get(class_id, str(class_id)),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )
    return pd.DataFrame(rows)


def _compute_confusion_rows(pred_frame: pd.DataFrame, *, variant_name: str) -> pd.DataFrame:
    labels = sorted(int(value) for value in np.unique(pred_frame["y_true"]))
    matrix = confusion_matrix(pred_frame["y_true"], pred_frame["y_pred"], labels=labels)
    rows: list[dict[str, object]] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append(
                {
                    "variant": variant_name,
                    "y_true": int(true_label),
                    "y_true_name": STAGE_NAME_MAP.get(int(true_label), str(true_label)),
                    "y_pred": int(pred_label),
                    "y_pred_name": STAGE_NAME_MAP.get(int(pred_label), str(pred_label)),
                    "count": int(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def _compute_subject_rows(pred_frame: pd.DataFrame, *, variant_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, group in pred_frame.groupby("subject_id", sort=False):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        rows.append(
            {
                "variant": variant_name,
                "subject_id": str(subject_id),
                "n_rows": int(len(group)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            }
        )
    return pd.DataFrame(rows)


def _compute_transition_rows(pred_frame: pd.DataFrame, *, variant_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, group in pred_frame.groupby("subject_id", sort=False):
        ordered = group.sort_values("time_seconds", kind="stable")
        true_values = ordered["y_true"].to_numpy()
        pred_values = ordered["y_pred"].to_numpy()
        times = ordered["time_seconds"].to_numpy(dtype=float)
        if len(true_values) < 2:
            continue
        for idx in range(1, len(true_values)):
            if not np.isfinite(times[idx] - times[idx - 1]) or abs((times[idx] - times[idx - 1]) - 30.0) > 0.25:
                continue
            rows.append(
                {
                    "variant": variant_name,
                    "subject_id": str(subject_id),
                    "true_from": int(true_values[idx - 1]),
                    "true_to": int(true_values[idx]),
                    "pred_from": int(pred_values[idx - 1]),
                    "pred_to": int(pred_values[idx]),
                    "true_changed": bool(true_values[idx - 1] != true_values[idx]),
                    "pred_changed": bool(pred_values[idx - 1] != pred_values[idx]),
                    "transition_correct": bool(
                        true_values[idx - 1] == pred_values[idx - 1] and true_values[idx] == pred_values[idx]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _group_feature_name(feature_name: str) -> str:
    return feature_name.split(".", 1)[0]


def _mean_bagging_feature_importance(model, feature_names: list[str]) -> np.ndarray:
    estimators = list(getattr(model, "estimators_", []))
    if not estimators:
        return np.zeros(len(feature_names), dtype=float)
    importances: list[np.ndarray] = []
    for estimator in estimators:
        values = getattr(estimator, "feature_importances_", None)
        if values is not None:
            importances.append(np.asarray(values, dtype=float))
    if not importances:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            return np.zeros(len(feature_names), dtype=float)
        return np.asarray(values, dtype=float)
    return np.mean(np.vstack(importances), axis=0)


def _compute_stage1_importances(
    repo_root: Path,
    *,
    outer_validation: str,
    outer_folds: int,
    inner_validation: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(repo_root / "src"))
    from physionet_sleep.analysis import build_staging_benchmark_table, load_ucddb_reference_dataset
    from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    specs = build_default_experiment_specs(table)
    feature_columns = list(specs["staging_plus_shared"].feature_columns)
    benchmark = build_staging_benchmark_table(table, feature_columns=feature_columns)
    frame = benchmark[["subject_id", "time_seconds", "label.stage_30s", *feature_columns]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["subject_id", "label.stage_30s"]).copy()
    frame["label.stage_30s"] = pd.to_numeric(frame["label.stage_30s"], errors="coerce")
    frame = frame.dropna(subset=["label.stage_30s"]).copy()
    frame["label.stage_30s"] = frame["label.stage_30s"].astype(int)

    groups = frame["subject_id"].astype(str).to_numpy()
    y = frame["label.stage_30s"].to_numpy()
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
    fold_rows: list[dict[str, object]] = []
    for split in splits:
        train_df = frame.iloc[split.train_idx].copy()
        feature_names = [col for col in feature_columns if train_df[col].notna().any()]
        estimator = build_model(
            model_family="ensemble",
            task="classification",
            random_state=random_state,
            class_weight="balanced",
        )
        estimator.fit(train_df[feature_names], train_df["label.stage_30s"])
        model = estimator.named_steps["model"]
        values = _mean_bagging_feature_importance(model, feature_names)
        for feature_name, importance in zip(feature_names, values.tolist(), strict=False):
            fold_rows.append(
                {
                    "fold_index": split.fold_index,
                    "feature_name": feature_name,
                    "feature_group": _group_feature_name(feature_name),
                    "importance": float(importance),
                }
            )
    feature_df = pd.DataFrame(fold_rows)
    family_df = (
        feature_df.groupby(["feature_group", "fold_index"], as_index=False)["importance"].sum()
        .groupby("feature_group", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_median=("importance", "median"),
        )
    )
    feature_summary = feature_df.groupby("feature_name", as_index=False).agg(
        importance_mean=("importance", "mean"),
        importance_median=("importance", "median"),
    )
    return feature_summary, family_df


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    predictions_dir = _find_predictions_dir(repo_root, args.predictions_dir)
    output_dir = repo_root / args.output_dir
    split_tag = _split_tag(
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
    )
    out_root = output_dir / split_tag
    out_root.mkdir(parents=True, exist_ok=True)

    prediction_paths = sorted(predictions_dir.glob("predictions_*.parquet"))
    if not prediction_paths:
        raise FileNotFoundError(f"No predictions_*.parquet files found in {predictions_dir}")

    overall_rows: list[pd.DataFrame] = []
    confusion_rows: list[pd.DataFrame] = []
    subject_rows: list[pd.DataFrame] = []
    transition_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for path in prediction_paths:
        variant = _variant_name_from_path(path)
        pred_frame = pd.read_parquet(path)
        pred_frame = pred_frame.dropna(subset=["y_true", "y_pred"]).copy()
        pred_frame["y_true"] = pred_frame["y_true"].astype(int)
        pred_frame["y_pred"] = pred_frame["y_pred"].astype(int)
        summary_rows.append(
            {
                "variant": variant,
                "accuracy": float(accuracy_score(pred_frame["y_true"], pred_frame["y_pred"])),
                "macro_f1": float(f1_score(pred_frame["y_true"], pred_frame["y_pred"], average="macro", zero_division=0)),
                "cohen_kappa": float(cohen_kappa_score(pred_frame["y_true"], pred_frame["y_pred"])),
                "n_rows": int(len(pred_frame)),
                "n_subjects": int(pred_frame["subject_id"].nunique()),
            }
        )
        overall_rows.append(_compute_overall_rows(pred_frame, variant_name=variant))
        confusion_rows.append(_compute_confusion_rows(pred_frame, variant_name=variant))
        subject_rows.append(_compute_subject_rows(pred_frame, variant_name=variant))
        transition_rows.append(_compute_transition_rows(pred_frame, variant_name=variant))

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["macro_f1", "cohen_kappa"], ascending=False)
    overall_df = pd.concat(overall_rows, axis=0, ignore_index=True)
    confusion_df = pd.concat(confusion_rows, axis=0, ignore_index=True)
    subject_df = pd.concat(subject_rows, axis=0, ignore_index=True)
    transition_df = pd.concat(transition_rows, axis=0, ignore_index=True)

    transition_summary = (
        transition_df.groupby("variant", as_index=False)
        .agg(
            transition_accuracy=("transition_correct", "mean"),
            true_change_rate=("true_changed", "mean"),
            pred_change_rate=("pred_changed", "mean"),
            n_transitions=("transition_correct", "size"),
        )
    )
    worst_subjects = (
        subject_df.sort_values(by=["variant", "macro_f1", "cohen_kappa"], ascending=[True, True, True])
        .groupby("variant", as_index=False)
        .head(5)
    )

    feature_importance_df, family_importance_df = _compute_stage1_importances(
        repo_root,
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
        random_state=args.random_state,
    )

    summary_df.to_parquet(out_root / "variant_summary.parquet", index=False)
    overall_df.to_parquet(out_root / "per_stage_metrics.parquet", index=False)
    confusion_df.to_parquet(out_root / "confusion_matrix_long.parquet", index=False)
    subject_df.to_parquet(out_root / "subject_metrics.parquet", index=False)
    worst_subjects.to_parquet(out_root / "worst_subjects.parquet", index=False)
    transition_df.to_parquet(out_root / "transition_rows.parquet", index=False)
    transition_summary.to_parquet(out_root / "transition_summary.parquet", index=False)
    feature_importance_df.to_parquet(out_root / "stage1_feature_importance.parquet", index=False)
    family_importance_df.to_parquet(out_root / "stage1_feature_family_importance.parquet", index=False)

    report = {
        "predictions_dir": str(predictions_dir),
        "output_dir": str(out_root),
        "best_variant": summary_df.iloc[0].to_dict() if not summary_df.empty else {},
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
