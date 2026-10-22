"""Family-level permutation importance for the epoch-only HGB staging model.

For each outer fold (subject-grouped, same splits as the dashboard), fit stage 1 on the training subjects, then for
each feature family shuffle that family's columns together across the held-out epochs and record the drop in
macro-F1 and per-class F1. Reported as mean over folds and repeats.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.analysis import build_staging_benchmark_table, load_ucddb_reference_dataset
from physionet_sleep.experiments.runner_utils import apply_featurewise_norm_map, load_selected_norm_map, prepare_model_frame
from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits
from physionet_sleep.experiments.tabular_specs import STAGING_FEATURE_FAMILIES, TOPOGRAPHY_FAMILY, build_default_experiment_specs

TARGET = "label.stage_30s"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/staging_diagnostics/kfold5__none__hgb")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucddb"],
        help="Datasets from physionet_sleep.analysis.datasets to use; non-default runs get their own output folder.",
    )
    args = parser.parse_args()
    if args.datasets != ["ucddb"]:
        args.output_dir = f"{args.output_dir}__{'+'.join(args.datasets)}"

    if args.datasets == ["ucddb"]:
        table = load_ucddb_reference_dataset(
            cache_dir=REPO_ROOT / "artifacts" / "ucddb_recipe_cache",
            root_dir=REPO_ROOT / "data" / "raw" / "ucddb",
        )
    else:
        from physionet_sleep.analysis.datasets import load_reference_dataset

        table = load_reference_dataset(args.datasets)
    features = list(build_default_experiment_specs(table)["staging_plus_shared"].feature_columns)
    frame = prepare_model_frame(
        build_staging_benchmark_table(table, feature_columns=features),
        subject_column="subject_id",
        target_column=TARGET,
        feature_columns=features,
        drop_negative_targets=True,
    )
    norm_map = load_selected_norm_map(
        REPO_ROOT / "artifacts" / "reference_diagnostics_by_task" / "staging" / "best_feature_norms.parquet"
    )
    splits = build_subject_eval_splits(
        groups=frame["subject_id"].astype(str).to_numpy(),
        y=frame[TARGET].to_numpy(),
        outer_validation="kfold",
        outer_folds=5,
        random_state=args.random_state,
        inner_validation="none",
    )
    families = [
        (group["label"], [f for f in group["feature_ids"] if f in features])
        for group in [*STAGING_FEATURE_FAMILIES, TOPOGRAPHY_FAMILY]
        if any(f in features for f in group["feature_ids"])
    ]
    rng = np.random.default_rng(args.random_state)
    rows = []
    for split in splits:
        train, test = apply_featurewise_norm_map(
            frame.iloc[split.train_idx].copy(),
            frame.iloc[split.test_idx].copy(),
            feature_columns=features,
            norm_map=norm_map,
            subject_column="subject_id",
        )
        model = build_model(model_family="hgb", task="classification", random_state=args.random_state)
        model.fit(train[features], train[TARGET])
        y_true = test[TARGET].to_numpy()
        base_pred = model.predict(test[features])
        base_macro = f1_score(y_true, base_pred, average="macro")
        base_class = f1_score(y_true, base_pred, labels=range(5), average=None, zero_division=0)
        for label, cols in families:
            for repeat in range(args.repeats):
                shuffled = test[features].copy()
                order = rng.permutation(len(shuffled))
                shuffled[cols] = shuffled[cols].to_numpy()[order]
                pred = model.predict(shuffled)
                class_f1 = f1_score(y_true, pred, labels=range(5), average=None, zero_division=0)
                rows.append(
                    {
                        "fold_index": split.fold_index,
                        "repeat": repeat,
                        "family": label,
                        "n_features": len(cols),
                        "macro_f1_drop": base_macro - f1_score(y_true, pred, average="macro"),
                        **{f"f1_drop_{stage}": base_class[i] - class_f1[i] for i, stage in enumerate(STAGES)},
                    }
                )
        print(f"fold {split.fold_index} done (base macro-F1 {base_macro:.3f})", flush=True)

    result = pd.DataFrame(rows)
    summary = (
        result.groupby(["family", "n_features"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["fold_index", "repeat"])
        .sort_values("macro_f1_drop", ascending=False)
    )
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir / "family_permutation_importance_rows.parquet", index=False)
    summary.to_parquet(out_dir / "family_permutation_importance.parquet", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
