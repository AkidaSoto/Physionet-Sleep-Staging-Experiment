from __future__ import annotations

import argparse
import json
import sys
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
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/staging_sweeps")
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-validation", choices=("none", "single_subject"), default="none")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=(
            "staging_interpretable",
            "staging_core",
            "staging_plus_shared",
            "staging_trim_conservative",
            "staging_trim_signal",
            "eeg_only",
            "eeg_eog",
        ),
    )
    parser.add_argument(
        "--model-families",
        nargs="+",
        default=("ensemble", "random_forest", "extra_trees"),
    )
    parser.add_argument("--context-radii", nargs="+", type=int, default=(1, 2, 3))
    return parser.parse_args()


def _split_tag(*, outer_validation: str, outer_folds: int, inner_validation: str) -> str:
    return build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
    )


def _load_norm_map(path: Path) -> dict[str, str]:
    return load_selected_norm_map(path)


def _apply_featurewise_best_norm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_map: dict[str, str],
    subject_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_featurewise_norm_map(
        train_df,
        test_df,
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


def _sequence_arrays(pred_frame: pd.DataFrame, *, subject_column: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return ordered_prediction_sequences(pred_frame, subject_column=subject_column)


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _build_context_prob_frame(
    pred_frame: pd.DataFrame,
    *,
    class_labels: list[int],
    context_radius: int,
    subject_column: str = "subject_id",
) -> pd.DataFrame:
    return build_probability_context_frame(
        pred_frame,
        class_labels=class_labels,
        subject_column=subject_column,
        context_radius=context_radius,
    )


def _context_feature_columns(class_labels: list[int], *, context_radius: int) -> list[str]:
    cols: list[str] = []
    for offset in range(-context_radius, context_radius + 1):
        label = _offset_label(offset)
        cols.extend([f"{label}.proba_{class_id}" for class_id in class_labels])
    return cols


def _predict_with_stage2(
    stage1_test_frame: pd.DataFrame,
    context_test: pd.DataFrame,
    *,
    y_pred_context: np.ndarray,
    subject_column: str,
) -> pd.DataFrame:
    return apply_prediction_updates(
        stage1_test_frame,
        context_test,
        y_pred_values=np.asarray(y_pred_context),
        subject_column=subject_column,
        time_column="time_seconds",
        stage1_column="y_pred_stage1",
    )


def _match_prefixes(columns: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [col for col in columns if col.startswith(prefixes)]


def _build_feature_sets(all_columns: list[str]) -> dict[str, list[str]]:
    eeg_cols = _match_prefixes(all_columns, ("spindle_features.", "slow_wave_features.", "eeg_arousal_features."))
    eog_cols = _match_prefixes(all_columns, ("eye_movement_activity.", "eog_rem_features."))
    emg_cols = _match_prefixes(all_columns, ("emg_tone_features.",))
    shared_cols = _match_prefixes(all_columns, ("pan_tompkins.",))
    dead_or_low_yield = {
        "eeg_arousal_features.eeg_arousal_signal",
        "eog_rem_features.rem_variance_epoch",
        "eog_rem_features.rem_power_epoch",
        "eog_rem_features.sem_power_epoch",
        "eog_rem_features.noise_power_epoch",
        "pan_tompkins.ecg_filtered",
    }
    signal_trim_keep = {
        "slow_wave_features.slow_wave_envelope",
        "slow_wave_features.high_frequency_envelope",
        "slow_wave_features.slow_wave_index",
        "slow_wave_features.slow_wave_ratio_power",
        "eye_movement_activity.eye_movement_activity",
        "eye_movement_activity.eye_movement_baseline",
        "eog_rem_features.rem_band_power_fast",
        "eog_rem_features.sem_band_power_fast",
        "eog_rem_features.noise_band_power_fast",
        "emg_tone_features.emg_tone_baseline",
        "emg_tone_features.emg_tone",
        "spindle_features.aperiodic_exponent",
        "spindle_features.aperiodic_intercept",
        "spindle_features.sigma_residual_power",
        "spindle_features.sigma_beta_ratio",
        "spindle_features.sigma_power",
        "spindle_features.splindex",
        "pan_tompkins.heart_rate_bpm_track",
        "pan_tompkins.hrv_rmssd_track",
        "pan_tompkins.hrv_sdnn_track",
        "pan_tompkins.heart_rate_min_track",
        "pan_tompkins.heart_rate_range_track",
    }
    interpretable_keep = {
        "slow_wave_features.slow_wave_envelope",
        "slow_wave_features.high_frequency_envelope",
        "spindle_features.aperiodic_exponent",
        "spindle_features.sigma_residual_power",
        "eye_movement_activity.eye_movement_activity",
        "eog_rem_features.rem_band_power_fast",
        "eog_rem_features.sem_band_power_fast",
        "emg_tone_features.emg_tone_baseline",
        "pan_tompkins.heart_rate_bpm_track",
        "pan_tompkins.hrv_rmssd_track",
        "respiration.respiration_rate_bpm",
        "respiration.respiration_envelope",
    }
    staging_core = list(dict.fromkeys(eeg_cols + eog_cols + emg_cols))
    staging_plus_shared = list(dict.fromkeys(staging_core + shared_cols))
    staging_trim_conservative = [col for col in staging_plus_shared if col not in dead_or_low_yield]
    staging_trim_signal = [col for col in staging_plus_shared if col in signal_trim_keep]
    sets = {
        "eeg_only": eeg_cols,
        "eeg_eog": list(dict.fromkeys(eeg_cols + eog_cols)),
        "staging_core": staging_core,
        "staging_plus_shared": staging_plus_shared,
        "staging_interpretable": [col for col in staging_plus_shared if col in interpretable_keep],
        "staging_trim_conservative": staging_trim_conservative,
        "staging_trim_signal": staging_trim_signal,
    }
    return {name: cols for name, cols in sets.items() if cols}


def _summarize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [col for col in df.columns if col not in {"feature_set", "model_family", "variant", "context_radius", "fold_index"}]
    summary_rows: list[dict[str, object]] = []
    for keys, group in df.groupby(["feature_set", "model_family", "variant", "context_radius"], sort=False):
        row = {
            "feature_set": keys[0],
            "model_family": keys[1],
            "variant": keys[2],
            "context_radius": keys[3],
            "n_folds": int(len(group)),
        }
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(group[col]):
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_median"] = float(group[col].median())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(
        by=["macro_f1_mean", "cohen_kappa_mean", "accuracy_mean"],
        ascending=False,
        na_position="last",
    )


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import build_staging_benchmark_table, load_ucddb_reference_dataset
    from physionet_sleep.decoders.hsmm import apply_hsmm_viterbi, learn_hsmm_params
    from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits, score_predictions
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    specs = build_default_experiment_specs(table)
    full_feature_columns = list(
        dict.fromkeys(
            col
            for spec in specs.values()
            if spec.target_column == "label.stage_seconds"
            for col in spec.feature_columns
        )
    )
    feature_set_map = _build_feature_sets(full_feature_columns)
    selected_feature_sets = {name: feature_set_map[name] for name in args.feature_sets if name in feature_set_map}
    if not selected_feature_sets:
        raise ValueError(f"No valid feature sets selected from: {args.feature_sets}")

    benchmark = build_staging_benchmark_table(table, feature_columns=full_feature_columns)
    groups = benchmark["subject_id"].astype(str).to_numpy()
    y = pd.to_numeric(benchmark["label.stage_30s"], errors="coerce").fillna(-1).astype(int).to_numpy()
    class_labels = sorted(int(value) for value in np.unique(y[y >= 0]))
    splits = list(
        build_subject_eval_splits(
            groups=groups,
            y=y,
            outer_validation=args.outer_validation,
            outer_folds=args.outer_folds,
            inner_validation=args.inner_validation,
            random_state=args.random_state,
        )
    )
    norm_map = _load_norm_map(repo_root / "artifacts" / "reference_diagnostics_by_task" / "staging" / "best_feature_norms.parquet")

    fold_rows: list[dict[str, object]] = []
    split_tag = _split_tag(
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
    )
    out_root = repo_root / args.output_dir / split_tag
    out_root.mkdir(parents=True, exist_ok=True)

    for feature_set_name, feature_columns in selected_feature_sets.items():
        frame = _prepare_frame(
            benchmark,
            subject_column="subject_id",
            target_column="label.stage_30s",
            feature_columns=feature_columns,
        )
        for model_family in args.model_families:
            for split in splits:
                train_df = frame.iloc[split.train_idx].copy()
                test_df = frame.iloc[split.test_idx].copy()
                calib_df = frame.iloc[split.validation_idx].copy() if split.validation_idx is not None else None
                train_df, test_df = _apply_featurewise_best_norm(
                    train_df,
                    test_df,
                    feature_columns=feature_columns,
                    norm_map=norm_map,
                    subject_column="subject_id",
                )
                if calib_df is not None:
                    train_df, calib_df = _apply_featurewise_best_norm(
                        train_df,
                        calib_df,
                        feature_columns=feature_columns,
                        norm_map=norm_map,
                        subject_column="subject_id",
                    )
                usable = [col for col in feature_columns if train_df[col].notna().any()]
                estimator = build_model(
                    model_family=model_family,
                    task="classification",
                    random_state=args.random_state,
                    class_weight="balanced",
                )
                estimator.fit(train_df[usable], train_df["label.stage_30s"])
                train_pred = np.asarray(estimator.predict(train_df[usable]))
                test_pred = np.asarray(estimator.predict(test_df[usable]))
                train_proba_raw = estimator.predict_proba(train_df[usable]) if hasattr(estimator, "predict_proba") else None
                test_proba_raw = estimator.predict_proba(test_df[usable]) if hasattr(estimator, "predict_proba") else None
                estimator_classes = np.asarray(getattr(estimator, "classes_", class_labels), dtype=int)
                train_aligned = _aligned_proba(train_proba_raw, estimator_classes=estimator_classes, class_labels=class_labels)
                test_aligned = _aligned_proba(test_proba_raw, estimator_classes=estimator_classes, class_labels=class_labels)

                raw_metrics = score_predictions(
                    y_true=test_df["label.stage_30s"].to_numpy(),
                    y_pred=test_pred,
                    proba=test_aligned,
                )
                fold_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model_family": model_family,
                        "variant": "raw",
                        "context_radius": 0,
                        "fold_index": split.fold_index,
                        **raw_metrics,
                    }
                )

                train_pred_frame = _attach_probabilities(
                    train_df[["subject_id", "time_seconds"]].assign(
                        y_true=train_df["label.stage_30s"].to_numpy(),
                        y_pred=train_pred,
                    ),
                    class_labels=class_labels,
                    aligned_proba=train_aligned,
                )
                test_pred_frame = _attach_probabilities(
                    test_df[["subject_id", "time_seconds"]].assign(
                        y_true=test_df["label.stage_30s"].to_numpy(),
                        y_pred=test_pred,
                    ),
                    class_labels=class_labels,
                    aligned_proba=test_aligned,
                )
                calib_pred_frame = train_pred_frame
                if calib_df is not None:
                    calib_pred = np.asarray(estimator.predict(calib_df[usable]))
                    calib_proba_raw = estimator.predict_proba(calib_df[usable]) if hasattr(estimator, "predict_proba") else None
                    calib_aligned = _aligned_proba(calib_proba_raw, estimator_classes=estimator_classes, class_labels=class_labels)
                    calib_pred_frame = _attach_probabilities(
                        calib_df[["subject_id", "time_seconds"]].assign(
                            y_true=calib_df["label.stage_30s"].to_numpy(),
                            y_pred=calib_pred,
                        ),
                        class_labels=class_labels,
                        aligned_proba=calib_aligned,
                    )

                hsmm_params = learn_hsmm_params(*_sequence_arrays(calib_pred_frame, subject_column="subject_id"))
                test_seq_true, test_seq_pred = _sequence_arrays(test_pred_frame, subject_column="subject_id")
                test_seq_hsmm = apply_hsmm_viterbi(test_seq_pred, hsmm_params)
                hsmm_flat = np.concatenate([np.asarray(seq) for seq in test_seq_hsmm if len(seq)], axis=0)
                true_flat = np.concatenate([np.asarray(seq) for seq in test_seq_true if len(seq)], axis=0)
                hsmm_metrics = score_predictions(y_true=true_flat, y_pred=hsmm_flat, proba=None)
                fold_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model_family": model_family,
                        "variant": "hsmm",
                        "context_radius": 0,
                        "fold_index": split.fold_index,
                        **hsmm_metrics,
                    }
                )

                for context_radius in args.context_radii:
                    context_train = _build_context_prob_frame(
                        calib_pred_frame,
                        class_labels=class_labels,
                        context_radius=context_radius,
                    )
                    context_test = _build_context_prob_frame(
                        test_pred_frame,
                        class_labels=class_labels,
                        context_radius=context_radius,
                    )
                    if context_train.empty or context_test.empty:
                        continue
                    context_cols = _context_feature_columns(class_labels, context_radius=context_radius)
                    for variant_name, stage2_family in (
                        ("context_linear", "linear"),
                        ("context_ensemble", "ensemble"),
                    ):
                        stage2 = build_model(
                            model_family=stage2_family,
                            task="classification",
                            random_state=args.random_state,
                            class_weight="balanced",
                        )
                        stage2.fit(context_train[context_cols], context_train["y_true"])
                        stage2_pred = np.asarray(stage2.predict(context_test[context_cols]))
                        refined = _predict_with_stage2(
                            test_pred_frame,
                            context_test,
                            y_pred_context=stage2_pred,
                            subject_column="subject_id",
                        )
                        stage2_metrics = score_predictions(
                            y_true=refined["y_true"].to_numpy(),
                            y_pred=refined["y_pred"].to_numpy(),
                            proba=None,
                        )
                        fold_rows.append(
                            {
                                "feature_set": feature_set_name,
                                "model_family": model_family,
                                "variant": variant_name,
                                "context_radius": int(context_radius),
                                "fold_index": split.fold_index,
                                **stage2_metrics,
                            }
                        )
                print(
                    f"{feature_set_name} {model_family} fold={split.fold_index}/{len(splits)} raw_macro_f1={raw_metrics.get('macro_f1')}",
                    flush=True,
                )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = _summarize_rows(fold_rows)
    fold_df.to_parquet(out_root / "fold_metrics.parquet", index=False)
    summary_df.to_parquet(out_root / "summary.parquet", index=False)
    report = {
        "output_dir": str(out_root),
        "best_row": summary_df.iloc[0].to_dict() if not summary_df.empty else {},
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
