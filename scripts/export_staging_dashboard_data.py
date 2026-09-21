from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
GENERATED = ROOT / "showcase" / "content" / "generated"
OUTPUT = GENERATED / "staging-dashboard.json"
FEATURE_EXAMPLES_OUTPUT = GENERATED / "staging-feature-examples.json"
CLINICAL_OUTPUT = GENERATED / "staging-clinical.json"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]

FEATURE_GROUPS = [
    {
        "id": "spindles",
        "label": "Spindles",
        "signal": "EEG",
        "feature_ids": [
            "spindle_features.splindex",
            "spindle_features.sigma_beta_ratio",
            "spindle_features.sigma_beta_residual_diff",
        ],
    },
    {
        "id": "spectrum",
        "label": "EEG spectrum",
        "signal": "EEG",
        "feature_ids": [
            "spindle_features.aperiodic_exponent",
            "spindle_features.aperiodic_intercept",
            *[
                f"spindle_features.{band}_{suffix}"
                for band in ("delta", "theta", "alpha", "sigma", "beta", "gamma")
                for suffix in ("power", "residual_power")
            ],
        ],
    },
    {
        "id": "slow-waves",
        "label": "Slow waves",
        "signal": "EEG",
        "feature_ids": [
            "slow_wave_features.slow_wave_envelope",
            "slow_wave_features.high_frequency_envelope",
            "slow_wave_features.slow_wave_raw_power",
            "slow_wave_features.slow_wave_ratio_power",
            "slow_wave_features.slow_wave_index",
        ],
    },
    {
        "id": "eye-movement",
        "label": "Eye movement",
        "signal": "EOG",
        "feature_ids": [
            "eye_movement_activity.eye_movement_activity",
            "eye_movement_activity.eye_movement_baseline",
            "eog_rem_features.rem_band_power_fast",
            "eog_rem_features.sem_band_power_fast",
            "eog_rem_features.noise_band_power_fast",
        ],
    },
    {
        "id": "muscle-tone",
        "label": "Muscle tone",
        "signal": "Chin EMG",
        "feature_ids": [
            "emg_tone_features.emg_tone",
            "emg_tone_features.emg_tone_baseline",
        ],
    },
    {
        "id": "cardiac",
        "label": "Cardiac",
        "signal": "ECG",
        "feature_ids": [
            "pan_tompkins.heart_rate_bpm_track",
            "pan_tompkins.hrv_rmssd_track",
            "pan_tompkins.hrv_sdnn_track",
            "pan_tompkins.heart_rate_min_track",
            "pan_tompkins.heart_rate_range_track",
        ],
    },
]

FEATURE_LABELS = {
    "spindle_features.splindex": "Spindle index",
    "spindle_features.sigma_beta_ratio": "Sigma / beta ratio",
    "spindle_features.sigma_beta_residual_diff": "Residual sigma–beta difference",
    "spindle_features.aperiodic_exponent": "Aperiodic exponent",
    "spindle_features.aperiodic_intercept": "Aperiodic intercept",
    "slow_wave_features.slow_wave_envelope": "Slow-wave envelope",
    "slow_wave_features.high_frequency_envelope": "High-frequency envelope",
    "slow_wave_features.slow_wave_raw_power": "Slow-wave raw power",
    "slow_wave_features.slow_wave_ratio_power": "Slow-wave power ratio",
    "slow_wave_features.slow_wave_index": "Slow-wave index",
    "eye_movement_activity.eye_movement_activity": "Eye-movement activity",
    "eye_movement_activity.eye_movement_baseline": "Eye-movement baseline",
    "eog_rem_features.rem_band_power_fast": "Rapid-eye band power",
    "eog_rem_features.sem_band_power_fast": "Slow-eye band power",
    "eog_rem_features.noise_band_power_fast": "EOG noise-band power",
    "emg_tone_features.emg_tone": "Chin EMG tone",
    "emg_tone_features.emg_tone_baseline": "Chin EMG baseline",
    "pan_tompkins.heart_rate_bpm_track": "Heart rate",
    "pan_tompkins.hrv_rmssd_track": "HRV RMSSD",
    "pan_tompkins.hrv_sdnn_track": "HRV SDNN",
    "pan_tompkins.heart_rate_min_track": "Minimum heart rate",
    "pan_tompkins.heart_rate_range_track": "Heart-rate range",
}
for _band in ("delta", "theta", "alpha", "sigma", "beta", "gamma"):
    FEATURE_LABELS[f"spindle_features.{_band}_power"] = f"{_band.title()} power"
    FEATURE_LABELS[f"spindle_features.{_band}_residual_power"] = f"{_band.title()} residual power"

FEATURE_IDS = [feature_id for group in FEATURE_GROUPS for feature_id in group["feature_ids"]]
FEATURE_SIGNAL = {
    feature_id: group["signal"]
    for group in FEATURE_GROUPS
    for feature_id in group["feature_ids"]
}

STAGE_FEATURE_TARGETS = {
    "Wake": [
        "slow_wave_features.high_frequency_envelope",
        "emg_tone_features.emg_tone_baseline",
        "eye_movement_activity.eye_movement_baseline",
        "pan_tompkins.heart_rate_bpm_track",
    ],
    "N1": [
        "spindle_features.aperiodic_exponent",
        "eye_movement_activity.eye_movement_activity",
        "eog_rem_features.sem_band_power_fast",
    ],
    "N2": [
        "spindle_features.splindex",
        "spindle_features.sigma_residual_power",
        "spindle_features.aperiodic_exponent",
        "pan_tompkins.hrv_rmssd_track",
    ],
    "N3": [
        "slow_wave_features.slow_wave_envelope",
        "slow_wave_features.slow_wave_index",
        "slow_wave_features.slow_wave_ratio_power",
        "slow_wave_features.high_frequency_envelope",
    ],
    "REM": [
        "eye_movement_activity.eye_movement_activity",
        "eog_rem_features.rem_band_power_fast",
        "eog_rem_features.sem_band_power_fast",
        "emg_tone_features.emg_tone_baseline",
    ],
}


def _summary(path: Path) -> dict[str, float]:
    row = pd.read_parquet(path).iloc[0]
    return {
        "accuracy": float(row["accuracy_mean"]),
        "balanced_accuracy": float(row["balanced_accuracy_mean"]),
        "macro_f1": float(row["macro_f1_mean"]),
        "cohen_kappa": float(row["cohen_kappa_mean"]),
    }


def _folds(path: Path) -> list[dict[str, float | int]]:
    frame = pd.read_parquet(path).sort_values("fold_index")
    return [
        {
            "fold": int(row.fold_index),
            "macro_f1": float(row.macro_f1),
            "accuracy": float(row.accuracy),
            "cohen_kappa": float(row.cohen_kappa),
        }
        for row in frame.itertuples(index=False)
    ]


def _model(
    *,
    model_id: str,
    name: str,
    role: str,
    hypothesis: str,
    summary_path: Path,
    folds_path: Path,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": name,
        "role": role,
        "hypothesis": hypothesis,
        "metrics": _summary(summary_path),
        "folds": _folds(folds_path),
    }


def _best_prediction_diagnostics(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path).sort_values(["subject_id", "time_seconds"])
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["y_pred"].astype(int).to_numpy()
    counts = confusion_matrix(y_true, y_pred, labels=range(len(STAGES)))
    normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(STAGES)),
        zero_division=0,
    )

    per_stage = [
        {
            "stage": STAGES[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(STAGES))
    ]

    subject_metrics = []
    for subject_id, subject_frame in frame.groupby("subject_id", sort=True):
        subject_true = subject_frame["y_true"].astype(int).to_numpy()
        subject_pred = subject_frame["y_pred"].astype(int).to_numpy()
        subject_metrics.append(
            {
                "subject": str(subject_id),
                "epochs": int(len(subject_frame)),
                "accuracy": float(accuracy_score(subject_true, subject_pred)),
                "macro_f1": float(
                    f1_score(
                        subject_true,
                        subject_pred,
                        labels=range(len(STAGES)),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "cohen_kappa": float(cohen_kappa_score(subject_true, subject_pred)),
            }
        )

    return {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "confusion_counts": counts.astype(int).tolist(),
        "confusion_normalized": normalized.tolist(),
        "per_stage": per_stage,
        "subject_metrics": subject_metrics,
    }


def _validation_folds(path: Path) -> list[dict[str, Any]]:
    membership = pd.read_parquet(
        path,
        columns=["subject_id", "fold_index"],
    ).drop_duplicates()
    all_subjects = sorted(membership["subject_id"].astype(str).unique().tolist())
    folds: list[dict[str, Any]] = []
    for fold_index, fold_frame in membership.groupby("fold_index", sort=True):
        test_subjects = sorted(fold_frame["subject_id"].astype(str).tolist())
        folds.append(
            {
                "fold": int(fold_index),
                "train_subjects": [subject for subject in all_subjects if subject not in test_subjects],
                "test_subjects": test_subjects,
            }
        )
    return folds


def _feature_matrix(prediction_path: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(prediction_path)[
        ["subject_id", "time_seconds", "y_true", "y_pred"]
    ].copy()
    rows: list[pd.DataFrame] = []

    for subject_id, subject_predictions in predictions.groupby("subject_id", sort=True):
        feature_path = (
            ARTIFACTS
            / "ucddb_recipe_cache"
            / str(subject_id)
            / "feature_table_py"
            / "artifacts"
            / "feature_table.parquet"
        )
        feature_frame = pd.read_parquet(
            feature_path,
            columns=["time_seconds", *FEATURE_IDS],
        )
        feature_frame["time_seconds"] = (
            np.floor(feature_frame["time_seconds"].astype(float) / 30.0) * 30.0
        )
        epoch_features = feature_frame.groupby("time_seconds", as_index=False)[
            FEATURE_IDS
        ].median()
        rows.append(
            subject_predictions.merge(epoch_features, on="time_seconds", how="left")
        )

    merged = pd.concat(rows, ignore_index=True).sort_values(
        ["subject_id", "time_seconds"]
    ).reset_index(drop=True)
    grouped = merged.groupby("subject_id", sort=False)
    previous_stage = grouped["y_true"].shift(1)
    next_stage = grouped["y_true"].shift(-1)
    previous_time = grouped["time_seconds"].shift(1)
    next_time = grouped["time_seconds"].shift(-1)
    merged["stable_stage"] = (
        previous_stage.eq(merged["y_true"])
        & next_stage.eq(merged["y_true"])
        & (merged["time_seconds"] - previous_time).eq(30.0)
        & (next_time - merged["time_seconds"]).eq(30.0)
    )
    return merged


def _cohens_d(target: pd.Series, rest: pd.Series) -> float:
    target_values = target.dropna().to_numpy(dtype=float)
    rest_values = rest.dropna().to_numpy(dtype=float)
    if target_values.size < 2 or rest_values.size < 2:
        return 0.0
    pooled_variance = (
        (target_values.size - 1) * np.var(target_values, ddof=1)
        + (rest_values.size - 1) * np.var(rest_values, ddof=1)
    ) / (target_values.size + rest_values.size - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return 0.0
    return float((np.mean(target_values) - np.mean(rest_values)) / np.sqrt(pooled_variance))


def _feature_distributions(merged: pd.DataFrame) -> list[dict[str, Any]]:
    distributions: list[dict[str, Any]] = []
    family_by_feature = {
        feature_id: group["id"]
        for group in FEATURE_GROUPS
        for feature_id in group["feature_ids"]
    }
    for feature_id in FEATURE_IDS:
        stage_rows = []
        for stage_index, stage in enumerate(STAGES):
            values = merged.loc[merged["y_true"] == stage_index, feature_id].dropna()
            rest = merged.loc[merged["y_true"] != stage_index, feature_id].dropna()
            quantiles = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            stage_rows.append(
                {
                    "stage": stage,
                    "count": int(len(values)),
                    "p10": float(quantiles.loc[0.10]),
                    "p25": float(quantiles.loc[0.25]),
                    "median": float(quantiles.loc[0.50]),
                    "p75": float(quantiles.loc[0.75]),
                    "p90": float(quantiles.loc[0.90]),
                    "one_vs_rest_cohens_d": _cohens_d(values, rest),
                }
            )
        distributions.append(
            {
                "id": feature_id,
                "family_id": family_by_feature[feature_id],
                "label": FEATURE_LABELS[feature_id],
                "signal": FEATURE_SIGNAL[feature_id],
                "stages": stage_rows,
            }
        )
    return distributions


def _canonical_channel(label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", label.lower())
    return {
        "lefteye": "left_eye",
        "righteye": "right_eye",
        "emg": "emg",
        "c3a2": "eeg_c3a2",
        "ecg": "ecg",
    }.get(key, key)


def _sample_signal(
    reader: Any,
    channel_indices: dict[str, int],
    signal_id: str,
    label: str,
    start_sec: float,
    end_sec: float,
    max_points: int = 900,
) -> dict[str, Any]:
    if signal_id == "eog":
        left_idx = channel_indices["left_eye"]
        right_idx = channel_indices["right_eye"]
        sample_rate = float(reader.getSampleFrequency(left_idx))
        start_idx = int(math.floor(start_sec * sample_rate))
        sample_count = int(math.ceil((end_sec - start_sec) * sample_rate))
        values = np.asarray(reader.readSignal(left_idx, start_idx, sample_count), dtype=float)
        values -= np.asarray(reader.readSignal(right_idx, start_idx, sample_count), dtype=float)
    else:
        channel_idx = channel_indices[signal_id]
        sample_rate = float(reader.getSampleFrequency(channel_idx))
        start_idx = int(math.floor(start_sec * sample_rate))
        sample_count = int(math.ceil((end_sec - start_sec) * sample_rate))
        values = np.asarray(reader.readSignal(channel_idx, start_idx, sample_count), dtype=float)

    stride = max(int(math.ceil(values.size / max_points)), 1)
    sampled = values[::stride]
    sampled -= np.nanmedian(sampled)
    sampled = np.round(sampled, 5)
    return {
        "id": signal_id,
        "label": label,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "effective_sample_rate_hz": sample_rate / stride,
        "values": sampled.tolist(),
    }


def _signal_spec_for_feature(feature_id: str) -> tuple[str, str]:
    if feature_id.startswith(("spindle_features", "slow_wave_features")):
        return "eeg_c3a2", "EEG C3–A2"
    if feature_id.startswith(("eye_movement_activity", "eog_rem_features")):
        return "eog", "EOG L–R"
    if feature_id.startswith("emg_tone_features"):
        return "emg", "Chin EMG"
    return "ecg", "ECG"


def _raw_feature_excerpt(
    record_id: str,
    start_sec: float,
    end_sec: float,
    feature_id: str,
) -> dict[str, Any]:
    import pyedflib  # type: ignore[import-not-found]

    path = ROOT / "data" / "raw" / "ucddb" / f"{record_id}.rec"
    reader = pyedflib.EdfReader(str(path))
    try:
        indices = {
            _canonical_channel(label): index
            for index, label in enumerate(reader.getSignalLabels())
        }
        signal_id, label = _signal_spec_for_feature(feature_id)
        return _sample_signal(
            reader,
            indices,
            signal_id,
            label,
            start_sec,
            end_sec,
            max_points=900,
        )
    finally:
        reader.close()


def _signal_is_usable(signal: dict[str, Any]) -> bool:
    values = np.asarray(signal["values"], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 100 or np.unique(np.round(finite, 6)).size < 50:
        return False
    q01, q25, q75, q99 = np.quantile(finite, [0.01, 0.25, 0.75, 0.99])
    robust_span = q75 - q25
    return bool(robust_span > 0 and (q99 - q01) / robust_span < 40)


def _representative_signal_examples(
    merged: pd.DataFrame,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    stage_to_index = {stage: index for index, stage in enumerate(STAGES)}
    family_by_feature = {
        feature_id: group["id"]
        for group in FEATURE_GROUPS
        for feature_id in group["feature_ids"]
    }

    for stage, feature_ids in STAGE_FEATURE_TARGETS.items():
      for feature_id in feature_ids:
        stage_index = stage_to_index[stage]
        target_values = merged.loc[merged["y_true"].eq(stage_index), feature_id]
        rest_values = merged.loc[~merged["y_true"].eq(stage_index), feature_id]
        effect = _cohens_d(target_values, rest_values)
        direction = "high" if effect >= 0 else "low"
        candidates = merged.loc[
            merged["stable_stage"]
            & merged["y_true"].eq(stage_index)
            & merged["y_pred"].eq(stage_index)
            & merged[feature_id].notna()
        ].copy()
        candidates["selection_percentile"] = candidates[feature_id].rank(
            pct=True,
            ascending=True,
        )
        target = 0.97 if direction == "high" else 0.03
        candidates["selection_distance"] = (
            candidates["selection_percentile"] - target
        ).abs()
        candidates = candidates.sort_values(
            ["selection_distance", "subject_id", "time_seconds"]
        )

        selected_row = None
        selected_signal = None
        for row in candidates.head(80).itertuples(index=False):
            epoch_start = float(row.time_seconds)
            signal = _raw_feature_excerpt(
                str(row.subject_id),
                epoch_start,
                epoch_start + 30.0,
                feature_id,
            )
            if _signal_is_usable(signal):
                selected_row = row
                selected_signal = signal
                break

        if selected_row is None or selected_signal is None:
            raise RuntimeError(f"No usable signal example found for {stage} / {feature_id}")

        record_id = str(selected_row.subject_id)
        epoch_start = float(selected_row.time_seconds)
        value = float(
            merged.loc[
                merged["subject_id"].eq(record_id)
                & merged["time_seconds"].eq(epoch_start),
                feature_id,
            ].iloc[0]
        )
        population = merged[feature_id].dropna().to_numpy(dtype=float)
        percentile = float(np.mean(population <= value) * 100.0)
        stage_percentile = float(selected_row.selection_percentile * 100.0)
        output.append(
            {
                "id": f"{stage.lower()}::{feature_id}",
                "family_id": family_by_feature[feature_id],
                "stage": stage,
                "record_id": record_id,
                "epoch_start_sec": epoch_start,
                "feature_id": feature_id,
                "feature_label": FEATURE_LABELS[feature_id],
                "feature_value": value,
                "feature_percentile": percentile,
                "stage_percentile": stage_percentile,
                "direction": direction,
                "selection_note": f"Stable, correctly classified {stage} epoch near the {'upper' if direction == 'high' else 'lower'} tail of this feature within {stage}.",
                "signal": selected_signal,
            }
        )
    return output


def main() -> int:
    smoother_root = (
        ARTIFACTS
        / "paper_matched_stage2_smoothers"
        / "kfold5__none"
        / "staging_30s"
    )
    tcn_root = (
        ARTIFACTS
        / "sequence_cnn_6s_epoch"
        / "kfold5__h10_c5_f5__dilated_tcn"
    )
    hierarchical_root = (
        ARTIFACTS
        / "hierarchical_epoch_staging"
        / "kfold5__eh1_ec1_ef1_s5__hierarchical_epoch"
    )

    models = [
        _model(
            model_id="raw-ensemble",
            name="Feature ensemble",
            role="Epoch baseline",
            hypothesis="Can physiology-derived features classify each epoch without temporal context?",
            summary_path=smoother_root / "ctx9" / "summary_raw.parquet",
            folds_path=smoother_root / "ctx9" / "fold_metrics_raw.parquet",
        ),
        _model(
            model_id="context-5",
            name="Learned context (5 epochs)",
            role="Simple temporal model",
            hypothesis="Does a compact learned probability context correct unstable epoch decisions?",
            summary_path=smoother_root / "ctx5" / "summary_context_linear.parquet",
            folds_path=smoother_root / "ctx5" / "fold_metrics_context_linear.parquet",
        ),
        _model(
            model_id="tcn-6s",
            name="6-second dilated TCN",
            role="Feature-sequence network",
            hypothesis="Can a deeper network learn useful short- and long-range structure directly?",
            summary_path=tcn_root / "summary.parquet",
            folds_path=tcn_root / "fold_metrics.parquet",
        ),
        _model(
            model_id="hierarchical",
            name="Hierarchical epoch model",
            role="Connected temporal network",
            hypothesis="Does jointly modeling within-epoch and across-epoch structure improve generalization?",
            summary_path=hierarchical_root / "summary_raw.parquet",
            folds_path=hierarchical_root / "fold_metrics_raw.parquet",
        ),
        _model(
            model_id="context-9",
            name="Learned context + confidence",
            role="Best overall",
            hypothesis="Can wider context, logits, and confidence preserve useful uncertainty across epochs?",
            summary_path=smoother_root / "ctx9" / "summary_context_linear_logit_conf.parquet",
            folds_path=smoother_root / "ctx9" / "fold_metrics_context_linear_logit_conf.parquet",
        ),
    ]

    best_prediction_path = (
        smoother_root / "ctx9" / "predictions_context_linear_logit_conf.parquet"
    )
    diagnostics = _best_prediction_diagnostics(best_prediction_path)
    feature_matrix = _feature_matrix(best_prediction_path)
    feature_frame = pd.read_parquet(
        ARTIFACTS
        / "staging_diagnostics"
        / "kfold5__none"
        / "stage1_feature_family_importance.parquet"
    ).sort_values("importance_mean", ascending=False)
    feature_labels = {
        "slow_wave_features": "Slow-wave features",
        "eog_rem_features": "EOG / REM features",
        "eye_movement_activity": "Eye-movement activity",
        "spindle_features": "Spindle features",
        "emg_tone_features": "EMG tone features",
        "pan_tompkins": "ECG / heart rate",
        "eeg_arousal_features": "EEG arousal features",
    }
    feature_importance = [
        {
            "group": feature_labels.get(
                str(row.feature_group),
                str(row.feature_group).replace("_features", "").replace("_", " ").title(),
            ),
            "importance": float(row.importance_mean),
        }
        for row in feature_frame.itertuples(index=False)
    ]

    baseline = next(model for model in models if model["id"] == "raw-ensemble")
    winner = next(model for model in models if model["id"] == "context-9")
    clinical_source_path = ARTIFACTS / "showcase" / "site-data.json"
    clinical_source = json.loads(clinical_source_path.read_text(encoding="utf-8"))

    examples = _representative_signal_examples(feature_matrix)
    payload = {
        "source": {
            "dataset": "University College Dublin Sleep Apnea Database",
            "short_name": "UCDDB",
            "subjects": diagnostics["subjects"],
            "epochs": diagnostics["rows"],
            "epoch_seconds": 30,
            "features": 34,
            "stages": STAGES,
        },
        "validation": {
            "outer_folds": 5,
            "train_subjects_per_fold": 20,
            "test_subjects_per_fold": 5,
            "neural_early_stop_subjects": 1,
            "grouping": "subject",
            "primary_metric": "Macro-F1",
            "folds": _validation_folds(best_prediction_path),
        },
        "headline": {
            "baseline_macro_f1": baseline["metrics"]["macro_f1"],
            "best_macro_f1": winner["metrics"]["macro_f1"],
            "absolute_lift": winner["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
        },
        "models": models,
        "diagnostics": diagnostics,
        "feature_importance": feature_importance,
        "feature_evidence": {
            "distributions": _feature_distributions(feature_matrix),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    FEATURE_EXAMPLES_OUTPUT.write_text(
        json.dumps(examples, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {FEATURE_EXAMPLES_OUTPUT}")
    for example in examples:
        print(
            f"  {example['family_id']}: {example['record_id']} "
            f"@ {example['epoch_start_sec']:.0f}s, {example['stage']}, "
            f"{example['feature_percentile']:.1f}th percentile"
        )

    stage_examples = []
    seen_stages: set[str] = set()
    for example in clinical_source["staging"]["examples"]:
        stage_label = str(example["class_label"]).lower()
        if stage_label in seen_stages:
            continue
        seen_stages.add(stage_label)
        stage_examples.append(
            {
                "id": example["id"],
                "class_id": example["class_id"],
                "class_label": example["class_label"],
                "run_start_sec": example["run_start_sec"],
                "run_end_sec": example["run_end_sec"],
                "excerpt_start_sec": example["excerpt_start_sec"],
                "excerpt_end_sec": example["excerpt_end_sec"],
                "signals": example["signals"][:3],
                "feature_window": example.get("feature_window"),
            }
        )

    clinical_payload = {
        "generated_at": clinical_source["generated_at"],
        "dataset": clinical_source["dataset"],
        "record_id": clinical_source["record_id"],
        "apnea": {"default_example_index": 0, "examples": []},
        "staging": {
            "record_duration_sec": clinical_source["staging"].get("record_duration_sec"),
            "hypnogram": clinical_source["staging"].get("hypnogram"),
            "anchors": [],
            "examples": stage_examples,
        },
    }
    CLINICAL_OUTPUT.write_text(json.dumps(clinical_payload, indent=2), encoding="utf-8")
    print(f"Wrote {CLINICAL_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
