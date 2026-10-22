from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from physionet_sleep.experiments.normalization import apply_normalization
from physionet_sleep.experiments.sequence_cnn import (
    _build_sequence_dataset,
    _impute_features,
    _predict_cnn_classifier,
    _prepare_frame as _prepare_sequence_frame,
    _require_torch,
    _split_inner_validation_subjects,
    _train_cnn_classifier,
    SequenceExperimentSpec,
)
from physionet_sleep.experiments.sequence_data import flatten_sequence_tensor
from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits, score_predictions


DEFAULT_STAGE_LABEL_MAP = {
    "wake": 0,
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "rem": 4,
}

DEFAULT_STAGE_EXPERT_FEATURES = {
    "wake": [
        "slow_wave_features.high_frequency_envelope",
        "emg_tone_features.emg_tone_baseline",
        "eye_movement_activity.eye_movement_baseline",
        "pan_tompkins.heart_rate_bpm_track",
    ],
    "n1": [
        "spindle_features.aperiodic_exponent",
        "eye_movement_activity.eye_movement_activity",
        "eog_rem_features.sem_band_power_fast",
        "respiration.respiration_envelope",
    ],
    "n2": [
        "spindle_features.splindex",
        "spindle_features.sigma_residual_power",
        "spindle_features.aperiodic_exponent",
        "pan_tompkins.hrv_rmssd_track",
    ],
    "n3": [
        "slow_wave_features.slow_wave_envelope",
        "slow_wave_features.slow_wave_index",
        "slow_wave_features.slow_wave_ratio_power",
        "slow_wave_features.high_frequency_envelope",
    ],
    "rem": [
        "eye_movement_activity.eye_movement_activity",
        "eog_rem_features.rem_band_power_fast",
        "eog_rem_features.sem_band_power_fast",
        "emg_tone_features.emg_tone_baseline",
        "respiration.respiration_envelope",
    ],
}


@dataclass(slots=True)
class StageExpertSpec:
    name: str
    positive_label: int
    feature_columns: list[str]


@dataclass(slots=True)
class StageExpertTemporalSpec:
    name: str
    expert_specs: list[StageExpertSpec]
    expert_model_family: str = "cnn"
    stage2_model_family: str = "linear"
    outer_validation: str = "kfold"
    outer_folds: int = 5
    inner_validation: str = "single_subject"
    normalization: str = "medianiqr"
    current_steps: int = 5
    context_radius: int = 2
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_channels: int = 32
    kernel_size: int = 3
    dropout: float = 0.1
    min_epochs: int = 10
    patience: int = 8
    tcn_levels: int = 4
    weighted_sampling: bool = True
    expected_step_seconds: float = 6.0
    step_tolerance_seconds: float = 0.25


@dataclass(slots=True)
class StageExpertTemporalRunResult:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    expert_fold_metrics: pd.DataFrame
    metadata: dict[str, Any]


def build_default_stage_expert_specs(feature_table: pd.DataFrame) -> list[StageExpertSpec]:
    available = set(feature_table.columns)
    specs: list[StageExpertSpec] = []
    for name, label in DEFAULT_STAGE_LABEL_MAP.items():
        cols = [col for col in DEFAULT_STAGE_EXPERT_FEATURES.get(name, []) if col in available]
        if cols:
            specs.append(StageExpertSpec(name=name, positive_label=int(label), feature_columns=cols))
    return specs


def run_stage_expert_temporal_experiment(
    feature_table: pd.DataFrame,
    *,
    spec: StageExpertTemporalSpec,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    target_column: str = "label.stage_seconds",
    random_state: int = 7,
) -> StageExpertTemporalRunResult:
    prepared = _prepare_sequence_frame(
        feature_table,
        feature_columns=_union_feature_columns(spec.expert_specs),
        target_column=target_column,
        subject_column=subject_column,
        time_column=time_column,
    )
    if prepared.empty:
        return StageExpertTemporalRunResult(
            summary=pd.DataFrame(),
            fold_metrics=pd.DataFrame(),
            predictions=pd.DataFrame(),
            expert_fold_metrics=pd.DataFrame(),
            metadata={"spec": asdict(spec), "rows": 0},
        )

    groups = prepared[subject_column].astype(str).to_numpy()
    y = prepared[target_column].to_numpy()
    splits = list(
        build_subject_eval_splits(
            groups=groups,
            y=y,
            outer_validation=spec.outer_validation,
            outer_folds=spec.outer_folds,
            inner_validation=spec.inner_validation,
            random_state=random_state,
        )
    )

    fold_rows: list[dict[str, Any]] = []
    expert_fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for split in splits:
        train_df_raw = prepared.iloc[split.train_idx].copy()
        test_df_raw = prepared.iloc[split.test_idx].copy()
        fit_train_raw, calib_raw = _split_calibration_subjects(
            train_df_raw,
            subject_column=subject_column,
            fold_index=split.fold_index,
            inner_validation=spec.inner_validation,
            random_state=random_state,
        )
        if calib_raw.empty:
            continue

        calib_frames: list[pd.DataFrame] = []
        test_frames: list[pd.DataFrame] = []
        for expert in spec.expert_specs:
            expert_result = _run_single_expert(
                fit_train_raw,
                calib_raw,
                test_df_raw,
                expert=expert,
                spec=spec,
                subject_column=subject_column,
                time_column=time_column,
                target_column=target_column,
                random_state=random_state,
            )
            calib_frames.append(expert_result["calib_frame"])
            test_frames.append(expert_result["test_frame"])
            expert_fold_rows.append(
                {
                    "experiment": spec.name,
                    "fold_index": split.fold_index,
                    "expert_name": expert.name,
                    "expert_model_family": spec.expert_model_family,
                    "positive_label": int(expert.positive_label),
                    "n_features": int(len(expert.feature_columns)),
                    "n_train": int(expert_result["n_train"]),
                    "n_calib": int(expert_result["n_calib"]),
                    "n_test": int(expert_result["n_test"]),
                    "n_positive_train": int(expert_result["n_positive_train"]),
                    "n_positive_calib": int(expert_result["n_positive_calib"]),
                    "n_positive_test": int(expert_result["n_positive_test"]),
                }
                | expert_result["metrics"]
            )

        calib_epoch = _merge_expert_epoch_frames(calib_frames, spec.expert_specs, subject_column=subject_column, time_column=time_column)
        test_epoch = _merge_expert_epoch_frames(test_frames, spec.expert_specs, subject_column=subject_column, time_column=time_column)
        if calib_epoch.empty or test_epoch.empty:
            continue

        calib_epoch = _attach_stage1_vote(calib_epoch, spec.expert_specs)
        test_epoch = _attach_stage1_vote(test_epoch, spec.expert_specs)

        raw_metrics = score_predictions(
            y_true=calib_epoch["y_true"].to_numpy(dtype=int),
            y_pred=calib_epoch["y_pred_stage1"].to_numpy(dtype=int),
            proba=_stage_probability_matrix(calib_epoch, spec.expert_specs),
        )
        test_raw_metrics = score_predictions(
            y_true=test_epoch["y_true"].to_numpy(dtype=int),
            y_pred=test_epoch["y_pred_stage1"].to_numpy(dtype=int),
            proba=_stage_probability_matrix(test_epoch, spec.expert_specs),
        )

        raw_pred_frame = test_epoch.copy()
        raw_pred_frame["y_pred"] = raw_pred_frame["y_pred_stage1"].to_numpy(dtype=int)
        raw_pred_frame["variant"] = "raw_expert_vote"
        prediction_frames.append(raw_pred_frame)

        fold_rows.append(
            {
                "experiment": spec.name,
                "fold_index": split.fold_index,
                "variant": "raw_expert_vote",
                "expert_model_family": spec.expert_model_family,
                "stage2_model_family": spec.stage2_model_family,
                "context_radius": int(spec.context_radius),
                "n_train": int(len(fit_train_raw)),
                "n_calib": int(len(calib_raw)),
                "n_test": int(len(test_df_raw)),
                "n_epoch_calib": int(len(calib_epoch)),
                "n_epoch_test": int(len(test_epoch)),
                "n_experts": int(len(spec.expert_specs)),
                "calib_macro_f1": float(raw_metrics["macro_f1"]),
            }
            | test_raw_metrics
        )

        stage2_result = _run_stage2_combiner(
            calib_epoch,
            test_epoch,
            spec=spec,
            subject_column=subject_column,
            time_column=time_column,
            random_state=random_state,
        )
        if stage2_result is None:
            continue
        stage2_pred_frame = stage2_result["predictions"]
        stage2_pred_frame["variant"] = "temporal_combiner"
        prediction_frames.append(stage2_pred_frame)
        fold_rows.append(
            {
                "experiment": spec.name,
                "fold_index": split.fold_index,
                "variant": "temporal_combiner",
                "expert_model_family": spec.expert_model_family,
                "stage2_model_family": spec.stage2_model_family,
                "context_radius": int(spec.context_radius),
                "n_train": int(len(fit_train_raw)),
                "n_calib": int(len(calib_raw)),
                "n_test": int(len(test_df_raw)),
                "n_epoch_calib": int(len(calib_epoch)),
                "n_epoch_test": int(len(test_epoch)),
                "n_experts": int(len(spec.expert_specs)),
                "n_stage2_rows_calib": int(stage2_result["n_stage2_rows_calib"]),
                "n_stage2_rows_test": int(stage2_result["n_stage2_rows_test"]),
                "n_stage2_features": int(stage2_result["n_stage2_features"]),
            }
            | stage2_result["metrics"]
        )

    fold_metrics = pd.DataFrame(fold_rows)
    expert_fold_metrics = pd.DataFrame(expert_fold_rows)
    predictions = pd.concat(prediction_frames, axis=0, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary = _summarize_fold_metrics(fold_metrics)
    metadata = {
        "spec": asdict(spec),
        "rows": int(len(prepared)),
        "subjects": sorted(prepared[subject_column].astype(str).unique().tolist()),
        "feature_union": _union_feature_columns(spec.expert_specs),
    }
    return StageExpertTemporalRunResult(
        summary=summary,
        fold_metrics=fold_metrics,
        predictions=predictions,
        expert_fold_metrics=expert_fold_metrics,
        metadata=metadata,
    )


def _union_feature_columns(expert_specs: list[StageExpertSpec]) -> list[str]:
    return list(dict.fromkeys(col for expert in expert_specs for col in expert.feature_columns))


def _split_calibration_subjects(
    train_df: pd.DataFrame,
    *,
    subject_column: str,
    fold_index: int,
    inner_validation: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if inner_validation == "single_subject":
        return _split_inner_validation_subjects(
            train_df,
            subject_column=subject_column,
            fold_index=fold_index,
            n_val_subjects=1,
            random_state=random_state,
        )
    if inner_validation == "none":
        return train_df.copy(), train_df.copy()
    raise ValueError(f"Unsupported inner_validation: {inner_validation}")


def _run_single_expert(
    train_df_raw: pd.DataFrame,
    calib_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    *,
    expert: StageExpertSpec,
    spec: StageExpertTemporalSpec,
    subject_column: str,
    time_column: str,
    target_column: str,
    random_state: int,
) -> dict[str, Any]:
    train_df, calib_df, _ = apply_normalization(
        train_df_raw,
        calib_df_raw,
        feature_columns=expert.feature_columns,
        norm_name=spec.normalization,
        group_column=subject_column,
    )
    _, test_df, _ = apply_normalization(
        train_df_raw,
        test_df_raw,
        feature_columns=expert.feature_columns,
        norm_name=spec.normalization,
        group_column=subject_column,
    )
    train_df, calib_df = _impute_features(train_df, calib_df, feature_columns=expert.feature_columns)
    train_eval_df, test_df = _impute_features(train_df, test_df, feature_columns=expert.feature_columns)

    sequence_spec = SequenceExperimentSpec(
        name=f"{expert.name}_expert",
        target_column=target_column,
        feature_columns=expert.feature_columns,
        model_arch="simple_cnn",
        sequence_mode="epoch_aligned",
        history_steps=0,
        current_steps=spec.current_steps,
        future_steps=0,
        normalization=spec.normalization,
        cv_folds=spec.outer_folds,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        hidden_channels=spec.hidden_channels,
        kernel_size=spec.kernel_size,
        dropout=spec.dropout,
        tcn_levels=spec.tcn_levels,
        min_epochs=spec.min_epochs,
        patience=spec.patience,
        internal_val_subjects=1,
        weighted_sampling=spec.weighted_sampling,
        expected_step_seconds=spec.expected_step_seconds,
        step_tolerance_seconds=spec.step_tolerance_seconds,
    )
    train_dataset = _build_sequence_dataset(train_df, spec=sequence_spec, subject_column=subject_column, time_column=time_column)
    calib_dataset = _build_sequence_dataset(calib_df, spec=sequence_spec, subject_column=subject_column, time_column=time_column)
    test_dataset = _build_sequence_dataset(test_df, spec=sequence_spec, subject_column=subject_column, time_column=time_column)
    if train_dataset.X.shape[0] == 0 or calib_dataset.X.shape[0] == 0 or test_dataset.X.shape[0] == 0:
        raise ValueError(f"Empty sequence dataset for expert {expert.name}")

    y_train = (train_dataset.y.astype(int) == int(expert.positive_label)).astype(int)
    y_calib = (calib_dataset.y.astype(int) == int(expert.positive_label)).astype(int)
    y_test = (test_dataset.y.astype(int) == int(expert.positive_label)).astype(int)

    train_proba, calib_proba, test_proba, test_pred = _fit_binary_sequence_model(
        train_dataset.X,
        y_train,
        calib_dataset.X,
        y_calib,
        test_dataset.X,
        expert_model_family=spec.expert_model_family,
        epochs=spec.epochs,
        min_epochs=spec.min_epochs,
        patience=spec.patience,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        hidden_channels=spec.hidden_channels,
        kernel_size=spec.kernel_size,
        dropout=spec.dropout,
        tcn_levels=spec.tcn_levels,
        current_steps=spec.current_steps,
        weighted_sampling=spec.weighted_sampling,
        random_state=random_state,
    )

    metrics = score_predictions(
        y_true=y_test,
        y_pred=test_pred,
        proba=np.column_stack([1.0 - test_proba, test_proba]),
    )
    calib_frame = calib_dataset.metadata.copy()
    calib_frame["y_true"] = calib_dataset.y.astype(int)
    calib_frame[f"expert_{expert.name}_proba"] = calib_proba
    calib_frame[f"expert_{expert.name}_binary_true"] = y_calib

    test_frame = test_dataset.metadata.copy()
    test_frame["y_true"] = test_dataset.y.astype(int)
    test_frame[f"expert_{expert.name}_proba"] = test_proba
    test_frame[f"expert_{expert.name}_binary_true"] = y_test
    test_frame[f"expert_{expert.name}_binary_pred"] = test_pred

    return {
        "calib_frame": calib_frame,
        "test_frame": test_frame,
        "metrics": metrics,
        "n_train": int(train_dataset.X.shape[0]),
        "n_calib": int(calib_dataset.X.shape[0]),
        "n_test": int(test_dataset.X.shape[0]),
        "n_positive_train": int(y_train.sum()),
        "n_positive_calib": int(y_calib.sum()),
        "n_positive_test": int(y_test.sum()),
        "train_proba_mean": float(np.mean(train_proba)) if train_proba.size else float("nan"),
    }


def _fit_binary_sequence_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    *,
    expert_model_family: str,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    kernel_size: int,
    dropout: float,
    tcn_levels: int,
    current_steps: int,
    weighted_sampling: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(y_train)
    if unique.size < 2:
        constant = float(unique[0]) if unique.size == 1 else 0.0
        train_proba = np.full(X_train.shape[0], constant, dtype=float)
        calib_proba = np.full(X_calib.shape[0], constant, dtype=float)
        test_proba = np.full(X_test.shape[0], constant, dtype=float)
        test_pred = np.rint(test_proba).astype(int)
        return train_proba, calib_proba, test_proba, test_pred

    if expert_model_family == "cnn":
        _require_torch()
        model, encoder, _ = _train_cnn_classifier(
            X_train,
            y_train,
            X_early_val=X_calib,
            y_early_val=y_calib,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            model_arch="simple_cnn",
            history_steps=0,
            current_steps=current_steps,
            future_steps=0,
            tcn_levels=tcn_levels,
            weighted_sampling=weighted_sampling,
            random_state=random_state,
        )
        y_train_pred, train_proba_full = _predict_cnn_classifier(model, encoder, X_train)
        _, calib_proba_full = _predict_cnn_classifier(model, encoder, X_calib)
        y_test_pred, test_proba_full = _predict_cnn_classifier(model, encoder, X_test)
        train_proba = _positive_column(train_proba_full, encoder.classes_)
        calib_proba = _positive_column(calib_proba_full, encoder.classes_)
        test_proba = _positive_column(test_proba_full, encoder.classes_)
        return train_proba, calib_proba, test_proba, np.asarray(y_test_pred, dtype=int)

    X_train_flat = flatten_sequence_tensor(X_train)
    X_calib_flat = flatten_sequence_tensor(X_calib)
    X_test_flat = flatten_sequence_tensor(X_test)
    estimator = build_model(
        model_family=expert_model_family,
        task="classification",
        random_state=random_state,
        class_weight="balanced",
    )
    estimator.fit(X_train_flat, y_train)
    train_proba = estimator.predict_proba(X_train_flat)[:, -1]
    calib_proba = estimator.predict_proba(X_calib_flat)[:, -1]
    test_proba = estimator.predict_proba(X_test_flat)[:, -1]
    test_pred = np.asarray(estimator.predict(X_test_flat), dtype=int)
    return train_proba, calib_proba, test_proba, test_pred


def _positive_column(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes)
    positive_idx = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else int(classes.size - 1)
    return proba[:, positive_idx]


def _merge_expert_epoch_frames(
    frames: list[pd.DataFrame],
    expert_specs: list[StageExpertSpec],
    *,
    subject_column: str,
    time_column: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keep_cols = [subject_column, time_column, "y_true"]
    for frame, expert in zip(frames, expert_specs, strict=False):
        cols = keep_cols if merged is None else [subject_column, time_column]
        expert_cols = [col for col in frame.columns if col.startswith(f"expert_{expert.name}_")]
        block = frame[cols + expert_cols].copy()
        if merged is None:
            merged = block
        else:
            merged = merged.merge(block, on=[subject_column, time_column], how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values([subject_column, time_column], kind="stable").reset_index(drop=True)


def _attach_stage1_vote(frame: pd.DataFrame, expert_specs: list[StageExpertSpec]) -> pd.DataFrame:
    out = frame.copy()
    prob_cols = [f"expert_{expert.name}_proba" for expert in expert_specs]
    prob_matrix = out[prob_cols].to_numpy(dtype=float)
    row_sum = prob_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        prob_matrix,
        row_sum,
        out=np.full_like(prob_matrix, 1.0 / max(prob_matrix.shape[1], 1), dtype=float),
        where=row_sum > 0,
    )
    stage_labels = np.asarray([expert.positive_label for expert in expert_specs], dtype=int)
    out["y_pred_stage1"] = stage_labels[np.argmax(normalized, axis=1)]
    for idx, expert in enumerate(expert_specs):
        out[f"stage_proba_{expert.positive_label}"] = normalized[:, idx]
    return out


def _stage_probability_matrix(frame: pd.DataFrame, expert_specs: list[StageExpertSpec]) -> np.ndarray:
    cols = [f"stage_proba_{expert.positive_label}" for expert in expert_specs]
    return frame[cols].to_numpy(dtype=float)


def _run_stage2_combiner(
    calib_epoch: pd.DataFrame,
    test_epoch: pd.DataFrame,
    *,
    spec: StageExpertTemporalSpec,
    subject_column: str,
    time_column: str,
    random_state: int,
) -> dict[str, Any] | None:
    if spec.stage2_model_family == "tcn":
        return _run_stage2_tcn(
            calib_epoch,
            test_epoch,
            spec=spec,
            subject_column=subject_column,
            time_column=time_column,
            random_state=random_state,
        )

    calib_context = _build_context_frame(
        calib_epoch,
        expert_specs=spec.expert_specs,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=spec.context_radius,
    )
    test_context = _build_context_frame(
        test_epoch,
        expert_specs=spec.expert_specs,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=spec.context_radius,
    )
    if calib_context.empty or test_context.empty:
        return None
    if calib_context["y_true"].nunique() < 2:
        return None

    context_cols = _context_feature_columns(spec.expert_specs, context_radius=spec.context_radius)
    stage2 = build_model(
        model_family=spec.stage2_model_family,
        task="classification",
        random_state=random_state,
        class_weight="balanced",
    )
    stage2.fit(calib_context[context_cols], calib_context["y_true"])
    y_pred = np.asarray(stage2.predict(test_context[context_cols]), dtype=int)

    refined = _apply_stage2_epoch_predictions(
        test_epoch,
        updates=test_context[[subject_column, time_column]].assign(y_pred=y_pred),
        subject_column=subject_column,
        time_column=time_column,
    )
    metrics = score_predictions(
        y_true=refined["y_true"].to_numpy(dtype=int),
        y_pred=refined["y_pred"].to_numpy(dtype=int),
        proba=None,
    )
    return {
        "predictions": refined,
        "metrics": metrics,
        "n_stage2_rows_calib": int(len(calib_context)),
        "n_stage2_rows_test": int(len(test_context)),
        "n_stage2_features": int(len(context_cols)),
    }


def _run_stage2_tcn(
    calib_epoch: pd.DataFrame,
    test_epoch: pd.DataFrame,
    *,
    spec: StageExpertTemporalSpec,
    subject_column: str,
    time_column: str,
    random_state: int,
) -> dict[str, Any] | None:
    calib_seq = _build_context_sequence_dataset(
        calib_epoch,
        expert_specs=spec.expert_specs,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=spec.context_radius,
    )
    test_seq = _build_context_sequence_dataset(
        test_epoch,
        expert_specs=spec.expert_specs,
        subject_column=subject_column,
        time_column=time_column,
        context_radius=spec.context_radius,
    )
    if calib_seq is None or test_seq is None:
        return None
    if np.unique(calib_seq["y"]).size < 2:
        return None

    _require_torch()
    model, encoder, _ = _train_cnn_classifier(
        calib_seq["X"],
        calib_seq["y"],
        X_early_val=calib_seq["X"],
        y_early_val=calib_seq["y"],
        epochs=spec.epochs,
        min_epochs=spec.min_epochs,
        patience=spec.patience,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        hidden_channels=spec.hidden_channels,
        kernel_size=spec.kernel_size,
        dropout=spec.dropout,
        model_arch="dilated_tcn",
        history_steps=spec.context_radius,
        current_steps=1,
        future_steps=spec.context_radius,
        tcn_levels=spec.tcn_levels,
        weighted_sampling=spec.weighted_sampling,
        random_state=random_state,
    )
    y_pred, _ = _predict_cnn_classifier(model, encoder, test_seq["X"])
    y_pred = np.asarray(y_pred, dtype=int)
    refined = _apply_stage2_epoch_predictions(
        test_epoch,
        updates=test_seq["metadata"][[subject_column, time_column]].assign(y_pred=y_pred),
        subject_column=subject_column,
        time_column=time_column,
    )
    metrics = score_predictions(
        y_true=refined["y_true"].to_numpy(dtype=int),
        y_pred=refined["y_pred"].to_numpy(dtype=int),
        proba=None,
    )
    return {
        "predictions": refined,
        "metrics": metrics,
        "n_stage2_rows_calib": int(calib_seq["X"].shape[0]),
        "n_stage2_rows_test": int(test_seq["X"].shape[0]),
        "n_stage2_features": int(calib_seq["X"].shape[2]),
    }


def _build_context_frame(
    epoch_frame: pd.DataFrame,
    *,
    expert_specs: list[StageExpertSpec],
    subject_column: str,
    time_column: str,
    context_radius: int,
    expected_step_seconds: float = 30.0,
    step_tolerance_seconds: float = 0.25,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prob_cols = [f"expert_{expert.name}_proba" for expert in expert_specs]
    window_length = (2 * int(context_radius)) + 1
    for subject_id, subject_df in epoch_frame.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        if len(ordered) < window_length:
            continue
        times = ordered[time_column].to_numpy(dtype=float, copy=False)
        for center in range(context_radius, len(ordered) - context_radius):
            start = center - context_radius
            stop = center + context_radius + 1
            seq_times = times[start:stop]
            if seq_times.size != window_length:
                continue
            diffs = np.diff(seq_times)
            if diffs.size and not np.all(np.abs(diffs - expected_step_seconds) <= step_tolerance_seconds):
                continue
            row: dict[str, object] = {
                subject_column: str(subject_id),
                time_column: float(times[center]),
                "y_true": int(ordered["y_true"].iloc[center]),
                "y_pred_stage1": int(ordered["y_pred_stage1"].iloc[center]),
            }
            block = ordered.iloc[start:stop]
            for offset in range(-context_radius, context_radius + 1):
                offset_label = _offset_label(offset)
                source = block.iloc[offset + context_radius]
                for prob_col in prob_cols:
                    row[f"{offset_label}.{prob_col}"] = float(source[prob_col])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_context_sequence_dataset(
    epoch_frame: pd.DataFrame,
    *,
    expert_specs: list[StageExpertSpec],
    subject_column: str,
    time_column: str,
    context_radius: int,
    expected_step_seconds: float = 30.0,
    step_tolerance_seconds: float = 0.25,
) -> dict[str, Any] | None:
    seq_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict[str, object]] = []
    prob_cols = [f"expert_{expert.name}_proba" for expert in expert_specs]
    window_length = (2 * int(context_radius)) + 1
    for subject_id, subject_df in epoch_frame.groupby(subject_column, sort=False):
        ordered = subject_df.sort_values(time_column, kind="stable").reset_index(drop=True)
        if len(ordered) < window_length:
            continue
        times = ordered[time_column].to_numpy(dtype=float, copy=False)
        probs = ordered[prob_cols].to_numpy(dtype=float, copy=False)
        y_true = ordered["y_true"].to_numpy(dtype=int, copy=False)
        y_stage1 = ordered["y_pred_stage1"].to_numpy(dtype=int, copy=False)
        for center in range(context_radius, len(ordered) - context_radius):
            start = center - context_radius
            stop = center + context_radius + 1
            seq_times = times[start:stop]
            if seq_times.size != window_length:
                continue
            diffs = np.diff(seq_times)
            if diffs.size and not np.all(np.abs(diffs - expected_step_seconds) <= step_tolerance_seconds):
                continue
            seq_list.append(probs[start:stop])
            y_list.append(int(y_true[center]))
            meta_rows.append(
                {
                    subject_column: str(subject_id),
                    time_column: float(times[center]),
                    "y_true": int(y_true[center]),
                    "y_pred_stage1": int(y_stage1[center]),
                }
            )
    if not seq_list:
        return None
    return {
        "X": np.stack(seq_list, axis=0),
        "y": np.asarray(y_list, dtype=int),
        "metadata": pd.DataFrame(meta_rows),
    }


def _context_feature_columns(expert_specs: list[StageExpertSpec], *, context_radius: int) -> list[str]:
    cols: list[str] = []
    for offset in range(-context_radius, context_radius + 1):
        offset_label = _offset_label(offset)
        for expert in expert_specs:
            cols.append(f"{offset_label}.expert_{expert.name}_proba")
    return cols


def _apply_stage2_epoch_predictions(
    epoch_frame: pd.DataFrame,
    *,
    updates: pd.DataFrame,
    subject_column: str,
    time_column: str,
) -> pd.DataFrame:
    refined = epoch_frame.copy()
    refined["y_pred"] = refined["y_pred_stage1"].to_numpy(dtype=int)
    refined = refined.merge(updates[[subject_column, time_column, "y_pred"]], on=[subject_column, time_column], how="left", suffixes=("", "__stage2"))
    mask = refined["y_pred__stage2"].notna()
    refined.loc[mask, "y_pred"] = refined.loc[mask, "y_pred__stage2"].astype(int)
    refined = refined.drop(columns=["y_pred__stage2"])
    return refined


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    group_cols = ["variant", "expert_model_family", "stage2_model_family", "context_radius"]
    skip_cols = {"experiment", "fold_index", *group_cols}
    numeric_cols = [col for col in fold_df.columns if col not in skip_cols and pd.api.types.is_numeric_dtype(fold_df[col])]
    for keys, group in fold_df.groupby(group_cols, sort=False):
        row: dict[str, object] = {
            "variant": keys[0],
            "expert_model_family": keys[1],
            "stage2_model_family": keys[2],
            "context_radius": keys[3],
            "n_folds": int(len(group)),
        }
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(
        by=["macro_f1_mean", "cohen_kappa_mean", "accuracy_mean"],
        ascending=False,
        na_position="last",
    )
