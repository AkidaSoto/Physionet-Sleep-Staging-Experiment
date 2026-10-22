from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physionet_sleep.analysis.benchmark_windows import build_staging_benchmark_table
from physionet_sleep.experiments.sequence_cnn import (
    _predict_cnn_classifier,
    _require_torch,
    _train_cnn_classifier,
)
from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits, score_predictions
from physionet_sleep.recipe.algo_store import ParquetAlgoStore


@dataclass(slots=True)
class SignalView:
    algo_name: str
    signal_name: str


@dataclass(slots=True)
class StageSignalExpertSpec:
    name: str
    positive_label: int
    signal_views: list[SignalView]


@dataclass(slots=True)
class StageSignalExpertTemporalSpec:
    name: str
    expert_specs: list[StageSignalExpertSpec]
    stage1_model_arch: str = "dilated_tcn"
    stage2_model_family: str = "tcn"
    outer_validation: str = "kfold"
    outer_folds: int = 5
    inner_validation: str = "single_subject"
    epoch_seconds: float = 30.0
    window_points: int = 64
    context_radius: int = 2
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_channels: int = 32
    kernel_size: int = 5
    dropout: float = 0.1
    min_epochs: int = 8
    patience: int = 6
    tcn_levels: int = 3
    weighted_sampling: bool = True


@dataclass(slots=True)
class StageSignalExpertTemporalRunResult:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    expert_fold_metrics: pd.DataFrame
    metadata: dict[str, Any]


DEFAULT_STAGE_SIGNAL_EXPERT_SPECS = [
    StageSignalExpertSpec(
        name="wake",
        positive_label=0,
        signal_views=[
            SignalView("final__slow_wave_features", "high_frequency_envelope"),
            SignalView("final__emg_tone_features", "emg_tone_baseline"),
            SignalView("final__eye_movement_activity", "eye_movement_baseline"),
        ],
    ),
    StageSignalExpertSpec(
        name="n2",
        positive_label=2,
        signal_views=[
            SignalView("final__spindle_features", "sigma_power"),
            SignalView("final__spindle_features", "sigma_residual_power"),
            SignalView("final__spindle_features", "splindex"),
            SignalView("final__slow_wave_features", "slow_wave_ratio_power"),
            SignalView("final__slow_wave_features", "high_frequency_envelope"),
            SignalView("final__emg_tone_features", "emg_tone_baseline"),
            SignalView("final__eye_movement_activity", "eye_movement_activity"),
        ],
    ),
    StageSignalExpertSpec(
        name="n3",
        positive_label=3,
        signal_views=[
            SignalView("final__slow_wave_features", "slow_wave_envelope"),
            SignalView("final__slow_wave_features", "slow_wave_ratio_power"),
            SignalView("final__slow_wave_features", "high_frequency_envelope"),
            SignalView("final__spindle_features", "sigma_power"),
        ],
    ),
    StageSignalExpertSpec(
        name="rem",
        positive_label=4,
        signal_views=[
            SignalView("final__eye_movement_activity", "eye_movement_activity"),
            SignalView("final__eog_rem_features", "rem_band_power_fast"),
            SignalView("final__eog_rem_features", "sem_band_power_fast"),
            SignalView("final__emg_tone_features", "emg_tone_baseline"),
        ],
    ),
]


def build_default_stage_signal_expert_specs() -> list[StageSignalExpertSpec]:
    return [
        StageSignalExpertSpec(
            name=spec.name,
            positive_label=spec.positive_label,
            signal_views=list(spec.signal_views),
        )
        for spec in DEFAULT_STAGE_SIGNAL_EXPERT_SPECS
    ]


def build_staging_epoch_frame(reference_table: pd.DataFrame) -> pd.DataFrame:
    epoch_frame = build_staging_benchmark_table(
        reference_table,
        feature_columns=[],
    )
    if epoch_frame.empty:
        return epoch_frame
    return epoch_frame[["subject_id", "time_seconds", "window_end_seconds", "label.stage_30s"]].copy()


def run_stage_signal_expert_temporal_experiment(
    epoch_frame: pd.DataFrame,
    *,
    cache_dir: str | Path,
    spec: StageSignalExpertTemporalSpec,
    random_state: int = 7,
) -> StageSignalExpertTemporalRunResult:
    frame = epoch_frame.copy()
    frame = frame.dropna(subset=["subject_id", "time_seconds", "label.stage_30s"]).copy()
    frame["label.stage_30s"] = pd.to_numeric(frame["label.stage_30s"], errors="coerce")
    frame = frame.dropna(subset=["label.stage_30s"]).copy()
    frame = frame.loc[frame["label.stage_30s"] >= 0].copy()
    frame["label.stage_30s"] = frame["label.stage_30s"].astype(int)
    if frame.empty:
        return StageSignalExpertTemporalRunResult(
            summary=pd.DataFrame(),
            fold_metrics=pd.DataFrame(),
            predictions=pd.DataFrame(),
            expert_fold_metrics=pd.DataFrame(),
            metadata={"spec": asdict(spec), "rows": 0},
        )

    subject_signals = _load_subject_signal_views(
        cache_dir=cache_dir,
        subject_ids=sorted(frame["subject_id"].astype(str).unique().tolist()),
        expert_specs=spec.expert_specs,
    )
    expert_datasets = {
        expert.name: _build_expert_signal_dataset(
            frame,
            subject_signals=subject_signals,
            expert=expert,
            epoch_seconds=spec.epoch_seconds,
            window_points=spec.window_points,
        )
        for expert in spec.expert_specs
    }

    groups = frame["subject_id"].astype(str).to_numpy()
    y = frame["label.stage_30s"].to_numpy()
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
        print(f"[stage_signal_expert_temporal] fold={split.fold_index} start", flush=True)
        fit_train_idx, calib_idx = _resolve_inner_split_indices(
            split,
            groups=groups,
            random_state=random_state,
        )
        test_idx = np.asarray(split.test_idx, dtype=int)
        if fit_train_idx.size == 0 or calib_idx.size == 0 or test_idx.size == 0:
            print(f"[stage_signal_expert_temporal] fold={split.fold_index} skipped empty split", flush=True)
            continue

        train_frames: list[pd.DataFrame] = []
        calib_frames: list[pd.DataFrame] = []
        test_frames: list[pd.DataFrame] = []
        for expert in spec.expert_specs:
            dataset = expert_datasets.get(expert.name)
            if dataset is None:
                continue
            print(
                "[stage_signal_expert_temporal] "
                f"fold={split.fold_index} expert={expert.name} start "
                f"channels={dataset['X'].shape[2]} window_points={dataset['X'].shape[1]} rows={dataset['X'].shape[0]}",
                flush=True,
            )
            expert_result = _run_single_signal_expert(
                dataset,
                fit_train_idx=fit_train_idx,
                calib_idx=calib_idx,
                test_idx=test_idx,
                expert=expert,
                spec=spec,
                random_state=random_state,
            )
            train_frames.append(expert_result["train_frame"])
            calib_frames.append(expert_result["calib_frame"])
            test_frames.append(expert_result["test_frame"])
            expert_fold_rows.append(
                {
                    "experiment": spec.name,
                    "fold_index": split.fold_index,
                    "expert_name": expert.name,
                    "positive_label": int(expert.positive_label),
                    "n_channels": int(dataset["X"].shape[2]),
                    "window_points": int(dataset["X"].shape[1]),
                    "n_train": int(expert_result["n_train"]),
                    "n_calib": int(expert_result["n_calib"]),
                    "n_test": int(expert_result["n_test"]),
                    "n_positive_train": int(expert_result["n_positive_train"]),
                    "n_positive_calib": int(expert_result["n_positive_calib"]),
                    "n_positive_test": int(expert_result["n_positive_test"]),
                }
                | expert_result["metrics"]
            )
            print(
                "[stage_signal_expert_temporal] "
                f"fold={split.fold_index} expert={expert.name} done "
                f"macro_f1={expert_result['metrics']['macro_f1']:.4f}",
                flush=True,
            )

        if not train_frames or not calib_frames or not test_frames:
            print(f"[stage_signal_expert_temporal] fold={split.fold_index} skipped no expert outputs", flush=True)
            continue

        train_epoch = _merge_expert_epoch_frames(train_frames, subject_column="subject_id", time_column="time_seconds")
        calib_epoch = _merge_expert_epoch_frames(calib_frames, subject_column="subject_id", time_column="time_seconds")
        test_epoch = _merge_expert_epoch_frames(test_frames, subject_column="subject_id", time_column="time_seconds")
        if train_epoch.empty or calib_epoch.empty or test_epoch.empty:
            print(f"[stage_signal_expert_temporal] fold={split.fold_index} skipped empty merged epoch frames", flush=True)
            continue

        train_epoch = _attach_stage1_vote(train_epoch, spec.expert_specs)
        calib_epoch = _attach_stage1_vote(calib_epoch, spec.expert_specs)
        test_epoch = _attach_stage1_vote(test_epoch, spec.expert_specs)

        raw_pred_frame = test_epoch.copy()
        raw_pred_frame["y_pred"] = raw_pred_frame["y_pred_stage1"].to_numpy(dtype=int)
        raw_pred_frame["variant"] = "raw_expert_vote"
        prediction_frames.append(raw_pred_frame)
        raw_metrics = score_predictions(
            y_true=raw_pred_frame["y_true"].to_numpy(dtype=int),
            y_pred=raw_pred_frame["y_pred"].to_numpy(dtype=int),
            proba=None,
        )
        fold_rows.append(
            {
                "experiment": spec.name,
                "fold_index": split.fold_index,
                "variant": "raw_expert_vote",
                "stage2_model_family": spec.stage2_model_family,
                "context_radius": int(spec.context_radius),
                "n_train": int(fit_train_idx.size),
                "n_calib": int(calib_idx.size),
                "n_test": int(test_idx.size),
                "n_epoch_train": int(len(train_epoch)),
                "n_epoch_calib": int(len(calib_epoch)),
                "n_epoch_test": int(len(test_epoch)),
                "n_experts": int(len(spec.expert_specs)),
            }
            | raw_metrics
        )

        stage2_result = _run_stage2(
            train_epoch,
            calib_epoch,
            test_epoch,
            expert_specs=spec.expert_specs,
            stage2_model_family=spec.stage2_model_family,
            context_radius=spec.context_radius,
            epochs=spec.epochs,
            min_epochs=spec.min_epochs,
            patience=spec.patience,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            hidden_channels=spec.hidden_channels,
            kernel_size=spec.kernel_size,
            dropout=spec.dropout,
            tcn_levels=spec.tcn_levels,
            weighted_sampling=spec.weighted_sampling,
            random_state=random_state,
        )
        if stage2_result is None:
            print(f"[stage_signal_expert_temporal] fold={split.fold_index} skipped stage2 unavailable", flush=True)
            continue
        stage2_pred_frame = stage2_result["predictions"]
        stage2_pred_frame["variant"] = "temporal_combiner"
        prediction_frames.append(stage2_pred_frame)
        fold_rows.append(
            {
                "experiment": spec.name,
                "fold_index": split.fold_index,
                "variant": "temporal_combiner",
                "stage2_model_family": spec.stage2_model_family,
                "context_radius": int(spec.context_radius),
                "n_train": int(fit_train_idx.size),
                "n_calib": int(calib_idx.size),
                "n_test": int(test_idx.size),
                "n_epoch_train": int(len(train_epoch)),
                "n_epoch_calib": int(len(calib_epoch)),
                "n_epoch_test": int(len(test_epoch)),
                "n_experts": int(len(spec.expert_specs)),
                "n_stage2_rows_train": int(stage2_result["n_stage2_rows_train"]),
                "n_stage2_rows_early_val": int(stage2_result["n_stage2_rows_early_val"]),
                "n_stage2_rows_test": int(stage2_result["n_stage2_rows_test"]),
                "n_stage2_features": int(stage2_result["n_stage2_features"]),
            }
            | stage2_result["metrics"]
        )
        print(
            "[stage_signal_expert_temporal] "
            f"fold={split.fold_index} done "
            f"raw_macro_f1={raw_metrics['macro_f1']:.4f} "
            f"stage2_macro_f1={stage2_result['metrics']['macro_f1']:.4f}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    expert_fold_metrics = pd.DataFrame(expert_fold_rows)
    predictions = pd.concat(prediction_frames, axis=0, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary = _summarize_fold_metrics(fold_metrics)
    metadata = {
        "spec": asdict(spec),
        "rows": int(len(frame)),
        "subjects": sorted(frame["subject_id"].astype(str).unique().tolist()),
    }
    return StageSignalExpertTemporalRunResult(
        summary=summary,
        fold_metrics=fold_metrics,
        predictions=predictions,
        expert_fold_metrics=expert_fold_metrics,
        metadata=metadata,
    )


def _load_subject_signal_views(
    *,
    cache_dir: str | Path,
    subject_ids: list[str],
    expert_specs: list[StageSignalExpertSpec],
) -> dict[str, dict[tuple[str, str], tuple[np.ndarray, float]]]:
    store = ParquetAlgoStore()
    needed = {(view.algo_name, view.signal_name) for expert in expert_specs for view in expert.signal_views}
    out: dict[str, dict[tuple[str, str], tuple[np.ndarray, float]]] = {}
    for subject_id in subject_ids:
        subject_root = Path(cache_dir) / subject_id
        view_map: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        loaded_algos: dict[str, Any] = {}
        for algo_name, signal_name in needed:
            if algo_name not in loaded_algos:
                loaded_algos[algo_name] = store.load_one(str(subject_root), algo_name)
            obj = loaded_algos.get(algo_name)
            if obj is None or signal_name not in obj.signals:
                continue
            series = obj.signals[signal_name]
            view_map[(algo_name, signal_name)] = (
                np.asarray(series.values, dtype=float),
                float(series.sample_rate),
            )
        out[subject_id] = view_map
    return out


def _build_expert_signal_dataset(
    epoch_frame: pd.DataFrame,
    *,
    subject_signals: dict[str, dict[tuple[str, str], tuple[np.ndarray, float]]],
    expert: StageSignalExpertSpec,
    epoch_seconds: float,
    window_points: int,
) -> dict[str, Any]:
    seq_list: list[np.ndarray] = []
    valid_indices: list[int] = []
    for row_idx, row in epoch_frame.reset_index(drop=True).iterrows():
        subject_id = str(row["subject_id"])
        start_sec = float(row["time_seconds"])
        subject_map = subject_signals.get(subject_id, {})
        channel_windows: list[np.ndarray] = []
        ok = True
        for view in expert.signal_views:
            signal_key = (view.algo_name, view.signal_name)
            signal_tuple = subject_map.get(signal_key)
            if signal_tuple is None:
                ok = False
                break
            values, fs = signal_tuple
            window = _extract_resampled_window(
                values,
                sample_rate=fs,
                start_seconds=start_sec,
                window_seconds=epoch_seconds,
                target_points=window_points,
            )
            if window is None:
                ok = False
                break
            channel_windows.append(window)
        if not ok or not channel_windows:
            continue
        seq_list.append(np.stack(channel_windows, axis=1))
        valid_indices.append(int(row_idx))

    if not seq_list:
        return {
            "X": np.empty((0, int(window_points), len(expert.signal_views)), dtype=float),
            "frame": epoch_frame.iloc[[]].copy(),
        }
    dataset_frame = epoch_frame.iloc[valid_indices].reset_index(drop=True).copy()
    dataset_frame["_master_index"] = np.asarray(valid_indices, dtype=int)
    return {
        "X": np.stack(seq_list, axis=0),
        "frame": dataset_frame,
    }


def _extract_resampled_window(
    values: np.ndarray,
    *,
    sample_rate: float,
    start_seconds: float,
    window_seconds: float,
    target_points: int,
) -> np.ndarray | None:
    if values.size == 0 or sample_rate <= 0:
        return None
    start_idx = max(int(np.floor(start_seconds * sample_rate)), 0)
    end_idx = min(int(np.ceil((start_seconds + window_seconds) * sample_rate)), values.size)
    if end_idx - start_idx < 2:
        return None
    segment = np.asarray(values[start_idx:end_idx], dtype=float)
    finite_mask = np.isfinite(segment)
    if finite_mask.sum() < 2:
        return None
    if not np.all(finite_mask):
        valid_idx = np.flatnonzero(finite_mask)
        segment = np.interp(np.arange(segment.size, dtype=float), valid_idx.astype(float), segment[finite_mask])
    source_time = (start_idx / sample_rate) + (np.arange(segment.size, dtype=float) / sample_rate)
    target_time = start_seconds + (np.arange(target_points, dtype=float) + 0.5) * (window_seconds / float(target_points))
    left = float(segment[0])
    right = float(segment[-1])
    return np.interp(target_time, source_time, segment, left=left, right=right).astype(float)


def _resolve_inner_split_indices(split, *, groups: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(split.train_idx, dtype=int)
    if split.validation_idx is not None:
        return train_idx, np.asarray(split.validation_idx, dtype=int)
    if getattr(split, "inner_validation", "none") == "none":
        return train_idx, train_idx
    candidate_groups = groups[train_idx].astype(str)
    unique_groups = sorted(np.unique(candidate_groups).tolist())
    if len(unique_groups) < 2:
        return train_idx, np.asarray([], dtype=int)
    offset = int((split.fold_index - 1 + random_state) % len(unique_groups))
    calib_group = unique_groups[offset]
    calib_mask = candidate_groups == calib_group
    return train_idx[~calib_mask], train_idx[calib_mask]


def _run_single_signal_expert(
    dataset: dict[str, Any],
    *,
    fit_train_idx: np.ndarray,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    expert: StageSignalExpertSpec,
    spec: StageSignalExpertTemporalSpec,
    random_state: int,
) -> dict[str, Any]:
    frame = dataset["frame"]
    X_all = np.asarray(dataset["X"], dtype=float)
    row_lookup = {
        (str(row.subject_id), float(row.time_seconds)): idx
        for idx, row in enumerate(frame.itertuples(index=False))
    }
    print(
        "[stage_signal_expert_temporal] "
        f"expert={expert.name} lookup_start total_rows={X_all.shape[0]}",
        flush=True,
    )
    train_rows = _lookup_dataset_rows(frame, fit_train_idx, row_lookup)
    calib_rows = _lookup_dataset_rows(frame, calib_idx, row_lookup)
    test_rows = _lookup_dataset_rows(frame, test_idx, row_lookup)
    if train_rows.size == 0 or calib_rows.size == 0 or test_rows.size == 0:
        raise ValueError(f"No signal rows for expert {expert.name}")
    print(
        "[stage_signal_expert_temporal] "
        f"expert={expert.name} lookup_done "
        f"train_rows={train_rows.size} calib_rows={calib_rows.size} test_rows={test_rows.size}",
        flush=True,
    )

    X_train = X_all[train_rows]
    X_calib = X_all[calib_rows]
    X_test = X_all[test_rows]
    print(
        "[stage_signal_expert_temporal] "
        f"expert={expert.name} tensors_sliced "
        f"train_shape={X_train.shape} calib_shape={X_calib.shape} test_shape={X_test.shape}",
        flush=True,
    )
    X_train, X_calib, X_test = _normalize_signal_tensors(X_train, X_calib, X_test)
    print(f"[stage_signal_expert_temporal] expert={expert.name} tensors_normalized", flush=True)

    train_frame = frame.iloc[train_rows].reset_index(drop=True)
    calib_frame = frame.iloc[calib_rows].reset_index(drop=True)
    test_frame = frame.iloc[test_rows].reset_index(drop=True)
    y_train = (train_frame["label.stage_30s"].to_numpy(dtype=int) == int(expert.positive_label)).astype(int)
    y_calib = (calib_frame["label.stage_30s"].to_numpy(dtype=int) == int(expert.positive_label)).astype(int)
    y_test = (test_frame["label.stage_30s"].to_numpy(dtype=int) == int(expert.positive_label)).astype(int)
    print(
        "[stage_signal_expert_temporal] "
        f"expert={expert.name} labels_ready "
        f"positive_train={int(y_train.sum())} positive_calib={int(y_calib.sum())} positive_test={int(y_test.sum())}",
        flush=True,
    )

    train_proba, calib_proba, test_proba, test_pred = _fit_binary_signal_model(
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        model_arch=spec.stage1_model_arch,
        epochs=spec.epochs,
        min_epochs=spec.min_epochs,
        patience=spec.patience,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        hidden_channels=spec.hidden_channels,
        kernel_size=spec.kernel_size,
        dropout=spec.dropout,
        tcn_levels=spec.tcn_levels,
        weighted_sampling=spec.weighted_sampling,
        random_state=random_state,
    )
    print(f"[stage_signal_expert_temporal] expert={expert.name} train_done", flush=True)
    metrics = score_predictions(
        y_true=y_test,
        y_pred=test_pred,
        proba=np.column_stack([1.0 - test_proba, test_proba]),
    )

    train_out = train_frame[["subject_id", "time_seconds", "label.stage_30s"]].rename(columns={"label.stage_30s": "y_true"}).copy()
    train_out[f"expert_{expert.name}_proba"] = train_proba
    calib_out = calib_frame[["subject_id", "time_seconds", "label.stage_30s"]].rename(columns={"label.stage_30s": "y_true"}).copy()
    calib_out[f"expert_{expert.name}_proba"] = calib_proba
    test_out = test_frame[["subject_id", "time_seconds", "label.stage_30s"]].rename(columns={"label.stage_30s": "y_true"}).copy()
    test_out[f"expert_{expert.name}_proba"] = test_proba
    test_out[f"expert_{expert.name}_binary_pred"] = test_pred

    return {
        "train_frame": train_out,
        "calib_frame": calib_out,
        "test_frame": test_out,
        "metrics": metrics,
        "n_train": int(X_train.shape[0]),
        "n_calib": int(X_calib.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_positive_train": int(y_train.sum()),
        "n_positive_calib": int(y_calib.sum()),
        "n_positive_test": int(y_test.sum()),
        "train_proba_mean": float(np.mean(train_proba)) if train_proba.size else float("nan"),
    }


def _lookup_dataset_rows(frame: pd.DataFrame, base_indices: np.ndarray, row_lookup: dict[tuple[str, float], int]) -> np.ndarray:
    if base_indices.size == 0 or "_master_index" not in frame.columns:
        return np.asarray([], dtype=int)
    wanted = set(int(idx) for idx in np.asarray(base_indices, dtype=int).tolist())
    rows = [row_idx for row_idx, master_idx in enumerate(frame["_master_index"].to_numpy(dtype=int, copy=False)) if int(master_idx) in wanted]
    return np.asarray(rows, dtype=int)


def _normalize_signal_tensors(X_train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    train = np.asarray(X_train, dtype=float).copy()
    other_list = [np.asarray(arr, dtype=float).copy() for arr in others]
    n_channels = train.shape[2]
    med = np.nanmedian(train.reshape(-1, n_channels), axis=0)
    q25 = np.nanpercentile(train.reshape(-1, n_channels), 25, axis=0)
    q75 = np.nanpercentile(train.reshape(-1, n_channels), 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0

    def _apply(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out = (out - med.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    return tuple([_apply(train)] + [_apply(arr) for arr in other_list])


def _fit_binary_signal_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    *,
    model_arch: str,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    kernel_size: int,
    dropout: float,
    tcn_levels: int,
    weighted_sampling: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(y_train)
    if unique.size < 2:
        constant = float(unique[0]) if unique.size == 1 else 0.0
        return (
            np.full(X_train.shape[0], constant, dtype=float),
            np.full(X_calib.shape[0], constant, dtype=float),
            np.full(X_test.shape[0], constant, dtype=float),
            np.full(X_test.shape[0], int(round(constant)), dtype=int),
        )

    _require_torch()
    print(
        "[stage_signal_expert_temporal] "
        f"binary_fit_start arch={model_arch} "
        f"train={X_train.shape} calib={X_calib.shape} test={X_test.shape} "
        f"epochs={epochs} batch_size={batch_size} hidden_channels={hidden_channels}",
        flush=True,
    )
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
        model_arch=model_arch,
        history_steps=0,
        current_steps=int(X_train.shape[1]),
        future_steps=0,
        tcn_levels=tcn_levels,
        weighted_sampling=weighted_sampling,
        random_state=random_state,
    )
    print("[stage_signal_expert_temporal] binary_fit_model_trained", flush=True)
    y_train_pred, train_proba_full = _predict_cnn_classifier(model, encoder, X_train)
    _, calib_proba_full = _predict_cnn_classifier(model, encoder, X_calib)
    y_test_pred, test_proba_full = _predict_cnn_classifier(model, encoder, X_test)
    print("[stage_signal_expert_temporal] binary_fit_predictions_ready", flush=True)
    train_proba = _positive_column(train_proba_full, encoder.classes_)
    calib_proba = _positive_column(calib_proba_full, encoder.classes_)
    test_proba = _positive_column(test_proba_full, encoder.classes_)
    return train_proba, calib_proba, test_proba, np.asarray(y_test_pred, dtype=int)


def _positive_column(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes)
    positive_idx = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else int(classes.size - 1)
    return proba[:, positive_idx]


def _merge_expert_epoch_frames(frames: list[pd.DataFrame], *, subject_column: str, time_column: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for frame in frames:
        keep_cols = [subject_column, time_column, "y_true"] if merged is None else [subject_column, time_column]
        expert_cols = [col for col in frame.columns if col.startswith("expert_")]
        block = frame[keep_cols + expert_cols].copy()
        if merged is None:
            merged = block
        else:
            merged = merged.merge(block, on=[subject_column, time_column], how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values([subject_column, time_column], kind="stable").reset_index(drop=True)


def _attach_stage1_vote(frame: pd.DataFrame, expert_specs: list[StageSignalExpertSpec]) -> pd.DataFrame:
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
    return out


def _run_stage2(
    train_epoch: pd.DataFrame,
    calib_epoch: pd.DataFrame,
    test_epoch: pd.DataFrame,
    *,
    expert_specs: list[StageSignalExpertSpec],
    stage2_model_family: str,
    context_radius: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    kernel_size: int,
    dropout: float,
    tcn_levels: int,
    weighted_sampling: bool,
    random_state: int,
) -> dict[str, Any] | None:
    if stage2_model_family == "tcn":
        train_seq = _build_context_sequence_dataset(train_epoch, expert_specs=expert_specs, context_radius=context_radius)
        calib_seq = _build_context_sequence_dataset(calib_epoch, expert_specs=expert_specs, context_radius=context_radius)
        test_seq = _build_context_sequence_dataset(test_epoch, expert_specs=expert_specs, context_radius=context_radius)
        if train_seq is None or test_seq is None or np.unique(train_seq["y"]).size < 2:
            return None
        early_val_seq = train_seq
        if calib_seq is not None and np.unique(calib_seq["y"]).size >= 2:
            early_val_seq = calib_seq
        _require_torch()
        model, encoder, _ = _train_cnn_classifier(
            train_seq["X"],
            train_seq["y"],
            X_early_val=early_val_seq["X"],
            y_early_val=early_val_seq["y"],
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            model_arch="dilated_tcn",
            history_steps=context_radius,
            current_steps=1,
            future_steps=context_radius,
            tcn_levels=tcn_levels,
            weighted_sampling=weighted_sampling,
            random_state=random_state,
        )
        y_pred, _ = _predict_cnn_classifier(model, encoder, test_seq["X"])
        refined = _apply_stage2_epoch_predictions(
            test_epoch,
            updates=test_seq["metadata"][["subject_id", "time_seconds"]].assign(y_pred=np.asarray(y_pred, dtype=int)),
        )
        metrics = score_predictions(
            y_true=refined["y_true"].to_numpy(dtype=int),
            y_pred=refined["y_pred"].to_numpy(dtype=int),
            proba=None,
        )
        return {
            "predictions": refined,
            "metrics": metrics,
            "n_stage2_rows_train": int(train_seq["X"].shape[0]),
            "n_stage2_rows_early_val": int(early_val_seq["X"].shape[0]),
            "n_stage2_rows_test": int(test_seq["X"].shape[0]),
            "n_stage2_features": int(train_seq["X"].shape[2]),
        }

    train_context = _build_context_frame(train_epoch, expert_specs=expert_specs, context_radius=context_radius)
    calib_context = _build_context_frame(calib_epoch, expert_specs=expert_specs, context_radius=context_radius)
    test_context = _build_context_frame(test_epoch, expert_specs=expert_specs, context_radius=context_radius)
    if train_context.empty or test_context.empty or train_context["y_true"].nunique() < 2:
        return None
    context_cols = _context_feature_columns(expert_specs, context_radius=context_radius)
    stage2 = build_model(
        model_family=stage2_model_family,
        task="classification",
        random_state=random_state,
        class_weight="balanced",
    )
    stage2.fit(train_context[context_cols], train_context["y_true"])
    y_pred = np.asarray(stage2.predict(test_context[context_cols]), dtype=int)
    refined = _apply_stage2_epoch_predictions(
        test_epoch,
        updates=test_context[["subject_id", "time_seconds"]].assign(y_pred=y_pred),
    )
    metrics = score_predictions(
        y_true=refined["y_true"].to_numpy(dtype=int),
        y_pred=refined["y_pred"].to_numpy(dtype=int),
        proba=None,
    )
    return {
        "predictions": refined,
        "metrics": metrics,
        "n_stage2_rows_train": int(len(train_context)),
        "n_stage2_rows_early_val": int(len(calib_context)),
        "n_stage2_rows_test": int(len(test_context)),
        "n_stage2_features": int(len(context_cols)),
    }


def _build_context_frame(epoch_frame: pd.DataFrame, *, expert_specs: list[StageSignalExpertSpec], context_radius: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prob_cols = [f"expert_{expert.name}_proba" for expert in expert_specs]
    window_length = (2 * int(context_radius)) + 1
    for subject_id, subject_df in epoch_frame.groupby("subject_id", sort=False):
        ordered = subject_df.sort_values("time_seconds", kind="stable").reset_index(drop=True)
        times = ordered["time_seconds"].to_numpy(dtype=float, copy=False)
        if len(ordered) < window_length:
            continue
        for center in range(context_radius, len(ordered) - context_radius):
            start = center - context_radius
            stop = center + context_radius + 1
            seq_times = times[start:stop]
            if seq_times.size != window_length:
                continue
            if np.any(np.abs(np.diff(seq_times) - 30.0) > 0.25):
                continue
            block = ordered.iloc[start:stop]
            row: dict[str, object] = {
                "subject_id": str(subject_id),
                "time_seconds": float(times[center]),
                "y_true": int(ordered["y_true"].iloc[center]),
                "y_pred_stage1": int(ordered["y_pred_stage1"].iloc[center]),
            }
            for offset in range(-context_radius, context_radius + 1):
                offset_label = _offset_label(offset)
                source = block.iloc[offset + context_radius]
                for prob_col in prob_cols:
                    row[f"{offset_label}.{prob_col}"] = float(source[prob_col])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_context_sequence_dataset(epoch_frame: pd.DataFrame, *, expert_specs: list[StageSignalExpertSpec], context_radius: int) -> dict[str, Any] | None:
    seq_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict[str, object]] = []
    prob_cols = [f"expert_{expert.name}_proba" for expert in expert_specs]
    window_length = (2 * int(context_radius)) + 1
    for subject_id, subject_df in epoch_frame.groupby("subject_id", sort=False):
        ordered = subject_df.sort_values("time_seconds", kind="stable").reset_index(drop=True)
        times = ordered["time_seconds"].to_numpy(dtype=float, copy=False)
        probs = ordered[prob_cols].to_numpy(dtype=float, copy=False)
        y_true = ordered["y_true"].to_numpy(dtype=int, copy=False)
        y_stage1 = ordered["y_pred_stage1"].to_numpy(dtype=int, copy=False)
        if len(ordered) < window_length:
            continue
        for center in range(context_radius, len(ordered) - context_radius):
            start = center - context_radius
            stop = center + context_radius + 1
            seq_times = times[start:stop]
            if seq_times.size != window_length:
                continue
            if np.any(np.abs(np.diff(seq_times) - 30.0) > 0.25):
                continue
            seq_list.append(probs[start:stop])
            y_list.append(int(y_true[center]))
            meta_rows.append(
                {
                    "subject_id": str(subject_id),
                    "time_seconds": float(times[center]),
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


def _context_feature_columns(expert_specs: list[StageSignalExpertSpec], *, context_radius: int) -> list[str]:
    cols: list[str] = []
    for offset in range(-context_radius, context_radius + 1):
        offset_label = _offset_label(offset)
        for expert in expert_specs:
            cols.append(f"{offset_label}.expert_{expert.name}_proba")
    return cols


def _offset_label(offset: int) -> str:
    if offset < 0:
        return f"ctx_m{abs(offset)}"
    if offset > 0:
        return f"ctx_p{offset}"
    return "ctx_0"


def _apply_stage2_epoch_predictions(epoch_frame: pd.DataFrame, *, updates: pd.DataFrame) -> pd.DataFrame:
    refined = epoch_frame.copy()
    refined["y_pred"] = refined["y_pred_stage1"].to_numpy(dtype=int)
    refined = refined.merge(updates[["subject_id", "time_seconds", "y_pred"]], on=["subject_id", "time_seconds"], how="left", suffixes=("", "__stage2"))
    mask = refined["y_pred__stage2"].notna()
    refined.loc[mask, "y_pred"] = refined.loc[mask, "y_pred__stage2"].astype(int)
    refined = refined.drop(columns=["y_pred__stage2"])
    return refined


def _summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    group_cols = ["variant", "stage2_model_family", "context_radius"]
    skip_cols = {"experiment", "fold_index", *group_cols}
    numeric_cols = [col for col in fold_df.columns if col not in skip_cols and pd.api.types.is_numeric_dtype(fold_df[col])]
    for keys, group in fold_df.groupby(group_cols, sort=False):
        row: dict[str, object] = {
            "variant": keys[0],
            "stage2_model_family": keys[1],
            "context_radius": keys[2],
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
