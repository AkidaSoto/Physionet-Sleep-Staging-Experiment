from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

from physionet_sleep.experiments.normalization import apply_normalization
from physionet_sleep.experiments.sequence_data import (
    build_centered_sequence_dataset,
    build_epoch_aligned_sequence_dataset,
    build_hierarchical_epoch_sequence_dataset,
)
from physionet_sleep.experiments.tabular_runner import build_validation_splits, score_predictions, summarize_fold_metrics


@dataclass(slots=True)
class SequenceExperimentSpec:
    name: str
    target_column: str
    feature_columns: list[str]
    task: str = "classification"
    model_arch: str = "simple_cnn"
    sequence_mode: str = "centered"
    context_radius: int = 2
    history_steps: int = 0
    current_steps: int = 1
    future_steps: int = 0
    epoch_steps: int = 5
    normalization: str = "medianiqr"
    cv_folds: int = 5
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_channels: int = 32
    kernel_size: int = 3
    dropout: float = 0.2
    tcn_levels: int = 4
    min_epochs: int = 10
    patience: int = 8
    internal_val_subjects: int = 1
    weighted_sampling: bool = True
    expected_step_seconds: float = 6.0
    step_tolerance_seconds: float = 0.25


@dataclass(slots=True)
class SequenceRunResult:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any]


def run_sequence_cnn_experiment(
    feature_table: pd.DataFrame,
    *,
    spec: SequenceExperimentSpec,
    subject_column: str = "subject_id",
    time_column: str = "time_seconds",
    validation: str = "kfold",
    random_state: int = 7,
) -> SequenceRunResult:
    _require_torch()
    if spec.task != "classification":
        raise ValueError(f"Unsupported task: {spec.task}")

    frame = _prepare_frame(
        feature_table,
        feature_columns=spec.feature_columns,
        target_column=spec.target_column,
        subject_column=subject_column,
        time_column=time_column,
    )
    if frame.empty:
        return SequenceRunResult(
            summary=pd.DataFrame(),
            fold_metrics=pd.DataFrame(),
            predictions=pd.DataFrame(),
            metadata={"spec": asdict(spec), "validation": validation, "rows": 0},
        )

    groups = frame[subject_column].astype(str).to_numpy()
    y_all = frame[spec.target_column].to_numpy()
    splits = list(
        build_validation_splits(
            groups=groups,
            y=y_all,
            folds=spec.cv_folds,
            validation=validation,
            random_state=random_state,
        )
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_index, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df_raw = frame.iloc[train_idx].copy()
        val_df_raw = frame.iloc[val_idx].copy()
        fit_train_raw, fit_val_raw = _split_inner_validation_subjects(
            train_df_raw,
            subject_column=subject_column,
            fold_index=fold_index,
            n_val_subjects=spec.internal_val_subjects,
            random_state=random_state,
        )

        fit_train_df, fit_val_df, norm_stats = apply_normalization(
            fit_train_raw,
            fit_val_raw,
            feature_columns=spec.feature_columns,
            norm_name=spec.normalization,
            group_column=subject_column,
        )
        _, val_df, _ = apply_normalization(
            fit_train_raw,
            val_df_raw,
            feature_columns=spec.feature_columns,
            norm_name=spec.normalization,
            group_column=subject_column,
        )
        fit_train_df, fit_val_df = _impute_features(
            fit_train_df,
            fit_val_df,
            feature_columns=spec.feature_columns,
        )
        fit_train_for_eval, val_df = _impute_features(
            fit_train_df,
            val_df,
            feature_columns=spec.feature_columns,
        )

        train_dataset = _build_sequence_dataset(
            fit_train_df,
            spec=spec,
            subject_column=subject_column,
            time_column=time_column,
        )
        train_eval_dataset = _build_sequence_dataset(
            fit_train_for_eval,
            spec=spec,
            subject_column=subject_column,
            time_column=time_column,
        )
        fit_val_dataset = _build_sequence_dataset(
            fit_val_df,
            spec=spec,
            subject_column=subject_column,
            time_column=time_column,
        )
        val_dataset = _build_sequence_dataset(
            val_df,
            spec=spec,
            subject_column=subject_column,
            time_column=time_column,
        )
        if train_dataset.X.shape[0] == 0 or fit_val_dataset.X.shape[0] == 0 or val_dataset.X.shape[0] == 0:
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
            history_steps=_history_steps_for_spec(spec),
            current_steps=_current_steps_for_spec(spec),
            future_steps=_future_steps_for_spec(spec),
            tcn_levels=spec.tcn_levels,
            weighted_sampling=spec.weighted_sampling,
            random_state=random_state,
        )

        y_train_pred, train_proba = _predict_cnn_classifier(model, encoder, train_eval_dataset.X)
        y_pred, proba = _predict_cnn_classifier(model, encoder, val_dataset.X)
        train_metrics = score_predictions(y_true=train_eval_dataset.y, y_pred=y_train_pred, proba=train_proba)
        metrics = score_predictions(y_true=val_dataset.y, y_pred=y_pred, proba=proba)
        fold_rows.append(
            {
                "experiment": spec.name,
                "model_family": "cnn",
                "norm_name": spec.normalization,
                "norm_scope": norm_stats.get("scope", ""),
                "fold_index": fold_index,
                "n_train": int(train_dataset.X.shape[0]),
                "n_val": int(val_dataset.X.shape[0]),
                "n_early_val": int(fit_val_dataset.X.shape[0]),
                "n_features_used": int(len(spec.feature_columns)),
                "sequence_length": int(train_dataset.window_length),
                "n_subjects_train": int(fit_train_df[subject_column].nunique()),
                "n_subjects_early_val": int(fit_val_df[subject_column].nunique()),
                "n_subjects_val": int(val_df[subject_column].nunique()),
                "best_epoch": int(training_info["best_epoch"]),
                "early_val_macro_f1": float(training_info["best_val_macro_f1"]),
                "early_val_loss": float(training_info["best_val_loss"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "train_accuracy": float(train_metrics["accuracy"]),
            }
            | metrics
        )

        pred_frame = val_dataset.metadata.copy()
        pred_frame["experiment"] = spec.name
        pred_frame["model_family"] = "cnn"
        pred_frame["norm_name"] = spec.normalization
        pred_frame["fold_index"] = fold_index
        pred_frame["y_true"] = val_dataset.y
        pred_frame["y_pred"] = y_pred
        prediction_frames.append(pred_frame)

    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        summary = pd.DataFrame()
    else:
        summary = pd.DataFrame(
            [
                summarize_fold_metrics(
                    fold_metrics,
                    experiment_name=spec.name,
                    norm_name=spec.normalization,
                    model_family="cnn",
                )
            ]
        )
    predictions = pd.concat(prediction_frames, axis=0, ignore_index=True) if prediction_frames else pd.DataFrame()
    return SequenceRunResult(
        summary=summary,
        fold_metrics=fold_metrics,
        predictions=predictions,
        metadata={
            "spec": asdict(spec),
            "validation": validation,
            "rows": int(len(frame)),
            "subjects": sorted(frame[subject_column].astype(str).unique().tolist()),
        },
    )


def _prepare_frame(
    feature_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str,
    time_column: str,
) -> pd.DataFrame:
    required = [subject_column, time_column, target_column, *feature_columns]
    missing = [col for col in required if col not in feature_table.columns]
    if missing:
        return pd.DataFrame()
    frame = feature_table[required].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[subject_column, time_column, target_column])
    if target_column in {"label.stage_seconds", "label.stage_30s"}:
        frame = frame.loc[pd.to_numeric(frame[target_column], errors="coerce") >= 0].copy()
    return frame


def _history_steps_for_spec(spec: SequenceExperimentSpec) -> int:
    if spec.sequence_mode == "epoch_aligned":
        return int(spec.history_steps)
    return int(spec.context_radius)


def _current_steps_for_spec(spec: SequenceExperimentSpec) -> int:
    if spec.sequence_mode == "epoch_aligned":
        return int(spec.current_steps)
    return 1


def _future_steps_for_spec(spec: SequenceExperimentSpec) -> int:
    if spec.sequence_mode == "epoch_aligned":
        return int(spec.future_steps)
    return int(spec.context_radius)


def _build_sequence_dataset(
    frame: pd.DataFrame,
    *,
    spec: SequenceExperimentSpec,
    subject_column: str,
    time_column: str,
):
    if spec.sequence_mode == "centered":
        return build_centered_sequence_dataset(
            frame,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column=subject_column,
            time_column=time_column,
            context_radius=spec.context_radius,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
    if spec.sequence_mode == "epoch_aligned":
        return build_epoch_aligned_sequence_dataset(
            frame,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column=subject_column,
            time_column=time_column,
            history_steps=spec.history_steps,
            current_steps=spec.current_steps,
            future_steps=spec.future_steps,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
    if spec.sequence_mode == "hierarchical_epochs":
        return build_hierarchical_epoch_sequence_dataset(
            frame,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column=subject_column,
            time_column=time_column,
            history_epochs=spec.history_steps,
            current_epochs=spec.current_steps,
            future_epochs=spec.future_steps,
            epoch_steps=spec.epoch_steps,
            expected_step_seconds=spec.expected_step_seconds,
            step_tolerance_seconds=spec.step_tolerance_seconds,
        )
    raise ValueError(f"Unsupported sequence_mode: {spec.sequence_mode}")


def _impute_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df.loc[:, feature_columns] = imputer.fit_transform(train_df[feature_columns])
    val_df.loc[:, feature_columns] = imputer.transform(val_df[feature_columns])
    return train_df, val_df


def _split_inner_validation_subjects(
    train_df: pd.DataFrame,
    *,
    subject_column: str,
    fold_index: int,
    n_val_subjects: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_val_subjects <= 0:
        raise ValueError("n_val_subjects must be positive")
    subjects = sorted(train_df[subject_column].astype(str).unique().tolist())
    if len(subjects) <= n_val_subjects:
        raise ValueError("Not enough training subjects for internal validation split")
    start = int((fold_index - 1 + random_state) % len(subjects))
    val_subjects = {subjects[(start + offset) % len(subjects)] for offset in range(n_val_subjects)}
    val_mask = train_df[subject_column].astype(str).isin(val_subjects)
    fit_val_df = train_df.loc[val_mask].copy()
    fit_train_df = train_df.loc[~val_mask].copy()
    return fit_train_df, fit_val_df


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError("PyTorch is required for sequence_cnn but is not installed.") from exc


def _train_cnn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_early_val: np.ndarray,
    y_early_val: np.ndarray,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    kernel_size: int,
    dropout: float,
    model_arch: str,
    history_steps: int,
    current_steps: int,
    future_steps: int,
    tcn_levels: int,
    weighted_sampling: bool,
    random_state: int,
):
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(int(random_state))

    encoder = _LabelEncoder(y_train)
    y_encoded = encoder.transform(y_train)
    y_early_val_encoded = encoder.transform(y_early_val)

    model = _SequenceClassifier(
        n_features=int(X_train.shape[-1]),
        n_classes=int(len(encoder.classes_)),
        hidden_channels=int(hidden_channels),
        kernel_size=int(kernel_size),
        dropout=float(dropout),
        model_arch=model_arch,
        history_steps=int(history_steps),
        current_steps=int(current_steps),
        future_steps=int(future_steps),
        tcn_levels=int(tcn_levels),
        epoch_steps=int(X_train.shape[2]) if X_train.ndim == 4 else 0,
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long),
    )
    sampler = None
    if weighted_sampling:
        sample_weights = _compute_sample_weights(y_encoded)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=int(len(sample_weights)),
            replacement=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=sampler is None,
        sampler=sampler,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    # Balance classes once: the inverse-frequency sampler already equalises classes per batch, so a
    # class-weighted loss on top would weight rare classes by ~1/count^2.
    criterion = nn.CrossEntropyLoss(weight=None if weighted_sampling else _compute_class_weights(y_encoded))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_state = None
    best_val_loss = float("inf")
    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    X_early_val_t = torch.tensor(X_early_val, dtype=torch.float32)
    y_early_val_t = torch.tensor(y_early_val_encoded, dtype=torch.long)

    model.train()
    for epoch in range(1, max(int(epochs), 1) + 1):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        val_loss, val_macro_f1 = _evaluate_model(
            model,
            X_early_val_t,
            y_early_val_t,
            y_early_val,
            criterion=criterion,
            encoder=encoder,
        )
        scheduler.step(val_loss)
        improved = (val_macro_f1 > best_val_macro_f1) or (
            np.isclose(val_macro_f1, best_val_macro_f1) and val_loss < best_val_loss
        )
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_loss = float(val_loss)
            best_val_macro_f1 = float(val_macro_f1)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= int(min_epochs) and stale_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), encoder, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
    }


def _compute_class_weights(y_encoded: np.ndarray):
    import torch

    counts = np.bincount(y_encoded, minlength=int(np.max(y_encoded)) + 1).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (counts * counts.size)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _compute_sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_encoded, minlength=int(np.max(y_encoded)) + 1).astype(np.float64)
    counts[counts <= 0] = 1.0
    inv = 1.0 / counts
    return inv[y_encoded]


def _evaluate_model(model, X_val_t, y_val_t, y_val_raw, *, criterion, encoder) -> tuple[float, float]:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        loss = criterion(logits, y_val_t).item()
        pred_encoded = torch.argmax(logits, dim=1).cpu().numpy()
    y_pred = encoder.inverse_transform(pred_encoded)
    macro_f1 = float(f1_score(y_val_raw, y_pred, average="macro", zero_division=0))
    model.train()
    return float(loss), macro_f1


def _predict_cnn_classifier(model, encoder, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_pred, proba, _, _ = _predict_cnn_classifier_detailed(model, encoder, X_val)
    return y_pred, proba


def _predict_cnn_classifier_detailed(model, encoder, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        features = model.forward_features(X_tensor)
        logits = model.head(features)
        proba_tensor = torch.softmax(logits, dim=1)
        pred_encoded = torch.argmax(proba_tensor, dim=1).cpu().numpy()
        proba = proba_tensor.cpu().numpy()
        logits_np = logits.cpu().numpy()
        features_np = features.cpu().numpy()
    y_pred = encoder.inverse_transform(pred_encoded)
    return y_pred, proba, logits_np, features_np


class _SequenceClassifier:
    def __new__(
        cls,
        *,
        n_features: int,
        n_classes: int,
        hidden_channels: int,
        kernel_size: int,
        dropout: float,
        model_arch: str,
        history_steps: int,
        current_steps: int,
        future_steps: int,
        tcn_levels: int,
        epoch_steps: int,
    ):
        import torch
        import torch.nn as nn

        padding = max(int(kernel_size) // 2, 0)
        history_steps = max(int(history_steps), 0)
        current_steps = max(int(current_steps), 1)
        future_steps = max(int(future_steps), 0)
        segment_lengths = (history_steps, current_steps, future_steps)
        epoch_steps = max(int(epoch_steps), 0)

        def _segment_pools(x: torch.Tensor) -> list[torch.Tensor]:
            outputs: list[torch.Tensor] = []
            start = 0
            for length in segment_lengths:
                if length <= 0:
                    zeros = torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
                    outputs.extend([zeros, zeros.clone()])
                    continue
                segment = x[:, :, start : start + length]
                outputs.append(segment.mean(dim=2))
                outputs.append(segment.amax(dim=2))
                start += length
            return outputs

        class ResidualDilatedBlock(nn.Module):
            def __init__(self, dilation: int) -> None:
                super().__init__()
                block_padding = dilation * max(int(kernel_size) - 1, 0) // 2
                self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=block_padding, dilation=dilation)
                self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=block_padding, dilation=dilation)
                self.act = nn.GELU()
                self.dropout = nn.Dropout(p=max(float(dropout), 0.0))

            def forward(self, x):
                residual = x
                x = self.act(self.conv1(x))
                x = self.dropout(x)
                x = self.act(self.conv2(x))
                x = self.dropout(x)
                return x + residual

        class TemporalConvNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model_arch = str(model_arch)
                if self.model_arch == "simple_cnn":
                    self.conv1 = nn.Conv1d(n_features, hidden_channels, kernel_size=kernel_size, padding=padding)
                    self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
                    self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
                    self.input_proj = None
                    self.blocks = None
                    self.branch_stem = None
                    self.branch_blocks = None
                    self.channel_gate = None
                    self.epoch_conv1 = None
                    self.epoch_conv2 = None
                    self.within_epoch_rnn = None
                    self.epoch_context_proj = None
                    self.epoch_blocks = None
                elif self.model_arch == "dilated_tcn":
                    self.input_proj = nn.Conv1d(n_features, hidden_channels, kernel_size=1)
                    self.blocks = nn.ModuleList(
                        [ResidualDilatedBlock(dilation=2**level) for level in range(max(int(tcn_levels), 1))]
                    )
                    self.conv1 = None
                    self.conv2 = None
                    self.conv3 = None
                    self.branch_stem = None
                    self.branch_blocks = None
                    self.channel_gate = None
                    self.epoch_conv1 = None
                    self.epoch_conv2 = None
                    self.within_epoch_rnn = None
                    self.epoch_context_proj = None
                    self.epoch_blocks = None
                elif self.model_arch == "multi_branch_tcn":
                    # Paper-inspired signal model: per-channel temporal branches followed by residual attention fusion.
                    self.branch_stem = nn.ModuleList(
                        [nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding) for _ in range(n_features)]
                    )
                    self.branch_blocks = nn.ModuleList(
                        [
                            nn.ModuleList(
                                [ResidualDilatedBlock(dilation=2**level) for level in range(max(int(tcn_levels), 1))]
                            )
                            for _ in range(n_features)
                        ]
                    )
                    self.channel_gate = nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.GELU(),
                        nn.Linear(hidden_channels, 1),
                    )
                    self.input_proj = None
                    self.blocks = None
                    self.conv1 = None
                    self.conv2 = None
                    self.conv3 = None
                    self.epoch_conv1 = None
                    self.epoch_conv2 = None
                    self.within_epoch_rnn = None
                    self.epoch_context_proj = None
                    self.epoch_blocks = None
                elif self.model_arch == "hierarchical_epoch":
                    if epoch_steps <= 0:
                        raise ValueError("hierarchical_epoch requires epoch_steps > 0")
                    self.conv1 = None
                    self.conv2 = None
                    self.conv3 = None
                    self.input_proj = None
                    self.blocks = None
                    self.branch_stem = None
                    self.branch_blocks = None
                    self.channel_gate = None
                    self.epoch_conv1 = nn.Conv1d(n_features, hidden_channels, kernel_size=kernel_size, padding=padding)
                    self.epoch_conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
                    self.within_epoch_rnn = nn.GRU(
                        input_size=hidden_channels,
                        hidden_size=hidden_channels,
                        batch_first=True,
                        bidirectional=True,
                    )
                    self.epoch_context_proj = nn.Linear(hidden_channels * 2, hidden_channels)
                    self.epoch_blocks = nn.ModuleList(
                        [ResidualDilatedBlock(dilation=2**level) for level in range(max(int(tcn_levels), 1))]
                    )
                else:
                    raise ValueError(f"Unsupported model_arch: {self.model_arch}")
                self.act = nn.GELU()
                self.dropout = nn.Dropout(p=max(float(dropout), 0.0))
                self.center_proj = nn.Linear(n_features, hidden_channels)
                self.head = nn.Sequential(
                    nn.Linear(hidden_channels * 7, hidden_channels),
                    nn.GELU(),
                    nn.Dropout(p=max(float(dropout), 0.0)),
                    nn.Linear(hidden_channels, n_classes),
                )

            def forward_features(self, x):
                if self.model_arch == "hierarchical_epoch":
                    if x.ndim != 4:
                        raise ValueError(f"hierarchical_epoch expects 4D input, got shape {tuple(x.shape)}")
                    current_start = history_steps
                    current_stop = history_steps + current_steps
                    current_raw = x[:, current_start:current_stop, :, :].mean(dim=(1, 2))
                    batch_size, n_epochs, n_epoch_steps, _ = x.shape
                    x = x.reshape(batch_size * n_epochs, n_epoch_steps, n_features).transpose(1, 2)
                    x = self.act(self.epoch_conv1(x))
                    x = self.dropout(x)
                    x = self.act(self.epoch_conv2(x))
                    x = self.dropout(x)
                    x = x.transpose(1, 2)
                    x, _ = self.within_epoch_rnn(x)
                    x = self.act(self.epoch_context_proj(x.mean(dim=1)))
                    x = x.view(batch_size, n_epochs, hidden_channels).transpose(1, 2)
                    for block in self.epoch_blocks:
                        x = block(x)
                else:
                    current_start = history_steps
                    current_stop = history_steps + current_steps
                    current_raw = x[:, current_start:current_stop, :].mean(dim=1)
                    x = x.transpose(1, 2)
                if self.model_arch == "simple_cnn":
                    x = self.act(self.conv1(x))
                    x = self.act(self.conv2(x))
                    x = self.act(self.conv3(x))
                elif self.model_arch == "dilated_tcn":
                    x = self.act(self.input_proj(x))
                    for block in self.blocks:
                        x = block(x)
                elif self.model_arch == "multi_branch_tcn":
                    branch_outputs = []
                    for channel_index, stem in enumerate(self.branch_stem):
                        branch_x = self.act(stem(x[:, channel_index : channel_index + 1, :]))
                        for block in self.branch_blocks[channel_index]:
                            branch_x = block(branch_x)
                        branch_outputs.append(branch_x)
                    stacked = torch.stack(branch_outputs, dim=1)
                    avg_fused = stacked.mean(dim=1)
                    gate_input = stacked.mean(dim=3)
                    gate_logits = self.channel_gate(gate_input).squeeze(-1)
                    gate_weights = torch.softmax(gate_logits, dim=1)
                    attn_fused = (stacked * gate_weights[:, :, None, None]).sum(dim=1)
                    x = avg_fused + attn_fused
                elif self.model_arch != "hierarchical_epoch":
                    raise ValueError(f"Unsupported model_arch: {self.model_arch}")
                x = self.dropout(x)
                history_mean, history_max, current_mean, current_max, future_mean, future_max = _segment_pools(x)
                center_skip = self.act(self.center_proj(current_raw))
                merged = torch.cat(
                    [history_mean, history_max, current_mean, current_max, future_mean, future_max, center_skip],
                    dim=1,
                )
                return merged

            def forward(self, x):
                return self.head(self.forward_features(x))

        return TemporalConvNet()


class _LabelEncoder:
    def __init__(self, y: np.ndarray) -> None:
        self.classes_ = np.unique(y)
        self._class_to_index = {value: idx for idx, value in enumerate(self.classes_)}

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray([self._class_to_index[value] for value in y], dtype=int)

    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        return self.classes_[encoded]
