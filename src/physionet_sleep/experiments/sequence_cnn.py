from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from physionet_sleep.experiments.normalization import apply_normalization
from physionet_sleep.experiments.sequence_data import build_centered_sequence_dataset
from physionet_sleep.experiments.tabular_runner import build_validation_splits, score_predictions, summarize_fold_metrics


@dataclass(slots=True)
class SequenceExperimentSpec:
    name: str
    target_column: str
    feature_columns: list[str]
    task: str = "classification"
    context_radius: int = 2
    normalization: str = "medianiqr"
    cv_folds: int = 5
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_channels: int = 32
    kernel_size: int = 3


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
        train_df = frame.iloc[train_idx].copy()
        val_df = frame.iloc[val_idx].copy()

        train_df, val_df, norm_stats = apply_normalization(
            train_df,
            val_df,
            feature_columns=spec.feature_columns,
            norm_name=spec.normalization,
            group_column=subject_column,
        )
        train_df, val_df = _impute_features(
            train_df,
            val_df,
            feature_columns=spec.feature_columns,
        )

        train_dataset = build_centered_sequence_dataset(
            train_df,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column=subject_column,
            time_column=time_column,
            context_radius=spec.context_radius,
        )
        val_dataset = build_centered_sequence_dataset(
            val_df,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            subject_column=subject_column,
            time_column=time_column,
            context_radius=spec.context_radius,
        )
        if train_dataset.X.shape[0] == 0 or val_dataset.X.shape[0] == 0:
            continue

        model, encoder = _train_cnn_classifier(
            train_dataset.X,
            train_dataset.y,
            epochs=spec.epochs,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            hidden_channels=spec.hidden_channels,
            kernel_size=spec.kernel_size,
            random_state=random_state,
        )

        y_pred, proba = _predict_cnn_classifier(model, encoder, val_dataset.X)
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
                "n_features_used": int(len(spec.feature_columns)),
                "sequence_length": int(train_dataset.window_length),
                "n_subjects_train": int(train_df[subject_column].nunique()),
                "n_subjects_val": int(val_df[subject_column].nunique()),
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
    if target_column == "label.stage_seconds":
        frame = frame.loc[pd.to_numeric(frame[target_column], errors="coerce") >= 0].copy()
    return frame


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


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError("PyTorch is required for sequence_cnn but is not installed.") from exc


def _train_cnn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    kernel_size: int,
    random_state: int,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(random_state))

    encoder = _LabelEncoder(y_train)
    y_encoded = encoder.transform(y_train)

    model = _TemporalConvClassifier(
        n_features=int(X_train.shape[2]),
        n_classes=int(len(encoder.classes_)),
        hidden_channels=int(hidden_channels),
        kernel_size=int(kernel_size),
    )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=max(int(batch_size), 1), shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(max(int(epochs), 1)):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    return model.eval(), encoder


def _predict_cnn_classifier(model, encoder, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32))
        proba_tensor = torch.softmax(logits, dim=1)
        pred_encoded = torch.argmax(proba_tensor, dim=1).cpu().numpy()
        proba = proba_tensor.cpu().numpy()
    y_pred = encoder.inverse_transform(pred_encoded)
    return y_pred, proba


class _TemporalConvClassifier:
    def __new__(cls, *, n_features: int, n_classes: int, hidden_channels: int, kernel_size: int):
        import torch.nn as nn

        padding = max(int(kernel_size) // 2, 0)

        class TemporalConvNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv1d(n_features, hidden_channels, kernel_size=kernel_size, padding=padding)
                self.bn1 = nn.BatchNorm1d(hidden_channels)
                self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
                self.bn2 = nn.BatchNorm1d(hidden_channels)
                self.act = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Linear(hidden_channels, n_classes)

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.act(self.bn1(self.conv1(x)))
                x = self.act(self.bn2(self.conv2(x)))
                x = self.pool(x).squeeze(-1)
                return self.head(x)

        return TemporalConvNet()


class _LabelEncoder:
    def __init__(self, y: np.ndarray) -> None:
        self.classes_ = np.unique(y)
        self._class_to_index = {value: idx for idx, value in enumerate(self.classes_)}

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray([self._class_to_index[value] for value in y], dtype=int)

    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        return self.classes_[encoded]
