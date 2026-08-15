from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SignalSeries:
    values: np.ndarray
    sample_rate: float
    units: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        if self.values.ndim == 0:
            raise ValueError("SignalSeries.values must be at least 1D")

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.values.shape[0] / self.sample_rate)


@dataclass(slots=True)
class Event:
    start: int
    end: int
    label: str
    kind: str = "period"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignalRecord:
    channels: dict[str, SignalSeries] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_channel(
        self,
        name: str,
        values: np.ndarray,
        sample_rate: float,
        *,
        units: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.channels[name] = SignalSeries(
            values=np.asarray(values),
            sample_rate=sample_rate,
            units=units,
            metadata=metadata or {},
        )

    def get(self, name: str) -> SignalSeries:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise KeyError(f"Signal '{name}' not found in record") from exc

    def require(self, *names: str) -> tuple[SignalSeries, ...]:
        return tuple(self.get(name) for name in names)


@dataclass(slots=True)
class AlgorithmResult:
    name: str
    signals: dict[str, SignalSeries] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabelTrack:
    values: np.ndarray
    step_seconds: float
    class_map: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)


@dataclass(slots=True)
class RecordLabels:
    tracks: dict[str, LabelTrack] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabelAlignment:
    tracks: dict[str, np.ndarray] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiagnosticResult:
    name: str
    summaries: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentResult:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetRun:
    record: SignalRecord
    labels: RecordLabels
    results: dict[str, AlgorithmResult]
    alignment: LabelAlignment
    diagnostics: dict[str, DiagnosticResult] = field(default_factory=dict)
    experiments: dict[str, ExperimentResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
