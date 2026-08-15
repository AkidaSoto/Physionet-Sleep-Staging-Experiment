from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from physionet_sleep.core.types import AlgorithmResult, SignalRecord


class MissingSignalError(ValueError):
    pass


class BaseAlgorithm(ABC):
    def __init__(self, *, name: str, references: tuple[str, ...] = ()) -> None:
        self.name = name
        self.references = references

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def validate(self, record: SignalRecord) -> None:
        missing = [name for name in self.required_signals() if name not in record.channels]
        if missing:
            raise MissingSignalError(
                f"{self.name} requires missing signals: {', '.join(missing)}"
            )

    def run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult] | None = None,
    ) -> AlgorithmResult:
        self.validate(record)
        result = self._run(record, prior=prior or {})
        result.metadata.setdefault("references", self.references)
        return result

    @abstractmethod
    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        raise NotImplementedError
