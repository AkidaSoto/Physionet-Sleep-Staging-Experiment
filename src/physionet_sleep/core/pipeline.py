from __future__ import annotations

from dataclasses import dataclass, field

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, SignalRecord


@dataclass(slots=True)
class AlgorithmPipeline:
    algorithms: list[BaseAlgorithm] = field(default_factory=list)

    def add(self, algorithm: BaseAlgorithm) -> None:
        self.algorithms.append(algorithm)

    def run(self, record: SignalRecord) -> dict[str, AlgorithmResult]:
        results: dict[str, AlgorithmResult] = {}
        for algorithm in self.algorithms:
            results[algorithm.name] = algorithm.run(record, prior=results)
        return results
