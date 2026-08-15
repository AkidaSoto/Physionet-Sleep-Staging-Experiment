from __future__ import annotations

from abc import ABC, abstractmethod

from physionet_sleep.core.types import DatasetRun, ExperimentResult


class BaseExperiment(ABC):
    name: str

    @abstractmethod
    def run(self, dataset_run: DatasetRun) -> ExperimentResult:
        raise NotImplementedError
