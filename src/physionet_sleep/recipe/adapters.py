from __future__ import annotations

from dataclasses import dataclass, field

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, SignalRecord
from physionet_sleep.recipe.algo_def import AlgoDef


@dataclass(slots=True)
class WrappedAlgorithmStep(AlgoDef):
    algorithm: BaseAlgorithm
    dependency_names: tuple[str, ...] = ()
    record_input_name: str = "signal_record"
    optional_streams: tuple[str, ...] = ()
    _params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        AlgoDef.__init__(self)
        self.params.update(self._params)

    def algo_name(self) -> str:
        return self.algorithm.name

    def requires(self) -> list[str]:
        return [self.record_input_name, *[f"{name}_algo" for name in self.dependency_names]]

    def optionals(self) -> list[str]:
        return list(self.optional_streams)

    def extract_outputs(self) -> list[str]:
        return [self.algorithm.name, f"final__{self.algorithm.name}"]

    def predict_outputs(self) -> list[str]:
        return []

    def extract(self, inputs: dict, params: dict) -> None:
        record = inputs[self.record_input_name]
        if not isinstance(record, SignalRecord):
            raise TypeError(f"{self.algorithm.name} expected SignalRecord at '{self.record_input_name}'")
        prior: dict[str, AlgorithmResult] = {}
        for dep_name in self.dependency_names:
            dep_key = f"{dep_name}_algo"
            dep_value = inputs.get(dep_key)
            if isinstance(dep_value, AlgorithmResult):
                prior[dep_name] = dep_value
        result = self.algorithm.run(record, prior=prior)
        self.features_ = {
            self.algorithm.name: result,
            f"final__{self.algorithm.name}": result,
        }
