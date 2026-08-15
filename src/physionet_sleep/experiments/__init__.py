from physionet_sleep.experiments.base import BaseExperiment
from physionet_sleep.experiments.sequence_cnn import SequenceExperimentSpec, SequenceRunResult, run_sequence_cnn_experiment
from physionet_sleep.experiments.sequence_data import (
    SequenceDataset,
    build_centered_sequence_dataset,
    build_flat_context_frame,
    build_mean_context_frame,
    flatten_sequence_tensor,
)
from physionet_sleep.experiments.tabular_specs import (
    DEFAULT_NORM_CANDIDATES,
    TabularExperimentSpec,
    build_default_experiment_specs,
)

__all__ = [
    "ApneaSecondwiseExperiment",
    "BaseExperiment",
    "DEFAULT_NORM_CANDIDATES",
    "SequenceDataset",
    "SequenceExperimentSpec",
    "SequenceRunResult",
    "TabularExperimentSpec",
    "build_default_experiment_specs",
    "build_centered_sequence_dataset",
    "build_flat_context_frame",
    "build_mean_context_frame",
    "flatten_sequence_tensor",
    "run_sequence_cnn_experiment",
]


def __getattr__(name: str):
    if name == "ApneaSecondwiseExperiment":
        from physionet_sleep.experiments.apnea_metrics import ApneaSecondwiseExperiment

        return ApneaSecondwiseExperiment
    raise AttributeError(name)
