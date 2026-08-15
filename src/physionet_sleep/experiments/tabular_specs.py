from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_NORM_CANDIDATES = ("raw", "zscore", "medianiqr")


@dataclass(slots=True)
class TabularExperimentSpec:
    name: str
    task: str
    target_column: str
    feature_columns: list[str]
    shared_columns: list[str]
    normalization_candidates: tuple[str, ...] = DEFAULT_NORM_CANDIDATES
    model_families: tuple[str, ...] = ("linear", "tree", "ensemble")
    cv_folds: int = 5


def build_default_experiment_specs(feature_table: pd.DataFrame) -> dict[str, TabularExperimentSpec]:
    columns = list(feature_table.columns)
    base_feature_columns = [
        col
        for col in columns
        if col != "time_seconds"
        and not col.startswith("label.")
        and not col.startswith("truth.")
        and not col.startswith("aligned.")
    ]

    shared_columns = _match_prefixes(
        base_feature_columns,
        prefixes=("pan_tompkins.",),
    )
    staging_columns = _match_prefixes(
        base_feature_columns,
        prefixes=(
            "spindle_features.",
            "slow_wave_features.",
            "eye_movement_activity.",
            "emg_tone_features.",
            "eeg_arousal_features.",
            "eog_rem_features.",
        ),
    )
    apnea_columns = _match_prefixes(
        base_feature_columns,
        prefixes=(
            "respiration.",
            "spo2_drop.",
            "effort_snore_features.",
            "airflow_morphology.",
            "paradoxical_breathing.",
            "apnea_events.",
        ),
    )

    specs = {
        "staging_core": TabularExperimentSpec(
            name="staging_core",
            task="classification",
            target_column="label.stage_seconds",
            feature_columns=staging_columns,
            shared_columns=[],
        ),
        "staging_plus_shared": TabularExperimentSpec(
            name="staging_plus_shared",
            task="classification",
            target_column="label.stage_seconds",
            feature_columns=_unique(staging_columns + shared_columns),
            shared_columns=shared_columns,
        ),
        "apnea_core": TabularExperimentSpec(
            name="apnea_core",
            task="classification",
            target_column="label.respiratory_event_truth",
            feature_columns=apnea_columns,
            shared_columns=[],
        ),
        "apnea_plus_shared": TabularExperimentSpec(
            name="apnea_plus_shared",
            task="classification",
            target_column="label.respiratory_event_truth",
            feature_columns=_unique(apnea_columns + shared_columns),
            shared_columns=shared_columns,
        ),
    }
    return {
        name: spec
        for name, spec in specs.items()
        if spec.target_column in columns and spec.feature_columns
    }


def _match_prefixes(columns: list[str], *, prefixes: tuple[str, ...]) -> list[str]:
    return [col for col in columns if col.startswith(prefixes)]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
