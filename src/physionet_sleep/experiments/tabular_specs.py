from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_NORM_CANDIDATES = ("raw", "zscore", "medianiqr")
STAGING_EXCLUDED_COLUMNS = {
    "eeg_arousal_features.eeg_arousal_signal",
    "eog_rem_features.rem_variance_epoch",
    "eog_rem_features.rem_power_epoch",
    "eog_rem_features.sem_power_epoch",
    "eog_rem_features.noise_power_epoch",
    "pan_tompkins.ecg_filtered",
    # Dropped from the staging models: ad hoc products, a duplicate column, a baseline-subtracted
    # envelope that cancels sustained N3 activity, and a residual below the 1/f fit range.
    # Removing them left 5-fold macro-F1 unchanged (0.608 -> 0.609).
    "spindle_features.splindex",
    "slow_wave_features.slow_wave_ratio_power",
    "slow_wave_features.slow_wave_index",
    "slow_wave_features.slow_wave_raw_power",
    "spindle_features.delta_residual_power",
    # Replaced by gain-free epoch-grid ratios (epoch_ratio_features.*): +0.028 macro-F1 with HGB on 6/6 subject
    # partitions (artifacts/exploration/ratio-normalization).
    "spindle_features.theta_power",
    "spindle_features.alpha_power",
    "spindle_features.sigma_power",
    "spindle_features.beta_power",
    "spindle_features.gamma_power",
    "slow_wave_features.high_frequency_envelope",
    "spindle_features.sigma_beta_ratio",
    "eog_rem_features.rem_band_power_fast",
    "eog_rem_features.sem_band_power_fast",
    "eog_rem_features.noise_band_power_fast",
    "emg_tone_features.emg_tone",
    "emg_tone_features.emg_tone_baseline",
    # Quality flag, not a model input.
    "hrv_features.nn_clean_frac_300s",
}

# Gain-free by construction; a per-subject z-score would reintroduce stage-mix dependence.
FEATURE_NORM_OVERRIDES_BY_PREFIX = {"epoch_ratio_features.": "none", "topography_features.": "none"}
STAGING_INTERPRETABLE_COLUMNS = {
    "slow_wave_features.slow_wave_envelope",
    "slow_wave_features.high_frequency_envelope",
    "spindle_features.aperiodic_exponent",
    "spindle_features.sigma_residual_power",
    "eye_movement_activity.eye_movement_activity",
    "eog_rem_features.rem_band_power_fast",
    "eog_rem_features.sem_band_power_fast",
    "emg_tone_features.emg_tone_baseline",
    "pan_tompkins.heart_rate_bpm_track",
    "pan_tompkins.hrv_rmssd_track",
    "respiration.respiration_rate_bpm",
    "respiration.respiration_envelope",
}


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
        and _is_model_feature_column(col)
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
            "epoch_ratio_features.",
            # Calibrated / frontal-occipital features; only present in datasets that support them (CinC 2018).
            "topography_features.",
        ),
    )
    staging_columns = [col for col in staging_columns if col not in STAGING_EXCLUDED_COLUMNS]
    # Staging uses the cleaned, literature-window cardio-respiratory set instead of the raw 30-s pan_tompkins tracks
    # (artifacts/exploration/hrv-features); the apnea specs keep pan_tompkins.
    staging_cardio_columns = [
        col
        for col in _match_prefixes(
            base_feature_columns,
            prefixes=("hrv_features.", "breath_regularity.", "cardiorespiratory_coupling."),
        )
        if col not in STAGING_EXCLUDED_COLUMNS
    ]
    shared_columns = [col for col in shared_columns if col not in STAGING_EXCLUDED_COLUMNS]
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
            feature_columns=_unique(staging_columns + staging_cardio_columns),
            shared_columns=staging_cardio_columns,
        ),
        "staging_interpretable": TabularExperimentSpec(
            name="staging_interpretable",
            task="classification",
            target_column="label.stage_seconds",
            feature_columns=[col for col in base_feature_columns if col in STAGING_INTERPRETABLE_COLUMNS],
            shared_columns=[],
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


def _is_model_feature_column(column_name: str) -> bool:
    lowered = column_name.lower()
    if any(token in lowered for token in ("mask", "vector")):
        return False
    if lowered.endswith("respiration_rate_peak_bpm") or lowered.endswith("respiration_rate_trough_bpm"):
        return False
    if lowered.startswith("effort_snore_features.snore_"):
        return False
    if lowered.startswith("apnea_events."):
        return False
    return True


# Physiological families of the staging feature set, shared by the dashboard export and the importance diagnostics.
STAGING_FEATURE_FAMILIES = [
    {
        "id": "spindles",
        "label": "Spindles",
        "signal": "EEG",
        "feature_ids": [
            "spindle_features.sigma_residual_power",
            "spindle_features.sigma_beta_residual_diff",
            "epoch_ratio_features.rel_sigma",
            "epoch_ratio_features.spindle_rel",
        ],
    },
    {
        "id": "spectrum",
        "label": "EEG spectrum",
        "signal": "EEG",
        "feature_ids": [
            "spindle_features.aperiodic_exponent",
            "spindle_features.aperiodic_intercept",
            "spindle_features.delta_power",
            "epoch_ratio_features.rel_theta",
            "epoch_ratio_features.rel_alpha",
            "epoch_ratio_features.rel_beta",
            "spindle_features.theta_residual_power",
            "spindle_features.alpha_residual_power",
            "spindle_features.beta_residual_power",
            "spindle_features.gamma_residual_power",
        ],
    },
    {"id": "slow-waves", "label": "Slow waves", "signal": "EEG", "feature_ids": ["slow_wave_features.slow_wave_envelope"]},
    {"id": "fast-eeg", "label": "Fast EEG (muscle)", "signal": "EEG", "feature_ids": ["epoch_ratio_features.muscle_frac_db"]},
    {
        "id": "eye-movement",
        "label": "Eye movement",
        "signal": "EOG",
        "feature_ids": [
            "eye_movement_activity.eye_movement_activity",
            "eye_movement_activity.eye_movement_baseline",
            "epoch_ratio_features.eog_rem_vs_noise_db",
            "epoch_ratio_features.eog_rem_vs_sem_db",
        ],
    },
    {
        "id": "muscle-tone",
        "label": "Muscle tone",
        "signal": "Chin EMG",
        "feature_ids": ["epoch_ratio_features.emg_floor_db", "epoch_ratio_features.emg_baseline_floor_db"],
    },
    {
        "id": "heart",
        "label": "Heart",
        "signal": "ECG",
        "feature_ids": [
            "hrv_features.hr_bpm_30s",
            "hrv_features.hr_surge_30s",
            "hrv_features.sdnn_150s",
            "hrv_features.hr_slow_sd_300s",
            "hrv_features.dfa_a2_300s",
            "hrv_features.ln_lf_hf_300s",
        ],
    },
    {
        "id": "breathing",
        "label": "Breathing",
        "signal": "Respiratory effort",
        "feature_ids": [
            "breath_regularity.breath_interval_cv_60s",
            "breath_regularity.breath_amp_cv_60s",
            "cardiorespiratory_coupling.hr_resp_coherence_120s",
        ],
    },
]

# Dataset-specific families, appended to the explorer/importance families when the features are present.
TOPOGRAPHY_FAMILY = {
    "id": "topography",
    "label": "Frontal slow waves / occipital alpha",
    "signal": "EEG",
    "feature_ids": ["topography_features.frontal_slow_wave_fraction", "topography_features.occipital_rel_alpha"],
}
