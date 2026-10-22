from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from physionet_sleep.experiments.runner_utils import build_split_tag


@dataclass(slots=True)
class ModelingContext:
    repo_root: Path
    raw_data_root: Path
    artifacts_root: Path
    cache_dir: Path
    datasets: tuple[str, ...] = ("ucddb",)


@dataclass(slots=True)
class ModelingJobOutcome:
    output_root: Path
    message: str
    summary: pd.DataFrame


@dataclass(slots=True)
class StagingReferenceBundle:
    reference_table: pd.DataFrame
    staging_table: pd.DataFrame
    feature_columns: list[str]


def build_modeling_context(
    repo_root: str | Path | None = None,
    *,
    datasets: tuple[str, ...] | list[str] = ("ucddb",),
) -> ModelingContext:
    resolved_root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    return ModelingContext(
        repo_root=resolved_root,
        raw_data_root=resolved_root / "data" / "raw" / "ucddb",
        artifacts_root=resolved_root / "artifacts",
        cache_dir=resolved_root / "artifacts" / "ucddb_recipe_cache",
        datasets=tuple(datasets),
    )


def dataset_tag(context: ModelingContext) -> str:
    return "" if context.datasets == ("ucddb",) else "__" + "+".join(context.datasets)


def build_output_root(
    context: ModelingContext,
    *,
    output_dir: str | Path,
    run_tag: str | None = None,
) -> Path:
    base = context.repo_root / Path(output_dir)
    if run_tag:
        run_tag = f"{run_tag}{dataset_tag(context)}"
    root = base / run_tag if run_tag else base
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_reference_table(
    context: ModelingContext,
    *,
    feature_table_name: str = "final__feature_table",
) -> pd.DataFrame:
    if context.datasets != ("ucddb",):
        from physionet_sleep.analysis.datasets import load_reference_dataset

        return load_reference_dataset(context.datasets, feature_table_name=feature_table_name)
    from physionet_sleep.analysis.reference_data import load_ucddb_reference_dataset

    return load_ucddb_reference_dataset(
        cache_dir=context.cache_dir,
        root_dir=context.raw_data_root,
        feature_table_name=feature_table_name,
    )


def load_staging_reference_bundle(
    context: ModelingContext,
    *,
    spec_name: str = "staging_plus_shared",
) -> StagingReferenceBundle:
    from physionet_sleep.analysis import build_staging_benchmark_table
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    reference_table = load_reference_table(context)
    specs = build_default_experiment_specs(reference_table)
    base_spec = specs[spec_name]
    staging_table = build_staging_benchmark_table(
        reference_table,
        feature_columns=base_spec.feature_columns,
    )
    return StagingReferenceBundle(
        reference_table=reference_table,
        staging_table=staging_table,
        feature_columns=list(base_spec.feature_columns),
    )


def save_result_bundle(
    result: Any,
    *,
    output_root: str | Path,
    metadata_updates: dict[str, Any] | None = None,
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}

    if hasattr(result, "save") and callable(result.save):
        result.save(root)
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            metadata = dict(result.metadata)
    elif is_dataclass(result):
        for field in fields(result):
            value = getattr(result, field.name)
            if field.name == "metadata" and isinstance(value, dict):
                metadata = dict(value)
                continue
            if isinstance(value, pd.DataFrame):
                value.to_parquet(root / f"{field.name}.parquet", index=False)
    else:
        raise TypeError(f"Unsupported result bundle type: {type(result)!r}")

    if metadata_updates:
        metadata.update(metadata_updates)
    if metadata:
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_sequence_baseline_job(
    context: ModelingContext,
    *,
    context_radius: int = 2,
    outer_validation: str = "kfold",
    outer_folds: int = 5,
    random_state: int = 7,
    output_dir: str = "artifacts/sequence_cnn_baseline",
) -> ModelingJobOutcome:
    from physionet_sleep.experiments.sequence_cnn import SequenceExperimentSpec, run_sequence_cnn_experiment

    bundle = load_staging_reference_bundle(context)
    split_tag = build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
    )
    run_tag = f"{split_tag}__ctx{(2 * int(context_radius)) + 1}"
    output_root = build_output_root(context, output_dir=output_dir, run_tag=run_tag)
    spec = SequenceExperimentSpec(
        name=f"staging_plus_shared_cnn_ctx{(2 * int(context_radius)) + 1}",
        target_column="label.stage_30s",
        feature_columns=bundle.feature_columns,
        context_radius=context_radius,
        normalization="medianiqr",
        cv_folds=outer_folds,
        epochs=60,
        batch_size=256,
        learning_rate=1e-3,
        hidden_channels=64,
        kernel_size=3,
        dropout=0.15,
        min_epochs=12,
        patience=10,
        internal_val_subjects=1,
        weighted_sampling=True,
        expected_step_seconds=30.0,
        step_tolerance_seconds=0.25,
    )
    result = run_sequence_cnn_experiment(
        bundle.staging_table,
        spec=spec,
        validation=outer_validation,
        random_state=random_state,
    )
    save_result_bundle(result, output_root=output_root)
    message = f"saved={output_root}"
    return ModelingJobOutcome(output_root=output_root, message=message, summary=result.summary)


def run_sequence_epoch_job(
    context: ModelingContext,
    *,
    history_steps: int = 10,
    current_steps: int = 5,
    future_steps: int = 5,
    model_arch: str = "dilated_tcn",
    tcn_levels: int = 4,
    outer_validation: str = "kfold",
    outer_folds: int = 5,
    random_state: int = 7,
    output_dir: str = "artifacts/sequence_cnn_6s_epoch",
) -> ModelingJobOutcome:
    from physionet_sleep.experiments.sequence_cnn import SequenceExperimentSpec, run_sequence_cnn_experiment

    bundle = load_staging_reference_bundle(context)
    split_tag = build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
    )
    shape_tag = f"h{history_steps}_c{current_steps}_f{future_steps}"
    run_tag = f"{split_tag}__{shape_tag}__{model_arch}"
    output_root = build_output_root(context, output_dir=output_dir, run_tag=run_tag)
    spec = SequenceExperimentSpec(
        name=f"staging_plus_shared_cnn_6s_{shape_tag}_{model_arch}",
        target_column="label.stage_seconds",
        feature_columns=bundle.feature_columns,
        model_arch=model_arch,
        sequence_mode="epoch_aligned",
        history_steps=history_steps,
        current_steps=current_steps,
        future_steps=future_steps,
        normalization="medianiqr",
        cv_folds=outer_folds,
        epochs=60,
        batch_size=256,
        learning_rate=1e-3,
        hidden_channels=64,
        kernel_size=3,
        dropout=0.15,
        tcn_levels=tcn_levels,
        min_epochs=12,
        patience=10,
        internal_val_subjects=3,
        weighted_sampling=True,
        expected_step_seconds=6.0,
        step_tolerance_seconds=0.25,
    )
    result = run_sequence_cnn_experiment(
        bundle.reference_table,
        spec=spec,
        validation=outer_validation,
        random_state=random_state,
    )
    save_result_bundle(
        result,
        output_root=output_root,
        metadata_updates={
            "history_steps": history_steps,
            "current_steps": current_steps,
            "future_steps": future_steps,
            "model_arch": model_arch,
            "tcn_levels": tcn_levels,
            "sequence_step_seconds": 6.0,
            "target_epoch_seconds": current_steps * 6.0,
        },
    )
    message = f"saved={output_root}"
    return ModelingJobOutcome(output_root=output_root, message=message, summary=result.summary)


def run_stage_expert_temporal_job(
    context: ModelingContext,
    *,
    expert_model_family: str = "cnn",
    stage2_model_family: str = "linear",
    outer_validation: str = "kfold",
    outer_folds: int = 5,
    inner_validation: str = "single_subject",
    context_radius: int = 2,
    current_steps: int = 5,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    kernel_size: int = 3,
    dropout: float = 0.1,
    min_epochs: int = 10,
    patience: int = 8,
    random_state: int = 7,
    output_dir: str = "artifacts/stage_expert_temporal",
) -> ModelingJobOutcome:
    from physionet_sleep.experiments.stage_expert_temporal import (
        StageExpertTemporalSpec,
        build_default_stage_expert_specs,
        run_stage_expert_temporal_experiment,
    )

    table = load_reference_table(context)
    split_tag = build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
    )
    run_tag = (
        f"{split_tag}__experts_{expert_model_family}"
        f"__stage2_{stage2_model_family}__ctx{(2 * int(context_radius)) + 1}"
    )
    output_root = build_output_root(context, output_dir=output_dir, run_tag=run_tag)
    spec = StageExpertTemporalSpec(
        name=f"stage_expert_temporal_{expert_model_family}_ctx{(2 * int(context_radius)) + 1}",
        expert_specs=build_default_stage_expert_specs(table),
        expert_model_family=expert_model_family,
        stage2_model_family=stage2_model_family,
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
        current_steps=current_steps,
        context_radius=context_radius,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        min_epochs=min_epochs,
        patience=patience,
    )
    result = run_stage_expert_temporal_experiment(
        table,
        spec=spec,
        random_state=random_state,
    )
    save_result_bundle(result, output_root=output_root)
    message = f"saved={output_root}"
    return ModelingJobOutcome(output_root=output_root, message=message, summary=result.summary)


def run_stage_signal_expert_temporal_job(
    context: ModelingContext,
    *,
    stage1_model_arch: str = "dilated_tcn",
    stage2_model_family: str = "tcn",
    outer_validation: str = "kfold",
    outer_folds: int = 5,
    inner_validation: str = "single_subject",
    window_points: int = 64,
    context_radius: int = 2,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_channels: int = 32,
    kernel_size: int = 5,
    dropout: float = 0.1,
    min_epochs: int = 8,
    patience: int = 6,
    tcn_levels: int = 3,
    random_state: int = 7,
    output_dir: str = "artifacts/stage_signal_expert_temporal",
) -> ModelingJobOutcome:
    from physionet_sleep.experiments.stage_signal_expert_temporal import (
        StageSignalExpertTemporalSpec,
        build_default_stage_signal_expert_specs,
        build_staging_epoch_frame,
        run_stage_signal_expert_temporal_experiment,
    )

    reference_table = load_reference_table(context)
    epoch_frame = build_staging_epoch_frame(reference_table)
    split_tag = build_split_tag(
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
    )
    run_tag = f"{split_tag}__stage2_{stage2_model_family}__wp{window_points}__ctx{(2 * int(context_radius)) + 1}"
    output_root = build_output_root(context, output_dir=output_dir, run_tag=run_tag)
    spec = StageSignalExpertTemporalSpec(
        name=f"stage_signal_expert_temporal_{stage2_model_family}",
        expert_specs=build_default_stage_signal_expert_specs(),
        stage1_model_arch=stage1_model_arch,
        stage2_model_family=stage2_model_family,
        outer_validation=outer_validation,
        outer_folds=outer_folds,
        inner_validation=inner_validation,
        window_points=window_points,
        context_radius=context_radius,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        min_epochs=min_epochs,
        patience=patience,
        tcn_levels=tcn_levels,
    )
    result = run_stage_signal_expert_temporal_experiment(
        epoch_frame,
        cache_dir=context.cache_dir,
        spec=spec,
        random_state=random_state,
    )
    save_result_bundle(result, output_root=output_root)
    message = f"saved={output_root}"
    return ModelingJobOutcome(output_root=output_root, message=message, summary=result.summary)


def run_reference_diagnostics_job(
    context: ModelingContext,
    *,
    output_dir: str = "artifacts/reference_diagnostics",
) -> ModelingJobOutcome:
    from physionet_sleep.diagnostics.reference_runner import run_reference_diagnostics

    table = load_reference_table(context)
    result = run_reference_diagnostics(table)
    output_root = build_output_root(context, output_dir=output_dir)
    save_result_bundle(result, output_root=output_root)
    summary = result.best_feature_norms.head(15).copy()
    message = f"saved={output_root} rows={len(table)} features={result.metadata.get('feature_count', 0)}"
    return ModelingJobOutcome(output_root=output_root, message=message, summary=summary)
