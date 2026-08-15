from __future__ import annotations

from pathlib import Path

from physionet_sleep.alignment.ucddb import align_ucddb_labels_to_results
from physionet_sleep.core.pipeline import AlgorithmPipeline
from physionet_sleep.core.types import DatasetRun, RecordLabels, SignalRecord
from physionet_sleep.diagnostics import (
    BaseDiagnostic,
    FeatureScreenDiagnostic,
    PairwiseFeatureDiagnostic,
    StageFitDiagnostic,
)
from physionet_sleep.experiments import ApneaSecondwiseExperiment, BaseExperiment
from physionet_sleep.io.ucddb import UCDDBRecordPaths, load_ucddb_waveforms, resolve_ucddb_record
from physionet_sleep.labels.ucddb import load_ucddb_labels
from physionet_sleep.pipelines.apnea import build_apnea_pipeline
from physionet_sleep.studies.ucddb import UCDDBApneaStudyDef
from physionet_sleep.utils import enrich_feature_tables


def load_ucddb_inputs(
    record_id: str,
    *,
    root_dir: str | Path = "data/raw/ucddb",
) -> tuple[UCDDBRecordPaths, SignalRecord, RecordLabels]:
    paths = resolve_ucddb_record(root_dir, record_id)
    record = load_ucddb_waveforms(paths)
    labels = load_ucddb_labels(paths)
    return paths, record, labels


def run_ucddb_algorithms(
    record: SignalRecord,
    *,
    pipeline: AlgorithmPipeline | None = None,
):
    chosen_pipeline = pipeline or build_apnea_pipeline()
    return chosen_pipeline.run(record)


def run_ucddb_algorithms_recipe(
    record_id: str,
    *,
    root_dir: str | Path = "data/raw/ucddb",
    cache_dir: str | Path = "artifacts/ucddb_recipe_cache",
):
    study = UCDDBApneaStudyDef(subjects=[record_id], root_dir=root_dir, cache_dir=cache_dir)
    built = study.run_build({"no_save": True, "force": True})
    if not built["cache"]:
        raise RuntimeError(f"Recipe runner did not complete subject '{record_id}'")
    cache = built["cache"][0]
    results = {}
    for k, v in cache.items():
        if not (hasattr(v, "signals") and hasattr(v, "events")):
            continue
        if k.startswith("final__"):
            results[k.replace("final__", "", 1)] = v
        elif k not in results:
            results[k] = v
    return results


def default_ucddb_diagnostics() -> list[BaseDiagnostic]:
    return [FeatureScreenDiagnostic(), PairwiseFeatureDiagnostic(), StageFitDiagnostic()]


def default_ucddb_experiments() -> list[BaseExperiment]:
    return [ApneaSecondwiseExperiment()]


def run_ucddb_pipeline(
    record_id: str,
    *,
    root_dir: str | Path = "data/raw/ucddb",
    pipeline: AlgorithmPipeline | None = None,
    diagnostics: list[BaseDiagnostic] | None = None,
    experiments: list[BaseExperiment] | None = None,
    engine: str = "recipe",
) -> DatasetRun:
    paths, record, labels = load_ucddb_inputs(record_id, root_dir=root_dir)
    if engine == "recipe" and pipeline is None:
        results = run_ucddb_algorithms_recipe(record_id, root_dir=root_dir)
    else:
        results = run_ucddb_algorithms(record, pipeline=pipeline)
    alignment = align_ucddb_labels_to_results(record, labels, results)
    dataset_run = DatasetRun(
        record=record,
        labels=labels,
        results=results,
        alignment=alignment,
        metadata={
            "dataset": "ucddb",
            "record_id": paths.record_id,
            "waveform_path": record.metadata.get("waveform_path"),
            "stage_path": str(paths.stage_path) if paths.stage_path else None,
            "respevt_path": str(paths.respevt_path) if paths.respevt_path else None,
        },
    )
    enrich_feature_tables(dataset_run)

    chosen_diagnostics = diagnostics if diagnostics is not None else default_ucddb_diagnostics()
    chosen_experiments = experiments if experiments is not None else default_ucddb_experiments()

    for diagnostic in chosen_diagnostics:
        dataset_run.diagnostics[diagnostic.name] = diagnostic.run(dataset_run)
    for experiment in chosen_experiments:
        dataset_run.experiments[experiment.name] = experiment.run(dataset_run)

    return dataset_run
