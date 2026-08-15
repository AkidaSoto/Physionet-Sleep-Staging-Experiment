from __future__ import annotations

from pathlib import Path

from physionet_sleep.core.pipeline import AlgorithmPipeline
from physionet_sleep.core.types import DatasetRun, SignalRecord
from physionet_sleep.diagnostics.base import BaseDiagnostic
from physionet_sleep.experiments.base import BaseExperiment
from physionet_sleep.pipelines.apnea import build_apnea_pipeline
from physionet_sleep.runners.ucddb import run_ucddb_pipeline


def preprocess_ucddb(record: SignalRecord) -> dict[str, object]:
    """Runs the apnea pipeline on an already-loaded SignalRecord."""

    pipeline = build_apnea_pipeline()
    return {
        "record_metadata": record.metadata,
        "results": pipeline.run(record),
    }


def run_ucddb_record(
    record_id: str,
    *,
    root_dir: str | Path = "data/raw/ucddb",
    pipeline: AlgorithmPipeline | None = None,
    diagnostics: list[BaseDiagnostic] | None = None,
    experiments: list[BaseExperiment] | None = None,
) -> DatasetRun:
    return run_ucddb_pipeline(
        record_id,
        root_dir=root_dir,
        pipeline=pipeline or build_apnea_pipeline(),
        diagnostics=diagnostics,
        experiments=experiments,
    )
