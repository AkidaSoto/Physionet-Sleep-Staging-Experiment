from __future__ import annotations

from dataclasses import dataclass

from physionet_sleep.core.types import DatasetRun
from physionet_sleep.diagnostics.label_agreement import compute_label_agreement
from physionet_sleep.diagnostics.pairwise import PairwiseFeatureDiagnostic
from physionet_sleep.diagnostics.subject_qc import SubjectQCReport, compute_subject_qc


@dataclass(slots=True)
class CollectionDiagnostics:
    pairwise_by_subject: dict[str, dict]
    subject_qc: SubjectQCReport
    label_agreement_by_subject: dict[str, list]


def run_collection_diagnostics(
    runs: list[DatasetRun],
    *,
    track_name: str = "stage_seconds",
    agreement_tracks: list[str] | None = None,
) -> CollectionDiagnostics:
    pairwise_diag = PairwiseFeatureDiagnostic(track_name=track_name)
    pairwise_by_subject = {
        str(run.metadata.get("record_id", i)): pairwise_diag.run(run).summaries
        for i, run in enumerate(runs)
    }
    label_agreement_by_subject = {
        str(run.metadata.get("record_id", i)): compute_label_agreement(run, track_names=agreement_tracks or [])
        for i, run in enumerate(runs)
    }
    return CollectionDiagnostics(
        pairwise_by_subject=pairwise_by_subject,
        subject_qc=compute_subject_qc(runs, track_name=track_name),
        label_agreement_by_subject=label_agreement_by_subject,
    )
