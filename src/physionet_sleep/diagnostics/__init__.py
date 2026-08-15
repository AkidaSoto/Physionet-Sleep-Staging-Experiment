from physionet_sleep.diagnostics.base import BaseDiagnostic

__all__ = [
    "BaseDiagnostic",
    "CollectionDiagnostics",
    "FeatureScreenDiagnostic",
    "PairwiseFeatureDiagnostic",
    "StageFitDiagnostic",
    "SubjectQCReport",
    "compute_subject_qc",
    "run_collection_diagnostics",
]


def __getattr__(name: str):
    if name in {"CollectionDiagnostics", "run_collection_diagnostics"}:
        from physionet_sleep.diagnostics.collection import CollectionDiagnostics, run_collection_diagnostics

        exports = {
            "CollectionDiagnostics": CollectionDiagnostics,
            "run_collection_diagnostics": run_collection_diagnostics,
        }
        return exports[name]
    if name == "FeatureScreenDiagnostic":
        from physionet_sleep.diagnostics.feature_screen import FeatureScreenDiagnostic

        return FeatureScreenDiagnostic
    if name == "PairwiseFeatureDiagnostic":
        from physionet_sleep.diagnostics.pairwise import PairwiseFeatureDiagnostic

        return PairwiseFeatureDiagnostic
    if name == "StageFitDiagnostic":
        from physionet_sleep.diagnostics.stage_fit import StageFitDiagnostic

        return StageFitDiagnostic
    if name in {"SubjectQCReport", "compute_subject_qc"}:
        from physionet_sleep.diagnostics.subject_qc import SubjectQCReport, compute_subject_qc

        exports = {
            "SubjectQCReport": SubjectQCReport,
            "compute_subject_qc": compute_subject_qc,
        }
        return exports[name]
    raise AttributeError(name)
