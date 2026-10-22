from __future__ import annotations

from pathlib import Path

from physionet_sleep.io.ucddb import discover_ucddb_records, load_ucddb_waveforms, resolve_ucddb_record
from physionet_sleep.pipelines.apnea import build_apnea_pipeline
from physionet_sleep.recipe import StudyDef, WrappedAlgorithmStep


def build_ucddb_apnea_recipe(*, ecg_detector_backend: str = "wfdb_xqrs"):
    pipeline = build_apnea_pipeline(ecg_detector_backend=ecg_detector_backend)
    algo_names = [algorithm.name for algorithm in pipeline.algorithms]
    deps: dict[str, tuple[str, ...]] = {}
    for algorithm in pipeline.algorithms:
        if algorithm.name == "apnea_events":
            deps[algorithm.name] = ("respiration", "spo2_drop")
        elif algorithm.name == "hrv_features":
            deps[algorithm.name] = ("pan_tompkins",)
        elif algorithm.name == "cardiorespiratory_coupling":
            deps[algorithm.name] = ("hrv_features",)
        elif algorithm.name.startswith("feature_table"):
            deps[algorithm.name] = tuple(name for name in algo_names if not name.startswith("feature_table"))
        else:
            deps[algorithm.name] = ()
    return [
        WrappedAlgorithmStep(
            algorithm=algorithm,
            dependency_names=deps.get(algorithm.name, ()),
        )
        for algorithm in pipeline.algorithms
    ]


class UCDDBApneaStudyDef(StudyDef):
    def __init__(
        self,
        *,
        subjects: list[str] | None = None,
        root_dir: str | Path = "data/raw/ucddb",
        cache_dir: str | Path = "artifacts/ucddb_recipe_cache",
        ecg_detector_backend: str = "wfdb_xqrs",
    ) -> None:
        self._root_dir = Path(root_dir)
        self._algo_path = str(Path(cache_dir))
        self._subjects = subjects or [p.record_id for p in discover_ucddb_records(self._root_dir)]
        self.recipe = build_ucddb_apnea_recipe(ecg_detector_backend=ecg_detector_backend)

    @property
    def subjects(self) -> list[str]:
        return self._subjects

    @property
    def algo_path(self) -> str:
        return self._algo_path

    def load_fn(self, subject_id: str, req: str, meta=None):
        try:
            paths = resolve_ucddb_record(self._root_dir, subject_id)
            if req == "signal_record":
                return load_ucddb_waveforms(paths)
        except Exception:
            return None
        return None
