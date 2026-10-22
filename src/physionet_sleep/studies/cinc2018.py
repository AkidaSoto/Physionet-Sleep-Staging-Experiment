from __future__ import annotations

import shutil
from pathlib import Path

from physionet_sleep.io.cinc2018 import discover_cinc2018_records, load_cinc2018_waveforms, resolve_cinc2018_record
from physionet_sleep.pipelines.cinc2018 import build_cinc2018_staging_pipeline
from physionet_sleep.recipe import StudyDef, WrappedAlgorithmStep

# Cache entries kept after a subject is built: the feature tables plus R-peak events (for heart-rate traces).
KEEP_AFTER_BUILD = ("feature_table_py", "final__feature_table_py")
KEEP_EVENTS = ("pan_tompkins_py",)


def build_cinc2018_recipe(*, ecg_detector_backend: str = "wfdb_xqrs"):
    pipeline = build_cinc2018_staging_pipeline(ecg_detector_backend=ecg_detector_backend)
    algo_names = [algorithm.name for algorithm in pipeline.algorithms]
    deps: dict[str, tuple[str, ...]] = {
        "hrv_features": ("pan_tompkins",),
        "cardiorespiratory_coupling": ("hrv_features",),
    }
    for name in algo_names:
        if name.startswith("feature_table"):
            deps[name] = tuple(other for other in algo_names if not other.startswith("feature_table"))
    return [
        WrappedAlgorithmStep(algorithm=algorithm, dependency_names=deps.get(algorithm.name, ()))
        for algorithm in pipeline.algorithms
    ]


def prune_subject_cache(subject_dir: str | Path) -> None:
    """Drop intermediate signal tracks (hundreds of MB per night at 200 Hz) once the feature table exists."""
    subject_dir = Path(subject_dir)
    if not (subject_dir / "final__feature_table_py").exists():
        return
    for entry in subject_dir.iterdir():
        if not entry.is_dir() or entry.name in KEEP_AFTER_BUILD:
            continue
        if entry.name in KEEP_EVENTS:
            for child in entry.iterdir():
                if child.name != "events":
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
            continue
        shutil.rmtree(entry, ignore_errors=True)


class CinC2018StagingStudyDef(StudyDef):
    def __init__(
        self,
        *,
        subjects: list[str] | None = None,
        root_dir: str | Path = "data/raw/cinc2018",
        cache_dir: str | Path = "artifacts/cinc2018_recipe_cache",
        ecg_detector_backend: str = "wfdb_xqrs",
    ) -> None:
        self._root_dir = Path(root_dir)
        self._algo_path = str(Path(cache_dir))
        self._subjects = subjects or [p.record_id for p in discover_cinc2018_records(self._root_dir)]
        self.recipe = build_cinc2018_recipe(ecg_detector_backend=ecg_detector_backend)

    @property
    def subjects(self) -> list[str]:
        return self._subjects

    @property
    def algo_path(self) -> str:
        return self._algo_path

    def load_fn(self, subject_id: str, req: str, meta=None):
        try:
            paths = resolve_cinc2018_record(self._root_dir, subject_id)
            if req == "signal_record":
                return load_cinc2018_waveforms(paths)
        except Exception:
            return None
        return None
