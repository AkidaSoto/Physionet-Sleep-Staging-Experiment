"""ISRUC-Sleep subgroup I through the CinC 2018 staging recipe (same algorithms, same 200 Hz montage)."""
from __future__ import annotations

from pathlib import Path

from physionet_sleep.io.isruc import discover_isruc_records, load_isruc_waveforms, resolve_isruc_record
from physionet_sleep.recipe import StudyDef
from physionet_sleep.studies.cinc2018 import build_cinc2018_recipe


class ISRUCStagingStudyDef(StudyDef):
    def __init__(
        self,
        *,
        subjects: list[str] | None = None,
        root_dir: str | Path = "data/raw/isruc",
        cache_dir: str | Path = "artifacts/isruc_recipe_cache",
        ecg_detector_backend: str = "wfdb_xqrs",
    ) -> None:
        self._root_dir = Path(root_dir)
        self._algo_path = str(Path(cache_dir))
        self._subjects = subjects or [p.record_id for p in discover_isruc_records(self._root_dir)]
        self.recipe = build_cinc2018_recipe(ecg_detector_backend=ecg_detector_backend)

    @property
    def subjects(self) -> list[str]:
        return self._subjects

    @property
    def algo_path(self) -> str:
        return self._algo_path

    def load_fn(self, subject_id: str, req: str, meta=None):
        try:
            if req == "signal_record":
                return load_isruc_waveforms(resolve_isruc_record(self._root_dir, subject_id))
        except Exception:
            return None
        return None
