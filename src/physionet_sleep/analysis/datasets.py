"""Registry of sleep datasets that share the staging feature-table format.

Each dataset knows where its raw data and recipe cache live, which subjects have a built feature table, and how to
load its labels. `load_reference_dataset` concatenates any combination into one table with a `dataset` column, so
models can be trained or evaluated on one dataset, several, or across them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from physionet_sleep.analysis.reference_data import attach_ucddb_labels_to_feature_table
from physionet_sleep.core.types import RecordLabels
from physionet_sleep.recipe.algo_store import ParquetAlgoStore

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SleepDataset:
    name: str
    raw_dir: Path
    cache_dir: Path
    load_labels: Callable[[Path, str], RecordLabels]

    def built_subjects(self) -> list[str]:
        if not self.cache_dir.exists():
            return []
        return sorted(p.name for p in self.cache_dir.iterdir() if (p / "final__feature_table_py").exists())


def _ucddb_labels(raw_dir: Path, subject_id: str) -> RecordLabels:
    from physionet_sleep.io.ucddb import resolve_ucddb_record
    from physionet_sleep.labels.ucddb import load_ucddb_labels

    return load_ucddb_labels(resolve_ucddb_record(raw_dir, subject_id))


def _cinc2018_labels(raw_dir: Path, subject_id: str) -> RecordLabels:
    from physionet_sleep.io.cinc2018 import resolve_cinc2018_record
    from physionet_sleep.labels.cinc2018 import load_cinc2018_labels

    return load_cinc2018_labels(resolve_cinc2018_record(raw_dir, subject_id))


DATASETS: dict[str, SleepDataset] = {
    "ucddb": SleepDataset("ucddb", REPO_ROOT / "data/raw/ucddb", REPO_ROOT / "artifacts/ucddb_recipe_cache", _ucddb_labels),
    "cinc2018": SleepDataset("cinc2018", REPO_ROOT / "data/raw/cinc2018", REPO_ROOT / "artifacts/cinc2018_recipe_cache", _cinc2018_labels),
}


def load_reference_dataset(
    datasets: str | Iterable[str] = "ucddb",
    *,
    subject_ids: Iterable[str] | None = None,
    feature_table_name: str = "final__feature_table",
) -> pd.DataFrame:
    names = [datasets] if isinstance(datasets, str) else list(datasets)
    wanted = set(subject_ids) if subject_ids is not None else None
    store = ParquetAlgoStore()
    tables: list[pd.DataFrame] = []
    for name in names:
        spec = DATASETS[name]
        for subject_id in spec.built_subjects():
            if wanted is not None and subject_id not in wanted:
                continue
            loaded = store.load_one(str(spec.cache_dir / subject_id), feature_table_name)
            table = loaded.artifacts.get("feature_table") if loaded is not None else None
            if not isinstance(table, pd.DataFrame) or table.empty:
                continue
            enriched = attach_ucddb_labels_to_feature_table(table, labels=spec.load_labels(spec.raw_dir, subject_id), subject_id=subject_id)
            enriched.insert(0, "dataset", name)
            tables.append(enriched)
    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, axis=0, ignore_index=True)
    return combined.drop(columns=[c for c in combined.columns if "__eeg_" in c])


def _isruc_labels(raw_dir: Path, subject_id: str) -> RecordLabels:
    from physionet_sleep.io.isruc import resolve_isruc_record
    from physionet_sleep.labels.isruc import load_isruc_labels

    return load_isruc_labels(resolve_isruc_record(raw_dir, subject_id))


DATASETS["isruc"] = SleepDataset("isruc", REPO_ROOT / "data/raw/isruc", REPO_ROOT / "artifacts/isruc_recipe_cache", _isruc_labels)
