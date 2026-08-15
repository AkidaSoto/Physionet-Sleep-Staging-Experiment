from __future__ import annotations

from abc import ABC, abstractmethod

from physionet_sleep.recipe.algo_store import AlgoStore, ParquetAlgoStore


class StudyDef(ABC):
    recipe: object = None
    rerun: list = []
    force_rerun: list = []
    subject_meta: list = []
    algo_store: AlgoStore = ParquetAlgoStore()

    def run_build(self, opts: dict | None = None) -> dict:
        from physionet_sleep.recipe.run_recipe import run_recipe

        return run_recipe(self, opts)

    @property
    @abstractmethod
    def subjects(self) -> list[str]: ...

    @abstractmethod
    def load_fn(self, subject_id: str, req: str, meta=None):
        ...

    @property
    @abstractmethod
    def algo_path(self) -> str: ...

    def label_cols(self) -> list[str]:
        return []

    def default_label(self) -> str:
        return ""

    def on_subject_exit(self, subject_id: str, exit_state: dict) -> None:
        pass

    def save_fn(self, subject_id: str, cache: dict) -> None:
        pass

    def label_atoms(self) -> list:
        return []
