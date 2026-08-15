from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from physionet_sleep.core.types import AlgorithmResult
from physionet_sleep.recipe.serialization import load_algorithm_result, save_algorithm_result


class AlgoStore(ABC):
    @abstractmethod
    def save(self, cache_dir: str, name: str, data) -> None: ...

    @abstractmethod
    def reload(self, subject_id: str, meta, cache_dir: str) -> tuple[list[str], list[str]]: ...

    @abstractmethod
    def load_one(self, cache_dir: str, name: str): ...


class ParquetAlgoStore(AlgoStore):
    SUFFIX = "_py"
    EXT = ".parquet"

    def save(self, cache_dir: str, name: str, data) -> None:
        dir_path = os.path.join(cache_dir, f"{name}{self.SUFFIX}")
        if isinstance(data, AlgorithmResult):
            save_algorithm_result(dir_path, data)
            return
        if isinstance(data, dict):
            path = os.path.join(cache_dir, f"{name}{self.SUFFIX}{self.EXT}")
            _write_parquet(data, path)

    def reload(self, subject_id: str, meta, cache_dir: str) -> tuple[list[str], list[str]]:
        file_pattern = os.path.join(cache_dir, f"*{self.SUFFIX}{self.EXT}")
        dir_pattern = os.path.join(cache_dir, f"*{self.SUFFIX}")
        listing = sorted(
            [path for path in glob.glob(file_pattern) if os.path.isfile(path)]
            + [path for path in glob.glob(dir_pattern) if os.path.isdir(path)]
        )
        names = list(dict.fromkeys(self._strip_suffix(os.path.basename(p)) for p in listing))
        return listing, names

    def load_one(self, cache_dir: str, name: str):
        dir_path = os.path.join(cache_dir, f"{name}{self.SUFFIX}")
        if os.path.isdir(dir_path):
            try:
                return load_algorithm_result(dir_path)
            except Exception:
                return None
        path = os.path.join(cache_dir, f"{name}{self.SUFFIX}{self.EXT}")
        if not os.path.isfile(path):
            return None
        try:
            return pd.read_parquet(path).to_dict(orient="list")
        except Exception:
            return None

    def _strip_suffix(self, filename: str) -> str:
        sfx = f"{self.SUFFIX}{self.EXT}"
        if filename.endswith(sfx):
            return filename[:-len(sfx)]
        if filename.endswith(self.SUFFIX):
            return filename[:-len(self.SUFFIX)]
        return filename


def _write_parquet(result: dict, path: str) -> None:
    lengths = [
        len(v)
        for v in result.values()
        if isinstance(v, (list, np.ndarray)) and hasattr(v, "__len__")
    ]
    n = max(lengths) if lengths else 1
    tabular = {}
    for k, v in result.items():
        if isinstance(v, (list, np.ndarray)) and hasattr(v, "__len__") and len(v) == n:
            tabular[k] = v
        else:
            tabular[k] = [v] * n
    pd.DataFrame(tabular).to_parquet(path, index=False)
