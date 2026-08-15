from __future__ import annotations

import copy
import re
import traceback
from abc import ABC, abstractmethod

from physionet_sleep.recipe.exit_state import make_exit_state


class AlgoDef(ABC):
    def __init__(self) -> None:
        self.params: dict = {}
        self.verbose: bool = False
        self.subject_id_: str = ""
        self.features_: dict = {}
        self.result_: dict = {}

    def vprint(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def pprint(self, hub: str, fmt: str, *args) -> None:
        kvs = fmt % args if args else fmt
        print(f"[PARITY][{self.algo_name()}][{self.subject_id_}][{hub}] {kvs}")

    @abstractmethod
    def algo_name(self) -> str: ...

    @abstractmethod
    def requires(self) -> list[str]: ...

    @abstractmethod
    def extract_outputs(self) -> list[str]: ...

    @abstractmethod
    def predict_outputs(self) -> list[str]: ...

    @abstractmethod
    def extract(self, inputs: dict, params: dict) -> None: ...

    def optionals(self) -> list[str]:
        return []

    def default_params(self) -> dict:
        return {}

    def predict(self, params: dict) -> None:
        raise NotImplementedError(f'"{self.algo_name()}" does not implement predict()')

    def post_fn(self, subject_id: str, out_dir: str) -> None:
        pass

    def format_params(self) -> dict:
        return {"lookback": 0, "lookforward": 0}

    def outputs(self) -> list[str]:
        return self.extract_outputs() + self.predict_outputs()

    def get_params(self) -> dict:
        p = copy.deepcopy(self.default_params())
        p.update(self.params)
        return p

    def ok_state(self) -> dict:
        return make_exit_state("ok", "OK", "", self.algo_name())

    def fail_state(self, code: str, msg: str) -> dict:
        return make_exit_state("fail", code, msg, self.algo_name())

    def outputs_cached(self, cache: dict) -> bool:
        cache_keys = list(cache.keys())
        for o in self.outputs():
            if "*" in o or "?" in o:
                regex = re.compile("^" + re.escape(o).replace(r"\*", ".*").replace(r"\?", ".") + "$")
                if not any(regex.match(k) for k in cache_keys):
                    return False
            elif o not in cache:
                return False
        return True

    def safe_extract(self, inputs: dict) -> dict:
        p = self.get_params()
        try:
            self.extract(inputs, p)
            return self.ok_state()
        except Exception as exc:
            traceback.print_exc()
            return self.fail_state("EXTRACT_FAILED", str(exc))

    def safe_predict(self) -> dict:
        p = self.get_params()
        try:
            self.predict(p)
            return self.ok_state()
        except Exception as exc:
            traceback.print_exc()
            return self.fail_state("PREDICT_FAILED", str(exc))
