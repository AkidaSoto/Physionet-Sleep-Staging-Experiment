from __future__ import annotations

import re

from physionet_sleep.recipe.algo_def import AlgoDef
from physionet_sleep.recipe.recipe_branch import RecipeBranch


class AlgoRecipe:
    def __init__(self, recipe):
        if isinstance(recipe, RecipeBranch):
            steps = recipe.steps
            desc = _flatten_descendants(recipe)
        else:
            assert isinstance(recipe, list) and recipe
            steps = recipe
            desc = []
        for i, step in enumerate(steps):
            assert isinstance(step, AlgoDef), f"step {i} is not an AlgoDef instance"
        self._steps = steps
        self._descendant_steps = desc

    def validate(self, available_in: list[str] | None = None) -> list[str]:
        available = list(available_in) if available_in else []
        for step in self._steps:
            for req in step.requires():
                if req == "all" or not req.endswith("_algo"):
                    continue
                output_name = req[:-5]
                if output_name not in available:
                    raise ValueError(
                        f'Step "{step.algo_name()}" requires "{req}" but no earlier step declares "{output_name}".'
                    )
            available.extend(step.outputs())
        return available

    def required_streams(self) -> list[str]:
        streams: list[str] = []
        for step in self._steps:
            for rname in _normalize_algo_refs(list(step.requires()) + list(step.optionals())):
                if rname != "all" and rname not in streams:
                    streams.append(rname)
        return streams

    def flush_schedule(self) -> dict[str, list[str]]:
        streams = self.required_streams()
        all_last_idx = self._last_all_step_index()
        desc_patterns: list[str] = []
        for step in self._descendant_steps:
            desc_patterns.extend(_normalize_algo_refs(list(step.requires()) + list(step.optionals())))

        schedule: dict[str, list[str]] = {}
        for sname in streams:
            if _stream_needed_by(sname, desc_patterns):
                continue
            last_idx = -1
            for i, step in enumerate(self._steps):
                req_all = _normalize_algo_refs(list(step.requires()) + list(step.optionals()))
                if "all" in req_all:
                    last_idx = max(last_idx, all_last_idx)
                if _stream_needed_by(sname, req_all):
                    last_idx = max(last_idx, i)
            if last_idx >= 0:
                schedule.setdefault(self._steps[last_idx].algo_name(), []).append(sname)
        return schedule

    def _last_all_step_index(self) -> int:
        idx = -1
        for i, step in enumerate(self._steps):
            if "all" in step.requires():
                idx = i
        return idx


def _normalize_algo_refs(patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        out.append(p[:-len("_algo")] if p.endswith("_algo") else p)
    return out


def _flatten_descendants(branch: RecipeBranch) -> list[AlgoDef]:
    out: list[AlgoDef] = []
    for child in branch.children:
        out.extend(child.steps)
        out.extend(_flatten_descendants(child))
    return out


def _stream_needed_by(sname: str, req_all: list[str]) -> bool:
    sname_is_wc = "*" in sname or "?" in sname
    sname_re = _wildcard_to_regex(sname) if sname_is_wc else None
    for pat in req_all:
        if pat == sname:
            return True
        if pat.startswith("^"):
            if sname.endswith(pat[1:]):
                return True
            continue
        pat_is_wc = "*" in pat or "?" in pat
        if pat_is_wc and _wildcard_to_regex(pat).match(sname):
            return True
        if sname_is_wc and not pat_is_wc and sname_re.match(pat):
            return True
    return False


def _wildcard_to_regex(pattern: str):
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{escaped}$")
