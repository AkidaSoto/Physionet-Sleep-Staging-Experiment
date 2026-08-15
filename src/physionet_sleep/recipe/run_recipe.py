from __future__ import annotations

import os
import re

from physionet_sleep.recipe.algo_recipe import AlgoRecipe
from physionet_sleep.recipe.exit_state import make_exit_state
from physionet_sleep.recipe.recipe_branch import RecipeBranch


def run_recipe(definition, opts: dict | None = None) -> dict:
    opts = opts or {}
    no_save = bool(opts.get("no_save", False))
    force = bool(opts.get("force", False))
    output_filter = list(opts.get("outputs") or [])
    if isinstance(definition.recipe, RecipeBranch):
        return _run_recipe_tree(definition, definition.recipe, no_save, force, output_filter)
    return _run_recipe_flat(definition, definition.recipe, no_save, force, output_filter)


def _run_recipe_flat(definition, recipe: list, no_save: bool, force: bool, output_filter: list[str]) -> dict:
    ar = AlgoRecipe(recipe)
    ar.validate()
    flush_sc = ar.flush_schedule()
    subjects = definition.subjects
    algo_order = [s.algo_name() for s in recipe]
    completed, completed_metas, completed_cache, completed_success = [], [], [], []

    for ed, raw_id in enumerate(subjects):
        subject_id = str(raw_id)
        cache_dir = os.path.join(definition.algo_path, subject_id)
        os.makedirs(cache_dir, exist_ok=True)
        meta = definition.subject_meta[ed] if definition.subject_meta else None
        rerun_set = set(definition.rerun)
        force_rerun_set = set(definition.force_rerun)
        cache: dict = {}
        vars_on_disk: list[str] = []
        pq_names: list[str] = []
        if not force:
            _, pq_names = definition.algo_store.reload(subject_id, meta, cache_dir)
            vars_on_disk = list(dict.fromkeys(vars_on_disk + list(pq_names)))

        needed: list[str] = []
        for step in recipe:
            if step.algo_name() in rerun_set or step.algo_name() in force_rerun_set:
                continue
            for oname in step.outputs():
                if "*" in oname or "?" in oname:
                    regex = _wildcard_to_regex(oname)
                    needed.extend(v for v in vars_on_disk if regex.match(v))
                elif oname in vars_on_disk:
                    needed.append(oname)
        needed = list(dict.fromkeys(needed))

        effectively_cached = [False] * len(recipe)
        for i in range(len(recipe) - 1, -1, -1):
            step = recipe[i]
            aname = step.algo_name()
            if aname in force_rerun_set:
                effectively_cached[i] = False
            elif i != len(recipe) - 1 and all(effectively_cached[i + 1 :]):
                effectively_cached[i] = True
            elif aname not in rerun_set:
                all_hit = True
                for o in step.outputs():
                    if "*" in o or "?" in o:
                        hit = any(_wildcard_to_regex(o).match(n) for n in needed)
                    else:
                        hit = o in needed
                    if not hit:
                        all_hit = False
                        break
                effectively_cached[i] = all_hit

        to_load = list(needed)
        if output_filter:
            requested: list[str] = []
            for pat in output_filter:
                regex = _wildcard_to_regex(pat)
                requested.extend(n for n in needed if regex.match(n))
            true_deps: list[str] = []
            for i, step in enumerate(recipe):
                if effectively_cached[i]:
                    continue
                dep_names = list(step.requires()) + _expand_caret_opts(step, i, recipe)
                for dname in dep_names:
                    if dname == "all":
                        true_deps = list(needed)
                        break
                    if dname.endswith("_algo"):
                        dname = dname[:-5]
                    if "*" in dname or "?" in dname:
                        true_deps.extend(n for n in needed if _wildcard_to_regex(dname).match(n))
                    elif dname in needed:
                        true_deps.append(dname)
            to_load = list(dict.fromkeys(requested + true_deps))

        for pq_name in pq_names:
            if pq_name in to_load:
                try:
                    cache[pq_name] = definition.algo_store.load_one(cache_dir, pq_name)
                except Exception:
                    pass

        run_ok = True
        algo_success = [False] * len(recipe)

        for i, step in enumerate(recipe):
            aname = step.algo_name()
            if effectively_cached[i]:
                cache = _flush_after_step(cache, flush_sc, aname)
                continue
            eff_opts = _expand_caret_opts(step, i, recipe)
            inputs, cache, ok, exit_state = _resolve_inputs(step, subject_id, definition, cache, eff_opts, meta)
            if not ok:
                if not no_save:
                    definition.on_subject_exit(subject_id, exit_state)
                run_ok = False
                break
            step.subject_id_ = subject_id
            exit_state = step.safe_extract(inputs)
            has_predict = bool(step.predict_outputs())
            if exit_state["status"] == "ok" and has_predict:
                exit_state = step.safe_predict()
            if exit_state["status"] == "fail":
                if not no_save:
                    definition.on_subject_exit(subject_id, exit_state)
                run_ok = False
                break
            algo_success[i] = exit_state["status"] != "fail"
            cache = _evict_dropped_streams(step, inputs, cache)

            for src, declared in ((step.features_, step.extract_outputs()), (step.result_, step.predict_outputs())):
                if not isinstance(src, dict) or not src:
                    continue
                for k, v in src.items():
                    cache[k] = v
                if not no_save and declared:
                    cache_keys = []
                    for pat in declared:
                        if "*" in pat or "?" in pat:
                            cache_keys.extend(k for k in src if _wildcard_to_regex(pat).match(k))
                        elif pat in src:
                            cache_keys.append(pat)
                    for fname in cache_keys:
                        try:
                            definition.algo_store.save(cache_dir, fname, src[fname])
                        except Exception:
                            pass
            try:
                step.post_fn(subject_id, cache_dir)
            except Exception:
                pass
            step.features_ = {}
            step.result_ = {}
            cache = _flush_after_step(cache, flush_sc, aname)

        if not run_ok:
            continue
        if not no_save:
            definition.save_fn(subject_id, cache)

        keep_pats = output_filter if output_filter else [o for s in recipe for o in s.outputs()]
        keep_res = [_wildcard_to_regex(p) for p in keep_pats]
        cache = {k: v for k, v in cache.items() if any(r.match(k) for r in keep_res)}

        completed.append(subject_id)
        completed_metas.append(meta)
        completed_cache.append(cache)
        completed_success.append(algo_success)

    return {
        "subjects": completed,
        "subject_meta": completed_metas,
        "cache": completed_cache,
        "algo_order": algo_order,
        "success": completed_success,
    }


def _run_recipe_tree(definition, root: RecipeBranch, no_save: bool, force: bool, output_filter: list[str]) -> dict:
    subjects = definition.subjects
    algo_order = [s.algo_name() for s in _flatten_all_steps(root)]
    completed, completed_metas, completed_results = [], [], []
    for ed, raw_id in enumerate(subjects):
        subject_id = str(raw_id)
        cache_dir = os.path.join(definition.algo_path, subject_id)
        os.makedirs(cache_dir, exist_ok=True)
        meta = definition.subject_meta[ed] if definition.subject_meta else None
        cache: dict = {}
        if not force:
            _, pq_names = definition.algo_store.reload(subject_id, meta, cache_dir)
            for pq_name in pq_names:
                try:
                    cache[pq_name] = definition.algo_store.load_one(cache_dir, pq_name)
                except Exception:
                    pass
        results = _run_branch(root, cache, definition, subject_id, meta, cache_dir, no_save)
        got_output = not output_filter
        for pat in output_filter:
            if _tree_has_output(results, pat):
                got_output = True
                break
        if not got_output:
            continue
        completed.append(subject_id)
        completed_metas.append(meta)
        completed_results.append(results)
    return {
        "subjects": completed,
        "subject_meta": completed_metas,
        "cache": completed_results,
        "algo_order": algo_order,
        "success": [],
    }


def _tree_has_output(results: dict, pat: str) -> bool:
    if not results["ok"]:
        return False
    if "/" in pat:
        child_name, rest = pat.split("/", 1)
        if child_name in results["children"]:
            return _tree_has_output(results["children"][child_name], rest)
    if "*" in pat or "?" in pat:
        if any(_wildcard_to_regex(pat).match(k) for k in results["cache"]):
            return True
    elif pat in results["cache"]:
        return True
    return any(_tree_has_output(child, pat) for child in results["children"].values())


def _flatten_all_steps(branch: RecipeBranch) -> list:
    steps = list(branch.steps)
    for child in branch.children:
        steps.extend(_flatten_all_steps(child))
    return steps


def _flatten_descendant_steps(branch: RecipeBranch) -> list:
    steps = []
    for child in branch.children:
        steps.extend(child.steps)
        steps.extend(_flatten_descendant_steps(child))
    return steps


def _run_branch(branch: RecipeBranch, cache: dict, definition, subject_id: str, meta, cache_dir: str, no_save: bool) -> dict:
    ar = AlgoRecipe(branch)
    flush_sc = ar.flush_schedule()
    desc_steps = _flatten_descendant_steps(branch)
    cache, run_ok, algo_success, exit_state = _run_step_list(
        branch.steps,
        cache,
        definition,
        subject_id,
        meta,
        flush_sc,
        cache_dir,
        no_save,
        set(definition.rerun),
        set(definition.force_rerun),
        branch.name,
        desc_steps,
    )
    results = {"name": branch.name, "ok": run_ok, "exit_state": exit_state, "cache": cache, "algo_success": algo_success, "children": {}}
    if run_ok and branch.is_leaf() and not no_save:
        definition.save_fn(subject_id, cache)
    if not run_ok or branch.is_leaf():
        return results
    for child in branch.children:
        child_cache = dict(cache)
        results["children"][child.name] = _run_branch(child, child_cache, definition, subject_id, meta, cache_dir, no_save)
    return results


def _run_step_list(steps: list, cache: dict, definition, subject_id: str, meta, flush_sc: dict, cache_dir: str, no_save: bool, rerun_set: set, force_rerun_set: set, branch_label: str, desc_steps: list) -> tuple:
    run_ok = True
    algo_success = [False] * len(steps)
    last_exit_state = None
    for i, step in enumerate(steps):
        aname = step.algo_name()
        if aname not in rerun_set and aname not in force_rerun_set and step.outputs_cached(cache):
            cache = _flush_after_step(cache, flush_sc, aname)
            algo_success[i] = True
            continue
        eff_opts = _expand_caret_opts(step, i, steps, desc_steps)
        inputs, cache, ok, exit_state = _resolve_inputs(step, subject_id, definition, cache, eff_opts, meta)
        if not ok:
            if not no_save:
                definition.on_subject_exit(subject_id, exit_state)
            return cache, False, algo_success, exit_state
        step.subject_id_ = subject_id
        exit_state = step.safe_extract(inputs)
        has_predict = bool(step.predict_outputs())
        if exit_state["status"] == "ok" and has_predict:
            exit_state = step.safe_predict()
        if exit_state["status"] == "fail":
            if not no_save:
                definition.on_subject_exit(subject_id, exit_state)
            return cache, False, algo_success, exit_state
        algo_success[i] = exit_state["status"] != "fail"
        last_exit_state = exit_state
        cache = _evict_dropped_streams(step, inputs, cache)
        for src, declared in ((step.features_, step.extract_outputs()), (step.result_, step.predict_outputs())):
            if not isinstance(src, dict) or not src:
                continue
            for k, v in src.items():
                cache[k] = v
            if not no_save and declared:
                cache_keys = []
                for pat in declared:
                    if "*" in pat or "?" in pat:
                        cache_keys.extend(k for k in src if _wildcard_to_regex(pat).match(k))
                    elif pat in src:
                        cache_keys.append(pat)
                for fname in cache_keys:
                    try:
                        definition.algo_store.save(cache_dir, fname, src[fname])
                    except Exception:
                        pass
        try:
            step.post_fn(subject_id, cache_dir)
        except Exception:
            pass
        step.features_ = {}
        step.result_ = {}
        cache = _flush_after_step(cache, flush_sc, aname)
    return cache, run_ok, algo_success, last_exit_state


def _flush_after_step(cache: dict, flush_sc: dict, aname: str) -> dict:
    for pat in flush_sc.get(aname, []):
        if "*" in pat or "?" in pat:
            regex = _wildcard_to_regex(pat)
            for k in [k for k in cache if regex.match(k)]:
                del cache[k]
        else:
            cache.pop(pat, None)
    return cache


def _expand_caret_opts(step, i: int, recipe: list, extra_downstream: list | None = None) -> list[str]:
    extra_downstream = extra_downstream or []
    raw = step.optionals()
    opts: list[str] = []
    for o in raw:
        if not o.startswith("^"):
            opts.append(o)
            continue
        suffix = o[1:]
        collected = []
        for ds in list(recipe[i + 1 :]) + list(extra_downstream):
            for p in list(ds.requires()) + list(ds.optionals()):
                if not p.startswith("^") and p.endswith(suffix):
                    collected.append(p)
        opts.extend(collected)
    return list(dict.fromkeys(opts))


def _resolve_inputs(step, subject_id: str, definition, cache: dict, override_opts: list[str] | None, meta) -> tuple:
    inputs: dict = {}
    aname = step.algo_name()
    excluded = cache.get("excluded_streams_", [])
    for pass_idx in range(2):
        required = pass_idx == 0
        names = step.requires() if required else (override_opts if override_opts is not None else step.optionals())
        for rname in names:
            if rname == "all":
                if required:
                    for k, v in cache.items():
                        if k not in excluded:
                            inputs[k] = v
                continue
            inputs, cache, ok, exit_state = _resolve_one(rname, inputs, cache, definition, subject_id, aname, required, meta, excluded)
            if not ok:
                return inputs, cache, False, exit_state
    return inputs, cache, True, None


def _resolve_one(rname: str, inputs: dict, cache: dict, definition, subject_id: str, aname: str, required: bool, meta, excluded: list) -> tuple:
    if "*" in rname or "?" in rname:
        regex = _wildcard_to_regex(rname)
        matched = {k: v for k, v in cache.items() if regex.match(k) and k not in excluded}
        if matched:
            inputs.update(matched)
            return inputs, cache, True, None
        try:
            streams = definition.load_fn(subject_id, rname, meta)
        except Exception:
            streams = None
        flds = {k: v for k, v in streams.items() if k not in excluded} if isinstance(streams, dict) else {}
        if flds:
            cache.update(flds)
            inputs.update(flds)
        elif required:
            return inputs, cache, False, make_exit_state("fail", "MISSING_INPUT", f'No streams matched required pattern "{rname}"', aname)
        return inputs, cache, True, None

    fkey = rname
    if rname.endswith("_algo"):
        key2 = rname[:-5]
        if key2 in cache and key2 not in excluded:
            inputs[fkey] = cache[key2]
        elif required:
            return inputs, cache, False, make_exit_state("fail", "MISSING_DEPENDENCY", f'Required output "{key2}" not in cache', aname)
    elif fkey in cache and fkey not in excluded:
        inputs[fkey] = cache[fkey]
    else:
        if fkey in excluded:
            data = None
        else:
            try:
                data = definition.load_fn(subject_id, rname, meta)
            except Exception:
                data = None
        if data is not None:
            cache[fkey] = data
            inputs[fkey] = data
        elif required:
            return inputs, cache, False, make_exit_state("fail", "MISSING_INPUT", f'Required stream "{rname}" could not be loaded', aname)
    return inputs, cache, True, None


def _is_transform_step(step) -> bool:
    return any(o.startswith("^") for o in step.optionals())


def _evict_dropped_streams(step, inputs: dict, cache: dict) -> dict:
    if not _is_transform_step(step):
        return cache
    kept = set(step.features_.keys()) if isinstance(step.features_, dict) else set()
    touched = set(inputs.keys())
    dropped = touched - kept
    if not dropped:
        return cache
    for dn in dropped:
        cache.pop(dn, None)
    prior = set(cache.get("excluded_streams_", []))
    cache["excluded_streams_"] = sorted(prior | dropped)
    return cache


def _wildcard_to_regex(pattern: str):
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{escaped}$")
