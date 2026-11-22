
"""Nested, resumable search over named PSG feature recipes and context size."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from run_staging_feature_tuning import BASE, FEATURES, SIGNALS, TRACKS, error_structure, profile_masks, save_frame
from run_waveform_development import (
    CACHE, EXPLORATION, KEYS, LABEL, PROBABILITIES, SEED, STAGES, atomic_json, development_frame,
    fold_frames, local_training_slot, log, outer_splits, pin_manifest, scores, write_predictions,
)
from physionet_sleep.analysis.epoch_windows import WindowEncoder
from physionet_sleep.experiments.modern_tabular import _dense_cached_night
from physionet_sleep.experiments import staging_recipe_features as recipes
from physionet_sleep.experiments.tabular_runner import build_model
from physionet_sleep.experiments.waveform_clock import active_clock

OUT = EXPLORATION / "cinc-recipe-search"
CONTEXTS = (0, 1, 2, 3, 5, 8)
BASE_CONTEXT = 2
THETA_REDUCTION = "epoch_ratio_features.rel_theta"
BASELINE = {"context": BASE_CONTEXT, "features": [], "reduce": False}
JOIN = [*KEYS, LABEL, "fold"]
STUDY_PARAMS = {"startup_trials": 12, "cheap_patients": 64, "confirm_patients": 96,
                "cheap_iterations": 18, "confirm_iterations": 35}
CLASSES = list(range(5))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def measurement_slot(recipe: dict) -> str:
    spec = recipes.FAMILIES[recipe["family"]]
    return spec.slot if recipe["placement"] == "replace" else f'recipe.{recipe["family"]}'


def strip_placement(recipe: dict) -> dict:
    return {key: value for key, value in recipe.items() if key != "placement"}


class FeatureStore:
    """Reuse source and output arrays within a bounded in-process cache."""

    def __init__(self):
        self.base = lru_cache(maxsize=176)(self.base)
        self._sources = lru_cache(maxsize=12)(self._sources)
        self._candidate = lru_cache(maxsize=640)(self._candidate)

    def base(self, subject: str) -> np.ndarray:
        return _dense_cached_night(CACHE, subject, TRACKS, FEATURES)

    def _sources(self, subject: str):
        return recipes.CachedRecipeNight(CACHE, subject)

    def _candidate(self, subject: str, config: str) -> np.ndarray:
        output = self._sources(subject).compute(json.loads(config))
        result = np.asarray(output, dtype=np.float32)
        if result.ndim != 1:
            raise ValueError("Feature modules must return one aligned value per recorded second")
        return result

    def candidate(self, subject: str, recipe: dict) -> np.ndarray:
        return self._candidate(subject, json.dumps(strip_placement(recipe), sort_keys=True))

    def names(self, config: dict) -> list[str]:
        names = list(SIGNALS)
        for recipe in config["features"]:
            if recipe["placement"] == "add":
                names.append(measurement_slot(recipe))
        if config["reduce"]:
            names.remove(THETA_REDUCTION)
        return names

    def night(self, subject: str, config: dict) -> np.ndarray:
        original = self.base(subject)
        if not (config["features"] or config["reduce"]):
            return original
        planned = {measurement_slot(r): r for r in config["features"]}
        names = self.names(config)
        channels = []
        for slot in names:
            if slot in planned:
                x = self.candidate(subject, planned[slot])
                if len(x) != len(original):
                    raise ValueError(f"{subject}: feature seconds differ from the baseline clock")
            else:
                x = original[:, SIGNALS.index(slot)]
            channels.append(x)
        return np.column_stack(channels)


def window_columns(encoder: WindowEncoder, radius: int) -> list[str]:
    result = encoder.column_names()
    for offset in remote_offsets(radius):
        result += [f"{name}@{offset:+d}epoch_mean" for name in encoder.signal_names]
    return result


def remote_offsets(radius: int) -> list[int]:
    return [offset for offset in range(-radius, radius + 1) if abs(offset) > BASE_CONTEXT]


def window_matrix(encoder: WindowEncoder, radius: int, dense: dict, frames: pd.DataFrame) -> np.ndarray:
    if radius <= BASE_CONTEXT:
        return encoder.transform(dense, frames).reshape(len(frames), -1)
    offsets = np.asarray(remote_offsets(radius))
    near = encoder.n_steps * len(encoder.signal_names)
    output = np.empty((len(frames), near + len(offsets) * len(encoder.signal_names)), np.float32)
    for subject, positions in frames.groupby("subject_id", sort=False).indices.items():
        night = dense[subject]
        epochs = frames.iloc[positions]
        steps = encoder._steps(night)
        pad = (encoder.context + 1) * encoder.steps_per_epoch
        ids = epochs.epoch_index.to_numpy()
        take = (pad + (ids - encoder.context) * encoder.steps_per_epoch)[:, None] + np.arange(encoder.n_steps)
        output[positions, :near] = steps[np.clip(take, 0, len(steps) - 1)].reshape(len(positions), -1)
        central = steps[pad:-pad]
        central = np.pad(central, ((0, -len(central) % encoder.steps_per_epoch), (0, 0)))
        means = central.reshape(-1, encoder.steps_per_epoch, central.shape[1]).mean(1)
        indices = ids[:, None] + offsets
        valid = (indices >= 0) & (indices < len(means))
        distant = means[np.clip(indices, 0, len(means) - 1)] * valid[:, :, None]
        output[positions, near:] = distant.reshape(len(positions), -1)
    return output


def fit_predict(store: FeatureStore, config: dict, train: pd.DataFrame, val: pd.DataFrame, iterations: int | None):
    train_ids = sorted(train.subject_id.unique())
    if set(train_ids) & set(val.subject_id):
        raise ValueError("Model training and prediction patients overlap")
    radius = config["context"]
    encoder = WindowEncoder(store.names(config), context=min(radius, BASE_CONTEXT))
    columns = window_columns(encoder, radius)
    started = perf_counter()
    dense = {s: store.night(s, config) for s in train_ids}
    encoder.fit(dense, train_ids)
    training = window_matrix(encoder, radius, dense, train)
    del dense
    prep = perf_counter() - started
    model = build_model(model_family="hgb", task="classification", random_state=SEED)
    if iterations is not None:
        model.set_params(model__max_iter=iterations, model__early_stopping=False)
    fit_start = perf_counter()
    model.fit(pd.DataFrame(training, columns=columns), train[LABEL].to_numpy())
    del training
    fit_seconds = perf_counter() - fit_start
    inference = perf_counter()
    proba = np.zeros((len(val), len(STAGES)))
    for subject, positions in val.groupby("subject_id", sort=False).indices.items():
        x = window_matrix(encoder, radius, {subject: store.night(subject, config)}, val.iloc[positions])
        proba[np.ix_(positions, model.classes_.astype(int))] = model.predict_proba(pd.DataFrame(x, columns=columns))
    return proba, {"inputs": len(encoder.signal_names), "columns": len(columns),
                   "prepare_s": prep, "fit_s": fit_seconds, "predict_s": perf_counter() - inference}


def effect(x: np.ndarray, y: np.ndarray, first: int, second: int) -> np.ndarray | None:
    a, b = x[y == first], x[y == second]
    if min(len(a), len(b)) < 3:
        return None
    variance = ((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2)
    if np.any(variance <= 1e-12):
        return None
    return (a.mean(0) - b.mean(0)) / np.sqrt(variance)


def role_masks(night: pd.DataFrame, spec) -> dict:
    quarter, interior, _ = profile_masks(night)
    labels = night[LABEL].to_numpy()
    epoch = night.epoch_index.to_numpy()
    a, b = sorted(spec.contrast)
    adjacent = (np.diff(epoch) == 1) & (np.minimum(labels[:-1], labels[1:]) == a)
    adjacent &= np.maximum(labels[:-1], labels[1:]) == b
    left, right = np.r_[False, adjacent], np.r_[adjacent, False]
    edge = np.where(left & right, 2, np.where(left, 0, np.where(right, 4, -1)))
    return {"quarter": quarter, "interior": interior, "edge_step": edge}


def feature_check(store: FeatureStore, recipe: dict, nights: list[pd.DataFrame], *, confirm: bool) -> dict:
    spec = recipes.FAMILIES[recipe["family"]]
    baseline_slot = SIGNALS.index(spec.slot)
    results = {r: {"scores": [], "eligible": 0, "covered": 0} for r in ("all", "difficult")}
    if spec.guard is not None:
        results["guard"] = {"scores": [], "eligible": 0, "covered": 0}
    for night in nights:
        subject = str(night.subject_id.iloc[0])
        y = night[LABEL].to_numpy()
        if not np.isin(spec.contrast, y).all():
            continue
        index = night.epoch_index.to_numpy()[:, None] * 30 + np.arange(30)
        base = store.base(subject)[:, baseline_slot]
        comparison = store.candidate(subject, recipe)
        pair = np.stack([base[index], comparison[index]], axis=-1)
        masks = role_masks(night, spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pooled = np.nanmean(pair, axis=1)
            step = np.nanmean(pair.reshape(len(y), 5, 6, 2), axis=2)
        boundary = masks["edge_step"] >= 0
        focal = step[np.arange(len(y)), np.maximum(masks["edge_step"], 0)]
        for role, info in results.items():
            contrast = spec.guard if role == "guard" else spec.contrast
            included = np.isin(y, contrast)
            x = pooled
            if role == "difficult":
                included &= boundary if spec.stratum == "boundary" else masks["interior"]
                if spec.stratum == "boundary":
                    x = focal
            effects = []
            for quarter in range(4):
                eligible = included & (masks["quarter"] == quarter) & np.isfinite(x[:, 0])
                if min(int((eligible & (y == target)).sum()) for target in contrast) < 3:
                    continue
                valid = eligible & np.isfinite(x[:, 1])
                info["eligible"] += int(eligible.sum())
                info["covered"] += int(valid.sum())
                value = effect(x[valid], y[valid], *contrast)
                if value is not None:
                    effects.append(value)
            if effects:
                info["scores"].append(np.mean(effects, axis=0))
    members = {}
    n_min, rate = (20, .95) if confirm else (12, .90)
    reasons = []
    for role, info in results.items():
        scores_ = np.stack(info["scores"]) if info["scores"] else np.empty((0, 2))
        n = len(scores_)
        coverage = info["covered"] / max(1, info["eligible"])
        base, new = scores_.mean(0).tolist() if n else (None, None)
        members[role] = {"patients": n, "coverage": coverage, "reference_d": base, "d": new}
        if n < n_min or coverage < rate:
            reasons.append(f"{role}: insufficient coverage")
            continue
        if role in {"all", "difficult"} and new <= 0:
            reasons.append(f"{role}: wrong contrast direction")
        if recipe["placement"] == "replace":
            if role == "all" and new < base:
                reasons.append("weaker than replaced feature")
            if role == "guard" and new < base - .10:
                reasons.append("secondary contrast declined")
        elif role == "guard" and new < 0:
            reasons.append("secondary contrast direction reversed")
    return {"recipe": recipe, "feasible": not reasons, "reasons": reasons, "contrasts": members,
            "metadata": recipes.feature_metadata(strip_placement(recipe))}


def recipe_search_space(trial) -> dict:
    layout = trial.suggest_categorical("layout", ["unchanged", "single", "paired"])
    radius = trial.suggest_categorical("context", list(CONTEXTS))
    reduced = trial.suggest_categorical("reduce", [False, True])
    result = {"context": radius, "features": [], "reduce": reduced}
    if layout == "unchanged":
        return result
    families = sorted(recipes.FAMILIES)
    first = trial.suggest_categorical("first_family", families)
    selected = [first]
    if layout == "paired":
        selected.append(trial.suggest_categorical(f"second_after.{first}", [f for f in families if f != first]))
    for i, family in enumerate(selected):
        prefix = ["first", "second"][i]
        def suggest(name, choices):
            return trial.suggest_categorical(f"{prefix}.{name}", list(choices))
        candidate = recipes.sample_recipe(family, suggest)
        candidate["placement"] = trial.suggest_categorical(f"{prefix}.placement", ["replace", "add"])
        result["features"].append(candidate)
    return result


def anchors() -> list[dict]:
    seeds = [{"layout": "unchanged", "context": c, "reduce": False} for c in CONTEXTS]
    for family, anchor in sorted(recipes.ANCHORS.items()):
        fields = {"layout": "single", "context": BASE_CONTEXT, "reduce": False, "first_family": family,
                  "first.placement": "add" if family == "atonia" else "replace"}

        def use_anchor(name, choices):
            key = name.removeprefix(f"{family}.")
            value = anchor[key]
            if value not in choices:
                raise ValueError(f"Invalid anchor option for {family}.{key}")
            fields[f"first.{name}"] = value
            return value

        recipes.sample_recipe(family, use_anchor)
        seeds.append(fields)
    # Include a measured reduction and a paired branch before adaptive sampling.
    seeds.append({"layout": "unchanged", "context": BASE_CONTEXT, "reduce": True})
    first, second = "sigma", "atonia"
    paired = {"layout": "paired", "context": BASE_CONTEXT, "reduce": False, "first_family": first,
              f"second_after.{first}": second, "first.placement": "replace", "second.placement": "add"}
    for choice, prefix in [(first, "first"), (second, "second")]:
        def use_pair(name, choices):
            key = name.removeprefix(f"{choice}.")
            value = recipes.ANCHORS[choice][key]
            if value not in choices:
                raise ValueError(f"Invalid anchor option for {choice}.{key}")
            paired[f"{prefix}.{name}"] = value
            return value
        recipes.sample_recipe(choice, use_pair)
    seeds.append(paired)
    return seeds


def freeze_recipe(request: dict, store: FeatureStore, train: pd.DataFrame, memo: dict, *, confirm: bool):
    check = []
    selected = []
    slots = set()
    for recipe in request["features"]:
        key = fingerprint(recipe)
        if key not in memo:
            nights = [g.sort_values("epoch_index") for _, g in train.groupby("subject_id", sort=False)]
            memo[key] = feature_check(store, recipe, nights, confirm=confirm)
        outcome = memo[key]
        slot = measurement_slot(recipe)
        check.append(outcome)
        if outcome["feasible"] and slot not in slots:
            selected.append(recipe)
            slots.add(slot)
    return {"context": request["context"], "features": selected,
            "reduce": request["reduce"] and THETA_REDUCTION not in slots}, check


def validation_metrics(values: np.ndarray, truth: pd.DataFrame) -> dict:
    y = truth[LABEL].to_numpy()
    stage = f1_score(y, values.argmax(1), labels=CLASSES, average=None, zero_division=0)
    return {"macro_f1": float(stage.mean()), "stage_f1": stage.tolist()}


def study_at(path: Path, seed: int):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    url = f"sqlite:///{path.resolve().as_posix()}"
    return optuna.create_study(study_name="recipes", storage=url, direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=True)


def proxy_search(store: FeatureStore, home: Path, train, val, confirm, confirm_val, count: int, seed: int, pause):
    study = study_at(home / "trials.sqlite3", seed)
    if not study.trials:
        for seed_params in anchors():
            study.enqueue_trial(seed_params)
    for t in study.get_trials(deepcopy=False, states=[TrialState.RUNNING]):
        study.tell(t.number, state=TrialState.FAIL)
    completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    prior = {fingerprint(t.user_attrs["resolved"]): (t.value, t.user_attrs["runtime"]) for t in completed}
    marker = {}
    while len(completed) < count:
        if pause():
            return None
        next_id = 1 + max((t.number for t in study.trials if t.state != TrialState.WAITING), default=-1)
        study.sampler = optuna.samplers.TPESampler(seed=seed + next_id, n_startup_trials=STUDY_PARAMS["startup_trials"])
        trial = study.ask()
        wanted = recipe_search_space(trial)
        resolved, checks = freeze_recipe(wanted, store, train, marker, confirm=False)
        key = fingerprint(resolved)
        reused = key in prior
        if reused:
            score, runtime = prior[key]
        else:
            probs, runtime = fit_predict(store, resolved, train, val, STUDY_PARAMS["cheap_iterations"])
            score = validation_metrics(probs, val)["macro_f1"]
            prior[key] = (score, runtime)
        for attr, value in {"requested": wanted, "resolved": resolved, "checks": checks,
                            "runtime": runtime, "reused": reused}.items():
            trial.set_user_attr(attr, value)
        study.tell(trial, score)
        completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        log(f"{home.name} trial {len(completed)}/{count}: {30*(2*resolved['context']+1)}s, "
            f"{len(resolved['features'])} changes, proxy {score:.4f}" + (" (reused)" if reused else ""))
    return promote(store, home, completed, confirm, confirm_val, pause)


def promote(store, home, trials, train, val, pause):
    finalists = []
    candidates = sorted(trials, key=lambda t: (-t.value, t.number))
    distinct = {}
    for trial in candidates:
        spec = trial.user_attrs["resolved"]
        if spec != BASELINE:
            distinct.setdefault(fingerprint(spec), spec)
    specs = list(distinct.values())
    if specs:
        best = specs[0]
        options = [best, next((r for r in specs if r["context"] != best["context"]), None),
                   next((r for r in specs if r["features"] != best["features"]), None)]
        for r in options + specs:
            if r is not None and r not in finalists:
                finalists.append(r)
            if len(finalists) == 3:
                break
    checked = []
    by_key = {}
    marker = {}
    for request in [BASELINE, *finalists]:
        if pause():
            return None
        resolved, checks = freeze_recipe(request, store, train, marker, confirm=True)
        key = fingerprint(resolved)
        output = home / f"confirm-{key[:12]}.json"
        if key not in by_key:
            if output.exists():
                data = json.loads(output.read_text(encoding="utf-8"))
            else:
                combined = pd.concat(val)
                probs, runtime = fit_predict(store, resolved, train, combined, STUDY_PARAMS["confirm_iterations"])
                n = len(val[0])
                metrics = [validation_metrics(p, f) for p, f in zip([probs[:n], probs[n:]], val)]
                data = {"recipe": resolved, "metrics": metrics, "runtime": runtime}
                atomic_json(output, data)
            by_key[key] = data
        checked.append({"request": request, "resolved": resolved, "checks": checks, **by_key[key]})
    base = checked[0]
    mean = lambda row: float(np.mean([m["macro_f1"] for m in row["metrics"]]))
    passed = []
    for row in checked:
        guarded = min(a["stage_f1"][i] - b["stage_f1"][i]
                      for a,b in zip(row["metrics"],base["metrics"]) for i in (1,2)) >= -0.010
        if guarded and mean(row) > mean(base):
            passed.append(row)
    winner = max(passed,key=mean) if passed else base
    return {"recipe": winner["resolved"], "finalists": checked,
            "proxy_trials": len(trials), "proxy_fits": sum(not t.user_attrs["reused"] for t in trials)}


def partitions(frame, split, fold):
    first, val_a, test = fold_frames(frame, split)
    second = next(iter(outer_splits(first, folds=9)))
    eligible, val_b = first.iloc[second.train_idx], first.iloc[second.test_idx]
    rng = np.random.default_rng(SEED + fold)
    order = rng.permutation(sorted(eligible.subject_id.unique()))
    cheap = eligible[eligible.subject_id.isin(order[:STUDY_PARAMS["cheap_patients"]])]
    confirm = eligible[eligible.subject_id.isin(order[:STUDY_PARAMS["confirm_patients"]])]
    groups = [set(f.subject_id) for f in (confirm, val_a, val_b, test)]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("Inner fitting, validation and outer patients must be disjoint")
    if not set(cheap.subject_id) <= groups[0]:
        raise ValueError("Cheap patients must be contained in confirmation training")
    return cheap, val_a, confirm, val_b, test


def check_smoke(frame, split, store):
    cheap, val, confirm, val_b, test = partitions(frame, split, 0)
    subject = sorted(cheap.subject_id.unique())[0]
    dense = _dense_cached_night(CACHE, subject, TRACKS, FEATURES)
    np.testing.assert_array_equal(dense, store.night(subject, BASELINE))
    sample = cheap[cheap.subject_id.eq(subject)].iloc[:12]
    encoder = WindowEncoder(list(SIGNALS), context=2).fit({subject: dense}, [subject])
    exact = encoder.transform({subject: dense}, sample).reshape(len(sample), -1)
    np.testing.assert_array_equal(exact, window_matrix(encoder, 2, {subject: dense}, sample))
    expanded = window_matrix(encoder, 8, {subject: dense}, sample)
    np.testing.assert_array_equal(exact, expanded[:, :exact.shape[1]])
    assert expanded.shape[1] == len(window_columns(encoder, 8))
    trial_path = OUT / "smoke"
    trial_path.mkdir(exist_ok=True)
    study = study_at(trial_path / "smoke.sqlite3", SEED)
    show = [a for a in anchors() if a["layout"] == "single" and a["first_family"] in {"sigma", "atonia"}]
    if len(show) != 2:
        raise ValueError("Smoke check needs two separate feature families")
    if not study.trials:
        tr_subjects = sorted(cheap.subject_id.unique())[:8]
        val_subjects = sorted(val.subject_id.unique())[:3]
        tr, va = cheap[cheap.subject_id.isin(tr_subjects)], val[val.subject_id.isin(val_subjects)]
        for seed_params in show:
            study.enqueue_trial(seed_params)
            trial = study.ask()
            config = recipe_search_space(trial)
            original = store.night(subject, config)
            tr_probs, _ = fit_predict(store, config, tr, va, 2)
            trial.set_user_attr("recipe", config)
            trial.set_user_attr("array_sha256", hashlib.sha256(original.tobytes()).hexdigest())
            study.tell(trial, validation_metrics(tr_probs, va)["macro_f1"])
    reopened = study_at(trial_path / "smoke.sqlite3", SEED)
    fresh_store = FeatureStore()
    completed = reopened.get_trials(states=[TrialState.COMPLETE])
    if len(completed) != 2:
        raise ValueError("Smoke study did not retain both completed trials")
    for trial in completed:
        array = fresh_store.night(subject, trial.user_attrs["recipe"])
        assert hashlib.sha256(array.tobytes()).hexdigest() == trial.user_attrs["array_sha256"]
    new_parts = partitions(frame, split, 0)
    for old,new in zip((cheap,val,confirm,val_b,test),new_parts):
        pd.testing.assert_frame_equal(old[JOIN],new[JOIN])
    atomic_json(trial_path / "summary.json", {"passed": True, "baseline_array_exact": True, "two_feature_families": True,
                 "test_patients_scored": 0, "resumed_trials": len(completed), "contexts_reproduced": [150,510]})
    log("Training-only smoke passed: exact baseline, long window, feature arrays, patient keys and SQLite replay")


def report(frame: pd.DataFrame):
    aligned = frame[JOIN + ["prediction"]].rename(columns={"prediction":"baseline"})
    base_prob = frame[PROBABILITIES].to_numpy()
    names = ["baseline","searched","fixed_context"]
    probs = {"baseline":base_prob}
    for name in names[1:]:
        saved = pd.concat([pd.read_parquet(OUT/f"fold-{f}/{name}.parquet") for f in range(5)])
        merged = frame[JOIN].merge(saved,on=JOIN,validate="one_to_one")
        if len(merged) != len(frame) or len(saved) != len(frame):
            raise ValueError("Predictions have missing or extra original fold keys")
        probs[name] = merged[PROBABILITIES].to_numpy()
        aligned[name] = probs[name].argmax(1)
        save_frame(OUT/f"{name}_predictions.parquet",merged)
    rows = [np.flatnonzero(aligned.fold.eq(f)) for f in range(5)]
    y = aligned[LABEL].to_numpy()
    ids = aligned.subject_id.to_numpy()
    cm, patients = patient_confusions(ids,y,aligned.baseline.to_numpy())
    groups = [np.flatnonzero(aligned.groupby("subject_id").fold.first().loc[patients].eq(f)) for f in range(5)]
    draws = cv_fold_draws(groups,2000,np.random.default_rng(SEED))
    base = cv_scores(cm,groups,draws)["macro_f1"]
    metrics = {}
    for v in names:
        predicted = aligned[v].to_numpy()
        metrics[v] = scores(aligned,predicted,rows)
        p = probs[v]
        metrics[v]["log_loss"] = float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
        metrics[v]["brier"] = float(np.square(p-np.eye(len(STAGES))[y]).sum(1).mean())
    for v in names[1:]:
        predicted = aligned[v].to_numpy()
        deltas = [float(f1_score(y[r],predicted[r],average="macro")-f1_score(y[r],aligned.baseline.to_numpy()[r],average="macro"))
                  for r in rows]
        stage = {s:metrics[v][f"f1_{s}"]-metrics["baseline"][f"f1_{s}"] for s in STAGES}
        other,_ = patient_confusions(ids,y,predicted)
        band = interval(cv_scores(other,groups,draws)["macro_f1"]-base)
        metrics[v].update(delta=float(np.mean(deltas)),fold_deltas=deltas,stage_deltas=stage,paired_ci=band,
                          passes=float(np.mean(deltas))>=.010 and sum(d>0 for d in deltas)>=4 and min(stage.values())>=-.010)
        result = []
        for sid,g in aligned.groupby("subject_id",sort=False):
            truth = g[LABEL]
            stages = sorted(truth.unique())
            s0,s1 = [f1_score(truth,g[k],labels=stages,average="macro",zero_division=0) for k in ("baseline",v)]
            result.append((sid,int(g.fold.iloc[0]),s0,s1,s1-s0))
        pd.DataFrame(result,columns=["subject_id","fold","baseline_f1","candidate_f1","delta"]).to_csv(
            OUT/f"{v}_vs_baseline_patients.csv",index=False)
    contexts = [json.loads((OUT/f"fold-{f}/selection.json").read_text(encoding="utf-8")) for f in range(5)]
    errors = {v:error_structure(aligned,v) for v in names}
    atomic_json(OUT/"summary.json",{"metrics":metrics,"fold_recipes":contexts,"errors":errors})
    log(f'Searched recipe {metrics["searched"]["macro_f1"]:.4f} macro-F1 ({metrics["searched"]["delta"]:+.4f})')


def execute(args):
    if not (OUT / "FINDINGS.md").exists():
        raise FileNotFoundError("Declare the search protocol first")
    frame, splits = development_frame()
    for fold, split in enumerate(splits):
        frame.loc[frame.index[split.test_idx], "fold"] = fold
    frame["fold"] = frame.fold.astype(int)
    store = FeatureStore()
    if args.smoke:
        check_smoke(frame, splits[0], store)
        return 0
    if not (OUT / "smoke/summary.json").is_file():
        raise ValueError("The training-only verification must finish first")
    if args.trials < len(anchors()):
        raise ValueError("Trial budget must include all declared anchor recipes")
    archived = pd.read_parquet(BASE)
    base = frame[JOIN].merge(archived,on=JOIN,validate="one_to_one")
    if len(base) != len(frame) or len(archived) != len(frame):
        raise ValueError("Development and archived baseline keys differ")
    src = [Path(__file__), Path(recipes.__file__)]
    folds = [partitions(frame,s,f) for f,s in enumerate(splits)]
    pin_manifest(OUT/"manifest.json", {"trial_ceiling":args.trials,"seed":SEED,"contexts":list(CONTEXTS),
                "fidelity":STUDY_PARAMS,"patient_partitions":[[sorted(p.subject_id.unique()) for p in f] for f in folds],
                "optuna":optuna.__version__,
                "inputs":SIGNALS,"baseline_sha256":hashlib.sha256(BASE.read_bytes()).hexdigest(),
                "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src}})
    budget = OUT/"active_budget.json"
    used = json.loads(budget.read_text())["seconds"] if budget.exists() else 0.0
    started = active_clock()

    def pause():
        elapsed = used + active_clock() - started
        atomic_json(budget, {"seconds": elapsed})
        return (OUT/"PAUSE").exists() or elapsed >= 86400

    try:
        for f,(cheap,first,confirm,second,test) in enumerate(folds):
            if pause():
                return 75
            home = OUT/f"fold-{f}"
            home.mkdir(exist_ok=True)
            chosen = home/"selection.json"
            if not chosen.exists():
                selection = proxy_search(store,home,cheap,first,confirm,[first,second],args.trials,SEED+f*1000,pause)
                if selection is None:
                    return 75
                atomic_json(chosen,selection)
            winner = json.loads(chosen.read_text(encoding="utf-8"))["recipe"]
            train = frame.iloc[splits[f].train_idx]
            completed = {}
            for name,config in [("searched",winner),("fixed_context",{**winner,"context":BASE_CONTEXT})]:
                path = home/f"{name}.parquet"
                if path.exists():
                    completed[fingerprint(config)] = pd.read_parquet(path)[PROBABILITIES].to_numpy()
                    continue
                if pause():
                    return 75
                if config == BASELINE:
                    anchored = test[JOIN].merge(base[JOIN+PROBABILITIES],on=JOIN,validate="one_to_one")
                    if len(anchored)!=len(test):
                        raise ValueError("Archived baseline could not be aligned to fold")
                    y = anchored[PROBABILITIES].to_numpy()
                    runtime = {"reused": "archived baseline"}
                elif fingerprint(config) in completed:
                    y, runtime = completed[fingerprint(config)], {"reused": "identical recipe"}
                else:
                    log(f"fold {f+1} refitting {name}: context {30*(2*config['context']+1)} s, {len(config['features'])} changes")
                    y, runtime = fit_predict(store,config,train,test,None)
                atomic_json(home/f"{name}_runtime.json",runtime)
                write_predictions(path,test,y,f)
                completed[fingerprint(config)] = y
        report(base)
    finally:
        pause()
    return 0


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--smoke", action="store_true", help="Run the training-only verification")
    cli.add_argument("--trials", type=int, default=20, help="First-rung trial ceiling per outer fold")
    with local_training_slot(), threadpool_limits(limits=4):
        raise SystemExit(execute(cli.parse_args()))

