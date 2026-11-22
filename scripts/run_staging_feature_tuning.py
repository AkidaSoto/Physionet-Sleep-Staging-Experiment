
"""Run the preregistered Boston feature tuning and reduction experiments, #44–46."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from run_waveform_development import (
    CACHE, EXPLORATION, KEYS, LABEL, PROBABILITIES, RAW, SEED, STAGES, REDUCTION,
    atomic_json, development_frame, fold_frames, local_training_slot, log, pin_manifest,
    scores, write_predictions,
)
from run_window_hmm_staging import adopted_signals, build_model
from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from physionet_sleep.analysis.epoch_windows import WindowEncoder, robust_scale
from physionet_sleep.experiments.cached_staging_tuning import (
    CACHE_COLUMNS, TRACK_COLUMNS, SIGMA_CANDIDATES, THETA_CANDIDATES, cached_staging_candidates,
)
from physionet_sleep.experiments.modern_tabular import REMOVED_TRACKS, _dense_cached_night
from physionet_sleep.experiments.staging_feature_waveform import (
    EOG_VARIANTS, SLOW_WAVE_VARIANTS, RawStagingFeatureCache,
)
from physionet_sleep.experiments.waveform_clock import active_clock

OUT = EXPLORATION / "cinc-feature-tuning"
BASE = REDUCTION / "reduction/predictions.parquet"
TRACKS, FEATURES = adopted_signals()
TRACKS = [s for s in TRACKS if s not in REMOVED_TRACKS]
SIGNALS = TRACKS + FEATURES
SLOW = "topography_features.frontal_slow_wave_fraction"
THETA = "c_rel_theta"
SIGMA = "c_rel_beta"
EOG = "epoch_ratio_features.eog_rem_vs_noise_db"
DROPPED = {"epoch_ratio_features.rel_theta", "eye_movement_activity.eye_movement_baseline"}
FAMILIES = {
    # Original slot, candidates, physiological contrast, secondary guard, difficult stratum.
    SLOW: (SLOW_WAVE_VARIANTS, (3, 2), None, "interior"),
    THETA: (THETA_CANDIDATES, (1, 0), None, "boundary"),
    SIGMA: (SIGMA_CANDIDATES, (2, 3), (2, 1), "interior"),
    EOG: (EOG_VARIANTS, (4, 1), (4, 0), "interior"),
}
CANDIDATES = [item for specs in FAMILIES.values() for item in specs[0]]


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(".part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def profile_masks(night: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epoch = night.epoch_index.to_numpy()
    y = night[LABEL].to_numpy()
    change = (np.diff(epoch) == 1) & (np.diff(y) != 0)
    left, right = np.r_[False, change], np.r_[change, False]
    same = (np.diff(epoch) == 1) & (np.diff(y) == 0)
    interior = np.r_[False, False, same[1:] & same[:-1]] & np.r_[same[1:] & same[:-1], False, False]
    quarter = np.minimum(3, 4 * (epoch - epoch.min()) // (epoch.max() - epoch.min() + 1))
    return quarter, interior, np.where(left & right, 2, np.where(left, 0, np.where(right, 4, -1)))


def subject_candidates(raw: RawStagingFeatureCache, subject: str) -> pd.DataFrame:
    slots_dir = OUT / "slot_cache"
    slots_dir.mkdir(exist_ok=True)
    path = slots_dir / f"{subject}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    tracks = pd.read_parquet(CACHE/"tracks1s"/f"{subject}.parquet", columns=list(TRACK_COLUMNS))
    quality = pd.read_parquet(CACHE/"quality1s"/f"{subject}.parquet", columns=list(CACHE_COLUMNS))
    slots = cached_staging_candidates(tracks, quality)
    waveform = raw.read(subject)
    if waveform.index.name != "epoch_index" and "epoch_index" in waveform:
        waveform = waveform.set_index("epoch_index", verify_integrity=True)
    supplemental = waveform.reindex(slots.index // 5).set_axis(slots.index)
    frame = slots.join(supplemental)[CANDIDATES].astype(np.float32)
    save_frame(path, frame)
    return frame


def per_night_profile(raw: RawStagingFeatureCache, night: pd.DataFrame) -> list[dict]:
    subject = str(night.subject_id.iloc[0])
    dense = _dense_cached_night(CACHE, subject, TRACKS, FEATURES)
    slots = subject_candidates(raw, subject)
    index = night.epoch_index.to_numpy()
    seconds = index[:, None] * 30 + np.arange(30)
    steps = index[:, None] * 5 + np.arange(5)
    if seconds.max() >= len(dense):
        raise ValueError(f"Epoch beyond observed track samples for {subject}")
    values = dense[seconds].reshape(len(night), 5, 6, len(SIGNALS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        originals = np.nanmean(values, axis=2)
    alt = slots.to_numpy(np.float32)[steps]
    labels = night[LABEL].to_numpy()
    quarter, interior, _ = profile_masks(night)
    rows = []
    wake_n1 = (np.diff(index) == 1) & (np.minimum(labels[:-1], labels[1:]) == 0)
    wake_n1 &= (np.maximum(labels[:-1], labels[1:]) == 1)
    left, right = np.r_[False, wake_n1], np.r_[wake_n1, False]
    edge_step = np.where(left & right, 2, np.where(left, 0, np.where(right, 4, -1)))
    for slot, (candidates, pair, guard, difficult) in FAMILIES.items():
        x = np.column_stack([originals[:, :, SIGNALS.index(slot)].mean(axis=1),
                            alt[:, :, [CANDIDATES.index(c) for c in candidates]].mean(axis=1)])
        stages = [("all", pair, np.ones(len(night), bool), x)]
        if difficult == "boundary":
            used = np.maximum(edge_step, 0)
            own = np.arange(len(night))
            at_boundary = np.column_stack([originals[own, used, SIGNALS.index(slot)],
                                           alt[own, used][:, [CANDIDATES.index(c) for c in candidates]]])
            stages.append(("difficult", pair, edge_step >= 0, at_boundary))
        else:
            stages.append(("difficult", pair, interior, x))
        if guard is not None:
            stages.append(("guard", guard, np.ones(len(night), bool), x))
        for name, (true, rival), mask, features in stages:
            for col, candidate in enumerate(candidates, start=1):
                common = mask & np.isfinite(features[:, 0]) & np.isfinite(features[:, col])
                per_quarter = []
                eligible_epochs = finite_epochs = use_epochs = 0
                for q in range(4):
                    a = (quarter == q) & (labels == true) & mask
                    b = (quarter == q) & (labels == rival) & mask
                    expected = np.isfinite(features[:, 0])
                    if min(int((a & expected).sum()), int((b & expected).sum())) < 3:
                        continue
                    eligible_epochs += int(((a | b) & expected).sum())
                    finite_epochs += int(((a | b) & common).sum())
                    group_a, group_b = features[a & common][:, [0, col]], features[b & common][:, [0, col]]
                    if min(len(group_a), len(group_b)) < 3:
                        continue
                    var = ((len(group_a)-1)*group_a.var(0, ddof=1) + (len(group_b)-1)*group_b.var(0, ddof=1))
                    pooled = np.sqrt(var / (len(group_a)+len(group_b)-2))
                    if np.any(pooled <= 1e-8):
                        continue
                    per_quarter.append((group_a.mean(0) - group_b.mean(0))/pooled)
                    use_epochs += len(group_a) + len(group_b)
                if eligible_epochs:
                    base, tuned = np.mean(per_quarter, axis=0) if per_quarter else (np.nan,np.nan)
                    rows.append({"subject_id": subject, "slot": slot, "candidate": candidate, "stratum": name,
                                 "base": float(base), "tuned": float(tuned), "matched_epochs": use_epochs,
                                 "eligible_epochs": eligible_epochs, "finite_epochs": finite_epochs})
    return rows


def marker_selection(profile: pd.DataFrame, patients: set[str]) -> tuple[dict, list[dict]]:
    selected = {}
    summary = []
    table = profile[profile.subject_id.isin(patients)]
    for slot, (candidates, _, guard, _) in FAMILIES.items():
        passing = []
        for candidate in candidates:
            rows = table[(table.slot == slot) & (table.candidate == candidate)]
            effects = rows.groupby("stratum")[["base","tuned"]].mean()
            n = rows.dropna(subset=["tuned"]).groupby("stratum").subject_id.nunique()
            coverage = rows.groupby("stratum")[["finite_epochs","eligible_epochs"]].sum()
            required = ["all", "difficult", "guard"] if guard else ["all", "difficult"]
            fractions = coverage["finite_epochs"] / coverage["eligible_epochs"]
            min_coverage = float(fractions.reindex(required,fill_value=0.0).min())
            def value(stratum: str, metric: str) -> float:
                return float(effects.loc[stratum, metric]) if stratum in effects.index else float("nan")

            overall = value("all", "tuned")
            hard = value("difficult", "tuned")
            baseline_overall = value("all", "base")
            baseline_hard = value("difficult", "base")
            guard_delta = value("guard", "tuned") - value("guard", "base") if guard else 0.0
            count = min(int(n.get(name,0)) for name in required)
            passes = bool(count >= 20 and min_coverage >= .95 and overall > 0 and hard > 0 and overall >= baseline_overall
                          and hard >= baseline_hard + 0.05 and guard_delta >= -0.10)
            summary.append({"slot": slot, "candidate": candidate, "minimum_patients": count, "minimum_finite_coverage":min_coverage,
                            "baseline_d": baseline_overall, "candidate_d": overall,
                            "baseline_difficult_d": baseline_hard, "candidate_difficult_d": hard,
                            "guard_delta": guard_delta, "passes": passes})
            if passes:
                passing.append((candidate, hard, overall))
        if passing:
            selected[slot] = max(passing, key=lambda item: item[1:])[0]
    return selected, summary


def dense_night(subject: str, replacements: dict, dropped: set[str]) -> np.ndarray:
    dense = _dense_cached_night(CACHE, subject, TRACKS, FEATURES)
    if replacements:
        alt = pd.read_parquet(OUT/"slot_cache"/f"{subject}.parquet", columns=list(replacements.values()))
        for slot, candidate in replacements.items():
            signal = np.repeat(alt[candidate].to_numpy(np.float32), 6)[:len(dense)]
            dense[:, SIGNALS.index(slot)] = robust_scale(signal[:,None], [slot])[:,0]
    return dense[:, [i for i, slot in enumerate(SIGNALS) if slot not in dropped]] if dropped else dense


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, replacements: dict,
                *, inner: bool = False, dropped: set[str] = frozenset()) -> tuple[np.ndarray, dict]:
    names = [s.split(".")[-1] for s in SIGNALS if s not in dropped]
    encoder = WindowEncoder(names, context=2)
    columns = encoder.column_names()
    subjects = sorted(train.subject_id.unique())
    start = perf_counter()
    dense = {s: dense_night(s, replacements, dropped) for s in subjects}
    encoder.fit(dense, subjects)
    windows = encoder.transform(dense, train).reshape(len(train), len(columns))
    del dense
    preparation = perf_counter() - start
    model = build_model(model_family="hgb", task="classification", random_state=SEED)
    if inner:
        model.set_params(model__max_iter=35, model__early_stopping=False)
    start = perf_counter()
    model.fit(pd.DataFrame(windows, columns=columns), train[LABEL].to_numpy())
    del windows
    training = perf_counter() - start
    probabilities = np.zeros((len(test), 5))
    classes = np.asarray(model.classes_, dtype=int)
    start = perf_counter()
    for subject, positions in test.groupby("subject_id", sort=False).indices.items():
        dense = dense_night(subject, replacements, dropped)
        windows = encoder.transform({subject:dense}, test.iloc[positions]).reshape(len(positions), len(columns))
        probabilities[np.ix_(positions,classes)] = model.predict_proba(pd.DataFrame(windows, columns=columns))
    return probabilities, {"inputs":len(names), "prepare_s":preparation, "fit_s":training,
                           "test_cache_to_predictions_s":perf_counter()-start}


def inner_search(frame: pd.DataFrame, split, profile: pd.DataFrame, fold: int) -> tuple[dict, list, list]:
    tr, val, _ = fold_frames(frame, split)
    patients = sorted(tr.subject_id.unique())
    selected, marker = marker_selection(profile, set(patients))
    if not selected:
        return {}, [], marker
    seed = np.random.default_rng(SEED + fold)
    tr = tr[tr.subject_id.isin(seed.choice(patients, min(96,len(patients)), replace=False))]
    core = {slot: candidate for slot,candidate in selected.items() if slot in {SLOW,THETA}}
    recipes = {"default":{}}
    for variant, recipe in [("core",core),("full",selected)]:
        if recipe not in recipes.values():
            recipes[variant] = recipe
    results = []
    for variant, recipe in recipes.items():
        start = perf_counter()
        probs, _ = fit_predict(tr, val, recipe, inner=True)
        per_class = f1_score(val[LABEL], probs.argmax(1), labels=np.arange(5), average=None, zero_division=0)
        results.append({"bundle":variant, "macro_f1":float(per_class.mean()), "f1_N1":float(per_class[1]),
                        "f1_N2":float(per_class[2]), "seconds":perf_counter()-start,
                        "train_patients":tr.subject_id.nunique(), "validation_patients":val.subject_id.nunique()})
        log(f"fold {fold + 1}: inner {variant}, macro-F1 {per_class.mean():.4f}")
    base = results[0]
    passing = [r for r in results if r["macro_f1"] > base["macro_f1"]
               and min(r[k]-base[k] for k in ("f1_N1","f1_N2")) >= -0.010]
    winner = max(passing,key=lambda r:r["macro_f1"])["bundle"] if passing else "default"
    return recipes[winner], results, marker


def compare(frame: pd.DataFrame, candidate: str, comparator: str) -> dict:
    truth = frame[LABEL].to_numpy()
    folds = [np.flatnonzero(frame.fold.to_numpy()==f) for f in range(5)]
    pred, base = frame[candidate].to_numpy(), frame[comparator].to_numpy()
    old = scores(frame, base, folds)
    new = scores(frame, pred, folds)
    deltas = [float(f1_score(truth[i],pred[i],average="macro")-f1_score(truth[i],base[i],average="macro"))
              for i in folds]
    cm_a, ids = patient_confusions(frame.subject_id.to_numpy(),truth,pred)
    cm_b, _ = patient_confusions(frame.subject_id.to_numpy(),truth,base)
    f = frame.groupby("subject_id").fold.first().loc[ids].to_numpy()
    groups = [np.flatnonzero(f==k) for k in range(5)]
    draws = cv_fold_draws(groups,2000,np.random.default_rng(SEED))
    ci = interval(cv_scores(cm_a,groups,draws)["macro_f1"]-cv_scores(cm_b,groups,draws)["macro_f1"])
    class_delta = {s:new[f"f1_{s}"]-old[f"f1_{s}"] for s in STAGES}
    patient_rows = []
    for subject, night in frame.groupby("subject_id", sort=False):
        y = night[LABEL].to_numpy()
        labels = np.unique(y)
        a = f1_score(y,night[candidate],labels=labels,average="macro",zero_division=0)
        b = f1_score(y,night[comparator],labels=labels,average="macro",zero_division=0)
        patient_rows.append((subject,int(night.fold.iloc[0]),a,b,a-b))
    pd.DataFrame(patient_rows, columns=["subject_id","fold","candidate_f1","comparison_f1","delta"]).to_csv(
        OUT/f"{candidate}_vs_{comparator}_patients.csv", index=False)
    return {"candidate":new,"comparison":old,"delta":float(np.mean(deltas)),"fold_deltas":deltas,
            "stage_deltas":class_delta,"paired_ci":ci, "evaluated_folds":len(deltas)}


def error_structure(frame: pd.DataFrame, variant: str) -> list[dict]:
    result = []
    for _, night in frame.sort_values(KEYS).groupby("subject_id",sort=False):
        _, interior, boundary = profile_masks(night)
        y = night[LABEL].to_numpy()
        pred = night[variant].to_numpy()
        epoch = night.epoch_index.to_numpy()
        neighbors = np.diff(epoch) == 1
        early_switch = np.r_[neighbors & (y[:-1] != y[1:]) & (pred[:-1] == y[1:]), False]
        late_switch = np.r_[False, neighbors & (y[:-1] != y[1:]) & (pred[1:] == y[:-1])]
        new = np.r_[True,(np.diff(epoch)!=1)|(np.diff(y)!=0)|(np.diff(pred)!=0)]
        run = np.cumsum(new)
        long = np.bincount(run)[run]>=6
        for i,j in [(0,1),(1,0),(2,1),(1,2),(2,3),(3,2),(1,4),(4,1)]:
            errors = (y==i)&(pred==j)
            result.append({"true_stage":STAGES[i],"predicted_stage":STAGES[j],"errors":int(errors.sum()),
                           "boundary":int((errors & (boundary>=0)).sum()),
                           "interior":int((errors & interior).sum()),
                           "calls_next_stage_early":int((errors & early_switch).sum()),
                           "keeps_previous_stage_late":int((errors & late_switch).sum()),
                           "sustained":int((errors & long).sum())})
    return pd.DataFrame(result).groupby(["true_stage","predicted_stage"],as_index=False).sum().to_dict("records")


def report(base: pd.DataFrame) -> None:
    report_frame = base[[*KEYS,LABEL,"fold","prediction"]].rename(columns={"prediction":"baseline"})
    for variant in ("joint","reduced"):
        pred = pd.concat([pd.read_parquet(OUT/f"fold-{f}"/f"{variant}.parquet") for f in range(5)])
        save_frame(OUT/f"{variant}_predictions.parquet",pred)
        pred[variant] = pred[PROBABILITIES].to_numpy().argmax(1)
        report_frame = report_frame.merge(pred[[*KEYS,LABEL,"fold",variant]],on=[*KEYS,LABEL,"fold"],
                                          validate="one_to_one")
        if len(report_frame) != len(base):
            raise ValueError("Cross-validation prediction keys are incomplete")
    joint = compare(report_frame,"joint","baseline")
    reduced = compare(report_frame,"reduced","baseline")
    reduced_joint = compare(report_frame,"reduced","joint")
    function_ok = lambda r: r["delta"] >= -0.005 and min(r["fold_deltas"]) >= -0.005 and min(r["stage_deltas"].values()) >= -0.010
    joint["advanced"] = joint["delta"]>=0.010 and sum(d>0 for d in joint["fold_deltas"])>=4 and min(joint["stage_deltas"].values())>=-0.010
    reduced["advanced"] = function_ok(reduced) and function_ok(reduced_joint)
    payload = {"joint":joint,"reduced":reduced,"reduced_vs_joint":reduced_joint,
               "errors":{name:error_structure(report_frame,name) for name in ("baseline","joint","reduced")}}
    payload["fold_recipes"] = [json.loads((OUT/f"fold-{fold}"/"recipe.json").read_text()) for fold in range(5)]
    payload["runtime"] = {variant:[json.loads((OUT/f"fold-{fold}"/f"{variant}_runtime.json").read_text())
                                   for fold in range(5)] for variant in ("joint","reduced")}
    payload["actual_fits"] = {variant:sum(not fit.get("reused",False) for fit in runtime)
                               for variant,runtime in payload["runtime"].items()}
    payload["actual_fits"]["inner"] = sum(len(recipe["inner"]) for recipe in payload["fold_recipes"])
    atomic_json(OUT/"summary.json",payload)
    for v in ("joint","reduced"):
        log(f'{v}: {payload[v]["candidate"]["macro_f1"]:.4f} macro-F1, {payload[v]["delta"]:+.4f} over HGB18')


def execute() -> int:
    if not (OUT/"FINDINGS.md").is_file():
        raise ValueError("Preregister the experiment before running")
    frame, splits = development_frame()
    base = pd.read_parquet(BASE)
    base_keys = base[[*KEYS,LABEL,"fold"]]
    frame["fold"] = -1
    for f,split in enumerate(splits):
        frame.iloc[split.test_idx,frame.columns.get_loc("fold")] = f
    if len(frame.merge(base_keys,on=[*KEYS,LABEL,"fold"],validate="one_to_one")) != len(frame) or len(base) != len(frame):
        raise ValueError("Saved baseline keys do not match the original evaluation folds")
    sources = [Path(__file__), Path(sys.modules[RawStagingFeatureCache.__module__].__file__),
               Path(sys.modules[cached_staging_candidates.__module__].__file__)]
    pin_manifest(OUT/"manifest.json", {"folds":[sorted(frame.loc[frame.fold==f,"subject_id"].unique()) for f in range(5)],
                                      "seed":SEED,"inputs":SIGNALS,"parameters":CANDIDATES,"removed":sorted(DROPPED),
                                      "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                                      "baseline_sha256":hashlib.sha256(BASE.read_bytes()).hexdigest()})
    raw = RawStagingFeatureCache(RAW,CACHE/"quality1s",OUT/"waveform_cache",permitted_subjects=set(frame.subject_id))
    profile_path = OUT/"marker_patient_effects.parquet"
    if not profile_path.exists():
        entries = []
        for i,(_,night) in enumerate(frame.sort_values(KEYS).groupby("subject_id",sort=False),start=1):
            entries.extend(per_night_profile(raw,night))
            if i%40==0:
                log(f"marker inputs checked for {i}/320 nights")
        save_frame(profile_path,pd.DataFrame(entries))
    profile = pd.read_parquet(profile_path)
    start = active_clock()
    budget_file = OUT/"active_budget.json"
    used = json.loads(budget_file.read_text())["seconds"] if budget_file.exists() else 0.0
    try:
        for fold,split in enumerate(splits):
            destination = OUT/f"fold-{fold}"
            destination.mkdir(exist_ok=True)
            recipe_file = destination/"recipe.json"
            if (OUT/"PAUSE").exists() or used+active_clock()-start >= 86400:
                log("Paused at a fold boundary")
                return 75
            train = frame.iloc[split.train_idx]
            test = frame.iloc[split.test_idx]
            if not recipe_file.exists():
                recipe, inner, marker = inner_search(frame,split,profile,fold)
                pd.DataFrame(marker).to_csv(destination/"marker_screen.csv",index=False)
                atomic_json(recipe_file,{"substitutions":recipe,"inner":inner})
            recipe = json.loads(recipe_file.read_text())["substitutions"]
            for variant,dropped in [("joint",set()),("reduced",DROPPED)]:
                file = destination/f"{variant}.parquet"
                if file.exists():
                    continue
                if (OUT/"PAUSE").exists() or used+active_clock()-start >= 86400:
                    log("Paused at a fit boundary")
                    return 75
                if recipe or dropped:
                    log(f"fold {fold + 1}: fitting {variant} using {recipe}")
                    probs, runtime = fit_predict(train,test,recipe,dropped=dropped)
                else:
                    aligned = test[[*KEYS,LABEL,"fold"]].merge(base,on=[*KEYS,LABEL,"fold"],validate="one_to_one")
                    probs, runtime = aligned[PROBABILITIES].to_numpy(), {"inputs":18,"reused":True}
                    log(f"fold {fold + 1}: joint selected the archived baseline")
                atomic_json(destination/f"{variant}_runtime.json",runtime)
                write_predictions(file,test,probs,fold)
        report(base)
    finally:
        atomic_json(budget_file,{"seconds":used+active_clock()-start})
    return 0


if __name__ == "__main__":
    with local_training_slot():
        raise SystemExit(execute())

