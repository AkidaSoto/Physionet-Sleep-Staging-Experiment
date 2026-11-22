
"""Reconstruct one development-fold tree pair and audit its class weights and interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from run_staging_feature_tuning import BASE, FEATURES, SIGNALS, TRACKS, profile_masks, save_frame
from run_staging_recipe_search import BASE_CONTEXT, window_columns, window_matrix
from run_waveform_development import (
    CACHE, EXPLORATION, KEYS, LABEL, PROBABILITIES, SEED, STAGES, atomic_json,
    development_frame, local_training_slot, log, pin_manifest, write_predictions,
)
from physionet_sleep.analysis.epoch_windows import WindowEncoder
from physionet_sleep.experiments.modern_tabular import _dense_cached_night
from physionet_sleep.experiments.tabular_runner import build_model

OUT = EXPLORATION / "cinc-error-interactions" / "tree-refit"
SEARCH = EXPLORATION / "cinc-recipe-search"
FOLD = 1
LONG_RADIUS = 8
KEY = [*KEYS, LABEL, "fold"]
CLASSES = np.arange(len(STAGES))
DONOR_REPEATS = 3
MIN_DONOR_GAP = 2 * LONG_RADIUS + 1
STRATUM_CAP = 4
GROUPS = {
    "slow_wave_delta": ("delta_power", "frontal_slow_wave_fraction"),
    "alpha_theta": ("c_rel_alpha", "o_rel_alpha", "occipital_rel_alpha", "c_rel_theta", "rel_theta"),
    "beta_spindle": ("c_rel_beta", "rel_beta", "spindle_bursts"),
    "eye": ("eog_high", "eye_movement_activity", "eye_movement_baseline", "eog_rem_vs_noise_db", "eog_rem_vs_sem_db"),
    "chin": ("emg_floor_db",),
    "broad_eeg": ("c_log_power", "aperiodic_exponent"),
}
PAIRS = [(2, 1), (1, 2), (2, 3), (3, 2)]
INTERACTIONS = (
    ("near:beta_spindle", "near:slow_wave_delta"),
    ("near:beta_spindle", "remote:slow_wave_delta"),
    ("near:alpha_theta", "remote:alpha_theta"),
)


def align_archive(test: pd.DataFrame, path: Path) -> np.ndarray:
    saved = pd.read_parquet(path, columns=KEY + PROBABILITIES, filters=[("fold", "==", FOLD)])
    aligned = test[KEY].merge(saved, on=KEY, how="left", validate="one_to_one", indicator=True)
    if len(aligned) != len(saved) or not aligned["_merge"].eq("both").all():
        raise ValueError(f"Saved predictions do not cover the exact fold keys: {path}")
    return aligned[PROBABILITIES].to_numpy(np.float64)


def read_study() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    frame, splits = development_frame()
    outer = splits[FOLD]
    train = frame.iloc[outer.train_idx].reset_index(drop=True)
    test = frame.iloc[outer.test_idx].reset_index(drop=True).assign(fold=FOLD)
    if set(train.subject_id) & set(test.subject_id):
        raise ValueError("Training and evaluation patients overlap")
    recipe = json.loads((SEARCH / f"fold-{FOLD}" / "selection.json").read_text(encoding="utf-8"))["recipe"]
    if recipe != {"context": LONG_RADIUS, "features": [], "reduce": False}:
        raise ValueError("This diagnostic requires a context-only change on the nominated fold")
    archived = {"baseline": align_archive(test, BASE), "searched": align_archive(test, SEARCH / f"fold-{FOLD}/searched.parquet")}
    return train, test, archived


def case_sample(test: pd.DataFrame, baseline: np.ndarray, searched: np.ndarray) -> pd.DataFrame:
    """Match errors between recipes on direction, patient, night quarter and reference position."""
    case = test[KEY].copy()
    y = case[LABEL].to_numpy()
    old, new = baseline.argmax(1), searched.argmax(1)
    case["cohort"] = np.select(
        [(old != y) & (new == y), (old == y) & (new != y), (old != y) & (new == old)],
        ["repair", "regression", "persistent"],
        default="exclude",
    )
    case["rival"] = np.where(case.cohort.eq("regression"), new, old)
    case["quarter"] = -1
    case["position"] = "unassigned"
    for _, night in case.groupby("subject_id", sort=False):
        ordered = night.sort_values("epoch_index")
        quarter, interior, boundary = profile_masks(ordered)
        case.loc[ordered.index, "quarter"] = quarter
        case.loc[ordered.index, "position"] = np.select(
            [interior, boundary >= 0], ["interior", "boundary"], default="other"
        )

    # Same donors for both HGBs and every perturbation. Candidate donors are independent
    # of recorded stage. Capped, matched samples therefore describe this diagnostic cohort.
    rng = np.random.default_rng(SEED)
    pool = {
        key: g.index.to_numpy()
        for key, g in case.groupby(["subject_id", "quarter"], sort=True)
    }
    selected = []
    epochs = case.epoch_index.to_numpy()
    pairs = pd.MultiIndex.from_arrays([case[LABEL], case.rival]).isin(PAIRS)
    for key, group in case[pairs & case.cohort.ne("exclude")].groupby(
        ["subject_id", "quarter", "position", LABEL, "rival"], sort=True
    ):
        donor_pool = pool[key[:2]]
        candidates = {}
        for cohort, cohort_rows in group.groupby("cohort", sort=True):
            candidates[cohort] = [
                (i, rng.choice(donors, size=DONOR_REPEATS, replace=False))
                for i in rng.permutation(cohort_rows.index)
                if len(donors := donor_pool[np.abs(epochs[donor_pool] - epochs[i]) >= MIN_DONOR_GAP]) >= DONOR_REPEATS
            ]
        n_match = min(STRATUM_CAP, len(candidates.get("repair", [])), len(candidates.get("regression", [])))
        if n_match == 0:
            continue
        for cohort in ("repair", "regression", "persistent"):
            selected.extend(candidates.get(cohort, [])[:n_match])
    if not selected:
        raise ValueError("No matched repairs and regressions on the nominated fold")
    positions = [i for i, _ in selected]
    out = case.loc[positions].reset_index(names="row")
    donor_rows = np.stack([rows for _, rows in selected])
    for draw in range(DONOR_REPEATS):
        out[f"donor_row_{draw}"] = donor_rows[:, draw]
        out[f"donor_epoch_{draw}"] = epochs[donor_rows[:, draw]]
    return out


def fit_archive_model(
    name: str, radius: int, train: pd.DataFrame, class_weight: dict[int, float] | None
):
    path = OUT / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    name_columns = [signal.split(".")[-1] for signal in SIGNALS] if name == "baseline" else list(SIGNALS)
    encoder = WindowEncoder(name_columns, context=min(radius, BASE_CONTEXT))
    columns = window_columns(encoder, radius)
    subjects = sorted(train.subject_id.unique())
    start = perf_counter()
    dense = {sid: _dense_cached_night(CACHE, sid, TRACKS, FEATURES) for sid in subjects}
    encoder.fit(dense, subjects)
    matrix = window_matrix(encoder, radius, dense, train)
    del dense
    prepared = perf_counter()
    estimator = build_model(model_family="hgb", task="classification", random_state=SEED, class_weight=class_weight)
    estimator.fit(pd.DataFrame(matrix, columns=columns), train[LABEL].to_numpy())
    del matrix
    state = {"pipeline": estimator, "encoder": encoder, "columns": columns,
             "prepare_s": prepared - start, "fit_s": perf_counter() - prepared}
    pending = path.with_suffix(".part")
    joblib.dump(state, pending, compress=3)
    pending.replace(path)
    return state


def evaluate_model(state: dict, test: pd.DataFrame, cases: pd.DataFrame, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the full fold with per-patient windows and retain matched rows for interventions."""
    model = state["pipeline"]
    encoder = state["encoder"]
    columns = state["columns"]
    target_rows = cases.row.to_numpy()
    donor_rows = cases[[f"donor_row_{i}" for i in range(DONOR_REPEATS)]].to_numpy()
    p = np.empty((len(test), len(STAGES)))
    original = np.empty((len(cases), len(columns)), dtype=np.float32)
    donors = np.empty((DONOR_REPEATS, len(cases), len(columns)), dtype=np.float32)
    for subject, indices in test.groupby("subject_id", sort=False).indices.items():
        dense = _dense_cached_night(CACHE, subject, TRACKS, FEATURES)
        x = window_matrix(encoder, radius, {subject: dense}, test.iloc[indices])
        p[indices] = model.predict_proba(pd.DataFrame(x, columns=columns))
        case_inds = np.flatnonzero(cases.subject_id.eq(subject))
        # GroupBy.indices preserves the order of the original positions.
        original[case_inds] = x[np.searchsorted(indices, target_rows[case_inds])]
        for i in range(DONOR_REPEATS):
            donors[i, case_inds] = x[np.searchsorted(indices, donor_rows[case_inds, i])]
    return p, original, donors


def describe_columns(columns: list[str]) -> pd.DataFrame:
    family = {signal: group for group, signals in GROUPS.items() for signal in signals}
    groups = []
    for column in columns:
        signal, location = column.rsplit("@", 1)
        signal = signal.split(".")[-1]
        groups.append((signal, family[signal], "remote" if location.endswith("epoch_mean") else "near", location))
    return pd.DataFrame(groups, columns=["signal", "family", "location", "offset"])


def split_gain(model, mapping: pd.DataFrame, name: str) -> pd.DataFrame:
    """Native HGB split gains as training diagnostics, with per-output tree class."""
    values = []
    fitted = model.named_steps["model"]
    for boosting in fitted._predictors:
        for tree_class, tree in zip(model.classes_, boosting, strict=True):
            nodes = tree.nodes
            if not {"is_leaf", "feature_idx", "gain"} <= set(nodes.dtype.names):
                raise TypeError("Native HGB node fields differ from the expected sklearn version")
            inner = nodes[nodes["is_leaf"] == 0]
            info = mapping.iloc[inner["feature_idx"].astype(int)].reset_index(drop=True)
            values.append(info.assign(tree_class=STAGES[tree_class], gain=inner["gain"], splits=1))
    summary = pd.concat(values).groupby(
        ["tree_class", "location", "family", "signal"], as_index=False
    )[["gain", "splits"]].sum()
    total_gain = summary.groupby("tree_class").gain.transform("sum")
    return summary.assign(model=name, class_gain_share=summary.gain.div(total_gain))


def contrastive_interventions(
    state: dict, name: str, x: np.ndarray, donors: np.ndarray, cases: pd.DataFrame, mapping: pd.DataFrame,
) -> pd.DataFrame:
    model = state["pipeline"]
    columns = state["columns"]
    truth, rival = cases[LABEL].to_numpy(), cases.rival.to_numpy()
    row = np.arange(len(cases))
    blocks = {f"{location}:{family}": group.index.to_numpy()
              for (location, family), group in mapping.groupby(["location", "family"], sort=True)}

    def log_odds(array: np.ndarray) -> np.ndarray:
        values = model.decision_function(pd.DataFrame(array, columns=columns))
        return values[row, truth] - values[row, rival]

    full = log_odds(x)
    original = cases[KEY + ["cohort", "rival", "position", "quarter"]]
    records = []
    for draw in range(DONOR_REPEATS):
        def intervene(indices: np.ndarray) -> np.ndarray:
            modified = x.copy()
            modified[:, indices] = donors[draw][:, indices]
            return log_odds(modified)

        swaps = {label: intervene(indices) for label, indices in blocks.items()}
        for term, shifted in swaps.items():
            records.append(original.assign(model=name, draw=draw, kind="group", term=term,
                                           unperturbed=full, effect=full - shifted))
        for first, second in INTERACTIONS:
            if first not in blocks or second not in blocks:
                continue
            both = intervene(np.r_[blocks[first], blocks[second]])
            records.append(original.assign(model=name, draw=draw, kind="mixed_difference",
                                           term=f"{first} × {second}", unperturbed=full,
                                           effect=full - swaps[first] - swaps[second] + both))
    return pd.concat(records, ignore_index=True)


def fold_metrics(test: pd.DataFrame, probabilities: np.ndarray) -> dict:
    y = test[LABEL].to_numpy()
    pred = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {"macro_f1": float(f1.mean()), "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
            "brier": float(np.square(probabilities - np.eye(len(STAGES))[y]).sum(axis=1).mean()),
            "stages": {name: {"precision": float(p), "recall": float(r), "f1": float(q), "support": int(n)}
                       for name, p, r, q, n in zip(STAGES, precision, recall, f1, support, strict=True)},
            "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist()}


def run(preflight: bool) -> int:
    train, test, archived = read_study()
    cases = case_sample(test, archived["baseline"], archived["searched"])
    sample_summary = cases.groupby([LABEL, "rival", "cohort"]).size().reset_index(name="epochs")
    log(f"Fold {FOLD}: {train.subject_id.nunique()} training patients, "
        f"{test.subject_id.nunique()} test patients, {len(cases)} matched cases")
    if preflight:
        print(sample_summary.to_string(index=False))
        return 0
    if not (OUT.parent / "FINDINGS.md").exists():
        raise FileNotFoundError("Declare experiment #55 before fitting")
    OUT.mkdir(parents=True, exist_ok=True)
    pin_manifest(OUT / "manifest.json", {
        "fold": FOLD, "train_subjects": sorted(train.subject_id.unique()), "test_subjects": sorted(test.subject_id.unique()),
        "rows": len(test), "sklearn": sklearn_version, "seed": SEED, "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    save_frame(OUT / "matched_cases.parquet", cases)

    labels = train[LABEL].to_numpy()
    counts = np.bincount(labels, minlength=len(STAGES)).astype(float)
    frequencies = counts / counts.sum()
    balanced = counts.sum() / (len(STAGES) * counts)
    tempered = frequencies ** -0.5
    tempered /= np.dot(frequencies, tempered)
    variants = (
        ("baseline", BASE_CONTEXT, None),
        ("searched", LONG_RADIUS, None),
        ("tempered", LONG_RADIUS, dict(enumerate(tempered.tolist()))),
    )
    audit = {}
    for name, radius, weights in variants:
        if (OUT / "PAUSE").exists():
            return 75
        checkpoint = OUT / f"{name}_result.json"
        if checkpoint.exists():
            audit[name] = json.loads(checkpoint.read_text(encoding="utf-8"))
            continue
        log(f"Reconstructing diagnostic fold {FOLD} {name} HGB")
        state = fit_archive_model(name, radius, train, weights)
        p, x, donor = evaluate_model(state, test, cases, radius)
        info = fold_metrics(test, p)
        if name in archived:
            maximum = float(np.abs(p - archived[name]).max())
            unchanged = int(np.sum(p.argmax(1) != archived[name].argmax(1)))
            info["archive_max_abs_probability_difference"] = maximum
            info["archive_changed_decisions"] = unchanged
            if unchanged or not np.allclose(p, archived[name], rtol=1e-8, atol=1e-9):
                raise ValueError(f"{name} archive parity failed: maximum difference {maximum:.3g}, changed decisions {unchanged}")
        write_predictions(OUT / f"{name}_predictions.parquet", test, p, FOLD)
        column_info = describe_columns(state["columns"])
        save_frame(OUT / f"{name}_split_gain.parquet", split_gain(state["pipeline"], column_info, name))
        perturbed = contrastive_interventions(state, name, x, donor, cases, column_info)
        save_frame(OUT / f"{name}_interventions.parquet", perturbed)
        info["iterations"] = int(state["pipeline"].named_steps["model"].n_iter_)
        info["fit_seconds"] = state["fit_s"]
        info["prepare_seconds"] = state["prepare_s"]
        atomic_json(checkpoint, info)
        audit[name] = info
        log(f"{name}: macro-F1 {info['macro_f1']:.4f}, fitted {info['fit_seconds']:.1f}s")
        del perturbed, p, x, donor, state

    data = pd.concat([pd.read_parquet(OUT / f"{name}_interventions.parquet") for name, _, _ in variants])
    position = ["model", "cohort", LABEL, "rival", "kind", "term"]
    cell = data.groupby(position + KEYS, as_index=False)[["effect", "unperturbed"]].mean()
    cell["absolute"] = cell.effect.abs()
    cell["positive"] = cell.effect > 0
    effects = cell.groupby(position, as_index=False).agg(
        patients=("subject_id", "nunique"), epochs=("subject_id", "size"), mean=("effect", "mean"),
        median=("effect", "median"), mean_absolute=("absolute", "mean"), positive_share=("positive", "mean"),
        mean_unperturbed=("unperturbed", "mean"),
    )
    save_frame(OUT / "interaction_summary.parquet", effects)
    selected = {}
    for a, b in PAIRS:
        for cohort in ("repair", "regression", "persistent"):
            rows = effects[effects.model.eq("searched") & effects.kind.eq("mixed_difference")
                           & effects[LABEL].eq(a) & effects.rival.eq(b) & effects.cohort.eq(cohort)]
            if not rows.empty:
                selected[f"{STAGES[a]}→{STAGES[b]} {cohort}"] = rows.sort_values(
                    "mean_absolute", ascending=False
                ).iloc[0][["term", "mean", "median", "mean_absolute", "epochs", "patients"]].to_dict()
    atomic_json(OUT / "tree_summary.json", {
        "fold_zero_based": FOLD, "training_patients": train.subject_id.nunique(), "evaluation_patients": test.subject_id.nunique(),
        "evaluation_epochs": len(test), "training_stage_counts": dict(zip(STAGES, counts.astype(int).tolist(), strict=True)),
        "balanced_class_weights": dict(zip(STAGES, balanced.tolist(), strict=True)),
        "tempered_class_weights": dict(zip(STAGES, tempered.tolist(), strict=True)),
        "matched_cases": sample_summary.to_dict("records"), "models": audit, "largest_pair_effects": selected,
    })
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Inspect archived cases with no fitting")
    args = parser.parse_args()
    if args.preflight:
        raise SystemExit(run(preflight=True))
    with local_training_slot(), threadpool_limits(limits=4):
        raise SystemExit(run(preflight=False))

