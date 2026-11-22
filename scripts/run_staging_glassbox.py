"""Additive and pairwise glass-box staging models for the Boston development folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")

from interpret.glassbox import ExplainableBoostingClassifier
import interpret
from joblib import dump
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from build_metric_intervals import (
    cv_fold_draws,
    cv_scores,
    interval,
    patient_confusions,
)
from run_staging_feature_tuning import (
    BASE,
    FEATURES,
    TRACKS,
    error_structure,
    save_frame,
)
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    SEED,
    STAGES,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
    pin_manifest,
    scores,
    write_predictions,
)
from physionet_sleep.experiments.modern_tabular import _dense_cached_night
from physionet_sleep.experiments.staging_recipe_features import (
    ANCHORS,
    CachedRecipeNight,
)
from physionet_sleep.experiments.waveform_clock import active_clock

OUT = EXPLORATION / "cinc-error-interactions"
KEY_FOLD = [*KEYS, LABEL, "fold"]
SPANS = {
    "center": (0,),
    "prior_min": (-2, -1),
    "next_min": (1, 2),
    "distant": (-8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8),
}
SIGNALS = [x.split(".")[-1] for x in TRACKS + FEATURES]
NAMES = [f"{s}@{span}" for span in SPANS for s in SIGNALS] + [
    "low_chin_persistence_10min"
]
INTERACTIONS = (
    ("spindle_bursts@center", "rel_beta@center"),
    ("frontal_slow_wave_fraction@center", "c_rel_theta@center"),
    ("delta_power@center", "aperiodic_exponent@center"),
    ("o_rel_alpha@center", "c_rel_theta@center"),
    ("emg_floor_db@center", "eog_rem_vs_noise_db@center"),
    ("low_chin_persistence_10min", "eog_rem_vs_noise_db@center"),
    ("frontal_slow_wave_fraction@center", "frontal_slow_wave_fraction@distant"),
)
PAIRS = tuple(tuple(NAMES.index(s) for s in pair) for pair in INTERACTIONS)
SETTINGS = {
    "max_bins": 128,
    "max_interaction_bins": 16,
    "outer_bags": 1,
    "inner_bags": 0,
    "learning_rate": 0.04,
    "max_rounds": 600,
    "early_stopping_rounds": 50,
    "n_jobs": 1,
    "random_state": SEED,
    "greedy_ratio": 5.0,
    "smoothing_rounds": 30,
    "interaction_smoothing_rounds": 15,
}
CLASSES = np.arange(5)


def night_panel(subject: str, epochs: np.ndarray) -> np.ndarray:
    dense = _dense_cached_night(CACHE, subject, TRACKS, FEATURES)
    x = np.pad(dense, ((0, -len(dense) % 30), (0, 0)), constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        epoch_view = np.nanmean(x.reshape(-1, 30, len(SIGNALS)), axis=1)
    if epochs.max() >= len(epoch_view):
        raise ValueError(f"{subject}: keyed epoch beyond the recorded night")
    future_padded = np.pad(epoch_view, ((8, 8), (0, 0)), constant_values=np.nan)
    parts = []
    for offsets in SPANS.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            parts.append(
                np.nanmean(
                    future_padded[epochs[:, None] + 8 + np.array(offsets)], axis=1
                )
            )
    atonia = CachedRecipeNight(CACHE, subject).compute(ANCHORS["atonia"])
    return np.column_stack([*parts, atonia[epochs * 30]]).astype(np.float32)


def named_panel(frame: pd.DataFrame) -> np.ndarray:
    values = np.empty((len(frame), len(NAMES)), dtype=np.float32)
    for subject, positions in frame.groupby("subject_id", sort=False).indices.items():
        values[positions] = night_panel(
            subject, frame.iloc[positions].epoch_index.to_numpy()
        )
    return values


def ebm(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    x: np.ndarray,
    *,
    pairs: bool,
    rounds: int | None = None,
):
    all_rows = pd.concat([fit, validation])
    if set(fit.subject_id) & set(validation.subject_id):
        raise ValueError("Training and validation patients overlap")
    y = all_rows[LABEL].to_numpy()
    counts = np.bincount(fit[LABEL], minlength=len(STAGES)).astype(np.float64)
    if not (counts > 0).all():
        raise ValueError("No fitting examples of at least one stage")
    weights = len(fit) / (len(STAGES) * counts[y])
    bags = np.ones((len(y), 1), dtype=np.int8)
    bags[len(fit) :, 0] = -1
    params = {**SETTINGS, "max_rounds": rounds or SETTINGS["max_rounds"]}
    if rounds is not None:
        params.update(
            smoothing_rounds=min(rounds, SETTINGS["smoothing_rounds"]),
            interaction_smoothing_rounds=min(
                rounds, SETTINGS["interaction_smoothing_rounds"]
            ),
        )
    model = ExplainableBoostingClassifier(
        feature_names=NAMES,
        feature_types=["continuous"] * len(NAMES),
        interactions=list(PAIRS) if pairs else 0,
        **params,
    )
    model.fit(x[all_rows.index.to_numpy()], y, sample_weight=weights, bags=bags)
    found = {tuple(idx) for idx in model.term_features_ if len(idx) > 1}
    if found != (set(PAIRS) if pairs else set()):
        raise ValueError(
            f"Fitted interaction terms differ from the declared pairs: {found}"
        )
    return model


def verify_locally(model, x: np.ndarray) -> None:
    term_scores = model.eval_terms(x)
    reconstructed = term_scores.sum(axis=1) + model.intercept_
    np.testing.assert_allclose(reconstructed, model.decision_function(x), atol=1e-9)
    np.testing.assert_allclose(
        softmax(reconstructed, axis=1), model.predict_proba(x), atol=1e-9
    )


def smoke_check(frame, splits):
    fit, val, _ = fold_frames(frame, splits[0])
    fit = fit[fit.subject_id.isin(sorted(fit.subject_id.unique())[:8])]
    val = val[val.subject_id.isin(sorted(val.subject_id.unique())[:3])]
    small = pd.concat([fit, val]).reset_index(drop=True)
    cells = named_panel(small)
    start = perf_counter()
    baseline = ebm(
        small.iloc[: len(fit)], small.iloc[len(fit) :], cells, pairs=False, rounds=5
    )
    paired = ebm(
        small.iloc[: len(fit)], small.iloc[len(fit) :], cells, pairs=True, rounds=5
    )
    for model in (baseline, paired):
        verify_locally(model, cells[-16:])
    atomic_json(
        OUT / "glassbox_smoke.json",
        {
            "passed": True,
            "held_out_patients_scored": 0,
            "seconds": perf_counter() - start,
            "terms": len(paired.term_features_),
        },
    )
    log(
        "Glass-box smoke verified fitted terms and exact reconstruction on inner patients"
    )


def case_terms(model, x, test, base_pred, p, fold):
    prediction = p.argmax(1)
    truth = test[LABEL].to_numpy()
    repaired = (base_pred != truth) & (prediction == truth)
    regressed = (base_pred == truth) & (prediction != truth)
    rival = np.where(regressed, prediction, base_pred)
    cards = []
    term_totals = []
    for outcome, mask in (("repair", repaired), ("regression", regressed)):
        for a, b in [(2, 1), (1, 2), (2, 3), (3, 2), (1, 0), (0, 1), (4, 1), (1, 4)]:
            selected = np.flatnonzero(mask & (truth == a) & (rival == b))
            if not len(selected):
                continue
            values = []
            margins = []
            for start in range(0, len(selected), 2048):
                batch = selected[start : start + 2048]
                terms = model.eval_terms(x[batch])
                values.append(np.sum(terms[:, :, a] - terms[:, :, b], axis=0))
                margins.append(
                    model.decision_function(x[batch])[:, a]
                    - model.decision_function(x[batch])[:, b]
                )
            for name, value in zip(
                model.term_names_, np.sum(values, axis=0) / len(selected)
            ):
                term_totals.append(
                    {
                        "fold": fold,
                        "outcome": outcome,
                        "true_stage": STAGES[a],
                        "rival": STAGES[b],
                        "n": len(selected),
                        "term": name,
                        "mean_log_odds_contribution": float(value),
                    }
                )
            # Representative case near the median margin of its directed cohort.
            median = np.median(np.concatenate(margins))
            pos = selected[np.argmin(np.abs(np.concatenate(margins) - median))]
            terms = model.eval_terms(x[pos : pos + 1])[0]
            local = terms[:, a] - terms[:, b]
            intercept = float(model.intercept_[a] - model.intercept_[b])
            raw_log_odds = float(
                model.decision_function(x[pos : pos + 1])[0, a]
                - model.decision_function(x[pos : pos + 1])[0, b]
            )
            np.testing.assert_allclose(intercept + local.sum(), raw_log_odds, atol=1e-9)
            order = np.argsort(-np.abs(local), kind="stable")[:6]
            cards.append(
                {
                    "fold": fold,
                    "sample": f"fold-{fold + 1}/{test.iloc[pos].epoch_index}",
                    "outcome": outcome,
                    "true_stage": STAGES[a],
                    "rival": STAGES[b],
                    "cohort_epochs": len(selected),
                    "raw_log_odds": raw_log_odds,
                    "intercept": intercept,
                    "omitted_terms": float(local.sum() - local[order].sum()),
                    "terms": [
                        {"name": model.term_names_[i], "contribution": float(local[i])}
                        for i in order
                    ],
                }
            )
    return cards, term_totals


def compare(aligned: pd.DataFrame, candidate: str, comparator: str) -> dict:
    y, a, b = (aligned[k].to_numpy() for k in (LABEL, candidate, comparator))
    groups = [np.flatnonzero(aligned.fold.eq(i)) for i in range(5)]
    x, z = scores(aligned, a, groups), scores(aligned, b, groups)
    fold_deltas = [
        float(
            f1_score(y[i], a[i], average="macro")
            - f1_score(y[i], b[i], average="macro")
        )
        for i in groups
    ]
    ca, patients = patient_confusions(aligned.subject_id.to_numpy(), y, a)
    cb, _ = patient_confusions(aligned.subject_id.to_numpy(), y, b)
    group = aligned.groupby("subject_id").fold.first().loc[patients].to_numpy()
    boot_groups = [np.flatnonzero(group == i) for i in range(5)]
    draws = cv_fold_draws(boot_groups, 2000, np.random.default_rng(SEED))
    bounds = interval(
        cv_scores(ca, boot_groups, draws)["macro_f1"]
        - cv_scores(cb, boot_groups, draws)["macro_f1"]
    )
    loss = {s: x[f"f1_{s}"] - z[f"f1_{s}"] for s in STAGES}
    return {
        "candidate": x,
        "comparator": z,
        "fold_deltas": fold_deltas,
        "delta": float(np.mean(fold_deltas)),
        "stage_deltas": loss,
        "paired_ci": bounds,
    }


def report(base):
    aligned = base[KEY_FOLD].copy()
    aligned["baseline"] = base["prediction"].to_numpy()
    probabilities = {"baseline": base[PROBABILITIES].to_numpy()}
    cards = []
    term_groups = []
    for variant in ("additive", "pairwise"):
        p = pd.concat(
            [
                pd.read_parquet(OUT / f"glassbox/fold-{f}/{variant}.parquet")
                for f in range(5)
            ]
        )
        p = base[KEY_FOLD].merge(p, on=KEY_FOLD, validate="one_to_one")
        if len(p) != len(base):
            raise ValueError("Missing original outer rows in the glass-box comparison")
        save_frame(OUT / f"glassbox_{variant}_predictions.parquet", p)
        probabilities[variant] = p[PROBABILITIES].to_numpy()
        aligned[variant] = probabilities[variant].argmax(1)
    for f in range(5):
        home = OUT / f"glassbox/fold-{f}"
        cards.extend(
            json.loads((home / "reason_cards.json").read_text(encoding="utf-8"))
        )
        term_groups.append(pd.read_csv(home / "pairwise_term_groups.csv"))
    atomic_json(OUT / "glassbox_reason_cards.json", {"cards": cards})
    pd.concat(term_groups, ignore_index=True).to_csv(
        OUT / "glassbox_pairwise_term_groups.csv", index=False
    )
    results = {
        f"{a}_vs_{b}": compare(aligned, a, b)
        for a, b in [
            ("additive", "baseline"),
            ("pairwise", "baseline"),
            ("pairwise", "additive"),
        ]
    }
    y = aligned[LABEL].to_numpy()
    metrics = {
        name: {
            "log_loss": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
            "brier": float(np.square(p - np.eye(5)[y]).sum(1).mean()),
        }
        for name, p in probabilities.items()
    }
    p = results["pairwise_vs_baseline"]
    p["passes"] = (
        p["delta"] >= 0.010
        and sum(d > 0 for d in p["fold_deltas"]) >= 4
        and min(p["stage_deltas"].values()) >= -0.010
    )
    atomic_json(
        OUT / "glassbox_summary.json",
        {
            "contrasts": results,
            "probabilities": metrics,
            "errors": {v: error_structure(aligned, v) for v in probabilities},
            "interactions": [list(p) for p in INTERACTIONS],
        },
    )
    for variant in ("additive", "pairwise"):
        z = results[f"{variant}_vs_baseline"]
        log(
            f"{variant}: {z['candidate']['macro_f1']:.4f}, {z['delta']:+.4f} versus compact HGB"
        )


def execute(smoke):
    if not (OUT / "FINDINGS.md").exists():
        raise FileNotFoundError("Declare the study before scoring")
    frame, splits = development_frame()
    if smoke:
        smoke_check(frame, splits)
        return 0
    if not (OUT / "glassbox_smoke.json").exists():
        raise FileNotFoundError("Run the training-only smoke before the full fit")
    base = pd.read_parquet(BASE)
    all_expected = pd.concat(
        [frame.iloc[s.test_idx].assign(fold=f)[KEY_FOLD] for f, s in enumerate(splits)]
    )
    if len(base) != len(frame) or len(
        all_expected.merge(base, on=KEY_FOLD, validate="one_to_one")
    ) != len(frame):
        raise ValueError("Original patient folds and baseline keys differ")
    src = Path(__file__)
    pin_manifest(
        OUT / "glassbox_manifest.json",
        {
            "settings": SETTINGS,
            "panel": NAMES,
            "library": interpret.__version__,
            "pairs": [list(p) for p in INTERACTIONS],
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "baseline_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        },
    )
    x = named_panel(frame)
    ledger = OUT / "glassbox_time.json"
    used = json.loads(ledger.read_text())["seconds"] if ledger.exists() else 0.0
    start = active_clock()
    try:
        for f, split in enumerate(splits):
            fit, val, test = fold_frames(frame, split)
            home = OUT / f"glassbox/fold-{f}"
            home.mkdir(parents=True, exist_ok=True)
            base_pred = (
                test[KEYS + [LABEL]]
                .assign(fold=f)
                .merge(
                    base[KEY_FOLD + ["prediction"]], on=KEY_FOLD, validate="one_to_one"
                )
                .prediction.to_numpy()
            )
            if len(base_pred) != len(test):
                raise ValueError("Archived baseline has incomplete evaluation keys")
            for name in ("additive", "pairwise"):
                path = home / f"{name}.parquet"
                if path.exists():
                    continue
                if (
                    OUT / "PAUSE_GLASSBOX"
                ).exists() or used + active_clock() - start > 86400:
                    return 75
                log(
                    f"Fitting glass-box fold {f + 1}/5 {name} on {fit.subject_id.nunique()} patients"
                )
                before = active_clock()
                model = ebm(fit, val, x, pairs=name == "pairwise")
                test_x = x[test.index.to_numpy()]
                verify_locally(model, test_x[:16])
                p = model.predict_proba(test_x)
                # Keep portable stage-shape functions and the trained estimator.
                np.savez_compressed(
                    home / f"{name}_term_scores.npz",
                    intercept=model.intercept_,
                    **{f"term_{i}": v for i, v in enumerate(model.term_scores_)},
                )
                pending = home / f"{name}.part"
                dump(model, pending, compress=3)
                pending.replace(home / f"{name}.joblib")
                atomic_json(
                    home / f"{name}_runtime.json",
                    {
                        "seconds": active_clock() - before,
                        "train_patients": fit.subject_id.nunique(),
                        "validation_patients": val.subject_id.nunique(),
                        "stage_rounds": model.best_iteration_.tolist(),
                        "terms": model.term_names_,
                    },
                )
                if name == "pairwise":
                    cards, term_totals = case_terms(
                        model, test_x, test, base_pred, p, f
                    )
                    atomic_json(home / "reason_cards.json", cards)
                    pd.DataFrame(term_totals).to_csv(
                        home / "pairwise_term_groups.csv", index=False
                    )
                write_predictions(path, test, p, f)
                log(
                    f"glass-box fold {f + 1}/5 {name}: {active_clock() - before:.1f}s, "
                    f"macro-F1 {f1_score(test[LABEL], p.argmax(1), average='macro'):.4f}"
                )
        report(base)
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - start})
    return 0


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--smoke", action="store_true")
    with local_training_slot(), threadpool_limits(limits=4):
        raise SystemExit(execute(cli.parse_args().smoke))
