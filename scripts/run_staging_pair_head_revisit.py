"""Refit a compact N1/N2 decision on a frozen Boston development encoder."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_staging_complementary_model as expanded
import run_staging_gated_temporal as gated
from build_metric_intervals import (
    cv_fold_draws,
    cv_scores,
    interval,
    patient_confusions,
)
from run_staging_feature_tuning import BASE, save_frame
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    SEED,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
)
from physionet_sleep.analysis.epoch_windows import spindle_burst_counts
from physionet_sleep.experiments.complementary_n2_events import _recent_support


OUT = EXPLORATION / "cinc-stage-pair-revisit"
KEY_FOLD = gated.KEY_FOLD
N1, N2 = 1, 2
BASE_FEATURES = [
    "local_margin",
    "past_margin",
    "future_margin",
    "n2_recent_event_support",
    "slow_wave_75_max_clean",
    "slow_wave_60_to_75_max_clean",
]
K_FEATURES = ["log_k_shape", "k_by_strong_waves"]
INTERRUPTION_FEATURES = ["post_interrupt_recent_event_support", "interruption_strength"]
SOURCE_HISTORY = ["recent_spindle_support", "recent_k_support"]
OFFSETS = (-0.20, 0.0, 0.20, 0.40)
PENALTY = 0.02


def pair_candidates(p: np.ndarray) -> np.ndarray:
    best = np.argsort(-p, axis=1, kind="stable")[:, :2]
    return (np.sort(best, axis=1) == [N1, N2]).all(1)


def old_margin(p: np.ndarray) -> np.ndarray:
    return np.log(p[:, N1].clip(1e-12)) - np.log(p[:, N2].clip(1e-12))


def pair_probabilities(p: np.ndarray, corrected_margin: np.ndarray) -> np.ndarray:
    mask = pair_candidates(p)
    result = p.copy()
    mass = p[mask, N1] + p[mask, N2]
    result[mask, N1] = mass * expit(corrected_margin[mask])
    result[mask, N2] = mass - result[mask, N1]
    return result


def fit_head(
    train: pd.DataFrame, measure: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, dict]:
    p = train[PROBABILITIES].to_numpy()
    keep = pair_candidates(p) & train[LABEL].isin([N1, N2]).to_numpy()
    y = train.loc[keep, LABEL].eq(N1).to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError(
            "Head fitting requires both pair labels among eligible inner cases"
        )
    scaler = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler())
    x = scaler.fit_transform(train.loc[keep, features])
    baseline = old_margin(p[keep])

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        z = baseline + weights[0] + x @ weights[1:]
        error = expit(z) - y
        loss = (np.logaddexp(0, z) - y * z).mean() + (PENALTY / 2) * np.dot(
            weights, weights
        )
        gradient = np.r_[error.mean(), x.T @ error / len(y)] + PENALTY * weights
        return float(loss), gradient

    result = minimize(objective, np.zeros(x.shape[1] + 1), method="L-BFGS-B", jac=True)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Residual head failed to converge: {result.message}")
    z = old_margin(measure[PROBABILITIES].to_numpy())
    corrected = z + result.x[0] + scaler.transform(measure[features]) @ result.x[1:]
    std = scaler.named_steps["standardscaler"]
    model = {
        "features": features,
        "weights": result.x.tolist(),
        "means": std.mean_.tolist(),
        "scales": std.scale_.tolist(),
        "imputation": scaler.named_steps["simpleimputer"].statistics_.tolist(),
        "fit_epochs": int(keep.sum()),
    }
    return corrected, model


def select_offset(
    inner: pd.DataFrame, features: list[str]
) -> tuple[float, list[dict], bool]:
    p = inner[PROBABILITIES].to_numpy()
    y = inner[LABEL].to_numpy()
    inner_oof = np.empty(len(inner))
    for fit, held in GroupKFold(n_splits=4).split(inner, groups=inner.subject_id):
        inner_oof[held], _ = fit_head(inner.iloc[fit], inner.iloc[held], features)
    reference = f1_score(
        y, p.argmax(1), labels=np.arange(5), average=None, zero_division=0
    )
    candidates = []
    for offset in OFFSETS:
        prediction = pair_probabilities(p, inner_oof + offset).argmax(1)
        stages = f1_score(
            y, prediction, labels=np.arange(5), average=None, zero_division=0
        )
        candidates.append(
            {
                "offset": offset,
                "n1_f1": float(stages[N1]),
                "n2_delta": float(stages[N2] - reference[N2]),
                "macro_delta": float((stages - reference).mean()),
            }
        )
    valid = [
        c for c in candidates if c["n2_delta"] >= -0.005 and c["macro_delta"] >= -0.002
    ]
    chosen = (
        max(valid, key=lambda c: (c["n1_f1"], -abs(c["offset"]), -c["offset"]))[
            "offset"
        ]
        if valid
        else 0.0
    )
    return chosen, candidates, bool(valid)


def marker_panel(frame: pd.DataFrame, split_history: bool) -> pd.DataFrame:
    records = []
    for subject, values in expanded.load_markers(set(frame.subject_id)).items():
        raw = pd.DataFrame(values, columns=expanded.FEATURES)
        raw["epoch_index"] = np.arange(len(raw))
        raw["subject_id"] = subject
        if split_history:
            tracks = pd.read_parquet(
                CACHE / "tracks1s" / f"{subject}.parquet",
                columns=["c_rel_alpha", "c_rel_beta", "c_rel_sigma"],
            )
            spindle = spindle_burst_counts(tracks)[: len(raw)]
            events = raw["kc_shape_events"].to_numpy()
            raw[SOURCE_HISTORY[0]] = _recent_support(np.zeros_like(events), spindle)
            raw[SOURCE_HISTORY[1]] = _recent_support(events, np.zeros_like(spindle))
            original = raw["n2_recent_event_support"].to_numpy()
            pooled = np.fmax(
                raw[SOURCE_HISTORY[0]].to_numpy(), raw[SOURCE_HISTORY[1]].to_numpy()
            )
            known = np.isfinite(original)
            np.testing.assert_allclose(
                pooled[known], original[known], atol=1e-7, rtol=0
            )
        records.append(raw)
    data = pd.concat(records, ignore_index=True)
    data["log_k_shape"] = np.log1p(data["kc_shape_events"])
    data["k_by_strong_waves"] = data["log_k_shape"] * data["slow_wave_75_max_clean"]
    return data.drop(columns="kc_shape_events")


def build_inputs(
    df: pd.DataFrame,
    fold: int,
    probabilities: np.ndarray,
    terms: np.ndarray,
    markers: pd.DataFrame,
) -> pd.DataFrame:
    output = df[[*KEYS, LABEL]].assign(fold=fold)
    output[PROBABILITIES] = probabilities
    for name in ("local", "past", "future"):
        output[f"{name}_margin"] = (
            terms[:, gated.TERM_COLUMNS.index(f"{name}_N1")]
            - terms[:, gated.TERM_COLUMNS.index(f"{name}_N2")]
        )
    joined = output.merge(
        markers, on=KEYS, how="left", validate="one_to_one", indicator=True
    )
    if len(joined) != len(output) or not joined["_merge"].eq("both").all():
        raise ValueError("A scored epoch lacks an aligned marker row")
    return joined.drop(columns="_merge")


def validate_histories(
    frame: pd.DataFrame, splits: list, markers: pd.DataFrame
) -> pd.DataFrame:
    joined = frame[[*KEYS, LABEL]].merge(
        markers[[*KEYS, *SOURCE_HISTORY]], on=KEYS, how="left", validate="one_to_one"
    )
    rows = []
    for fold, split in enumerate(splits):
        train, _, _ = fold_frames(frame, split)
        values = joined.iloc[train.index].groupby(LABEL)[SOURCE_HISTORY].mean()
        contrast = values.loc[N2] - values.loc[N1]
        rows.append({"fold": fold, **contrast.to_dict()})
    directions = pd.DataFrame(rows)
    if not (directions[SOURCE_HISTORY] > 0).all(axis=None):
        raise ValueError(
            "At least one source history lacks the declared direction on inner-fit patients"
        )
    return directions


def replay(
    frame: pd.DataFrame,
    splits: list,
    base: pd.DataFrame,
    variants: dict,
    markers: pd.DataFrame,
) -> dict:
    panels = expanded.load_markers(set(frame.subject_id))
    predictions = {name: [] for name in variants}
    history_cases = []
    settings = []
    for f, split in enumerate(splits):
        train, inner, test = fold_frames(frame, split)
        training_ids = set(train.subject_id)
        recordings = gated.sequence.prepare_recordings(frame, training_ids)
        scaling = expanded.add_markers(recordings, panels, training_ids)
        path = expanded.MODEL / f"fold-{f}"
        archived = json.loads((path / "runtime.json").read_text())["marker_scaling"]
        if scaling != archived:
            raise ValueError(f"Fold {f} scaling differs from the frozen encoder")
        model = expanded.expanded_model()
        state = torch.load(
            path / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state["best_weights"])
        inner_p, inner_terms = gated.score_components(model, recordings, inner)
        outer_p, outer_terms = gated.score_components(model, recordings, test)
        prior = gated.align(
            test[[*KEYS, LABEL]].assign(fold=f),
            pd.read_parquet(path / "predictions.parquet"),
        )
        np.testing.assert_allclose(prior[PROBABILITIES], outer_p, rtol=0, atol=2e-7)
        cal = build_inputs(inner, f, inner_p, inner_terms, markers)
        held = build_inputs(test, f, outer_p, outer_terms, markers)
        if "interruption_head" in variants:
            resets = (
                held["post_interrupt_recent_event_support"].notna()
                & held["n2_recent_event_support"].notna()
                & (
                    held["post_interrupt_recent_event_support"]
                    < held["n2_recent_event_support"] - 1e-6
                )
            )
            history_cases.append(held[KEY_FOLD].assign(history_case=resets))
        elif "continuation_head" in variants:
            spindle, k_shape = (held[column] for column in SOURCE_HISTORY)
            different = (
                spindle.notna() & k_shape.notna() & ((spindle - k_shape).abs() > 1e-6)
            )
            history_cases.append(held[KEY_FOLD].assign(history_case=different))
        for name, features in variants.items():
            offset, options, guard = select_offset(cal, features)
            margin, parameters = fit_head(cal, held, features)
            corrected = pair_probabilities(outer_p, margin + offset)
            use = held[KEY_FOLD].copy()
            use[PROBABILITIES] = corrected
            predictions[name].append(use)
            settings.append(
                {
                    "fold": f,
                    "head": name,
                    "offset": offset,
                    "inner_guards": guard,
                    "candidates": options,
                    "parameters": parameters,
                }
            )
        log(f"Fold {f + 1}/5 encoder verified and {len(variants)} heads scored")
        del recordings
    final = {n: gated.align(base, pd.concat(rows)) for n, rows in predictions.items()}
    changed = (
        gated.align(base, pd.concat(history_cases))["history_case"]
        if history_cases
        else None
    )
    return {"predictions": final, "history_case": changed, "settings": settings}


def report(base: pd.DataFrame, results: dict) -> dict:
    y = base[LABEL].to_numpy()
    compact = base[PROBABILITIES].to_numpy()
    order = np.argsort(-compact, axis=1, kind="stable")[:, :2]
    best = np.take_along_axis(compact, order, axis=1)
    hard = {
        (i, j): (best[:, 0] <= 2 * best[:, 1])
        & (np.sort(order, axis=1) == [i, j]).all(1)
        for i, j in gated.TARGET_PAIRS
    }
    old = gated.align(
        base, pd.read_parquet(expanded.OUT / "expanded_predictions.parquet")
    )
    comparisons = {"expanded": old[PROBABILITIES].to_numpy()}
    comparisons.update(
        {n: df[PROBABILITIES].to_numpy() for n, df in results["predictions"].items()}
    )
    aligned = base[KEY_FOLD].copy()
    for name, p in comparisons.items():
        aligned[name] = p.argmax(1)
    groups = [np.flatnonzero(aligned.fold.eq(f)) for f in range(5)]
    parents = {
        "head": "expanded",
        "k_head": "head",
        "interruption_head": "k_head",
        "continuation_head": "k_head",
    }
    metrics = {
        name: gated.probability_metrics(y, p, hard) for name, p in comparisons.items()
    }
    reference_stage = metrics["expanded"]["stages"]
    prior = gated.align(base, pd.read_parquet(expanded.ARCHIVED_GATED))
    recall_floor = float(
        (prior[PROBABILITIES].to_numpy().argmax(1)[y == N1] == N1).mean()
    )

    subjects = base.subject_id.to_numpy()
    cm = {
        name: patient_confusions(subjects, y, aligned[name].to_numpy())
        for name in comparisons
    }
    membership = (
        base.groupby("subject_id").fold.first().loc[cm["expanded"][1]].to_numpy()
    )
    folds = [np.flatnonzero(membership == f) for f in range(5)]
    draws = cv_fold_draws(folds, 2000, np.random.default_rng(SEED))
    kappa = {n: cv_scores(c, folds, draws)["kappa"] for n, (c, _) in cm.items()}
    contrasts = {}
    for name in results["predictions"]:
        parent = parents[name]
        main = gated.compare(aligned, name, "expanded", groups)
        match = (
            main
            if parent == "expanded"
            else gated.compare(aligned, name, parent, groups)
        )
        local_stage = metrics[name]["stages"]
        main["kappa_delta"] = float(
            main["candidate"]["kappa"] - main["reference"]["kappa"]
        )
        main["kappa_ci"] = interval(kappa[name] - kappa["expanded"])
        main["passes"] = bool(
            local_stage["N1"]["recall"] >= recall_floor
            and local_stage["N1"]["f1"] >= reference_stage["N1"]["f1"] + 0.005
            and local_stage["N2"]["f1"] >= reference_stage["N2"]["f1"] - 0.005
            and main["delta"] >= -0.002
            and main["kappa_delta"] >= -0.005
        )
        contrasts[name] = {
            "against_expanded": main,
            "matched_reference": parent,
            "matched": match,
        }
    history_comparison = None
    if results["history_case"] is not None:
        in_pair = (
            pair_candidates(comparisons["expanded"])
            & results["history_case"].to_numpy()
        )
        history_comparison = {
            "eligible_epochs": int(in_pair.sum()),
            "label_pair_epochs": int((in_pair & np.isin(y, [N1, N2])).sum()),
            "correct_by_model": {
                n: int((in_pair & aligned[n].eq(y)).sum()) for n in comparisons
            },
        }
    log(f"{len(contrasts)} fitted head variants completed on the original five folds")
    return {
        "contrasts": contrasts,
        "metrics": metrics,
        "settings": results["settings"],
        "sustained_runs": {
            name: expanded.sustained_runs(aligned, name) for name in comparisons
        },
        "history_sensitive_subset": history_comparison,
    }


def main() -> None:
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the head studies before fitting")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frame, splits = development_frame()
    base = pd.read_parquet(BASE)
    summary = json.loads(
        (OUT / "interruption_summary.json").read_text(encoding="utf-8")
    )
    markers = marker_panel(frame, split_history=not summary["qualified"])
    variants = {"head": BASE_FEATURES, "k_head": BASE_FEATURES + K_FEATURES}
    if summary["qualified"]:
        break_features = pd.read_parquet(
            OUT / "interruptions.parquet", columns=[*KEYS, *INTERRUPTION_FEATURES]
        )
        markers = markers.merge(
            break_features, on=KEYS, how="left", validate="one_to_one"
        )
        variants["interruption_head"] = (
            BASE_FEATURES + K_FEATURES + INTERRUPTION_FEATURES
        )
    else:
        directions = validate_histories(frame, splits, markers)
        directions.to_csv(OUT / "continuation_marker_directions.csv", index=False)
        variants["continuation_head"] = BASE_FEATURES + K_FEATURES + SOURCE_HISTORY
    with local_training_slot(), threadpool_limits(limits=4):
        result = replay(frame, splits, base, variants, markers)
        summary = report(base, result)
        for name, rows in result["predictions"].items():
            save_frame(OUT / f"{name}_predictions.parquet", rows)
        atomic_json(OUT / "head_summary.json", summary)


if __name__ == "__main__":
    main()
