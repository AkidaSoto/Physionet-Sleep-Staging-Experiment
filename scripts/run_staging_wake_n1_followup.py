"""Test theta timing and past-context attenuation on the original development folds."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_staging_complementary_model as expanded
import run_staging_gated_temporal as gated
from audit_staging_wake_n1 import annotation_phases
from run_staging_feature_tuning import BASE, save_frame
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    STAGES,
    atomic_json,
    development_frame,
    local_training_slot,
    log,
    pin_manifest,
)


OUT = EXPLORATION / "cinc-wake-n1-followup"
KEY_FOLD = gated.KEY_FOLD
WAKE, N1 = 0, 1
PAIR = [WAKE, N1]
THETA = ["theta_mean", "theta_late_minus_early", "theta_peak_minus_mean"]
ATTENUATION = (0.0, 0.25, 0.50, 1.0)
RESIDUAL_CAP = 0.5
PENALTY = 0.02


def clock_means(values: np.ndarray, minimum_seconds: int) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=-1)
    return np.divide(
        np.where(valid, values, 0).sum(axis=-1),
        count,
        out=np.full(count.shape, np.nan),
        where=count >= minimum_seconds,
    )


def markers(frame: pd.DataFrame) -> pd.DataFrame:
    panel = np.full((len(frame), len(THETA) + 1), np.nan)
    for subject, positions in frame.groupby("subject_id", sort=False).indices.items():
        track = pd.read_parquet(
            CACHE / "tracks1s" / f"{subject}.parquet",
            columns=["second", "c_rel_theta", "o_rel_alpha"],
        )
        if not np.array_equal(track.second, np.arange(len(track))):
            raise ValueError(f"{subject}: waveform track is off the one-second clock")
        epoch = frame.iloc[positions].epoch_index.to_numpy()
        sec = max(len(track), (int(epoch.max()) + 1) * 30)
        values = np.pad(
            track[["c_rel_theta", "o_rel_alpha"]].to_numpy(dtype=float),
            ((0, sec - len(track) + (-sec % 30)), (0, 0)),
            constant_values=np.nan,
        ).reshape(-1, 30, 2)
        theta, alpha = values[epoch, :, 0], values[:, :, 1]
        mean = clock_means(theta, 24)
        thirds = clock_means(theta.reshape(-1, 3, 10), 8)
        alpha = clock_means(alpha, 24)
        panel[positions] = np.column_stack(
            (
                mean,
                np.where(np.isfinite(thirds).all(1), thirds[:, 2] - thirds[:, 0], np.nan),
                thirds.max(1) - mean,
                np.r_[np.nan, np.diff(alpha)][epoch],
            )
        )
    return pd.DataFrame(panel, columns=[*THETA, "alpha_rise"], index=frame.index)


def pair_top_two(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = np.argsort(-p, axis=1, kind="stable")[:, :2]
    scores = np.take_along_axis(p, top, axis=1)
    return (np.sort(top, axis=1) == PAIR).all(1), scores[:, 0] <= 2 * scores[:, 1]


def margin(p: np.ndarray) -> np.ndarray:
    return np.log(p[:, N1].clip(1e-12)) - np.log(p[:, WAKE].clip(1e-12))


def with_pair_residual(p: np.ndarray, residual: np.ndarray) -> np.ndarray:
    edited = p.copy()
    use = residual != 0
    if not pair_top_two(p[use])[0].all():
        raise ValueError("Residual used outside the leading Wake/N1 pair")
    mass = p[use][:, PAIR].sum(1)
    edited[use, N1] = mass * expit(margin(p[use]) + residual[use])
    edited[use, WAKE] = mass - edited[use, N1]
    np.testing.assert_allclose(p[:, PAIR].sum(1), edited[:, PAIR].sum(1), atol=1e-7)
    return edited


def run_probe(partitions: list, panel: pd.DataFrame, hard: np.ndarray) -> None:
    available = np.isfinite(panel[THETA].to_numpy()).all(1)
    variants = {"mean_only": THETA[:1], "mean_and_timing": THETA}
    scored = []
    metrics = []
    for fold, (train, _, outer) in enumerate(partitions):
        fit = train.index[train[LABEL].isin(PAIR) & available[train.index]]
        test = outer.index[hard[outer.index] & available[outer.index]]
        y_train = train.loc[fit, LABEL].eq(N1)
        y_test = outer.loc[test, LABEL].eq(N1)
        results = outer.loc[test, [*KEYS, LABEL]].assign(fold=fold)
        row = {
            "fold": fold,
            "fitting_epochs": len(fit),
            "hard_epochs": int(hard[outer.index].sum()),
            "scored_epochs": len(test),
            "scored_patients": int(results.subject_id.nunique()),
        }
        for name, features in variants.items():
            model = make_pipeline(StandardScaler(), LogisticRegression())
            model.fit(panel.loc[fit, features], y_train)
            prediction = model.predict_proba(panel.loc[test, features])[:, 1]
            results[name] = prediction
            row[name] = float(roc_auc_score(y_test, prediction))
            row[f"{name}_coefficients"] = model[-1].coef_.ravel().tolist()
        row["delta"] = row["mean_and_timing"] - row["mean_only"]
        metrics.append(row)
        scored.append(results)
        log(f"Theta timing fold {fold + 1}/5 AUC change {row['delta']:+.4f}")
    predictions = pd.concat(scored, ignore_index=True)
    save_frame(OUT / "probe_predictions.parquet", predictions)
    atomic_json(
        OUT / "probe_summary.json",
        {
            "folds": metrics,
            "mean_auc": {
                key: float(np.mean([row[key] for row in metrics])) for key in variants
            },
            "pooled_auc": {
                key: float(roc_auc_score(predictions[LABEL].eq(N1), predictions[key]))
                for key in variants
            },
            "mean_delta": float(np.mean([row["delta"] for row in metrics])),
            "wins": sum(row["delta"] > 0 for row in metrics),
            "hard_epochs": int(hard.sum()),
            "scored_epochs": len(predictions),
        },
    )


def theta_residual(
    inner_p: np.ndarray,
    inner_y: np.ndarray,
    inner_values: np.ndarray,
    outer_p: np.ndarray,
    outer_values: np.ndarray,
) -> tuple[np.ndarray, dict]:
    fit = (
        pair_top_two(inner_p)[0]
        & np.isin(inner_y, PAIR)
        & np.isfinite(inner_values).all(1)
    )
    apply = pair_top_two(outer_p)[0] & np.isfinite(outer_values).all(1)
    scaler = StandardScaler()
    features = scaler.fit_transform(inner_values[fit])
    offset = margin(inner_p[fit])
    y = inner_y[fit] == N1

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        activation = np.tanh(features @ weights)
        adjusted = offset + RESIDUAL_CAP * activation
        loss = np.mean(np.logaddexp(0, adjusted) - y * adjusted)
        derivative = (expit(adjusted) - y) * RESIDUAL_CAP * (1 - activation**2)
        return (
            loss + PENALTY * float(weights @ weights) / 2,
            features.T @ derivative / len(y) + PENALTY * weights,
        )

    result = minimize(objective, np.zeros(features.shape[1]), method="L-BFGS-B", jac=True)
    if not result.success:
        raise RuntimeError(f"Theta residual optimization failed: {result.message}")
    residual = np.zeros(len(outer_p))
    residual[apply] = RESIDUAL_CAP * np.tanh(scaler.transform(outer_values[apply]) @ result.x)
    return residual, {
        "coefficients": result.x.tolist(),
        "fit_mean": scaler.mean_.tolist(),
        "fit_scale": scaler.scale_.tolist(),
        "inner_fit_epochs": int(fit.sum()),
        "eligible_outer_epochs": int(apply.sum()),
    }


def past_context_residual(p: np.ndarray, terms: pd.DataFrame, rise: np.ndarray) -> np.ndarray:
    past = terms[f"past_{STAGES[N1]}"] - terms[f"past_{STAGES[WAKE]}"]
    local = terms[f"local_{STAGES[N1]}"] - terms[f"local_{STAGES[WAKE]}"]
    eligible = pair_top_two(p)[0] & (past > 0) & (local < 0) & (rise > 0)
    return np.where(eligible, -past, 0.0)


def count_direction(y: np.ndarray, pred: np.ndarray, source: int) -> int:
    return int(((y == source) & (pred == (N1 if source == WAKE else WAKE))).sum())


def select_attenuation(
    p: np.ndarray, y: np.ndarray, residual: np.ndarray
) -> tuple[float, list]:
    base_f1 = f1_score(y, p.argmax(1), average=None, labels=range(len(STAGES)))
    choices = []
    for strength in ATTENUATION:
        pred = with_pair_residual(p, strength * residual).argmax(1)
        diff = f1_score(y, pred, average=None, labels=range(len(STAGES))) - base_f1
        choices.append({
            "strength": strength,
            "inner_wake_as_n1": count_direction(y, pred, WAKE),
            "inner_n1_as_wake": count_direction(y, pred, N1),
            "inner_macro_f1_delta": float(diff.mean()),
            "inner_stage_f1_delta": diff[PAIR].tolist(),
            "inside_guard": bool(diff.mean() >= -0.002 and diff[PAIR].min() >= -0.005),
        })
    preferred = min(
        (option for option in choices if option["inside_guard"]),
        key=lambda option: (option["inner_wake_as_n1"], option["strength"]),
    )
    return preferred["strength"], choices


def replay_overlays(
    frame: pd.DataFrame, partitions: list, panel: pd.DataFrame
) -> tuple[dict, list]:
    panels = expanded.load_markers(set(frame.subject_id))
    predictions = {name: [] for name in ("frozen", "theta_timing", "alpha_past_gate")}
    settings = []
    for fold, (train, inner, outer) in enumerate(partitions):
        training_ids = set(train.subject_id)
        recordings = gated.sequence.prepare_recordings(frame, training_ids)
        scaling = expanded.add_markers(recordings, panels, training_ids)
        path = expanded.MODEL / f"fold-{fold}"
        archived = json.loads((path / "runtime.json").read_text())["marker_scaling"]
        if scaling != archived:
            raise ValueError(f"Fold {fold}: archive marker normalization does not match")
        model = expanded.expanded_model()
        checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["best_weights"])
        inner_p, inner_terms = gated.score_components(model, recordings, inner)
        del recordings
        keys = outer[[*KEYS, LABEL]].assign(fold=fold)
        outer_terms = gated.align(keys, pd.read_parquet(path / "components.parquet"))
        archived_p = gated.align(keys, pd.read_parquet(path / "predictions.parquet"))
        outer_p = archived_p[PROBABILITIES].to_numpy()
        theta, timing = theta_residual(
            inner_p,
            inner[LABEL].to_numpy(),
            panel.loc[inner.index, THETA[1:]].to_numpy(),
            outer_p,
            panel.loc[outer.index, THETA[1:]].to_numpy(),
        )
        context = past_context_residual(
            inner_p,
            pd.DataFrame(inner_terms, columns=gated.TERM_COLUMNS),
            panel.loc[inner.index, "alpha_rise"].to_numpy(),
        )
        strength, choices = select_attenuation(inner_p, inner[LABEL].to_numpy(), context)
        alpha = past_context_residual(
            outer_p, outer_terms, panel.loc[outer.index, "alpha_rise"].to_numpy()
        )
        overlays = {
            "frozen": outer_p,
            "theta_timing": with_pair_residual(outer_p, theta),
            "alpha_past_gate": with_pair_residual(outer_p, strength * alpha),
        }
        for name, p in overlays.items():
            result = keys.copy()
            result[PROBABILITIES] = p
            predictions[name].append(result)
        settings.append({
            "fold": fold,
            "timing": timing,
            "attenuation": strength,
            "attenuation_choices": choices,
            "alpha_gate_eligible_outer_epochs": int(np.count_nonzero(alpha)),
            "theta_prediction_changes": int(np.count_nonzero(overlays["theta_timing"].argmax(1) != outer_p.argmax(1))),
            "alpha_gate_prediction_changes": int(np.count_nonzero(overlays["alpha_past_gate"].argmax(1) != outer_p.argmax(1))),
        })
        log(f"Wake/N1 overlays fold {fold + 1}/5 scored; past attenuation {strength:.2f}")
    return {name: pd.concat(rows, ignore_index=True) for name, rows in predictions.items()}, settings


def report_overlays(base: pd.DataFrame, hard: np.ndarray, predictions: dict, settings: list) -> None:
    y = base[LABEL].to_numpy()
    use_pair = np.isin(y, PAIR)
    align = base[KEY_FOLD].copy()
    phases, _ = annotation_phases(align)
    fold_rows = [np.flatnonzero(align.fold.eq(i)) for i in range(5)]
    direction = {"wake_as_n1": WAKE, "n1_as_wake": N1}
    metrics = {}
    for name, rows in predictions.items():
        p = gated.align(base, rows)[PROBABILITIES].to_numpy()
        pred = align[name] = p.argmax(1)
        metrics[name] = {
            "recall": dict(zip((STAGES[i] for i in PAIR), recall_score(y, pred, average=None, labels=PAIR).tolist())),
            "auc_all": float(roc_auc_score(y[use_pair] == N1, margin(p)[use_pair])),
            "auc_hard": float(roc_auc_score(y[hard] == N1, margin(p)[hard])),
            "hard_auc_by_fold": [float(roc_auc_score(y[i][hard[i]] == N1, margin(p[i])[hard[i]])) for i in fold_rows],
            "direction_errors": {k: count_direction(y, pred, v) for k, v in direction.items()},
            "direction_errors_by_fold": [{k: count_direction(y[i], pred[i], v) for k, v in direction.items()} for i in fold_rows],
            "phase_errors": {
                phase: {k: count_direction(y[phases == phase], pred[phases == phase], v) for k, v in direction.items()}
                for phase in np.unique(phases)
            },
        }
        ordered = align.sort_values(KEYS)
        metrics[name]["sustained_runs"] = expanded.sustained_runs(ordered, name)
        metrics[name]["sustained_direction_runs"] = {
            k: expanded.sustained_runs(ordered[ordered[LABEL].eq(v) & ordered[name].isin(PAIR)], name)
            for k, v in direction.items()
        }
        save_frame(OUT / f"{name}_predictions.parquet", rows)
    contrasts = {}
    for name, target in (("theta_timing", "n1_as_wake"), ("alpha_past_gate", "wake_as_n1")):
        compare = gated.compare(align, name, "frozen", fold_rows)
        count = metrics[name]["direction_errors"][target]
        baseline = metrics["frozen"]["direction_errors"][target]
        reduction = 1 - count / baseline
        compare.update(
            target=target,
            target_error_reduction=reduction,
            passes_guard=reduction >= 0.05
            and compare["delta"] >= -0.002
            and min(compare["stage_deltas"][STAGES[i]] for i in PAIR) >= -0.005,
        )
        compare["paired_patient_metrics"] = f"{name}_paired_patient_scores.parquet"
        save_frame(
            OUT / compare["paired_patient_metrics"],
            expanded.paired_patient_scores(align.rename(columns={"frozen": "gated", name: "expanded"}))
            .rename(columns={"gated_f1": "frozen_f1", "expanded_f1": "candidate_f1"}),
        )
        contrasts[name] = compare
    atomic_json(OUT / "model_summary.json", {"hard_epochs": int(hard.sum()), "metrics": metrics, "contrasts": contrasts, "settings": settings})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["probe", "overlays"])
    args = parser.parse_args()
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare Wake/N1 follow-up before scoring")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frame, splits = development_frame()
    partitions = [gated.fold_frames(frame, split) for split in splits]
    expected = pd.concat([test[[*KEYS, LABEL]].assign(fold=i) for i, (_, _, test) in enumerate(partitions)]).loc[frame.index]
    base = gated.align(expected, pd.read_parquet(BASE))
    order, close = pair_top_two(base[PROBABILITIES].to_numpy())
    hard = order & close & base[LABEL].isin(PAIR).to_numpy()
    pin_manifest(OUT / "manifest.json", {
        "patients": len(frame.subject_id.unique()),
        "epochs": len(frame),
        "hard_epochs": int(hard.sum()),
        "folds": [
            {part: sorted(rows.subject_id.unique()) for part, rows in zip(("fit", "inner", "outer"), parts)}
            for parts in partitions
        ],
        "theta_columns": THETA,
        "residual_cap": RESIDUAL_CAP,
        "residual_l2": PENALTY,
        "attenuation_options": list(ATTENUATION),
    })
    panel = markers(frame)
    with local_training_slot(), threadpool_limits(limits=4):
        if args.only != "overlays":
            run_probe(partitions, panel, hard)
        if args.only != "probe":
            predictions, settings = replay_overlays(frame, partitions, panel)
            report_overlays(base, hard, predictions, settings)


if __name__ == "__main__":
    main()
