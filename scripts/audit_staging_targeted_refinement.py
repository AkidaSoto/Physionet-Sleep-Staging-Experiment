"""Audit agreement and test narrow refinements of the frozen expanded model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score
import torch

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
    EXPLORATION,
    LABEL,
    PROBABILITIES,
    SEED,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
)


OUT = EXPLORATION / "cinc-targeted-refinement"
PARAMETERS = {
    "n1_bias": (0.0, 0.10, 0.20, 0.30, 0.40),
    "past_attenuation": (0.0, 0.25, 0.50, 1.0),
}
KEYS = gated.KEY_FOLD
PAIR = [1, 2]


def adjust(
    p: np.ndarray,
    local_difference: np.ndarray,
    past_difference: np.ndarray,
    name: str,
    setting: float,
) -> np.ndarray:
    if setting == 0:
        return p
    best = np.argsort(-p, axis=1, kind="stable")[:, :2]
    eligible = (np.sort(best, axis=1) == PAIR).all(1)
    if name == "past_attenuation":
        eligible &= (local_difference > 0) & (past_difference < 0)
        bonus = -setting * past_difference[eligible]
    else:
        bonus = setting
    a, b = p[eligible, 1], p[eligible, 2]
    mass = a + b
    log_ratio = np.log(a.clip(1e-12)) - np.log(b.clip(1e-12))
    probability_n1 = expit(log_ratio + bonus) * mass
    output = p.copy()
    output[eligible, 1] = probability_n1
    output[eligible, 2] = mass - probability_n1
    return output


def choose(
    y: np.ndarray, p: np.ndarray, local: np.ndarray, past: np.ndarray, name: str
) -> tuple[float, list[dict]]:
    baseline = f1_score(
        y, p.argmax(1), labels=np.arange(5), average=None, zero_division=0
    )
    options = []
    for setting in PARAMETERS[name]:
        pred = adjust(p, local, past, name, setting).argmax(1)
        stages = f1_score(y, pred, labels=np.arange(5), average=None, zero_division=0)
        options.append(
            {
                "setting": setting,
                "n1_f1": float(stages[1]),
                "delta_macro_f1": float(stages.mean() - baseline.mean()),
                "delta_n2_f1": float(stages[2] - baseline[2]),
            }
        )
    choice = max(
        (
            x
            for x in options
            if x["delta_macro_f1"] >= -0.002 and x["delta_n2_f1"] >= -0.005
        ),
        key=lambda x: (x["n1_f1"], -x["setting"]),
    )
    return choice["setting"], options


def replay(
    frame: pd.DataFrame, splits: list, base: pd.DataFrame
) -> tuple[dict, list[dict]]:
    markers = expanded.load_markers(set(frame.subject_id))
    distributions = {
        name: [] for name in (*PARAMETERS, "without_k_shape", "without_recent_event")
    }
    choices = []
    for fold, split in enumerate(splits):
        train, val, test = fold_frames(frame, split)
        inner = set(train.subject_id)
        recordings = gated.sequence.prepare_recordings(frame, inner)
        scale = expanded.add_markers(recordings, markers, inner)
        fold_path = expanded.MODEL / f"fold-{fold}"
        archived_scale = json.loads((fold_path / "runtime.json").read_text())[
            "marker_scaling"
        ]
        if scale != archived_scale:
            raise ValueError(
                f"Fold {fold}: marker normalization differs from the frozen fit"
            )
        ckpt = torch.load(
            fold_path / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        model = expanded.expanded_model()
        model.load_state_dict(ckpt["best_weights"])
        val_prob, val_terms = gated.score_components(model, recordings, val)
        test_prob, test_terms = gated.score_components(model, recordings, test)
        reference = pd.read_parquet(fold_path / "predictions.parquet")
        test_keys = test[[*gated.KEYS, LABEL]].assign(fold=fold)
        reference = gated.align(test_keys, reference)
        np.testing.assert_allclose(
            reference[PROBABILITIES], test_prob, rtol=0, atol=2e-7
        )

        def difference(terms: np.ndarray, component: str) -> np.ndarray:
            return (
                terms[:, gated.TERM_COLUMNS.index(f"{component}_N1")]
                - terms[:, gated.TERM_COLUMNS.index(f"{component}_N2")]
            )

        for name in PARAMETERS:
            setting, options = choose(
                val[LABEL].to_numpy(),
                val_prob,
                difference(val_terms, "local"),
                difference(val_terms, "past"),
                name,
            )
            choices.append(
                {
                    "fold": fold,
                    "method": name,
                    "setting": setting,
                    "inner_results": options,
                }
            )
            p = adjust(
                test_prob,
                difference(test_terms, "local"),
                difference(test_terms, "past"),
                name,
                setting,
            )
            distributions[name].append(
                test_keys.assign(**dict(zip(PROBABILITIES, p.T, strict=True)))
            )
        for name, feature in zip(
            ("without_k_shape", "without_recent_event"),
            expanded.FEATURES[-2:],
            strict=True,
        ):
            column = expanded.ORIGINAL_WIDTH + expanded.FEATURES.index(feature)
            p = gated.sequence.score_fold(
                expanded.WithoutMarkerFamily(model, slice(column, column + 1)),
                recordings,
                test,
            )
            distributions[name].append(
                test_keys.assign(**dict(zip(PROBABILITIES, p.T, strict=True)))
            )
        log(f"Fold {fold + 1}/5 replayed and both marker sensitivities scored")
        del recordings
    return {
        name: gated.align(base, pd.concat(frames))
        for name, frames in distributions.items()
    }, choices


def report(base: pd.DataFrame, details: dict, choices: list[dict]) -> None:
    y = base[LABEL].to_numpy()
    compact = base[PROBABILITIES].to_numpy()
    prior = gated.align(base, pd.read_parquet(expanded.ARCHIVED_GATED))
    control = gated.align(
        base, pd.read_parquet(expanded.OUT / "expanded_predictions.parquet")
    )
    blend = gated.align(
        base,
        pd.read_parquet(
            EXPLORATION / "cinc-feature-context/paired_predictions.parquet",
            columns=[*KEYS, "long_blend"],
        ),
    )
    p = {
        "compact_hgb": compact,
        "gated18": prior[PROBABILITIES].to_numpy(),
        "expanded": control[PROBABILITIES].to_numpy(),
    }
    p.update({name: table[PROBABILITIES].to_numpy() for name, table in details.items()})
    prediction = {name: value.argmax(1) for name, value in p.items()}
    prediction["earlier_feature_blend"] = blend["long_blend"].to_numpy()
    subjects = base.subject_id.to_numpy()
    ids = np.unique(subjects.astype(str))
    membership = base.groupby("subject_id").fold.first().loc[ids].to_numpy()
    groups = [np.flatnonzero(membership == i) for i in range(5)]
    draws = cv_fold_draws(groups, 2000, np.random.default_rng(SEED))
    matrices = {
        name: patient_confusions(subjects, y, pred)[0]
        for name, pred in prediction.items()
    }
    points = {name: cv_scores(cm, groups, None) for name, cm in matrices.items()}
    boot = {name: cv_scores(cm, groups, draws) for name, cm in matrices.items()}
    order = np.argsort(-compact, axis=1, kind="stable")[:, :2]
    best = np.take_along_axis(compact, order, axis=1)
    hard = {
        (i, j): (best[:, 0] <= 2 * best[:, 1])
        & (np.sort(order, axis=1) == [i, j]).all(1)
        for i, j in gated.TARGET_PAIRS
    }
    metrics = {
        name: gated.probability_metrics(y, value, hard) for name, value in p.items()
    }
    agreements = {}
    for name, val in points.items():
        agreements[name] = {
            metric: float(val[metric]) for metric in ("macro_f1", "kappa")
        }
        agreements[name]["paired_kappa_vs_expanded"] = interval(
            boot[name]["kappa"] - boot["expanded"]["kappa"]
        )
        agreements[name]["fold_kappa"] = [
            float(cv_scores(matrices[name], [g], None)["kappa"]) for g in groups
        ]
    for name in PARAMETERS:
        stage = metrics[name]["stages"]
        ref = metrics["expanded"]["stages"]
        agreements[name]["passes"] = bool(
            stage["N1"]["recall"] >= metrics["gated18"]["stages"]["N1"]["recall"]
            and stage["N1"]["f1"] - ref["N1"]["f1"] >= 0.005
            and stage["N2"]["f1"] - ref["N2"]["f1"] >= -0.005
            and points[name]["macro_f1"] - points["expanded"]["macro_f1"] >= -0.002
        )
    cohort = gated.align(
        base,
        pd.read_parquet(
            gated.ARCHIVED,
            columns=[*KEYS, "paired_run_6", "baseline_to_searched", "searched"],
        ),
    )
    selected = (
        cohort.paired_run_6
        & cohort.baseline_to_searched.eq("persistent_same")
        & cohort[LABEL].eq(2)
        & cohort.searched.eq(1)
    )
    stress = {
        name: int((selected & (pred == 2)).sum()) for name, pred in prediction.items()
    }
    for name, table in details.items():
        save_frame(OUT / f"{name}_predictions.parquet", table)
    atomic_json(
        OUT / "summary.json",
        {
            "agreement": agreements,
            "metrics": metrics,
            "settings": choices,
            "sustained_N2_N1_correct": stress,
        },
    )


def main() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the diagnostics before evaluating them")
    frame, splits = development_frame()
    base = pd.read_parquet(BASE)
    with local_training_slot():
        details, choices = replay(frame, splits, base)
        report(base, details, choices)


if __name__ == "__main__":
    main()
