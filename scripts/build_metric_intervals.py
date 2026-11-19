"""Patient-bootstrap 95% intervals for the dashboard's headline scores (no refitting, no model selection).

Reads saved predictions only:
  CV (Boston, 320 patients)   artifacts/staging_window_model/cinc2018/predictions.parquet + attention-CNN fold probs
  Held-out (200 patients)     artifacts/staging_window_model/cinc2018/holdout/predictions_recovered_g20.parquet
  Coimbra (99 nights)         artifacts/staging_window_model/transfer_isruc/predictions.parquet
and writes showcase/content/generated/metric-intervals.json.

Resampling unit is the patient. CV estimates are the mean of the five fold macro-F1s, so patients are resampled
within each test fold and the fold mean is recomputed; held-out and Coimbra scores are pooled over all epochs, so
patients are resampled across the cohort. Paired differences use the same resampled patients for both models.

Usage: python scripts/build_metric_intervals.py [--reps 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_staging_dashboard_data as ex  # noqa: E402

HOLDOUT = ex.WINDOW_RUN / "holdout" / "predictions_recovered_g20.parquet"
ISRUC = ex.ARTIFACTS / "staging_window_model" / "transfer_isruc" / "predictions.parquet"
OUTPUT = ex.GENERATED / "metric-intervals.json"
STAGES = ex.STAGES
SEED = 7


def patient_confusions(subject: np.ndarray, y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(patients, 5, 5) confusion counts and the patient ids, in a fixed order."""
    ids, inv = np.unique(subject.astype(str), return_inverse=True)
    cm = np.zeros((len(ids), 5, 5), np.int64)
    np.add.at(cm, (inv, y.astype(int), pred.astype(int)), 1)
    return cm, ids


def scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Macro-F1, kappa and per-stage F1 from (..., 5, 5) confusion counts."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    true, pred = cm.sum(-1).astype(float), cm.sum(-2).astype(float)
    denom = true + pred
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    n = cm.sum((-2, -1)).astype(float)
    po = tp.sum(-1) / n
    pe = (true * pred).sum(-1) / n**2
    return {"macro_f1": f1.mean(-1), "kappa": (po - pe) / (1 - pe), "stage_f1": f1}


def interval(samples: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(samples, [2.5, 97.5], axis=0)]


def summary(point: dict[str, np.ndarray], boot: dict[str, np.ndarray], n_patients: int) -> dict:
    return {"n_patients": n_patients,
            "macro_f1": float(point["macro_f1"]), "macro_f1_ci": interval(boot["macro_f1"]),
            "kappa": float(point["kappa"]), "kappa_ci": interval(boot["kappa"]),
            "stage_f1": {s: float(v) for s, v in zip(STAGES, point["stage_f1"])},
            "stage_f1_ci": {s: [float(a), float(b)] for s, a, b in zip(STAGES, *np.percentile(boot["stage_f1"], [2.5, 97.5], axis=0))}}


def pooled(cm: np.ndarray, reps: int, rng: np.random.Generator) -> dict:
    """Pooled scores over all patients, resampling patients across the cohort."""
    draws = rng.integers(0, len(cm), size=(reps, len(cm)))
    return summary(scores(cm.sum(0)), scores(cm[draws].sum(1)), len(cm))


def cv_fold_draws(folds: list[np.ndarray], reps: int, rng: np.random.Generator) -> list[np.ndarray]:
    """For each fold, (reps, fold size) indices of patients resampled within that fold."""
    return [f[rng.integers(0, len(f), size=(reps, len(f)))] for f in folds]


def cv_scores(cm: np.ndarray, folds: list[np.ndarray], draws: list[np.ndarray] | None) -> dict[str, np.ndarray]:
    """Mean over folds of each fold's pooled scores (the dashboard's CV estimator)."""
    per_fold = [scores(cm[f].sum(0) if draws is None else cm[d].sum(1)) for f, d in zip(folds, draws or [None] * len(folds))]
    return {k: np.mean([p[k] for p in per_fold], axis=0) for k in per_fold[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    # ---- Boston cross-validation: every main-page model on the same patients and folds
    preds, splits = ex._prediction_frame()
    preds["cnn__epoch"], preds["cnn__hmm"] = ex._cnn_predictions(preds, splits)
    if (preds["cnn__epoch"] < 0).any():
        raise SystemExit("CNN predictions missing for some epochs")
    columns = {model_id: f"{src[0]}__{src[1]}" for model_id, *_, src in ex.MODELS}
    subject = preds["subject_id"].astype(str).to_numpy()
    y = preds["y_true"].to_numpy()
    cms = {}
    for model_id, col in columns.items():
        cms[model_id], ids = patient_confusions(subject, y, preds[col].to_numpy())
    fold_of = preds.groupby(preds["subject_id"].astype(str))["fold_index"].agg(["min", "max"])
    if (fold_of["min"] != fold_of["max"]).any() or (fold_of["min"] < 0).any():
        raise SystemExit("a patient spans several test folds")  # leakage guard: folds must be patient-grouped
    fold_ids = fold_of.loc[ids, "min"].to_numpy()
    folds = [np.flatnonzero(fold_ids == k) for k in range(len(splits))]
    draws = cv_fold_draws(folds, args.reps, rng)
    cv_point = {m: cv_scores(cm, folds, None) for m, cm in cms.items()}
    cv_boot = {m: cv_scores(cm, folds, draws) for m, cm in cms.items()}
    cv = {m: summary(cv_point[m], cv_boot[m], len(ids)) for m in cms}
    diffs = {}
    for other in cms:
        if other == ex.ADOPTED:
            continue
        d = cv_boot[other]["macro_f1"] - cv_boot[ex.ADOPTED]["macro_f1"]
        diffs[other] = {"minus_adopted": float(cv_point[other]["macro_f1"] - cv_point[ex.ADOPTED]["macro_f1"]),
                        "ci": interval(d), "share_above_zero": float(np.mean(d > 0))}

    # ---- Held-out Boston (model C, evaluation look 2) and Coimbra (both scorers)
    h = pd.read_parquet(HOLDOUT)
    holdout_cm, _ = patient_confusions(h["subject_id"].to_numpy(), h["label.stage_30s"].to_numpy(), h["prediction"].to_numpy())
    i = pd.read_parquet(ISRUC)
    if i["subject_id"].nunique() != 99 or (i[["scorer1", "scorer2", "pred"]] < 0).any().any():
        raise SystemExit("unexpected Coimbra predictions (expected 99 nights, scored epochs only)")
    coimbra = {}
    for key, ref, test in [("model_vs_scorer1", "scorer1", "pred"), ("model_vs_scorer2", "scorer2", "pred"),
                           ("scorer1_vs_scorer2", "scorer1", "scorer2")]:
        cm, _ = patient_confusions(i["subject_id"].to_numpy(), i[ref].to_numpy(), i[test].to_numpy())
        coimbra[key] = pooled(cm, args.reps, rng)
    # Per-night macro-F1 against scorer 2, as already saved by the Coimbra run (scorer 1's is in isruc-status.json).
    nights = pd.read_csv(ISRUC.parent / "isruc_subjects.csv")
    coimbra["model_vs_scorer2"]["subject_macro_f1"] = [float(v) for v in nights["macro_f1_scorer2"]]

    payload = {
        "method": f"Patient bootstrap, {args.reps} resamples, 2.5th-97.5th percentiles. CV: patients resampled within "
                  "each test fold, mean of fold macro-F1. Held-out and Coimbra: patients resampled across the cohort, "
                  "pooled over epochs.",
        "reps": args.reps,
        "cv": cv,
        "cv_paired_vs_adopted": diffs,
        "holdout": pooled(holdout_cm, args.reps, rng),
        "coimbra": coimbra,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    for name, s in [("CV C", cv[ex.ADOPTED]), ("held-out", payload["holdout"]), *[(f"Coimbra {k}", v) for k, v in coimbra.items()]]:
        print(f"{name:28s} macro-F1 {s['macro_f1']:.3f} [{s['macro_f1_ci'][0]:.3f}, {s['macro_f1_ci'][1]:.3f}]  "
              f"kappa {s['kappa']:.3f} [{s['kappa_ci'][0]:.3f}, {s['kappa_ci'][1]:.3f}]")
    for m, d in diffs.items():
        print(f"CV {m:18s} - C  {d['minus_adopted']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
