"""Evaluate Experiment 78: unified multi-family ensemble + OOF calibration across #73-#77."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions  # noqa: E402
from run_staging_wake_n1_data_curve import APNEA_BANDS, apnea_bands  # noqa: E402
from run_waveform_development import (  # noqa: E402
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    SEED,
    STAGES,
    atomic_json,
    write_predictions,
)

OUT = EXPLORATION / "cinc-unified-ensemble"
KEY_FOLD = [*KEYS, LABEL, "fold"]
CLASSES = np.arange(len(STAGES))

CANDIDATE_MODELS: dict[str, Path] = {
    "73_expanded704": EXPLORATION / "cinc-all-additional-training/predictions.parquet",
    "53_searched_hgb": EXPLORATION / "cinc-recipe-search/searched_predictions.parquet",
    "47_seq_long": EXPLORATION / "cinc-feature-context/long_predictions.parquet",
    "40_resnet": EXPLORATION / "cinc-waveform/scratch/predictions.parquet",
    "75_bigru_unified": EXPLORATION / "cinc-bigru-unified/predictions.parquet",
    "76_hgb_704": EXPLORATION / "cinc-hgb-704/predictions.parquet",
    "77_waveform_subepoch": EXPLORATION / "cinc-waveform-subepoch/predictions.parquet",
}


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = np.bincount(y_true * 5 + y_pred, minlength=25).reshape(5, 5)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(2.0 * tp / np.maximum(denom, 1)))


def load_aligned_probs(base: pd.DataFrame, path: Path) -> np.ndarray:
    source = pd.read_parquet(path, columns=[*KEY_FOLD, *PROBABILITIES])
    joined = base[KEY_FOLD].merge(source, on=KEY_FOLD, validate="one_to_one", how="left", indicator=True)
    if len(source) != len(base) or not joined["_merge"].eq("both").all():
        raise ValueError(f"{path}: mismatched development keys or folds")
    probs = joined[PROBABILITIES].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError(f"{path}: non-finite or negative probabilities")
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def eval_predictions(
    y: np.ndarray,
    probs: np.ndarray,
    fold_indices: list[np.ndarray],
) -> dict:
    pred = probs.argmax(axis=1)
    fold_f1 = [float(f1_score(y[idx], pred[idx], labels=CLASSES, average="macro", zero_division=0)) for idx in fold_indices]
    fold_kappa = [float(cohen_kappa_score(y[idx], pred[idx])) for idx in fold_indices]
    stage_f1 = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    return {
        "macro_f1": float(np.mean(fold_f1)),
        "fold_macro_f1": fold_f1,
        "kappa": float(np.mean(fold_kappa)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-12, 1.0))),
        "stage_f1": {s: float(v) for s, v in zip(STAGES, stage_f1, strict=True)},
        "confusion_matrix": cm.tolist(),
    }


def oof_prior_power(y: np.ndarray, fold_arr: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, list[float]]:
    oof = np.zeros_like(probs)
    alphas: list[float] = []
    for f in range(5):
        tr, te = fold_arr != f, fold_arr == f
        counts = np.bincount(y[tr], minlength=5).astype(np.float64)
        priors = counts / counts.sum()
        best_a, best_s = 0.0, -1.0
        for a in np.linspace(-0.3, 0.8, 45):
            s = fast_macro_f1(y[tr], (probs[tr] * (priors ** a)).argmax(axis=1))
            if s > best_s:
                best_s, best_a = s, float(a)
        alphas.append(round(best_a, 4))
        adj = probs[te] * (priors ** best_a)
        oof[te] = adj / adj.sum(axis=1, keepdims=True)
    return oof, alphas


def oof_logit_offsets(y: np.ndarray, fold_arr: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    oof = np.zeros_like(probs)
    biases: list[list[float]] = []
    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    for f in range(5):
        tr, te = fold_arr != f, fold_arr == f
        lp_tr = log_p[tr]
        y_tr = y[tr]
        bias = np.zeros(5, dtype=np.float64)
        best = fast_macro_f1(y_tr, lp_tr.argmax(axis=1))
        for _ in range(3):
            for c in range(1, 5):
                cur = lp_tr + bias
                for delta in np.linspace(-0.7, 0.7, 29):
                    cur[:, c] = lp_tr[:, c] + delta
                    s = fast_macro_f1(y_tr, cur.argmax(axis=1))
                    if s > best:
                        best = s
                        bias[c] = float(delta)
        biases.append([round(float(v), 4) for v in bias])
        logits_te = log_p[te] + bias
        logits_te -= logits_te.max(axis=1, keepdims=True)
        e = np.exp(logits_te)
        oof[te] = e / e.sum(axis=1, keepdims=True)
    return oof, biases


def fit_oof_simplex_blend(
    y: np.ndarray,
    fold_arr: np.ndarray,
    prob_list: list[np.ndarray],
    *,
    mode: str = "prob",
) -> tuple[np.ndarray, list[list[float]]]:
    """Optimize non-negative simplex weights minimizing training-fold log loss, evaluated OOF."""
    m = len(prob_list)
    oof = np.zeros_like(prob_list[0])
    weights_by_fold: list[list[float]] = []
    stacked = np.stack(prob_list, axis=0)  # (M, N, 5)
    log_stacked = np.log(np.clip(stacked, 1e-8, 1.0))

    for f in range(5):
        tr, te = fold_arr != f, fold_arr == f
        y_tr = y[tr]
        sub_idx = np.arange(len(y_tr))[::4]  # fast deterministic subsample on 230k training rows
        y_sub = y_tr[sub_idx]
        if mode == "prob":
            X_sub = stacked[:, tr, :][:, sub_idx, :]
        else:
            X_sub = log_stacked[:, tr, :][:, sub_idx, :]

        def obj(raw_w: np.ndarray) -> float:
            w = np.exp(raw_w - raw_w.max())
            w /= w.sum()
            if mode == "prob":
                p = np.tensordot(w, X_sub, axes=(0, 0))
                return float(-np.log(np.clip(p[np.arange(len(y_sub)), y_sub], 1e-8, 1.0)).mean())
            logits = np.tensordot(w, X_sub, axes=(0, 0))
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=1, keepdims=True)
            return float(-np.log(np.clip(p[np.arange(len(y_sub)), y_sub], 1e-8, 1.0)).mean())

        res = minimize(obj, np.zeros(m, dtype=np.float64), method="BFGS")
        w = np.exp(res.x - res.x.max())
        w /= w.sum()
        weights_by_fold.append([round(float(v), 4) for v in w])
        if mode == "prob":
            oof[te] = np.tensordot(w, stacked[:, te, :], axes=(0, 0))
        else:
            logits_te = np.tensordot(w, log_stacked[:, te, :], axes=(0, 0))
            logits_te -= logits_te.max(axis=1, keepdims=True)
            e = np.exp(logits_te)
            oof[te] = e / e.sum(axis=1, keepdims=True)
    return oof, weights_by_fold


def paired_bootstrap_vs_ref(
    base: pd.DataFrame,
    y: np.ndarray,
    pred_cand: np.ndarray,
    pred_ref: np.ndarray,
) -> dict:
    subjects = base["subject_id"].to_numpy()
    cm_cand, patients = patient_confusions(subjects, y, pred_cand)
    cm_ref, _ = patient_confusions(subjects, y, pred_ref)
    membership = base.groupby("subject_id")["fold"].first().loc[patients].to_numpy()
    groups = [np.flatnonzero(membership == f) for f in range(5)]
    draws = cv_fold_draws(groups, 2000, np.random.default_rng(SEED))
    boot_cand = cv_scores(cm_cand, groups, draws)
    boot_ref = cv_scores(cm_ref, groups, draws)
    point_cand = cv_scores(cm_cand, groups, None)
    point_ref = cv_scores(cm_ref, groups, None)
    return {
        "macro_f1_delta": float(point_cand["macro_f1"] - point_ref["macro_f1"]),
        "macro_f1_ci": interval(boot_cand["macro_f1"] - boot_ref["macro_f1"]),
        "kappa_delta": float(point_cand["kappa"] - point_ref["kappa"]),
        "kappa_ci": interval(boot_cand["kappa"] - boot_ref["kappa"]),
        "stage_f1_deltas": {
            s: float(point_cand["stage_f1"][i] - point_ref["stage_f1"][i])
            for i, s in enumerate(STAGES)
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-all", action="store_true", help="Fail if any candidate model is missing")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref_path = CANDIDATE_MODELS["73_expanded704"]
    base = pd.read_parquet(ref_path, columns=[*KEY_FOLD, *PROBABILITIES]).sort_values(KEYS, ignore_index=True)
    if len(base) != 288923 or base["subject_id"].nunique() != 320 or base.duplicated(KEYS).any():
        raise ValueError("Expected complete 320-patient development reference")

    y = base[LABEL].to_numpy(dtype=np.int64)
    fold_arr = base["fold"].to_numpy(dtype=np.int64)
    fold_indices = [np.flatnonzero(fold_arr == f) for f in range(5)]

    available: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name, path in CANDIDATE_MODELS.items():
        if path.is_file():
            available[name] = load_aligned_probs(base, path)
        else:
            missing.append(name)
    if args.require_all and missing:
        raise FileNotFoundError(f"Missing candidate model predictions: {missing}")

    single_results = {}
    for name, probs in available.items():
        cal_p, alphas = oof_prior_power(y, fold_arr, probs)
        single_results[name] = {
            "raw": eval_predictions(y, probs, fold_indices),
            "prior_calibrated": {**eval_predictions(y, cal_p, fold_indices), "alphas": alphas},
        }

    # Build OOF simplex ensembles over all available diverse models
    model_names = list(available.keys())
    prob_list = [available[n] for n in model_names]
    blend_prob, w_prob = fit_oof_simplex_blend(y, fold_arr, prob_list, mode="prob")
    blend_log, w_log = fit_oof_simplex_blend(y, fold_arr, prob_list, mode="logit")

    blend_prob_prior, a_prob = oof_prior_power(y, fold_arr, blend_prob)
    blend_prob_offsets, b_prob = oof_logit_offsets(y, fold_arr, blend_prob)
    blend_log_prior, a_log = oof_prior_power(y, fold_arr, blend_log)
    blend_log_offsets, b_log = oof_logit_offsets(y, fold_arr, blend_log)

    ensembles = {
        "oof_prob_simplex": {
            **eval_predictions(y, blend_prob, fold_indices),
            "weights_by_fold": w_prob,
        },
        "oof_prob_simplex_prior": {
            **eval_predictions(y, blend_prob_prior, fold_indices),
            "weights_by_fold": w_prob,
            "alphas": a_prob,
        },
        "oof_prob_simplex_offsets": {
            **eval_predictions(y, blend_prob_offsets, fold_indices),
            "weights_by_fold": w_prob,
            "offsets": b_prob,
        },
        "oof_logit_simplex": {
            **eval_predictions(y, blend_log, fold_indices),
            "weights_by_fold": w_log,
        },
        "oof_logit_simplex_prior": {
            **eval_predictions(y, blend_log_prior, fold_indices),
            "weights_by_fold": w_log,
            "alphas": a_log,
        },
        "oof_logit_simplex_offsets": {
            **eval_predictions(y, blend_log_offsets, fold_indices),
            "weights_by_fold": w_log,
            "offsets": b_log,
        },
    }

    # Leave-one-model-out ablation on prob_simplex + prior_power
    ablations = {}
    if len(model_names) >= 3:
        for drop in model_names:
            keep = [n for n in model_names if n != drop]
            sub_p, _ = fit_oof_simplex_blend(y, fold_arr, [available[n] for n in keep], mode="prob")
            sub_cal, _ = oof_prior_power(y, fold_arr, sub_p)
            ablations[f"drop_{drop}"] = {
                "raw_macro_f1": eval_predictions(y, sub_p, fold_indices)["macro_f1"],
                "prior_cal_macro_f1": eval_predictions(y, sub_cal, fold_indices)["macro_f1"],
            }

    best_name = max(ensembles, key=lambda k: ensembles[k]["macro_f1"])
    best_probs = {
        "oof_prob_simplex": blend_prob,
        "oof_prob_simplex_prior": blend_prob_prior,
        "oof_prob_simplex_offsets": blend_prob_offsets,
        "oof_logit_simplex": blend_log,
        "oof_logit_simplex_prior": blend_log_prior,
        "oof_logit_simplex_offsets": blend_log_offsets,
    }[best_name]

    # Paired bootstrap vs #73
    ref_pred = available["73_expanded704"].argmax(axis=1)
    best_pred = best_probs.argmax(axis=1)
    paired_vs_73 = paired_bootstrap_vs_ref(base, y, best_pred, ref_pred)

    # AHI-band breakdown
    band_map = apnea_bands(sorted(base["subject_id"].unique()))
    band_col = base["subject_id"].map(band_map)
    ahi_breakdown = {}
    for code, band_name in APNEA_BANDS.items():
        mask = band_col.eq(code).to_numpy()
        if not mask.any():
            continue
        ahi_breakdown[band_name] = {
            "patients": int(base.loc[mask, "subject_id"].nunique()),
            "epochs": int(mask.sum()),
            "ref73_macro_f1": float(f1_score(y[mask], ref_pred[mask], labels=CLASSES, average="macro")),
            "best_macro_f1": float(f1_score(y[mask], best_pred[mask], labels=CLASSES, average="macro")),
        }

    # Save fold-aligned predictions
    base_keys = base[[*KEYS, LABEL]].copy()
    for f in range(5):
        pass
    # write_predictions takes (path, frame, probabilities, fold)
    # Build combined DataFrame directly to preserve per-row fold column
    out_raw = base[KEY_FOLD].copy()
    out_cal = base[KEY_FOLD].copy()
    for j, stage in enumerate(STAGES):
        out_raw[f"p_{stage}"] = blend_prob[:, j].astype(np.float32)
        out_cal[f"p_{stage}"] = best_probs[:, j].astype(np.float32)
    out_raw.to_parquet(OUT / "predictions.parquet", index=False)
    out_cal.to_parquet(OUT / "calibrated_predictions.parquet", index=False)

    summary = {
        "experiment": 78,
        "models_included": model_names,
        "models_missing": missing,
        "single_models": single_results,
        "ensembles": ensembles,
        "ablations": ablations,
        "best_ensemble": best_name,
        "paired_vs_73": paired_vs_73,
        "ahi_bands": ahi_breakdown,
    }
    atomic_json(OUT / "summary.json", summary)

    print("=== Experiment 78 Unified Ensemble Summary ===")
    print("Models included:", model_names)
    for k, v in single_results.items():
        print(f"  {k:22s}: raw={v['raw']['macro_f1']:.5f} (kap={v['raw']['kappa']:.5f}) | prior_cal={v['prior_calibrated']['macro_f1']:.5f}")
    for k, v in ensembles.items():
        print(f"  {k:28s}: mF1={v['macro_f1']:.5f} | kap={v['kappa']:.5f} | ll={v['log_loss']:.4f}")
    print(f"Best: {best_name} -> {ensembles[best_name]['macro_f1']:.5f}, delta vs #73 = {paired_vs_73['macro_f1_delta']:+.5f} {paired_vs_73['macro_f1_ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
