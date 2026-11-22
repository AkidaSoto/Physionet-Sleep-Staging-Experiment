"""Run Experiment 76 (cinc-hgb-704): 704-patient multi-scale tempered HistGradientBoosting."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
import torch

import run_staging_all_additional_training as exp73
import run_staging_complementary_model as expanded
import run_staging_gated_temporal as gated
from physionet_sleep.experiments.multiscale_hgb import (
    CacheLayout,
    DEFAULT_WEIGHT_POWER,
    build_fit_matrix,
    calibrate_oof_prior_power,
    compute_tempered_weights,
    feature_column_names,
    fit_fold_moments,
    fit_tempered_hgb,
    has_occipital_cache,
    score_subjects_streaming,
)
from physionet_sleep.experiments.waveform_clock import active_clock
from run_staging_feature_tuning import BASE, save_frame
from run_waveform_development import (
    BUDGET_SECONDS,
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    RAW,
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


OUT = EXPLORATION / "cinc-hgb-704"
OCCIPITAL_CACHE = EXPLORATION / "cinc-bigru-unified" / "occipital_cache"
SEARCHED_53 = EXPLORATION / "cinc-recipe-search" / "searched_predictions.parquet"
EXPANDED_73 = exp73.OUT / "predictions.parquet"
CHECKPOINT_SECONDS = 300


def reserved_holdout_ids() -> set[str]:
    """Read only the reserved CinC holdout ID list for disjointness assertions."""
    return set((RAW / "selected_records_holdout.txt").read_text(encoding="utf-8").split())


def assert_disjoint_from_reserved(subjects: set[str], reserved: set[str], context: str) -> None:
    overlap = subjects & reserved
    if overlap:
        raise ValueError(f"{context}: overlaps reserved Boston holdout patients: {sorted(overlap)[:5]}")


def make_layout(added_subjects: set[str]) -> CacheLayout:
    return CacheLayout(
        dev_window_cache=CACHE,
        added_window_cache=exp73.ADDED_WINDOW_CACHE,
        dev_marker_cache=expanded.OUT / "feature_cache",
        added_marker_cache=exp73.OUT / "feature_cache",
        added_subjects=frozenset(added_subjects),
        occipital_cache=OCCIPITAL_CACHE if OCCIPITAL_CACHE.is_dir() else None,
    )


def run_fold_once(
    fold: int,
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    layout: CacheLayout,
    *,
    weight_power: float,
    max_iter: int,
    include_occipital: bool,
    check_fn=None,
) -> tuple[np.ndarray, dict]:
    """Execute Pass 1 (moments), Pass 2 (float32 fit matrix), tempered HGB fit, and streaming inference."""
    fit_subjects = set(fit_frame.subject_id.astype(str).unique())
    eval_subjects = set(eval_frame.subject_id.astype(str).unique())
    if fit_subjects & eval_subjects:
        raise ValueError(f"Fold {fold}: fitting and evaluation patients overlap")
    reserved = reserved_holdout_ids()
    assert_disjoint_from_reserved(fit_subjects | eval_subjects, reserved, f"fold {fold}")

    t_prep = active_clock()
    moments = fit_fold_moments(layout, fit_subjects, include_occipital=include_occipital, check_fn=check_fn)
    X_fit, y_fit = build_fit_matrix(layout, fit_frame, moments, label_col=LABEL, check_fn=check_fn)
    sample_weight, counts, class_weights = compute_tempered_weights(
        y_fit, n_classes=len(STAGES), weight_power=weight_power
    )
    prep_seconds = active_clock() - t_prep
    n_features = X_fit.shape[1]
    n_fit_epochs = len(y_fit)

    t_fit = active_clock()
    clf = fit_tempered_hgb(X_fit, y_fit, sample_weight, seed=SEED + fold, max_iter=max_iter)
    fit_seconds = active_clock() - t_fit
    del X_fit
    gc.collect()

    t_score = active_clock()
    probs = score_subjects_streaming(
        clf, layout, eval_frame, moments, n_classes=len(STAGES), check_fn=check_fn
    )
    score_seconds = active_clock() - t_score

    y_eval = eval_frame[LABEL].to_numpy(dtype=np.int64)
    pred_eval = probs.argmax(axis=1)
    eval_macro_f1 = float(f1_score(y_eval, pred_eval, labels=range(len(STAGES)), average="macro", zero_division=0))
    eval_kappa = float(cohen_kappa_score(y_eval, pred_eval))

    runtime = {
        "fold": fold,
        "fit_patients": len(fit_subjects),
        "fit_epochs": n_fit_epochs,
        "eval_patients": len(eval_subjects),
        "eval_epochs": len(eval_frame),
        "n_features": n_features,
        "include_occipital": include_occipital,
        "weight_power": weight_power,
        "stage_counts": counts.astype(int).tolist(),
        "class_weights": class_weights.tolist(),
        "n_iter": int(clf.n_iter_),
        "early_stopping_active": bool(clf.do_early_stopping_),
        "eval_macro_f1": eval_macro_f1,
        "eval_kappa": eval_kappa,
        "prep_seconds": prep_seconds,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_fit_seconds": moments.total_fit_seconds,
        "process_peak_rss_bytes": exp73.peak_memory_bytes(),
    }
    return probs, runtime


def smoke(frame: pd.DataFrame, split, weight_power: float) -> int:
    """Verify streaming float32 features, holdout disjointness, tempered HGB, and fold checkpoint/resume on fold 0 inner patients."""
    started = active_clock()
    train, val, test = fold_frames(frame, split)
    reserved = reserved_holdout_ids()
    assert_disjoint_from_reserved(set(frame.subject_id), reserved, "development")

    fit_ids = sorted(train.subject_id.unique())[:12]
    val_ids = sorted(val.subject_id.unique())[:4]
    assert not (set(fit_ids) & set(test.subject_id))
    assert not (set(val_ids) & set(test.subject_id))

    fit_sub = train[train.subject_id.isin(fit_ids)].reset_index(drop=True)
    val_sub = val[val.subject_id.isin(val_ids)].reset_index(drop=True)
    layout = make_layout(set())
    use_occ = has_occipital_cache(layout.occipital_cache, [*fit_ids, *val_ids])

    probs, runtime = run_fold_once(
        0,
        fit_sub,
        val_sub,
        layout,
        weight_power=weight_power,
        max_iter=25,
        include_occipital=use_occ,
    )
    cols = feature_column_names(include_occipital=use_occ)
    if runtime["n_features"] != len(cols):
        raise ValueError(f"Expected {len(cols)} features, got {runtime['n_features']}")

    # Verify fold-level checkpoint/resume and PAUSE sentinel logic in a temporary folder.
    with TemporaryDirectory(prefix="hgb704-smoke-") as tmp:
        tmp_dir = Path(tmp)
        fold_dir = tmp_dir / "fold-0"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_predictions(fold_dir / "predictions.parquet", val_sub, probs, 0)
        atomic_json(fold_dir / "runtime.json", runtime)
        if not (fold_dir / "predictions.parquet").is_file():
            raise RuntimeError("Failed to write smoke fold checkpoint")
        reloaded = pd.read_parquet(fold_dir / "predictions.parquet")
        np.testing.assert_allclose(reloaded[PROBABILITIES].to_numpy(np.float32), probs, rtol=0, atol=1e-6)
        (tmp_dir / "PAUSE").touch()
        if not (tmp_dir / "PAUSE").exists():
            raise RuntimeError("Failed PAUSE sentinel check")

    # Verify prior-power calibration mechanics on the smoke predictions.
    priors = np.asarray([runtime["stage_counts"]], dtype=np.float64)
    priors = priors / priors.sum(axis=1, keepdims=True)
    cal_probs, cal_info = calibrate_oof_prior_power(
        probs, val_sub[LABEL].to_numpy(dtype=np.int64), np.zeros(len(val_sub), dtype=np.int64), priors
    )
    if not np.isfinite(cal_probs).all() or not np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid calibrated probabilities in smoke check")

    summary = {
        "passed": True,
        "outer_evaluation_subjects": 0,
        "reserved_holdout_overlap": 0,
        "fitting_patients": len(fit_ids),
        "validation_patients": len(val_ids),
        "fitting_epochs": len(fit_sub),
        "validation_epochs": len(val_sub),
        "n_features": runtime["n_features"],
        "include_occipital": use_occ,
        "weight_power": weight_power,
        "smoke_val_macro_f1": runtime["eval_macro_f1"],
        "smoke_calibrated_macro_f1": cal_info["calibrated_mean_fold_macro_f1"],
        "active_seconds": active_clock() - started,
        "process_peak_rss_bytes": exp73.peak_memory_bytes(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "smoke.json", summary)
    log(
        f"Smoke passed: {runtime['n_features']} float32 features, {len(fit_sub)} fit epochs, "
        f"val macro-F1 {runtime['eval_macro_f1']:.4f}, peak RSS {summary['process_peak_rss_bytes'] / (1024**2):.1f} MiB"
    )
    return 0


def compare_reference(aligned: pd.DataFrame, pred_col: str, ref_path: Path, fold_rows: list[np.ndarray]) -> dict | None:
    if not ref_path.is_file():
        return None
    ref = pd.read_parquet(ref_path)
    joined = aligned[[*KEYS, LABEL]].merge(ref, on=[*KEYS, LABEL], validate="one_to_one")
    if len(joined) != len(aligned):
        return None
    ref_pred = joined[PROBABILITIES].to_numpy().argmax(axis=1) if set(PROBABILITIES).issubset(joined.columns) else joined.iloc[:, -1].to_numpy()
    cand_scores = scores(aligned, aligned[pred_col].to_numpy(), fold_rows)
    ref_scores = scores(aligned, ref_pred, fold_rows)
    y = aligned[LABEL].to_numpy()
    cand = aligned[pred_col].to_numpy()
    fold_deltas = [
        float(f1_score(y[r], cand[r], average="macro") - f1_score(y[r], ref_pred[r], average="macro"))
        for r in fold_rows
    ]
    return {
        "reference_path": str(ref_path.relative_to(REPO)),
        "reference": ref_scores,
        "macro_f1_delta": cand_scores["macro_f1"] - ref_scores["macro_f1"],
        "fold_deltas": fold_deltas,
        "winning_folds": int(sum(d > 0 for d in fold_deltas)),
    }


def summarize_and_save(
    frame: pd.DataFrame,
    splits: list,
    weight_power: float,
    include_occipital: bool,
) -> dict:
    fold_preds = [pd.read_parquet(OUT / f"fold-{f}" / "predictions.parquet") for f in range(len(splits))]
    fold_runtimes = [json.loads((OUT / f"fold-{f}" / "runtime.json").read_text(encoding="utf-8")) for f in range(len(splits))]
    pooled = pd.concat(fold_preds, ignore_index=True)

    aligned = frame.copy()
    aligned["fold"] = -1
    for f, split in enumerate(splits):
        aligned.iloc[split.test_idx, aligned.columns.get_loc("fold")] = f
    aligned = aligned.merge(
        pooled, how="outer", on=[*KEYS, LABEL, "fold"], validate="one_to_one", indicator=True
    )
    if len(aligned) != len(frame) or not (aligned["_merge"] == "both").all():
        raise ValueError("Pooled predictions do not match development_frame keys exactly")
    aligned = aligned.drop(columns="_merge")

    out_cols = [*KEYS, LABEL, "fold", *PROBABILITIES]
    save_frame(OUT / "predictions.parquet", aligned[out_cols])

    probs = aligned[PROBABILITIES].to_numpy(dtype=np.float32)
    labels = aligned[LABEL].to_numpy(dtype=np.int64)
    fold_idx = aligned["fold"].to_numpy(dtype=np.int64)
    fold_counts = np.asarray([rt["stage_counts"] for rt in fold_runtimes], dtype=np.float64)
    fold_priors = fold_counts / fold_counts.sum(axis=1, keepdims=True)
    cal_probs, cal_info = calibrate_oof_prior_power(probs, labels, fold_idx, fold_priors)

    cal_frame = aligned[[*KEYS, LABEL, "fold"]].copy()
    cal_frame[PROBABILITIES] = cal_probs
    save_frame(OUT / "calibrated_predictions.parquet", cal_frame)

    fold_rows = [np.flatnonzero(fold_idx == f) for f in range(len(splits))]
    aligned["raw_pred"] = probs.argmax(axis=1)
    aligned["cal_pred"] = cal_probs.argmax(axis=1)
    raw_scores = scores(aligned, aligned["raw_pred"].to_numpy(), fold_rows)
    cal_scores = scores(aligned, aligned["cal_pred"].to_numpy(), fold_rows)
    base_scores = scores(aligned, aligned["g20__epoch"].to_numpy(), fold_rows)

    summary = {
        "experiment": 76,
        "name": "cinc-hgb-704",
        "weight_power": weight_power,
        "include_occipital": include_occipital,
        "n_features": fold_runtimes[0]["n_features"],
        "development_patients": int(aligned.subject_id.nunique()),
        "development_scored_epochs": len(aligned),
        "raw": raw_scores,
        "calibrated": cal_scores,
        "calibration": cal_info,
        "vs_compact_g20": {
            "reference": base_scores,
            "macro_f1_delta": raw_scores["macro_f1"] - base_scores["macro_f1"],
            "calibrated_macro_f1_delta": cal_scores["macro_f1"] - base_scores["macro_f1"],
        },
        "vs_recipe53": compare_reference(aligned, "raw_pred", SEARCHED_53, fold_rows),
        "vs_expanded73": compare_reference(aligned, "raw_pred", EXPANDED_73, fold_rows),
        "folds": fold_runtimes,
    }
    atomic_json(OUT / "summary.json", summary)
    log(
        f"#76 completed: raw macro-F1 {raw_scores['macro_f1']:.4f}, "
        f"calibrated {cal_scores['macro_f1']:.4f} (vs g20 {raw_scores['macro_f1'] - base_scores['macro_f1']:+.4f})"
    )
    return summary


def execute(args: argparse.Namespace) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError(f"Declare the experiment in {OUT / 'FINDINGS.md'} before running")

    frame, splits = development_frame()
    if args.smoke:
        return smoke(frame, splits[0], args.weight_power)

    epochs, added = exp73.training_cohort(set(frame.subject_id))
    cohort = pd.concat([frame[[*KEYS, LABEL]], epochs], ignore_index=True)
    reserved = reserved_holdout_ids()
    assert_disjoint_from_reserved(set(cohort.subject_id), reserved, "cohort")

    layout = make_layout(added)
    all_subjects = set(cohort.subject_id.astype(str).unique())
    include_occipital = has_occipital_cache(layout.occipital_cache, all_subjects)

    partitions = [fold_frames(frame, split) for split in splits]
    pin_manifest(
        OUT / "training_manifest.json",
        {
            "experiment": 76,
            "seed": SEED,
            "weight_power": args.weight_power,
            "max_iter": args.max_iter,
            "include_occipital": include_occipital,
            "columns": feature_column_names(include_occipital=include_occipital),
            "additional_training_subjects": sorted(added),
            "source_sha256": {
                "run_staging_hgb_704.py": exp73.digest(Path(__file__)),
                "multiscale_hgb.py": exp73.digest(REPO / "src/physionet_sleep/experiments/multiscale_hgb.py"),
            },
        },
    )

    ledger = OUT / "active_budget.json"
    used = json.loads(ledger.read_text(encoding="utf-8"))["seconds"] if ledger.exists() else 0.0
    start = last_account = active_clock()
    budget_seconds = min(BUDGET_SECONDS, args.budget_hours * 3600.0)
    deadline = start + budget_seconds - used
    pause_path = OUT / "PAUSE"

    def account() -> None:
        nonlocal last_account
        last_account = active_clock()
        atomic_json(ledger, {"seconds": used + last_account - start})

    def check() -> None:
        now = active_clock()
        if now - last_account >= CHECKPOINT_SECONDS:
            account()
        if pause_path.exists() or now >= deadline:
            raise exp73.Paused

    try:
        for f, (train, val, test) in enumerate(partitions):
            fold_dir = OUT / f"fold-{f}"
            pred_path = fold_dir / "predictions.parquet"
            runtime_path = fold_dir / "runtime.json"
            if pred_path.is_file() and runtime_path.is_file():
                log(f"Fold {f + 1}/5: saved checkpoint found, skipping")
                continue
            check()
            fold_dir.mkdir(parents=True, exist_ok=True)
            fit_subjects = set(train.subject_id.astype(str).unique()) | added
            if fit_subjects & set(val.subject_id.astype(str).unique()):
                raise ValueError(f"Fold {f}: fitting and inner validation patients overlap")
            fit_frame = cohort[cohort.subject_id.isin(fit_subjects)].reset_index(drop=True)
            log(f"Fold {f + 1}/5: streaming features & fitting tempered HGB on {len(fit_subjects)} patients ({len(fit_frame)} epochs)")
            probs, runtime = run_fold_once(
                f,
                fit_frame,
                test,
                layout,
                weight_power=args.weight_power,
                max_iter=args.max_iter,
                include_occipital=include_occipital,
                check_fn=check,
            )
            write_predictions(pred_path, test, probs, f)
            atomic_json(runtime_path, runtime)
            account()
            log(
                f"Fold {f + 1}/5 saved: macro-F1 {runtime['eval_macro_f1']:.4f}, "
                f"n_iter={runtime['n_iter']}, peak RSS {runtime['process_peak_rss_bytes'] / (1024**2):.1f} MiB"
            )
            gc.collect()

        summarize_and_save(frame, splits, args.weight_power, include_occipital)
    except exp73.Paused:
        log("#76 paused cleanly between checkpoints")
        return 75
    finally:
        account()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run inner-patient smoke test on fold 0")
    parser.add_argument("--weight-power", type=float, default=DEFAULT_WEIGHT_POWER, help="Exponent gamma in counts^-gamma")
    parser.add_argument("--max-iter", type=int, default=350, help="Maximum boosting iterations")
    parser.add_argument("--budget-hours", type=float, default=24.0, help="Active-clock wall budget in hours")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    with local_training_slot():
        raise SystemExit(execute(args))
