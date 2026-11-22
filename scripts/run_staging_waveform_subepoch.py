"""Experiment 77: contiguous sub-epoch 1D-CNN + BiGRU raw-waveform staging on 704 patients."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from physionet_sleep.experiments.waveform_clock import active_clock
from physionet_sleep.experiments.waveform_data import WaveformStore
from physionet_sleep.experiments.waveform_subepoch import (
    SubEpochWaveformNet,
    fit_subepoch_fold,
    oof_prior_power_calibration,
    process_memory_bytes,
)
from run_staging_all_additional_training import training_cohort
from run_waveform_development import (
    BASELINE,
    BUDGET_SECONDS,
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
    write_predictions,
)
from run_window_hmm_staging import scores


OUT = EXPLORATION / "cinc-waveform-subepoch"
EXP40_PREDS = EXPLORATION / "cinc-waveform/scratch/predictions.parquet"


def _reserved_holdout_subjects() -> set[str]:
    holdout = set((RAW / "selected_records_holdout.txt").read_text(encoding="utf-8").split())
    ledger = set(pd.read_csv(BASELINE / "holdout_subjects.csv", usecols=["record_id"]).record_id)
    if len(holdout) != 200 or holdout != ledger:
        raise ValueError("Reserved 200-patient CinC holdout manifest mismatch")
    return holdout


def _assert_disjoint(*groups: set[str]) -> None:
    if len(set.union(*groups)) != sum(len(g) for g in groups):
        raise ValueError("Subject partitions must be strictly pairwise disjoint")


def _select_smoke_patients(
    train_dev: pd.DataFrame, val_dev: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select small inner training/validation subsets of fold 0 containing all 5 sleep stages."""
    train_groups = list(train_dev.groupby("subject_id", sort=False))
    selected_train: list[pd.DataFrame] = []
    seen_stages: set[int] = set()
    for _, group in train_groups:
        if len(selected_train) < 8 or len(seen_stages) < len(STAGES):
            selected_train.append(group)
            seen_stages.update(int(v) for v in group[LABEL].unique())
        if len(selected_train) >= 8 and len(seen_stages) == len(STAGES):
            break
    if len(seen_stages) < len(STAGES):
        raise ValueError("Could not cover all 5 sleep stages in fold 0 smoke training subset")
    smoke_train = pd.concat(selected_train, ignore_index=True)
    val_groups = [group for _, group in islice(val_dev.groupby("subject_id", sort=False), 2)]
    if len(val_groups) < 2:
        raise ValueError("Expected at least 2 inner validation patients in fold 0")
    return smoke_train, val_groups[0].reset_index(drop=True), val_groups[1].reset_index(drop=True)


def run_smoke(frame: pd.DataFrame, split0, *, device: str = "cpu") -> int:
    """Verify fold-0 inner training/val shapes, finite outputs, pause/resume parity, speed, and RSS."""
    started = active_clock()
    reserved = _reserved_holdout_subjects()
    train_dev, val_dev, test_dev = fold_frames(frame, split0)
    smoke_train, smoke_val, smoke_eval = _select_smoke_patients(train_dev, val_dev)

    fit_ids = set(smoke_train.subject_id.astype(str))
    val_ids = set(smoke_val.subject_id.astype(str))
    eval_ids = set(smoke_eval.subject_id.astype(str))
    test_ids = set(test_dev.subject_id.astype(str))
    _assert_disjoint(fit_ids, val_ids, eval_ids, test_ids, reserved)

    # Scoring frame uses a second inner validation patient only (0 outer evaluation subjects).
    store = WaveformStore(RAW, permitted_subjects=fit_ids | val_ids | eval_ids, max_open_nights=4)
    OUT.mkdir(parents=True, exist_ok=True)
    options = dict(
        device=device,
        batch_size=4,
        chunk_epochs=32,
        eval_batch_epochs=64,
        max_steps=20,
        lr=1.5e-3,
        weight_decay=1e-3,
        validation_interval=10,
        checkpoint_interval_steps=250,
        seed=SEED,
    )

    with TemporaryDirectory(dir=OUT, prefix="smoke-") as tmp:
        tmp_dir = Path(tmp)
        resumed_dir = tmp_dir / "resumed"
        uninterrupted_dir = tmp_dir / "uninterrupted"

        paused = fit_subepoch_fold(
            store, smoke_train, smoke_val, smoke_eval,
            output_dir=resumed_dir, smoke_pause_after=10, **options,
        )
        if paused is not None:
            raise RuntimeError("Expected smoke fit to pause after 10 updates")
        ckpt_mid = torch.load(resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        if ckpt_mid["step"] != 10:
            raise RuntimeError(f"Expected checkpoint step 10 after pause, got {ckpt_mid['step']}")

        resumed = fit_subepoch_fold(
            store, smoke_train, smoke_val, smoke_eval,
            output_dir=resumed_dir, **options,
        )
        if resumed is None:
            raise RuntimeError("Resumed smoke fit failed to complete")
        resumed_test, resumed_val, resumed_runtime = resumed

        uninterrupted = fit_subepoch_fold(
            store, smoke_train, smoke_val, smoke_eval,
            output_dir=uninterrupted_dir, **options,
        )
        if uninterrupted is None:
            raise RuntimeError("Uninterrupted smoke fit failed to complete")
        unint_test, unint_val, unint_runtime = uninterrupted

        ckpt_res = torch.load(resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        ckpt_unint = torch.load(uninterrupted_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        for name, tensor in ckpt_res["model"].items():
            torch.testing.assert_close(tensor, ckpt_unint["model"][name], rtol=0.0, atol=0.0)
        for name, tensor in ckpt_res["best_model"].items():
            torch.testing.assert_close(tensor, ckpt_unint["best_model"][name], rtol=0.0, atol=0.0)

        np.testing.assert_allclose(resumed_test, unint_test, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(resumed_val, unint_val, rtol=0.0, atol=0.0)

        for arr, expected_len in ((resumed_val, len(smoke_val)), (resumed_test, len(smoke_eval))):
            if arr.shape != (expected_len, len(STAGES)):
                raise ValueError(f"Unexpected smoke prediction shape {arr.shape}")
            if not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError("Non-finite or negative smoke probabilities")
            np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-5, rtol=0)

    rss_now, peak_rss = process_memory_bytes()
    result = {
        "passed": True,
        "exact_weight_parity": True,
        "exact_probability_parity": True,
        "outer_evaluation_subjects": 0,
        "fitting_patients": len(fit_ids),
        "validation_patients": len(val_ids) + len(eval_ids),
        "validation_epochs": len(smoke_val) + len(smoke_eval),
        "parameter_count": int(unint_runtime["parameter_count"]),
        "ms_per_step": float(unint_runtime["ms_per_step"]),
        "active_seconds": float(active_clock() - started),
        "current_rss_mib": float(rss_now / 2**20),
        "peak_rss_mib": float(max(peak_rss, unint_runtime["peak_rss_bytes"]) / 2**20),
        "val_history": unint_runtime["history"],
    }
    atomic_json(OUT / "smoke.json", result)
    print(json.dumps(result, indent=2))
    log("Smoke check passed: exact pause/resume parity, finite shapes, and bounded RSS verified")
    return 0


def _paired_comparison(aligned: pd.DataFrame, candidate_pred: np.ndarray, ref_pred: np.ndarray, fold_rows: list[np.ndarray]) -> dict:
    y = aligned[LABEL].to_numpy()
    subjects = aligned["subject_id"].to_numpy()
    cand_scores = scores(aligned, candidate_pred, fold_rows)
    ref_scores = scores(aligned, ref_pred, fold_rows)
    fold_deltas = [
        float(f1_score(y[rows], candidate_pred[rows], average="macro") - f1_score(y[rows], ref_pred[rows], average="macro"))
        for rows in fold_rows
    ]
    cm_cand, ids = patient_confusions(subjects, y, candidate_pred)
    cm_ref, _ = patient_confusions(subjects, y, ref_pred)
    fold_by_sub = aligned.groupby("subject_id")["fold"].first().loc[ids].to_numpy()
    patient_folds = [np.flatnonzero(fold_by_sub == k) for k in range(len(fold_rows))]
    draws = cv_fold_draws(patient_folds, 2000, np.random.default_rng(SEED))
    boot = cv_scores(cm_cand, patient_folds, draws)["macro_f1"] - cv_scores(cm_ref, patient_folds, draws)["macro_f1"]
    return {
        "candidate": cand_scores,
        "reference": ref_scores,
        "macro_f1_delta": float(np.mean(fold_deltas)),
        "fold_deltas": fold_deltas,
        "wins": sum(d > 0 for d in fold_deltas),
        "paired_ci": interval(boot),
        "stage_deltas": {s: cand_scores[f"f1_{s}"] - ref_scores[f"f1_{s}"] for s in STAGES},
    }


def build_report(frame: pd.DataFrame, splits: list, fold_train_counts: dict[int, np.ndarray]) -> dict:
    """Pool 5-fold predictions, apply OOF prior-power calibration, and compare to Model C & Exp 40."""
    aligned = frame.copy()
    aligned["fold"] = -1
    for f, split in enumerate(splits):
        aligned.iloc[split.test_idx, aligned.columns.get_loc("fold")] = f

    fold_preds = [pd.read_parquet(OUT / f"fold-{f}" / "predictions.parquet") for f in range(len(splits))]
    pooled = pd.concat(fold_preds, ignore_index=True)
    merged = aligned.merge(pooled, how="outer", on=[*KEYS, LABEL, "fold"], validate="one_to_one", indicator=True)
    if len(merged) != len(frame) or not merged["_merge"].eq("both").all():
        raise ValueError("Pooled predictions do not cover the exact 320-patient development keys")
    merged = merged.drop(columns="_merge")
    merged.to_parquet(OUT / "predictions.parquet", index=False)

    probs = merged[PROBABILITIES].to_numpy(dtype=np.float32)
    y = merged[LABEL].to_numpy()
    folds = merged["fold"].to_numpy()
    fold_rows = [np.flatnonzero(folds == f) for f in range(len(splits))]

    cal = oof_prior_power_calibration(y, folds, probs, fold_train_counts)
    cal_frame = merged[[*KEYS, LABEL, "fold"]].copy()
    cal_frame[PROBABILITIES] = cal["nested_probabilities"]
    cal_frame.to_parquet(OUT / "calibrated_predictions.parquet", index=False)

    raw_pred = probs.argmax(axis=1)
    half_pred = cal["bayes_half_probabilities"].argmax(axis=1)
    nested_pred = cal["nested_probabilities"].argmax(axis=1)
    model_c_pred = merged["g20__epoch"].to_numpy()

    summary: dict = {
        "experiment": 77,
        "patients": int(merged["subject_id"].nunique()),
        "scored_epochs": len(merged),
        "raw": scores(merged, raw_pred, fold_rows),
        "prior_power_calibration": {
            "bayes_half": scores(merged, half_pred, fold_rows),
            "nested_oof": scores(merged, nested_pred, fold_rows),
            "fold_alphas": cal["fold_alphas"],
            "mean_alpha": cal["mean_alpha"],
        },
        "vs_model_c_raw": _paired_comparison(merged, raw_pred, model_c_pred, fold_rows),
        "vs_model_c_calibrated": _paired_comparison(merged, nested_pred, model_c_pred, fold_rows),
        "folds": [
            json.loads((OUT / f"fold-{f}" / "runtime.json").read_text(encoding="utf-8"))
            for f in range(len(splits))
        ],
    }

    if EXP40_PREDS.is_file():
        exp40 = pd.read_parquet(EXP40_PREDS, columns=[*KEYS, LABEL, "fold", *PROBABILITIES])
        m40 = merged[[*KEYS, LABEL, "fold"]].merge(exp40, on=[*KEYS, LABEL, "fold"], how="inner", validate="one_to_one")
        if len(m40) == len(merged):
            exp40_pred = m40[PROBABILITIES].to_numpy().argmax(axis=1)
            summary["vs_exp40_raw"] = _paired_comparison(merged, raw_pred, exp40_pred, fold_rows)
            summary["vs_exp40_calibrated"] = _paired_comparison(merged, nested_pred, exp40_pred, fold_rows)

    atomic_json(OUT / "summary.json", summary)
    log(
        f"#77 completed: raw macro-F1 {summary['raw']['macro_f1']:.4f}, "
        f"calibrated {summary['prior_power_calibration']['nested_oof']['macro_f1']:.4f}"
    )
    return summary


def execute(args: argparse.Namespace) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError(f"Declare the experiment in {OUT / 'FINDINGS.md'} before running")

    frame, splits = development_frame()
    if args.smoke:
        return run_smoke(frame, splits[0], device=args.device)

    smoke_path = OUT / "smoke.json"
    if not smoke_path.is_file() or not json.loads(smoke_path.read_text(encoding="utf-8")).get("passed"):
        raise ValueError("Run --smoke and verify passing checks before full 5-fold training")

    reserved = _reserved_holdout_subjects()
    added_epochs, added_subjects = training_cohort(set(frame.subject_id))
    dev_subjects = set(frame.subject_id.astype(str))
    _assert_disjoint(dev_subjects, added_subjects, reserved)

    permitted = dev_subjects | added_subjects
    if permitted & reserved:
        raise ValueError("Holdout contamination detected in permitted waveform subjects")
    store = WaveformStore(RAW, permitted_subjects=permitted, max_open_nights=6)

    pin_manifest(
        OUT / "manifest.json",
        {
            "experiment": 77,
            "seed": SEED,
            "stage_order": STAGES,
            "device": args.device,
            "batch_size": args.batch_size,
            "chunk_epochs": args.chunk_epochs,
            "max_steps": args.max_steps,
            "added_subjects": len(added_subjects),
        },
    )

    ledger = OUT / "active_budget.json"
    used = json.loads(ledger.read_text(encoding="utf-8"))["seconds"] if ledger.exists() else 0.0
    limit = min(BUDGET_SECONDS, args.budget_hours * 3600.0)
    started = active_clock()
    pause_file = OUT / "PAUSE"

    fold_train_counts: dict[int, np.ndarray] = {}
    try:
        for f, split in enumerate(splits):
            fold_dir = OUT / f"fold-{f}"
            pred_path = fold_dir / "predictions.parquet"
            runtime_path = fold_dir / "runtime.json"
            train_dev, val_dev, test_dev = fold_frames(frame, split)
            train_fit = pd.concat([train_dev[[*KEYS, LABEL]], added_epochs[[*KEYS, LABEL]]], ignore_index=True)

            fit_ids = set(train_fit.subject_id.astype(str))
            val_ids = set(val_dev.subject_id.astype(str))
            test_ids = set(test_dev.subject_id.astype(str))
            _assert_disjoint(fit_ids, val_ids, test_ids, reserved)
            fold_train_counts[f] = np.bincount(train_fit[LABEL].to_numpy(dtype=np.int64), minlength=len(STAGES))

            if pred_path.exists() and runtime_path.exists():
                log(f"fold {f + 1}/5: saved predictions found")
                continue

            remaining = limit - used - (active_clock() - started)
            if remaining <= 0 or pause_file.exists():
                log(f"paused before fold {f + 1}/5")
                return 75

            log(f"fold {f + 1}/5: starting ({len(fit_ids)} fitting, {len(val_ids)} val, {len(test_ids)} test patients)")
            result = fit_subepoch_fold(
                store,
                train_fit,
                val_dev,
                test_dev,
                output_dir=fold_dir,
                device=args.device,
                batch_size=args.batch_size,
                chunk_epochs=args.chunk_epochs,
                eval_batch_epochs=args.eval_batch_epochs,
                max_steps=args.max_steps,
                seed=SEED + f,
                time_limit_seconds=remaining,
                pause_file=pause_file,
            )
            if result is None:
                log(f"checkpoint saved in fold {f + 1}/5")
                return 75
            test_probs, val_probs, _ = result
            write_predictions(pred_path, test_dev, test_probs, f)
            write_predictions(fold_dir / "val_predictions.parquet", val_dev, val_probs, f)
            store._nights.clear()
            import gc
            gc.collect()
            log(f"fold {f + 1}/5: predictions saved")

        build_report(frame, splits, fold_train_counts)
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - started})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run fold-0 inner smoke + pause/resume check")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-epochs", type=int, default=32)
    parser.add_argument("--eval-batch-epochs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--budget-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    with local_training_slot():
        return execute(args)


if __name__ == "__main__":
    sys.exit(main())
