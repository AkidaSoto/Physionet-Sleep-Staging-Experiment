"""Experiment 82: Feature-pruned Concept-BiGRU variants (<= 20 features) vs #79."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, log_loss

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_all_additional_training as additional
import run_staging_bigru_concept as exp79
import run_staging_bigru_unified as unified
import run_staging_feature_context as sequence
from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from physionet_sleep.experiments.concept_bigru_staging import (
    CONCEPT_NAMES,
    FEATURE_NAMES_79,
    STEP_PER_EPOCH,
    evaluate_conformal_oof,
    fit_prior_power_oof,
)
from physionet_sleep.experiments.waveform_clock import active_clock
from run_waveform_development import (
    BUDGET_SECONDS,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    SEED,
    STAGES,
    atomic_json,
    development_frame,
    fold_frames,
    log,
)

OUT = EXPLORATION / "cinc-bigru-pruned"

# Each variant selects a strict subset of #79's 29 features (81 inputs) grouped into the 4 concepts.
# All 3 variants have <= 20 distinct features AND drop all non-causal multi-minute/night-clock markers!
VARIANTS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    # 18 features (70 inputs): all 13 6-s tracks + 5 single-epoch AASM markers (zero rolling/clock heuristics)
    "core18": {
        "slow_wave": (
            ("spindle_features.delta_power", "spindle_features.aperiodic_exponent"),
            ("slow_wave_75_max_clean",),
        ),
        "n2_events": (
            ("c_rel_sigma", "sigma_ratio"),
            ("spindle_bursts", "kc_shape_events"),
        ),
        "wake_n1": (
            ("o_rel_alpha", "c_rel_alpha", "c_rel_theta", "o_rel_theta", "c_rel_beta", "c_log_power"),
            (),
        ),
        "rem_tone": (
            ("eog_high", "eye_movement_activity.eye_movement_activity", "eye_movement_activity.eye_movement_baseline"),
            ("epoch_ratio_features.eog_rem_vs_sem_db", "epoch_ratio_features.emg_floor_db"),
        ),
    },
    # 13 features (65 inputs): pure 6-s tracks only, zero 30-s event detectors or rolling markers
    "tracks13": {
        "slow_wave": (
            ("spindle_features.delta_power", "spindle_features.aperiodic_exponent"),
            (),
        ),
        "n2_events": (
            ("c_rel_sigma", "sigma_ratio"),
            (),
        ),
        "wake_n1": (
            ("o_rel_alpha", "c_rel_alpha", "c_rel_theta", "o_rel_theta", "c_rel_beta", "c_log_power"),
            (),
        ),
        "rem_tone": (
            ("eog_high", "eye_movement_activity.eye_movement_activity", "eye_movement_activity.eye_movement_baseline"),
            (),
        ),
    },
    # 13 features (49 inputs): ultra-lean non-overlapping cues (9 tracks + 4 single-epoch AASM markers)
    "minimal13": {
        "slow_wave": (
            ("spindle_features.delta_power", "spindle_features.aperiodic_exponent"),
            ("slow_wave_75_max_clean",),
        ),
        "n2_events": (
            ("c_rel_sigma",),
            ("spindle_bursts", "kc_shape_events"),
        ),
        "wake_n1": (
            ("o_rel_alpha", "c_rel_theta", "c_rel_beta", "c_log_power"),
            (),
        ),
        "rem_tone": (
            ("eog_high", "eye_movement_activity.eye_movement_activity"),
            ("epoch_ratio_features.emg_floor_db",),
        ),
    },
}


def variant_indices(name: str) -> tuple[list[int], tuple[int, ...], int]:
    """Map a variant's (step_signals, epoch_signals) onto #79's 81-column FEATURE_NAMES_79."""
    spec = VARIANTS[name]
    col_to_idx = {col: i for i, col in enumerate(FEATURE_NAMES_79)}
    indices: list[int] = []
    dims: list[int] = []
    n_distinct = 0
    for concept in CONCEPT_NAMES:
        step_sigs, epoch_sigs = spec[concept]
        c_idx: list[int] = []
        for sig in step_sigs:
            for step in range(STEP_PER_EPOCH):
                key = f"{sig}@+{step * 6}s"
                if key not in col_to_idx:
                    raise KeyError(f"Missing step column {key} in FEATURE_NAMES_79")
                c_idx.append(col_to_idx[key])
        for sig in epoch_sigs:
            if sig not in col_to_idx:
                raise KeyError(f"Missing epoch column {sig} in FEATURE_NAMES_79")
            c_idx.append(col_to_idx[sig])
        n_distinct += len(step_sigs) + len(epoch_sigs)
        dims.append(len(c_idx))
        indices.extend(c_idx)
    return indices, tuple(dims), n_distinct


def slice_recordings(
    recordings: dict[str, sequence.Recording],
    indices: list[int],
) -> dict[str, sequence.Recording]:
    return {
        sub: replace(rec, features=rec.features[:, indices].contiguous())
        for sub, rec in recordings.items()
    }


def score_metrics(pred_df: pd.DataFrame) -> dict:
    y = pred_df[LABEL].to_numpy(dtype=np.int64)
    p = pred_df["prediction"].to_numpy(dtype=np.int64)
    probs = pred_df[PROBABILITIES].to_numpy(dtype=np.float64)
    f1_per = f1_score(y, p, labels=list(range(len(STAGES))), average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(f1_per)),
        "kappa": float(cohen_kappa_score(y, p)),
        "accuracy": float((y == p).mean()),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-7, 1.0), labels=list(range(len(STAGES))))),
        "stage_f1": {st: float(f1_per[i]) for i, st in enumerate(STAGES)},
    }


def finalize_variant_5fold(vname: str, config: exp79.TrainConfig) -> dict:
    """Aggregate all 5 folds into summary.json, calibrated_predictions.parquet, and conformal_summary.json."""
    indices, c_dims, n_distinct = variant_indices(vname)
    vroot = OUT / vname
    fold_preds, fold_comps, fold_concepts = [], [], []
    fold_histories: dict[str, dict] = {}
    param_count = 0
    for fold in range(5):
        fdir = vroot / f"fold-{fold}"
        fold_preds.append(pd.read_parquet(fdir / "predictions.parquet"))
        fold_comps.append(pd.read_parquet(fdir / "components.parquet"))
        fold_concepts.append(pd.read_parquet(fdir / "concept_logits.parquet"))
        hist = json.loads((fdir / "history.json").read_text(encoding="utf-8"))
        fold_histories[str(fold)] = hist
        param_count = int(hist.get("param_count", param_count))

    oof = pd.concat(fold_preds, ignore_index=True)
    comp_df = pd.concat(fold_comps, ignore_index=True)
    concept_df = pd.concat(fold_concepts, ignore_index=True)

    y = oof[LABEL].to_numpy()
    probs = oof[PROBABILITIES].to_numpy()
    folds_col = oof["fold"].to_numpy()
    subjects = oof["subject_id"].astype(str).to_numpy()

    cal_probs, cal_details = fit_prior_power_oof(probs, y, folds_col)
    cal_df = oof.copy()
    cal_df["prediction"] = cal_probs.argmax(axis=1).astype(np.int64)
    for idx, col in enumerate(PROBABILITIES):
        cal_df[col] = cal_probs[:, idx].astype(np.float32)

    unified.save_frame(vroot / "predictions.parquet", oof)
    unified.save_frame(vroot / "calibrated_predictions.parquet", cal_df)

    fold_masks = [folds_col == f for f in sorted(np.unique(folds_col).tolist())]
    raw_metrics = unified._stage_metrics(y, probs, fold_masks)
    cal_metrics = unified._stage_metrics(y, cal_probs, fold_masks)

    is_trans = exp79._transition_flags(oof)
    conformal = {
        "raw": evaluate_conformal_oof(probs, y, folds_col, is_trans, stages=tuple(STAGES)),
        "calibrated": evaluate_conformal_oof(cal_probs, y, folds_col, is_trans, stages=tuple(STAGES)),
    }
    atomic_json(vroot / "conformal_summary.json", conformal)

    concept_diag = exp79._compute_concept_diagnostics(comp_df, concept_df)

    # Paired patient bootstrap comparisons vs #79, #75, #73
    rng = np.random.default_rng(SEED)
    cm_new_raw, pats_raw = patient_confusions(subjects, y, probs.argmax(axis=1))
    cm_new_cal, pats_cal = patient_confusions(subjects, y, cal_probs.argmax(axis=1))
    assert (pats_raw == pats_cal).all()
    pat_fold = np.array([folds_col[subjects == pt][0] for pt in pats_raw])
    f_idx = [np.flatnonzero(pat_fold == f) for f in sorted(np.unique(folds_col).tolist())]
    draws = cv_fold_draws(f_idx, 2000, rng)
    sc_new_raw = cv_scores(cm_new_raw, f_idx, draws)
    sc_new_cal = cv_scores(cm_new_cal, f_idx, draws)
    comparisons = {}
    for ref_name, ref_path in (
        ("vs_79_calibrated", exp79.OUT / "calibrated_predictions.parquet"),
        ("vs_75_calibrated", exp79.REFERENCE_75),
        ("vs_73", exp79.REFERENCE_73),
    ):
        if not ref_path.exists():
            continue
        ref = pd.read_parquet(ref_path)
        if "prediction" not in ref.columns:
            ref["prediction"] = ref[PROBABILITIES].to_numpy().argmax(axis=1)
        merged = oof[[*KEYS, LABEL, "fold"]].merge(
            ref[[*KEYS, "prediction"]], on=[*KEYS], how="inner"
        )
        if len(merged) != len(oof):
            continue
        cm_ref, pats_ref = patient_confusions(
            merged["subject_id"].astype(str).to_numpy(),
            merged[LABEL].to_numpy(),
            merged["prediction"].to_numpy(),
        )
        if not (pats_ref == pats_raw).all():
            continue
        sc_ref = cv_scores(cm_ref, f_idx, draws)
        d_raw = sc_new_raw["macro_f1"] - sc_ref["macro_f1"]
        d_cal = sc_new_cal["macro_f1"] - sc_ref["macro_f1"]
        comparisons[ref_name] = {
            "raw_macro_f1_delta": {"mean": float(d_raw.mean()), "ci95": interval(d_raw)},
            "calibrated_macro_f1_delta": {"mean": float(d_cal.mean()), "ci95": interval(d_cal)},
        }

    summary = {
        "experiment": 82,
        "name": f"cinc-bigru-pruned-{vname}",
        "variant": vname,
        "n_features": int(n_distinct),
        "config": asdict(config),
        "model_params": int(param_count),
        "input_dim": int(sum(c_dims)),
        "concept_dims": {k: int(v) for k, v in zip(CONCEPT_NAMES, c_dims)},
        "feature_names": [FEATURE_NAMES_79[i] for i in indices],
        "feature_indices_79": indices,
        "raw": raw_metrics,
        "calibrated": cal_metrics,
        "calibration_oof_details": cal_details,
        "conformal": conformal,
        "concept_diagnostics": concept_diag,
        "comparisons": comparisons,
        "fold_histories": fold_histories,
    }
    atomic_json(vroot / "summary.json", summary)
    log(
        f"EXPERIMENT 82 [{vname}] 5-FOLD COMPLETE: params={param_count}, "
        f"n_features={n_distinct}, input_dim={sum(c_dims)}, "
        f"raw macro-F1={raw_metrics['mean_fold_macro_f1']:.5f}, "
        f"cal macro-F1={cal_metrics['mean_fold_macro_f1']:.5f}, "
        f"kappa={cal_metrics['kappa']:.5f}, "
        f"local-only={concept_diag['local_only_macro_f1']:.5f}"
    )
    return summary


def run_screen(
    variants: list[str],
    folds: list[int],
    *,
    smoke: bool = False,
    finalize: bool = False,
) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pause_path = OUT / "PAUSE"

    dev_all, splits = development_frame()
    added_frame, eligible_added = additional.training_cohort(
        set(dev_all.subject_id.astype(str))
    )
    added_subjects = {str(s) for s in eligible_added}
    unified.assert_holdout_disjoint(
        set(dev_all.subject_id.astype(str).unique()) | added_subjects
    )

    ref79_f0 = pd.read_parquet(exp79.OUT / "fold-0" / "predictions.parquet")
    ref79_f0_metrics = score_metrics(ref79_f0)
    log(
        f"#79 Fold-0 raw baseline: macro_f1={ref79_f0_metrics['macro_f1']:.5f}, "
        f"kappa={ref79_f0_metrics['kappa']:.5f}, log_loss={ref79_f0_metrics['log_loss']:.5f}"
    )

    config = exp79.TrainConfig(max_passes=2) if smoke else exp79.TrainConfig()
    deadline = active_clock() + (600.0 if smoke else BUDGET_SECONDS * 2.0)
    results_path = OUT / "fold0_screen.json"
    summary: dict = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.is_file() and not smoke
        else {"ref79_fold0_raw": ref79_f0_metrics, "variants": {}}
    )

    for fold in folds:
        # Skip loading recordings if every requested variant is already complete for this fold.
        if not smoke and all((OUT / v / f"fold-{fold}" / "predictions.parquet").exists() for v in variants):
            log(f"Fold {fold}: all requested variants already complete, skipping load.")
            continue

        split = splits[fold]
        train, val, test = fold_frames(dev_all, split)
        if smoke:
            fit_subs = sorted(train.subject_id.astype(str).unique())[:6]
            val_subs = sorted(val.subject_id.astype(str).unique())[:2]
            test = test[test.subject_id.astype(str).isin(sorted(test.subject_id.astype(str).unique())[:2])]
            fold_frame = pd.concat(
                [
                    train[train.subject_id.astype(str).isin(fit_subs)],
                    val[val.subject_id.astype(str).isin(val_subs)],
                    test,
                ],
                ignore_index=True,
            )
            fit_set = set(fit_subs)
        else:
            full_fit = pd.concat([train, added_frame.copy()], ignore_index=True)
            fit_subs = sorted(full_fit.subject_id.astype(str).unique())
            val_subs = sorted(val.subject_id.astype(str).unique())
            fold_frame = pd.concat([full_fit, val, test], ignore_index=True)
            fit_set = set(fit_subs)

        log(
            f"Fold {fold}: loading 81-col #79 recordings ONCE for {len(fit_subs)} fit + "
            f"{len(val_subs)} val + {test.subject_id.nunique()} test..."
        )
        base_recordings, _ = exp79.prepare_concept_recordings(
            fold_frame,
            fit_set,
            lambda: None,
            added=added_subjects if not smoke else set(),
            occipital_dir=exp79.OCCIPITAL_CACHE,
        )

        for vname in variants:
            indices, c_dims, n_distinct = variant_indices(vname)
            vdir = OUT / vname / f"fold-{fold}"
            pred_file = vdir / "predictions.parquet"
            if pred_file.exists() and not smoke:
                log(f"[{vname} fold-{fold}] already complete, loading cached predictions.")
                pred_df = pd.read_parquet(pred_file)
                metrics = score_metrics(pred_df)
            else:
                sliced = slice_recordings(base_recordings, indices)
                log(
                    f"[{vname} fold-{fold}] training ({n_distinct} features, {sum(c_dims)} inputs, "
                    f"dims={c_dims})..."
                )
                model, state = exp79.fit_concept_bigru(
                    sliced,
                    fit_subs,
                    val_subs,
                    SEED + fold,
                    vdir,
                    pause_path,
                    deadline,
                    config,
                    concept_dims=c_dims,
                    concept_names=CONCEPT_NAMES,
                )
                if model is None:
                    log(f"[{vname} fold-{fold}] paused/timed out cleanly.")
                    return 75
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                pred_df, comp_df, concept_df = exp79.score_fold_with_concepts(
                    model, sliced, test, fold
                )
                if not smoke:
                    unified.save_frame(pred_file, pred_df)
                    unified.save_frame(vdir / "components.parquet", comp_df)
                    unified.save_frame(vdir / "concept_logits.parquet", concept_df)
                    atomic_json(
                        vdir / "history.json",
                        {
                            "variant": vname,
                            "n_features": n_distinct,
                            "input_dim": int(sum(c_dims)),
                            "concept_dims": dict(zip(CONCEPT_NAMES, c_dims)),
                            "param_count": int(param_count),
                            "best_val": float(state["best"]),
                            "best_pass": int(state["best_pass"]),
                            "pass_seconds": float(state["pass_seconds"]),
                        },
                    )
                metrics = score_metrics(pred_df)
                metrics["param_count"] = int(param_count)
                del sliced, model
                gc.collect()

            delta = metrics["macro_f1"] - ref79_f0_metrics["macro_f1"]
            log(
                f"[{vname} fold-{fold}] macro_f1={metrics['macro_f1']:.5f} "
                f"(delta vs #79_f0={delta:+.5f}), kappa={metrics['kappa']:.5f}, "
                f"N1={metrics['stage_f1']['N1']:.4f}, N3={metrics['stage_f1']['N3']:.4f}, "
                f"REM={metrics['stage_f1']['REM']:.4f}"
            )
            if not smoke and fold == 0:
                summary["variants"].setdefault(vname, {})[f"fold_{fold}"] = {
                    "n_features": n_distinct,
                    "input_dim": int(sum(c_dims)),
                    "concept_dims": dict(zip(CONCEPT_NAMES, c_dims)),
                    **metrics,
                    "delta_vs_79_fold0": float(delta),
                }
                atomic_json(results_path, summary)
        del base_recordings
        gc.collect()

    if finalize and not smoke:
        for vname in variants:
            if all((OUT / vname / f"fold-{f}" / "predictions.parquet").exists() for f in range(5)):
                finalize_variant_5fold(vname, config)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=[*VARIANTS.keys(), "all"], default="all")
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    return run_screen(variants, args.folds, smoke=args.smoke, finalize=args.finalize)


if __name__ == "__main__":
    raise SystemExit(main())
