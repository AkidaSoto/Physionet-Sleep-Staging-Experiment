"""Experiment 80: Training-fold Cohen's d cue screen, extraction, and ConceptBiGRU ablations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_all_additional_training as additional
import run_staging_bigru_concept as concept_run
import run_staging_bigru_unified as unified
import run_staging_feature_context as sequence
from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from physionet_sleep.analysis.epoch_windows import WindowEncoder, robust_scale
from physionet_sleep.experiments.concept_bigru_staging import (
    CONCEPT_NAMES,
    ConceptBiGRUStagingNet,
    STEP_PER_EPOCH,
    evaluate_conformal_oof,
    fit_prior_power_oof,
)
from physionet_sleep.experiments.cues_bigru_staging import (
    CUE_COLUMNS,
    extract_missing_cues_for_subject,
    get_variant_layout,
)
from physionet_sleep.experiments.waveform_clock import active_clock
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
)

OUT = EXPLORATION / "cinc-bigru-cues"
CUE_CACHE = OUT / "cue_cache"
REFERENCE_79_RAW = EXPLORATION / "cinc-bigru-concept/predictions.parquet"
REFERENCE_79_CAL = EXPLORATION / "cinc-bigru-concept/calibrated_predictions.parquet"
REFERENCE_75_CAL = EXPLORATION / "cinc-bigru-unified/calibrated_predictions.parquet"
CLASSES = np.arange(len(STAGES))
_NO_ROBUST_SCALE = {"eog_burst_guarded"}


def _build_one_subject(args: tuple[str, Path, Path | None, tuple[str, ...]]) -> tuple[str, bool]:
    subject, out_path, dev_tracks_path, permitted = args
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path, columns=["second", *CUE_COLUMNS])
            if len(df) >= 300:
                return subject, True
        except Exception:
            pass
    df = extract_missing_cues_for_subject(
        subject, RAW, dev_tracks_path, permitted_subjects=permitted
    )
    tmp = out_path.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return subject, False


def build_cue_cache(workers: int = 4) -> None:
    """Build holdout-disjoint 1-Hz cue cache across 320 dev + 474 added recordings."""
    started = perf_counter()
    CUE_CACHE.mkdir(parents=True, exist_ok=True)
    dev_all, _ = development_frame()
    dev_subjects = set(dev_all.subject_id.astype(str).unique())
    _, elig_added = additional.training_cohort(dev_subjects)
    all_subjects = sorted(dev_subjects | {str(s) for s in elig_added})
    unified.assert_holdout_disjoint(set(all_subjects))

    # Extract fast 320 dev nights first (reuses Dev tracks1s + computes E1-M2 guarded bursts),
    # then batch the 474 added nights cleanly across bounded worker processes.
    permitted_tuple = tuple(all_subjects)
    jobs: list[tuple[str, Path, Path | None, tuple[str, ...]]] = []
    for s in all_subjects:
        out_path = CUE_CACHE / f"{s}.parquet"
        dev_tracks = (CACHE / "tracks1s" / f"{s}.parquet") if s in dev_subjects else None
        jobs.append((s, out_path, dev_tracks, permitted_tuple))

    pending = [j for j in jobs if not j[1].exists()]
    log(
        f"Cue cache: {len(all_subjects)} permitted subjects "
        f"({len(all_subjects) - len(pending)} cached, {len(pending)} to extract, workers={workers})"
    )
    done = 0
    if workers <= 1:
        for job in pending:
            _build_one_subject(job)
            done += 1
            if done % 50 == 0 or done == len(pending):
                log(f"  extracted {done}/{len(pending)} ({perf_counter() - started:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_build_one_subject, job) for job in pending]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 50 == 0 or done == len(pending):
                    log(
                        f"  extracted {done}/{len(pending)} "
                        f"({perf_counter() - started:.1f}s)"
                    )
    log(f"Cue cache complete ({len(all_subjects)} files) in {perf_counter() - started:.1f}s")


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    sp = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / max(1, len(a) + len(b) - 2)
    )
    return float((np.mean(a) - np.mean(b)) / max(float(sp), 1e-9))


def run_cohen_d_screen() -> dict:
    """Evaluate 5-fold strictly outer-training Cohen's d across all candidate cues."""
    dev_all, splits = development_frame()
    unified.assert_holdout_disjoint(set(dev_all.subject_id.astype(str).unique()))

    screen_cols = [*CUE_COLUMNS, "resp_effort"]
    rows: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sub, grp in dev_all.groupby("subject_id", sort=False):
            s = str(sub)
            cue_path = CUE_CACHE / f"{s}.parquet"
            if cue_path.exists():
                cue_df = pd.read_parquet(cue_path, columns=list(CUE_COLUMNS))
            else:
                cue_df = pd.read_parquet(
                    CACHE / "tracks1s" / f"{s}.parquet",
                    columns=[c for c in CUE_COLUMNS if c != "eog_burst_guarded"],
                )
            resp_df = pd.read_parquet(
                CACHE / "tracks1s" / f"{s}.parquet", columns=["resp_effort"]
            )
            merged = cue_df.copy()
            merged["resp_effort"] = resp_df["resp_effort"].to_numpy(dtype=np.float32)[: len(merged)]
            present = [c for c in screen_cols if c in merged.columns]
            arr = np.array(merged[present].to_numpy(dtype=np.float32), copy=True)
            # Match training scaling: robust_scale continuous signals, raw fraction for eog_burst_guarded
            cont_cols = [c for c in present if c not in _NO_ROBUST_SCALE]
            if cont_cols:
                idx_cont = [present.index(c) for c in cont_cols]
                arr[:, idx_cont] = robust_scale(arr[:, idx_cont], cont_cols)
            n_ep = len(arr) // 30
            ep = np.nanmean(arr[: n_ep * 30].reshape(n_ep, 30, len(present)), axis=1)
            ei = grp["epoch_index"].to_numpy()
            sub_df = pd.DataFrame(ep[ei], columns=present)
            sub_df["subject_id"] = s
            sub_df["y"] = grp[LABEL].to_numpy()
            rows.append(sub_df)

    full = pd.concat(rows, ignore_index=True)
    contrasts = [
        ("Wake_vs_N1", 0, 1),
        ("N1_vs_N2", 1, 2),
        ("N2_vs_N3", 2, 3),
        ("N1_vs_REM", 1, 4),
        ("N2_vs_REM", 2, 4),
        ("Wake_vs_REM", 0, 4),
    ]
    report: dict[str, dict] = {}
    for col in [c for c in screen_cols if c in full.columns]:
        per_contrast = {c[0]: [] for c in contrasts}
        for split in splits:
            train, _, _ = fold_frames(dev_all, split)
            sub_df = full[full.subject_id.isin(set(train.subject_id.astype(str)))]
            x, y = sub_df[col].to_numpy(), sub_df["y"].to_numpy()
            for cname, c1, c2 in contrasts:
                per_contrast[cname].append(_cohen_d(x[y == c1], x[y == c2]))
        means = {k: round(float(np.mean(v)), 4) for k, v in per_contrast.items()}
        stds = {k: round(float(np.std(v)), 4) for k, v in per_contrast.items()}
        max_abs = round(max(abs(v) for v in means.values()), 4)
        report[col] = {
            "mean_cohen_d": means,
            "fold_std": stds,
            "max_abs_d": max_abs,
            "qualified": bool(max_abs >= 0.20),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "cohen_d_screen.json", report)
    log("Cohen's d outer-training screen saved to cohen_d_screen.json")
    return report


def _subject_variant_raw(
    subject: str,
    *,
    step_cols: list[str],
    ep_cols: list[str],
    added: set[str],
    occipital_dir: Path | None,
    cue_dir: Path,
    allow_missing_occipital: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble base concept signals plus Experiment 80 cue tracks for one night."""
    dense_base, ep_base = concept_run._subject_concept_raw(
        subject,
        added=added,
        occipital_dir=occipital_dir,
        allow_missing_occipital=allow_missing_occipital,
    )
    base_step_cols: list[str] = []
    for c in CONCEPT_NAMES:
        base_step_cols.extend(concept_run.CONCEPT_STEP_SIGNALS[c])
    base_step_pos = {col: i for i, col in enumerate(base_step_cols)}

    extra_cols = [c for c in step_cols if c not in base_step_pos]
    if extra_cols:
        cue_path = cue_dir / f"{subject}.parquet"
        if not cue_path.exists():
            raise FileNotFoundError(
                f"Missing cue cache for {subject}: {cue_path}. Run --build-cache first."
            )
        cue_df = pd.read_parquet(cue_path, columns=extra_cols)
        n_sec = min(len(dense_base), len(cue_df))
        extra_arr = np.array(cue_df[extra_cols].to_numpy(dtype=np.float32)[:n_sec], copy=True)
        cont_cols = [c for c in extra_cols if c not in _NO_ROBUST_SCALE]
        if cont_cols:
            cont_idx = [extra_cols.index(c) for c in cont_cols]
            extra_arr[:, cont_idx] = robust_scale(extra_arr[:, cont_idx], cont_cols)
        dense_base = dense_base[:n_sec]
        extra_map = {col: i for i, col in enumerate(extra_cols)}
        parts = [
            dense_base[:, base_step_pos[c]]
            if c in base_step_pos
            else extra_arr[:, extra_map[c]]
            for c in step_cols
        ]
        dense_out = np.column_stack(parts).astype(np.float32)
    else:
        dense_out = dense_base[:, [base_step_pos[c] for c in step_cols]].astype(
            np.float32
        )
    return dense_out, ep_base


def _variant_gather_order(layout: dict) -> tuple[list[str], list[str], np.ndarray]:
    concept_names = layout["concept_names"]
    step_signals = layout["step_signals"]
    epoch_signals = layout["epoch_signals"]

    step_cols: list[str] = []
    ep_cols: list[str] = []
    for concept in concept_names:
        step_cols.extend(step_signals[concept])
        ep_cols.extend(epoch_signals[concept])
    n_step = len(step_cols)
    step_pos = {c: i for i, c in enumerate(step_cols)}
    ep_pos = {c: i for i, c in enumerate(ep_cols)}

    gather_idx: list[int] = []
    for concept in concept_names:
        for sig in step_signals[concept]:
            si = step_pos[sig]
            for step_idx in range(STEP_PER_EPOCH):
                gather_idx.append(step_idx * n_step + si)
        for sig in epoch_signals[concept]:
            gather_idx.append(STEP_PER_EPOCH * n_step + ep_pos[sig])
    return step_cols, ep_cols, np.asarray(gather_idx, dtype=np.int64)


def prepare_variant_recordings(
    frame: pd.DataFrame,
    fit: set[str],
    layout: dict,
    *,
    added: set[str] | None = None,
    occipital_dir: Path | None = unified.OCCIPITAL_CACHE,
    cue_dir: Path = CUE_CACHE,
    allow_missing_occipital: bool = False,
) -> tuple[dict[str, sequence.Recording], dict]:
    """Stream per-subject variant features with strictly fitting-only standardization."""
    added = added or set()
    all_subjects = set(frame.subject_id.astype(str).unique())
    unified.assert_holdout_disjoint(all_subjects | fit)

    step_cols, ep_cols, gather_idx = _variant_gather_order(layout)
    n_step = len(step_cols)
    n_ep_cols = len(ep_cols)

    step_count = np.zeros(n_step, dtype=np.int64)
    step_total = np.zeros(n_step, dtype=np.float64)
    step_squared = np.zeros(n_step, dtype=np.float64)

    ep_count = np.zeros(n_ep_cols, dtype=np.int64)
    ep_total = np.zeros(n_ep_cols, dtype=np.float64)
    ep_squared = np.zeros(n_ep_cols, dtype=np.float64)

    fit_seconds = 0
    fit_epochs = 0
    for subject in sorted(fit):
        dense_step, ep_unique = _subject_variant_raw(
            subject,
            step_cols=step_cols,
            ep_cols=ep_cols,
            added=added,
            occipital_dir=occipital_dir,
            cue_dir=cue_dir,
            allow_missing_occipital=allow_missing_occipital,
        )
        fit_seconds += len(dense_step)
        fit_epochs += len(ep_unique)
        step_count += np.isfinite(dense_step).sum(axis=0)
        step_total += np.nansum(dense_step, axis=0, dtype=np.float64)
        step_squared += np.nansum(np.square(dense_step, dtype=np.float64), axis=0)

        ep_count += np.isfinite(ep_unique).sum(axis=0)
        ep_total += np.nansum(ep_unique, axis=0, dtype=np.float64)
        ep_squared += np.nansum(np.square(ep_unique, dtype=np.float64), axis=0)

    step_allow = (
        {step_cols.index("o_rel_theta")}
        if allow_missing_occipital and "o_rel_theta" in step_cols
        else set()
    )
    step_mean, step_scale, step_count = unified._finalize_moments(
        step_count, step_total, step_squared, allow_all_nan_cols=step_allow
    )
    ep_mean, ep_scale, ep_count = unified._finalize_moments(
        ep_count, ep_total, ep_squared
    )

    encoder = WindowEncoder(step_cols, context=0, mean=step_mean, sd=step_scale)
    pad = encoder.steps_per_epoch
    recordings: dict[str, sequence.Recording] = {}

    for subject, night in frame.groupby("subject_id", sort=False):
        subject_id = str(subject)
        dense_step, ep_unique = _subject_variant_raw(
            subject_id,
            step_cols=step_cols,
            ep_cols=ep_cols,
            added=added,
            occipital_dir=occipital_dir,
            cue_dir=cue_dir,
            allow_missing_occipital=allow_missing_occipital,
        )
        steps = encoder._steps(dense_step)[pad:-pad]
        steps = np.pad(steps, ((0, -len(steps) % pad), (0, 0)))
        step_values = steps.reshape(-1, STEP_PER_EPOCH * n_step).astype(np.float32)

        # Match length cleanly in case a truncated second differs between base and cue cache
        n_ep = min(len(ep_unique), len(step_values))
        step_values = step_values[:n_ep]
        ep_unique = ep_unique[:n_ep]
        ep_norm = np.clip((ep_unique - ep_mean) / ep_scale, -10.0, 10.0).astype(
            np.float32
        )
        ep_norm = np.where(np.isfinite(ep_norm), ep_norm, 0.0).astype(np.float32)
        flat = np.hstack((step_values, ep_norm)).astype(np.float32)
        concept_x = np.ascontiguousarray(flat[:, gather_idx], dtype=np.float32)

        labels = np.full(len(concept_x), -1, dtype=np.int64)
        at = night["epoch_index"].to_numpy()
        labels[at] = night[LABEL].to_numpy()
        recordings[subject_id] = sequence.Recording(
            torch.from_numpy(concept_x), torch.from_numpy(labels)
        )

    return recordings, {
        "variant": layout["variant"],
        "input_dim": int(layout["input_dim"]),
        "concept_names": list(layout["concept_names"]),
        "concept_dims": list(layout["concept_dims"]),
        "concept_widths": list(layout["concept_widths"]),
        "total_fit_seconds": fit_seconds,
        "total_fit_epochs": fit_epochs,
    }


def _paired_comparison(oof: pd.DataFrame, ref_path: Path) -> dict | None:
    if not ref_path.exists():
        return None
    ref = pd.read_parquet(ref_path)
    if "prediction" not in ref.columns:
        ref["prediction"] = ref[PROBABILITIES].to_numpy().argmax(axis=1)
    merged = oof[[*KEYS, LABEL, "fold", "prediction"]].merge(
        ref[[*KEYS, "prediction"]], on=[*KEYS], how="inner", suffixes=("_new", "_ref")
    )
    if len(merged) != len(oof):
        return None
    y = merged[LABEL].to_numpy()
    subs = merged["subject_id"].astype(str).to_numpy()
    folds = merged["fold"].to_numpy()
    cm_new, pats_new = patient_confusions(subs, y, merged["prediction_new"].to_numpy())
    cm_ref, pats_ref = patient_confusions(subs, y, merged["prediction_ref"].to_numpy())
    assert (pats_new == pats_ref).all()
    pat_fold = np.array([folds[subs == pt][0] for pt in pats_new])
    f_idx = [np.flatnonzero(pat_fold == f) for f in sorted(np.unique(folds).tolist())]
    draws = cv_fold_draws(f_idx, 2000, np.random.default_rng(SEED))
    d = cv_scores(cm_new, f_idx, draws)["macro_f1"] - cv_scores(cm_ref, f_idx, draws)["macro_f1"]
    return {"mean": float(d.mean()), "ci95": interval(d)}


def execute_variant(
    variant: str,
    folds_to_run: list[int],
    config: concept_run.TrainConfig,
    budget_seconds: float,
) -> int:
    layout = get_variant_layout(variant)
    var_dir = OUT / variant
    var_dir.mkdir(parents=True, exist_ok=True)
    pause_path = var_dir / "PAUSE"
    if pause_path.exists():
        log(f"PAUSE marker present at {pause_path}; exiting cleanly.")
        return 75

    dev_all, splits = development_frame()
    added_frame, eligible_added = additional.training_cohort(
        set(dev_all.subject_id.astype(str))
    )
    added_subjects = {str(s) for s in eligible_added}
    unified.assert_holdout_disjoint(
        set(dev_all.subject_id.astype(str).unique()) | added_subjects
    )

    deadline = active_clock() + budget_seconds
    fold_preds: list[pd.DataFrame] = []
    fold_comps: list[pd.DataFrame] = []
    fold_concepts: list[pd.DataFrame] = []
    fold_histories: dict[int, dict] = {}
    param_count = 0

    for fold in folds_to_run:
        train, val, test = fold_frames(dev_all, splits[fold])
        fold_dir = var_dir / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_file = fold_dir / "predictions.parquet"
        comp_file = fold_dir / "components.parquet"
        concept_file = fold_dir / "concept_logits.parquet"
        hist_file = fold_dir / "history.json"
        norm_file = fold_dir / "norm_meta.json"

        if (
            pred_file.exists()
            and comp_file.exists()
            and concept_file.exists()
            and hist_file.exists()
        ):
            log(f"[{variant}] Fold {fold}: loading cached outputs.")
            fold_preds.append(pd.read_parquet(pred_file))
            fold_comps.append(pd.read_parquet(comp_file))
            fold_concepts.append(pd.read_parquet(concept_file))
            fold_histories[fold] = json.loads(hist_file.read_text(encoding="utf-8"))
            param_count = fold_histories[fold].get("param_count", param_count)
            continue

        full_fit = pd.concat([train, added_frame], ignore_index=True)
        fit_subjects = sorted(full_fit.subject_id.astype(str).unique())
        val_subjects = sorted(val.subject_id.astype(str).unique())
        fold_frame = pd.concat([full_fit, val, test], ignore_index=True)

        log(
            f"[{variant}] Fold {fold}: preparing {len(fit_subjects)} fit + "
            f"{len(val_subjects)} val + {test.subject_id.nunique()} test..."
        )
        recordings, norm_meta = prepare_variant_recordings(
            fold_frame,
            set(fit_subjects),
            layout,
            added=added_subjects,
        )
        atomic_json(norm_file, norm_meta)

        model, state = concept_run.fit_concept_bigru(
            recordings,
            fit_subjects,
            val_subjects,
            SEED + fold,
            fold_dir,
            pause_path,
            deadline,
            config,
            concept_dims=layout["concept_dims"],
            concept_names=layout["concept_names"],
            width_per_concept=layout["concept_widths"],
        )
        if model is None:
            log(
                f"[{variant}] Fold {fold}: paused cleanly at pass {state['next_pass']} "
                f"batch {state['next_batch']}."
            )
            return 75

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        concept_run.verify_concept_model(model)
        pred_df, comp_df, concept_df = concept_run.score_fold_with_concepts(
            model, recordings, test, fold
        )
        unified.save_frame(pred_file, pred_df)
        unified.save_frame(comp_file, comp_df)
        unified.save_frame(concept_file, concept_df)
        hist_payload = {
            "param_count": int(param_count),
            "best": float(state["best"]),
            "best_pass": int(state["best_pass"]),
            "pass_seconds": float(state["pass_seconds"]),
            "history": state["history"],
        }
        atomic_json(hist_file, hist_payload)

        fold_preds.append(pred_df)
        fold_comps.append(comp_df)
        fold_concepts.append(concept_df)
        fold_histories[fold] = hist_payload

        del recordings, model
        gc.collect()

    oof = pd.concat(fold_preds, ignore_index=True)
    comp_df = pd.concat(fold_comps, ignore_index=True)
    concept_df = pd.concat(fold_concepts, ignore_index=True)
    unified.save_frame(var_dir / "predictions.parquet", oof)

    y = oof[LABEL].to_numpy()
    probs = oof[PROBABILITIES].to_numpy()
    folds_col = oof["fold"].to_numpy()
    fold_masks = [folds_col == f for f in sorted(np.unique(folds_col).tolist())]
    raw_metrics = unified._stage_metrics(y, probs, fold_masks)
    concept_diag = concept_run._compute_concept_diagnostics(
        comp_df, concept_df, concept_names=layout["concept_names"]
    )

    summary: dict = {
        "experiment": 80,
        "variant": variant,
        "folds": folds_to_run,
        "model_params": int(param_count),
        "input_dim": int(layout["input_dim"]),
        "concept_dims": {
            k: int(v) for k, v in zip(layout["concept_names"], layout["concept_dims"])
        },
        "concept_widths": {
            k: int(v) for k, v in zip(layout["concept_names"], layout["concept_widths"])
        },
        "raw": raw_metrics,
        "concept_diagnostics": concept_diag,
        "fold_histories": {str(k): v for k, v in fold_histories.items()},
        "peak_rss_mib": round(additional.peak_memory_bytes() / 1024**2, 1),
    }

    # Fold-matched paired bootstrap delta against #79 raw (works for 1 fold OR 5 folds!)
    if REFERENCE_79_RAW.exists():
        ref79 = pd.read_parquet(REFERENCE_79_RAW)
        ref79_sub = ref79[ref79.fold.isin(folds_to_run)]
        if len(ref79_sub) == len(oof):
            tmp_ref = ref79_sub.copy()
            summary["vs_79_raw_delta"] = _paired_comparison(oof, REFERENCE_79_RAW) if len(folds_to_run) == 5 else None
            if len(folds_to_run) < 5:
                merged = oof[[*KEYS, LABEL, "prediction"]].merge(
                    ref79_sub[[*KEYS, "prediction"]], on=[*KEYS], suffixes=("_new", "_79")
                )
                y_m = merged[LABEL].to_numpy()
                subs_m = merged["subject_id"].astype(str).to_numpy()
                cm_n, pats_n = patient_confusions(subs_m, y_m, merged["prediction_new"].to_numpy())
                cm_r, pats_r = patient_confusions(subs_m, y_m, merged["prediction_79"].to_numpy())
                assert (pats_n == pats_r).all()
                f_idx = [np.arange(len(pats_n))]
                draws = cv_fold_draws(f_idx, 2000, np.random.default_rng(SEED))
                delta = cv_scores(cm_n, f_idx, draws)["macro_f1"] - cv_scores(cm_r, f_idx, draws)["macro_f1"]
                summary["vs_79_raw_subset_delta"] = {
                    "mean": float(delta.mean()),
                    "ci95": interval(delta),
                }

    if len(folds_to_run) == 5:
        cal_probs, cal_details = fit_prior_power_oof(probs, y, folds_col)
        cal_df = oof.copy()
        cal_df["prediction"] = cal_probs.argmax(axis=1).astype(np.int64)
        for idx, col in enumerate(PROBABILITIES):
            cal_df[col] = cal_probs[:, idx].astype(np.float32)
        unified.save_frame(var_dir / "calibrated_predictions.parquet", cal_df)
        cal_metrics = unified._stage_metrics(y, cal_probs, fold_masks)
        is_trans = concept_run._transition_flags(oof)
        conformal = {
            "raw": evaluate_conformal_oof(probs, y, folds_col, is_trans, stages=tuple(STAGES)),
            "calibrated": evaluate_conformal_oof(
                cal_probs, y, folds_col, is_trans, stages=tuple(STAGES)
            ),
        }
        atomic_json(var_dir / "conformal_summary.json", conformal)
        summary["calibrated"] = cal_metrics
        summary["calibration_oof_details"] = cal_details
        summary["conformal"] = conformal
        summary["vs_79_calibrated_delta"] = _paired_comparison(cal_df, REFERENCE_79_CAL)
        summary["vs_75_calibrated_delta"] = _paired_comparison(cal_df, REFERENCE_75_CAL)

    atomic_json(var_dir / "summary.json", summary)
    log(
        f"[{variant}] COMPLETE (folds={folds_to_run}, params={param_count}): "
        f"raw macro-F1={raw_metrics['mean_fold_macro_f1']:.5f}, "
        f"N1={raw_metrics['stage_f1']['N1']:.4f}, N3={raw_metrics['stage_f1']['N3']:.4f}, "
        f"REM={raw_metrics['stage_f1']['REM']:.4f}, rss={summary['peak_rss_mib']:.1f} MiB"
    )
    return 0


def smoke() -> None:
    """Verify cue extraction parity, 5-concept model isolation, and mini resume."""
    dev_all, splits = development_frame()
    train, val, _ = fold_frames(dev_all, splits[0])
    subjects = sorted(train.subject_id.astype(str).unique())[:4]
    val_subjects = sorted(val.subject_id.astype(str).unique())[:2]
    permitted = frozenset([*subjects, *val_subjects])
    layout = get_variant_layout("all")

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cue_dir = tmp_dir / "cues"
        cue_dir.mkdir()
        for s in permitted:
            df = extract_missing_cues_for_subject(
                s, RAW, CACHE / "tracks1s" / f"{s}.parquet", permitted
            )
            assert list(df.columns) == ["second", *CUE_COLUMNS]
            df.to_parquet(cue_dir / f"{s}.parquet", index=False)

        mini = pd.concat(
            [
                train[train.subject_id.astype(str).isin(subjects)],
                val[val.subject_id.astype(str).isin(val_subjects)],
            ],
            ignore_index=True,
        )
        recs, meta = prepare_variant_recordings(mini, set(subjects), layout, cue_dir=cue_dir)
        assert recs[subjects[0]].features.shape[1] == layout["input_dim"]
        assert meta["concept_widths"] == [16, 16, 16, 16, 8]

        cfg = concept_run.TrainConfig(max_passes=2, min_passes=1, batch_size=2, patience=5)
        model, _ = concept_run.fit_concept_bigru(
            recs,
            subjects,
            val_subjects,
            SEED,
            tmp_dir / "run",
            tmp_dir / "PAUSE",
            active_clock() + 600,
            cfg,
            concept_dims=layout["concept_dims"],
            concept_names=layout["concept_names"],
            width_per_concept=layout["concept_widths"],
        )
        assert model is not None
        concept_run.verify_concept_model(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(
        f"EXP 80 SMOKE PASSED: variant=all, dims={layout['input_dim']} "
        f"({layout['concept_dims']}), params={n_params}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["delta", "rem_emg", "autonomic", "rem_auto", "all"],
        default=None,
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget-seconds", type=float, default=BUDGET_SECONDS)
    parser.add_argument("--max-passes", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        with local_training_slot():
            smoke()
        return 0
    if args.build_cache:
        with local_training_slot():
            build_cue_cache(workers=args.workers)
    if args.screen:
        with local_training_slot():
            run_cohen_d_screen()
    if args.variant is not None:
        cfg = concept_run.TrainConfig(max_passes=args.max_passes)
        with local_training_slot():
            return execute_variant(args.variant, args.folds, cfg, args.budget_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
