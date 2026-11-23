#!/usr/bin/env python
"""External evaluation of #82 minimal13 (13 features / 49 inputs) on Boston holdout (200) and Coimbra ISRUC (99).

Reuses the pre-extracted raw external feature cache from artifacts/exploration/cinc-bigru-concept/external_cache/,
applies exact fit-only fold moments, slices the 49 minimal13 inputs, averages the 5 fold checkpoints
strictly out-of-sample with zero holdout tuning, and logs Look #4 to LEDGER.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_concept_external_eval as ext79
import run_staging_all_additional_training as additional
import run_window_hmm_staging as window_run
from physionet_sleep.analysis.epoch_windows import WindowEncoder, epoch_frame
from physionet_sleep.experiments.bigru_staging import calibrate_prior_power
from physionet_sleep.experiments.concept_bigru_staging import CONCEPT_NAMES, ConceptBiGRUStagingNet
from physionet_sleep.io.isruc import resolve_isruc_record
from physionet_sleep.labels.isruc import read_isruc_hypnogram
from run_waveform_development import (
    CACHE,
    KEYS,
    LABEL,
    RAW,
    STAGES,
    atomic_json,
    development_frame,
    local_training_slot,
    log,
)

PRUNED_ROOT = REPO / "artifacts/exploration/cinc-bigru-pruned"


def _score_pruned_cohort(
    dataset: str,
    frame: pd.DataFrame,
    moments: dict[int, tuple],
    model_dir: Path,
    summary: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Score external cohort by averaging 5 fold minimal13 checkpoints strictly out-of-sample."""
    concept_dims = tuple(int(summary["concept_dims"][c]) for c in CONCEPT_NAMES)
    feat_idx = np.asarray(summary["feature_indices_79"], dtype=np.int64)
    cal_details = summary["calibration_oof_details"]

    models: list[ConceptBiGRUStagingNet] = []
    for f in range(5):
        ckpt = torch.load(model_dir / f"fold-{f}/checkpoint.pt", map_location="cpu", weights_only=True)
        m = ConceptBiGRUStagingNet(concept_dims=concept_dims)
        m.load_state_dict(ckpt["best_weights"])
        m.eval()
        models.append(m)

    subjects = frame["subject_id"].astype(str).unique().tolist()
    n_rows = len(frame)
    raw_probs_out = np.zeros((n_rows, 5), dtype=np.float64)
    cal_probs_out = np.zeros((n_rows, 5), dtype=np.float64)
    local_logits_out = np.zeros((n_rows, 5), dtype=np.float64)
    concept_logits_out = np.zeros((n_rows, 4, 5), dtype=np.float64)
    past_gates_out = np.zeros((n_rows, 5), dtype=np.float64)
    future_gates_out = np.zeros((n_rows, 5), dtype=np.float64)

    sub_positions: dict[str, np.ndarray] = {}
    subj_col = frame["subject_id"].astype(str).to_numpy()
    for s in subjects:
        sub_positions[s] = np.flatnonzero(subj_col == s)

    with torch.inference_mode():
        for idx, s in enumerate(subjects):
            dense_step, ep_unique = ext79._external_subject_raw(dataset, s)
            pos = sub_positions[s]
            ei = frame.iloc[pos]["epoch_index"].to_numpy()
            if ei.max() >= len(ep_unique):
                raise ValueError(f"{dataset}:{s} scored epoch {ei.max()} >= cached {len(ep_unique)}")

            raw_sum = np.zeros((len(pos), 5), dtype=np.float64)
            cal_sum = np.zeros((len(pos), 5), dtype=np.float64)
            for f in range(5):
                s_mean, s_scale, e_mean, e_scale, gather_idx = moments[f]
                enc = WindowEncoder(ext79.STEP_COLS, context=0, mean=s_mean, sd=s_scale)
                pad = enc.steps_per_epoch
                steps = enc._steps(dense_step)[pad:-pad]
                steps = np.pad(steps, ((0, -len(steps) % pad), (0, 0)))
                step_values = steps.reshape(-1, ext79.STEP_PER_EPOCH * len(ext79.STEP_COLS)).astype(np.float32)[: len(ep_unique)]
                ep_norm = np.clip((ep_unique - e_mean) / e_scale, -10.0, 10.0).astype(np.float32)
                ep_norm = np.where(np.isfinite(ep_norm), ep_norm, 0.0).astype(np.float32)
                x_flat = np.hstack((step_values, ep_norm)).astype(np.float32)
                x_81 = x_flat[:, gather_idx]
                x = torch.from_numpy(np.ascontiguousarray(x_81[:, feat_idx], dtype=np.float32)).unsqueeze(0)
                rec = torch.ones((1, len(ep_unique)), dtype=torch.bool)
                terms = models[f](x, rec, return_components=True)

                probs_f = softmax(terms.logits[0, ei].cpu().numpy(), axis=1)
                prior_f = np.asarray(cal_details[f]["outer_train_prior"], dtype=np.float64)
                exp_f = float(cal_details[f]["exponent"])
                cal_f = calibrate_prior_power(probs_f, prior_f, exp_f)
                raw_sum += probs_f
                cal_sum += cal_f

                local_logits_out[pos] += terms.local[0, ei].cpu().numpy() / 5.0
                concept_logits_out[pos] += terms.concept_logits[0, ei].cpu().numpy() / 5.0
                past_gates_out[pos] += terms.past_gate[0, ei].cpu().numpy() / 5.0
                future_gates_out[pos] += terms.future_gate[0, ei].cpu().numpy() / 5.0

            raw_probs_out[pos] = raw_sum / 5.0
            cal_probs_out[pos] = cal_sum / 5.0
            if (idx + 1) % 50 == 0 or idx + 1 == len(subjects):
                log(f"  scored [{dataset}] {idx + 1}/{len(subjects)}")

    pred_df = frame[[*KEYS]].copy()
    for col in frame.columns:
        if col not in KEYS and col not in pred_df.columns:
            pred_df[col] = frame[col].to_numpy()
    pred_df["prediction_raw"] = raw_probs_out.argmax(axis=1).astype(np.int64)
    pred_df["prediction"] = cal_probs_out.argmax(axis=1).astype(np.int64)
    for idx, st in enumerate(STAGES):
        pred_df[f"p_{st}"] = cal_probs_out[:, idx].astype(np.float32)
        pred_df[f"p_raw_{st}"] = raw_probs_out[:, idx].astype(np.float32)
        pred_df[f"local_{st}"] = local_logits_out[:, idx].astype(np.float32)
        pred_df[f"past_gate_{st}"] = past_gates_out[:, idx].astype(np.float32)
        pred_df[f"future_gate_{st}"] = future_gates_out[:, idx].astype(np.float32)
    for c_idx, cname in enumerate(CONCEPT_NAMES):
        for s_idx, st in enumerate(STAGES):
            pred_df[f"concept_{cname}_{st}"] = concept_logits_out[:, c_idx, s_idx].astype(np.float32)
    return pred_df, raw_probs_out, cal_probs_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="minimal13")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model_dir = PRUNED_ROOT / args.variant
    ext_out = model_dir / "external_eval"
    ext_out.mkdir(parents=True, exist_ok=True)
    look_file = ext_out / "single_look.json"
    if look_file.exists():
        raise SystemExit(
            f"#82 {args.variant} external look already recorded at {look_file}; refusing duplicate holdout look."
        )

    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))

    dev_all, splits = development_frame()
    dev_subjects = set(dev_all.subject_id.astype(str).unique())
    added_frame, eligible_added = additional.training_cohort(dev_subjects)
    added_subjects = {str(s) for s in eligible_added}

    wanted_holdout = (RAW / "selected_records_holdout.txt").read_text().split()
    holdout_subjects = [s for s in wanted_holdout if s not in dev_subjects and s not in added_subjects]
    assert len(holdout_subjects) == 200, f"Expected 200 holdout subjects, got {len(holdout_subjects)}"

    isruc_all = [p.stem for p in sorted((ext79.ISRUC_CACHE / "tracks1s").glob("*.parquet"), key=lambda x: int(x.stem))]
    isruc_subjects = [s for s in isruc_all if s not in ext79.ISRUC_EXCLUDED]
    assert len(isruc_subjects) == 99, f"Expected 99 ISRUC subjects, got {len(isruc_subjects)}"

    with local_training_slot():
        ext79.build_external_caches(holdout_subjects, isruc_subjects, workers=args.workers)
        moments = ext79._compute_all_fold_moments(dev_all, splits, added_frame, added_subjects)

        # 1. Boston Holdout (200)
        _, rows_h, _ = window_run.load_inputs("cinc2018", CACHE, holdout_subjects)
        ho_frame = epoch_frame(rows_h, window_run.WINDOW_FEATURES)[[
            "subject_id", "epoch_index", LABEL
        ]].copy()
        ref_ho = pd.read_parquet(REPO / "artifacts/staging_window_model/cinc2018/holdout/predictions_recovered_g20.parquet")
        assert len(ho_frame) == len(ref_ho), f"Holdout epoch count mismatch {len(ho_frame)} vs {len(ref_ho)}"

        ho_preds, ho_raw_p, ho_cal_p = _score_pruned_cohort("holdout", ho_frame, moments, model_dir, summary)
        y_ho = ho_preds[LABEL].to_numpy()
        s_ho = ho_preds["subject_id"].astype(str).to_numpy()
        ho_metrics_cal = ext79._aggregate_metrics(y_ho, ho_cal_p, s_ho)
        ho_metrics_raw = ext79._aggregate_metrics(y_ho, ho_raw_p, s_ho)
        ho_conformal = ext79._eval_conformal_external(ho_cal_p, y_ho, ho_preds, summary)

        ho_summary = {
            "experiment": 82,
            "variant": args.variant,
            "n_features": int(summary["n_features"]),
            "input_dim": int(summary["input_dim"]),
            "model_params": int(summary["model_params"]),
            "cohort": "cinc2018_holdout_200",
            "model": f"ConceptBiGRU-{args.variant} (5-fold ensemble, zero holdout tuning)",
            "calibrated": {k: v for k, v in ho_metrics_cal.items() if k != "per_night_macro_f1"},
            "raw": {k: v for k, v in ho_metrics_raw.items() if k != "per_night_macro_f1"},
            "conformal_calibrated": ho_conformal,
            "per_night_macro_f1": ho_metrics_cal["per_night_macro_f1"],
        }
        ho_preds.to_parquet(ext_out / "holdout_predictions.parquet", index=False)
        atomic_json(ext_out / "holdout_summary.json", ho_summary)

        # Log Look #4 cleanly in LEDGER.csv
        look_num = (int(pd.read_csv(ext79.HOLDOUT_LEDGER)["look"].max()) if ext79.HOLDOUT_LEDGER.exists() else 0) + 1
        ledger_row = pd.DataFrame([{
            "look": look_num,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "subjects": int(len(holdout_subjects)),
            "configs": f"g82_{args.variant}",
            "adopted_macro_f1": float(ho_metrics_cal["macro_f1"]),
            "note": f"Adopted #82 Concept-BiGRU {args.variant} ({summary['n_features']} features / {summary['input_dim']} inputs, {summary['model_params']:,} params)",
        }])
        ledger_row.to_csv(ext79.HOLDOUT_LEDGER, mode="a", header=False, index=False)

        # 2. Coimbra ISRUC (99)
        _, rows_i, _ = window_run.load_inputs("isruc", ext79.ISRUC_CACHE, isruc_subjects)
        is_frame = epoch_frame(rows_i, window_run.WINDOW_FEATURES)[[
            "subject_id", "epoch_index", LABEL
        ]].rename(columns={LABEL: "scorer1"}).copy()
        s2_vals = np.full(len(is_frame), -1, dtype=np.int64)
        subj_arr = is_frame["subject_id"].astype(str).to_numpy()
        ei_arr = is_frame["epoch_index"].to_numpy()
        for s in isruc_subjects:
            idx = np.flatnonzero(subj_arr == s)
            h2 = read_isruc_hypnogram(resolve_isruc_record(ext79.ISRUC_RAW, s), scorer=2)
            ok = ei_arr[idx] < len(h2)
            s2_vals[idx[ok]] = h2[ei_arr[idx[ok]]]
        is_frame["scorer2"] = s2_vals
        is_frame[LABEL] = is_frame["scorer1"]

        is_preds, is_raw_p, is_cal_p = _score_pruned_cohort("isruc", is_frame, moments, model_dir, summary)
        y_is1 = is_preds["scorer1"].to_numpy()
        s_is = is_preds["subject_id"].astype(str).to_numpy()
        both = is_preds["scorer2"].to_numpy() >= 0

        is_s1_cal = ext79._aggregate_metrics(y_is1, is_cal_p, s_is)
        is_s2_cal = ext79._aggregate_metrics(is_preds["scorer2"].to_numpy()[both], is_cal_p[both], s_is[both])
        is_s1_raw = ext79._aggregate_metrics(y_is1, is_raw_p, s_is)
        is_s2_raw = ext79._aggregate_metrics(is_preds["scorer2"].to_numpy()[both], is_raw_p[both], s_is[both])
        is_conformal_s1 = ext79._eval_conformal_external(
            is_cal_p, y_is1, is_preds[["subject_id", "epoch_index", LABEL]], summary
        )

        is_summary = {
            "experiment": 82,
            "variant": args.variant,
            "n_features": int(summary["n_features"]),
            "input_dim": int(summary["input_dim"]),
            "model_params": int(summary["model_params"]),
            "cohort": "isruc_coimbra_99",
            "excluded": ext79.ISRUC_EXCLUDED,
            "calibrated_scorer1": {k: v for k, v in is_s1_cal.items() if k != "per_night_macro_f1"},
            "calibrated_scorer2": {k: v for k, v in is_s2_cal.items() if k != "per_night_macro_f1"},
            "raw_scorer1": {k: v for k, v in is_s1_raw.items() if k != "per_night_macro_f1"},
            "raw_scorer2": {k: v for k, v in is_s2_raw.items() if k != "per_night_macro_f1"},
            "conformal_calibrated_scorer1": is_conformal_s1,
            "per_night_scorer1": is_s1_cal["per_night_macro_f1"],
            "per_night_scorer2": is_s2_cal["per_night_macro_f1"],
        }
        is_preds.to_parquet(ext_out / "isruc_predictions.parquet", index=False)
        atomic_json(ext_out / "isruc_summary.json", is_summary)

        single_look = {
            "experiment": 82,
            "variant": args.variant,
            "date": pd.Timestamp.now().isoformat(),
            "holdout_look": int(look_num),
            "holdout_subjects": len(holdout_subjects),
            "holdout_calibrated_macro_f1": ho_metrics_cal["macro_f1"],
            "isruc_subjects": len(isruc_subjects),
            "isruc_calibrated_scorer1_macro_f1": is_s1_cal["macro_f1"],
            "isruc_calibrated_scorer2_macro_f1": is_s2_cal["macro_f1"],
        }
        atomic_json(look_file, single_look)
        log(
            f"#82 {args.variant} EXTERNAL EVAL COMPLETE (Look #{look_num}): "
            f"Boston Holdout (200) cal={ho_metrics_cal['macro_f1']:.4f} (raw={ho_metrics_raw['macro_f1']:.4f}, kappa={ho_metrics_cal['kappa']:.4f}); "
            f"Coimbra ISRUC (99) s1={is_s1_cal['macro_f1']:.4f} / s2={is_s2_cal['macro_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
