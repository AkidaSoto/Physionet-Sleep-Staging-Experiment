"""Final single-look external evaluation of adopted #79 Concept-BiGRU on 200 CinC holdouts + 99 ISRUC nights."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scipy.special import softmax
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_all_additional_training as additional
import run_staging_bigru_concept as concept_run
import run_staging_bigru_unified as unified
import run_staging_complementary_markers as markers_script
import run_staging_feature_context as sequence
import run_window_hmm_staging as window_run
from physionet_sleep.algorithms.eeg_quality import channel_quality
from physionet_sleep.algorithms.eeg_tracks import FS, power_spectra
from physionet_sleep.analysis.epoch_windows import (
    WindowEncoder,
    dense_signals,
    epoch_frame,
    spindle_burst_counts,
    subject_rows6,
)
from physionet_sleep.experiments import complementary_n2_events as n2_ev
from physionet_sleep.experiments import complementary_slow_waves as sw_ev
from physionet_sleep.experiments.bigru_staging import (
    SLOW_WAVE_PANEL_COLUMNS,
    N2_EVENT_COLUMNS,
    block_epoch_mean,
    calibrate_prior_power,
    compute_night_history,
    compute_sigma_ratio,
    compute_split_support,
    extract_occipital_theta,
)
from physionet_sleep.experiments.concept_bigru_staging import (
    CONCEPT_EPOCH_SIGNALS,
    CONCEPT_NAMES,
    CONCEPT_STEP_SIGNALS,
    ConceptBiGRUStagingNet,
    DEFAULT_CONCEPT_DIMS,
    STEP_PER_EPOCH,
    TOTAL_CONCEPT_DIM,
    conformal_prediction_sets,
)
from physionet_sleep.experiments.waveform_data import EPOCH_SAMPLES, WaveformStore, resolve_cinc2018_record
from physionet_sleep.io.isruc import read_isruc_channels, resolve_isruc_record
from physionet_sleep.labels.isruc import read_isruc_hypnogram
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    RAW,
    STAGES,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
)

CONCEPT_DIR = EXPLORATION / "cinc-bigru-concept"
EXT_CACHE = CONCEPT_DIR / "external_cache"
EXT_OUT = CONCEPT_DIR / "external_eval"
HOLDOUT_LEDGER = REPO / "artifacts/staging_window_model/cinc2018/holdout/LEDGER.csv"
ISRUC_RAW = window_run.DATASETS["isruc"].raw_dir
ISRUC_CACHE = REPO / "artifacts/isruc_window_cache"
ISRUC_EXCLUDED = {"8": "night 8 excluded before scoring: no frontal EEG (F3/F4)"}
CLASSES = np.arange(len(STAGES))
MARKER_COLS = (*SLOW_WAVE_PANEL_COLUMNS, *N2_EVENT_COLUMNS)
STEP_COLS = [*CONCEPT_STEP_SIGNALS["slow_wave"], *CONCEPT_STEP_SIGNALS["n2_events"], *CONCEPT_STEP_SIGNALS["wake_n1"], *CONCEPT_STEP_SIGNALS["rem_tone"]]
EP_COLS = [*CONCEPT_EPOCH_SIGNALS["slow_wave"], *CONCEPT_EPOCH_SIGNALS["n2_events"], *CONCEPT_EPOCH_SIGNALS["wake_n1"], *CONCEPT_EPOCH_SIGNALS["rem_tone"]]


def _build_external_subject(args: tuple[str, str, Path, Path, tuple[str, ...]]) -> str:
    dataset, subject, occ_path, marker_path, permitted = args
    if occ_path.exists() and marker_path.exists():
        return subject

    if dataset == "holdout":
        store = WaveformStore(RAW, permitted_subjects=permitted, max_open_nights=1)
        try:
            f3 = store.channel_uv(subject, "F3-M2")
            f4 = store.channel_uv(subject, "F4-M1")
            o1 = store.channel_uv(subject, "O1-M2")
            o2 = store.channel_uv(subject, "O2-M1")
        finally:
            store._nights.clear()
        import wfdb

        rec = resolve_cinc2018_record(RAW, subject)
        ecg = None
        if rec.header_path is not None:
            sig, _ = wfdb.rdsamp(str(rec.header_path.with_suffix("")), channel_names=["ECG"])
            ecg = sig[:, 0]
        tracks = pd.read_parquet(
            CACHE / "tracks1s" / f"{subject}.parquet",
            columns=["c_rel_sigma", "c_rel_alpha", "c_rel_beta"],
        )
    elif dataset == "isruc":
        channels, _ = read_isruc_channels(resolve_isruc_record(ISRUC_RAW, subject))
        f3, f4 = channels["eeg_f3"][0], channels["eeg_f4"][0]
        o1, o2 = channels["eeg_o1"][0], channels["eeg_o2"][0]
        ecg = channels["ecg"][0] if "ecg" in channels else None
        n_min = min(len(f3), len(f4), len(o1), len(o2))
        f3, f4, o1, o2 = f3[:n_min], f4[:n_min], o1[:n_min], o2[:n_min]
        if ecg is not None:
            ecg = ecg[:n_min]
        tracks = pd.read_parquet(
            ISRUC_CACHE / "tracks1s" / f"{subject}.parquet",
            columns=["c_rel_sigma", "c_rel_alpha", "c_rel_beta"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_sec = min(len(f3) // FS, len(f4) // FS, len(o1) // FS, len(o2) // FS, len(tracks))
    n_epochs = n_sec // 30
    n_sec_ep = n_epochs * 30
    spectra = {
        "F3-M2": power_spectra(f3, n_sec_ep),
        "F4-M1": power_spectra(f4, n_sec_ep),
    }
    quality = channel_quality(
        {"F3-M2": f3[: n_sec_ep * FS], "F4-M1": f4[: n_sec_ep * FS]},
        spectra,
        ecg[: n_sec_ep * FS] if ecg is not None else None,
        n_sec_ep,
        channels=("F3-M2", "F4-M1"),
    )[markers_script.FRONTAL_QUALITY]

    spindle = spindle_burst_counts(tracks.iloc[:n_sec_ep])[:n_epochs]
    waves = sw_ev.slow_wave_panel(f3[: n_epochs * EPOCH_SAMPLES], f4[: n_epochs * EPOCH_SAMPLES], quality)
    events = n2_ev.n2_event_panel(f3[: n_epochs * EPOCH_SAMPLES], f4[: n_epochs * EPOCH_SAMPLES], quality, spindle)
    markers_df = pd.concat([waves, events], axis=1)

    occ_df = extract_occipital_theta(o1, o2)

    occ_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_occ = occ_path.with_suffix(".part")
    occ_df.to_parquet(tmp_occ, index=False)
    tmp_occ.replace(occ_path)
    tmp_mk = marker_path.with_suffix(".part")
    markers_df.to_parquet(tmp_mk, index=True)
    tmp_mk.replace(marker_path)
    return subject


def build_external_caches(holdout_subjects: list[str], isruc_subjects: list[str], workers: int = 4) -> None:
    """Build occipital theta and complementary markers for 200 CinC holdouts + 99 ISRUC nights."""
    started = perf_counter()
    jobs: list[tuple[str, str, Path, Path, tuple[str, ...]]] = []
    h_tuple = tuple(holdout_subjects)
    i_tuple = tuple(isruc_subjects)
    for s in holdout_subjects:
        jobs.append(
            (
                "holdout",
                s,
                EXT_CACHE / "holdout/occipital" / f"{s}.parquet",
                EXT_CACHE / "holdout/markers" / f"{s}.parquet",
                h_tuple,
            )
        )
    for s in isruc_subjects:
        jobs.append(
            (
                "isruc",
                s,
                EXT_CACHE / "isruc/occipital" / f"{s}.parquet",
                EXT_CACHE / "isruc/markers" / f"{s}.parquet",
                i_tuple,
            )
        )
    pending = [j for j in jobs if not (j[2].exists() and j[3].exists())]
    log(
        f"External caches: {len(jobs)} subjects total "
        f"({len(jobs) - len(pending)} cached, {len(pending)} to extract, workers={workers})"
    )
    if not pending:
        return
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_build_external_subject, j) for j in pending]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(pending):
                log(f"  external extracted {done}/{len(pending)} ({perf_counter() - started:.1f}s)")


def _compute_all_fold_moments(
    dev_all: pd.DataFrame, splits: list, added_frame: pd.DataFrame, added_subjects: set[str]
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Recover exact fitting-only standardization moments for each of the 5 folds in one 794-night pass."""
    fit_subs_by_fold: dict[int, set[str]] = {}
    all_fit: set[str] = set()
    for f, split in enumerate(splits):
        train, _, _ = fold_frames(dev_all, split)
        subs = set(train.subject_id.astype(str).unique()) | added_subjects
        fit_subs_by_fold[f] = subs
        all_fit |= subs
    unified.assert_holdout_disjoint(all_fit)

    n_step = len(STEP_COLS)
    n_ep = len(EP_COLS)
    fold_stats = {
        f: {
            "step_count": np.zeros(n_step, dtype=np.int64),
            "step_total": np.zeros(n_step, dtype=np.float64),
            "step_sq": np.zeros(n_step, dtype=np.float64),
            "ep_count": np.zeros(n_ep, dtype=np.int64),
            "ep_total": np.zeros(n_ep, dtype=np.float64),
            "ep_sq": np.zeros(n_ep, dtype=np.float64),
        }
        for f in range(len(splits))
    }

    log(f"Recovering exact fitting-only moments across {len(all_fit)} fitting subjects (1 pass)...")
    for idx, sub in enumerate(sorted(all_fit)):
        dense_step, ep_unique = concept_run._subject_concept_raw(
            sub,
            added=added_subjects,
            occipital_dir=concept_run.OCCIPITAL_CACHE,
            allow_missing_occipital=False,
        )
        sc = np.isfinite(dense_step).sum(axis=0)
        st = np.nansum(dense_step, axis=0, dtype=np.float64)
        ss = np.nansum(np.square(dense_step, dtype=np.float64), axis=0)
        ec = np.isfinite(ep_unique).sum(axis=0)
        et = np.nansum(ep_unique, axis=0, dtype=np.float64)
        es = np.nansum(np.square(ep_unique, dtype=np.float64), axis=0)

        for f, f_subs in fit_subs_by_fold.items():
            if sub in f_subs:
                fs_dict = fold_stats[f]
                fs_dict["step_count"] += sc
                fs_dict["step_total"] += st
                fs_dict["step_sq"] += ss
                fs_dict["ep_count"] += ec
                fs_dict["ep_total"] += et
                fs_dict["ep_sq"] += es
        if (idx + 1) % 200 == 0 or idx + 1 == len(all_fit):
            log(f"  moments pass {idx + 1}/{len(all_fit)}")

    _, _, gather_idx = concept_run._concept_feature_order_indices()
    moments: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for f in range(len(splits)):
        fs_dict = fold_stats[f]
        s_mean, s_scale, s_cnt = unified._finalize_moments(
            fs_dict["step_count"], fs_dict["step_total"], fs_dict["step_sq"]
        )
        e_mean, e_scale, e_cnt = unified._finalize_moments(
            fs_dict["ep_count"], fs_dict["ep_total"], fs_dict["ep_sq"]
        )
        # Bit-exactness check against saved fold-{f}/norm_meta.json!
        norm_meta = json.loads((CONCEPT_DIR / f"fold-{f}/norm_meta.json").read_text(encoding="utf-8"))
        assert s_cnt.tolist() == norm_meta["step_finite_counts"], f"Fold {f} step count mismatch"
        assert e_cnt.tolist() == norm_meta["ep_finite_counts"], f"Fold {f} ep count mismatch"
        moments[f] = (s_mean, s_scale, e_mean, e_scale, gather_idx)
    log("All 5 fold moment vectors verified bit-for-bit against saved norm_meta.json counts.")
    return moments


def _external_subject_raw(dataset: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """Assemble unstandardized #79 step and epoch signals for one CinC holdout or ISRUC night."""
    if dataset == "holdout":
        cache_dir = CACHE
        ds_name = "cinc2018"
        occ_path = EXT_CACHE / "holdout/occipital" / f"{subject}.parquet"
        mk_path = EXT_CACHE / "holdout/markers" / f"{subject}.parquet"
    elif dataset == "isruc":
        cache_dir = ISRUC_CACHE
        ds_name = "isruc"
        occ_path = EXT_CACHE / "isruc/occipital" / f"{subject}.parquet"
        mk_path = EXT_CACHE / "isruc/markers" / f"{subject}.parquet"
    else:
        raise ValueError(dataset)

    rows6_cols = list(sequence.FEATURES[:-1])  # all except derived spindle_bursts
    rows6 = subject_rows6(ds_name, subject, rows6_cols, cache_dir / "rows6")
    tracks = pd.read_parquet(
        cache_dir / "tracks1s" / f"{subject}.parquet",
        columns=[*unified.BASE_STEP_TRACKS, "c_rel_sigma"],
    )
    occ = pd.read_parquet(occ_path, columns=["second", "o_rel_theta"])
    markers_df = pd.read_parquet(mk_path, columns=list(MARKER_COLS))

    tracks = tracks.copy()
    tracks["sigma_ratio"] = compute_sigma_ratio(tracks)
    n_occ = min(len(tracks), len(occ))
    o_theta = np.full(len(tracks), np.nan, dtype=np.float32)
    o_theta[:n_occ] = occ["o_rel_theta"].to_numpy(dtype=np.float32)[:n_occ]
    tracks["o_rel_theta"] = o_theta

    # Construct raw dense matching STEP_COLS
    all_raw_step_names = [*unified.STEP_TRACKS, *sequence.FEATURES]
    raw_dense = dense_signals(tracks, rows6, list(unified.STEP_TRACKS), list(sequence.FEATURES))
    step_idx_map = {c: i for i, c in enumerate(all_raw_step_names)}
    dense_step = raw_dense[:, [step_idx_map[c] for c in STEP_COLS]].astype(np.float32)

    n_ep = min(len(markers_df), len(tracks) // 30)
    spindle = spindle_burst_counts(tracks)[:n_ep]
    kc = markers_df["kc_shape_events"].to_numpy(dtype=np.float32)[:n_ep]
    recent_k, recent_spindle = compute_split_support(kc, spindle)
    hist_cols = compute_night_history(rows6, len(tracks))[:n_ep]  # (n_ep, 7)

    # Assemble EP_COLS exactly in order
    from physionet_sleep.experiments.bigru_staging import NIGHT_HISTORY_COLUMNS

    hist_idx = {col: i for i, col in enumerate(NIGHT_HISTORY_COLUMNS)}
    ep_parts: list[np.ndarray] = []
    for col in EP_COLS:
        if col in SLOW_WAVE_PANEL_COLUMNS or col == "kc_shape_events":
            ep_parts.append(markers_df[col].to_numpy(dtype=np.float32)[:n_ep])
        elif col == "spindle_bursts":
            ep_parts.append(spindle.astype(np.float32)[:n_ep])
        elif col == "recent_k_support":
            ep_parts.append(recent_k)
        elif col == "recent_spindle_support":
            ep_parts.append(recent_spindle)
        elif col in hist_idx:
            ep_parts.append(hist_cols[:, hist_idx[col]])
        elif col in step_idx_map:
            raw_1hz = raw_dense[: n_ep * 30, step_idx_map[col]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ep_parts.append(np.nanmean(raw_1hz.reshape(n_ep, 30), axis=1).astype(np.float32))
        else:
            raise KeyError(f"Unknown EP_COL: {col}")
    ep_unique = np.column_stack(ep_parts).astype(np.float32)
    return dense_step[: n_ep * 30], ep_unique


def _score_cohort_with_fold_ensemble(
    dataset: str,
    frame: pd.DataFrame,
    moments: dict[int, tuple],
    summary79: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Score external cohort by averaging 5 fold ConceptBiGRU checkpoints strictly out-of-sample."""
    models: list[ConceptBiGRUStagingNet] = []
    cal_details = summary79["calibration_oof_details"]
    for f in range(5):
        ckpt = torch.load(CONCEPT_DIR / f"fold-{f}/checkpoint.pt", map_location="cpu", weights_only=True)
        m = ConceptBiGRUStagingNet()
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

    # Index positions per subject preserving original frame order
    sub_positions: dict[str, np.ndarray] = {}
    subj_col = frame["subject_id"].astype(str).to_numpy()
    for s in subjects:
        sub_positions[s] = np.flatnonzero(subj_col == s)

    with torch.inference_mode():
        for idx, s in enumerate(subjects):
            dense_step, ep_unique = _external_subject_raw(dataset, s)
            pos = sub_positions[s]
            ei = frame.iloc[pos]["epoch_index"].to_numpy()
            if ei.max() >= len(ep_unique):
                raise ValueError(f"{dataset}:{s} scored epoch {ei.max()} >= cached {len(ep_unique)}")

            raw_sum = np.zeros((len(pos), 5), dtype=np.float64)
            cal_sum = np.zeros((len(pos), 5), dtype=np.float64)
            for f in range(5):
                s_mean, s_scale, e_mean, e_scale, gather_idx = moments[f]
                enc = WindowEncoder(STEP_COLS, context=0, mean=s_mean, sd=s_scale)
                pad = enc.steps_per_epoch
                steps = enc._steps(dense_step)[pad:-pad]
                steps = np.pad(steps, ((0, -len(steps) % pad), (0, 0)))
                step_values = steps.reshape(-1, STEP_PER_EPOCH * len(STEP_COLS)).astype(np.float32)[: len(ep_unique)]
                ep_norm = np.clip((ep_unique - e_mean) / e_scale, -10.0, 10.0).astype(np.float32)
                ep_norm = np.where(np.isfinite(ep_norm), ep_norm, 0.0).astype(np.float32)
                x_flat = np.hstack((step_values, ep_norm)).astype(np.float32)
                x = torch.from_numpy(np.ascontiguousarray(x_flat[:, gather_idx], dtype=np.float32)).unsqueeze(0)
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


def _night_scores(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    out = {}
    for s in sorted(np.unique(subjects).tolist(), key=lambda v: (int(v) if str(v).isdigit() else str(v))):
        mask = subjects == s
        labels_present = sorted(set(y[mask].tolist()))
        out[str(s)] = float(
            f1_score(y[mask], pred[mask], labels=labels_present, average="macro", zero_division=0)
        )
    return out


def _eval_conformal_external(
    probs: np.ndarray, y: np.ndarray, frame: pd.DataFrame, summary79: dict
) -> dict:
    """Apply fixed 5-fold mean OOF Mondrian thresholds (zero external tuning!) to unseen cohorts."""
    trans = concept_run._transition_flags(frame)
    out = {}
    for cov_key in ("coverage_90", "coverage_95"):
        folds_list = summary79["conformal"]["calibrated"][cov_key]["mondrian"]["folds"]
        mean_thr = np.mean([row["threshold"] for row in folds_list], axis=0)
        sets, empty_rate = conformal_prediction_sets(probs, mean_thr)
        covered = sets[np.arange(len(y)), y]
        sizes = sets.sum(axis=1)
        single = sizes == 1
        pred = probs.argmax(axis=1)
        out[cov_key] = {
            "target_coverage": float(summary79["conformal"]["calibrated"][cov_key]["mondrian"]["target_coverage"]),
            "empirical_coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "singleton_rate": float(single.mean()),
            "singleton_accuracy": float((pred[single] == y[single]).mean()) if single.any() else None,
            "empty_before_fallback_rate": float(empty_rate),
            "per_stage_coverage": {
                st: float(covered[y == i].mean()) for i, st in enumerate(STAGES) if (y == i).any()
            },
            "transition_set_size": float(sizes[trans].mean()) if trans.any() else None,
            "interior_set_size": float(sizes[~trans].mean()) if (~trans).any() else None,
        }
    return out


def _aggregate_metrics(y: np.ndarray, probs: np.ndarray, subjects: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    st_f1 = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    per_night = _night_scores(y, pred, subjects)
    vals = np.array(list(per_night.values()), dtype=np.float64)
    return {
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro")),
        "kappa": float(cohen_kappa_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, probs, labels=CLASSES)),
        "n_epochs": int(len(y)),
        "n_subjects": int(len(per_night)),
        "stage_f1": {st: float(v) for st, v in zip(STAGES, st_f1)},
        "subj_median": float(np.median(vals)),
        "subj_q1": float(np.percentile(vals, 25)),
        "subj_q3": float(np.percentile(vals, 75)),
        "subj_min": float(vals.min()),
        "subj_max": float(vals.max()),
        "per_night_macro_f1": per_night,
    }


def smoke() -> None:
    """Verify CinC and ISRUC external extraction helpers without touching holdout predictions."""
    dev_all, _ = development_frame()
    sub = sorted(dev_all.subject_id.astype(str).unique())[0]
    with TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        _build_external_subject(
            ("holdout", sub, tdir / "occ.parquet", tdir / "mk.parquet", (sub,))
        )
        occ = pd.read_parquet(tdir / "occ.parquet")
        mk = pd.read_parquet(tdir / "mk.parquet")
        assert "o_rel_theta" in occ.columns and len(occ) > 300
        assert tuple(mk.columns) == MARKER_COLS
    # Also verify ISRUC channel loader on subject 1
    channels, calib = read_isruc_channels(resolve_isruc_record(ISRUC_RAW, "1"))
    assert all(k in channels for k in ("eeg_f3", "eeg_f4", "eeg_o1", "eeg_o2"))
    log(f"EXTERNAL EVAL SMOKE PASSED (cinc sub={sub}, isruc sub=1 calib={calib['exg_gain_fixed']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.smoke:
        with local_training_slot():
            smoke()
        return 0

    EXT_OUT.mkdir(parents=True, exist_ok=True)
    look_file = EXT_OUT / "single_look.json"
    if look_file.exists():
        raise SystemExit(
            f"#79 external single look already recorded at {look_file}; refusing duplicate holdout look."
        )

    dev_all, splits = development_frame()
    dev_subjects = set(dev_all.subject_id.astype(str).unique())
    added_frame, eligible_added = additional.training_cohort(dev_subjects)
    added_subjects = {str(s) for s in eligible_added}

    wanted_holdout = (RAW / "selected_records_holdout.txt").read_text().split()
    holdout_subjects = [s for s in wanted_holdout if s not in dev_subjects and s not in added_subjects]
    assert len(holdout_subjects) == 200, f"Expected 200 holdout subjects, got {len(holdout_subjects)}"

    isruc_all = [p.stem for p in sorted((ISRUC_CACHE / "tracks1s").glob("*.parquet"), key=lambda x: int(x.stem))]
    isruc_subjects = [s for s in isruc_all if s not in ISRUC_EXCLUDED]
    assert len(isruc_subjects) == 99, f"Expected 99 ISRUC subjects, got {len(isruc_subjects)}"

    with local_training_slot():
        build_external_caches(holdout_subjects, isruc_subjects, workers=args.workers)
        moments = _compute_all_fold_moments(dev_all, splits, added_frame, added_subjects)
        summary79 = json.loads((CONCEPT_DIR / "summary.json").read_text(encoding="utf-8"))

        # 1. Build exact Boston Holdout (200) evaluation frame matching prior holdout look
        _, rows_h, _ = window_run.load_inputs("cinc2018", CACHE, holdout_subjects)
        ho_frame = epoch_frame(rows_h, window_run.WINDOW_FEATURES)[[
            "subject_id", "epoch_index", LABEL
        ]].copy()
        ref_ho = pd.read_parquet(REPO / "artifacts/staging_window_model/cinc2018/holdout/predictions_recovered_g20.parquet")
        assert len(ho_frame) == len(ref_ho), f"Holdout epoch count mismatch {len(ho_frame)} vs {len(ref_ho)}"

        ho_preds, ho_raw_p, ho_cal_p = _score_cohort_with_fold_ensemble("holdout", ho_frame, moments, summary79)
        y_ho = ho_preds[LABEL].to_numpy()
        s_ho = ho_preds["subject_id"].astype(str).to_numpy()
        ho_metrics_cal = _aggregate_metrics(y_ho, ho_cal_p, s_ho)
        ho_metrics_raw = _aggregate_metrics(y_ho, ho_raw_p, s_ho)
        ho_conformal = _eval_conformal_external(ho_cal_p, y_ho, ho_preds, summary79)

        ho_summary = {
            "experiment": 79,
            "cohort": "cinc2018_holdout_200",
            "model": "ConceptBiGRU (5-fold ensemble, zero holdout tuning)",
            "calibrated": {k: v for k, v in ho_metrics_cal.items() if k != "per_night_macro_f1"},
            "raw": {k: v for k, v in ho_metrics_raw.items() if k != "per_night_macro_f1"},
            "conformal_calibrated": ho_conformal,
            "per_night_macro_f1": ho_metrics_cal["per_night_macro_f1"],
        }
        ho_preds.to_parquet(EXT_OUT / "holdout_predictions.parquet", index=False)
        atomic_json(EXT_OUT / "holdout_summary.json", ho_summary)

        # Log Look #3 cleanly in LEDGER.csv
        look_num = (int(pd.read_csv(HOLDOUT_LEDGER)["look"].max()) if HOLDOUT_LEDGER.exists() else 0) + 1
        ledger_row = pd.DataFrame([{
            "look": look_num,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "subjects": int(len(holdout_subjects)),
            "configs": "g79_concept_bigru",
            "adopted_macro_f1": float(ho_metrics_cal["macro_f1"]),
            "note": "Adopted #79 Concept-BiGRU 5-fold ensemble (61,599 params)",
        }])
        ledger_row.to_csv(HOLDOUT_LEDGER, mode="a", header=False, index=False)

        # 2. Build exact Coimbra ISRUC (99) evaluation frame matching prior ISRUC look
        _, rows_i, _ = window_run.load_inputs("isruc", ISRUC_CACHE, isruc_subjects)
        is_frame = epoch_frame(rows_i, window_run.WINDOW_FEATURES)[[
            "subject_id", "epoch_index", LABEL
        ]].rename(columns={LABEL: "scorer1"}).copy()
        # Attach scorer2
        s2_vals = np.full(len(is_frame), -1, dtype=np.int64)
        subj_arr = is_frame["subject_id"].astype(str).to_numpy()
        ei_arr = is_frame["epoch_index"].to_numpy()
        for s in isruc_subjects:
            idx = np.flatnonzero(subj_arr == s)
            h2 = read_isruc_hypnogram(resolve_isruc_record(ISRUC_RAW, s), scorer=2)
            ok = ei_arr[idx] < len(h2)
            s2_vals[idx[ok]] = h2[ei_arr[idx[ok]]]
        is_frame["scorer2"] = s2_vals
        is_frame[LABEL] = is_frame["scorer1"]

        is_preds, is_raw_p, is_cal_p = _score_cohort_with_fold_ensemble("isruc", is_frame, moments, summary79)
        y_is1 = is_preds["scorer1"].to_numpy()
        s_is = is_preds["subject_id"].astype(str).to_numpy()
        both = is_preds["scorer2"].to_numpy() >= 0

        is_s1_cal = _aggregate_metrics(y_is1, is_cal_p, s_is)
        is_s2_cal = _aggregate_metrics(is_preds["scorer2"].to_numpy()[both], is_cal_p[both], s_is[both])
        is_s1_raw = _aggregate_metrics(y_is1, is_raw_p, s_is)
        is_s2_raw = _aggregate_metrics(is_preds["scorer2"].to_numpy()[both], is_raw_p[both], s_is[both])
        is_conformal_s1 = _eval_conformal_external(
            is_cal_p, y_is1, is_preds[["subject_id", "epoch_index", LABEL]], summary79
        )

        is_summary = {
            "experiment": 79,
            "cohort": "isruc_coimbra_99",
            "excluded": ISRUC_EXCLUDED,
            "calibrated_scorer1": {k: v for k, v in is_s1_cal.items() if k != "per_night_macro_f1"},
            "calibrated_scorer2": {k: v for k, v in is_s2_cal.items() if k != "per_night_macro_f1"},
            "raw_scorer1": {k: v for k, v in is_s1_raw.items() if k != "per_night_macro_f1"},
            "raw_scorer2": {k: v for k, v in is_s2_raw.items() if k != "per_night_macro_f1"},
            "conformal_calibrated_scorer1": is_conformal_s1,
            "per_night_scorer1": is_s1_cal["per_night_macro_f1"],
            "per_night_scorer2": is_s2_cal["per_night_macro_f1"],
        }
        is_preds.to_parquet(EXT_OUT / "isruc_predictions.parquet", index=False)
        atomic_json(EXT_OUT / "isruc_summary.json", is_summary)

        single_look = {
            "experiment": 79,
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
            f"EXTERNAL EVAL COMPLETE (Look #{look_num}): "
            f"Boston Holdout (200) cal={ho_metrics_cal['macro_f1']:.4f} (raw={ho_metrics_raw['macro_f1']:.4f}, κ={ho_metrics_cal['kappa']:.4f}); "
            f"Coimbra ISRUC (99) s1={is_s1_cal['macro_f1']:.4f} / s2={is_s2_cal['macro_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
