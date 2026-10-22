"""Window staging model: HGB on 150-s windows of 1-s tracks and 6-s features, decoded with a sleep-stage HMM.

Each epoch is scored from the epoch and two neighbours on each side, as 25 six-second steps of 33 signals
(14 one-second tracks averaged to 6 s, 16 staging features, 3 cardiorespiratory features). Class-balanced HGB
probabilities are then decoded with forward-backward over a 5x5 transition matrix counted from training labels.

Configurations (each scored epoch-only and with the HMM, on the same 5 subject-grouped folds):
  epoch19   the 19 features averaged over the epoch (reference: no context, no tracks)
  window19  the 19 features over the window (context, no tracks)
  window33  tracks + 19 features over the window (adopted)
Ablations (run with --tag so the main outputs are not overwritten):
  nocardio27   window33 without heart and breathing (3 tracks + 3 features removed)
  cardio6      heart and breathing only (hr, resp_effort, resp_env + the 3 cardiorespiratory features)
  cardio6_c5   heart and breathing only, 5 epochs each side (5.5 minutes)
  cardiofull_c5 / _c10   all 9 heart and breathing features + 3 tracks, 5 or 10 epochs each side
  --unbalanced  equal class weights instead of balanced (what published cardiorespiratory models optimise)

Outputs (artifacts/staging_window_model/<dataset>/):
  summary.csv, folds.csv, predictions.parquet          scores and per-epoch predictions
  transitions.csv, bouts.csv                           HMM transition matrix and bout lengths, all labelled nights
  importance.csv                                       adopted-model permutation importance: per signal and per 6-s step
Outputs of earlier runs are kept: a run replaces only the configs it ran.

Usage: python scripts/run_window_hmm_staging.py [--dataset cinc2018] [--configs epoch19,window19,window33]
                                                 [--importance-folds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.algorithms.eeg_tracks import TRACKS, cinc2018_subject_tracks  # noqa: E402
from physionet_sleep.analysis.datasets import DATASETS  # noqa: E402
from physionet_sleep.analysis.epoch_windows import (  # noqa: E402
    BASE16,
    CARDIO3,
    SPINDLE_BURSTS,
    SUBJ,
    TARGET,
    WINDOW_FEATURES,
    WindowEncoder,
    dense_signals,
    epoch_frame,
    subject_rows6,
)
from physionet_sleep.decoders.hmm import bout_lengths, learn_transition_matrix, posterior_decode  # noqa: E402
from physionet_sleep.experiments.tabular_runner import build_model, build_subject_eval_splits  # noqa: E402

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
CARDIO_FULL = [
    "hrv_features.hr_bpm_30s", "hrv_features.hr_surge_30s", "hrv_features.sdnn_150s", "hrv_features.hr_slow_sd_300s",
    "hrv_features.dfa_a2_300s", "hrv_features.ln_lf_hf_300s", "breath_regularity.breath_interval_cv_60s",
    "breath_regularity.breath_amp_cv_60s", "cardiorespiratory_coupling.hr_resp_coherence_120s",
]
CLASS_WEIGHT = None  # None -> balanced (build_model default); set by --unbalanced
ADOPTED = "g20"  # model C (2026-10-22): final21 with two spectral fits swapped for spindle_bursts (feature_reduction/FINDINGS.md)
CARDIO_TRACKS = ["hr", "resp_effort", "resp_env"]
# Removed from F: lowest permutation importance in nocardio27, CV macro-F1 unchanged (0.7754 vs 0.7755).
PRUNED_TRACKS = ["emg_log_rms", "eog_low", "c_rel_delta"]
PRUNED_FEATURES = ["spindle_features.theta_residual_power", "spindle_features.beta_residual_power",
                   "spindle_features.alpha_residual_power"]
# Swapped out of F for g20 in favour of spindle_bursts (round 3 swapS2: CV 0.7728, worst fold -0.0039).
G20_SWAPPED_OUT = ["spindle_features.aperiodic_intercept", "spindle_features.gamma_residual_power"]


def nocardio27_signals() -> tuple[list[str], list[str]]:
    """(tracks, features) of model D: 11 one-second tracks and the 16 base features."""
    return [t for t in TRACKS if t not in CARDIO_TRACKS], list(BASE16)


def final21_signals() -> tuple[list[str], list[str]]:
    """(tracks, features) of model F: 8 one-second tracks and 13 features."""
    trk, feats = nocardio27_signals()
    return [t for t in trk if t not in PRUNED_TRACKS], [f for f in feats if f not in PRUNED_FEATURES]


def adopted_signals() -> tuple[list[str], list[str]]:
    """(tracks, features) of the adopted model C (g20): 8 one-second tracks, 11 features and spindle_bursts."""
    trk, feats = final21_signals()
    return trk, [f for f in feats if f not in G20_SWAPPED_OUT] + [SPINDLE_BURSTS]
CLASSES = [0, 1, 2, 3, 4]
SEED = 7


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nights(frame: pd.DataFrame):
    """Row positions of each subject's epochs, in time order."""
    subj = frame[SUBJ].astype(str).to_numpy()
    order = np.lexsort((frame["epoch_index"].to_numpy(), subj))
    s = subj[order]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield order[a:b]


def hmm_decode(frame: pd.DataFrame, proba: np.ndarray, model) -> np.ndarray:
    post = np.zeros_like(proba)
    for idx in nights(frame):
        post[idx] = posterior_decode(proba[idx], model)
    return post


def transition_model(frame: pd.DataFrame):
    y = frame[TARGET].to_numpy()
    return learn_transition_matrix([y[idx] for idx in nights(frame)])


def fit_proba(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, columns: list[str]):
    model = build_model(model_family="hgb", task="classification", random_state=SEED, class_weight=CLASS_WEIGHT)
    model.fit(pd.DataFrame(Xtr, columns=columns), ytr)
    proba = np.zeros((len(Xte), 5))
    proba[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(pd.DataFrame(Xte, columns=columns))
    return model, proba


def scores(frame: pd.DataFrame, pred: np.ndarray, folds: list[np.ndarray]) -> dict:
    y = frame[TARGET].to_numpy()
    per_fold = [f1_score(y[i], pred[i], average="macro") for i in folds]
    row = {"macro_f1": float(np.mean(per_fold)), "folds": " ".join(f"{v:.3f}" for v in per_fold),
           "kappa": float(np.mean([cohen_kappa_score(y[i], pred[i]) for i in folds]))}
    for st, v in zip(STAGES, f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)):
        row[f"f1_{st}"] = float(v)
    subj = frame[SUBJ].astype(str).to_numpy()
    per_subj = pd.Series({s: f1_score(y[subj == s], pred[subj == s], labels=sorted(set(y[subj == s])), average="macro")
                          for s in np.unique(subj)})
    row.update(subj_median=float(per_subj.median()), subj_min=float(per_subj.min()), worst_subject=str(per_subj.idxmin()))
    return row


def development_subjects(dataset: str) -> list[str]:
    """The fixed development set (dev_subjects.txt), so held-out records built later never enter cross-validation."""
    path = REPO / "artifacts/staging_window_model" / dataset / "dev_subjects.txt"
    return path.read_text().split() if path.exists() else DATASETS[dataset].built_subjects()


def load_inputs(dataset: str, cache: Path, subjects: list[str] | None = None):
    spec = DATASETS[dataset]
    subjects = subjects or development_subjects(dataset)
    rows, tracks = {}, {}
    for i, s in enumerate(subjects):
        rows[s] = subject_rows6(dataset, s, WINDOW_FEATURES + [c for c in CARDIO_FULL if c not in WINDOW_FEATURES], cache / "rows6")
        tp = cache / "tracks1s" / f"{s}.parquet"
        if not tp.exists():
            tp.parent.mkdir(parents=True, exist_ok=True)
            cinc2018_subject_tracks(spec.raw_dir, spec.cache_dir, s).to_parquet(tp, index=False)
            log(f"tracks {i + 1}/{len(subjects)} {s}")
        tracks[s] = pd.read_parquet(tp)
    return subjects, rows, tracks


def label_structure(frame: pd.DataFrame, out: Path) -> None:
    """Transition matrix and bout lengths over every labelled night (the talk's description of sleep structure)."""
    tm = transition_model(frame)
    pd.DataFrame(tm.trans, index=STAGES, columns=STAGES).to_csv(out / "transitions.csv")
    y = frame[TARGET].to_numpy()
    lens: list[list[int]] = [[] for _ in STAGES]
    for idx in nights(frame):
        st, ln = bout_lengths(y[idx])
        for a, b in zip(st, ln):
            lens[int(a)].append(int(b))
    rows = []
    for k, st in enumerate(STAGES):
        L = np.asarray(lens[k])
        rows.append({"stage": st, "bouts": len(L), "median_epochs": float(np.median(L)), "mean_epochs": float(L.mean()),
                     "p90_epochs": float(np.percentile(L, 90)), "share_single_epoch": float(np.mean(L == 1)),
                     "self_transition": float(tm.trans[k, k]), "initial": float(tm.initial[k])})
    pd.DataFrame(rows).to_csv(out / "bouts.csv", index=False)


def permutation_importance(model, X: np.ndarray, y: np.ndarray, columns: list[str], signal_names: list[str],
                           step_offsets: list[int], fold: int, rng) -> list[dict]:
    """Macro-F1 drop when one signal is shuffled across epochs at every step, or one step across every signal."""
    def f1(Xp):
        pred = model.predict(pd.DataFrame(Xp.reshape(len(Xp), -1), columns=columns))
        return f1_score(y, pred, average="macro"), f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)

    base, base_cls = f1(X)
    out = []
    for j, name in enumerate(signal_names):
        Xp = X.copy()
        Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
        m, cls = f1(Xp)
        out.append({"fold": fold, "group": "signal", "name": name, "drop": base - m,
                    **{f"drop_{st}": float(b - c) for st, b, c in zip(STAGES, base_cls, cls)}})
    for t, off in enumerate(step_offsets):
        Xp = X.copy()
        Xp[:, t, :] = Xp[rng.permutation(len(Xp)), t, :]
        m, cls = f1(Xp)
        out.append({"fold": fold, "group": "step", "name": f"{off:+d}s", "drop": base - m,
                    **{f"drop_{st}": float(b - c) for st, b, c in zip(STAGES, base_cls, cls)}})
    log(f"importance fold {fold}: base {base:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cinc2018")
    ap.add_argument("--configs", default="epoch19,window19,window33")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--importance-folds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="", help="write outputs to a subfolder (for ablations)")
    ap.add_argument("--unbalanced", action="store_true", help="equal class weights instead of balanced")
    args = ap.parse_args()
    configs = args.configs.split(",")
    global CLASS_WEIGHT
    if args.unbalanced:
        CLASS_WEIGHT = {k: 1.0 for k in range(5)}
    imp_folds = {int(f) for f in args.importance_folds.split(",") if f}
    cache = REPO / f"artifacts/{args.dataset}_window_cache"
    out = REPO / "artifacts/staging_window_model" / args.dataset / args.tag
    out.mkdir(parents=True, exist_ok=True)

    subjects, rows, tracks = load_inputs(args.dataset, cache)
    frame = epoch_frame(rows, WINDOW_FEATURES)
    log(f"{len(subjects)} subjects, {len(frame)} labelled epochs")
    label_structure(frame, out)
    splits = list(build_subject_eval_splits(groups=frame[SUBJ].astype(str).to_numpy(), y=frame[TARGET].to_numpy(),
                                            outer_validation="kfold", outer_folds=5, random_state=SEED))
    folds = [np.asarray(sp.test_idx) for sp in splits]
    y = frame[TARGET].to_numpy()

    cardio_tracks = ["hr", "resp_effort", "resp_env"]
    specs = {  # config -> (tracks, features, context epochs each side)
        "window19": ([], WINDOW_FEATURES, args.context),
        "window33": (TRACKS, WINDOW_FEATURES, args.context),
        "nocardio27": (*nocardio27_signals(), args.context),
        "final21": (*final21_signals(), args.context),
        "g20": (*adopted_signals(), args.context),
        "cardio6": (cardio_tracks, CARDIO3, args.context),
        "cardio6_c5": (cardio_tracks, CARDIO3, 5),
        "cardiofull_c5": (cardio_tracks, CARDIO_FULL, 5),
        "cardiofull_c10": (cardio_tracks, CARDIO_FULL, 10),
    }
    short = lambda c: c.split(".")[-1]  # noqa: E731

    preds = frame[[SUBJ, "epoch_index", TARGET]].copy()
    if (out / "predictions.parquet").exists():  # keep earlier configs' columns (same frame, same order)
        old = pd.read_parquet(out / "predictions.parquet")
        if len(old) == len(preds) and (old["epoch_index"].to_numpy() == preds["epoch_index"].to_numpy()).all():
            keep = [c for c in old.columns if c not in preds.columns and c.split("__")[0] not in configs
                    and not (c.startswith("p_") and ADOPTED in configs)]
            preds = pd.concat([preds, old[keep]], axis=1)
    old_summary = pd.read_csv(out / "summary.csv") if (out / "summary.csv").exists() else pd.DataFrame()
    old_folds = pd.read_csv(out / "folds.csv") if (out / "folds.csv").exists() else pd.DataFrame()
    if len(old_summary):
        old_summary = old_summary[~old_summary["config"].isin(configs)]
        old_folds = old_folds[~old_folds["config"].isin(configs)]
    summary, fold_rows, importance = [], [], []
    rng = np.random.default_rng(SEED)
    for cfg in configs:
        proba, post = np.zeros((len(frame), 5)), np.zeros((len(frame), 5))
        for k, sp in enumerate(splits):
            tr, te = np.asarray(sp.train_idx), np.asarray(sp.test_idx)
            train, test = frame.iloc[tr], frame.iloc[te]
            if cfg == "epoch19":
                cols = WINDOW_FEATURES
                Xtr, Xte = train[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)
                model, p = fit_proba(Xtr, y[tr], Xte, [short(c) for c in cols])
            else:
                trk, feats, ctx = specs[cfg]
                if k == 0:
                    dense = {s: dense_signals(tracks[s], rows[s], trk, feats) for s in subjects}
                enc = WindowEncoder([short(c) for c in trk + feats], context=ctx)
                enc.fit(dense, sorted(set(train[SUBJ])))
                Wtr, Wte = enc.transform(dense, train), enc.transform(dense, test)
                cols = enc.column_names()
                model, p = fit_proba(Wtr.reshape(len(Wtr), -1), y[tr], Wte.reshape(len(Wte), -1), cols)
                del Wtr
                if cfg == ADOPTED and k in imp_folds:
                    offsets = [(t - ctx * enc.steps_per_epoch) * enc.step_seconds for t in range(enc.n_steps)]
                    importance += permutation_importance(model, Wte, y[te], cols, enc.signal_names, offsets, k, rng)
            proba[te] = p
            post[te] = hmm_decode(test, p, transition_model(train))
            log(f"{cfg} fold {k}: epoch-only {f1_score(y[te], p.argmax(1), average='macro'):.4f}, "
                f"+HMM {f1_score(y[te], post[te].argmax(1), average='macro'):.4f}")
        for decoder, P in (("epoch-only", proba), ("hmm", post)):
            row = {"config": cfg, "decoder": decoder, **scores(frame, P.argmax(1), folds)}
            summary.append(row)
            log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
            yt = frame[TARGET].to_numpy()
            for k, idx in enumerate(folds):
                fold_rows.append({"config": cfg, "decoder": decoder, "fold": k,
                                  "macro_f1": f1_score(yt[idx], P[idx].argmax(1), average="macro")})
        preds[f"{cfg}__epoch"] = proba.argmax(1)
        preds[f"{cfg}__hmm"] = post.argmax(1)
        if cfg == ADOPTED:
            for j, st in enumerate(STAGES):
                preds[f"p_{st}"] = post[:, j].astype(np.float32)
        pd.concat([old_summary, pd.DataFrame(summary)], ignore_index=True).to_csv(out / "summary.csv", index=False)
        pd.concat([old_folds, pd.DataFrame(fold_rows)], ignore_index=True).to_csv(out / "folds.csv", index=False)
        preds.to_parquet(out / "predictions.parquet", index=False)
        if importance:
            pd.DataFrame(importance).to_csv(out / "importance.csv", index=False)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
