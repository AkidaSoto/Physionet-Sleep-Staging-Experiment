"""Shared pieces of the no-EEG (heart + breathing) staging experiments on the CinC 2018 development subjects.

Same labelled epochs, subject-grouped folds (5, seed 7) and HMM decoder as scripts/run_window_hmm_staging.py.

Per subject we build an epoch grid covering the whole recording (labelled or not, so context is never cut), with
per-epoch summaries scaled by the subject's median and IQR:
  base    the 3 heart/breathing 1-s tracks (hr, resp_effort, resp_env): mean and SD within the epoch, plus the epoch
          mean of the 9 heart/breathing 6-s features (CARDIO_FULL) -> 15 columns
  raw     summaries recomputed from raw ECG R-peaks, effort belts and SaO2 (scripts/build_noeeg_cache.py)
Night position is kept apart (not scaled): hours since recording start, fraction of the recording, hours to its end.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from physionet_sleep.analysis.epoch_windows import (
    SUBJ,
    TARGET,
    WINDOW_FEATURES,
    dense_signals,
    epoch_frame,
    robust_scale,
    subject_rows6,
)
from physionet_sleep.decoders.hmm import learn_transition_matrix, posterior_decode
from physionet_sleep.experiments.tabular_runner import build_subject_eval_splits

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "artifacts/cinc2018_window_cache"
NOEEG = REPO / "artifacts/exploration/no-eeg"
RAW_CACHE = NOEEG / "cache"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
STAGES4 = ["Wake", "Light", "Deep", "REM"]
TO4 = np.array([0, 1, 1, 2, 3])
SEED = 7
CARDIO_TRACKS = ["hr", "resp_effort", "resp_env"]
CARDIO_FULL = [
    "hrv_features.hr_bpm_30s", "hrv_features.hr_surge_30s", "hrv_features.sdnn_150s", "hrv_features.hr_slow_sd_300s",
    "hrv_features.dfa_a2_300s", "hrv_features.ln_lf_hf_300s", "breath_regularity.breath_interval_cv_60s",
    "breath_regularity.breath_amp_cv_60s", "cardiorespiratory_coupling.hr_resp_coherence_120s",
]
RAW_COLS = ["hr_mean", "hr_sd", "hr_min", "hr_max", "ln_rmssd", "ln_sdnn", "pnn50", "ln_lf", "ln_hf", "lf_hf",
            "br_rate", "br_interval_cv", "br_amp_log", "br_amp_cv", "br_count", "resp_peak_hz", "abd_chest_corr",
            "effort_env_log", "effort_move", "effort_hf", "ecg_noise", "spo2_mean", "spo2_min", "spo2_sd", "spo2_drop"]
SPO2_COLS = ["spo2_mean", "spo2_min", "spo2_sd", "spo2_drop"]
RAW_GROUPS = {
    "heart": RAW_COLS[:10],
    "breath": ["br_rate", "br_interval_cv", "br_amp_log", "br_amp_cv", "br_count", "resp_peak_hz", "abd_chest_corr",
               "effort_env_log"],
    "move": ["effort_move", "effort_hf", "ecg_noise"],
    "spo2": SPO2_COLS,
}
POSITION = ["pos_hours", "pos_frac", "pos_hours_left"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def development_subjects() -> list[str]:
    subjects = (REPO / "artifacts/staging_window_model/cinc2018/dev_subjects.txt").read_text().split()
    hold = REPO / "data/raw/cinc2018/selected_records_holdout.txt"
    held = set(hold.read_text().split()) if hold.exists() else set()
    assert not held & set(subjects), "held-out records in the development list"
    return subjects


def load_rows_tracks(subjects: list[str]):
    rows, tracks = {}, {}
    for s in subjects:
        rows[s] = subject_rows6("cinc2018", s, WINDOW_FEATURES + [c for c in CARDIO_FULL if c not in WINDOW_FEATURES],
                                CACHE / "rows6")
        tracks[s] = pd.read_parquet(CACHE / "tracks1s" / f"{s}.parquet")
    return rows, tracks


def labelled_frame(rows: dict) -> tuple[pd.DataFrame, list]:
    """The window script's labelled epochs and its 5 subject-grouped folds (seed 7)."""
    frame = epoch_frame(rows, WINDOW_FEATURES)
    splits = list(build_subject_eval_splits(groups=frame[SUBJ].astype(str).to_numpy(), y=frame[TARGET].to_numpy(),
                                            outer_validation="kfold", outer_folds=5, random_state=SEED))
    return frame, splits


def n_epochs(tracks: pd.DataFrame) -> int:
    return len(tracks) // 30


def base_epoch_summaries(tracks: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(n_epochs, 15): epoch mean and SD of the 3 subject-scaled tracks, epoch mean of the 9 scaled features."""
    D = dense_signals(tracks, rows, CARDIO_TRACKS, CARDIO_FULL)
    ne = n_epochs(tracks)
    E = D[: ne * 30].reshape(ne, 30, D.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(E, 1)
        sd = np.nanstd(E[:, :, :3], 1)
    names = [f"{c}_mean" for c in CARDIO_TRACKS] + [f"{c}_sd" for c in CARDIO_TRACKS] + [c.split(".")[-1] for c in CARDIO_FULL]
    return np.hstack([mean[:, :3], sd, mean[:, 3:]]).astype(np.float32), names


def raw_epoch_summaries(sid: str, ne: int, cols: list[str] = RAW_COLS) -> np.ndarray:
    """(n_epochs, len(cols)) raw-derived summaries, subject-scaled (median/IQR), aligned to the epoch grid."""
    df = pd.read_parquet(RAW_CACHE / "epochs" / f"{sid}.parquet")
    X = np.full((ne, len(cols)), np.nan, np.float32)
    ei = df["epoch_index"].to_numpy()
    ok = ei < ne
    X[ei[ok]] = df[cols].to_numpy(np.float32)[ok]
    return robust_scale(X, cols)


def position_features(ne: int) -> np.ndarray:
    e = np.arange(ne, dtype=np.float32)
    return np.stack([e / 120, e / max(ne - 1, 1), (ne - 1 - e) / 120], 1).astype(np.float32)


# ---------------------------------------------------------------- multi-scale context for trees
def block_means(S: np.ndarray, center: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of S over epochs [center+lo, center+hi] (inclusive), NaN-aware, NaN outside the night."""
    ne = len(S)
    cs = np.vstack([np.zeros((1, S.shape[1])), np.nancumsum(S, 0)])
    cn = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(np.isfinite(S), 0)])
    a = np.clip(center + lo, 0, ne)
    b = np.clip(center + hi + 1, 0, ne)
    tot, cnt = cs[b] - cs[a], cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32)


def context_matrix(S: np.ndarray, names: list[str], epochs: np.ndarray, scales: list[tuple[int, int, int]]):
    """Features for the given epochs: for each (start, stop, width), blocks of `width` epochs from offset start to stop
    on both sides (width 1 = single epochs, start 0 includes the epoch itself once)."""
    cols, parts = [], []
    for start, stop, width in scales:
        offs = []
        for lo in range(start, stop + 1, width):
            offs.append((lo, lo + width - 1))
            if lo > 0:
                offs.append((-(lo + width - 1), -lo))
        for lo, hi in sorted(set(offs)):
            parts.append(block_means(S, epochs, lo, hi))
            tag = f"{lo:+d}" if lo == hi else f"{lo:+d}..{hi:+d}"
            cols += [f"{n}@{tag}" for n in names]
    return np.hstack(parts), cols


# ---------------------------------------------------------------- decoding and scores
def nights(frame: pd.DataFrame):
    subj = frame[SUBJ].astype(str).to_numpy()
    order = np.lexsort((frame["epoch_index"].to_numpy(), subj))
    s = subj[order]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield order[a:b]


def hmm_decode(train: pd.DataFrame, test: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    y = train[TARGET].to_numpy()
    model = learn_transition_matrix([y[idx] for idx in nights(train)])
    post = np.zeros_like(proba)
    for idx in nights(test):
        post[idx] = posterior_decode(proba[idx], model)
    return post


def scores(y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray]) -> dict:
    f1s = [f1_score(y[i], pred[i], average="macro") for i in folds]
    k4 = [cohen_kappa_score(TO4[y[i]], TO4[pred[i]]) for i in folds]
    row = {"macro_f1": float(np.mean(f1s)), "folds": " ".join(f"{v:.3f}" for v in f1s),
           "kappa5": float(np.mean([cohen_kappa_score(y[i], pred[i]) for i in folds])),
           "kappa4": float(np.mean(k4)), "kappa4_folds": " ".join(f"{v:.3f}" for v in k4),
           "acc4": float(np.mean(TO4[y] == TO4[pred]))}
    for st, v in zip(STAGES, f1_score(y, pred, labels=range(5), average=None, zero_division=0)):
        row[f"f1_{st}"] = float(v)
    for st, v in zip(STAGES4, f1_score(TO4[y], TO4[pred], labels=range(4), average=None, zero_division=0)):
        row[f"f1_4{st}"] = float(v)
    return row


def table_line(date: str, variant: str, decoder: str, row: dict) -> str:
    return (f"| {date} | {variant} | {decoder} | {row['macro_f1']:.3f} | {row['folds']} | {row['kappa4']:.3f} | "
            + " | ".join(f"{row[f'f1_{s}']:.3f}" for s in STAGES) + " |")


# ---------------------------------------------------------------- cached per-night inputs for the sequence models
SEQ_CACHE = RAW_CACHE / "sequence_inputs.pkl"


def sequence_inputs() -> dict:
    """{'frame': labelled epochs (subject, epoch_index, label), 'splits': [(train_idx, test_idx)] x 5,
    'subjects': [...], 'blocks': {subject: {'base', 'fine', 'raw', 'pos': (n_epochs, k) arrays}}}, cached as a pickle."""
    import pickle

    if SEQ_CACHE.exists():
        with open(SEQ_CACHE, "rb") as fh:
            return pickle.load(fh)
    subjects = development_subjects()
    rows, tracks = load_rows_tracks(subjects)
    frame, splits = labelled_frame(rows)
    blocks = {}
    for s in subjects:
        nep = n_epochs(tracks[s])
        D = dense_signals(tracks[s], rows[s], CARDIO_TRACKS, [])[: nep * 30]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fine = np.nanmean(D.reshape(nep, 5, 6, 3), 2).reshape(nep, 15)
        blocks[s] = {"base": base_epoch_summaries(tracks[s], rows[s])[0], "fine": fine.astype(np.float32),
                     "raw": raw_epoch_summaries(s, nep), "pos": position_features(nep)}
    data = {"frame": frame[[SUBJ, "epoch_index", TARGET]].reset_index(drop=True),
            "splits": [(np.asarray(sp.train_idx), np.asarray(sp.test_idx)) for sp in splits],
            "subjects": subjects, "blocks": blocks}
    with open(SEQ_CACHE, "wb") as fh:
        pickle.dump(data, fh)
    return data
