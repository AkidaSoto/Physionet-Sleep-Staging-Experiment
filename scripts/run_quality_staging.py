"""Adopted window model (nocardio27) with quality-aware tracks and/or quality-based training exclusion.

Same 320 development subjects, same 5 subject-grouped folds (seed 7), same HGB and WindowEncoder as
scripts/run_window_hmm_staging.py. Only two things can change:
  --tracks base     the adopted tracks1s cache (C3/C4, F3/F4, O1/O2 averaged)
  --tracks clean    eeg_quality.quality_weighted_tracks: each pair averaged over its good leads per epoch, chin notched
  --tracks clean_leads   the same lead choice, chin RMS as in base
  --exclude LEVEL   drop epochs that are unusable at LEVEL (scripts/run_quality_report.py unusable_epochs) from the
                    training targets of every fold; they are still predicted, so scoring can use all epochs or not.
The 16 six-second features are unchanged in every variant (they come from the feature pipeline, not from these tracks).

Writes artifacts/exploration/signal-quality/runs/<tag>.parquet: subject_id, epoch_index, label, fold, p_W..p_REM
(epoch-only HGB probabilities). Scores are computed by run_quality_report.py.

Usage: python scripts/run_quality_staging.py --tracks clean [--exclude L2] --tag clean
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import run_window_hmm_staging as W  # noqa: E402
from physionet_sleep.algorithms.eeg_quality import quality_weighted_tracks  # noqa: E402
from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET, WINDOW_FEATURES, WindowEncoder, dense_signals, epoch_frame  # noqa: E402

OUT = REPO / "artifacts/exploration/signal-quality"
QCACHE = REPO / "artifacts/cinc2018_window_cache/quality1s"


def load_frame():
    cache = REPO / "artifacts/cinc2018_window_cache"
    subjects, rows, tracks = W.load_inputs("cinc2018", cache)
    frame = epoch_frame(rows, WINDOW_FEATURES)
    splits = list(W.build_subject_eval_splits(groups=frame[SUBJ].astype(str).to_numpy(), y=frame[TARGET].to_numpy(),
                                              outer_validation="kfold", outer_folds=5, random_state=W.SEED))
    return subjects, rows, tracks, frame, splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", choices=["base", "clean", "clean_leads"], default="base")
    ap.add_argument("--exclude", default="", help="unusable level to drop from training (see run_quality_report.py)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)

    subjects, rows, tracks, frame, splits = load_frame()
    if args.tracks != "base":
        notch = args.tracks == "clean"
        tracks = {s: quality_weighted_tracks(pd.read_parquet(QCACHE / f"{s}.parquet"), emg_notch=notch) for s in subjects}
    drop = np.zeros(len(frame), bool)
    if args.exclude:
        from run_quality_report import epoch_quality, unusable_epochs

        uq = unusable_epochs(epoch_quality(subjects), args.exclude)
        key = frame[[SUBJ, "epoch_index"]].merge(uq, on=[SUBJ, "epoch_index"], how="left")
        drop = key["unusable"].fillna(False).to_numpy(bool)
        W.log(f"excluding {drop.sum()} of {len(drop)} epochs ({drop.mean():.2%}) from training at level {args.exclude}")
    W.log(f"{len(subjects)} subjects, {len(frame)} epochs, tracks={args.tracks}")

    y = frame[TARGET].to_numpy()
    trk, feats = W.adopted_signals()
    short = lambda c: c.split(".")[-1]  # noqa: E731
    dense = {s: dense_signals(tracks[s], rows[s], trk, feats) for s in subjects}
    proba, fold = np.zeros((len(frame), 5)), np.zeros(len(frame), int)
    for k, sp in enumerate(splits):
        tr, te = np.asarray(sp.train_idx), np.asarray(sp.test_idx)
        tr = tr[~drop[tr]]
        train, test = frame.iloc[tr], frame.iloc[te]
        enc = WindowEncoder([short(c) for c in trk + feats], context=2)
        enc.fit(dense, sorted(set(train[SUBJ])))
        Wtr, Wte = enc.transform(dense, train), enc.transform(dense, test)
        _, p = W.fit_proba(Wtr.reshape(len(Wtr), -1), y[tr], Wte.reshape(len(Wte), -1), enc.column_names())
        del Wtr, Wte
        proba[te], fold[te] = p, k
        W.log(f"{args.tag} fold {k}: macro-F1 all epochs {f1_score(y[te], p.argmax(1), average='macro'):.4f}")
    out = frame[[SUBJ, "epoch_index", TARGET]].rename(columns={TARGET: "label"}).copy()
    out["fold"] = fold
    for j, st in enumerate(W.STAGES):
        out[f"p_{st}"] = proba[:, j].astype(np.float32)
    out.to_parquet(OUT / "runs" / f"{args.tag}.parquet", index=False)
    W.log(f"done -> {OUT / 'runs' / (args.tag + '.parquet')}")


if __name__ == "__main__":
    main()
