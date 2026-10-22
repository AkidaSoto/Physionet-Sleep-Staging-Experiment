"""No-EEG staging with gradient boosting over multi-scale context (heart + breathing only, CinC 2018 dev subjects).

Every variant: HistGradientBoosting (balanced; defaults except early_stopping=False with the default 100 iterations,
which avoids the internal validation copy that does not fit in memory on this shared machine), same 5 subject-grouped folds (seed 7) and HMM decoder as
scripts/run_window_hmm_staging.py. Inputs (src/physionet_sleep/experiments/noeeg.py):
  fine   6-s steps of the 3 tracks over the epoch +-1 epoch (15 steps x 3; the 9 features are 30-300-s smooth, so
         their epoch summaries carry them)
  ctx    per-epoch summaries (15 base, optionally + raw) as single epochs to +-10, 1-min blocks to +-20,
         2.5-min blocks to +-40 epochs (depending on the variant)
  pos    night position (hours since start, fraction, hours to end)

Variants:
  ms10        fine + single epochs +-10
  ms20        ms10 + 1-min blocks 11..20
  ms40        ms20 + 2.5-min blocks 21..40
  ms40pos     ms40 + position
  ms40pos_raw ms40pos + raw-derived summaries at the same scales (incl. SpO2)
  ms40pos_raw_nospo2  same without the 4 SpO2 columns
  raw40pos    raw-derived summaries only (no fine steps, no base) + position

Outputs: artifacts/exploration/no-eeg/trees/summary.csv (appended), proba_<variant>.npz (epoch-only and HMM posteriors).
Usage: python scripts/run_noeeg_trees.py --variants ms10,ms20,ms40,ms40pos
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET, dense_signals  # noqa: E402
from physionet_sleep.experiments import noeeg as ne  # noqa: E402

OUT = ne.NOEEG / "trees"
SCALES = {"ms10": [(0, 10, 1)], "ms20": [(0, 10, 1), (11, 20, 2)], "ms40": [(0, 10, 1), (11, 20, 2), (21, 40, 5)]}


def fine_steps(D: np.ndarray, epochs: np.ndarray, context: int = 2) -> np.ndarray:
    n6 = -(-len(D) // 6)
    Dp = np.vstack([D, np.full((n6 * 6 - len(D), D.shape[1]), np.nan, np.float32)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S6 = np.nanmean(Dp.reshape(n6, 6, D.shape[1]), 1)
    pad = (context + 1) * 5
    S6 = np.vstack([np.full((pad, D.shape[1]), np.nan), S6, np.full((pad, D.shape[1]), np.nan)]).astype(np.float32)
    start = pad + (epochs - context) * 5
    idx = start[:, None] + np.arange((2 * context + 1) * 5)[None, :]
    return S6[np.clip(idx, 0, len(S6) - 1)].reshape(len(epochs), -1)


def build_features(variant: str, frame: pd.DataFrame, rows: dict, tracks: dict) -> tuple[np.ndarray, list[str]]:
    use_raw = "raw" in variant
    base = not variant.startswith("raw40")
    scales = SCALES["ms40" if "40" in variant else variant[:4]]
    raw_cols = [c for c in ne.RAW_COLS if not ("nospo2" in variant and c in ne.SPO2_COLS)]
    subj = frame[SUBJ].astype(str).to_numpy()
    blocks, cols = [None] * len(frame), None
    X = None
    for s in pd.unique(subj):
        idx = np.flatnonzero(subj == s)
        ep = frame["epoch_index"].to_numpy()[idx]
        nep = ne.n_epochs(tracks[s])
        parts, names = [], []
        if base:
            D = dense_signals(tracks[s], rows[s], ne.CARDIO_TRACKS, [])
            parts.append(fine_steps(D, ep, context=1))
            names += [f"{n}@{(t - 5) * 6:+d}s" for t in range(15) for n in ne.CARDIO_TRACKS]
            S, sn = ne.base_epoch_summaries(tracks[s], rows[s])
        else:
            S, sn = np.zeros((nep, 0), np.float32), []
        if use_raw:
            S = np.hstack([S, ne.raw_epoch_summaries(s, nep, raw_cols)])
            sn = sn + raw_cols
        C, cn = ne.context_matrix(S, sn, ep, scales)
        parts.append(C)
        names += cn
        if "pos" in variant:
            parts.append(ne.position_features(nep)[np.clip(ep, 0, nep - 1)])
            names += ne.POSITION
        M = np.hstack(parts).astype(np.float32)
        if X is None:
            X = np.empty((len(frame), M.shape[1]), np.float32)
            cols = names
        X[idx] = M
    return X, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="ms10,ms20,ms40,ms40pos")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = ne.development_subjects()
    rows, tracks = ne.load_rows_tracks(subjects)
    frame, splits = ne.labelled_frame(rows)
    y = frame[TARGET].to_numpy()
    folds = [np.asarray(sp.test_idx) for sp in splits]
    ne.log(f"{len(subjects)} subjects, {len(frame)} labelled epochs")
    for variant in args.variants.split(","):
        t0 = time.time()
        X, cols = build_features(variant, frame, rows, tracks)
        ne.log(f"{variant}: {X.shape[1]} features")
        proba, post = np.zeros((len(frame), 5)), np.zeros((len(frame), 5))
        for k, sp in enumerate(splits):
            tr, te = np.asarray(sp.train_idx), np.asarray(sp.test_idx)
            model = HistGradientBoostingClassifier(random_state=ne.SEED, class_weight="balanced", early_stopping=False)
            model.fit(X[tr], y[tr])
            p = np.zeros((len(te), 5))
            p[:, model.classes_.astype(int)] = model.predict_proba(X[te])
            proba[te] = p
            post[te] = ne.hmm_decode(frame.iloc[tr], frame.iloc[te], p)
            ne.log(f"{variant} fold {k}: epoch-only {f1_score(y[te], p.argmax(1), average='macro'):.4f}, "
                   f"+HMM {f1_score(y[te], post[te].argmax(1), average='macro'):.4f} ({model.n_iter_} iters)")
        out = []
        for decoder, P in (("epoch-only", proba), ("hmm", post)):
            row = {"date": time.strftime("%Y-%m-%d"), "variant": variant, "decoder": decoder, "n_features": X.shape[1],
                   **ne.scores(y, P.argmax(1), folds), "minutes": (time.time() - t0) / 60}
            out.append(row)
            print(ne.table_line(row["date"], variant, decoder, row), flush=True)
        path = OUT / "summary.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(out)], ignore_index=True).to_csv(path, index=False)
        np.savez_compressed(OUT / f"proba_{variant}.npz", proba=proba.astype(np.float32), post=post.astype(np.float32))
    ne.log("done")


if __name__ == "__main__":
    main()
