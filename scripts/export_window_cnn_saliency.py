"""Export attention-CNN explanations for the dashboard (showcase/content/generated/staging-saliency.json).

Uses the 320-subject attention CNN (14 one-second tracks + 16 staging features, target epoch +-1, two seeds) trained
in artifacts/exploration/cinc-context, evaluated on fold 0's held-out subjects:
  * share of |gradient x input| per signal and expert stage (what the network leans on for each stage);
  * mean attention per second for each stage (where in the 90-s window it looks);
  * one confident, correct example epoch per stage: z-scored input, gradient x input, attention.

Usage: python scripts/export_window_cnn_saliency.py [--fold 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.algorithms.eeg_tracks import TRACKS  # noqa: E402
from physionet_sleep.analysis.datasets import DATASETS  # noqa: E402
from physionet_sleep.analysis.epoch_windows import (  # noqa: E402
    BASE16,
    SUBJ,
    TARGET,
    WINDOW_FEATURES,
    WindowEncoder,
    dense_signals,
    epoch_frame,
    subject_rows6,
)
from physionet_sleep.experiments.tabular_runner import build_subject_eval_splits  # noqa: E402
from physionet_sleep.experiments.window_cnn import EpochNet, explain  # noqa: E402

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
WEIGHTS = REPO / "artifacts/exploration/cinc-context/cache"
CACHE = REPO / "artifacts/cinc2018_window_cache"
OUT = REPO / "showcase/content/generated/staging-saliency.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--examples", action="store_true", help="also export one heatmap example per stage")
    args = ap.parse_args()
    dev = REPO / "artifacts/staging_window_model/cinc2018/dev_subjects.txt"
    subjects = dev.read_text().split() if dev.exists() else DATASETS["cinc2018"].built_subjects()
    rows = {s: subject_rows6("cinc2018", s, WINDOW_FEATURES, CACHE / "rows6") for s in subjects}
    frame = epoch_frame(rows, WINDOW_FEATURES)
    sp = list(build_subject_eval_splits(groups=frame[SUBJ].astype(str).to_numpy(), y=frame[TARGET].to_numpy(),
                                        outer_validation="kfold", outer_folds=5, random_state=7))[args.fold]
    train, test = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx].reset_index(drop=True)
    names = TRACKS + [c.split(".")[-1] for c in BASE16]
    dense = {s: dense_signals(pd.read_parquet(CACHE / "tracks1s" / f"{s}.parquet"), rows[s], TRACKS, BASE16) for s in subjects}
    enc = WindowEncoder(names, context=1, step_seconds=1).fit(dense, sorted(set(train[SUBJ])))
    X = enc.transform({s: dense[s] for s in set(test[SUBJ])}, test)
    y = test[TARGET].to_numpy()

    models = []
    for seed in (0, 1):
        m = EpochNet(len(names))
        m.load_state_dict(torch.load(WEIGHTS / f"model_320_p2+16_c1_cnn_fold{args.fold}_seed{seed}.pt"))
        m.eval()
        models.append(m)

    prob, att, share = np.zeros((len(y), 5)), np.zeros((len(y), enc.n_steps)), np.zeros((len(y), len(names)))
    for b in np.array_split(np.arange(len(y)), max(1, len(y) // 1024)):
        p, a, s = explain(models, X[b])
        prob[b], att[b] = p, a
        mag = np.abs(s).sum(1)
        share[b] = mag / np.maximum(mag.sum(1, keepdims=True), 1e-12)
    pred = prob.argmax(1)
    correct = pred == y
    print(f"fold {args.fold}: {len(y)} test epochs, accuracy {correct.mean():.3f}")

    seconds = [int(t - enc.context * 30) for t in range(enc.n_steps)]
    by_stage, examples = [], []
    with_examples = args.examples
    for k, st in enumerate(STAGES):
        sel = (y == k) & correct
        by_stage.append({
            "stage": st,
            "epochs": int(sel.sum()),
            "signal_share": np.round(share[sel].mean(0), 4).tolist(),
            "attention": np.round(att[sel].mean(0), 5).tolist(),
        })
        if not with_examples:
            continue
        # The most typical confident example: the confident, correct epoch whose signal shares are closest to the stage mean.
        cand = np.flatnonzero(sel & (prob.max(1) > 0.8))
        centre = share[sel].mean(0)
        i = int(cand[np.argmin(np.abs(share[cand] - centre).sum(1))])
        _, a, s = explain(models, X[i : i + 1])
        examples.append({
            "stage": st,
            "record_id": str(test[SUBJ].iloc[i]),
            "epoch_start_sec": float(test["epoch_index"].iloc[i] * 30),
            "probability": round(float(prob[i].max()), 3),
            "input": np.round(np.clip(X[i].T, -3, 3), 2).tolist(),
            "saliency": np.round(s[0].T / max(np.abs(s[0]).max(), 1e-9), 3).tolist(),
            "attention": np.round(a[0], 5).tolist(),
        })
    payload = {
        "model": "Attention CNN, 14 one-second tracks + 16 staging features, target epoch ±1 (90 s), 2 seeds",
        "fold": args.fold,
        "test_subjects": int(test[SUBJ].nunique()),
        "test_epochs": int(len(y)),
        "signals": names,
        "seconds": seconds,
        "by_stage": by_stage,
        "examples": examples,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
