"""Where in the 150-s window the window model's evidence comes from, without the tie-break artifact.

Seven epoch-grid features hold one value per epoch, so their five 6-s copies are identical and the trees always split
on the first copy. Shuffling one 6-s step therefore credits those features entirely to each epoch's first step.
This script separates the two kinds of input:
  * fine-scale signals (the adopted model's one-second tracks and features that change every 6 s): shuffled one 6-s step at a time;
  * epoch-level features (constant within an epoch): shuffled one epoch (all five copies) at a time.
Macro-F1 drop on each held-out fold, refitting the same model as scripts/run_window_hmm_staging.py.

Writes artifacts/staging_window_model/<dataset>/time_importance.csv
Usage: python scripts/run_window_time_importance.py [--folds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_window_hmm_staging as run  # noqa: E402

EPOCH_LEVEL = [
    "epoch_ratio_features.rel_theta", "epoch_ratio_features.rel_beta", "epoch_ratio_features.eog_rem_vs_noise_db",
    "epoch_ratio_features.eog_rem_vs_sem_db", "epoch_ratio_features.emg_floor_db",
    "topography_features.frontal_slow_wave_fraction", "topography_features.occipital_rel_alpha",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cinc2018")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    out = REPO / "artifacts/staging_window_model" / args.dataset
    subjects, rows, tracks = run.load_inputs(args.dataset, REPO / f"artifacts/{args.dataset}_window_cache")
    frame = run.epoch_frame(rows, run.WINDOW_FEATURES)
    splits = list(run.build_subject_eval_splits(groups=frame[run.SUBJ].astype(str).to_numpy(), y=frame[run.TARGET].to_numpy(),
                                                outer_validation="kfold", outer_folds=5, random_state=run.SEED))
    trk, feats = run.adopted_signals()
    signals = trk + feats
    names = [c.split(".")[-1] for c in signals]
    epoch_idx = [signals.index(c) for c in EPOCH_LEVEL]
    fine_idx = [j for j in range(len(signals)) if j not in epoch_idx]
    dense = {s: run.dense_signals(tracks[s], rows[s], trk, feats) for s in subjects}
    rng = np.random.default_rng(run.SEED)
    records = []
    for k in [int(f) for f in args.folds.split(",")]:
        sp = splits[k]
        train, test = frame.iloc[sp.train_idx], frame.iloc[sp.test_idx]
        enc = run.WindowEncoder(names).fit(dense, sorted(set(train[run.SUBJ])))
        Wtr, Wte = enc.transform(dense, train), enc.transform(dense, test)
        cols = enc.column_names()
        model, p = run.fit_proba(Wtr.reshape(len(Wtr), -1), train[run.TARGET].to_numpy(), Wte.reshape(len(Wte), -1), cols)
        del Wtr
        y = test[run.TARGET].to_numpy()
        base = f1_score(y, p.argmax(1), average="macro")

        def drop(steps, idx):
            Xp = Wte.copy()
            perm = rng.permutation(len(Xp))
            for t in steps:
                Xp[:, t, idx] = Wte[perm][:, t, idx]
            pred = model.predict(pd.DataFrame(Xp.reshape(len(Xp), -1), columns=cols))
            return base - f1_score(y, pred, average="macro")

        spe = enc.steps_per_epoch
        for t in range(enc.n_steps):
            records.append({"fold": k, "group": "fine", "offset_sec": (t - enc.context * spe) * enc.step_seconds,
                            "span_sec": enc.step_seconds, "drop": drop([t], fine_idx)})
        for e in range(2 * enc.context + 1):
            records.append({"fold": k, "group": "epoch", "offset_sec": (e - enc.context) * 30, "span_sec": 30,
                            "drop": drop(range(e * spe, (e + 1) * spe), epoch_idx)})
        run.log(f"time importance fold {k}: base {base:.4f}")
        pd.DataFrame(records).to_csv(out / "time_importance.csv", index=False)
    run.log(f"done -> {out / 'time_importance.csv'}")


if __name__ == "__main__":
    main()
