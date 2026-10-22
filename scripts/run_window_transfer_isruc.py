"""Cross-lab transfer: train the adopted window model on CinC 2018 (MGH), score ISRUC-Sleep subgroup I (Coimbra) once.

ISRUC records the full CinC montage at the same 200 Hz (F3/F4, C3/C4, O1/O2 to the contralateral mastoid, E1-M2,
chin EMG), so the adopted model runs with all its inputs (run.adopted_signals(): g20, 8 one-second tracks,
11 features and spindle_bursts) over the epoch +-2 epochs in 6-s steps, default class-balanced HGB, no decoder. Features come from the same
CinC staging recipe (studies.isruc), tracks from eeg_tracks.isruc_subject_tracks.

Reports, for the same inputs:
  in_lab              5-fold subject-grouped CV within the 320 CinC development subjects
  transfer_scorer1/2  trained on all 320, scored once on all ISRUC nights against each ISRUC scorer
  scorer1_vs_scorer2  the two ISRUC scorers against each other (human agreement on the same epochs)
Writes artifacts/staging_window_model/transfer_isruc/{summary.csv, isruc_subjects.csv, confusion_*.csv,
predictions.parquet}.

This is the single ISRUC look for the final model: the script refuses to run if all nights are not built (--expect)
or if transfer_isruc/single_look.json already exists, and records the config, signals and time there.
The 36-night pilot (nocardio27, 2026-10-21) is kept as *_partial36.

Usage: python scripts/run_window_transfer_isruc.py [--skip-cv] [--expect 100]
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
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_window_hmm_staging as run  # noqa: E402
from physionet_sleep.analysis.epoch_windows import subject_rows6  # noqa: E402
from physionet_sleep.io.isruc import resolve_isruc_record  # noqa: E402
from physionet_sleep.labels.isruc import read_isruc_hypnogram  # noqa: E402

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
OUT = REPO / "artifacts/staging_window_model/transfer_isruc"


def isruc_inputs():
    spec = run.DATASETS["isruc"]
    cache = REPO / "artifacts/isruc_window_cache"
    rows, tracks = {}, {}
    for s in spec.built_subjects():
        tp = cache / "tracks1s" / f"{s}.parquet"
        if not tp.exists():
            continue
        rows[s] = subject_rows6("isruc", s, run.WINDOW_FEATURES, cache / "rows6")
        tracks[s] = pd.read_parquet(tp)
    return rows, tracks


def score(y, p, subj) -> tuple[dict, dict]:
    per = {s: f1_score(y[subj == s], p[subj == s], labels=sorted(set(y[subj == s])), average="macro") for s in np.unique(subj)}
    return {"macro_f1": float(f1_score(y, p, average="macro")), "kappa": float(cohen_kappa_score(y, p)),
            "accuracy": float(accuracy_score(y, p)), "n_epochs": int(len(y)), "n_subjects": int(len(per)),
            **{f"f1_{s}": float(v) for s, v in zip(STAGES, f1_score(y, p, labels=range(5), average=None))},
            "subj_median": float(np.median(list(per.values()))), "subj_min": float(min(per.values()))}, per


def scorer2_labels(frame: pd.DataFrame) -> np.ndarray:
    """Scorer 2's stage for each epoch of the frame (-1 where scorer 2 left it unscored)."""
    raw = run.DATASETS["isruc"].raw_dir
    out = np.full(len(frame), -1)
    subj = frame[run.SUBJ].astype(str).to_numpy()
    ei = frame["epoch_index"].to_numpy()
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        h = read_isruc_hypnogram(resolve_isruc_record(raw, s), scorer=2)
        ok = ei[idx] < len(h)
        out[idx[ok]] = h[ei[idx[ok]]]
    return out


def confusion(y, p, name: str) -> None:
    cm = pd.DataFrame(confusion_matrix(y, p, labels=range(5)), index=[f"true_{s}" for s in STAGES], columns=[f"pred_{s}" for s in STAGES])
    cm.to_csv(OUT / f"confusion_{name}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true", help="skip the in-lab CinC cross-validation")
    ap.add_argument("--expect", type=int, default=100, help="ISRUC nights that must be built")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    look = OUT / "single_look.json"
    if look.exists():
        raise SystemExit(f"ISRUC has already been scored once ({look}); not scoring again")
    tracks_used, features = run.adopted_signals()
    short = lambda c: c.split(".")[-1]  # noqa: E731
    names = [short(c) for c in tracks_used + features]
    cinc_subjects, rows_c, tracks_c = run.load_inputs("cinc2018", REPO / "artifacts/cinc2018_window_cache")
    holdout = set((REPO / "data/raw/cinc2018/selected_records_holdout.txt").read_text().split())
    assert not holdout & set(cinc_subjects), "held-out CinC records in the training set"
    rows_i, tracks_i = isruc_inputs()
    if len(rows_i) < args.expect:
        raise SystemExit(f"only {len(rows_i)} of {args.expect} ISRUC nights built; not scoring")
    look.write_text(json.dumps({"config": run.ADOPTED, "tracks": tracks_used, "features": features,
                                "isruc_nights": sorted(rows_i, key=int), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    frame_c = run.epoch_frame(rows_c, run.WINDOW_FEATURES)
    frame_i = run.epoch_frame(rows_i, run.WINDOW_FEATURES)
    dense = {s: run.dense_signals(tracks_c[s], rows_c[s], tracks_used, features) for s in cinc_subjects}
    dense.update({s: run.dense_signals(tracks_i[s], rows_i[s], tracks_used, features) for s in rows_i})
    y_c, y_i = frame_c[run.TARGET].to_numpy(), frame_i[run.TARGET].to_numpy()
    run.log(f"inputs: {len(names)} signals x 25 steps; CinC {len(cinc_subjects)} subjects / {len(frame_c)} epochs; "
            f"ISRUC {len(rows_i)} subjects / {len(frame_i)} epochs")
    results = []

    if not args.skip_cv:
        splits = list(run.build_subject_eval_splits(groups=frame_c[run.SUBJ].astype(str).to_numpy(), y=y_c,
                                                    outer_validation="kfold", outer_folds=5, random_state=run.SEED))
        pred_c = np.zeros(len(frame_c), dtype=int)
        for k, sp in enumerate(splits):
            tr, te = np.asarray(sp.train_idx), np.asarray(sp.test_idx)
            enc = run.WindowEncoder(names).fit(dense, sorted(set(frame_c.iloc[tr][run.SUBJ])))
            Wtr, Wte = enc.transform(dense, frame_c.iloc[tr]), enc.transform(dense, frame_c.iloc[te])
            _, p = run.fit_proba(Wtr.reshape(len(Wtr), -1), y_c[tr], Wte.reshape(len(Wte), -1), enc.column_names())
            pred_c[te] = p.argmax(1)
            run.log(f"in-lab fold {k}: {f1_score(y_c[te], pred_c[te], average='macro'):.4f}")
            del Wtr, Wte
        folds = [f1_score(y_c[np.asarray(sp.test_idx)], pred_c[np.asarray(sp.test_idx)], average="macro") for sp in splits]
        in_lab, _ = score(y_c, pred_c, frame_c[run.SUBJ].astype(str).to_numpy())
        in_lab.update(evaluation="in_lab_cinc_cv", pooled_macro_f1=in_lab["macro_f1"], macro_f1=float(np.mean(folds)),
                      folds=" ".join(f"{v:.3f}" for v in folds))
        confusion(y_c, pred_c, "in_lab_cinc")
        results.append(in_lab)

    # transfer: all 320 CinC development subjects -> every ISRUC night, scored once
    enc = run.WindowEncoder(names).fit(dense, cinc_subjects)
    Wtr = enc.transform(dense, frame_c)
    Wte = enc.transform(dense, frame_i)
    _, proba = run.fit_proba(Wtr.reshape(len(Wtr), -1), y_c, Wte.reshape(len(Wte), -1), enc.column_names())
    del Wtr, Wte
    pred = proba.argmax(1)
    subj = frame_i[run.SUBJ].astype(str).to_numpy()
    y2 = scorer2_labels(frame_i)
    both = y2 >= 0

    t1, per1 = score(y_i, pred, subj)
    t1["evaluation"] = "transfer_cinc_to_isruc_scorer1"
    t2, per2 = score(y2[both], pred[both], subj[both])
    t2["evaluation"] = "transfer_cinc_to_isruc_scorer2"
    hh, perh = score(y_i[both], y2[both], subj[both])
    hh["evaluation"] = "isruc_scorer1_vs_scorer2"
    t1b, _ = score(y_i[both], pred[both], subj[both])
    t1b["evaluation"] = "transfer_cinc_to_isruc_scorer1_on_epochs_both_scored"
    agree = both & (y_i == y2)
    tc, _ = score(y_i[agree], pred[agree], subj[agree])
    tc["evaluation"] = "transfer_on_epochs_where_scorers_agree"
    results += [t1, t1b, t2, hh, tc]
    for r in results:
        run.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    for r in results:
        r["config"] = run.ADOPTED
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)
    confusion(y_i, pred, "transfer_scorer1")
    confusion(y2[both], pred[both], "transfer_scorer2")
    confusion(y_i[both], y2[both], "scorer1_vs_scorer2")
    pd.DataFrame([{"subject_id": s, "macro_f1_scorer1": per1[s], "macro_f1_scorer2": per2.get(s, np.nan),
                   "macro_f1_scorer1_vs_scorer2": perh.get(s, np.nan), "n_epochs": int((subj == s).sum())}
                  for s in per1]).sort_values("subject_id", key=lambda c: c.astype(int)).to_csv(OUT / "isruc_subjects.csv", index=False)
    pred_df = frame_i[[run.SUBJ, "epoch_index"]].copy()
    pred_df["scorer1"], pred_df["scorer2"], pred_df["pred"] = y_i, y2, pred
    for k, st in enumerate(STAGES):
        pred_df[f"p_{st}"] = proba[:, k].astype(np.float32)
    pred_df.to_parquet(OUT / "predictions.parquet", index=False)
    run.log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
