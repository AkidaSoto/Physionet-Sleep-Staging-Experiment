"""Cross-lab transfer: train the window model on CinC 2018 (MGH), test on UCDDB (Dublin) without retraining.

Uses the adopted model's signals that both datasets record: 6 one-second tracks (central EEG bands and power, fast EOG)
and 10 features (the adopted model's 12 minus the two frontal/occipital ones, spindle_bursts included). A first run that also used
heart rate and breathing scored 0.769 in-lab and 0.754 on UCDDB (2026-10-21). UCDDB differs in lab, montage (two EOG channels, no F/O leads), sample rates
(EEG 128 Hz, EOG 64 Hz, effort 8 Hz), amplifier calibration and scoring rules (R&K, S3+S4 merged to N3).

Reports, for the same input set:
  in_lab   5-fold subject-grouped CV within CinC (what the model does on unseen patients from the same lab)
  transfer trained on all CinC development subjects, scored once on all UCDDB subjects
Writes artifacts/staging_window_model/transfer/{summary.csv, ucddb_subjects.csv}.

Usage: python scripts/run_window_transfer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_window_hmm_staging as run  # noqa: E402
from physionet_sleep.algorithms.eeg_tracks import TRANSFER_TRACKS, ucddb_subject_tracks  # noqa: E402
from physionet_sleep.analysis.epoch_windows import subject_rows6  # noqa: E402

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
# The adopted model's signals that both labs record: no frontal/occipital leads, no heart or breathing.
# spindle_bursts is built from the central EEG tracks, so UCDDB has it too.
SHARED_TRACKS = [t for t in run.adopted_signals()[0] if t in TRANSFER_TRACKS]
FEATURES = [c for c in run.adopted_signals()[1] if not c.startswith("topography_features.")]


def ucddb_inputs():
    spec = run.DATASETS["ucddb"]
    cache = REPO / "artifacts/ucddb_window_cache"
    rows, tracks = {}, {}
    for s in spec.built_subjects():
        rows[s] = subject_rows6("ucddb", s, run.WINDOW_FEATURES, cache / "rows6")
        tp = cache / "tracks1s" / f"{s}.parquet"
        if not tp.exists():
            tp.parent.mkdir(parents=True, exist_ok=True)
            ucddb_subject_tracks(spec.raw_dir, spec.cache_dir, s).to_parquet(tp, index=False)
        tracks[s] = pd.read_parquet(tp)
    return rows, tracks


def score(y, p, subj) -> dict:
    per = {s: f1_score(y[subj == s], p[subj == s], labels=sorted(set(y[subj == s])), average="macro") for s in np.unique(subj)}
    return {"macro_f1": float(f1_score(y, p, average="macro")), "kappa": float(cohen_kappa_score(y, p)),
            "accuracy": float(accuracy_score(y, p)),
            **{f"f1_{s}": float(v) for s, v in zip(STAGES, f1_score(y, p, labels=range(5), average=None))},
            "subj_median": float(np.median(list(per.values()))), "subj_min": float(min(per.values()))}, per


def main() -> None:
    out = REPO / "artifacts/staging_window_model/transfer"
    out.mkdir(parents=True, exist_ok=True)
    short = lambda c: c.split(".")[-1]  # noqa: E731
    names = [short(c) for c in SHARED_TRACKS + FEATURES]
    cinc_subjects, rows_c, tracks_c = run.load_inputs("cinc2018", REPO / "artifacts/cinc2018_window_cache")
    rows_u, tracks_u = ucddb_inputs()
    frame_c = run.epoch_frame(rows_c, run.WINDOW_FEATURES)
    frame_u = run.epoch_frame(rows_u, run.WINDOW_FEATURES)
    dense = {s: run.dense_signals(tracks_c[s], rows_c[s], SHARED_TRACKS, FEATURES) for s in cinc_subjects}
    dense.update({s: run.dense_signals(tracks_u[s], rows_u[s], SHARED_TRACKS, FEATURES) for s in rows_u})
    y_c, y_u = frame_c[run.TARGET].to_numpy(), frame_u[run.TARGET].to_numpy()
    run.log(f"CinC {len(cinc_subjects)} subjects / {len(frame_c)} epochs; UCDDB {len(rows_u)} subjects / {len(frame_u)} epochs")

    # in-lab reference: 5-fold CV on CinC with the same inputs
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
    in_lab.update(evaluation="in_lab_cinc_cv", macro_f1=float(np.mean(folds)), folds=" ".join(f"{v:.3f}" for v in folds))

    # transfer: all CinC -> UCDDB
    enc = run.WindowEncoder(names).fit(dense, cinc_subjects)
    Wtr, Wte = enc.transform(dense, frame_c), enc.transform(dense, frame_u)
    _, p = run.fit_proba(Wtr.reshape(len(Wtr), -1), y_c, Wte.reshape(len(Wte), -1), enc.column_names())
    transfer, per = score(y_u, p.argmax(1), frame_u[run.SUBJ].astype(str).to_numpy())
    transfer["evaluation"] = "transfer_cinc_to_ucddb"
    rows = [in_lab, transfer]
    for r in rows:
        run.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame([{"subject_id": s, "macro_f1": v} for s, v in per.items()]).to_csv(out / "ucddb_subjects.csv", index=False)
    run.log(f"done -> {out}")


if __name__ == "__main__":
    main()
