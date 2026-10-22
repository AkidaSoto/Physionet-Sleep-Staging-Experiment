"""Final held-out test: train on all 320 development subjects, score once on records never used in development.

The development set is artifacts/staging_window_model/cinc2018/dev_subjects.txt; the held-out set is
data/raw/cinc2018/selected_records_holdout.txt (scripts/select_cinc2018_holdout.py), restricted to records whose
recipe cache is built. Normalisation statistics come from the development subjects only. Nothing here was tuned on
the held-out records.

Configs, scored together as ONE look at the held-out set: nocardio27 (adopted), window33 (with heart/breathing),
epoch19 (single-epoch reference), window33 + decoder. Every run appends to holdout/LEDGER.csv so the number of looks
is on record; keep it to one per adopted-model change. Only headline metrics are written (no per-subject or
per-condition breakdown), so the held-out set is not mined for model choices.
Writes artifacts/staging_window_model/cinc2018/holdout/{summary.csv, LEDGER.csv}.

Usage: python scripts/run_window_holdout.py
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

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
CARDIO_TRACKS = ["hr", "resp_effort", "resp_env"]


CONFIGS = sys.argv[1].split(",") if len(sys.argv) > 1 else [run.ADOPTED]


def main() -> None:
    out = REPO / "artifacts/staging_window_model/cinc2018/holdout"
    out.mkdir(parents=True, exist_ok=True)
    spec = run.DATASETS["cinc2018"]
    dev = run.development_subjects("cinc2018")
    built = set(spec.built_subjects())
    wanted = (spec.raw_dir / "selected_records_holdout.txt").read_text().split()
    held = [s for s in wanted if s in built and s not in dev]
    run.log(f"{len(dev)} development subjects, {len(held)} held-out subjects built of {len(wanted)} selected")
    cache = REPO / "artifacts/cinc2018_window_cache"
    _, rows_d, tracks_d = run.load_inputs("cinc2018", cache, dev)
    _, rows_h, tracks_h = run.load_inputs("cinc2018", cache, held)
    rows, tracks = {**rows_d, **rows_h}, {**tracks_d, **tracks_h}
    train = run.epoch_frame(rows_d, run.WINDOW_FEATURES)
    test = run.epoch_frame(rows_h, run.WINDOW_FEATURES)
    y_tr, y_te = train[run.TARGET].to_numpy(), test[run.TARGET].to_numpy()
    short = lambda c: c.split(".")[-1]  # noqa: E731
    all_configs = {
        "window33": (run.TRACKS, run.WINDOW_FEATURES),
        "nocardio27": run.nocardio27_signals(),
        "final21": run.final21_signals(),
        "g20": run.adopted_signals(),
    }
    wanted_cfgs = [c for c in CONFIGS if c in all_configs]
    configs = {c: all_configs[c] for c in wanted_cfgs}
    preds = test[[run.SUBJ, "epoch_index", run.TARGET]].copy()
    summary = []
    if "epoch19" in CONFIGS:  # single-epoch reference
        model, p = run.fit_proba(train[run.WINDOW_FEATURES].to_numpy(np.float32), y_tr,
                                 test[run.WINDOW_FEATURES].to_numpy(np.float32), [short(c) for c in run.WINDOW_FEATURES])
        preds["epoch19"] = p.argmax(1)
    for cfg, (trk, feats) in configs.items():
        names = [short(c) for c in trk + feats]
        dense = {s: run.dense_signals(tracks[s], rows[s], trk, feats) for s in dev + held}
        enc = run.WindowEncoder(names).fit(dense, dev)
        Wtr, Wte = enc.transform(dense, train), enc.transform(dense, test)
        model, p = run.fit_proba(Wtr.reshape(len(Wtr), -1), y_tr, Wte.reshape(len(Wte), -1), enc.column_names())
        preds[cfg] = p.argmax(1)
        if cfg == "window33":
            preds["window33_decoder"] = run.hmm_decode(test, p, run.transition_model(train)).argmax(1)
            for j, st in enumerate(STAGES):
                preds[f"p_{st}"] = p[:, j].astype(np.float32)
        del Wtr, Wte, dense
        run.log(f"{cfg}: held-out macro-F1 {f1_score(y_te, p.argmax(1), average='macro'):.4f}")
    for col in [c for c in [*CONFIGS, "window33_decoder"] if c in preds.columns]:
        yp = preds[col].to_numpy()
        row = {"config": col, "macro_f1": float(f1_score(y_te, yp, average="macro")), "kappa": float(cohen_kappa_score(y_te, yp)),
               "accuracy": float(accuracy_score(y_te, yp))}
        summary.append(row)
        run.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    ledger = out / "LEDGER.csv"
    # A rerun that only repairs a failed build keeps its look number (edit the row by hand); new runs take the next one.
    looks = (int(pd.read_csv(ledger)["look"].max()) if ledger.exists() else 0) + 1
    entry = pd.DataFrame([{"look": looks, "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "subjects": int(test[run.SUBJ].nunique()),
                           "configs": "; ".join(r["config"] for r in summary),
                           "adopted_macro_f1": next(r["macro_f1"] for r in summary if r["config"] == run.ADOPTED), "note": ""}])
    entry.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    run.log(f"held-out look #{looks} recorded in {ledger}")
    run.log(f"done -> {out} ({test[run.SUBJ].nunique()} subjects, {len(test)} epochs)")


if __name__ == "__main__":
    main()
