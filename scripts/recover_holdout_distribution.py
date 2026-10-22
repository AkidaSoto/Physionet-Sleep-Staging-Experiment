"""Reproduce the fixed g20 evaluation to recover missing patient scores; do not select or tune models."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_window_hmm_staging as run


def main() -> None:
    out = ROOT / "artifacts/staging_window_model/cinc2018/holdout"
    before = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in ("summary.csv", "LEDGER.csv")}
    expected = pd.read_csv(out / "summary.csv").set_index("config").loc["g20"]
    dev = run.development_subjects("cinc2018")
    held = (run.DATASETS["cinc2018"].raw_dir / "selected_records_holdout.txt").read_text().split()
    assert len(dev) == 320 and len(held) == 200 and not set(dev) & set(held)
    cache = ROOT / "artifacts/cinc2018_window_cache"
    _, rows_d, tracks_d = run.load_inputs("cinc2018", cache, dev)
    _, rows_h, tracks_h = run.load_inputs("cinc2018", cache, held)
    train = run.epoch_frame(rows_d, run.WINDOW_FEATURES)
    test = run.epoch_frame(rows_h, run.WINDOW_FEATURES)
    trk, feats = run.adopted_signals()
    names = [c.split(".")[-1] for c in trk + feats]
    rows, tracks = {**rows_d, **rows_h}, {**tracks_d, **tracks_h}
    dense = {s: run.dense_signals(tracks[s], rows[s], trk, feats) for s in dev + held}
    enc = run.WindowEncoder(names).fit(dense, dev)
    Wtr, Wte = enc.transform(dense, train), enc.transform(dense, test)
    run.log(f"Recovery: fixed g20, {len(dev)} training patients, {len(held)} held-out patients")
    _, probabilities = run.fit_proba(Wtr.reshape(len(Wtr), -1), train[run.TARGET].to_numpy(),
                                    Wte.reshape(len(Wte), -1), enc.column_names())
    pred = probabilities.argmax(1)
    y = test[run.TARGET].to_numpy()
    metrics = {"macro_f1": float(f1_score(y, pred, average="macro")),
               "accuracy": float(accuracy_score(y, pred)), "kappa": float(cohen_kappa_score(y, pred))}
    matches = all(np.isclose(metrics[k], float(expected[k]), rtol=0, atol=1e-12) for k in metrics)
    audit = {"time": datetime.now(ZoneInfo("America/New_York")).isoformat(),
             "purpose": "Recover patient distribution from unchanged evaluation; no tuning or model selection",
             "evaluation_look": 2, "config": "g20", "seed": run.SEED,
             "expected": {k: float(expected[k]) for k in metrics}, "recovered": metrics,
             "matches_original": matches, "original_hashes": before}
    (out / "distribution_recovery.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not matches:
        raise SystemExit(f"Scores differ; distribution NOT published: {audit}")
    frame = test[[run.SUBJ, "epoch_index", run.TARGET]].copy()
    frame["prediction"] = pred
    frame.to_parquet(out / "predictions_recovered_g20.parquet", index=False)
    scores = []
    for subject, group in frame.groupby(run.SUBJ):
        scores.append({"subject": str(subject), "macro_f1": float(f1_score(
            group[run.TARGET], group["prediction"], labels=range(5), average="macro", zero_division=0))})
    pd.DataFrame(scores).to_csv(out / "subject_scores_recovered_g20.csv", index=False)
    values = [r["macro_f1"] for r in scores]
    payload = {"verified": True, "metrics": metrics, "subject_macro_f1": values}
    (ROOT / "showcase/content/generated/holdout-distribution.json").write_text(json.dumps(payload, indent=2) + "\n")
    assert all(hashlib.sha256((out / name).read_bytes()).hexdigest() == digest for name, digest in before.items())
    run.log(f"Verified original scores: {metrics}; patient min/median/max: {np.min(values):.4f}/{np.median(values):.4f}/{np.max(values):.4f}")


if __name__ == "__main__":
    main()
