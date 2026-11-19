"""Data for the dashboard's Figure 1: one typical held-out night, and every patient's score in every test set.

Per-patient macro-F1 uses one definition for all cohorts: the mean F1 over the stages the expert scored that night.
(Averaging over all five stages scores a stage that never occurred as 0 and so penalises nights without N3; 31 of
the 200 held-out nights have no epochs of at least one stage.)

The example night is the held-out patient whose score is closest to the held-out median (ties: lowest record id);
it is chosen by rule, not by eye. Reads saved predictions only; no refitting.

Writes showcase/content/generated/lead-figure.json.
Usage: python scripts/export_lead_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "artifacts" / "staging_window_model" / "cinc2018"
CV = WINDOW / "predictions.parquet"
HOLDOUT = WINDOW / "holdout" / "predictions_recovered_g20.parquet"
ISRUC = ROOT / "artifacts" / "staging_window_model" / "transfer_isruc" / "predictions.parquet"
OUTPUT = ROOT / "showcase" / "content" / "generated" / "lead-figure.json"
ADOPTED = "g20__epoch"


def night_scores(y: np.ndarray, pred: np.ndarray, subject: np.ndarray) -> dict[str, float]:
    """Macro-F1 per night over the stages present in the expert's labels for that night."""
    out = {}
    for s in sorted(np.unique(subject)):
        m = subject == s
        out[str(s)] = float(f1_score(y[m], pred[m], labels=sorted(set(y[m].tolist())), average="macro", zero_division=0))
    return out


def main() -> int:
    cv = pd.read_parquet(CV, columns=["subject_id", "epoch_index", "label.stage_30s", ADOPTED])
    ho = pd.read_parquet(HOLDOUT)
    co = pd.read_parquet(ISRUC, columns=["subject_id", "epoch_index", "scorer1", "scorer2", "pred"])
    if set(cv["subject_id"].astype(str)) & set(ho["subject_id"].astype(str)):
        raise SystemExit("held-out patients overlap the cross-validation patients")  # leakage guard
    if (co[["scorer1", "pred"]] < 0).any().any():
        raise SystemExit("unscored Coimbra epochs in the scorer-1 comparison")
    both = co["scorer2"].to_numpy() >= 0

    patients = {
        "cv": night_scores(cv["label.stage_30s"].to_numpy(), cv[ADOPTED].to_numpy(), cv["subject_id"].astype(str).to_numpy()),
        "holdout": night_scores(ho["label.stage_30s"].to_numpy(), ho["prediction"].to_numpy(), ho["subject_id"].astype(str).to_numpy()),
        "coimbra_scorer1": night_scores(co["scorer1"].to_numpy(), co["pred"].to_numpy(), co["subject_id"].astype(str).to_numpy()),
        "coimbra_scorer2": night_scores(co["scorer2"].to_numpy()[both], co["pred"].to_numpy()[both], co["subject_id"].astype(str).to_numpy()[both]),
        "coimbra_scorers": night_scores(co["scorer1"].to_numpy()[both], co["scorer2"].to_numpy()[both], co["subject_id"].astype(str).to_numpy()[both]),
    }

    held = patients["holdout"]
    median = float(np.median(list(held.values())))
    subject = min(held, key=lambda s: (abs(held[s] - median), s))
    night = ho[ho["subject_id"].astype(str) == subject].sort_values("epoch_index")
    expected = np.arange(night["epoch_index"].min(), night["epoch_index"].max() + 1)
    if not np.array_equal(night["epoch_index"].to_numpy(), expected):
        raise SystemExit(f"night {subject} has gaps in its scored epochs; the hypnogram would misplace them")
    y, p = night["label.stage_30s"].to_numpy(), night["prediction"].to_numpy()

    summary = {}
    for cohort, scores in patients.items():
        v = np.array(list(scores.values()))
        summary[cohort] = {"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                           "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}
    payload = {
        "definition": "Per-night macro-F1 over the stages the expert scored that night.",
        "stages": ["Wake", "N1", "N2", "N3", "REM"],
        "night": {"cohort": "holdout", "record": subject, "first_epoch": int(night["epoch_index"].iloc[0]),
                  "macro_f1": held[subject], "accuracy": float(accuracy_score(y, p)), "epochs": int(len(y)),
                  "disagreements": int((y != p).sum()), "expert": y.astype(int).tolist(), "model": p.astype(int).tolist()},
        "summary": summary,
        "patients": {k: [round(v, 4) for _, v in sorted(s.items())] for k, s in patients.items()},
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}; example night {subject}: macro-F1 {held[subject]:.3f} "
          f"(held-out median {median:.3f}), {len(y)} epochs, {(y != p).sum()} disagreements")
    for k, s in summary.items():
        print(f"  {k:16s} n={s['n']:3d} median {s['median']:.3f} min {s['min']:.3f} max {s['max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
