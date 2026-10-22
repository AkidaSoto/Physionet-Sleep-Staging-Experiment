"""Quality-based exclusion and fallback, scored from saved per-epoch probabilities (run_quality_staging.py).

Epoch quality (signals only, never labels or model errors): for every scored 30-s epoch and each channel, the share of
its seconds that break eeg_quality.QUALITY_RULES (mains, flat, ECG correlation, peak-to-peak). An epoch is UNUSABLE at a
level when a signal the model relies on is bad (LEVELS below). For each run and level this reports:
  all        every labelled epoch (the run's plain score)
  retained   unusable epochs dropped from scoring
  fallback   all epochs, an unusable epoch's probabilities replaced by the mean of the nearest usable epoch before and
             after it in the same night (one side when the other does not exist)
  fallback_gap5  (post hoc) the same, only for runs of <= 5 unusable epochs; longer runs keep their own prediction
Also excluded_accuracy.csv: the model's accuracy on excluded vs retained epochs (and stage-matched: excluded per-stage
accuracy weighted by the retained stage mix), to test whether "unusable" epochs are actually predicted worse.
with macro-F1 (fold mean and folds), kappa (fold mean), per-stage F1 (pooled), per-subject macro-F1 min / median
(stages present in the subject; subjects with >= 30 scored epochs) and % of epochs excluded. Exclusion statistics
(overall, per true stage, per subject) go to exclusion_<level>.csv and exclusion_summary.csv.

Usage: python scripts/run_quality_report.py [run tag ...]   (default: every runs/*.parquet)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.algorithms.eeg_quality import SHORT, bad_seconds  # noqa: E402

OUT = REPO / "artifacts/exploration/signal-quality"
QCACHE = REPO / "artifacts/cinc2018_window_cache/quality1s"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
SUBJ = "subject_id"
CHANNELS = list(SHORT.values())  # F3 F4 C3 C4 O1 O2 EOG EMG
# level -> (pairs that count as bad only when both leads are bad, single channels, min bad share of the epoch)
LEVELS = {
    "L1": (["C"], ["EOG"], 0.5),
    "L2": (["C", "F", "O"], ["EOG", "EMG"], 0.5),
    "L3": (["C", "F", "O"], ["EOG", "EMG"], 0.2),
}


def epoch_quality(subjects: list[str]) -> pd.DataFrame:
    """Per subject x epoch: bad-second share of each channel ('<ch>_bad') and of all-channels-flat ('all_flat')."""
    path = OUT / "epoch_quality.parquet"
    if path.exists():
        eq = pd.read_parquet(path)
        if set(subjects) <= set(eq[SUBJ]):
            return eq[eq[SUBJ].isin(subjects)]
    frames = []
    for s in subjects:
        q = pd.read_parquet(QCACHE / f"{s}.parquet")
        ne = len(q) // 30
        d = {SUBJ: s, "epoch_index": np.arange(ne)}
        for c in CHANNELS:
            d[f"{c}_bad"] = bad_seconds(q, c)[: ne * 30].reshape(ne, 30).mean(1)
        flat = np.all([q[f"{c}_flat"].to_numpy() > 0 for c in CHANNELS], axis=0)
        d["all_flat"] = flat[: ne * 30].reshape(ne, 30).mean(1)
        frames.append(pd.DataFrame(d))
    eq = pd.concat(frames, ignore_index=True)
    eq.to_parquet(path, index=False)
    return eq


def unusable_epochs(eq: pd.DataFrame, level: str) -> pd.DataFrame:
    pairs, singles, share = LEVELS[level]
    bad = eq["all_flat"].to_numpy() >= share
    for p in pairs:
        a, b = {"C": ("C3", "C4"), "F": ("F3", "F4"), "O": ("O1", "O2")}[p]
        bad |= (eq[f"{a}_bad"].to_numpy() >= share) & (eq[f"{b}_bad"].to_numpy() >= share)
    for c in singles:
        bad |= eq[f"{c}_bad"].to_numpy() >= share
    return pd.DataFrame({SUBJ: eq[SUBJ].to_numpy(), "epoch_index": eq["epoch_index"].to_numpy(), "unusable": bad})


def neighbour_fallback(df: pd.DataFrame, P: np.ndarray, bad: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Replace each unusable epoch's probabilities with the mean of the nearest usable epochs before and after.
    max_gap: only runs of at most this many unusable epochs are filled; longer runs keep their own predictions."""
    P = P.copy()
    if max_gap is not None:
        bad = bad & (run_lengths(df, bad) <= max_gap)
    for _, g in df.groupby(SUBJ, sort=False):
        idx = g.sort_values("epoch_index").index.to_numpy()
        b = bad[idx]
        if not b.any() or b.all():
            continue
        good = np.flatnonzero(~b)
        for i in np.flatnonzero(b):
            k = np.searchsorted(good, i)
            nb = [good[j] for j in (k - 1, k) if 0 <= j < len(good)]
            P[idx[i]] = P[idx[nb]].mean(0)
    return P


def run_lengths(df: pd.DataFrame, bad: np.ndarray) -> np.ndarray:
    """Per epoch, the length of the run of consecutive unusable epochs it belongs to (0 for usable epochs)."""
    out = np.zeros(len(df), int)
    for _, g in df.groupby(SUBJ, sort=False):
        idx = g.sort_values("epoch_index").index.to_numpy()
        b = bad[idx].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, b, 0]))
        for a, e in zip(edges[::2], edges[1::2]):
            out[idx[a:e]] = e - a
    return out


def score(df: pd.DataFrame, pred: np.ndarray, keep: np.ndarray) -> dict:
    y, f, subj = df["label"].to_numpy(), df["fold"].to_numpy(), df[SUBJ].to_numpy()
    fold_f1 = [f1_score(y[keep & (f == k)], pred[keep & (f == k)], average="macro") for k in range(5)]
    row = {"macro_f1": float(np.mean(fold_f1)), "folds": " ".join(f"{v:.3f}" for v in fold_f1),
           "kappa": float(np.mean([cohen_kappa_score(y[keep & (f == k)], pred[keep & (f == k)]) for k in range(5)]))}
    for st, v in zip(STAGES, f1_score(y[keep], pred[keep], labels=range(5), average=None, zero_division=0)):
        row[f"f1_{st}"] = float(v)
    per = {}
    for s in np.unique(subj):
        m = keep & (subj == s)
        if m.sum() >= 30:
            per[s] = f1_score(y[m], pred[m], labels=sorted(set(y[m])), average="macro")
    per = pd.Series(per)
    row.update(subj_min=float(per.min()), subj_median=float(per.median()), worst_subject=str(per.idxmin()),
               subjects_scored=len(per), pct_excluded=float(100 * (1 - keep.mean())))
    return row


def exclusion_stats(df: pd.DataFrame, bad: np.ndarray, level: str) -> dict:
    y = df["label"].to_numpy()
    per_subj = pd.Series(bad).groupby(df[SUBJ].to_numpy()).mean().sort_values(ascending=False)
    per_subj.rename("excluded_share").to_csv(OUT / f"exclusion_{level}.csv", index_label=SUBJ)
    row = {"level": level, "pct_epochs": 100 * bad.mean(), "pct_in_runs_le5": 100 * (bad & (run_lengths(df, bad) <= 5)).mean()}
    for k, st in enumerate(STAGES):
        row[f"pct_{st}"] = 100 * bad[y == k].mean()
    row.update(subjects_any=int((per_subj > 0).sum()), subjects_gt5pct=int((per_subj > 0.05).sum()),
               subjects_gt20pct=int((per_subj > 0.2).sum()), subjects_gt50pct=int((per_subj > 0.5).sum()),
               subj_median_pct=100 * per_subj.median(), subj_p90_pct=100 * per_subj.quantile(0.9),
               worst=" ".join(f"{s}:{100 * v:.0f}%" for s, v in per_subj.head(6).items()))
    return row


def main() -> None:
    tags = sys.argv[1:] or sorted(p.stem for p in (OUT / "runs").glob("*.parquet"))
    rows, excl, err = [], {}, []
    for tag in tags:
        df = pd.read_parquet(OUT / "runs" / f"{tag}.parquet")
        eq = epoch_quality(sorted(df[SUBJ].unique()))
        P = df[[f"p_{s}" for s in STAGES]].to_numpy()
        allm = np.ones(len(df), bool)
        rows.append({"run": tag, "level": "-", "scoring": "all", **score(df, P.argmax(1), allm)})
        for level in LEVELS:
            u = df[[SUBJ, "epoch_index"]].merge(unusable_epochs(eq, level), on=[SUBJ, "epoch_index"], how="left")
            bad = u["unusable"].fillna(False).to_numpy(bool)
            if level not in excl:
                excl[level] = exclusion_stats(df, bad, level)
            rows.append({"run": tag, "level": level, "scoring": "retained", **score(df, P.argmax(1), ~bad)})
            rows.append({"run": tag, "level": level, "scoring": "fallback",
                         **score(df, neighbour_fallback(df, P, bad).argmax(1), allm)})
            # post hoc (added after the base results): fill only gaps of <= 5 epochs (2.5 min)
            rows.append({"run": tag, "level": level, "scoring": "fallback_gap5",
                         **score(df, neighbour_fallback(df, P, bad, max_gap=5).argmax(1), allm)})
            y, pred = df["label"].to_numpy(), P.argmax(1)
            err.append({"run": tag, "level": level, "acc_excluded": float((y == pred)[bad].mean()),
                         "acc_retained": float((y == pred)[~bad].mean()),
                         "acc_excluded_stage_matched": float(np.nansum([(y == pred)[bad & (y == k)].mean() * (y[~bad] == k).mean()
                                                                        for k in range(5) if (bad & (y == k)).any()])),
                         "n_excluded": int(bad.sum())})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "report.csv", index=False)
    pd.DataFrame(excl.values()).to_csv(OUT / "exclusion_summary.csv", index=False)
    pd.DataFrame(err).to_csv(OUT / "excluded_accuracy.csv", index=False)
    print(pd.DataFrame(err).round(4).to_string(index=False))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(excl.values()).round(2).to_string(index=False))
    print(res.drop(columns=["folds"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
