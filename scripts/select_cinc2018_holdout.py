"""Pick a held-out CinC 2018 test set: records never used in development, apnea-stratified like the 320 where the pool allows.

Reads every downloaded header/annotation, computes AHI from the annotations, removes the development records
(those with a built recipe cache), and samples the same AHI-band proportions as the development set (seed 7).
Writes data/raw/cinc2018/selected_records_holdout.txt and artifacts/staging_window_model/cinc2018/holdout_subjects.csv.

Usage: python scripts/select_cinc2018_holdout.py [--n 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.analysis.datasets import DATASETS  # noqa: E402
from physionet_sleep.io.cinc2018 import discover_cinc2018_records  # noqa: E402
from physionet_sleep.labels.cinc2018 import subject_summary  # noqa: E402

BANDS = [(-np.inf, 5, "<5"), (5, 15, "5-15"), (15, 30, "15-30"), (30, np.inf, ">=30")]


def band(ahi: float) -> str:
    return next(name for lo, hi, name in BANDS if lo <= ahi < hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    spec = DATASETS["cinc2018"]
    dev_file = REPO / "artifacts/staging_window_model/cinc2018/dev_subjects.txt"
    dev = set(dev_file.read_text().split()) if dev_file.exists() else set(spec.built_subjects())
    rows = []
    for paths in discover_cinc2018_records(spec.raw_dir, require_signals=False):
        if paths.annotation_path is None:
            continue
        info = subject_summary(paths)
        rows.append({"record_id": paths.record_id, "dev": paths.record_id in dev, **info})
    table = pd.DataFrame(rows)
    table["ahi_band"] = table["ahi"].map(band)
    share = table[table["dev"]]["ahi_band"].value_counts(normalize=True)
    pool = table[~table["dev"]]
    rng = np.random.default_rng(7)
    picks = []
    for name, frac in share.items():
        candidates = pool[pool["ahi_band"] == name]["record_id"].to_numpy()
        k = min(len(candidates), int(round(frac * args.n)))
        picks += list(rng.choice(candidates, size=k, replace=False))
    if len(picks) < args.n:  # a band ran out (most AHI < 5 records are already in development): top up at random
        rest = pool[~pool["record_id"].isin(picks)]["record_id"].to_numpy()
        picks += list(rng.choice(rest, size=min(len(rest), args.n - len(picks)), replace=False))
    picks = sorted(picks)
    (spec.raw_dir / "selected_records_holdout.txt").write_text("\n".join(picks) + "\n", encoding="utf-8")
    out = REPO / "artifacts/staging_window_model/cinc2018"
    out.mkdir(parents=True, exist_ok=True)
    table[table["record_id"].isin(picks)].to_csv(out / "holdout_subjects.csv", index=False)
    print(f"{len(table)} labelled records, {len(dev)} development, {len(pool)} available; picked {len(picks)}")
    print(table[table["record_id"].isin(picks)]["ahi_band"].value_counts().to_dict())


if __name__ == "__main__":
    main()
