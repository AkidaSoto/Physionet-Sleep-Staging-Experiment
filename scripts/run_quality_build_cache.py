"""Per-second quality inputs for the 320 CinC 2018 development subjects (eeg_quality.per_second_inputs).

Writes artifacts/cinc2018_window_cache/quality1s/<id>.parquet: per-lead band powers, single-channel tracks and per-channel
quality metrics, one row per second. The existing tracks1s cache is not touched. Only dev_subjects.txt is read.

Usage: python scripts/run_quality_build_cache.py [--workers 3] [subject ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "artifacts/cinc2018_window_cache/quality1s"


def build(subject: str) -> str:
    from physionet_sleep.algorithms.eeg_quality import cinc2018_per_second_inputs
    from physionet_sleep.analysis.datasets import DATASETS

    path = OUT / f"{subject}.parquet"
    if not path.exists():
        spec = DATASETS["cinc2018"]
        df = cinc2018_per_second_inputs(spec.raw_dir, spec.cache_dir, subject)
        df.to_parquet(path.with_suffix(".tmp"), index=False)
        os.replace(path.with_suffix(".tmp"), path)
    return subject


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("subjects", nargs="*")
    args = ap.parse_args()
    dev = (REPO / "artifacts/staging_window_model/cinc2018/dev_subjects.txt").read_text().split()
    subjects = args.subjects or dev
    assert set(subjects) <= set(dev), "development subjects only"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for i, s in enumerate(ex.map(build, subjects)):
            print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(subjects)} {s}", flush=True)


if __name__ == "__main__":
    main()
