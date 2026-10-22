"""Compute 1-s window-model tracks for CinC records as their recipe caches appear (runs alongside the cache build).

Usage: python scripts/precompute_cinc2018_tracks.py --records-file data/raw/cinc2018/selected_records_holdout.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.algorithms.eeg_tracks import cinc2018_subject_tracks  # noqa: E402
from physionet_sleep.analysis.datasets import DATASETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-file", required=True)
    ap.add_argument("--wait-minutes", type=float, default=300)
    args = ap.parse_args()
    spec = DATASETS["cinc2018"]
    out = REPO / "artifacts/cinc2018_window_cache/tracks1s"
    out.mkdir(parents=True, exist_ok=True)
    todo = Path(args.records_file).read_text().split()
    deadline = time.time() + args.wait_minutes * 60
    while todo and time.time() < deadline:
        ready = [s for s in todo if (spec.cache_dir / s / "final__feature_table_py").exists()]
        for s in ready:
            if not (out / f"{s}.parquet").exists():
                cinc2018_subject_tracks(spec.raw_dir, spec.cache_dir, s).to_parquet(out / f"{s}.parquet", index=False)
                print(f"tracks {s}", flush=True)
            todo.remove(s)
        if todo:
            time.sleep(30)
    print(f"done; {len(todo)} records never became ready", flush=True)


if __name__ == "__main__":
    main()
