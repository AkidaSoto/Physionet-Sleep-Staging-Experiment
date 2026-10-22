"""Build the ISRUC-Sleep subgroup I staging feature cache (CinC 2018 recipe), pruning intermediate signals per subject.

Run several copies with --worker i --workers n to split subjects 1-100. Builds whichever subjects have finished
downloading, so it can run alongside scripts/download_isruc.py. Also writes the 1-s tracks the window model reads
(artifacts/isruc_window_cache/tracks1s/<id>.parquet).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.algorithms.eeg_tracks import isruc_subject_tracks  # noqa: E402
from physionet_sleep.studies.cinc2018 import prune_subject_cache  # noqa: E402
from physionet_sleep.studies.isruc import ISRUCStagingStudyDef  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=[str(i) for i in range(1, 101)])
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-in-progress", action="store_true",
                        help="leave subjects whose cache folder exists without a feature table (another builder has them)")
    parser.add_argument("--wait-minutes", type=float, default=600.0, help="give up on a subject whose .rec never arrives")
    args = parser.parse_args()

    raw = REPO_ROOT / "data/raw/isruc"
    cache = REPO_ROOT / "artifacts/isruc_recipe_cache"
    tracks_dir = REPO_ROOT / "artifacts/isruc_window_cache/tracks1s"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    pending = list(args.subjects[args.worker :: args.workers])
    failed = []
    deadline = time.time() + args.wait_minutes * 60
    while pending and time.time() < deadline:
        if args.skip_in_progress:
            taken = [s for s in pending if (cache / s).exists() and not (cache / s / "final__feature_table_py").exists()]
            for s in taken:
                pending.remove(s)
                print(f"[w{args.worker}] {s} is being built elsewhere; skipped", flush=True)
        ready = [s for s in pending if (cache / s / "final__feature_table_py").exists()
                 or all(p.exists() for p in (raw / s / f"{s}.rec", raw / s / f"{s}_1.txt", raw / s / f"{s}_2.txt"))]
        if not ready:
            time.sleep(30)
            continue
        s = ready[0]
        pending.remove(s)
        t0 = time.time()
        if not (cache / s / "final__feature_table_py").exists():
            try:
                out = ISRUCStagingStudyDef(subjects=[s], root_dir=raw, cache_dir=cache).run_build({})
                ok = bool(out.get("subjects"))
            except Exception:
                traceback.print_exc()
                ok = False
            if not ok:
                failed.append(s)
                print(f"[w{args.worker} {len(pending)} left] {s} FAILED {time.time() - t0:.0f}s", flush=True)
                continue
        prune_subject_cache(cache / s)
        tp = tracks_dir / f"{s}.parquet"
        if not tp.exists():
            isruc_subject_tracks(raw, cache, s).to_parquet(tp, index=False)
        print(f"[w{args.worker} {len(pending)} left] {s} ok {time.time() - t0:.0f}s", flush=True)
    failed += pending
    print(f"[w{args.worker}] done; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
