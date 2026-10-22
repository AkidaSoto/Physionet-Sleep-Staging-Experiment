"""Build the CinC 2018 staging feature cache for downloaded records, pruning intermediate signals per subject.

Run several copies with --worker i --workers n to split the record list. Waits for each record's signal file to
finish downloading, so it can run alongside scripts/download_cinc2018.py.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.studies.cinc2018 import CinC2018StagingStudyDef, prune_subject_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-file", default=str(REPO_ROOT / "data/raw/cinc2018/selected_records.txt"))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--wait-minutes", type=float, default=60.0, help="give up on a record whose signals never arrive")
    args = parser.parse_args()

    raw = REPO_ROOT / "data/raw/cinc2018"
    cache = REPO_ROOT / "artifacts/cinc2018_recipe_cache"
    records = [r.strip() for r in Path(args.records_file).read_text().splitlines() if r.strip()]
    mine = records[args.worker :: args.workers]
    failed = []
    for index, record in enumerate(mine, 1):
        if (cache / record / "final__feature_table_py").exists():
            prune_subject_cache(cache / record)
            continue
        signal = raw / record / f"{record}.mat"
        deadline = time.time() + args.wait_minutes * 60
        while not signal.exists() and time.time() < deadline:
            time.sleep(15)
        if not signal.exists():
            failed.append(record)
            print(f"[w{args.worker} {index}/{len(mine)}] {record} signals never arrived", flush=True)
            continue
        t0 = time.time()
        try:
            out = CinC2018StagingStudyDef(subjects=[record], root_dir=raw, cache_dir=cache).run_build({})
            ok = bool(out.get("subjects"))
        except Exception:
            traceback.print_exc()
            ok = False
        if ok:
            prune_subject_cache(cache / record)
        else:
            failed.append(record)
        print(f"[w{args.worker} {index}/{len(mine)}] {record} {'ok' if ok else 'FAILED'} {time.time() - t0:.0f}s", flush=True)
    print(f"[w{args.worker}] done; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
