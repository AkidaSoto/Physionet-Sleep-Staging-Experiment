from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


TARGET_ALGOS = [
    "spindle_features",
    "slow_wave_features",
    "eeg_arousal_features",
    "feature_table",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.io.ucddb import discover_ucddb_records
    from physionet_sleep.studies.ucddb import UCDDBApneaStudyDef

    root_dir = repo_root / "data" / "raw" / "ucddb"
    cache_dir = repo_root / "artifacts" / "ucddb_recipe_cache"
    subjects = [record.record_id for record in discover_ucddb_records(root_dir)]

    print(
        f"rerun_start subjects={len(subjects)} target_algos={','.join(TARGET_ALGOS)}",
        flush=True,
    )
    t_all = time.time()
    completed = 0
    failed: list[str] = []

    for idx, subject_id in enumerate(subjects, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(subjects)}] start {subject_id}", flush=True)
        try:
            study = UCDDBApneaStudyDef(
                subjects=[subject_id],
                root_dir=root_dir,
                cache_dir=cache_dir,
            )
            study.force_rerun = list(TARGET_ALGOS)
            out = study.run_build({})
            if out.get("subjects"):
                completed += 1
                print(
                    f"[{idx}/{len(subjects)}] done {subject_id} elapsed={round(time.time() - t0, 2)}s",
                    flush=True,
                )
            else:
                failed.append(subject_id)
                print(
                    f"[{idx}/{len(subjects)}] empty {subject_id} elapsed={round(time.time() - t0, 2)}s",
                    flush=True,
                )
        except Exception:
            failed.append(subject_id)
            print(f"[{idx}/{len(subjects)}] fail {subject_id}", flush=True)
            traceback.print_exc()

    print(
        f"rerun_done completed={completed} failed={len(failed)} elapsed={round(time.time() - t_all, 2)}s",
        flush=True,
    )
    if failed:
        print("failed_subjects=" + ",".join(failed), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
