"""Copy ISRUC-Sleep subgroup I nights that the MEGA desktop app has finished into data/raw/isruc/ (copy, never move).

Every few minutes, for each subject still missing `data/raw/isruc/<n>/<n>.rec`: if the app's folder `<n>/` has no
`.getxfer*` temp file and every file has the size the MEGA listing gives, its files are copied in and any leftover
`.part` is deleted. A subject whose `.part` changed in the last two minutes (scripts/download_isruc.py writing it) is
left for a later round; the downloader itself stops once the copied file appears.

Usage: python scripts/sync_isruc_mega_app.py [--app-dir DIR] [--every 180]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_isruc as dl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ACTIVE_SECONDS = 120


def sync_once(app_dir: Path, out: Path, by_subject: dict[str, list[tuple[str, int]]]) -> list[str]:
    copied = []
    for s, files in sorted(by_subject.items(), key=lambda kv: int(kv[0])):
        if (out / s / f"{s}.rec").exists():
            continue
        if any((app_dir / s).glob("*.getxfer*")):
            continue
        if not all((app_dir / rel).exists() and (app_dir / rel).stat().st_size == size for rel, size in files):
            continue
        parts = list((out / s).glob("*.part")) if (out / s).exists() else []
        if any(time.time() - p.stat().st_mtime < ACTIVE_SECONDS for p in parts):
            continue
        # the .rec goes last: the builder treats a subject as ready once its .rec and hypnograms exist
        for rel, size in sorted(files, key=lambda f: f[0].endswith(".rec")):
            dst = out / rel
            if not (dst.exists() and dst.stat().st_size == size) and not dl.copy_from_app(dst, size, rel, app_dir):
                break
        else:
            copied.append(s)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", default=str(dl.APP_DIR))
    ap.add_argument("--out", default=str(REPO / "data/raw/isruc"))
    ap.add_argument("--every", type=int, default=180, help="seconds between rounds")
    args = ap.parse_args()
    app_dir, out = Path(args.app_dir), Path(args.out)
    by_subject: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for rel, _, size, _, _ in dl.listing():
        s = rel.split("/")[0]
        if s.isdigit() and "/" in rel:
            by_subject[s].append((rel, size))
    while True:
        copied = sync_once(app_dir, out, by_subject)
        if copied:
            print(f"[{time.strftime('%H:%M')}] copied from app: {' '.join(copied)}", flush=True)
        missing = [s for s in by_subject if not (out / s / f"{s}.rec").exists()]
        if not missing:
            print("all subjects present", flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
