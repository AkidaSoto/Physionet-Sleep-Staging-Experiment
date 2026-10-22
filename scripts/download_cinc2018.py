"""Resumable download of PhysioNet/CinC Challenge 2018 training records (open access, ODC-By 1.0).

Examples:
  # annotations + headers for every training record (~6 MB) plus age-sex.csv
  python scripts/download_cinc2018.py --kinds hea arousal
  # full signals for selected records
  python scripts/download_cinc2018.py --records-file data/raw/cinc2018/selected_records.txt --kinds hea arousal mat
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
import urllib.request
from pathlib import Path

# PhysioNet's AWS Open Data mirror of the same files; the physionet.org server is ~60x slower from here.
BASE = "https://physionet-open.s3.amazonaws.com/challenge-2018/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
SUFFIX = {"hea": ".hea", "arousal": ".arousal", "mat": ".mat", "arousal_mat": "-arousal.mat"}


def _fetch(url: str, dest: Path, retries: int = 3) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
                expected = int(response.headers.get("Content-Length", -1))
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            written = tmp.stat().st_size
            if expected >= 0 and written != expected:
                raise IOError(f"truncated: {written} of {expected} bytes")
            tmp.replace(dest)
            return written
        except Exception as exc:  # network hiccup: retry, then report
            if attempt == retries:
                print(f"FAILED {url}: {exc}", flush=True)
                return -1
            time.sleep(5 * attempt)
    return -1


def _training_records() -> list[str]:
    with urllib.request.urlopen("https://physionet.org/files/challenge-2018/1.0.0/training/RECORDS", timeout=60) as response:
        return [line.strip().rstrip("/") for line in response.read().decode().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "raw" / "cinc2018"))
    parser.add_argument("--records-file", help="one record id per line; default = all training records")
    parser.add_argument("--kinds", nargs="+", default=["hea", "arousal"], choices=sorted(SUFFIX))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root)
    _fetch(f"{BASE}/age-sex.csv", root / "age-sex.csv")
    records = (
        [line.strip() for line in Path(args.records_file).read_text().splitlines() if line.strip()]
        if args.records_file
        else _training_records()
    )
    jobs = [
        (f"{BASE}/training/{record}/{record}{SUFFIX[kind]}", root / record / f"{record}{SUFFIX[kind]}")
        for record in records
        for kind in args.kinds
    ]
    total, failed = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, size in enumerate(pool.map(lambda job: _fetch(*job), jobs), 1):
            failed += size < 0
            total += max(size, 0)
            if index % 50 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)} files] downloaded {total / 1e9:.2f} GB this run, failures {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
