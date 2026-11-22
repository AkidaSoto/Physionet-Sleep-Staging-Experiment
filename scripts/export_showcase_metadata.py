"""Export algorithm band edges and track metadata for the showcase dashboard.

Reads canonical constants from ``src/physionet_sleep/algorithms/`` so frequency
band labels in the dashboard cannot drift from the Python feature extractors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "showcase" / "content" / "generated" / "algorithm-metadata.json"

sys.path.insert(0, str(ROOT / "src"))
from physionet_sleep.algorithms.eeg_tracks import BANDS as TRACK_BANDS, FS, TRACKS  # noqa: E402
from physionet_sleep.algorithms.staging import EpochRatioFeatureGenerator  # noqa: E402


def _fmt_hz(lo: float, hi: float) -> str:
    lo_s = f"{lo:g}"
    hi_s = f"{hi:g}"
    return f"{lo_s}\u2013{hi_s} Hz"


def build_metadata() -> dict:
    """Return canonical algorithm band-edge labels and track IDs."""
    epoch_bands = {name: _fmt_hz(lo, hi) for name, (lo, hi) in EpochRatioFeatureGenerator.BANDS.items()}
    epoch_bands["gamma"] = _fmt_hz(30.0, FS / 2.0)
    track_bands = {name: _fmt_hz(lo, hi) for name, (lo, hi) in TRACK_BANDS.items()}
    return {
        "sample_rate_hz": FS,
        "epoch_ratio_bands": epoch_bands,
        "track_bands": track_bands,
        "track_ids": [f"track.{name}" for name in TRACKS],
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_metadata(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
