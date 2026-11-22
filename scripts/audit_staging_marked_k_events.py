"""Evaluate the frozen K-like detector on expert-marked DREAMS EEG excerpts."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physionet_sleep.algorithms.eeg_quality import LINE_BAND, _frame_corr, bad_seconds
from physionet_sleep.algorithms.eeg_tracks import EPS, band_power, power_spectra
from physionet_sleep.experiments.complementary_n2_events import (
    DETECTOR_HZ,
    EPOCH_SAMPLES,
    EPOCH_SECONDS,
    SAMPLE_RATE,
    _clean_epochs,
    _event_troughs,
)


OUT = ROOT / "artifacts/exploration/cinc-stage-pair-revisit/k_events"
EEG = ("CZ-A1", "FP1-A2", "FP2-A1")
FRONTAL_JOIN_SECONDS = 0.20


def read_dreams_channels(path: Path) -> dict[str, np.ndarray]:
    """Read the selected channels in documented fixed-record, 16-bit DREAMS EDFs."""
    with path.open("rb") as stream:
        header = stream.read(256)
        count, records = int(header[252:256]), int(header[236:244])
        stream.seek(256)
        fields = []
        for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
            block = stream.read(width * count)
            fields.append(
                [
                    block[k : k + width].decode("latin-1").strip()
                    for k in range(0, len(block), width)
                ]
            )
        labels, _, units, pmin, pmax, dmin, dmax, _, sizes, _ = fields
    header_bytes = int(header[184:192])
    duration = float(header[244:252])
    offsets = np.r_[0, np.cumsum([int(value) for value in sizes])]
    if (
        header_bytes != 256 * (count + 1)
        or path.stat().st_size != header_bytes + records * offsets[-1] * 2
    ):
        raise ValueError(f"{path.name}: EDF record grid is inconsistent")
    raw = np.memmap(
        path, mode="r", dtype="<i2", offset=header_bytes, shape=(records, offsets[-1])
    )
    signals = {}
    for lead in (*EEG, "ECG"):
        j = labels.index(lead)
        rate = int(sizes[j]) / duration
        if rate != SAMPLE_RATE or units[j] != ("MV" if lead == "ECG" else "UV"):
            raise ValueError(f"{path.name}: unexpected clock or units for {lead}")
        digital = np.asarray(raw[:, offsets[j] : offsets[j + 1]].ravel(), dtype=float)
        physical_span = float(pmax[j]) - float(pmin[j])
        digital_span = float(dmax[j]) - float(dmin[j])
        if min(physical_span, digital_span) <= 0:
            raise ValueError(f"{path.name}: invalid calibration for {lead}")
        signals[lead] = (digital - float(dmin[j])) * (
            physical_span / digital_span
        ) + float(pmin[j])
    return signals


def detect(
    signal: np.ndarray, ecg_filtered: np.ndarray, quality_filter: np.ndarray
) -> np.ndarray:
    """Apply Boston's existing quality gate, preparation, and trough detector."""
    seconds = len(signal) // SAMPLE_RATE
    spectra = power_spectra(signal, seconds)
    amplitude = np.ptp(signal.astype(np.float32).reshape(seconds, SAMPLE_RATE), axis=1)
    quality = pd.DataFrame(
        {
            "CZ_flat": amplitude == 0,
            "CZ_ptp": amplitude,
            "CZ_line": spectra[:, LINE_BAND].sum(axis=1)
            / (band_power(spectra, 0.5, 30) + EPS),
            "CZ_ecg_corr": _frame_corr(
                sosfiltfilt(quality_filter, signal), ecg_filtered, seconds
            ),
        }
    )
    bad = bad_seconds(quality, "CZ").reshape(-1, EPOCH_SECONDS)
    filtered, bad, usable = _clean_epochs(signal.reshape(-1, EPOCH_SAMPLES), bad)
    return _event_troughs(filtered, bad, usable)


def intervals(path: Path, scorer: int, elapsed: float) -> np.ndarray:
    with path.open(encoding="utf-8-sig") as handle:
        if handle.readline().strip() != f"[vis{scorer}_Kcomplexes/CZ-A1]":
            raise ValueError(f"{path.name}: unexpected visual annotation channel")
        marks = np.loadtxt(handle, ndmin=2)
    if (
        marks.shape[1] != 2
        or not np.isfinite(marks).all()
        or (
            (marks[:, 0] < 0).any()
            or (marks[:, 1] <= 0).any()
            or (marks.sum(1) > elapsed).any()
        )
    ):
        raise ValueError(f"{path.name}: invalid event intervals")
    return np.column_stack((marks[:, 0], marks.sum(axis=1)))


def match_points(points: np.ndarray, windows: np.ndarray) -> int:
    """Maximum-cardinality one-to-one point-and-interval matches."""
    queue = []
    windows = windows[np.argsort(windows[:, 0], kind="stable")]
    cursor = 0
    matches = 0
    for point in points:
        while cursor < len(windows) and windows[cursor, 0] <= point:
            heapq.heappush(queue, windows[cursor, 1])
            cursor += 1
        while queue and queue[0] < point:
            heapq.heappop(queue)
        if queue:
            heapq.heappop(queue)
            matches += 1
    return matches


def excerpt(n: int, quality_filter: np.ndarray) -> list[dict]:
    name = f"excerpt{n}"
    channels = read_dreams_channels(OUT / f"{name}.edf")
    central = channels["CZ-A1"]
    with (OUT / f"{name}.txt").open(encoding="utf-8-sig") as handle:
        if handle.readline().strip() != "[CZ-A1]":
            raise ValueError(f"{name}: missing central EEG channel declaration")
        text_values = np.loadtxt(handle)
    np.testing.assert_allclose(central, text_values, atol=1e-6, rtol=0)
    if len(central) != SAMPLE_RATE * 30 * 60:
        raise ValueError(f"{name}: expected a 30-minute excerpt")
    ecg_filtered = sosfiltfilt(quality_filter, channels["ECG"] * 1_000)
    events = {
        lead: detect(channels[lead], ecg_filtered, quality_filter) for lead in EEG
    }
    both = np.sort(np.r_[events["FP1-A2"], events["FP2-A1"]])
    if len(both):
        both = both[
            np.r_[True, np.diff(both) > round(FRONTAL_JOIN_SECONDS * DETECTOR_HZ)]
        ]
    elapsed = len(central) / SAMPLE_RATE
    results = []
    for scorer in range(1, 3 if n <= 5 else 2):
        truth = intervals(OUT / f"Visual_scoring{scorer}_{name}.txt", scorer, elapsed)
        midpoint = truth.mean(axis=1)
        for montage, sample in (("central", events["CZ-A1"]), ("frontal_poles", both)):
            points = sample / DETECTOR_HZ
            for match, windows in (
                ("scored_interval", truth),
                (
                    "midpoint_half_second",
                    np.column_stack((midpoint - 0.5, midpoint + 0.5)),
                ),
            ):
                results.append(
                    {
                        "excerpt": n,
                        "scorer": scorer,
                        "montage": montage,
                        "match": match,
                        "reference": len(truth),
                        "detected": len(points),
                        "matched": match_points(points, windows),
                        "hours": elapsed / 3600,
                    }
                )
    return results


def main() -> None:
    filt = butter(4, [5, 40], btype="band", fs=SAMPLE_RATE, output="sos")
    counts = pd.DataFrame([row for n in range(1, 11) for row in excerpt(n, filt)])
    keys = ["scorer", "montage", "match"]
    nums = ["reference", "detected", "matched", "hours"]
    all_excerpts = (
        counts.groupby(keys)[nums].sum().reset_index().assign(subset="available")
    )
    shared = counts.loc[counts.excerpt.le(5)].groupby(keys)[nums].sum().reset_index()
    total = pd.concat(
        [all_excerpts, shared.assign(subset="first_five")], ignore_index=True
    )
    total["sensitivity"] = total["matched"] / total["reference"]
    total["precision"] = total["matched"] / total["detected"]
    total["unmatched_per_hour"] = (total["detected"] - total["matched"]) / total[
        "hours"
    ]
    counts.to_csv(OUT / "event_counts_by_excerpt.csv", index=False)
    total.to_csv(OUT / "event_metrics.csv", index=False)
    summary = {
        "dataset": "DREAMS K-complexes Database",
        "doi": "10.5281/zenodo.2650142",
        "score_reference": "CZ-A1",
        "waveform_thresholds": "frozen Boston detector",
        "metrics": total.to_dict("records"),
    }
    (OUT / "event_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(total.to_string(index=False))


if __name__ == "__main__":
    main()
