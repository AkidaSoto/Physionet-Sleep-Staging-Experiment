"""Inventory an 80-patient training expansion from locally indexed Boston records."""

from __future__ import annotations

from shutil import disk_usage

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from run_waveform_development import (
    BASELINE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    RAW,
    SEED,
    atomic_json,
)
from select_cinc2018_holdout import band
from physionet_sleep.io.cinc2018 import discover_cinc2018_records
from physionet_sleep.labels.cinc2018 import (
    RESPIRATORY_TYPES,
    load_cinc2018_annotations,
    stage_epochs,
)


OUT = EXPLORATION / "cinc-wake-n1-data"
GIB = 2**30
INCREMENT = 80
AHI_BANDS = ["<5", "5-15", "15-30", ">=30"]


def patient_metadata(record, dev: set[str]) -> dict:
    changes, events, length = load_cinc2018_annotations(record)
    stages = stage_epochs(changes, length)
    sleep = np.count_nonzero(stages > 0) / 120
    if sleep == 0:
        raise ValueError(f"{record.record_id}: no scored sleep")
    apnea = sum(name in RESPIRATORY_TYPES for _, _, name in events)
    left, right = stages[:-1], stages[1:]
    starts = int(((left == 0) & (right == 1)).sum())
    wakes = int(((left == 1) & (right == 0)).sum())
    header = record.header_path.read_text().splitlines()[0].split()
    expected_bytes = 24 + 2 * int(header[1]) * int(header[3])
    local_waveform = (
        record.signal_path is not None
        and record.signal_path.stat().st_size == expected_bytes
    )
    return {
        "subject_id": record.record_id,
        "cohort": "development" if record.record_id in dev else "unused",
        "ahi_band": band(apnea / sleep),
        "hours": length / 3600,
        "n1_epochs": int((stages == 1).sum()),
        "wake_epochs": int((stages == 0).sum()),
        "wake_to_n1": starts,
        "n1_to_wake": wakes,
        "transitions_per_hour": (starts + wakes) / (length / 3600),
        "waveform_bytes": expected_bytes,
        "waveform_present": local_waveform,
    }


def quotas(target: np.ndarray, capacity: np.ndarray, total: int) -> np.ndarray:
    """Allocate to the largest target shortfall that still has members available."""
    if capacity.sum() < total:
        raise ValueError("The unused training pool cannot fill the proposed cohort")
    selected = np.zeros_like(capacity)
    for _ in range(total):
        deficits = np.where(selected < capacity, target - selected, -np.inf)
        selected[deficits.argmax()] += 1
    return selected


def propose(data: pd.DataFrame) -> pd.DataFrame:
    dev = data.loc[data.cohort.eq("development")]
    pool = data.loc[data.cohort.eq("unused")]
    dev_bands = dev.ahi_band.value_counts().reindex(AHI_BANDS, fill_value=0).to_numpy()
    available = pool.ahi_band.value_counts().reindex(AHI_BANDS, fill_value=0).to_numpy()
    n_band = quotas(INCREMENT * dev_bands / len(dev), available, INCREMENT)
    density_cuts = np.quantile(dev.transitions_per_hour, [0.25, 0.50, 0.75])
    rng = np.random.default_rng(SEED)
    chosen = []
    for name, goal in zip(AHI_BANDS, n_band, strict=True):
        candidates = pool.loc[pool.ahi_band.eq(name)].copy()
        candidates["transition_quartile"] = np.searchsorted(
            density_cuts, candidates.transitions_per_hour
        )
        reference = np.searchsorted(
            density_cuts, dev.loc[dev.ahi_band.eq(name), "transitions_per_hour"]
        )
        if not len(reference):
            reference = np.searchsorted(density_cuts, dev.transitions_per_hour)
        target = goal * np.bincount(reference, minlength=4) / len(reference)
        capacity = (
            candidates.transition_quartile.value_counts()
            .reindex(range(4), fill_value=0)
            .to_numpy()
        )
        for quartile, n in enumerate(quotas(target, capacity, int(goal))):
            group = candidates.loc[
                candidates.transition_quartile.eq(quartile)
            ].sort_values("subject_id")
            chosen.extend(rng.choice(group.index, size=n, replace=False).tolist())
    return pool.loc[chosen].sort_values("subject_id")


def describe(rows: pd.DataFrame) -> dict:
    return {
        "patients": len(rows),
        "with_complete_waveform": int(rows.waveform_present.sum()),
        "ahi_bands": {key: int(rows.ahi_band.eq(key).sum()) for key in AHI_BANDS},
        "hours": float(rows.hours.sum()),
        "n1_epochs": int(rows.n1_epochs.sum()),
        "wake_epochs": int(rows.wake_epochs.sum()),
        "wake_to_n1": int(rows.wake_to_n1.sum()),
        "n1_to_wake": int(rows.n1_to_wake.sum()),
        "median_transitions_per_hour": float(rows.transitions_per_hour.median()),
        "waveform_gib_to_fetch": float(
            rows.loc[~rows.waveform_present, "waveform_bytes"].sum() / GIB
        ),
    }


def compare_by_ahi(dev: pd.DataFrame) -> None:
    small = OUT / "data_curve/predictions.parquet"
    if not small.exists():
        return
    reference = EXPLORATION / "cinc-complementary-markers/expanded_predictions.parquet"
    keys = [*KEYS, LABEL, "fold"]
    a, b = (
        pd.read_parquet(p, columns=[*keys, *PROBABILITIES]) for p in (small, reference)
    )
    merged = a.merge(b, on=keys, validate="one_to_one", suffixes=("_128", "_full"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Data-curve arms differ on development evaluation epochs")
    merged = merged.merge(
        dev[["subject_id", "ahi_band"]], on="subject_id", validate="many_to_one"
    )
    if len(merged) != len(a):
        raise ValueError("Expected an AHI band for every development prediction")
    rows = []
    for band_name, case in merged.groupby("ahi_band", sort=False):
        label = case[LABEL].to_numpy()
        for name in ("128", "full"):
            pred = case[[f"{p}_{name}" for p in PROBABILITIES]].to_numpy().argmax(1)
            f1 = f1_score(label, pred, labels=np.arange(5), average=None)
            rows.append(
                {
                    "ahi_band": band_name,
                    "model": name,
                    "patients": case.subject_id.nunique(),
                    "pooled_macro_f1": f1.mean(),
                    "wake_f1": f1[0],
                    "n1_f1": f1[1],
                    "n1_as_wake_rate": (pred[label == 1] == 0).mean(),
                    "wake_as_n1_rate": (pred[label == 0] == 1).mean(),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "data_curve_ahi.csv", index=False)


def main() -> None:
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the expansion inventory first")
    dev = set((BASELINE / "dev_subjects.txt").read_text().split())
    reserved = set((RAW / "selected_records_holdout.txt").read_text().split())
    ledger = set(
        pd.read_csv(BASELINE / "holdout_subjects.csv", usecols=["record_id"]).record_id
    )
    if len(dev) != 320 or len(reserved) != 200 or dev & reserved or ledger != reserved:
        raise ValueError(
            "Boston development and independent holdout manifests differ from expectations"
        )
    records = discover_cinc2018_records(RAW, require_signals=False)
    metadata = [
        patient_metadata(record, dev)
        for record in records
        if record.record_id not in reserved and record.annotation_path is not None
    ]
    data = pd.DataFrame(metadata)
    if set(data.loc[data.cohort.eq("development"), "subject_id"]) != dev:
        raise ValueError("Development metadata are incomplete")
    if len(data) != 794 or data.subject_id.nunique() != len(data):
        raise ValueError(
            "Expected one locally indexed record per listed unused or development patient"
        )
    selected = propose(data)
    if len(selected) != INCREMENT or set(selected.subject_id) & (dev | reserved):
        raise ValueError(
            "The prospective list contains overlapping or missing patients"
        )
    selected.to_csv(OUT / "prospective_train80.csv", index=False)
    compare_by_ahi(data[data.cohort.eq("development")])
    (OUT / "prospective_train80.txt").write_text(
        "\n".join(selected.subject_id) + "\n", encoding="utf-8"
    )
    atomic_json(
        OUT / "expansion_inventory.json",
        {
            "free_disk_gib": disk_usage(RAW).free / GIB,
            "reserved_patients_excluded": len(reserved),
            "development": describe(data[data.cohort.eq("development")]),
            "unused": describe(data[data.cohort.eq("unused")]),
            "prospective": describe(selected),
        },
    )
    print(
        f"Inventoried {len(data) - len(dev)} unused patients and prepared {len(selected)} training-only candidates"
    )


if __name__ == "__main__":
    main()
