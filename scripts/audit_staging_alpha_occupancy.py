"""Compare alpha occupancy with mean share on fixed hard Wake/N1 cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit_staging_wake_n1 import OUT, annotation_phases
from run_staging_feature_tuning import BASE
from run_waveform_development import (
    CACHE,
    KEYS,
    LABEL,
    PROBABILITIES,
    SEED,
    atomic_json,
)


MARKERS = ("mean_alpha", "alpha_occupancy")
TIMING = ("later_return_to_sleep", "later_awakening", "stable_interior")


def alpha_epochs(subject: str, epoch: np.ndarray) -> np.ndarray:
    track = pd.read_parquet(
        CACHE / "tracks1s" / f"{subject}.parquet", columns=["second", "o_rel_alpha"]
    )
    if not np.array_equal(track.second, np.arange(len(track))):
        raise ValueError(f"{subject}: occipital track must follow the 1-second clock")
    alpha = track.o_rel_alpha.to_numpy(dtype=float)
    if not np.isfinite(alpha).any():
        return np.full((len(epoch), len(MARKERS)), np.nan)
    threshold = 0.70 * np.nanpercentile(alpha, 90)
    end = max(len(alpha), (int(epoch.max()) + 1) * 30)
    values = np.pad(
        alpha, (0, end - len(alpha) + (-end % 30)), constant_values=np.nan
    ).reshape(-1, 30)[epoch]
    count = np.isfinite(values).sum(1)
    safe = np.maximum(count, 1)
    mean = np.nansum(values, axis=1) / safe
    occupancy = (np.isfinite(values) & (values >= threshold)).sum(1) / safe
    return np.where(count[:, None] >= 24, np.column_stack([mean, occupancy]), np.nan)


def matched_auc(frame: pd.DataFrame) -> dict:
    entries = []
    for (patient, _), rows in frame.groupby(["subject_id", "quarter"], sort=False):
        truth = rows[LABEL].eq(0).to_numpy()
        if min(truth.sum(), (~truth).sum()) < 2:
            continue
        scores = {col: roc_auc_score(truth, rows[col]) for col in MARKERS}
        entries.append({"subject_id": patient, "epochs": len(rows), **scores})
    if not entries:
        return {"cells": 0}
    cells = pd.DataFrame(entries)
    by_patient = cells.groupby("subject_id")[list(MARKERS)].mean()
    delta = (by_patient["alpha_occupancy"] - by_patient["mean_alpha"]).to_numpy()
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(1)
    return {
        "patients": len(by_patient),
        "epochs": int(cells.epochs.sum()),
        "cells": len(cells),
        **by_patient.mean().to_dict(),
        "delta": float(delta.mean()),
        "paired_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the marker comparison before scoring")
    frame = pd.read_parquet(BASE).sort_values(KEYS).reset_index(drop=True)
    frame["phase"], _ = annotation_phases(frame)
    frame["quarter"] = (
        frame.epoch_index
        * 4
        // (frame.groupby("subject_id").epoch_index.transform("max") + 1)
    )
    p = frame[PROBABILITIES].to_numpy()
    top = np.argsort(-p, axis=1, kind="stable")[:, :2]
    scores = np.take_along_axis(p, top, axis=1)
    hard = (
        frame[LABEL].isin([0, 1])
        & (np.sort(top, axis=1) == [0, 1]).all(1)
        & (scores[:, 0] <= 2 * scores[:, 1])
    )
    hard_frame = frame.loc[hard].copy()
    results = []
    for patient, rows in hard_frame.groupby("subject_id", sort=False):
        sub = rows[[*KEYS, LABEL, "phase", "quarter"]].copy()
        sub[list(MARKERS)] = alpha_epochs(patient, rows.epoch_index.to_numpy())
        results.append(sub)
    matched = pd.concat(results).dropna(subset=list(MARKERS))
    comparisons = {"all_hard_epochs": matched_auc(matched)}
    comparisons.update(
        {phase: matched_auc(matched[matched.phase.eq(phase)]) for phase in TIMING}
    )
    atomic_json(OUT / "wake_n1_alpha_occupancy.json", comparisons)


if __name__ == "__main__":
    main()
