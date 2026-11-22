"""Build calibrated candidate features for the registered CinC development cohort."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physionet_sleep.experiments.staging_feature_waveform import RawStagingFeatureCache


def main() -> None:
    manifest = ROOT / "artifacts/staging_window_model/cinc2018/dev_subjects.txt"
    subjects = manifest.read_text().split()
    if len(subjects) != 320 or len(set(subjects)) != 320:
        raise ValueError("expected the fixed 320-patient CinC development manifest")
    input_cache = ROOT / "artifacts/cinc2018_window_cache"
    output_cache = ROOT / "artifacts/exploration/cinc-feature-tuning/waveform_cache"
    cache = RawStagingFeatureCache(
        ROOT / "data/raw/cinc2018", input_cache / "quality1s", output_cache, permitted_subjects=subjects
    )
    baseline = "topography_features.frontal_slow_wave_fraction"
    start = perf_counter()
    max_reference_difference = 0.0
    total_epochs = 0
    for i, subject in enumerate(subjects, 1):
        candidate = cache.read(subject)
        rows = pd.read_parquet(input_cache / "rows6" / f"{subject}.parquet", columns=["slot", baseline])
        rows = rows.loc[rows.slot.mod(5).eq(0)].copy()
        rows["epoch_index"] = rows.pop("slot") // 5
        reference = rows.set_index("epoch_index")[baseline].dropna()
        matched = reference.loc[reference.index.intersection(candidate.index)]
        differences = np.abs(candidate.loc[matched.index, "slow_wave_75_mean"] - matched)
        if differences.empty or (differences > 1e-7).any():
            raise ValueError(f"{subject}: waveform clock or slow-wave baseline differs from the reference cache")
        max_reference_difference = max(max_reference_difference, float(differences.max()))
        total_epochs += len(candidate)
        if i == 1 or i % 10 == 0:
            print(f"{i}/{len(subjects)} patients, {total_epochs:,} complete epochs, {perf_counter() - start:.1f} s", flush=True)
    print(f"Maximum reference slow-wave difference {max_reference_difference:.3g}", flush=True)


if __name__ == "__main__":
    main()
