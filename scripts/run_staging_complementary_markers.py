"""Cache complementary waveform descriptors and run the matched stage-pair probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_staging_feature_tuning import BASE, save_frame
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    RAW,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
    pin_manifest,
)
from physionet_sleep.analysis.epoch_windows import spindle_burst_counts
from physionet_sleep.experiments import complementary_n2_events as n2
from physionet_sleep.experiments import complementary_slow_waves as slow
from physionet_sleep.experiments.waveform_clock import active_clock
from physionet_sleep.experiments.waveform_data import EPOCH_SAMPLES, WaveformStore


OUT = EXPLORATION / "cinc-complementary-markers"
FEATURE_CACHE = OUT / "feature_cache"
KEY_FOLD = [*KEYS, LABEL, "fold"]
COLUMNS = (*slow.SLOW_WAVE_PANEL_COLUMNS, *n2.N2_EVENT_COLUMNS)
FRONTAL_QUALITY = [
    "second",
    *(
        f"{lead}_{q}"
        for lead in ("F3", "F4")
        for q in ("flat", "line", "ecg_corr", "ptp")
    ),
]
PROBES = {
    "N2_N3": (
        3,
        ["frontal_slow_wave_fraction", "delta_power"],
        list(slow.SLOW_WAVE_PANEL_COLUMNS),
    ),
    "N2_N1": (1, ["burst_seconds", "rel_beta", "rel_theta"], list(n2.N2_EVENT_COLUMNS)),
}
ARCHIVED_MEASUREMENTS = EXPLORATION / "cinc-cap-audit/feature_failure_epochs.parquet"
ARCHIVED_ERRORS = EXPLORATION / "cinc-error-interactions/archived_error_epochs.parquet"
ARCHIVED_WAVES = EXPLORATION / "cinc-feature-tuning/waveform_cache"


def extract(waveforms: WaveformStore, subject: str) -> pd.DataFrame:
    raw_front = [waveforms.channel_uv(subject, name) for name in ("F3-M2", "F4-M1")]
    quality = pd.read_parquet(
        CACHE / "quality1s" / f"{subject}.parquet", columns=FRONTAL_QUALITY
    )
    tracks = pd.read_parquet(
        CACHE / "tracks1s" / f"{subject}.parquet",
        columns=["c_rel_sigma", "c_rel_alpha", "c_rel_beta"],
    )
    n_epochs = len(raw_front[0]) // EPOCH_SAMPLES
    spindle = spindle_burst_counts(tracks)[:n_epochs]
    waves = slow.slow_wave_panel(*raw_front, quality)
    events = n2.n2_event_panel(*raw_front, quality, spindle)
    if not waves.index.equals(events.index):
        raise ValueError(f"{subject}: feature families use different epoch clocks")
    return pd.concat([waves, events], axis=1)


def check_night(features: pd.DataFrame, epochs: np.ndarray, subject: str) -> None:
    if tuple(features.columns) != COLUMNS or not features.index.equals(
        pd.RangeIndex(len(features), name="epoch_index")
    ):
        raise ValueError(f"{subject}: feature columns or contiguous epoch grid differ")
    if len(features) <= epochs.max() or np.isinf(features.to_numpy()).any():
        raise ValueError(f"{subject}: missing scored epoch or infinite feature")


def build_cache(frame: pd.DataFrame, subjects: list[str], smoke: bool) -> bool:
    all_subjects = sorted(frame.subject_id.unique())
    sources = [Path(n2.__file__), Path(slow.__file__)]
    pin_manifest(
        OUT / "feature_manifest.json",
        {
            "subjects": all_subjects,
            "columns": list(COLUMNS),
            "detectors": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
        },
    )
    by_subject = frame.groupby("subject_id", sort=False).epoch_index.apply(np.asarray)
    waveforms = WaveformStore(RAW, permitted_subjects=all_subjects, max_open_nights=1)
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    ledger = OUT / "extraction_time.json"
    used = json.loads(ledger.read_text())["seconds"] if ledger.exists() else 0.0
    started = active_clock()
    complete = 0
    cached = 0
    finite = np.zeros(len(COLUMNS), dtype=int)
    slow_reference = 0.0
    total_epochs = 0
    try:
        for subject in subjects:
            if (
                OUT / "PAUSE_CACHE"
            ).exists() or used + active_clock() - started >= 86400:
                return False
            path = FEATURE_CACHE / f"{subject}.parquet"
            if path.exists():
                features = pd.read_parquet(path)
                check_night(features, by_subject[subject], subject)
                cached += 1
            else:
                features = extract(waveforms, subject)
                check_night(features, by_subject[subject], subject)
                pending = path.with_suffix(".part")
                features.to_parquet(pending)
                pending.replace(path)
            if smoke:
                old = pd.read_parquet(
                    ARCHIVED_WAVES / f"{subject}.parquet",
                    columns=["slow_wave_75_best_clean"],
                )
                target = features["slow_wave_75_max_clean"].dropna()
                if not len(target):
                    raise ValueError(f"{subject}: missing nominal slow-wave reference")
                diff = np.max(
                    np.abs(old.loc[target.index, "slow_wave_75_best_clean"] - target)
                )
                if diff > 1e-7:
                    raise ValueError(
                        f"{subject}: calibrated 75 microvolt coverage changed by {diff}"
                    )
                slow_reference = max(slow_reference, float(diff))
            finite += features.notna().sum().to_numpy()
            total_epochs += len(features)
            complete += 1
            if complete == 1 or complete % 32 == 0:
                log(
                    f"Complementary markers: {complete}/{len(subjects)} nights, {active_clock() - started:.1f}s"
                )
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - started})
    summary = {
        "patients": complete,
        "complete_epochs": total_epochs,
        "reused_nights": cached,
        "active_seconds": used + active_clock() - started,
        "availability": dict(
            zip(COLUMNS, (finite / total_epochs).tolist(), strict=True)
        ),
    }
    if smoke:
        summary["nominal_slow_wave_max_abs_difference"] = slow_reference
    atomic_json(
        OUT / ("extraction_smoke.json" if smoke else "extraction_summary.json"), summary
    )
    return True


def align(base: pd.DataFrame, extra: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    joined = base.merge(
        extra, on=keys, how="left", validate="one_to_one", indicator=True
    )
    if len(joined) != len(base) or not joined["_merge"].eq("both").all():
        raise ValueError("Saved development keys or features differ")
    return joined.drop(columns="_merge")


def fit_probe(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=300),
    )
    model.fit(train[columns], train[LABEL].eq(2))
    return model.predict_proba(test[columns])[:, 1]


def read_marker_rows(subjects: list[str]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(FEATURE_CACHE / f"{subject}.parquet")
            .reset_index()
            .assign(subject_id=subject)
            for subject in subjects
        ],
        ignore_index=True,
    )


def run_profiles() -> None:
    from audit_staging_sustained_errors import (
        aggregate_cells,
        read_archives,
        reference_and_error_runs,
        within_cells,
    )

    frame = reference_and_error_runs(read_archives())
    frame = align(frame, read_marker_rows(sorted(frame.subject_id.unique())), KEYS)
    tables = [
        within_cells(frame, list(COLUMNS), pair)
        for pair in [(2, 3), (3, 2), (2, 1), (1, 2)]
    ]
    profile = aggregate_cells(pd.concat(tables, ignore_index=True))
    profile.to_csv(
        OUT / "complementary_matched_features.csv", index=False, float_format="%.6f"
    )
    log("Saved matched patient and night-quarter profiles for the new measurements")


def probe_metrics(frame: pd.DataFrame, name: str) -> dict:
    def score(sub: pd.DataFrame) -> dict:
        y = sub[LABEL].eq(2)
        return {
            "epochs": len(sub),
            "patients": sub.subject_id.nunique(),
            "auc": float(roc_auc_score(y, sub[name])),
            "log_loss": float(log_loss(y, sub[name])),
        }

    results = {
        "all": score(frame),
        "hard": score(frame[frame["hard"]]),
        "folds": [
            dict(fold=int(fold), all=score(rows), hard=score(rows[rows["hard"]]))
            for fold, rows in frame.groupby("fold")
        ],
    }
    hard_errors = frame[frame["strict_sustained"]]
    results["sustained"] = {}
    for label, rows in hard_errors.groupby(LABEL):
        p_true = rows[name] if label == 2 else 1 - rows[name]
        results["sustained"][str(label)] = {
            "epochs": len(rows),
            "patients": rows.subject_id.nunique(),
            "mean_true_probability": float(p_true.mean()),
            "truth_choices": int((p_true > 0.5).sum()),
        }
    return results


def run_probes(frame: pd.DataFrame, splits) -> None:
    old_columns = sorted({col for _, old, _ in PROBES.values() for col in old})
    data = align(
        frame[KEYS + [LABEL]],
        pd.read_parquet(ARCHIVED_MEASUREMENTS, columns=KEY_FOLD + old_columns),
        KEYS + [LABEL],
    )
    probabilities = pd.read_parquet(BASE, columns=KEY_FOLD + PROBABILITIES)
    data = align(data, probabilities, KEY_FOLD)
    sustained = pd.read_parquet(
        ARCHIVED_ERRORS,
        columns=KEY_FOLD
        + [
            "searched",
            "baseline_to_searched",
            "paired_run_6",
        ],
    )
    data = align(data, sustained, KEY_FOLD)
    data = align(data, read_marker_rows(sorted(data.subject_id.unique())), KEYS)
    probs = data[PROBABILITIES].to_numpy()
    rank = np.argsort(-probs, axis=1, kind="stable")[:, :2]
    p = np.take_along_axis(probs, rank, axis=1)
    data["close"] = p[:, 0] <= 2 * p[:, 1]
    data[["top_a", "top_b"]] = np.sort(rank, axis=1)
    results = {}
    for pair, (rival, old, additions) in PROBES.items():
        outputs = []
        for fold, split in enumerate(splits):
            train_keys, _, test_keys = fold_frames(frame, split)
            train = data.iloc[train_keys.index]
            test = data.iloc[test_keys.index]
            train = train.loc[train[LABEL].isin([2, rival])]
            test = test.loc[test[LABEL].isin([2, rival])]
            if not test.fold.eq(fold).all() or set(train.subject_id) & set(
                test.subject_id
            ):
                raise ValueError(
                    "Probe partitions differ from the original held-out patient folds"
                )
            a, b = sorted([2, rival])
            outcomes = test[KEY_FOLD].copy()
            outcomes["hard"] = test["close"] & test.top_a.eq(a) & test.top_b.eq(b)
            outcomes["strict_sustained"] = (
                test.paired_run_6
                & test.baseline_to_searched.eq("persistent_same")
                & test.searched.isin([2, rival])
            )
            outcomes["original"] = fit_probe(train, test, old)
            outcomes["complementary"] = fit_probe(train, test, old + additions)
            outputs.append(outcomes)
        scored = pd.concat(outputs, ignore_index=True)
        results[pair] = {
            name: probe_metrics(scored, name) for name in ("original", "complementary")
        }
        save_frame(OUT / f"{pair.lower()}_probe_predictions.parquet", scored)
        log(
            f"{pair} hard-case AUC: {results[pair]['original']['hard']['auc']:.4f} to "
            f"{results[pair]['complementary']['hard']['auc']:.4f}"
        )
    atomic_json(OUT / "probe_summary.json", results)


def execute(smoke: bool, probe_only: bool, profile_only: bool) -> int:
    if not (OUT / "FINDINGS.md").exists():
        raise FileNotFoundError(
            "Declare the experiments before raw extraction or scoring"
        )
    frame, splits = development_frame()
    if smoke:
        inner, _, _ = fold_frames(frame, splits[0])
        return (
            0 if build_cache(frame, sorted(inner.subject_id.unique())[:2], True) else 75
        )
    if not (OUT / "extraction_smoke.json").exists():
        raise FileNotFoundError("Run the training-only extraction smoke first")
    if not (probe_only or profile_only) and not build_cache(
        frame, sorted(frame.subject_id.unique()), False
    ):
        return 75
    if not profile_only:
        start = perf_counter()
        run_probes(frame, splits)
        log(f"Complementary feature probes saved in {perf_counter() - start:.1f}s")
    run_profiles()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--probe-only", action="store_true")
    mode.add_argument("--profile-only", action="store_true")
    options = parser.parse_args()
    with local_training_slot(), threadpool_limits(limits=4):
        raise SystemExit(
            execute(options.smoke, options.probe_only, options.profile_only)
        )
