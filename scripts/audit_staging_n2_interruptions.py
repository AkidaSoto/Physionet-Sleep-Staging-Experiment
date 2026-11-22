"""Measure EEG interruptions and test them against recorded non-REM arousals."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from run_staging_feature_tuning import save_frame
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    KEYS,
    LABEL,
    RAW,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
)

from physionet_sleep.algorithms.eeg_quality import QUALITY_RULES, bad_seconds
from physionet_sleep.analysis.epoch_windows import spindle_burst_counts
from physionet_sleep.experiments.complementary_n2_events import MAX_LAG, _recent_support
from physionet_sleep.io.cinc2018 import resolve_cinc2018_record
from physionet_sleep.labels.cinc2018 import load_cinc2018_annotations

OUT = EXPLORATION / "cinc-stage-pair-revisit"
MARKERS = EXPLORATION / "cinc-complementary-markers/feature_cache"
COLUMNS = (
    "interruption_strength",
    "post_interrupt_recent_event_support",
    "recent_interruption",
)
CENTRAL = ("C3", "C4")
BANDS = ("delta", "theta", "alpha", "beta", "total")
QUALITY_FIELDS = (
    "second",
    *(f"{lead}_{key}" for lead in CENTRAL for key in (*BANDS, *QUALITY_RULES)),
)
SPINDLE_FIELDS = ("second", "c_rel_sigma", "c_rel_alpha", "c_rel_beta")
THRESHOLD = np.log10(2)
EPOCH_SECONDS = 30
FRAME_END_OFFSET = 2
WINDOW = 10
LABELS = {
    "arousal_spontaneous": "Spontaneous arousal",
    "arousal_rera": "Respiratory effort-related arousal",
}


def sustained_shift(log_ratio: np.ndarray) -> np.ndarray:
    """Use nonoverlapping trailing EEG as background and emit at frame availability."""
    series = pd.Series(log_ratio)
    background = series.rolling(10, min_periods=8).median().shift(2)
    score = (series - background).rolling(3, min_periods=3).min()
    # Cached frame t covers [t - 0.5, t + 1.5]; its last sample is known by t + 2.
    return score.shift(FRAME_END_OFFSET).to_numpy()


def eeg_strength(quality: pd.DataFrame) -> np.ndarray:
    shifts = []
    for lead in CENTRAL:
        power = quality[[f"{lead}_{key}" for key in BANDS]].to_numpy(dtype=float)
        slow, theta, alpha, beta, total = power.T
        numerator = theta + alpha + beta
        usable = (
            np.isfinite(power).all(axis=1)
            & np.isfinite(
                quality[[f"{lead}_{key}" for key in QUALITY_RULES]].to_numpy()
            ).all(axis=1)
            & ~bad_seconds(quality, lead)
            & (total > 0)
            & (numerator >= 0)
            & (slow >= 0)
        )
        floor = np.maximum(total, 0) * 1e-6 + np.finfo(float).tiny
        ratio = np.log10((numerator + floor) / (slow + floor))
        shifts.append(sustained_shift(np.where(usable, ratio, np.nan)))
    # Any retained burst must persist within an individual lead.
    return np.fmax(shifts[0], shifts[1])


def continuation(
    events: np.ndarray, spindle: np.ndarray, strength: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positive = (events > 0) | (spindle >= 2)
    known_event = positive | (np.isfinite(events) & np.isfinite(spindle))
    observed = np.isfinite(strength)
    interrupted = observed & (strength >= THRESHOLD)
    reset_support = np.zeros(len(events), dtype=np.float32)
    last_interrupt = np.zeros(len(events), dtype=np.float32)
    has_event_history = np.zeros(len(events), dtype=bool)
    has_eeg_history = np.zeros(len(events), dtype=bool)
    clear_after_event = np.ones(len(events), dtype=bool)
    for lag in range(1, min(MAX_LAG, len(events) - 1) + 1):
        clear_after_event[lag:] &= ~interrupted[:-lag]
        decay = 2 ** (-lag / 2)
        available = positive[:-lag] & clear_after_event[lag:]
        reset_support[lag:] = np.maximum(reset_support[lag:], available * decay)
        last_interrupt[lag:] = np.maximum(
            last_interrupt[lag:], interrupted[:-lag] * decay
        )
        has_event_history[lag:] |= known_event[:-lag]
        has_eeg_history[lag:] |= observed[:-lag]
    return (
        np.where(has_event_history & has_eeg_history, reset_support, np.nan),
        np.where(has_eeg_history, last_interrupt, np.nan),
    )


def extract(subject: str) -> tuple[pd.DataFrame, np.ndarray]:
    saved = pd.read_parquet(
        MARKERS / f"{subject}.parquet",
        columns=["kc_shape_events", "n2_recent_event_support"],
    )
    count = len(saved)
    seconds = count * EPOCH_SECONDS
    quality = pd.read_parquet(
        CACHE / "quality1s" / f"{subject}.parquet", columns=QUALITY_FIELDS
    ).iloc[:seconds]
    tracks = pd.read_parquet(
        CACHE / "tracks1s" / f"{subject}.parquet", columns=SPINDLE_FIELDS
    )
    clock = np.arange(seconds)
    if not saved.index.equals(pd.RangeIndex(count, name="epoch_index")) or not (
        np.array_equal(quality["second"], clock)
        and np.array_equal(tracks["second"].iloc[:seconds], clock)
    ):
        raise ValueError(f"{subject}: source clocks do not align")
    spindle = spindle_burst_counts(tracks)[:count]
    events = saved.kc_shape_events.to_numpy()
    np.testing.assert_allclose(spindle.shape, events.shape)
    np.testing.assert_allclose(
        _recent_support(events, spindle), saved.n2_recent_event_support, rtol=0, atol=0
    )
    score = eeg_strength(quality)
    epoch = score.reshape(-1, EPOCH_SECONDS)
    positive = np.maximum(0, np.where(np.isfinite(epoch), epoch, -np.inf).max(axis=1))
    positive[~np.isfinite(epoch).any(axis=1)] = np.nan
    support, recent = continuation(events, spindle, positive)
    panel = pd.DataFrame(
        np.column_stack((positive, support, recent)), columns=COLUMNS, index=saved.index
    )
    return panel.astype(np.float32), score


def event_probe(subject: str, night: pd.DataFrame, score: np.ndarray) -> list[dict]:
    _, events, _ = load_cinc2018_annotations(resolve_cinc2018_record(RAW, subject))
    stages = np.full(len(score) // EPOCH_SECONDS, -1, dtype=int)
    stages[night.epoch_index] = night[LABEL]
    stages = np.repeat(stages, EPOCH_SECONDS)
    blocked = np.zeros(len(score), dtype=bool)
    for start, end, _ in events:
        begin, stop = (
            max(0, int(np.floor(start)) - 10),
            min(len(score), int(np.ceil(end)) + 11),
        )
        blocked[begin:stop] = True

    def strength(start: int) -> float:
        values = score[start : start + WINDOW]
        measured = values[np.isfinite(values)]
        return float(measured.max()) if len(measured) >= 5 else np.nan

    rows = []
    for start, _, kind in events:
        if kind not in LABELS:
            continue
        t = int(np.floor(start))
        if not (
            0 <= t <= len(score) - WINDOW
            and stages[t] in (1, 2, 3)
            and np.all(stages[t : t + WINDOW] == stages[t])
        ):
            continue
        case = strength(t)
        if not np.isfinite(case):
            continue
        controls = []
        for direction in (-1, 1):
            for step in range(30, 301, 30):
                candidate = t + direction * step
                if candidate < 0 or candidate + WINDOW > len(score):
                    continue
                span = slice(candidate, candidate + WINDOW)
                if blocked[span].any() or not np.all(stages[span] == stages[t]):
                    continue
                value = strength(candidate)
                if np.isfinite(value):
                    controls.append(value)
                    break
        for counter in controls:
            rows.append(
                {
                    "subject_id": subject,
                    "event": kind,
                    "stage": int(stages[t]),
                    "start_second": start,
                    "event_score": case,
                    "control_score": counter,
                    "ranking": float(case > counter) + 0.5 * float(case == counter),
                    "hit": bool(case >= THRESHOLD),
                    "control_hit": bool(counter >= THRESHOLD),
                }
            )
    return rows


def summarize(rows: pd.DataFrame) -> list[dict]:
    summary = []
    for event, subset in rows.groupby("event", sort=True):
        for fold in [None, *range(5)]:
            group = subset if fold is None else subset[subset.fold.eq(fold)]
            if group.empty:
                continue
            # Give the two flanks equal weight within an event and give patients equal weight.
            keys = ["subject_id", "start_second"]
            event_metrics = group.groupby(keys)[
                ["ranking", "hit", "control_hit"]
            ].mean()
            mean = event_metrics.groupby("subject_id").mean().mean()
            summary.append(
                {
                    "annotation": event,
                    "fold": fold,
                    "patients": event_metrics.index.levels[0].size,
                    "events": len(event_metrics),
                    "matched_windows": len(group),
                    "patient_matched_auc": float(mean.ranking),
                    "event_threshold_rate": float(mean.hit),
                    "control_threshold_rate": float(mean.control_hit),
                }
            )
    return summary


def smoke() -> None:
    ideal = np.zeros(120)
    ideal[35:40] = 0.9
    ideal[79:81] = 0.9
    result = sustained_shift(ideal)
    hits = np.flatnonzero(result >= THRESHOLD)
    np.testing.assert_array_equal(hits, [39, 40, 41])
    event = np.array([1, 0, 0, 0, 1, 0], dtype=float)
    spindle = np.zeros(len(event))
    activity = np.array([0, 0, 0.5, 0, 0, 0], dtype=float)
    support, recent = continuation(event, spindle, activity)
    np.testing.assert_allclose(support, [np.nan, 2**-0.5, 0.5, 0, 0, 2**-0.5])
    np.testing.assert_allclose(recent, [np.nan, 0, 0, 2**-0.5, 0.5, 2**-1.5])


def main(smoke_only: bool) -> None:
    started = perf_counter()
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the interruption audit before extraction")
    smoke()
    frame, splits = development_frame()
    memberships = np.full(len(frame), -1, dtype=int)
    for fold, part in enumerate(splits):
        memberships[part.test_idx] = fold
    frame["fold"] = memberships
    if smoke_only:
        inner, _, _ = fold_frames(frame, splits[0])
        subjects = sorted(inner.subject_id.unique())[:2]
        frame = frame[frame.subject_id.isin(subjects)]
    features = []
    records = []
    for subject, night in frame.groupby("subject_id", sort=True):
        panel, score = extract(subject)
        measured = night[KEYS + ["fold"]].merge(
            panel.reset_index(), on="epoch_index", validate="one_to_one"
        )
        if len(measured) != len(night):
            raise ValueError(
                f"{subject}: missing feature rows at scored development epochs"
            )
        features.append(measured)
        records.extend(
            dict(fold=int(night.fold.iloc[0]), **r)
            for r in event_probe(subject, night, score)
        )
    result = pd.concat(features, ignore_index=True)
    probes = pd.DataFrame(records)
    validation = summarize(probes) if len(probes) else []
    spontaneous = [r for r in validation if r["annotation"] == "arousal_spontaneous"]
    qualifying_folds = sum(
        r["fold"] is not None and r["patient_matched_auc"] >= 0.55 for r in spontaneous
    )
    pooled_positive = any(
        r["fold"] is None and r["patient_matched_auc"] > 0.50 for r in spontaneous
    )
    summary = {
        "patients": int(result.subject_id.nunique()),
        "epochs": len(result),
        "finite_share": {col: float(result[col].notna().mean()) for col in COLUMNS},
        "current_interruption_rate": float(
            result.interruption_strength.ge(THRESHOLD).mean()
        ),
        "annotations": validation,
        "qualifying_folds": qualifying_folds,
        "qualified": not smoke_only and qualifying_folds >= 4 and pooled_positive,
        "elapsed_seconds": perf_counter() - started,
    }
    if smoke_only:
        atomic_json(OUT / "interruption_smoke.json", summary)
    else:
        save_frame(OUT / "interruptions.parquet", result[[*KEYS, *COLUMNS]])
        save_frame(OUT / "interruption_annotation_matches.parquet", probes)
        atomic_json(OUT / "interruption_summary.json", summary)
    log(
        f"Interruption audit scored {summary['patients']} recordings in {summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    with local_training_slot():
        main(parser.parse_args().smoke)
