"""Evaluate the registered 21-measurement ablation of the expanded gated model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import run_staging_complementary_model as expanded
import run_staging_gated_temporal as gated

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score

from build_metric_intervals import (
    cv_fold_draws,
    cv_scores,
    interval,
    patient_confusions,
)
from physionet_sleep.experiments.waveform_clock import active_clock
from run_staging_feature_tuning import BASE, error_structure, save_frame
from run_waveform_development import (
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    SEED,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
    pin_manifest,
    write_predictions,
)


OUT = EXPLORATION / "cinc-stage-pair-revisit" / "reduction"
REFERENCE = expanded.OUT / "expanded_predictions.parquet"
REMOVED_ORIGINAL = "topography_features.frontal_slow_wave_fraction"
REMOVED_MARKER = "slow_wave_75_bilateral_overlap"
INPUT_SIGNALS = [*gated.sequence.TRACKS, *gated.sequence.FEATURES]
FULL_DIMENSION = expanded.ORIGINAL_WIDTH + len(expanded.FEATURES)
NIGHT_STEPS = expanded.ORIGINAL_WIDTH // len(INPUT_SIGNALS)
REMOVED_COLUMNS = [
    *(
        INPUT_SIGNALS.index(REMOVED_ORIGINAL) + step * len(INPUT_SIGNALS)
        for step in range(NIGHT_STEPS)
    ),
    expanded.ORIGINAL_WIDTH + expanded.FEATURES.index(REMOVED_MARKER),
]
RETAINED_COLUMNS = [
    column for column in range(FULL_DIMENSION) if column not in REMOVED_COLUMNS
]


def remove_measurements(recordings: dict[str, gated.sequence.Recording]) -> None:
    """Neutralize the two measurements in all epochs after the original scaling."""
    for recording in recordings.values():
        if recording.features.shape[1] != FULL_DIMENSION:
            raise ValueError("Expanded recording has an unexpected measurement layout")
        recording.features[:, REMOVED_COLUMNS] = 0.0


def prepare(
    frame: pd.DataFrame,
    train: set[str],
    markers: dict[str, np.ndarray],
) -> tuple[dict[str, gated.sequence.Recording], dict]:
    recordings = gated.sequence.prepare_recordings(frame, train)
    scaling = expanded.add_markers(recordings, markers, train)
    remove_measurements(recordings)
    return recordings, scaling


def smoke(frame: pd.DataFrame, split) -> None:
    train, val, _ = fold_frames(frame, split)
    train_ids = sorted(train.subject_id.unique())[:8]
    val_ids = sorted(val.subject_id.unique())[:4]
    subjects = set(train_ids + val_ids)
    sample = frame[frame.subject_id.isin(subjects)]
    recordings = gated.sequence.prepare_recordings(sample, set(train_ids))
    expanded.add_markers(recordings, expanded.load_markers(subjects), set(train_ids))
    before = {
        name: recording.features.clone() for name, recording in recordings.items()
    }
    remove_measurements(recordings)
    for name, recording in recordings.items():
        torch.testing.assert_close(
            recording.features[:, RETAINED_COLUMNS],
            before[name][:, RETAINED_COLUMNS],
            rtol=0,
            atol=0,
        )
        if torch.count_nonzero(recording.features[:, REMOVED_COLUMNS]):
            raise ValueError("An ablated feature still carries a signal")

    options = {"seed": SEED, "deadline": float("inf"), "max_passes": 2}
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT, prefix="smoke-") as temporary:
        scratch = Path(temporary)
        model, state = expanded.fit(
            recordings,
            train_ids,
            val_ids,
            scratch / "resume",
            **options,
            stop_after=1,
        )
        if model is not None or state["next_pass"] != 1:
            raise ValueError("The smoke fit did not stop after its first pass")
        resumed, _ = expanded.fit(
            recordings, train_ids, val_ids, scratch / "resume", **options
        )
        control, _ = expanded.fit(
            recordings, train_ids, val_ids, scratch / "uninterrupted", **options
        )
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(
                value, control.state_dict()[name], rtol=0, atol=0
            )
        sample_val = val[val.subject_id.isin(val_ids)].iloc[:25]
        probabilities = gated.sequence.score_fold(resumed, recordings, sample_val)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    atomic_json(
        OUT / "smoke.json",
        {
            "passed": True,
            "exact_resume": True,
            "retained_values_exact": True,
            "scored_outer_patients": 0,
        },
    )
    log("Both measurement masks and the resumed fit were verified on inner patients")


def report(base: pd.DataFrame, fold_count: int) -> None:
    scores = pd.concat(
        [
            pd.read_parquet(OUT / f"fold-{f}/predictions.parquet")
            for f in range(fold_count)
        ]
    )
    reduced = gated.align(base, scores)
    reference = gated.align(base, pd.read_parquet(REFERENCE))
    probabilities = {
        "expanded": reference[PROBABILITIES].to_numpy(),
        "reduced": reduced[PROBABILITIES].to_numpy(),
    }
    aligned = base[gated.KEY_FOLD].copy()
    for name, p in probabilities.items():
        aligned[name] = p.argmax(1)
    y = aligned[LABEL].to_numpy()
    fold_rows = [np.flatnonzero(aligned.fold.eq(f)) for f in range(fold_count)]
    matched = gated.compare(aligned, "reduced", "expanded", fold_rows)
    per_fold = []
    for fold, rows in enumerate(fold_rows):
        per_fold.append(
            {
                "fold": fold,
                **{
                    name: {
                        "macro_f1": float(
                            f1_score(
                                y[rows], aligned[name].to_numpy()[rows], average="macro"
                            )
                        ),
                        "kappa": float(
                            cohen_kappa_score(y[rows], aligned[name].to_numpy()[rows])
                        ),
                    }
                    for name in probabilities
                },
            }
        )
    reduced_cm, patients = patient_confusions(
        aligned.subject_id.to_numpy(), y, aligned.reduced.to_numpy()
    )
    expanded_cm, _ = patient_confusions(
        aligned.subject_id.to_numpy(), y, aligned.expanded.to_numpy()
    )
    membership = aligned.groupby("subject_id").fold.first().loc[patients].to_numpy()
    groups = [np.flatnonzero(membership == f) for f in range(fold_count)]
    draws = cv_fold_draws(groups, 2000, np.random.default_rng(SEED))
    matched["kappa_delta"] = (
        matched["candidate"]["kappa"] - matched["reference"]["kappa"]
    )
    matched["paired_kappa_ci"] = interval(
        cv_scores(reduced_cm, groups, draws)["kappa"]
        - cv_scores(expanded_cm, groups, draws)["kappa"]
    )
    compact = base[PROBABILITIES].to_numpy()
    rank = np.argsort(-compact, axis=1, kind="stable")[:, :2]
    ranked = np.take_along_axis(compact, rank, axis=1)
    close = ranked[:, 0] <= 2 * ranked[:, 1]
    hard = {
        (i, j): close & (np.sort(rank, axis=1) == [i, j]).all(1)
        for i, j in gated.TARGET_PAIRS
    }
    metrics = {
        name: gated.probability_metrics(y, p, hard) for name, p in probabilities.items()
    }
    n1_delta = (
        metrics["reduced"]["stages"]["N1"]["recall"]
        - metrics["expanded"]["stages"]["N1"]["recall"]
    )
    matched["n1_recall_delta"] = n1_delta
    matched["guards"] = {
        "mean_macro_f1": matched["delta"] >= -0.005,
        "every_fold_macro_f1": min(matched["fold_deltas"]) >= -0.005,
        "pooled_stage_f1": min(matched["stage_deltas"].values()) >= -0.010,
        "mean_kappa": matched["kappa_delta"] >= -0.005,
        "n1_recall": n1_delta >= -0.010,
    }
    matched["passes"] = all(matched["guards"].values())
    ordered = aligned.sort_values(KEYS)
    structure = {name: error_structure(ordered, name) for name in probabilities}
    runs = {name: expanded.sustained_runs(ordered, name) for name in probabilities}
    changes = {
        "reduced_wrong_and_expanded_correct": expanded.sustained_runs(
            ordered, "reduced", other_correct="expanded"
        ),
        "expanded_wrong_and_reduced_correct": expanded.sustained_runs(
            ordered, "expanded", other_correct="reduced"
        ),
    }
    paired = expanded.paired_patient_scores(
        aligned.rename(columns={"expanded": "gated", "reduced": "expanded"})
    )
    paired = paired.rename(
        columns={"gated_f1": "expanded_f1", "expanded_f1": "reduced_f1"}
    )
    save_frame(OUT / "paired_patient_scores.parquet", paired)
    save_frame(OUT / "predictions.parquet", reduced)
    atomic_json(
        OUT / "summary.json",
        {
            "measurements": {
                "expanded": 23,
                "reduced": 21,
                "removed": [REMOVED_ORIGINAL, REMOVED_MARKER],
            },
            "method": "Set the removed normalized inputs to zero in fit, validation and evaluation while retaining the original model width and extraction.",
            "contrasts": {"reduced_vs_expanded": matched},
            "folds": per_fold,
            "probabilities": metrics,
            "errors": structure,
            "sustained_errors": runs,
            "changed_sustained_errors": changes,
        },
    )
    log(
        f"Reduced {matched['candidate']['macro_f1']:.4f} macro-F1, "
        f"{matched['delta']:+.4f} vs expanded, passed guards: {matched['passes']}"
    )


def execute(smoke_only: bool) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT.parent / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the reduction before fitting")
    frame, splits = development_frame()
    if smoke_only:
        smoke(frame, splits[0])
        return 0
    if not json.loads((OUT / "smoke.json").read_text())["passed"]:
        raise ValueError("Verify the training-only reduction and resume first")
    base = pd.read_parquet(BASE)
    partitions = [fold_frames(frame, split) for split in splits]
    expected = pd.concat(
        [
            test[KEYS + [LABEL]].assign(fold=f)
            for f, (_, _, test) in enumerate(partitions)
        ]
    )
    gated.align(base, expected)
    gated.align(base, pd.read_parquet(REFERENCE))
    sources = [
        Path(__file__),
        Path(expanded.__file__),
        Path(gated.__file__),
        Path(gated.sequence.__file__),
        Path(gated.gated_staging.__file__),
        expanded.FEATURE_MANIFEST,
        expanded.MODEL / "manifest.json",
        BASE,
        REFERENCE,
    ]
    pin_manifest(
        OUT / "manifest.json",
        {
            "removed_columns": REMOVED_COLUMNS,
            "removed_measurements": [REMOVED_ORIGINAL, REMOVED_MARKER],
            "seed": SEED,
            "torch": torch.__version__,
            "sources": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
        },
    )
    ledger = OUT / "active_budget.json"
    used = json.loads(ledger.read_text())["seconds"] if ledger.exists() else 0.0
    started = active_clock()
    deadline = started + 86400 - used
    try:
        panels = expanded.load_markers(set(frame.subject_id))
        for fold, (train, val, test) in enumerate(partitions):
            path = OUT / f"fold-{fold}"
            if (path / "predictions.parquet").exists():
                continue
            if (OUT / "PAUSE").exists() or active_clock() >= deadline:
                return 75
            train_ids, val_ids = (
                sorted(train.subject_id.unique()),
                sorted(val.subject_id.unique()),
            )
            recordings, scaling = prepare(frame, set(train_ids), panels)
            model, timing = expanded.fit(
                recordings, train_ids, val_ids, path, SEED + fold, deadline
            )
            if model is None:
                log("Paused at a saved training pass")
                return 75
            probabilities = gated.sequence.score_fold(model, recordings, test)
            atomic_json(path / "runtime.json", {**timing, "marker_scaling": scaling})
            write_predictions(path / "predictions.parquet", test, probabilities, fold)
            log(f"Fold {fold + 1}/{len(partitions)} saved")
            del recordings
        report(base, len(partitions))
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - started})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Verify masks and resume on inner patients"
    )
    with local_training_slot():
        raise SystemExit(execute(parser.parse_args().smoke))
