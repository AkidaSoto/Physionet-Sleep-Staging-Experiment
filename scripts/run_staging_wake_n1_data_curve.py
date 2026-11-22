"""Compare 128 fitting patients with the archived full expanded-model cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import run_staging_complementary_model as expanded
import run_staging_gated_temporal as gated
import torch
from build_metric_intervals import (
    cv_fold_draws,
    cv_scores,
    interval,
    patient_confusions,
)
from run_staging_feature_tuning import BASE, save_frame
from run_waveform_development import (
    BUDGET_SECONDS,
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
    RAW,
    SEED,
    STAGES,
    atomic_json,
    development_frame,
    fold_frames,
    local_training_slot,
    log,
    pin_manifest,
    write_predictions,
)
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from physionet_sleep.experiments.waveform_clock import active_clock
from physionet_sleep.io.cinc2018 import resolve_cinc2018_record
from physionet_sleep.labels.cinc2018 import subject_summary

OUT = EXPLORATION / "cinc-wake-n1-data" / "data_curve"
REFERENCE = expanded.OUT / "expanded_predictions.parquet"
PATIENTS = 128
N1 = STAGES.index("N1")
WAKE = STAGES.index("Wake")
APNEA_BANDS = {-1: "unavailable", 0: "<5", 1: "5-<15", 2: "15-<30", 3: ">=30"}


def apnea_bands(subjects: list[str]) -> pd.Series:
    """Use only annotations from the declared development cohort."""
    rows = {}
    for subject in subjects:
        paths = resolve_cinc2018_record(RAW, subject)
        ahi = subject_summary(paths)["ahi"]
        rows[subject] = (
            int(np.searchsorted([5, 15, 30], ahi, side="right"))
            if np.isfinite(ahi)
            else -1
        )
    return pd.Series(rows, name="apnea_band")


def strata(train: pd.DataFrame, bands: pd.Series) -> pd.DataFrame:
    fraction = train[LABEL].eq(N1).groupby(train.subject_id).mean()
    profiles = fraction.rename("n1_share").sort_values(kind="stable").to_frame()
    profiles["n1_quartile"] = np.arange(len(profiles)) * 4 // len(profiles)
    return profiles.join(bands, how="left", validate="one_to_one")


def stratified_choice(
    profiles: pd.DataFrame, count: int, rng: np.random.Generator
) -> list[str]:
    """Allocate the total by largest remainder, then sample without replacement."""
    if not 0 <= count <= len(profiles):
        raise ValueError("Cannot draw the requested number of unique fitting patients")
    groups = [
        values.index.to_numpy()
        for _, values in profiles.groupby(["n1_quartile", "apnea_band"], dropna=False)
    ]
    target = np.array([len(group) for group in groups]) * (count / len(profiles))
    actual = np.floor(target).astype(int)
    order = np.argsort(actual - target, kind="stable")
    actual[order[: count - actual.sum()]] += 1
    selected = [
        subject
        for group, n in zip(groups, actual)
        for subject in rng.choice(group, n, replace=False)
    ]
    return sorted(selected)


def training_design(train: pd.DataFrame, bands: pd.Series, fold: int) -> dict:
    # Allocate patient slots once, then let the archived fitter shuffle the list
    # and checkpoint the RNG exactly as for its full-cohort fit.
    profiles = strata(train, bands)
    rng = np.random.default_rng(SEED + fold)
    selected = stratified_choice(profiles, PATIENTS, rng)
    additional = stratified_choice(
        profiles.loc[selected], len(profiles) - PATIENTS, rng
    )
    slots = sorted(selected + additional)
    counts = {
        name: {
            band: int(group.apnea_band.eq(code).sum())
            for code, band in APNEA_BANDS.items()
        }
        for name, group in (("full", profiles), ("selected", profiles.loc[selected]))
    }
    return {
        "fold": fold,
        "selected_patients": selected,
        "optimizer_patient_slots": slots,
        "apnea_bands": counts,
        "full_n1_share": float(profiles.n1_share.mean()),
        "selected_n1_share": float(profiles.loc[selected].n1_share.mean()),
        "effective_n1_share": float(profiles.loc[slots].n1_share.mean()),
    }


def prepare(
    frame: pd.DataFrame,
    selected: set[str],
    other: set[str],
    markers: dict[str, np.ndarray],
) -> tuple[dict[str, gated.sequence.Recording], dict]:
    sample = frame[frame.subject_id.isin(selected | other)]
    recordings = gated.sequence.prepare_recordings(sample, selected)
    scaling = expanded.add_markers(recordings, markers, selected)
    return recordings, scaling


def smoke(frame: pd.DataFrame, design: dict, inner_validation: pd.DataFrame) -> None:
    selected = set(design["selected_patients"][:8])
    val_ids = sorted(inner_validation.subject_id.unique())[:4]
    slots = [
        subject for subject in design["optimizer_patient_slots"] if subject in selected
    ]
    if len(slots) <= len(selected):
        raise ValueError("The training-only smoke requires repeated optimizer slots")
    recordings, _ = prepare(
        frame, selected, set(val_ids), expanded.load_markers(selected | set(val_ids))
    )
    options = {"seed": SEED, "deadline": float("inf"), "max_passes": 2}
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT, prefix="smoke-") as temporary:
        scratch = Path(temporary)
        paused, state = expanded.fit(
            recordings, slots, val_ids, scratch / "resume", **options, stop_after=1
        )
        if paused is not None or state["next_pass"] != 1:
            raise ValueError("The smoke fit did not stop after its first pass")
        resumed, _ = expanded.fit(
            recordings, slots, val_ids, scratch / "resume", **options
        )
        control, _ = expanded.fit(
            recordings, slots, val_ids, scratch / "uninterrupted", **options
        )
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(
                value, control.state_dict()[name], rtol=0, atol=0
            )
        sample = inner_validation[inner_validation.subject_id.isin(val_ids)].iloc[:25]
        p = gated.sequence.score_fold(resumed, recordings, sample)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    atomic_json(
        OUT / "smoke.json",
        {
            "passed": True,
            "exact_resume": True,
            "scored_outer_patients": 0,
            "train_unique_patients": len(selected),
            "optimizer_patient_slots": len(slots),
        },
    )
    log("Repeated-slot training and exact resume verified on inner patients")


def wake_n1_metrics(y: np.ndarray, p: np.ndarray, hard: np.ndarray) -> dict:
    pred = p.argmax(1)
    classes = [WAKE, N1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        pred,
        labels=classes,
        zero_division=0,
    )
    pair = np.isin(y, classes)
    margin = np.log(p[:, WAKE].clip(1e-12)) - np.log(p[:, N1].clip(1e-12))
    return {
        "stages": {
            STAGES[label]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "epochs": int(support[i]),
            }
            for i, label in enumerate(classes)
        },
        "Wake_as_N1": int(((y == WAKE) & (pred == N1)).sum()),
        "N1_as_Wake": int(((y == N1) & (pred == WAKE)).sum()),
        "auc_all": float(roc_auc_score(y[pair] == WAKE, margin[pair])),
        "auc_hard": float(roc_auc_score(y[hard] == WAKE, margin[hard])),
        "hard_epochs": int(hard.sum()),
    }


def sustained_directions(aligned: pd.DataFrame, name: str) -> dict:
    ordered = aligned.sort_values(KEYS)
    y, pred = ordered[LABEL], ordered[name]
    return {
        f"{STAGES[true]}_as_{STAGES[wrong]}": expanded.sustained_runs(
            ordered.assign(pair_prediction=pred.where(y.eq(true) & pred.eq(wrong), y)),
            "pair_prediction",
        )
        for true, wrong in ((WAKE, N1), (N1, WAKE))
    }


def report(base: pd.DataFrame, full: pd.DataFrame, fold_count: int) -> None:
    small = gated.align(
        base,
        pd.concat(
            [
                pd.read_parquet(OUT / f"fold-{fold}/predictions.parquet")
                for fold in range(fold_count)
            ]
        ),
    )
    p = {
        "full": full[PROBABILITIES].to_numpy(),
        "train_128": small[PROBABILITIES].to_numpy(),
    }
    aligned = base[gated.KEY_FOLD].copy()
    for name, probabilities in p.items():
        aligned[name] = probabilities.argmax(1)
    y = aligned[LABEL].to_numpy()
    fold_rows = [np.flatnonzero(aligned.fold.eq(fold)) for fold in range(fold_count)]
    comparison = gated.compare(aligned, "full", "train_128", fold_rows)
    cm_full, patients = patient_confusions(
        aligned.subject_id.to_numpy(), y, aligned.full.to_numpy()
    )
    cm_small, _ = patient_confusions(
        aligned.subject_id.to_numpy(), y, aligned.train_128.to_numpy()
    )
    membership = aligned.groupby("subject_id").fold.first().loc[patients].to_numpy()
    patient_groups = [np.flatnonzero(membership == fold) for fold in range(fold_count)]
    draws = cv_fold_draws(patient_groups, 2000, np.random.default_rng(SEED))
    comparison["kappa_delta"] = (
        comparison["candidate"]["kappa"] - comparison["reference"]["kappa"]
    )
    comparison["paired_kappa_ci"] = interval(
        cv_scores(cm_full, patient_groups, draws)["kappa"]
        - cv_scores(cm_small, patient_groups, draws)["kappa"]
    )
    comparison["passes"] = bool(
        comparison["wins"] >= 4
        and (comparison["delta"] >= 0.003 or comparison["stage_deltas"]["N1"] >= 0.010)
        and comparison["stage_deltas"]["Wake"] >= -0.010
    )
    compact = base[PROBABILITIES].to_numpy()
    order = np.argsort(-compact, axis=1, kind="stable")[:, :2]
    ranked = np.take_along_axis(compact, order, axis=1)
    hard = (
        np.isin(y, [WAKE, N1])
        & (np.sort(order, axis=1) == [WAKE, N1]).all(1)
        & (ranked[:, 0] <= 2 * ranked[:, 1])
    )
    per_fold = []
    for fold, rows in enumerate(fold_rows):
        fold_result = {"fold": fold}
        for name, probabilities in p.items():
            pred = aligned[name].to_numpy()[rows]
            fold_result[name] = {
                "macro_f1": float(f1_score(y[rows], pred, average="macro")),
                "kappa": float(cohen_kappa_score(y[rows], pred)),
                **wake_n1_metrics(y[rows], probabilities[rows], hard[rows]),
            }
        fold_result["train_128_training"] = json.loads(
            (OUT / f"fold-{fold}/runtime.json").read_text()
        )
        full_runtime = json.loads(
            (expanded.MODEL / f"fold-{fold}/runtime.json").read_text()
        )
        fold_result["full_training"] = optimizer_budget(
            full_runtime,
            full_runtime["fit_patients"],
            full_runtime["fit_patients"],
        )
        per_fold.append(fold_result)
    atomic_json(
        OUT / "summary.json",
        {
            "contrasts": {"full_vs_train_128": comparison},
            "wake_n1": {
                name: wake_n1_metrics(y, probabilities, hard)
                for name, probabilities in p.items()
            },
            "sustained_wake_n1": {
                name: sustained_directions(aligned, name) for name in p
            },
            "folds": per_fold,
            "n1_f1_fold_wins": sum(
                r["full"]["stages"]["N1"]["f1"] > r["train_128"]["stages"]["N1"]["f1"]
                for r in per_fold
            ),
        },
    )
    save_frame(OUT / "predictions.parquet", small)
    log(
        f"128 patients {comparison['reference']['macro_f1']:.4f}, "
        f"full {comparison['candidate']['macro_f1']:.4f}, "
        f"paired delta {comparison['delta']:+.4f}, passed rule {comparison['passes']}"
    )


def optimizer_budget(runtime: dict, unique: int, slots: int) -> dict:
    per_pass = ceil(slots / gated.sequence.BATCH_NIGHTS)
    return {
        "unique_patients": unique,
        "patient_slots_per_pass": slots,
        "passes": runtime["passes"],
        "best_pass": runtime["best_pass"],
        "optimizer_updates": per_pass * runtime["passes"],
        "updates_at_best": per_pass * runtime["best_pass"],
        "best_validation_macro_f1": runtime["best_validation_macro_f1"],
    }


def execute(smoke_only: bool) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT.parent / "FINDINGS.md").exists():
        raise FileNotFoundError("Declare the subject-budget study before fitting")
    frame, splits = development_frame()
    partitions = [fold_frames(frame, split) for split in splits]
    bands = apnea_bands(sorted(frame.subject_id.unique()))
    designs = [
        training_design(train, bands, fold)
        for fold, (train, _, _) in enumerate(partitions)
    ]
    if smoke_only:
        smoke(frame, designs[0], partitions[0][1])
        return 0
    if not json.loads((OUT / "smoke.json").read_text())["passed"]:
        raise ValueError("Verify the training-only smoke first")
    base = pd.read_parquet(BASE)
    expected = pd.concat(
        [
            test[KEYS + [LABEL]].assign(fold=f)
            for f, (_, _, test) in enumerate(partitions)
        ]
    )
    gated.align(base, expected)
    full = gated.align(base, pd.read_parquet(REFERENCE))
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
            "designs": designs,
            "seed": SEED,
            "torch": torch.__version__,
            "sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
        },
    )
    ledger = OUT / "active_budget.json"
    used = json.loads(ledger.read_text())["seconds"] if ledger.exists() else 0.0
    started = active_clock()
    deadline = started + BUDGET_SECONDS - used
    try:
        markers = expanded.load_markers(set(frame.subject_id))
        for design, (_, val, test) in zip(designs, partitions):
            fold = design["fold"]
            path = OUT / f"fold-{fold}"
            if (path / "predictions.parquet").exists():
                continue
            if (OUT / "PAUSE").exists() or active_clock() >= deadline:
                return 75
            selected = set(design["selected_patients"])
            slots = design["optimizer_patient_slots"]
            val_ids = sorted(val.subject_id.unique())
            evaluation = set(val_ids) | set(test.subject_id)
            if (
                selected & evaluation
                or len(selected) != PATIENTS
                or set(slots) != selected
            ):
                raise ValueError(
                    "The subject-budget partition overlaps evaluation patients"
                )
            recordings, scaling = prepare(frame, selected, evaluation, markers)
            model, runtime = expanded.fit(
                recordings, slots, val_ids, path, SEED + fold, deadline
            )
            if model is None:
                log("Paused at a saved training pass")
                return 75
            probabilities = gated.sequence.score_fold(model, recordings, test)
            atomic_json(
                path / "runtime.json",
                {
                    **optimizer_budget(runtime, len(selected), len(slots)),
                    "marker_scaling": scaling,
                },
            )
            write_predictions(path / "predictions.parquet", test, probabilities, fold)
            log(f"Fold {fold + 1}/{len(partitions)} 128-patient fit scored")
            del recordings
        report(base, full, len(partitions))
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - started})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Verify resumability on inner patients"
    )
    with local_training_slot():
        raise SystemExit(execute(parser.parse_args().smoke))
