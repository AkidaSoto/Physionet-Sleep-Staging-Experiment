"""Run the fixed-update, five-fold Boston training expansion (#73)."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import run_staging_complementary_model as expanded
import run_staging_gated_temporal as gated
from run_staging_wake_n1_data_curve import (
    APNEA_BANDS, N1, WAKE, apnea_bands, optimizer_budget, sustained_directions,
    wake_n1_metrics,
)

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
import torch

from audit_staging_wake_n1 import annotation_phases, model_components
from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from build_metric_intervals import scores as confusion_scores
from physionet_sleep.experiments.gated_staging import stage_loss
from physionet_sleep.experiments.modern_tabular import _dense_cached_night
from physionet_sleep.experiments.waveform_clock import active_clock
from physionet_sleep.io.cinc2018 import discover_cinc2018_records
from run_staging_feature_tuning import BASE, save_frame
from run_waveform_development import (
    BASELINE, BUDGET_SECONDS, CACHE, EXPLORATION, KEYS, LABEL, PROBABILITIES, RAW,
    SEED, STAGES, atomic_json, development_frame, fold_frames, local_training_slot,
    log, pin_manifest, write_predictions,
)


OUT = EXPLORATION / "cinc-all-additional-training"
SUBJECTS = OUT / "subjects.txt"
ADDED_EPOCHS = OUT / "additional_epochs.parquet"
ADDED_WINDOW_CACHE = OUT / "window_cache"
REFERENCE = expanded.OUT / "expanded_predictions.parquet"
SEQUENCE = gated.sequence
SIGNALS = SEQUENCE.TRACKS + SEQUENCE.FEATURES
CHECKPOINT_SECONDS = 300


class Paused(Exception):
    """The pause file or active time budget stopped the current session."""


def training_cohort(development: set[str]) -> tuple[pd.DataFrame, set[str]]:
    selected = SUBJECTS.read_text(encoding="utf-8").split()
    added = set(selected)
    reserved = set((RAW / "selected_records_holdout.txt").read_text().split())
    ledger = set(pd.read_csv(BASELINE / "holdout_subjects.csv", usecols=["record_id"]).record_id)
    if len(development) != 320 or len(reserved) != 200 or ledger != reserved:
        raise ValueError("Boston development or independent holdout membership changed")
    if development & reserved or len(selected) != 474 or len(added) != 474:
        raise ValueError("Expected 474 distinct added patients and disjoint Boston reserves")
    if added & (development | reserved):
        raise ValueError("An added patient appears in development or independent holdout")
    labeled = {
        record.record_id
        for record in discover_cinc2018_records(RAW, require_signals=False)
        if record.annotation_path is not None
    }
    if added != labeled - development - reserved:
        raise ValueError("The expansion list differs from the full unused labeled Boston pool")
    epochs = pd.read_parquet(ADDED_EPOCHS, columns=[*KEYS, LABEL])
    if set(epochs.subject_id) != added or epochs.duplicated(KEYS).any():
        raise ValueError("Added epoch keys differ from the declared training-only cohort")
    if (not pd.api.types.is_integer_dtype(epochs.epoch_index)
            or epochs.epoch_index.isna().any() or epochs.epoch_index.lt(0).any()
            or not epochs[LABEL].isin(range(len(STAGES))).all()):
        raise ValueError("Added training input has invalid scored epochs")
    return epochs, added


def marker_panel(subject: str) -> np.ndarray:
    panel = pd.read_parquet(OUT / "feature_cache" / f"{subject}.parquet")
    if (panel.empty or not panel.columns.is_unique
            or set(panel.columns) != set(expanded.FEATURES)
            or panel.index.name != "epoch_index" or not panel.index.equals(pd.RangeIndex(len(panel)))):
        raise ValueError(f"{subject}: added marker panel differs from the #61 feature contract")
    values = panel.loc[:, list(expanded.FEATURES)].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def prepare_recordings(
    frame: pd.DataFrame, fit: set[str], check, *, added: set[str] | None = None,
) -> tuple[dict, dict]:
    """Keep one dense 1 Hz night in memory while estimating the archived moments."""
    added = added or set()

    def dense(subject: str) -> np.ndarray:
        cache = ADDED_WINDOW_CACHE if subject in added else CACHE
        return _dense_cached_night(cache, subject, SEQUENCE.TRACKS, SEQUENCE.FEATURES)

    count = np.zeros(len(SIGNALS), dtype=np.int64)
    total = np.zeros(len(SIGNALS), dtype=np.float64)
    squared = np.zeros(len(SIGNALS), dtype=np.float64)
    fit_seconds = 0
    for subject in sorted(fit):
        check()
        data = dense(subject)
        fit_seconds += len(data)
        count += np.isfinite(data).sum(axis=0)
        total += np.nansum(data, axis=0, dtype=np.float64)
        squared += np.nansum(np.square(data, dtype=np.float64), axis=0)
    if not np.all(count):
        raise ValueError("Every signal must have observed values in the fitting patients")
    mean = total / count
    scale = np.sqrt(np.maximum(0.0, squared / count - mean**2)) + 1e-6
    encoder = SEQUENCE.WindowEncoder(SIGNALS, context=0, mean=mean, sd=scale)
    pad = encoder.steps_per_epoch
    recordings = {}
    for subject, night in frame.groupby("subject_id", sort=False):
        check()
        data = dense(subject)
        steps = encoder._steps(data)[pad:-pad]
        steps = np.pad(steps, ((0, -len(steps) % pad), (0, 0)))
        values = steps.reshape(-1, expanded.ORIGINAL_WIDTH)
        labels = np.full(len(values), -1, dtype=np.int64)
        at = night.epoch_index.to_numpy()
        if at.max() >= len(values):
            raise ValueError(f"{subject}: scored epoch extends beyond the feature cache")
        labels[at] = night[LABEL]
        recordings[subject] = SEQUENCE.Recording(torch.from_numpy(values), torch.from_numpy(labels))
    return recordings, {
        "signals": list(SIGNALS), "total_fit_seconds": fit_seconds,
        "finite_fit_seconds": count.tolist(),
        "fit_mean": mean.tolist(), "fit_scale": scale.tolist(),
    }


def slots_for_pass(cycle: np.ndarray, slots: int, step: int) -> np.ndarray:
    return cycle[np.arange(step * slots, (step + 1) * slots) % len(cycle)]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fit_expanded(
    recordings: dict, fit: set[str], validation: list[str], slots: int, passes: int,
    seed: int, directory: Path, pause_path: Path, deadline: float,
    account=lambda: None, *, added: set[str] | None = None,
    smoke_pause_updates: int | None = None,
) -> tuple[torch.nn.Module | None, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    torch.manual_seed(seed)
    model = expanded.expanded_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, SEQUENCE.MAX_PASSES, eta_min=0.0003,
    )
    labels = torch.cat([recordings[s].scored_labels for s in sorted(fit)])
    counts = torch.bincount(labels[labels >= 0], minlength=len(STAGES)).float()
    if (counts == 0).any():
        raise ValueError("Every stage must occur in the augmented fitting pool")
    weights = counts.pow(-0.70)
    cycle = np.random.default_rng(seed).permutation(sorted(fit))
    first_complete_pass = ceil(len(fit) / slots)
    if not (1 <= slots <= len(fit) and first_complete_pass <= passes <= SEQUENCE.MAX_PASSES):
        raise ValueError("The original update budget must cover the entire fitting pool")
    state = {"next_pass": 0, "next_batch": 0, "updates": 0, "best": -1.0,
             "best_pass": 0, "history": [], "pass_seconds": 0.0}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])

    def save() -> float:
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), rng=torch.get_rng_state())
        SEQUENCE.checkpoint(checkpoint, state)
        account()
        return active_clock()

    last_checkpoint = active_clock()
    for step in range(state["next_pass"], passes):
        if pause_path.exists() or active_clock() >= deadline:
            return None, state
        selected = slots_for_pass(cycle, slots, step)
        batches = SEQUENCE.batches(selected, np.random.default_rng(seed + step))
        began = active_clock()
        model.train()
        for i, batch in enumerate(batches):
            if i < state["next_batch"]:
                continue
            if pause_path.exists() or active_clock() >= deadline:
                state["pass_seconds"] += active_clock() - began
                save()
                return None, state
            x, y, recorded = SEQUENCE.collate(recordings, batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, recorded, return_components=True)
            loss = (stage_loss(output.logits, y, weights)
                    + gated.LOCAL_LOSS * stage_loss(output.local, y, weights))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss in augmented #61 fit")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state["updates"] += 1
            state["next_batch"] = i + 1
            now = active_clock()
            smoke_stop = state["updates"] == smoke_pause_updates
            stopping = pause_path.exists() or now >= deadline or smoke_stop
            if stopping or now - last_checkpoint >= CHECKPOINT_SECONDS:
                state["pass_seconds"] += now - began
                last_checkpoint = began = save()
            if stopping:
                return None, state

        scheduler.step()
        score = SEQUENCE.validation_score(model, recordings, validation)
        # An eligible checkpoint must have trained on every augmented patient.
        if step + 1 >= first_complete_pass and score > state["best"]:
            state.update(best=score, best_pass=step + 1, best_weights={
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            })
        elapsed = state["pass_seconds"] + active_clock() - began
        state["history"].append({"pass": step + 1, "val_macro_f1": score,
                                 "updates": state["updates"], "seconds": elapsed})
        state.update(next_pass=step + 1, next_batch=0, pass_seconds=0.0)
        last_checkpoint = save()
        log(f"{directory.name} pass {step + 1}/{passes}: inner macro-F1 {score:.4f}, {elapsed:.1f}s")

    per_pass = ceil(slots / SEQUENCE.BATCH_NIGHTS)
    if state["updates"] != passes * per_pass:
        raise ValueError("Actual optimizer update count differs from the archived budget")
    model.load_state_dict(state["best_weights"])
    added = added or set()

    def visits_through(n_passes: int) -> dict:
        visits = Counter(subject for step in range(n_passes) for subject in slots_for_pass(cycle, slots, step))
        return {
            "patient_slots": n_passes * slots, "unique_patients": len(visits),
            "original_patient_slots": sum(count for subject, count in visits.items() if subject not in added),
            "added_patient_slots": sum(count for subject, count in visits.items() if subject in added),
            "visits_frequency": dict(sorted(Counter(visits.values()).items())),
        }

    return model, {
        "unique_patients": len(fit), "validation_patients": len(validation),
        "patient_slots_per_pass": slots, "patients_seen_by_pass": first_complete_pass,
        "visits_through_scored_checkpoint": visits_through(state["best_pass"]),
        "visits_through_final_pass": visits_through(passes),
        "passes": passes, "optimizer_updates": state["updates"],
        "best_pass": state["best_pass"], "updates_at_best": state["best_pass"] * per_pass,
        "best_validation_macro_f1": state["best"], "stage_counts": counts.int().tolist(),
        "class_weights": weights.tolist(), "history": state["history"],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def smoke(partition) -> None:
    started = active_clock()
    train, validation, _ = partition
    fit = set(sorted(train.subject_id.unique())[:12])
    val = sorted(validation.subject_id.unique())[:2]
    frame = pd.concat([train[train.subject_id.isin(fit)], validation[validation.subject_id.isin(val)]])
    recordings, _ = prepare_recordings(frame, fit, lambda: None)
    control = SEQUENCE.prepare_recordings(frame, fit)
    for subject, recording in recordings.items():
        torch.testing.assert_close(recording.features, control[subject].features, rtol=0, atol=0)
        torch.testing.assert_close(recording.scored_labels, control[subject].scored_labels, rtol=0, atol=0)
    expanded.add_markers(recordings, expanded.load_markers(fit | set(val)), fit)
    with TemporaryDirectory(dir=OUT, prefix="smoke-") as directory:
        path = Path(directory)
        # A one-batch checkpoint ends in the middle of a two-batch pass.
        args = (recordings, fit, val, 9, 2, SEED)
        pause = path / "PAUSE"
        model, state = fit_expanded(*args, path / "resume", pause, float("inf"), smoke_pause_updates=1)
        if model is not None or state["next_pass"] or state["next_batch"] != 1:
            raise ValueError("Smoke fit failed to stop inside a pass")
        resumed, timing = fit_expanded(*args, path / "resume", pause, float("inf"))
        uninterrupted, _ = fit_expanded(*args, path / "uninterrupted", pause, float("inf"))
        for name, parameter in resumed.state_dict().items():
            torch.testing.assert_close(parameter, uninterrupted.state_dict()[name], rtol=0, atol=0)
        if timing["optimizer_updates"] != 4 or timing["patients_seen_by_pass"] != 2:
            raise ValueError("Smoke fit did not reach the intended patient and update budgets")
    atomic_json(OUT / "smoke.json", {
        "passed": True, "exact_resume": True, "encoder_matches_archive": True,
        "outer_evaluation_subjects": 0, "fitting_patients": len(fit), "validation_patients": len(val),
        "active_seconds": active_clock() - started, "process_peak_rss_bytes": peak_memory_bytes(),
        "pass_metrics": timing["history"],
    })
    log("Streaming preparation and exact mid-pass resume verified on inner patients")


def wake_n1_mechanisms(
    base: pd.DataFrame, probabilities: dict[str, np.ndarray], hard: np.ndarray, designs: list[dict],
) -> dict:
    order = np.lexsort((base.epoch_index.to_numpy(), base.subject_id.to_numpy()))
    ordered = base.iloc[order][gated.KEY_FOLD].reset_index(drop=True)
    ordered["phase"], _ = annotation_phases(ordered)
    target = ordered[LABEL].to_numpy()[hard[order]] == WAKE
    results = {}
    for model, folder in (("original", expanded.MODEL), ("expanded", OUT)):
        terms = gated.align(ordered, pd.concat([
            pd.read_parquet(folder / f"fold-{item['fold']}" / "components.parquet") for item in designs
        ], ignore_index=True))
        evidence = terms.loc[hard[order], "local_Wake"] - terms.loc[hard[order], "local_N1"]
        results[model] = {
            "hard_local_auc": float(roc_auc_score(target, evidence)),
            "phases": model_components(ordered, probabilities[model][order], terms),
        }
    return results


def report(base: pd.DataFrame, reference: pd.DataFrame, designs: list[dict]) -> None:
    candidate = gated.align(base, pd.concat([
        pd.read_parquet(OUT / f"fold-{item['fold']}" / "predictions.parquet") for item in designs
    ], ignore_index=True))
    aligned = base[gated.KEY_FOLD].copy()
    probabilities = {name: table[PROBABILITIES].to_numpy()
                     for name, table in (("original", reference), ("expanded", candidate))}
    for name, probs in probabilities.items():
        aligned[name] = probs.argmax(axis=1)
    y = aligned[LABEL].to_numpy()
    subjects = aligned.subject_id.to_numpy()
    folds = [np.flatnonzero(aligned.fold.eq(item["fold"])) for item in designs]
    comparison = gated.compare(aligned, "expanded", "original", folds)
    comparison["kappa_delta"] = comparison["candidate"]["kappa"] - comparison["reference"]["kappa"]
    cm_ref, patients = patient_confusions(subjects, y, aligned.original.to_numpy())
    cm_new, _ = patient_confusions(subjects, y, aligned.expanded.to_numpy())
    membership = aligned.groupby("subject_id").fold.first().loc[patients].to_numpy()
    groups = [np.flatnonzero(membership == item["fold"]) for item in designs]
    draws = cv_fold_draws(groups, 2000, np.random.default_rng(SEED))
    comparison["paired_kappa_ci"] = interval(
        cv_scores(cm_new, groups, draws)["kappa"] - cv_scores(cm_ref, groups, draws)["kappa"]
    )
    comparison["nominate"] = bool(comparison["delta"] >= .005 and comparison["wins"] >= 4
                                   and min(comparison["stage_deltas"].values()) >= -.010)

    compact = base[PROBABILITIES].to_numpy()
    order = np.argsort(-compact, axis=1, kind="stable")[:, :2]
    ranked = np.take_along_axis(compact, order, axis=1)
    hard = (np.isin(y, [WAKE, N1]) & (np.sort(order, axis=1) == [WAKE, N1]).all(axis=1)
            & (ranked[:, 0] <= 2 * ranked[:, 1]))
    band = aligned.subject_id.map(apnea_bands(sorted(aligned.subject_id.unique())))
    bands_by_patient = band.groupby(aligned.subject_id).first()
    paired_bands = bands_by_patient.loc[patients].to_numpy()
    rng = np.random.default_rng(SEED)
    apnea = {}
    for code, name in APNEA_BANDS.items():
        rows = np.flatnonzero(band.eq(code))
        if not len(rows):
            continue
        band_scores = {}
        for model, probs in probabilities.items():
            pred = probs[rows].argmax(axis=1)
            band_scores[model] = {
                "macro_f1": float(f1_score(y[rows], pred, labels=range(len(STAGES)), average="macro", zero_division=0)),
                "kappa": float(cohen_kappa_score(y[rows], pred)),
                **wake_n1_metrics(y[rows], probs[rows], hard[rows]),
            }
        patients_in_band = paired_bands == code
        n_patients = int(patients_in_band.sum())
        band_draws = rng.integers(n_patients, size=(2000, n_patients))
        original_boot = confusion_scores(cm_ref[patients_in_band][band_draws].sum(axis=1))
        expanded_boot = confusion_scores(cm_new[patients_in_band][band_draws].sum(axis=1))
        delta_ci = {metric: interval(expanded_boot[metric] - original_boot[metric]) for metric in ("macro_f1", "kappa")}
        for stage in (WAKE, N1):
            delta_ci[f"f1_{STAGES[stage]}"] = interval(expanded_boot["stage_f1"][:, stage] - original_boot["stage_f1"][:, stage])
        apnea[name] = {
            "patients": n_patients, "epochs": len(rows), **band_scores,
            "macro_f1_delta": band_scores["expanded"]["macro_f1"] - band_scores["original"]["macro_f1"],
            "paired_delta_ci": delta_ci,
        }
    paired = expanded.paired_patient_scores(aligned.rename(columns={"original": "gated"}))
    paired = paired.rename(columns={"gated_f1": "original_f1"})
    paired["apnea_band"] = paired.subject_id.map(bands_by_patient).map(APNEA_BANDS)
    save_frame(OUT / "paired_patients.parquet", paired)
    save_frame(OUT / "predictions.parquet", candidate)
    runtime = [json.loads((OUT / f"fold-{item['fold']}" / "runtime.json").read_text()) for item in designs]
    atomic_json(OUT / "summary.json", {
        "experiment": 73, "development_patients": len(patients), "development_scored_epochs": len(y),
        "comparison": comparison,
        "wake_n1": {name: wake_n1_metrics(y, probs, hard) for name, probs in probabilities.items()},
        "wake_n1_mechanisms": wake_n1_mechanisms(base, probabilities, hard, designs),
        "sustained_wake_n1": {name: sustained_directions(aligned, name) for name in probabilities},
        "apnea_bands": apnea, "folds": runtime,
    })
    log(f"#73 completed: paired macro-F1 {comparison['delta']:+.4f}, {comparison['wins']}/5 winning folds")


def peak_memory_bytes() -> int:
    import sys

    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    import ctypes as c
    from ctypes import wintypes as w

    class ProcessMemoryCounters(c.Structure):
        _fields_ = [("size", w.DWORD), ("fault_count", w.DWORD), ("peak_working_set", c.c_size_t),
                    ("remaining", c.c_size_t * 7)]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    query = c.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [w.HANDLE, c.POINTER(ProcessMemoryCounters), w.DWORD]
    query.restype = w.BOOL
    counters = ProcessMemoryCounters()
    counters.size = c.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), c.byref(counters), counters.size):
        raise c.WinError(c.get_last_error())
    return counters.peak_working_set


def execute(smoke_only: bool) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the training expansion before fitting")
    frame, splits = development_frame()
    partitions = [fold_frames(frame, split) for split in splits]
    if smoke_only:
        smoke(partitions[0])
        return 0
    if not json.loads((OUT / "smoke.json").read_text())["passed"]:
        raise ValueError("Verify the training-only smoke before fitting")
    epochs, added = training_cohort(set(frame.subject_id))
    cohort = pd.concat([frame[[*KEYS, LABEL]], epochs], ignore_index=True)
    base = pd.read_parquet(BASE)
    expected = pd.concat([test.assign(fold=f)[gated.KEY_FOLD] for f, (_, _, test) in enumerate(partitions)])
    gated.align(base, expected)
    reference = gated.align(base, pd.read_parquet(REFERENCE))
    designs = []
    for f, (train, val, test) in enumerate(partitions):
        runtime = json.loads((expanded.MODEL / f"fold-{f}" / "runtime.json").read_text())
        slots = runtime["fit_patients"]
        if slots != train.subject_id.nunique() or runtime["validation_patients"] != val.subject_id.nunique():
            raise ValueError(f"Fold {f}: archive and reconstructed partition sizes disagree")
        designs.append({"fold": f, "original_fitting": sorted(train.subject_id.unique()),
                        "validation": sorted(val.subject_id.unique()), "evaluation": sorted(test.subject_id.unique()),
                        "archive": optimizer_budget(runtime, slots, slots)})

    archived_sources = (Path(expanded.__file__), Path(gated.__file__), Path(SEQUENCE.__file__),
                        Path(gated.gated_staging.__file__), expanded.FEATURE_MANIFEST, BASE)
    archived_manifest = expanded.MODEL / "manifest.json"
    expected_archived = json.loads(archived_manifest.read_text())
    current_archived = {path.name: digest(path) for path in archived_sources}
    if any(expected_archived["sources"].get(name) != value for name, value in current_archived.items()):
        raise ValueError("Current source or reference data differ from experiment #61")
    paths = (Path(__file__), archived_manifest, REFERENCE, SUBJECTS, ADDED_EPOCHS)
    pin_manifest(OUT / "training_manifest.json", {
        "designs": designs, "additional_training_subjects": sorted(added), "seed": SEED, "torch": torch.__version__,
        "sources": {**current_archived, **{path.name: digest(path) for path in paths}},
    })
    ledger = OUT / "active_budget.json"
    used = json.loads(ledger.read_text())["seconds"] if ledger.exists() else 0.0
    start = last_account = active_clock()
    deadline = start + BUDGET_SECONDS - used
    pause = OUT / "PAUSE"

    def account() -> None:
        nonlocal last_account
        last_account = active_clock()
        atomic_json(ledger, {"seconds": used + last_account - start})

    def check() -> None:
        now = active_clock()
        if now - last_account >= CHECKPOINT_SECONDS:
            account()
        if pause.exists() or now >= deadline:
            raise Paused

    try:
        check()
        panels = expanded.load_markers(set(frame.subject_id))
        for subject in sorted(added):
            check()
            panels[subject] = marker_panel(subject)
        for design, (_, _, test) in zip(designs, partitions, strict=True):
            f = design["fold"]
            path = OUT / f"fold-{f}"
            if (path / "predictions.parquet").exists():
                continue
            check()
            train = set(design["original_fitting"]) | added
            archive = design["archive"]
            log(f"Fold {f + 1}/5 preparing features for {len(train)} fitting patients")
            began = active_clock()
            recordings, scaling = prepare_recordings(cohort, train, check, added=added)
            marker_scaling = expanded.add_markers(recordings, panels, train)
            preparation_seconds = active_clock() - began
            model, runtime = fit_expanded(
                recordings, train, design["validation"], archive["patient_slots_per_pass"],
                archive["passes"], SEED + f, path, pause, deadline, account, added=added,
            )
            if model is None:
                raise Paused
            probs, terms = gated.score_components(model, recordings, test)
            keys = test[[*KEYS, LABEL]].assign(fold=f).reset_index(drop=True)
            save_frame(path / "components.parquet", keys.join(pd.DataFrame(terms, columns=gated.TERM_COLUMNS)))
            atomic_json(path / "runtime.json", {
                "fold": f, "archive": archive, "expanded": runtime,
                "scaling": scaling, "marker_scaling": marker_scaling,
                "preparation_seconds": preparation_seconds, "process_peak_rss_bytes": peak_memory_bytes(),
                "scored_patients": test.subject_id.nunique(), "scored_epochs": len(test),
            })
            write_predictions(path / "predictions.parquet", test, probs, f)
            log(f"Fold {f + 1}/5 saved with {runtime['optimizer_updates']} matched optimizer updates")
            del recordings
        report(base, reference, designs)
    except Paused:
        log("#73 stopped at a saved checkpoint or before the next work unit")
        return 75
    finally:
        account()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Verify the encoder and exact mid-pass resume on inner patients")
    args = parser.parse_args()
    with local_training_slot():
        raise SystemExit(execute(args.smoke))
