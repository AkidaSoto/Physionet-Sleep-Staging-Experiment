"""Fit the five-fold complementary-marker extension of the archived gated network (#61)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import run_staging_gated_temporal as gated

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from physionet_sleep.experiments.complementary_n2_events import N2_EVENT_COLUMNS
from physionet_sleep.experiments.complementary_slow_waves import SLOW_WAVE_PANEL_COLUMNS
from physionet_sleep.experiments.gated_staging import GatedStagingNet, stage_loss
from physionet_sleep.experiments.waveform_clock import active_clock
from run_staging_feature_tuning import BASE, error_structure, save_frame
from run_waveform_development import (
    EXPLORATION,
    KEYS,
    LABEL,
    PROBABILITIES,
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


OUT = EXPLORATION / "cinc-complementary-markers"
MODEL = OUT / "model"
FEATURE_MANIFEST = OUT / "feature_manifest.json"
FEATURES = (*SLOW_WAVE_PANEL_COLUMNS, *N2_EVENT_COLUMNS)
ARCHIVED_GATED = gated.OUT / "gated_predictions.parquet"
ORIGINAL_WIDTH = 90
FAMILY_SLICES = {
    "without_slow_waves": slice(
        ORIGINAL_WIDTH, ORIGINAL_WIDTH + len(SLOW_WAVE_PANEL_COLUMNS)
    ),
    "without_n2_events": slice(
        ORIGINAL_WIDTH + len(SLOW_WAVE_PANEL_COLUMNS), ORIGINAL_WIDTH + len(FEATURES)
    ),
}


class MarkerProjection(nn.Module):
    """Keep the original stem numerics and add a learnable, initially zero correction."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.original = original
        with torch.random.fork_rng(devices=[]):
            self.markers = nn.Linear(len(FEATURES), original.out_features, bias=False)
        nn.init.zeros_(self.markers.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old = self.original(x[..., : self.original.in_features].contiguous())
        added = self.markers(x[..., self.original.in_features :].contiguous())
        return old + added


def expanded_model() -> GatedStagingNet:
    model = GatedStagingNet(gated=True)
    model.stem[0] = MarkerProjection(model.stem[0])
    return model


class WithoutMarkerFamily(nn.Module):
    """Set one standardized family to its fitting-patient average at inference."""

    def __init__(self, model: GatedStagingNet, columns: slice):
        super().__init__()
        self.model = model
        self.columns = columns

    def forward(self, x: torch.Tensor, recorded: torch.Tensor) -> torch.Tensor:
        adjusted = x.clone()
        adjusted[..., self.columns] = 0.0
        return self.model(adjusted, recorded)


def load_markers(subjects: set[str]) -> dict[str, np.ndarray]:
    panels = {}
    for subject in sorted(subjects):
        path = OUT / "feature_cache" / f"{subject}.parquet"
        panel = pd.read_parquet(path)
        expected_index = pd.RangeIndex(len(panel), name="epoch_index")
        if not panel.index.equals(expected_index) or panel.index.name != "epoch_index":
            raise ValueError(
                f"{subject}: expected complete consecutive marker epoch keys"
            )
        if (
            not len(panel)
            or set(panel.columns) != set(FEATURES)
            or not panel.columns.is_unique
        ):
            raise ValueError(
                f"{subject}: marker columns or rows differ from the declared panel"
            )
        values = panel.loc[:, list(FEATURES)].to_numpy(dtype=np.float32, copy=True)
        values[~np.isfinite(values)] = np.nan
        panels[subject] = values
    return panels


def add_markers(
    recordings: dict[str, gated.sequence.Recording],
    panels: dict[str, np.ndarray],
    fitting_subjects: set[str],
) -> dict:
    count = np.zeros(len(FEATURES), dtype=np.int64)
    total = np.zeros(len(FEATURES), dtype=np.float64)
    squared = np.zeros(len(FEATURES), dtype=np.float64)
    for subject, rec in recordings.items():
        values = panels[subject]
        if (rec.scored_labels[len(values) :] >= 0).any():
            raise ValueError(f"{subject}: marker stream ends before a scored epoch")
        if subject in fitting_subjects:
            valid = values[: len(rec.features)]
            count += np.isfinite(valid).sum(axis=0)
            total += np.nansum(valid, axis=0, dtype=np.float64)
            squared += np.nansum(np.square(valid, dtype=np.float64), axis=0)
    if (count == 0).any():
        missing = [feature for feature, n in zip(FEATURES, count) if not n]
        raise ValueError(f"No fitting-patient data for {missing}")
    mean = total / count
    scale = np.sqrt(np.maximum(0.0, squared / count - mean**2)) + 1e-6
    for subject, rec in recordings.items():
        original = rec.features
        values = panels[subject][: len(original)]
        standardized = np.zeros((len(original), len(FEATURES)), dtype=np.float32)
        normalized = (values - mean) / scale
        standardized[: len(values)] = np.where(np.isfinite(normalized), normalized, 0.0)
        rec.features = torch.cat((original, torch.from_numpy(standardized)), dim=-1)
    return {
        "features": list(FEATURES),
        "finite_fit_epochs": count.tolist(),
        "fit_mean": mean.tolist(),
        "fit_scale": scale.tolist(),
    }


def fit(
    recordings: dict[str, gated.sequence.Recording],
    train: list[str],
    val: list[str],
    directory: Path,
    seed: int,
    deadline: float,
    *,
    max_passes: int = gated.sequence.MAX_PASSES,
    stop_after: int | None = None,
) -> tuple[GatedStagingNet | None, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    torch.manual_seed(seed)
    model = expanded_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_passes, eta_min=0.0003
    )
    labels = torch.cat([recordings[s].scored_labels for s in train])
    counts = torch.bincount(labels[labels >= 0], minlength=len(STAGES)).float()
    if (counts == 0).any():
        raise ValueError("Every stage must occur in the fitting patients")
    stage_weight = counts.pow(-0.70)
    state = {"next_pass": 0, "best": -1.0, "best_pass": 0, "history": []}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])

    while not gated.stopped(state, max_passes):
        if (OUT / "PAUSE").exists() or active_clock() >= deadline:
            return None, state
        step = state["next_pass"]
        began = active_clock()
        model.train()
        for batch in gated.sequence.batches(train, np.random.default_rng(seed + step)):
            x, y, recorded = gated.sequence.collate(recordings, batch)
            optimizer.zero_grad(set_to_none=True)
            parts = model(x, recorded, return_components=True)
            loss = stage_loss(parts.logits, y, stage_weight)
            loss = loss + gated.LOCAL_LOSS * stage_loss(parts.local, y, stage_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss in expanded model")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        metric = gated.sequence.validation_score(model, recordings, val)
        pass_number = step + 1
        if metric > state["best"]:
            state.update(
                best=metric,
                best_pass=pass_number,
                best_weights={
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                },
            )
        elapsed = active_clock() - began
        state["history"].append(
            {"pass": pass_number, "val_macro_f1": metric, "seconds": elapsed}
        )
        state.update(
            next_pass=pass_number,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=torch.get_rng_state(),
        )
        gated.sequence.checkpoint(checkpoint, state)
        if pass_number == 1 or pass_number % 5 == 0:
            log(
                f"{directory.name} expanded pass {pass_number}: validation {metric:.4f}, "
                f"best {state['best']:.4f}, {elapsed:.1f}s"
            )
        if pass_number == stop_after:
            return None, state
    model.load_state_dict(state["best_weights"])
    return model, {
        "fit_patients": len(train),
        "validation_patients": len(val),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stage_counts": counts.int().tolist(),
        "class_weights": stage_weight.tolist(),
        "best_pass": state["best_pass"],
        "best_validation_macro_f1": state["best"],
        "passes": state["next_pass"],
        "history": state["history"],
    }


@torch.inference_mode()
def verify_initialization(x: torch.Tensor, recorded: torch.Tensor, seed: int) -> None:
    torch.manual_seed(seed)
    reference = GatedStagingNet(gated=True).eval()
    reference_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    expanded = expanded_model().eval()
    if not torch.equal(torch.get_rng_state(), reference_rng):
        raise ValueError("The marker projection changed the archived initial RNG state")
    torch.testing.assert_close(
        expanded(x, recorded),
        reference(x[..., :ORIGINAL_WIDTH].contiguous(), recorded),
        rtol=0,
        atol=0,
    )


def verify_smoke(frame: pd.DataFrame, split) -> None:
    train, val, _ = fold_frames(frame, split)
    train_ids = sorted(train.subject_id.unique())[:8]
    val_ids = sorted(val.subject_id.unique())[:4]
    subjects = set(train_ids + val_ids)
    sample = frame[frame.subject_id.isin(subjects)]
    recordings = gated.sequence.prepare_recordings(sample, set(train_ids))
    add_markers(recordings, load_markers(subjects), set(train_ids))
    x, _, valid = gated.sequence.collate(recordings, train_ids[:2])
    verify_initialization(x, valid, SEED)
    MODEL.mkdir(parents=True, exist_ok=True)
    options = {"seed": SEED, "deadline": float("inf"), "max_passes": 2}
    with TemporaryDirectory(dir=MODEL, prefix="smoke-") as temporary:
        scratch = Path(temporary)
        paused, state = fit(
            recordings, train_ids, val_ids, scratch / "resume", **options, stop_after=1
        )
        if paused is not None or state["next_pass"] != 1:
            raise ValueError("The smoke fit did not stop after its first pass")
        resumed, _ = fit(recordings, train_ids, val_ids, scratch / "resume", **options)
        control, _ = fit(
            recordings, train_ids, val_ids, scratch / "uninterrupted", **options
        )
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(
                value, control.state_dict()[name], rtol=0, atol=0
            )
        if (resumed.stem[0].markers.weight == 0).all():
            raise ValueError("The added marker projection received no training update")
        sample_val = val[val.subject_id.isin(val_ids)].iloc[:25]
        probabilities, _ = gated.score_components(resumed, recordings, sample_val)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    atomic_json(
        MODEL / "smoke.json",
        {
            "passed": True,
            "scored_outer_patients": 0,
            "initial_logits_exact": True,
            "initial_rng_exact": True,
            "exact_resume": True,
        },
    )
    log(
        "Training-only initialization, learnable marker projection, and exact resume verified"
    )


def sustained_runs(
    ordered: pd.DataFrame, name: str, *, other_correct: str | None = None
) -> dict:
    y, predicted, epoch, subject = (
        ordered[column].to_numpy()
        for column in (LABEL, name, "epoch_index", "subject_id")
    )
    eligible = y != predicted
    if other_correct is not None:
        eligible &= ordered[other_correct].to_numpy() == y
    new = np.r_[
        True,
        (np.diff(epoch) != 1)
        | (subject[1:] != subject[:-1])
        | (y[1:] != y[:-1])
        | (predicted[1:] != predicted[:-1])
        | (eligible[1:] != eligible[:-1]),
    ]
    run = np.cumsum(new) - 1
    sustained = (np.bincount(run)[run] >= 6) & eligible
    return {"epochs": int(sustained.sum()), "runs": int((new & sustained).sum())}


def paired_patient_scores(aligned: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for (fold, subject), night in aligned.groupby(["fold", "subject_id"], sort=True):
        y = night[LABEL].to_numpy()
        labels = np.unique(y)
        control, expanded = (
            f1_score(y, night[name], labels=labels, average="macro", zero_division=0)
            for name in ("gated", "expanded")
        )
        scores.append(
            {
                "fold": fold,
                "subject_id": subject,
                "epochs": len(night),
                "gated_f1": control,
                "expanded_f1": expanded,
                "delta": expanded - control,
            }
        )
    return pd.DataFrame(scores)


def sustained_cohorts(
    cohort: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> list[dict]:
    y, rival = (cohort[column].to_numpy() for column in (LABEL, "searched"))
    shared = cohort.paired_run_6 & cohort.baseline_to_searched.eq("persistent_same")
    rows = []
    for stage, other in [(2, 1), (2, 3)]:
        affected = shared & (y == stage) & (rival == other)
        row = {
            "stage": STAGES[stage],
            "rival": STAGES[other],
            "epochs": int(affected.sum()),
        }
        for name, pred in predictions.items():
            row[name] = {
                "correct": int((affected & (pred == stage)).sum()),
                "same_rival": int((affected & (pred == other)).sum()),
            }
        control, expanded = predictions["gated"], predictions["expanded"]
        row["repaired_vs_gated"] = int(
            (affected & (control != stage) & (expanded == stage)).sum()
        )
        row["regressed_vs_gated"] = int(
            (affected & (control == stage) & (expanded != stage)).sum()
        )
        rows.append(row)
    return rows


def report(base: pd.DataFrame, folds_count: int) -> None:
    cohort = gated.align(
        base,
        pd.read_parquet(
            gated.ARCHIVED,
            columns=[
                *gated.KEY_FOLD,
                "searched",
                "baseline_to_searched",
                "paired_run_6",
            ],
        ),
    )
    paths = [MODEL / f"fold-{fold}" for fold in range(folds_count)]
    expanded = gated.align(
        base, pd.concat([pd.read_parquet(p / "predictions.parquet") for p in paths])
    )
    original = gated.align(base, pd.read_parquet(ARCHIVED_GATED))
    parts = gated.align(
        base, pd.concat([pd.read_parquet(p / "components.parquet") for p in paths])
    )
    distributions = {
        "baseline": base[PROBABILITIES].to_numpy(),
        "gated": original[PROBABILITIES].to_numpy(),
        "expanded": expanded[PROBABILITIES].to_numpy(),
    }
    for name in FAMILY_SLICES:
        masked = gated.align(
            base, pd.concat([pd.read_parquet(p / f"{name}.parquet") for p in paths])
        )
        distributions[name] = masked[PROBABILITIES].to_numpy()
    aligned = base[gated.KEY_FOLD].copy()
    predictions = {name: values.argmax(1) for name, values in distributions.items()}
    for name, predictions_for_model in predictions.items():
        aligned[name] = predictions_for_model
    fold_rows = [np.flatnonzero(aligned.fold.eq(fold)) for fold in range(folds_count)]
    contrasts = {
        f"expanded_vs_{reference}": gated.compare(
            aligned, "expanded", reference, fold_rows
        )
        for reference in ("gated", "baseline")
    }
    p = distributions["baseline"]
    order = np.argsort(-p, axis=1, kind="stable")[:, :2]
    ordered_p = np.take_along_axis(p, order, axis=1)
    close = ordered_p[:, 0] <= 2 * ordered_p[:, 1]
    hard = {
        (i, j): close & (np.sort(order, axis=1) == [i, j]).all(1)
        for i, j in gated.TARGET_PAIRS
    }
    y = aligned[LABEL].to_numpy()
    metrics = {
        name: gated.probability_metrics(y, p, hard) for name, p in distributions.items()
    }
    ordered = aligned.sort_values(KEYS)
    run_counts = {name: sustained_runs(ordered, name) for name in distributions}
    errors = {name: error_structure(aligned, name) for name in ("gated", "expanded")}
    traces = gated.sustained_trace("expanded", parts, cohort, distributions["expanded"])
    matched = contrasts["expanded_vs_gated"]
    matched["stronger"] = (
        matched["candidate"]["macro_f1"] > 0.800
        and matched["wins"] >= 4
        and min(matched["stage_deltas"].values()) >= -0.010
    )
    matched["diagnostic_gain"] = (
        matched["delta"] >= 0.005
        and matched["wins"] >= 4
        and all(
            result["errors"] < metrics["gated"]["pairs"][pair]["errors"]
            for pair, result in metrics["expanded"]["pairs"].items()
        )
    )
    summary = {
        "added_features": list(FEATURES),
        "fold_fits": folds_count,
        "contrasts": contrasts,
        "sustained_errors": run_counts,
        "changed_sustained_errors": {
            "expanded_wrong_and_gated_correct": sustained_runs(
                ordered, "expanded", other_correct="gated"
            ),
            "gated_wrong_and_expanded_correct": sustained_runs(
                ordered, "gated", other_correct="expanded"
            ),
        },
        "errors": errors,
        "sustained_cohorts": sustained_cohorts(cohort, predictions),
        "sustained_logit_contributions": traces,
        "probabilities": {k: metrics[k] for k in ("baseline", "gated", "expanded")},
        "inference_sensitivity": {
            "method": "Set one fitted marker family to its fitting-patient mean without retraining.",
            "probabilities": {k: metrics[k] for k in FAMILY_SLICES},
        },
    }
    save_frame(OUT / "expanded_predictions.parquet", expanded)
    save_frame(OUT / "paired_patient_scores.parquet", paired_patient_scores(aligned))
    atomic_json(OUT / "model_summary.json", summary)
    result = contrasts["expanded_vs_gated"]
    log(
        f"Expanded {result['candidate']['macro_f1']:.4f} macro-F1, "
        f"{result['delta']:+.4f} over archived gated model"
    )


def execute(smoke: bool) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the study before fitting")
    frame, splits = development_frame()
    if smoke:
        verify_smoke(frame, splits[0])
        return 0
    if not json.loads((MODEL / "smoke.json").read_text())["passed"]:
        raise ValueError("The training-only smoke must pass before scoring")
    base = pd.read_parquet(BASE)
    partitions = [fold_frames(frame, split) for split in splits]
    expected = pd.concat(
        [
            test[KEYS + [LABEL]].assign(fold=f)
            for f, (_, _, test) in enumerate(partitions)
        ]
    )
    gated.align(base, expected)
    gated.align(base, pd.read_parquet(ARCHIVED_GATED))
    sources = [
        Path(__file__),
        Path(gated.__file__),
        Path(gated.sequence.__file__),
        Path(gated.gated_staging.__file__),
        FEATURE_MANIFEST,
        BASE,
        ARCHIVED_GATED,
        gated.ARCHIVED,
    ]
    pin_manifest(
        MODEL / "manifest.json",
        {
            "added_features": list(FEATURES),
            "seed": SEED,
            "torch": torch.__version__,
            "sources": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
        },
    )
    ledger = MODEL / "active_budget.json"
    used = json.loads(ledger.read_text())["seconds"] if ledger.exists() else 0.0
    started = active_clock()
    deadline = started + 86400 - used
    try:
        panels = load_markers(set(frame.subject_id))
        for f, (train, val, test) in enumerate(partitions):
            path = MODEL / f"fold-{f}"
            if (path / "predictions.parquet").exists():
                continue
            if (OUT / "PAUSE").exists() or active_clock() >= deadline:
                return 75
            train_ids = sorted(train.subject_id.unique())
            val_ids = sorted(val.subject_id.unique())
            recordings = gated.sequence.prepare_recordings(frame, set(train_ids))
            scaling = add_markers(recordings, panels, set(train_ids))
            model, timing = fit(
                recordings, train_ids, val_ids, path, SEED + f, deadline
            )
            if model is None:
                log("Paused at a saved pass boundary")
                return 75
            probabilities, terms = gated.score_components(model, recordings, test)
            keys = test[KEYS + [LABEL]].assign(fold=f).reset_index(drop=True)
            save_frame(
                path / "components.parquet",
                keys.join(pd.DataFrame(terms, columns=gated.TERM_COLUMNS)),
            )
            for name, columns in FAMILY_SLICES.items():
                masked = WithoutMarkerFamily(model, columns)
                p = gated.sequence.score_fold(masked, recordings, test)
                write_predictions(path / f"{name}.parquet", test, p, f)
            atomic_json(path / "runtime.json", {**timing, "marker_scaling": scaling})
            write_predictions(path / "predictions.parquet", test, probabilities, f)
            metric = f1_score(test[LABEL], probabilities.argmax(1), average="macro")
            log(f"Fold {f + 1}/{len(partitions)} expanded saved: macro-F1 {metric:.4f}")
            del recordings
        report(base, len(partitions))
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - started})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Verify resume on inner patients"
    )
    with local_training_slot():
        raise SystemExit(execute(parser.parse_args().smoke))
