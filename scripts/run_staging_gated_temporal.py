"""Compare directed context gates and a pair-specific stage loss on Boston development."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

import run_staging_feature_context as sequence
from build_metric_intervals import (
    cv_fold_draws,
    cv_scores,
    interval,
    patient_confusions,
)
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
    scores,
    write_predictions,
)
from physionet_sleep.experiments import gated_staging
from physionet_sleep.experiments.waveform_clock import active_clock


OUT = EXPLORATION / "cinc-gated-temporal"
ARCHIVED = EXPLORATION / "cinc-error-interactions/archived_error_epochs.parquet"
PREVIOUS_SEQUENCE = EXPLORATION / "cinc-feature-context/long_predictions.parquet"
KEY_FOLD = [*KEYS, LABEL, "fold"]
VARIANTS = {"ungated": (False, 0.0), "gated": (True, 0.0), "gated_pair": (True, 0.35)}
LOCAL_LOSS = 0.25
TERMS = ("local", "past", "future", "past_gate", "future_gate")
TERM_COLUMNS = [f"{term}_{stage}" for term in TERMS for stage in STAGES]
TARGET_PAIRS = [(1, 2), (2, 3)]


def stopped(state: dict, max_passes: int) -> bool:
    return state["next_pass"] >= max_passes or (
        state["next_pass"] >= 12 and state["next_pass"] - state["best_pass"] >= 5
    )


def fit(
    variant: str,
    recordings: dict[str, sequence.Recording],
    train: list[str],
    val: list[str],
    directory: Path,
    seed: int,
    deadline: float,
    *,
    max_passes: int = sequence.MAX_PASSES,
    stop_after: int | None = None,
) -> tuple[gated_staging.GatedStagingNet | None, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    gated, pair_strength = VARIANTS[variant]
    torch.manual_seed(seed)
    model = gated_staging.GatedStagingNet(gated=gated)
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

    while not stopped(state, max_passes):
        if (OUT / "PAUSE").exists() or active_clock() >= deadline:
            return None, state
        step = state["next_pass"]
        began = active_clock()
        model.train()
        for batch in sequence.batches(train, np.random.default_rng(seed + step)):
            x, y, recorded = sequence.collate(recordings, batch)
            optimizer.zero_grad(set_to_none=True)
            parts = model(x, recorded, return_components=True)
            loss = gated_staging.stage_loss(
                parts.logits, y, stage_weight, pair_strength=pair_strength
            )
            loss = loss + LOCAL_LOSS * gated_staging.stage_loss(
                parts.local, y, stage_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {variant}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        metric = sequence.validation_score(model, recordings, val)
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
        sequence.checkpoint(checkpoint, state)
        if pass_number == 1 or pass_number % 5 == 0:
            log(
                f"{directory.parent.name} {variant} pass {pass_number}: "
                f"validation {metric:.4f}, best {state['best']:.4f}, {elapsed:.1f}s"
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
def score_components(
    model: gated_staging.GatedStagingNet,
    recordings: dict[str, sequence.Recording],
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows = test.groupby("subject_id", sort=False).indices
    probabilities = np.empty((len(test), len(STAGES)), dtype=np.float32)
    values = np.empty((len(test), len(TERM_COLUMNS)), dtype=np.float32)
    for batch in sequence.batches(list(rows)):
        x, _, recorded = sequence.collate(recordings, batch)
        parts = model(x, recorded, return_components=True)
        torch.testing.assert_close(
            parts.logits, parts.local + parts.past + parts.future
        )
        p = parts.logits.softmax(dim=-1).numpy()
        terms = torch.cat([getattr(parts, term) for term in TERMS], dim=-1).numpy()
        for i, subject in enumerate(batch):
            positions = rows[subject]
            epoch = test.iloc[positions].epoch_index.to_numpy()
            probabilities[positions] = p[i, epoch]
            values[positions] = terms[i, epoch]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite component scores in evaluated epochs")
    return probabilities, values


def verify_smoke(frame: pd.DataFrame, split) -> None:
    train, val, _ = fold_frames(frame, split)
    train_ids = sorted(train.subject_id.unique())[:8]
    val_ids = sorted(val.subject_id.unique())[:4]
    recordings = sequence.prepare_recordings(
        frame[frame.subject_id.isin(train_ids + val_ids)], set(train_ids)
    )
    directory = OUT / "smoke"
    options = {"seed": SEED, "deadline": float("inf"), "max_passes": 2}
    model, state = fit(
        "gated_pair",
        recordings,
        train_ids,
        val_ids,
        directory / "resume",
        **options,
        stop_after=1,
    )
    if model is not None or state["next_pass"] != 1:
        raise ValueError("Smoke checkpoint did not stop after the first pass")
    resumed, _ = fit(
        "gated_pair", recordings, train_ids, val_ids, directory / "resume", **options
    )
    control, _ = fit(
        "gated_pair",
        recordings,
        train_ids,
        val_ids,
        directory / "uninterrupted",
        **options,
    )
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
    evaluated = val[val.subject_id.isin(val_ids)].iloc[:25]
    probabilities, _ = score_components(resumed, recordings, evaluated)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    atomic_json(
        directory / "summary.json", {"passed": True, "scored_outer_patients": 0}
    )
    log("Training-only checkpoint, exact resume, and decomposed scores verified")


def align(base: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    joined = base[KEY_FOLD].merge(
        predictions, on=KEY_FOLD, validate="one_to_one", how="left", indicator=True
    )
    if len(base) != len(predictions) or not joined["_merge"].eq("both").all():
        raise ValueError("A comparison has mismatched held-out patient or epoch keys")
    return joined.drop(columns="_merge")


def compare(
    aligned: pd.DataFrame, candidate: str, reference: str, folds: list[np.ndarray]
) -> dict:
    y, a, b = (aligned[name].to_numpy() for name in (LABEL, candidate, reference))
    case, control = scores(aligned, a, folds), scores(aligned, b, folds)
    deltas = [
        float(
            f1_score(y[i], a[i], average="macro")
            - f1_score(y[i], b[i], average="macro")
        )
        for i in folds
    ]
    ca, patients = patient_confusions(aligned.subject_id.to_numpy(), y, a)
    cb, _ = patient_confusions(aligned.subject_id.to_numpy(), y, b)
    membership = aligned.groupby("subject_id").fold.first().loc[patients].to_numpy()
    groups = [np.flatnonzero(membership == fold) for fold in range(len(folds))]
    draws = cv_fold_draws(groups, 2000, np.random.default_rng(SEED))
    bounds = interval(
        cv_scores(ca, groups, draws)["macro_f1"]
        - cv_scores(cb, groups, draws)["macro_f1"]
    )
    return {
        "candidate": case,
        "reference": control,
        "delta": float(np.mean(deltas)),
        "fold_deltas": deltas,
        "wins": sum(value > 0 for value in deltas),
        "paired_ci": bounds,
        "stage_deltas": {s: case[f"f1_{s}"] - control[f"f1_{s}"] for s in STAGES},
    }


def probability_metrics(
    y: np.ndarray, p: np.ndarray, hard: dict[tuple[int, int], np.ndarray]
) -> dict:
    pred = p.argmax(1)
    precision, recall, f1, counts = precision_recall_fscore_support(
        y, pred, labels=range(len(STAGES)), zero_division=0
    )
    pairs = {}
    for i, j in TARGET_PAIRS:
        selected = np.isin(y, [i, j])
        both = selected & hard[i, j]
        margin = np.log(p[:, i].clip(1e-12)) - np.log(p[:, j].clip(1e-12))
        pairs[f"{STAGES[i]}/{STAGES[j]}"] = {
            f"{STAGES[i]}_as_{STAGES[j]}": int(((y == i) & (pred == j)).sum()),
            f"{STAGES[j]}_as_{STAGES[i]}": int(((y == j) & (pred == i)).sum()),
            "errors": int((((y == i) & (pred == j)) | ((y == j) & (pred == i))).sum()),
            "auc_all": float(roc_auc_score(y[selected] == i, margin[selected])),
            "auc_hard": float(roc_auc_score(y[both] == i, margin[both])),
            "hard_epochs": int(both.sum()),
        }
    return {
        "stages": {
            s: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "epochs": int(counts[k]),
            }
            for k, s in enumerate(STAGES)
        },
        "pairs": pairs,
        "log_loss": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        "brier": float(np.square(p - np.eye(len(STAGES))[y]).sum(1).mean()),
    }


def sustained_trace(
    name: str, parts: pd.DataFrame, cohort: pd.DataFrame, probabilities: np.ndarray
) -> list[dict]:
    y, rival = (cohort[c].to_numpy() for c in (LABEL, "searched"))
    shared = cohort.paired_run_6 & cohort.baseline_to_searched.eq("persistent_same")
    local = parts[[f"local_{s}" for s in STAGES]].to_numpy().argmax(1)
    pred = probabilities.argmax(1)
    rows = []
    for i, j in [
        *TARGET_PAIRS,
        *(pair[::-1] for pair in TARGET_PAIRS),
        (1, 0),
        (0, 1),
        (4, 1),
        (1, 4),
    ]:
        mask = shared & (y == i) & (rival == j)
        sub = parts.loc[mask]
        if sub.empty:
            continue
        record = {
            "model": name,
            "stage": STAGES[i],
            "rival": STAGES[j],
            "epochs": len(sub),
            "patients": cohort.loc[mask, "subject_id"].nunique(),
            "local_correct": int((local[mask] == i).sum()),
            "fused_correct": int((pred[mask] == i).sum()),
            "local_rival": int((local[mask] == j).sum()),
            "fused_rival": int((pred[mask] == j).sum()),
            "correct_local_to_rival": int(((local == i) & (pred == j) & mask).sum()),
            "rival_local_to_correct": int(((local == j) & (pred == i) & mask).sum()),
        }
        for term in TERMS[:3]:
            difference = sub[f"{term}_{STAGES[i]}"] - sub[f"{term}_{STAGES[j]}"]
            record[f"{term}_median_log_odds"] = float(difference.median())
        for term in TERMS[3:]:
            for stage in (STAGES[i], STAGES[j]):
                record[f"{term}_{stage}_median"] = float(
                    sub[f"{term}_{stage}"].median()
                )
        rows.append(record)
    return rows


def report(base: pd.DataFrame) -> None:
    cohort = align(
        base,
        pd.read_parquet(
            ARCHIVED,
            columns=[
                *KEY_FOLD,
                "searched",
                "baseline_to_searched",
                "paired_run_6",
            ],
        ),
    )
    aligned = base[KEY_FOLD].copy()
    y = base[LABEL].to_numpy()
    p = base[PROBABILITIES].to_numpy()
    order = np.argsort(-p, axis=1, kind="stable")[:, :2]
    ordered_p = np.take_along_axis(p, order, axis=1)
    close = ordered_p[:, 0] <= 2 * ordered_p[:, 1]
    hard = {
        (i, j): close & (np.sort(order, axis=1) == [i, j]).all(1)
        for i, j in TARGET_PAIRS
    }
    previous = align(base, pd.read_parquet(PREVIOUS_SEQUENCE))
    distributions = {
        "baseline": p,
        "previous_sequence": previous[PROBABILITIES].to_numpy(),
    }
    traces = []
    for name in VARIANTS:
        path = [OUT / f"fold-{fold}" / name for fold in range(5)]
        predictions = align(
            base, pd.concat([pd.read_parquet(p / "predictions.parquet") for p in path])
        )
        parts = align(
            base, pd.concat([pd.read_parquet(p / "components.parquet") for p in path])
        )
        distributions[name] = predictions[PROBABILITIES].to_numpy()
        traces.extend(sustained_trace(name, parts, cohort, distributions[name]))
        save_frame(OUT / f"{name}_predictions.parquet", predictions)
    metrics = {
        name: probability_metrics(y, p, hard) for name, p in distributions.items()
    }
    for name, probabilities in distributions.items():
        aligned[name] = probabilities.argmax(1)
    folds = [np.flatnonzero(aligned.fold.eq(f)) for f in range(5)]
    comparisons = [(name, "baseline") for name in VARIANTS] + [
        ("gated", "ungated"),
        ("gated_pair", "gated"),
        ("gated", "previous_sequence"),
    ]
    contrasts = {f"{a}_vs_{b}": compare(aligned, a, b, folds) for a, b in comparisons}
    for name in VARIANTS:
        fit_result = contrasts[f"{name}_vs_baseline"]
        fit_result["stronger"] = (
            fit_result["candidate"]["macro_f1"] > 0.800
            and fit_result["wins"] >= 4
            and min(fit_result["stage_deltas"].values()) >= -0.010
        )
    g = contrasts["gated_vs_ungated"]
    g["passes"] = (
        g["delta"] >= 0.005 and g["wins"] >= 4 and g["stage_deltas"]["N1"] >= -0.010
    )
    m = contrasts["gated_pair_vs_gated"]
    m["passes"] = (
        m["delta"] >= 0.005
        and m["wins"] >= 4
        and min(m["stage_deltas"].values()) >= -0.010
        and all(
            value["errors"] < metrics["gated"]["pairs"][pair]["errors"]
            for pair, value in metrics["gated_pair"]["pairs"].items()
        )
    )
    errors = {name: error_structure(aligned, name) for name in distributions}
    pd.DataFrame(traces).to_csv(OUT / "sustained_logit_contributions.csv", index=False)
    save_frame(OUT / "paired_predictions.parquet", aligned)
    atomic_json(
        OUT / "summary.json",
        {"contrasts": contrasts, "probabilities": metrics, "errors": errors},
    )
    for name in VARIANTS:
        result = contrasts[f"{name}_vs_baseline"]
        log(
            f"{name}: {result['candidate']['macro_f1']:.4f} macro-F1, {result['delta']:+.4f} over HGB18"
        )


def execute(smoke: bool) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT / "FINDINGS.md").exists():
        raise FileNotFoundError("Declare the study before fitting")
    frame, splits = development_frame()
    if smoke:
        verify_smoke(frame, splits[0])
        return 0
    if not json.loads((OUT / "smoke/summary.json").read_text())["passed"]:
        raise ValueError("The training-only smoke must pass before scoring")
    base = pd.read_parquet(BASE)
    partitions = [fold_frames(frame, split) for split in splits]
    expected = pd.concat(
        [
            test[KEYS + [LABEL]].assign(fold=f)
            for f, (_, _, test) in enumerate(partitions)
        ]
    )
    align(base, expected)
    sources = [
        Path(__file__),
        Path(gated_staging.__file__),
        Path(sequence.__file__),
        BASE,
        ARCHIVED,
        PREVIOUS_SEQUENCE,
    ]
    pin_manifest(
        OUT / "manifest.json",
        {
            "variants": {k: list(v) for k, v in VARIANTS.items()},
            "local_loss": LOCAL_LOSS,
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
        for f, (train, val, test) in enumerate(partitions):
            paths = {name: OUT / f"fold-{f}" / name for name in VARIANTS}
            remaining = [
                name
                for name, path in paths.items()
                if not (path / "predictions.parquet").exists()
            ]
            if not remaining:
                continue
            if (OUT / "PAUSE").exists() or active_clock() >= deadline:
                return 75
            train_ids = sorted(train.subject_id.unique())
            val_ids = sorted(val.subject_id.unique())
            recordings = sequence.prepare_recordings(frame, set(train_ids))
            for name in remaining:
                path = paths[name]
                model, timing = fit(
                    name, recordings, train_ids, val_ids, path, SEED + f, deadline
                )
                if model is None:
                    log("Paused at a saved pass boundary")
                    return 75
                probabilities, terms = score_components(model, recordings, test)
                keys = test[KEYS + [LABEL]].assign(fold=f).reset_index(drop=True)
                save_frame(
                    path / "components.parquet",
                    keys.join(pd.DataFrame(terms, columns=TERM_COLUMNS)),
                )
                atomic_json(path / "runtime.json", timing)
                write_predictions(path / "predictions.parquet", test, probabilities, f)
                metric = f1_score(test[LABEL], probabilities.argmax(1), average="macro")
                log(
                    f"Fold {f + 1}/{len(partitions)} {name} saved: macro-F1 {metric:.4f}"
                )
            del recordings
        report(base)
    finally:
        atomic_json(ledger, {"seconds": used + active_clock() - started})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Verify exact resume on inner patients"
    )
    with local_training_slot():
        raise SystemExit(execute(parser.parse_args().smoke))
