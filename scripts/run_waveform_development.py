"""Run resumable raw-waveform and feature-reduction experiments on Boston development patients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from importlib.metadata import version
from itertools import islice
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from run_window_hmm_staging import REPO, SEED, STAGES, adopted_signals, build_subject_eval_splits, log, scores
from physionet_sleep.experiments.waveform_clock import active_clock

BASELINE = REPO / "artifacts/staging_window_model/cinc2018"
EXPLORATION = REPO / "artifacts/exploration"
REDUCTION = EXPLORATION / "cinc-reduction18"
WAVEFORM = EXPLORATION / "cinc-waveform"
CACHE = REPO / "artifacts/cinc2018_window_cache"
RAW = REPO / "data/raw/cinc2018"
KEYS = ["subject_id", "epoch_index"]
LABEL = "label.stage_30s"
PROBABILITIES = [f"p_{stage}" for stage in STAGES]
BUDGET_SECONDS = 24 * 60 * 60


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def pin_manifest(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Saved manifest differs from requested run at {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, payload)


@contextmanager
def local_training_slot():
    """Use an OS lock so a terminated process cannot leave a stale training slot."""
    EXPLORATION.mkdir(parents=True, exist_ok=True)
    with (EXPLORATION / "cinc-waveform-training.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError("Another development job is running on this machine") from exc
        yield


def development_frame() -> tuple[pd.DataFrame, list]:
    subjects = (BASELINE / "dev_subjects.txt").read_text(encoding="utf-8").split()
    if len(subjects) != 320 or len(set(subjects)) != 320:
        raise ValueError("Expected the original 320-patient development manifest")
    frame = pd.read_parquet(BASELINE / "predictions.parquet", columns=[*KEYS, LABEL, "g20__epoch"])
    if set(frame.subject_id) != set(subjects) or frame.duplicated(KEYS).any():
        raise ValueError("Saved predictions and development patient/epoch keys do not agree")
    splits = list(outer_splits(frame, folds=5))
    expected = pd.read_csv(BASELINE / "folds.csv").query("config == 'g20' and decoder == 'epoch-only'")
    actual = [f1_score(frame.iloc[s.test_idx][LABEL], frame.iloc[s.test_idx]["g20__epoch"], average="macro") for s in splits]
    if len(splits) != 5 or not np.allclose(actual, expected.sort_values("fold").macro_f1, atol=1e-10, rtol=0):
        raise ValueError("Reconstructed fold assignments do not reproduce saved Model C fold scores")
    return frame, splits


def outer_splits(frame: pd.DataFrame, folds: int):
    return build_subject_eval_splits(
        groups=frame.subject_id.to_numpy(), y=frame[LABEL].to_numpy(),
        outer_validation="kfold", outer_folds=folds, random_state=SEED,
    )


def fold_frames(frame: pd.DataFrame, split) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_train = frame.iloc[split.train_idx]
    inner = next(iter(outer_splits(outer_train, folds=10)))
    train, val = outer_train.iloc[inner.train_idx], outer_train.iloc[inner.test_idx]
    test = frame.iloc[split.test_idx]
    patient_sets = [set(part.subject_id) for part in (train, val, test)]
    if any(a & b for i, a in enumerate(patient_sets) for b in patient_sets[i + 1:]):
        raise ValueError("A patient appears in multiple partitions")
    return train, val, test


def write_predictions(path: Path, test: pd.DataFrame, probabilities: np.ndarray, fold: int) -> None:
    if (
        probabilities.shape != (len(test), len(STAGES))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError(f"Invalid probabilities for fold {fold}")
    predicted = test[[*KEYS, LABEL]].copy()
    predicted["fold"] = fold
    predicted[PROBABILITIES] = probabilities
    tmp = path.with_suffix(".part")
    predicted.to_parquet(tmp, index=False)
    tmp.replace(path)


def report(run_dir: Path, frame: pd.DataFrame, splits: list, variant: str) -> dict:
    from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions

    paths = [run_dir / f"fold-{fold}" / "predictions.parquet" for fold in range(len(splits))]
    aligned = frame.copy()
    aligned["fold"] = -1
    for fold, split in enumerate(splits):
        aligned.iloc[split.test_idx, aligned.columns.get_loc("fold")] = fold
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    aligned = aligned.merge(predictions, how="outer", on=[*KEYS, LABEL, "fold"], validate="one_to_one", indicator=True)
    if len(aligned) != len(frame) or not (aligned["_merge"] == "both").all():
        raise ValueError("Predictions do not cover the original labeled epoch and fold keys exactly")
    aligned["prediction"] = aligned[PROBABILITIES].to_numpy().argmax(axis=1)
    fold_rows = [np.flatnonzero(aligned.fold.to_numpy() == k) for k in range(len(splits))]
    baseline, candidate = aligned["g20__epoch"].to_numpy(), aligned["prediction"].to_numpy()
    y, subjects = aligned[LABEL].to_numpy(), aligned.subject_id.to_numpy()
    baseline_scores, candidate_scores = scores(aligned, baseline, fold_rows), scores(aligned, candidate, fold_rows)
    fold_deltas = [
        float(f1_score(y[rows], candidate[rows], average="macro") - f1_score(y[rows], baseline[rows], average="macro"))
        for rows in fold_rows
    ]
    delta = float(np.mean(fold_deltas))
    stage_deltas = {stage: candidate_scores[f"f1_{stage}"] - baseline_scores[f"f1_{stage}"] for stage in STAGES}
    cm_baseline, ids = patient_confusions(subjects, y, baseline)
    cm_candidate, _ = patient_confusions(subjects, y, candidate)
    fold_by_subject = aligned.groupby("subject_id").fold.first().loc[ids].to_numpy()
    patient_folds = [np.flatnonzero(fold_by_subject == k) for k in range(len(splits))]
    draws = cv_fold_draws(patient_folds, 2000, np.random.default_rng(SEED))
    boot = cv_scores(cm_candidate, patient_folds, draws)["macro_f1"] - cv_scores(cm_baseline, patient_folds, draws)["macro_f1"]
    advanced = (
        delta >= -0.005 and min(fold_deltas) >= -0.005 and min(stage_deltas["N1"], stage_deltas["N2"]) >= -0.010
        if variant == "reduction"
        else delta >= 0.010 and sum(d > 0 for d in fold_deltas) >= 4 and stage_deltas["N1"] >= -0.010
    )
    runtime = [
        json.loads(path.with_name("runtime.json").read_text(encoding="utf-8"))
        for path in paths if path.with_name("runtime.json").is_file()
    ]
    result = {
        "variant": variant,
        "patients": len(ids),
        "scored_epochs": len(aligned),
        "baseline": baseline_scores,
        "candidate": candidate_scores,
        "macro_f1_delta": delta,
        "macro_f1_paired_ci": interval(boot),
        "fold_deltas": fold_deltas,
        "stage_deltas": stage_deltas,
        "passes_development_rule": advanced,
        "fold_runtime": runtime,
    }
    aligned.drop(columns="_merge").to_parquet(run_dir / "predictions.parquet", index=False)
    atomic_json(run_dir / "summary.json", result)
    log(f"{variant}: {candidate_scores['macro_f1']:.4f} macro-F1, paired delta {delta:+.4f}, passes={advanced}")
    return result


def execute(args: argparse.Namespace) -> int:
    frame, splits = development_frame()
    variant = args.variant
    branch = REDUCTION if variant == "reduction" else WAVEFORM
    declaration = branch / ("MASKED.md" if variant == "masked" else "FINDINGS.md")
    if not declaration.is_file():
        raise FileNotFoundError(f"Predeclare the experiment in {declaration} before running it")
    if variant in {"profile", "smoke"}:
        return (profile if variant == "profile" else smoke)(frame, splits[0], args)

    run_dir = branch / variant
    partitions = []
    for split in splits:
        train, val, test = fold_frames(frame, split)
        if variant == "reduction":
            train, val = frame.iloc[split.train_idx], frame.iloc[:0]
        partitions.append({name: sorted(part.subject_id.unique()) for name, part in (
            ("train", train), ("validation", val), ("evaluation", test),
        )})
    with (BASELINE / "predictions.parquet").open("rb") as saved:
        baseline_digest = hashlib.file_digest(saved, "sha256").hexdigest()
    manifest = {
        "variant": variant, "stage_order": STAGES, "seed": SEED,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "python": platform.python_version(),
        "baseline_sha256": baseline_digest,
        "device": args.device, "batch_size": args.batch_size,
        "folds": partitions,
    }
    if variant != "reduction":
        package_dir = REPO / "src/physionet_sleep/experiments"
        sources = [Path(__file__), *(package_dir / f"waveform_{name}.py" for name in ("clock", "data", "resnet", "training"))]
        manifest["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        manifest["dependencies"] = {name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "pyarrow")}
    pin_manifest(run_dir / "manifest.json", manifest)

    active_file = EXPLORATION / "cinc-waveform-active-budget.json"
    state = json.loads(active_file.read_text(encoding="utf-8")) if active_file.exists() else {"elapsed_seconds": 0.0}
    started = active_clock()
    limit = min(BUDGET_SECONDS, args.budget_hours * 3600)
    try:
        if variant != "reduction":
            from physionet_sleep.experiments.waveform_data import WaveformStore
            from physionet_sleep.experiments.waveform_resnet import WaveformResNet
            from physionet_sleep.experiments.waveform_training import fit_waveform_fold

            store = WaveformStore(RAW, permitted_subjects=set(frame.subject_id))
        for fold, split in enumerate(splits):
            fold_dir = run_dir / f"fold-{fold}"
            output = fold_dir / "predictions.parquet"
            if output.exists():
                log(f"{variant} fold {fold + 1}/5: saved predictions found")
                continue
            remaining = limit - state["elapsed_seconds"] - (active_clock() - started)
            pause_file = run_dir / "PAUSE"
            if remaining <= 0 or pause_file.exists():
                log(f"{variant}: paused before fold {fold + 1}/5")
                return 0
            fold_dir.mkdir(parents=True, exist_ok=True)
            train, val, test = fold_frames(frame, split)
            log(f"{variant} fold {fold + 1}/5: starting")
            if variant == "reduction":
                from physionet_sleep.experiments.modern_tabular import fit_feature_reduction_fold

                probabilities, metadata = fit_feature_reduction_fold(
                    CACHE, frame.iloc[split.train_idx], test, *adopted_signals(), seed=SEED,
                )
                atomic_json(fold_dir / "runtime.json", metadata)
            else:
                probabilities = fit_waveform_fold(
                    store, WaveformResNet, train, val, test,
                    pretrained=variant == "masked", output_dir=fold_dir,
                    device=args.device, batch_size=args.batch_size, seed=SEED + fold,
                    time_limit_seconds=remaining, pause_file=pause_file,
                )
                if probabilities is None:
                    log(f"{variant}: checkpoint saved in fold {fold + 1}/5")
                    return 0
            write_predictions(output, test, probabilities, fold)
            log(f"{variant} fold {fold + 1}/5: predictions saved")
        report(run_dir, frame, splits, variant)
    finally:
        state["elapsed_seconds"] += active_clock() - started
        atomic_json(active_file, state)
    return 0


def profile(frame: pd.DataFrame, split, args: argparse.Namespace) -> int:
    from physionet_sleep.experiments.waveform_data import WaveformStore
    from physionet_sleep.experiments.waveform_resnet import WaveformResNet
    from physionet_sleep.experiments.waveform_training import profile_waveform

    train, val, _ = fold_frames(frame, split)
    store = WaveformStore(RAW, permitted_subjects=set(train.subject_id))
    result = profile_waveform(store, WaveformResNet, train, device=args.device, batch_size=args.batch_size)
    result["parameter_count"] = sum(p.numel() for p in WaveformResNet().parameters())
    result["rough_scratch_cv_seconds"] = (
        5000 * result["supervised_step_seconds"]
        + (20 * len(val) + len(frame)) * result["inference_epoch_seconds"]
    )
    atomic_json(WAVEFORM / f"profile-{args.device}-{args.batch_size}.json", result)
    print(json.dumps(result, indent=2))
    return 0


def smoke(frame: pd.DataFrame, split, args: argparse.Namespace) -> int:
    import tempfile
    import torch

    from physionet_sleep.experiments.waveform_data import WaveformStore
    from physionet_sleep.experiments.waveform_resnet import WaveformResNet
    from physionet_sleep.experiments.waveform_training import fit_waveform_fold

    train, _, _ = fold_frames(frame, split)
    patients = [group.iloc[:6] for _, group in islice(train.groupby("subject_id", sort=False), 3)]
    store = WaveformStore(RAW, permitted_subjects={str(part.subject_id.iloc[0]) for part in patients})
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    model = WaveformResNet().to(device)
    items = [store.window(row.subject_id, int(row.epoch_index)) for row in patients[0].itertuples()]
    batch = [torch.as_tensor(np.stack([item[key] for item in items]), device=device) for key in ("x", "valid", "channels")]
    labels = torch.tensor(patients[0][LABEL].to_numpy(), dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(*batch), labels)
        if step == 0:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        last_loss = torch.nn.functional.cross_entropy(model(*batch), labels).item()
    if not np.isfinite(last_loss) or last_loss >= first_loss * 0.8:
        raise RuntimeError("The ResNet failed the training-only overfit check")

    with tempfile.TemporaryDirectory(prefix="waveform-smoke-") as temp:
        temp_dir = Path(temp)
        options = dict(pretrained=False, device=args.device, batch_size=2, seed=SEED, supervised_steps=4)
        continuous = fit_waveform_fold(store, WaveformResNet, *patients, output_dir=temp_dir / "continuous", **options)
        resumed_dir = temp_dir / "resumed"
        pause_file = temp_dir / "PAUSE"

        class PauseAfterBatch:
            epochs = store.epochs
            windows = 0

            def window(self, subject, epoch):
                window = store.window(subject, epoch)
                self.windows += 1
                if self.windows == options["batch_size"]:
                    pause_file.touch()
                return window

        partial = fit_waveform_fold(
            PauseAfterBatch(), WaveformResNet, *patients, output_dir=resumed_dir, pause_file=pause_file, **options,
        )
        checkpoint = torch.load(resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        if partial is not None or checkpoint["step"] != 1:
            raise RuntimeError("Expected a durable pause after the first optimizer update")
        pause_file.unlink()
        resumed = fit_waveform_fold(store, WaveformResNet, *patients, output_dir=resumed_dir, **options)
        np.testing.assert_allclose(resumed, continuous, atol=1e-5, rtol=1e-5)
    result = {"overfit_initial_loss": first_loss, "overfit_final_loss": last_loss,
              "resume_max_probability_error": float(np.abs(resumed - continuous).max())}
    atomic_json(WAVEFORM / f"smoke-{args.device}.json", result)
    print(json.dumps(result, indent=2))
    return 0


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("reduction", "smoke", "profile", "scratch", "masked"))
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--budget-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.budget_hours <= 24:
        parser.error("batch size must be positive and the active budget must be within 24 hours")
    with local_training_slot():
        return execute(args)


if __name__ == "__main__":
    sys.exit(run())
