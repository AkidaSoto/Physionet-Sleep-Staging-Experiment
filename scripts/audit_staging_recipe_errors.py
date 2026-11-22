"""Compare archived Boston development errors and fixed class-prior sensitivity."""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from run_waveform_development import EXPLORATION, KEYS, LABEL, PROBABILITIES, SEED, STAGES, atomic_json, development_frame


OUT = EXPLORATION / "cinc-error-interactions"
EPOCH_KEYS = [*KEYS, LABEL, "fold"]
CLASSES = np.arange(len(STAGES))
PATHS = {
    "baseline": "cinc-reduction18/reduction/predictions.parquet",
    "searched": "cinc-recipe-search/searched_predictions.parquet",
    "fixed": "cinc-recipe-search/fixed_context_predictions.parquet",
    "sequence": "cinc-feature-context/long_predictions.parquet",
    "waveform": "cinc-waveform/scratch/predictions.parquet",
}
COMPARISONS = (
    ("baseline", "searched"),
    ("baseline", "fixed"),
    ("fixed", "searched"),
    ("baseline", "blend"),
    ("baseline", "waveform"),
)


def load_predictions() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    frame, splits = development_frame()
    membership = np.full(len(frame), -1, dtype=np.int8)
    frequencies = []
    for fold, split in enumerate(splits):
        if np.any(membership[split.test_idx] != -1):
            raise ValueError("The test folds overlap")
        membership[split.test_idx] = fold
        y = frame.iloc[split.train_idx][LABEL].to_numpy()
        frequencies.append(np.bincount(y, minlength=len(STAGES)) / len(y))
    if len(frame) != 288923 or np.any(membership < 0):
        raise ValueError("Unexpected development keys or uncovered test epochs")
    frame = frame.assign(fold=membership)[EPOCH_KEYS].sort_values(KEYS).reset_index(drop=True)

    probabilities = {}
    for name, path in PATHS.items():
        p = pd.read_parquet(EXPLORATION / path, columns=[*EPOCH_KEYS, *PROBABILITIES])
        joined = frame.merge(p, on=EPOCH_KEYS, validate="one_to_one")
        result = joined[PROBABILITIES].to_numpy(dtype=np.float64)
        if len(joined) != len(frame) or len(p) != len(frame):
            raise ValueError(f"Archived prediction keys differ for {name}")
        if not (np.isfinite(result).all() and (result >= 0).all() and np.allclose(result.sum(1), 1, atol=1e-6)):
            raise ValueError(f"Invalid archived probabilities for {name}")
        probabilities[name] = result
    probabilities["blend"] = 0.5 * (probabilities["baseline"] + probabilities["sequence"])
    return frame, probabilities, np.stack(frequencies)


def previous_link(frame: pd.DataFrame) -> np.ndarray:
    subject = frame["subject_id"].to_numpy()
    epoch = frame["epoch_index"].to_numpy()
    return np.r_[False, (subject[1:] == subject[:-1]) & (np.diff(epoch) == 1)]


def run_lengths(start_of_run: np.ndarray) -> np.ndarray:
    index = np.cumsum(start_of_run) - 1
    return np.bincount(index)[index]


def temporal_masks(frame: pd.DataFrame, adjacent: np.ndarray) -> dict[str, np.ndarray]:
    y = frame[LABEL].to_numpy()
    same_stage = adjacent & np.r_[False, y[1:] == y[:-1]]
    transition = adjacent & ~same_stage
    boundary = transition | np.r_[transition[1:], False]
    links = same_stage[2:] & same_stage[1:-1]
    interior = np.r_[False, False, links] & np.r_[links, False, False]
    return {"boundary": boundary, "interior": interior}


def paired_outcome(y: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (before != y) & (after == y),
            (before == y) & (after != y),
            (before != y) & (after != y) & (before == after),
            (before != y) & (after != y),
        ],
        ["repaired", "regressed", "persistent_same", "persistent_shifted"],
        default="both_correct",
    )


def summarize_errors(
    frame: pd.DataFrame, probabilities: dict[str, np.ndarray], predicted: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame[LABEL].to_numpy()
    n = len(frame)
    rows = np.arange(n)
    adjacent = previous_link(frame)
    reference = temporal_masks(frame, adjacent)
    margins = {name: np.diff(np.partition(p, -2, axis=1)[:, -2:], axis=1).ravel() for name, p in probabilities.items()}
    sustained = {}
    wrong_run_lengths = {}
    for name, p in predicted.items():
        start = ~adjacent | np.r_[True, (y[1:] != y[:-1]) | (p[1:] != p[:-1])]
        sustained[name] = (p != y) & (run_lengths(start) >= 6)
        wrong = p != y
        start = ~adjacent | np.r_[True, wrong[1:] != wrong[:-1]]
        wrong_run_lengths[name] = np.where(wrong, run_lengths(start), 0)

    epochs = frame.copy()
    epochs = epochs.assign(**reference, **predicted, **{f"{k}_sustained": v for k, v in sustained.items()})
    epochs = epochs.assign(**{f"{name}_margin": value for name, value in margins.items()})
    flows, totals = [], []
    for old, new in COMPARISONS:
        before, after = predicted[old], predicted[new]
        outcomes = paired_outcome(y, before, after)
        start = ~adjacent | np.r_[True, (y[1:] != y[:-1]) | (before[1:] != before[:-1])
                                  | (after[1:] != after[:-1])]
        paired_run = run_lengths(start) >= 6
        start = ~adjacent | np.r_[True, outcomes[1:] != outcomes[:-1]]
        outcome_run_lengths = run_lengths(start)
        top_two = np.argpartition(probabilities[old], -2, axis=1)[:, -2:]
        direction = pd.DataFrame({
            "old_model": old, "new_model": new, "outcome": outcomes,
            "reference": y, "old_stage": before, "new_stage": after,
            "subject_id": frame["subject_id"], "fold": frame["fold"],
            **reference,
            "old_sustained": sustained[old], "new_sustained": sustained[new], "paired_run_6": paired_run,
            "new_wrong_run_1": wrong_run_lengths[new] == 1, "new_wrong_run_6": wrong_run_lengths[new] >= 6,
            "outcome_run_1": outcome_run_lengths == 1, "outcome_run_6": outcome_run_lengths >= 6,
            "waveform_correct": predicted["waveform"] == y, "old_truth_top_two": (top_two == y[:, None]).any(1),
            "old_margin": margins[old], "new_margin": margins[new],
            "old_reference_probability": probabilities[old][rows, y],
            "new_reference_probability": probabilities[new][rows, y],
        })
        alts = ["old_model", "new_model", "outcome"]
        counts = {key: (key, "sum") for key in (
            "boundary", "interior", "old_sustained", "new_sustained", "paired_run_6", "waveform_correct",
            "old_truth_top_two", "new_wrong_run_1", "new_wrong_run_6", "outcome_run_1", "outcome_run_6",
        )}
        metrics = {f"median_{key}": (key, "median") for key in (
            "old_margin", "new_margin", "old_reference_probability", "new_reference_probability",
        )}
        by = {"epochs": ("outcome", "size"), "patients": ("subject_id", "nunique"), **counts, **metrics}
        error_part = direction[direction["outcome"] != "both_correct"]
        flows.append(error_part.groupby([*alts, "reference", "old_stage", "new_stage"]).agg(**by).reset_index())
        totals.append(direction.groupby(alts).agg(**by).reset_index())
        epochs[f"{old}_to_{new}"] = outcomes
        if (old, new) == ("baseline", "searched"):
            epochs["paired_run_6"] = paired_run

    recent = epochs["baseline_to_searched"]
    overlaps = []
    for earlier in ("baseline_to_fixed", "baseline_to_blend", "baseline_to_waveform"):
        old = epochs[earlier]
        table = pd.crosstab(recent, old).stack().rename("epochs").reset_index()
        table.columns = ["searched_outcome", "earlier_outcome", "epochs"]
        overlaps.append(table.assign(earlier_comparison=earlier))
    return epochs, pd.concat(totals, ignore_index=True), pd.concat(flows, ignore_index=True), pd.concat(overlaps, ignore_index=True)


def scores(frame: pd.DataFrame, p: np.ndarray) -> dict:
    y = frame[LABEL].to_numpy()
    pred = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    folds = frame["fold"].to_numpy()
    macro = [f1_score(y[folds == f], pred[folds == f], average="macro", labels=CLASSES, zero_division=0)
             for f in sorted(set(folds))]
    return {
        "macro_f1": float(np.mean(macro)),
        "fold_macro_f1": macro,
        "log_loss": float(-np.mean(np.log(p[np.arange(len(y)), y].clip(1e-12)))),
        "brier": float(np.square(p - np.eye(len(STAGES))[y]).sum(1).mean()),
        "stages": {s: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                       "support": int(support[i]), "predicted": int((pred == i).sum())} for i, s in enumerate(STAGES)},
    }


def bias_sensitivity(frame: pd.DataFrame, p: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, dict]:
    y = frame[LABEL].to_numpy()
    labels = p.argmax(1)
    factor = np.sqrt(frequencies[frame["fold"].to_numpy()])
    weighted = p * factor
    weighted /= weighted.sum(1, keepdims=True)
    rescored = weighted.argmax(1)
    patient = frame["subject_id"].to_numpy()
    cm_a, ids_a = patient_confusions(patient, y, labels)
    cm_b, ids_b = patient_confusions(patient, y, rescored)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Patient rows do not align in the prior sensitivity diagnostic")
    membership = frame.groupby("subject_id")["fold"].first().loc[ids_a].to_numpy()
    folds = [np.flatnonzero(membership == k) for k in sorted(set(membership))]
    draws = cv_fold_draws(folds, 2000, np.random.default_rng(SEED))
    a = cv_scores(cm_a, folds, draws)["macro_f1"]
    b = cv_scores(cm_b, folds, draws)["macro_f1"]
    changed = labels != rescored
    return weighted, {"changed": int(changed.sum()), "repairs": int((changed & (rescored == y)).sum()),
                      "regressions": int((changed & (labels == y)).sum()), "macro_f1_delta_ci95": interval(b - a)}


def main() -> int:
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("The archived analysis must be preregistered in FINDINGS.md")
    frame, probabilities, frequencies = load_predictions()
    predicted = {name: p.argmax(1) for name, p in probabilities.items()}
    epochs, totals, directions, overlaps = summarize_errors(frame, probabilities, predicted)
    model_scores = {name: scores(frame, p) for name, p in probabilities.items()}
    sensitivity = {}
    for model in ("baseline", "searched"):
        weighted, sensitivity[model] = bias_sensitivity(frame, probabilities[model], frequencies)
        model_scores[f"{model}_half_prior"] = scores(frame, weighted)
        epochs[f"{model}_half_prior"] = weighted.argmax(1)

    correction = paired_outcome(
        frame[LABEL].to_numpy(), epochs["searched"].to_numpy(), epochs["searched_half_prior"].to_numpy(),
    )
    sensitivity["searched"]["outcomes_on_baseline_search_comparison"] = (
        pd.crosstab(epochs["baseline_to_searched"], correction).to_dict(orient="index")
    )
    prior_cases = epochs[[LABEL, "searched", "searched_half_prior", "baseline_to_searched"]].assign(
        prior_outcome=correction,
    )
    prior_cases = prior_cases[prior_cases["searched"] != prior_cases["searched_half_prior"]]
    prior_directions = prior_cases.value_counts().rename("epochs").reset_index()

    patient = epochs[["subject_id", "fold", "baseline_to_searched"]].copy()
    risks = patient["baseline_to_searched"]
    patient = patient.assign(repaired=risks.eq("repaired"), regressed=risks.eq("regressed"),
                             persistent=risks.isin(["persistent_same", "persistent_shifted"]))
    patient_table = patient.groupby(["fold", "subject_id"]).agg(
        epochs=("baseline_to_searched", "size"), repaired=("repaired", "sum"),
        regressed=("regressed", "sum"), persistent=("persistent", "sum"),
    ).reset_index()
    y = frame[LABEL].to_numpy()
    cases = epochs["baseline_to_searched"]
    common = cases.isin(["persistent_same", "persistent_shifted"]).to_numpy()
    raw_rescue = common & (predicted["waveform"] == y)
    old_common = (predicted["baseline"] != y) & (predicted["sequence"] != y) & (predicted["blend"] != y)
    became_correct = old_common & (predicted["searched"] == y)
    tree_only_rescue = old_common & (predicted["waveform"] != y) & (predicted["searched"] == y)
    unresolved = {
        "shared_tree_errors": int(common.sum()),
        "waveform_correct_on_shared_tree_errors": int(raw_rescue.sum()),
        "shared_tree_errors_waveform_correct_and_previous_feature_models_wrong": int((raw_rescue & old_common).sum()),
        "previous_feature_model_shared_errors_repaired_by_searched": int(became_correct.sum()),
        "previous_feature_and_waveform_shared_errors_repaired_by_searched": int(tree_only_rescue.sum()),
        "previous_feature_model_shared_errors": int(old_common.sum()),
    }
    epochs.to_parquet(OUT / "archived_error_epochs.parquet", index=False)
    totals.to_csv(OUT / "archived_error_totals.csv", index=False)
    directions.to_csv(OUT / "archived_error_directions.csv", index=False)
    overlaps.to_csv(OUT / "archived_previous_overlap.csv", index=False)
    patient_table.to_csv(OUT / "archived_patient_errors.csv", index=False)
    prior_directions.to_csv(OUT / "archived_prior_directions.csv", index=False)
    prior_table = [{"model": name, "stage": stage, **values} for name, results in model_scores.items()
                   for stage, values in results["stages"].items()]
    pd.DataFrame(prior_table).to_csv(OUT / "archived_prior_stage.csv", index=False)
    atomic_json(OUT / "archived_error_summary.json", {
        "epochs": len(frame), "patients": frame["subject_id"].nunique(), "stages": STAGES,
        "scores": model_scores, "half_prior_changed": sensitivity, "fold_training_stage_frequencies": frequencies.tolist(),
        "shared_errors": unresolved, "error_totals": totals.to_dict("records"),
    })
    print(f"Archived audit complete for {len(frame):,} Boston development epochs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
