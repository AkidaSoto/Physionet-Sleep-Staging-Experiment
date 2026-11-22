"""Experiment 74. Compare saved changes on the expanded stager's residual errors."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from audit_staging_wake_n1 import annotation_phases
from run_staging_probability_diagnostic import PHYSIOLOGY, night_history
from run_waveform_development import BASELINE, EXPLORATION, KEYS, LABEL, PROBABILITIES, STAGES, atomic_json


OUT = EXPLORATION / "cinc-combination-audit"
REFERENCE = EXPLORATION / "cinc-all-additional-training/predictions.parquet"
KEY_FOLD = [*KEYS, LABEL, "fold"]
PAIRS = ((0, 1), (1, 2), (2, 3), (1, 4), (2, 4))


def aligned(base: pd.DataFrame, path: Path, label_column: str | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if label_column is None:
        keys, columns = KEY_FOLD, PROBABILITIES
    else:
        keys = [*KEYS, LABEL] if path == BASELINE / "predictions.parquet" else KEY_FOLD
        columns = [label_column]
    source = pd.read_parquet(path, columns=[*keys, *columns])
    joined = base[keys].merge(source, on=keys, validate="one_to_one", how="left", indicator=True)
    if len(source) != len(base) or not joined._merge.eq("both").all():
        raise ValueError(f"{path}: mismatched development patient, epoch, truth or fold")
    if label_column is not None:
        if not joined[label_column].isin(range(len(STAGES))).all():
            raise ValueError(f"{path}: invalid labels in {label_column}")
        return joined[label_column].to_numpy(np.int8), None
    probability = joined[columns].to_numpy(np.float32)
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=1e-5):
        raise ValueError(f"{path}: invalid stage probabilities")
    return probability.argmax(axis=1), probability


def verify_legacy_folds(base: pd.DataFrame, old: dict[str, np.ndarray]) -> None:
    archive = pd.read_csv(BASELINE / "folds.csv")
    for name, labels in old.items():
        config, decoder = name.split("__")
        if decoder == "epoch":
            decoder = "epoch-only"
        expected = archive.loc[archive.config.eq(config) & archive.decoder.eq(decoder)].sort_values("fold").macro_f1
        actual = [f1_score(base.loc[base.fold.eq(f), LABEL], labels[base.fold.eq(f)], average="macro") for f in range(5)]
        np.testing.assert_allclose(actual, expected.to_numpy(), atol=1e-10, rtol=0)


def pair_auc(proba: np.ndarray, keep: np.ndarray, positive: np.ndarray, a: int, b: int) -> float:
    left, right = proba[keep, a], proba[keep, b]
    return float(roc_auc_score(positive[keep], np.log(left.clip(1e-12)) - np.log(right.clip(1e-12))))


def chin_continuity(base: pd.DataFrame, ranked: np.ndarray, close: np.ndarray) -> dict:
    marker = np.empty(len(base), np.float32)
    epoch = base.epoch_index.to_numpy()
    persistence_column = 3 + PHYSIOLOGY.index("atonia_fraction_10min")
    for subject, rows in base.groupby("subject_id", sort=False).indices.items():
        marker[rows] = night_history(subject)[epoch[rows], persistence_column]
    result = {}
    for other in (1, 2):
        selected = close & (ranked == [other, 4]).all(axis=1) & base[LABEL].isin([other, 4]).to_numpy()
        subset = base.loc[selected, ["subject_id", LABEL]].assign(chin=marker[selected])
        per_patient = [
            roc_auc_score(epochs[LABEL].eq(4), epochs.chin)
            for _, epochs in subset.groupby("subject_id") if epochs[LABEL].nunique() == 2
        ]
        result[f"{STAGES[other]}/REM"] = {
            "epochs": len(subset), "paired_patients": len(per_patient),
            "pooled_auc": float(roc_auc_score(subset[LABEL].eq(4), subset.chin)),
            "mean_within_patient_auc": float(np.mean(per_patient)),
        }
    return result


def main() -> None:
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the saved-prediction audit before running it")
    base = pd.read_parquet(REFERENCE, columns=[*KEY_FOLD, *PROBABILITIES]).sort_values(KEYS, ignore_index=True)
    if len(base) != 288923 or base.subject_id.nunique() != 320 or base.duplicated(KEYS).any():
        raise ValueError("Expected the complete 320-patient reference")
    if set(base.fold) != set(range(5)) or base.groupby("subject_id").fold.nunique().max() != 1:
        raise ValueError("Each patient must belong to exactly one of the five reference folds")
    base["phase"], _ = annotation_phases(base)
    awakening = base.phase.eq("later_awakening").to_numpy()
    truth = base[LABEL].to_numpy()
    p_ref = base[PROBABILITIES].to_numpy()
    reference = p_ref.argmax(axis=1)
    ranked = np.argsort(-p_ref, axis=1, kind="stable")[:, :2]
    scores = np.take_along_axis(p_ref, ranked, axis=1)
    close = scores[:, 0] <= 2 * scores[:, 1]
    pair_order = np.sort(ranked, axis=1)
    specs = {
        "window27": (BASELINE / "predictions.parquet", "nocardio27__epoch"),
        "window33": (BASELINE / "predictions.parquet", "window33__epoch"),
        "window33_hmm": (BASELINE / "predictions.parquet", "window33__hmm"),
        "tree": (EXPLORATION / "cinc-reduction18/reduction/predictions.parquet", None),
        "tuned": (EXPLORATION / "cinc-feature-tuning/joint_predictions.parquet", None),
        "long_blend": (EXPLORATION / "cinc-feature-context/paired_predictions.parquet", "long_blend"),
        "clock": (EXPLORATION / "cinc-probability-diagnostic/clock_predictions.parquet", None),
        "physiology": (EXPLORATION / "cinc-probability-diagnostic/physiology_predictions.parquet", None),
        "fixed_context": (EXPLORATION / "cinc-recipe-search/fixed_context_predictions.parquet", None),
        "searched_context": (EXPLORATION / "cinc-recipe-search/searched_predictions.parquet", None),
        "gated": (EXPLORATION / "cinc-gated-temporal/gated_predictions.parquet", None),
        "expanded": (EXPLORATION / "cinc-complementary-markers/expanded_predictions.parquet", None),
        "head": (EXPLORATION / "cinc-stage-pair-revisit/head_predictions.parquet", None),
        "k_head": (EXPLORATION / "cinc-stage-pair-revisit/k_head_predictions.parquet", None),
        "continuation": (EXPLORATION / "cinc-stage-pair-revisit/continuation_head_predictions.parquet", None),
        "reduced": (EXPLORATION / "cinc-stage-pair-revisit/reduction/predictions.parquet", None),
        "theta_timing": (EXPLORATION / "cinc-wake-n1-followup/theta_timing_predictions.parquet", None),
        "alpha_guard": (EXPLORATION / "cinc-wake-n1-followup/alpha_past_gate_predictions.parquet", None),
    }
    outputs = {name: aligned(base, path, column) for name, (path, column) in specs.items()}
    verify_legacy_folds(base, {column: outputs[name][0] for name, (path, column) in specs.items() if path == BASELINE / "predictions.parquet"})
    contrasts = {
        "23_hmm": ("window33", "window33_hmm"),
        "24_cardio": ("window27", "window33"),
        "45_tuned_bundle": ("tree", "tuned"),
        "48_blend_diagnostic": ("tree", "long_blend"),
        "51_night_physiology": ("clock", "physiology"),
        "53_wider_window": ("fixed_context", "searched_context"),
        "61_markers": ("gated", "expanded"),
        "65_n1_n2_head": ("expanded", "head"),
        "66_k_shape": ("head", "k_head"),
        "67_history_split": ("k_head", "continuation"),
        "68_reduction": ("expanded", "reduced"),
        "72_theta": ("expanded", "theta_timing"),
        "72_awakening_guard": ("expanded", "alpha_guard"),
    }

    measurements, coverage = {}, {}
    for name, (before, after) in contrasts.items():
        p_before, dist_before = outputs[before]
        p_after, dist_after = outputs[after]
        rows = {}
        for a, b in PAIRS:
            pair = f"{STAGES[a]}/{STAGES[b]}"
            eligible = (pair_order == [a, b]).all(axis=1) & np.isin(truth, [a, b])
            hard = eligible & close
            current_miss = eligible & (truth != reference)
            surviving_repairs = eligible & current_miss & (p_before == reference) & (p_after == truth)
            coverage[pair, name] = surviving_repairs
            result = {"eligible_epochs": int(eligible.sum()), "hard_epochs": int(hard.sum())}
            if dist_before is not None and dist_after is not None:
                result["hard_auc_control"] = pair_auc(dist_before, hard, truth == a, a, b)
                result["hard_auc_candidate"] = pair_auc(dist_after, hard, truth == a, a, b)
                result["hard_auc_current_reference"] = pair_auc(p_ref, hard, truth == a, a, b)
            for target, rival in ((a, b), (b, a)):
                scope = eligible & (truth == target)
                fix = scope & (p_before == rival) & (p_after == target)
                harm = scope & (p_before == target) & (p_after == rival)
                surviving_risk = harm & (reference == target)
                direction = {
                    "current_errors": int((scope & (reference == rival)).sum()),
                    "archived_direct_repairs": int(fix.sum()),
                    "archived_direct_regressions": int(harm.sum()),
                    "surviving_repairs": int((surviving_repairs & scope).sum()),
                    "surviving_risks": int(surviving_risk.sum()),
                }
                if (a, b) == (0, 1):
                    direction["later_awakening_repairs"] = int((awakening & surviving_repairs & scope).sum())
                    direction["later_awakening_risks"] = int((awakening & surviving_risk).sum())
                result[f"{STAGES[target]}_as_{STAGES[rival]}"] = direction
            rows[pair] = result
        measurements[name] = {"control": before, "candidate": after, "pairs": rows}

    overlap = {}
    for pair in (f"{STAGES[a]}/{STAGES[b]}" for a, b in PAIRS):
        overlap[pair] = {
            f"{left}__{right}": int((coverage[pair, left] & coverage[pair, right]).sum())
            for left, right in combinations(contrasts, 2)
        }
    atomic_json(OUT / "summary.json", {
        "reference": str(REFERENCE.relative_to(EXPLORATION)),
        "patients": 320, "epochs": len(base), "folds": 5,
        "contrasts": measurements, "surviving_repair_overlap": overlap,
        "chin_continuity": chin_continuity(base, pair_order, close),
    })
    print(f"Audited {len(contrasts)} archived comparisons across 288,923 keyed predictions")


if __name__ == "__main__":
    main()
