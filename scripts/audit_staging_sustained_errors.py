"""Profile sustained, same-rival Boston development errors using frozen predictions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from audit_staging_recipe_errors import EPOCH_KEYS, PATHS, previous_link
from audit_staging_regression_features import FEATURES as ARCHIVED_FEATURES
from physionet_sleep.algorithms.eeg_quality import bad_seconds
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    LABEL,
    PROBABILITIES,
    STAGES,
    atomic_json,
)

OUT = EXPLORATION / "cinc-gated-temporal"
PREVIOUS = EXPLORATION / "cinc-error-interactions"
SIGNALS = EXPLORATION / "cinc-cap-audit/feature_failure_epochs.parquet"
WAVEFORM_CACHE = EXPLORATION / "cinc-feature-tuning/waveform_cache"
FEATURE_MODELS = ("sequence", "blend", "pairwise_ebm")
PAIRS = [(2, 3), (2, 1), (3, 2), (1, 2), (1, 0), (0, 1), (4, 1), (1, 4)]
TRAJECTORY_FLANK = 10
WAVE_COLUMNS = [
    *(f"slow_wave_{v}_{lead}" for v in (60, 75, 90) for lead in ("mean", "best_clean")),
    "eog_short_4s_q90_unguarded",
    "eog_short_4s_q90_frontal2",
]
EEG_LEADS = ["F3", "F4", "C3", "C4"]
QUALITY_COLUMNS = [
    "second",
    *(f"{lead}_{q}" for lead in EEG_LEADS for q in ("flat", "line", "ecg_corr", "ptp")),
    *(f"{lead}_{band}" for lead in ("C3", "C4") for band in ("sigma", "alpha", "beta")),
]
STRATA = ["subject_id", "quarter"]


def read_archives() -> pd.DataFrame:
    names = ["baseline", "searched", "sequence", "blend", "waveform"]
    pred = pd.read_parquet(
        PREVIOUS / "archived_error_epochs.parquet",
        columns=EPOCH_KEYS
        + names
        + ["paired_run_6", "baseline_margin", "searched_margin"],
    )
    ebm = pd.read_parquet(
        PREVIOUS / "glassbox_pairwise_predictions.parquet",
        columns=EPOCH_KEYS + PROBABILITIES,
    )
    ebm["pairwise_ebm"] = ebm[PROBABILITIES].to_numpy().argmax(1)
    measurements = pd.read_parquet(
        SIGNALS,
        columns=EPOCH_KEYS + ["quarter", "ref_position", *ARCHIVED_FEATURES],
    )
    result = pred.merge(
        ebm[EPOCH_KEYS + ["pairwise_ebm"]], on=EPOCH_KEYS, validate="one_to_one"
    )
    result = result.merge(measurements, on=EPOCH_KEYS, validate="one_to_one")
    if (
        len(result) != len(pred)
        or len(result) != len(ebm)
        or len(result) != len(measurements)
    ):
        raise ValueError(
            "Expected identical label-scored keys in the saved development data"
        )
    return result.sort_values(["subject_id", "epoch_index"]).reset_index(drop=True)


def reference_and_error_runs(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame[LABEL].to_numpy()
    baseline, searched = frame["baseline"].to_numpy(), frame["searched"].to_numpy()
    adjacent = previous_link(frame)
    reference_start = ~adjacent | np.r_[True, y[1:] != y[:-1]]
    reference_first = np.flatnonzero(reference_start)
    reference_last = np.r_[reference_first[1:] - 1, len(frame) - 1]
    segment = np.cumsum(reference_start) - 1
    previous = np.full(len(reference_first), -1, dtype=np.int8)
    following = previous.copy()
    starts_at_change = adjacent[reference_first]
    ends_at_change = np.r_[adjacent[reference_last[:-1] + 1], False]
    previous[starts_at_change] = y[reference_first[starts_at_change] - 1]
    following[ends_at_change] = y[reference_last[ends_at_change] + 1]

    model_change = (baseline[1:] != baseline[:-1]) | (searched[1:] != searched[:-1])
    error_start = reference_start | np.r_[True, model_change]
    error_run = np.cumsum(error_start) - 1
    length = np.bincount(error_run)[error_run]
    sustained = (baseline == searched) & (baseline != y) & (length >= 6)
    if not np.array_equal(
        sustained,
        frame["paired_run_6"].to_numpy() & (baseline == searched) & (baseline != y),
    ):
        raise ValueError(
            "Independent reconstruction of the paired error runs disagrees with experiment 54"
        )
    positions = np.arange(len(frame))
    return frame.assign(
        reference_bout=segment,
        from_reference_start=positions - reference_first[segment],
        from_reference_end=reference_last[segment] - positions,
        previous_reference=previous[segment],
        next_reference=following[segment],
        sustained_id=np.where(sustained, error_run, -1),
        strict_sustained=sustained,
    )


def sigma_hits(quality: pd.DataFrame, lead: str, usable: np.ndarray) -> np.ndarray:
    p = quality[[f"{lead}_{band}" for band in ("sigma", "alpha", "beta")]].to_numpy()
    ratio = p[:, 0] / (p[:, 1] + p[:, 2] + 1e-6)
    good = usable & np.isfinite(ratio)
    if not good.any():
        return np.zeros(len(ratio), dtype=bool)
    typical = np.median(ratio[good])
    robust_sigma = 1.4826 * np.median(np.abs(ratio[good] - typical))
    return good & (ratio > typical + 2 * robust_sigma)


def max_consecutive(rows: np.ndarray) -> np.ndarray:
    age = np.zeros(len(rows), dtype=np.int16)
    best = age.copy()
    for present in rows.T:
        age = np.where(present, age + 1, 0)
        best = np.maximum(best, age)
    return best


def extra_measurements(subject: str, epochs: np.ndarray) -> pd.DataFrame:
    raw = pd.read_parquet(WAVEFORM_CACHE / f"{subject}.parquet", columns=WAVE_COLUMNS)
    if not raw.index.equals(
        pd.RangeIndex(len(raw), name="epoch_index")
    ) or epochs.max() >= len(raw):
        raise ValueError(
            f"Unexpected epoch grid in the cached waveform measurements for {subject}"
        )
    quality = pd.read_parquet(
        CACHE / f"quality1s/{subject}.parquet", columns=QUALITY_COLUMNS
    )
    if not np.array_equal(quality.second.to_numpy(), np.arange(len(quality))):
        raise ValueError(
            f"Unexpected second grid in the cached spectral measurements for {subject}"
        )
    # Spectral estimates are sampled once per second over overlapping two-second frames.
    # These bins measure activity and persistence, without independent event markings.
    bad = {lead: bad_seconds(quality, lead) for lead in EEG_LEADS}
    first, last = epochs * 30, (epochs + 1) * 30 - 1
    seconds = first[:, None] + np.arange(30)
    if seconds.max() >= len(quality):
        raise ValueError(f"Epoch exceeds the waveform quality recording for {subject}")
    samples = [sigma_hits(quality, lead, ~bad[lead]) for lead in ("C3", "C4")]
    any_hit = np.logical_or(*samples)
    same_time_hit = np.logical_and(*samples)
    bins = any_hit[seconds]
    jointly_observed = ~bad["C3"] & ~bad["C4"]
    some_observed = ~bad["C3"] | ~bad["C4"]
    closest = np.flatnonzero(any_hit)
    prev_idx = np.searchsorted(closest, first, side="left") - 1
    next_idx = np.searchsorted(closest, last, side="right")
    prev_sec = np.r_[-1_000_000, closest][prev_idx + 1]
    next_sec = np.r_[closest, 1_000_000_000][next_idx]
    derived = {
        "one_or_more_clean_central_leads": some_observed[seconds].mean(1),
        "one_or_more_clean_frontal_leads": (~bad["F3"] | ~bad["F4"])[seconds].mean(1),
        "earliest_six_sigma_bins": bins[:, :6].sum(1),
        "latest_six_sigma_bins": bins[:, -6:].sum(1),
        "clean_central_sigma_bins": bins.sum(1),
        "clean_central_longest_sigma_run": max_consecutive(bins),
        "bilateral_central_sigma_bins": np.where(
            jointly_observed[seconds].any(1), same_time_hit[seconds].sum(1), np.nan
        ),
        "seconds_since_prior_sigma_bin_capped600": np.minimum(first - prev_sec, 600),
        "seconds_until_next_sigma_bin_capped600": np.minimum(next_sec - last, 600),
    }
    for key in (
        "earliest_six_sigma_bins",
        "latest_six_sigma_bins",
        "clean_central_sigma_bins",
        "clean_central_longest_sigma_run",
    ):
        derived[key] = np.where(some_observed[seconds].any(1), derived[key], np.nan)
    extras = raw.reindex(epochs).reset_index(drop=True)
    return pd.concat((extras, pd.DataFrame(derived, dtype=np.float32)), axis=1)


def load_sustained_patients(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame.loc[frame.strict_sustained, "subject_id"].unique()
    parts = []
    for subject, night in frame.loc[frame.subject_id.isin(keep)].groupby(
        "subject_id", sort=False
    ):
        extras = extra_measurements(subject, night.epoch_index.to_numpy())
        parts.append(pd.concat((night.reset_index(drop=True), extras), axis=1))
    return pd.concat(parts, ignore_index=True)


def placement(cohort: np.ndarray, rivals: np.ndarray) -> float:
    """Estimate the fraction of case–rival pairs where the case value is higher."""
    return float(
        (
            np.searchsorted(rivals, cohort, side="left")
            + np.searchsorted(rivals, cohort, side="right")
        ).mean()
        / (2 * len(rivals))
    )


def within_cells(
    frame: pd.DataFrame, features: list[str], pair: tuple[int, int]
) -> pd.DataFrame:
    truth, rival = pair
    correct = frame.baseline.eq(frame[LABEL]) & frame.searched.eq(frame[LABEL])
    cases = frame.strict_sustained & frame[LABEL].eq(truth) & frame.baseline.eq(rival)
    keep = frame.ref_position.eq("interior") & (
        cases | (correct & frame[LABEL].isin(pair))
    )
    result = []
    for (subject, quarter), context in frame.loc[keep].groupby(STRATA, sort=False):
        reference = context[LABEL].to_numpy()
        sustained = context.strict_sustained.to_numpy()
        rescued = context.waveform.eq(truth).to_numpy()
        control_true = (reference == truth) & ~sustained
        control_rival = (reference == rival) & ~sustained
        if min(control_true.sum(), control_rival.sum()) < 3 or not sustained.any():
            continue
        masks = {
            "correct_true": control_true,
            "correct_rival": control_rival,
            "sustained": sustained,
            "waveform_correct": sustained & rescued,
            "waveform_wrong": sustained & ~rescued,
        }
        for feature in features:
            x = context[feature].to_numpy()
            cohorts = {k: x[m & np.isfinite(x)] for k, m in masks.items()}
            if min(
                len(cohorts["correct_true"]), len(cohorts["correct_rival"])
            ) < 3 or not len(cohorts["sustained"]):
                continue
            controls = np.sort(cohorts["correct_rival"])
            row = {
                "subject_id": subject,
                "quarter": quarter,
                "fold": int(context.fold.iloc[0]),
                "feature": feature,
                "truth": STAGES[truth],
                "rival": STAGES[rival],
            }
            for name, a in cohorts.items():
                row[f"{name}_n"] = len(a)
                row[f"{name}_auc"] = placement(a, controls) if len(a) else np.nan
                row[f"{name}_mean"] = a.mean() if len(a) else np.nan
            result.append(row)
    return pd.DataFrame(result)


def bootstrap_patient_difference(differences: pd.Series) -> tuple[float, float]:
    values = differences.to_numpy()
    if len(values) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(57)
    boot = values[rng.integers(0, len(values), (1000, len(values)))].mean(1)
    return tuple(np.quantile(boot, (0.025, 0.975)))


def aggregate_cells(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (truth, rival, feature), group in cells.groupby(
        ["truth", "rival", "feature"], sort=False
    ):
        views = {
            "all": (group, "sustained", "correct_true"),
            "waveform_split": (
                group.loc[
                    group.waveform_correct_n.ge(2) & group.waveform_wrong_n.ge(2)
                ],
                "waveform_correct",
                "waveform_wrong",
            ),
        }
        for contrast, (sample, a, b) in views.items():
            if sample.empty:
                continue
            patient = sample.groupby("subject_id", sort=False).mean(numeric_only=True)
            sign = 1 if patient.correct_true_auc.mean() >= 0.5 else -1
            differences = sign * (patient[f"{a}_auc"] - patient[f"{b}_auc"])
            low, high = bootstrap_patient_difference(differences)
            row = {
                "truth": truth,
                "rival": rival,
                "feature": feature,
                "contrast": contrast,
                "matched_patients": len(patient),
                "matched_cells": len(sample),
                "truth_supported_by": "higher" if sign > 0 else "lower",
                "auc_difference_a_minus_b": differences.mean(),
                "paired_low": low,
                "paired_high": high,
            }
            for cohort in ("sustained", "correct_true", "correct_rival", a, b):
                row[f"{cohort}_n"] = int(sample[f"{cohort}_n"].sum())
                auc = patient[f"{cohort}_auc"].mean()
                row[f"{cohort}_auc"] = auc if sign > 0 else 1 - auc
                row[f"{cohort}_mean"] = patient[f"{cohort}_mean"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def stage_agreement(cases: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(
        np.select(
            (cases[column].eq(cases[LABEL]), cases[column].eq(cases.baseline)),
            ("correct_reference", "same_wrong_rival"),
            default="different_stage",
        ),
        index=cases.index,
    )


def summarize_bouts(
    sustained: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = sustained.groupby("sustained_id", sort=False)
    bouts = grouped.agg(
        subject_id=("subject_id", "first"),
        fold=("fold", "first"),
        start_epoch=("epoch_index", "first"),
        end_epoch=("epoch_index", "last"),
        length=("epoch_index", "size"),
        truth=(LABEL, "first"),
        rival=("baseline", "first"),
        gap_after_reference_start=("from_reference_start", "first"),
        gap_before_reference_end=("from_reference_end", "last"),
        previous_reference=("previous_reference", "first"),
        next_reference=("next_reference", "last"),
        baseline_median_margin=("baseline_margin", "median"),
    ).reset_index()
    for model in (*FEATURE_MODELS, "waveform"):
        outcomes = stage_agreement(sustained, model)
        proportions = pd.crosstab(sustained.sustained_id, outcomes, normalize="index")
        for outcome in ("correct_reference", "same_wrong_rival", "different_stage"):
            bouts[f"{model}_{outcome}_share"] = bouts.sustained_id.map(
                proportions[outcome]
                if outcome in proportions
                else pd.Series(dtype=float)
            ).fillna(0.0)
    bouts["at_baseline_transition_from_rival"] = bouts.gap_after_reference_start.eq(
        0
    ) & bouts.previous_reference.eq(bouts.rival)
    bouts["at_baseline_transition_to_rival"] = bouts.gap_before_reference_end.eq(
        0
    ) & bouts.next_reference.eq(bouts.rival)
    bouts["at_least_3min_from_reference_change"] = bouts.gap_after_reference_start.ge(
        6
    ) & bouts.gap_before_reference_end.ge(6)
    counts = []
    stage_pairs = sustained.groupby([LABEL, "baseline"], sort=False)
    for (truth, rival), cases in stage_pairs:
        b = bouts.loc[bouts.truth.eq(truth) & bouts.rival.eq(rival)]
        row = {
            "truth": STAGES[truth],
            "rival": STAGES[rival],
            "bouts": len(b),
            "patients": cases.subject_id.nunique(),
            "epochs": len(cases),
            "median_epochs_per_bout": b.length.median(),
            "fully_interior_bouts": int(b.at_least_3min_from_reference_change.sum()),
            "epochs_at_least_3min_from_reference_change": int(
                (
                    cases.from_reference_start.ge(6) & cases.from_reference_end.ge(6)
                ).sum()
            ),
            "bouts_touching_reference_change_from_rival": int(
                b.at_baseline_transition_from_rival.sum()
            ),
            "bouts_touching_reference_change_to_rival": int(
                b.at_baseline_transition_to_rival.sum()
            ),
            "bouts_touching_reference_change_to_or_from_rival": int(
                (
                    b.at_baseline_transition_from_rival
                    | b.at_baseline_transition_to_rival
                ).sum()
            ),
        }
        for model in (*FEATURE_MODELS, "waveform"):
            counts_by_model = stage_agreement(cases, model).value_counts()
            for outcome in ("correct_reference", "same_wrong_rival", "different_stage"):
                row[f"{model}_{outcome}"] = int(counts_by_model.get(outcome, 0))
            row[f"{model}_correct_full_bouts"] = int(
                b[f"{model}_correct_reference_share"].eq(1).sum()
            )
        counts.append(row)
    chain = np.ones(len(sustained), dtype=bool)
    for model in FEATURE_MODELS:
        chain &= sustained[model].eq(sustained.baseline).to_numpy()
    chain_frame = sustained.loc[chain].sort_values(["subject_id", "epoch_index"])
    chained = (
        previous_link(chain_frame)
        & np.r_[
            False,
            chain_frame.sustained_id.to_numpy()[1:]
            == chain_frame.sustained_id.to_numpy()[:-1],
        ]
    )
    chain_id = np.cumsum(~chained)
    chain_sizes = pd.Series(chain_id).value_counts()
    strict_all = chain_frame.loc[
        pd.Series(chain_id).map(chain_sizes).to_numpy() >= 6
    ].copy()
    concordance = (
        strict_all.groupby([LABEL, "baseline"], sort=False)
        .agg(
            epochs=(LABEL, "size"),
            patients=("subject_id", "nunique"),
        )
        .reset_index()
    )
    if not concordance.empty:
        concordance["truth"] = concordance[LABEL].map(dict(enumerate(STAGES)))
        concordance["rival"] = concordance.baseline.map(dict(enumerate(STAGES)))
    return bouts, pd.DataFrame(counts), concordance


def representative_trajectories(frame: pd.DataFrame, bouts: pd.DataFrame) -> dict:
    """Choose complete matched-length N2→N3 examples without inspecting signals."""
    candidates = bouts.loc[bouts.truth.eq(2) & bouts.rival.eq(3)].copy()
    broad = frame.set_index(["subject_id", "epoch_index"])
    if not broad.index.is_unique:
        raise ValueError(
            "Representative trajectories require unique patient and epoch keys"
        )
    has_neighbors = []
    for bout in candidates.itertuples():
        expected = np.arange(
            bout.start_epoch - TRAJECTORY_FLANK, bout.end_epoch + TRAJECTORY_FLANK + 1
        )
        has_neighbors.append(
            pd.Index(expected).isin(broad.loc[bout.subject_id].index).all()
        )
    candidates = candidates.loc[has_neighbors]
    consensus = [
        f"{model}_same_wrong_rival_share" for model in (*FEATURE_MODELS, "waveform")
    ]
    groups = {
        "waveform_resolved": candidates.loc[
            candidates.waveform_correct_reference_share.eq(1)
        ],
        "all_models_same_rival": candidates.loc[
            candidates[consensus].eq(1).all(axis=1)
        ],
    }
    shared_lengths = set.intersection(*(set(g.length) for g in groups.values()))
    if not shared_lengths:
        raise ValueError("Representative cohorts have no common run lengths")
    median_length = pd.concat(list(groups.values())).length.median()
    length = min(shared_lengths, key=lambda n: (abs(n - median_length), n))
    figure = []
    for cohort, population in groups.items():
        common = population.loc[population.length.eq(length)]
        gap = (
            common.baseline_median_margin - common.baseline_median_margin.median()
        ).abs()
        bout = (
            common.assign(margin_gap=gap)
            .sort_values(["margin_gap", "subject_id", "start_epoch"], kind="stable")
            .iloc[0]
        )
        subject = bout.subject_id
        keys = np.arange(
            bout.start_epoch - TRAJECTORY_FLANK, bout.end_epoch + TRAJECTORY_FLANK + 1
        )
        data = (
            broad.loc[subject]
            .loc[
                keys,
                [
                    LABEL,
                    "fold",
                    "baseline",
                    "searched",
                    "waveform",
                    "frontal_delta_control",
                    "slow_wave_75_mean",
                ],
            ]
            .reset_index()
        )
        data["subject_id"] = subject
        for model in ("baseline", "searched", "waveform"):
            probs = pd.read_parquet(
                EXPLORATION / PATHS[model],
                columns=EPOCH_KEYS + ["p_N2", "p_N3"],
                filters=[("subject_id", "==", subject)],
            ).rename(
                columns={f"p_{stage}": f"{model}_p_{stage}" for stage in ("N2", "N3")}
            )
            data = data.merge(probs, on=EPOCH_KEYS, how="left", validate="one_to_one")
            if data[[f"{model}_p_N2", f"{model}_p_N3"]].isna().any().any():
                raise ValueError(
                    f"Representative trajectory has missing {model} predictions"
                )
        stage_names = dict(enumerate(STAGES))
        data = data.rename(columns={LABEL: "reference"})
        for model in ("reference", "baseline", "searched", "waveform"):
            data[model] = data[model].map(stage_names)
        data["cohort"] = cohort
        data["relative_epoch"] = data.epoch_index - bout.start_epoch
        data["relative_minutes"] = 0.5 * data.relative_epoch
        data["inside_sustained_bout"] = data.relative_epoch.between(0, length - 1)
        figure.append(data.drop(columns=["subject_id", "epoch_index", "fold"]))
    pd.concat(figure, ignore_index=True).to_csv(
        OUT / "sustained_representative_trajectories.csv",
        index=False,
        float_format="%.6f",
    )
    return {
        "eligible_bouts": {kind: len(group) for kind, group in groups.items()},
        "shared_selected_bout_length": int(length),
        "flank_epochs_on_each_side": TRAJECTORY_FLANK,
        "selection": (
            "Among complete-flank N2-to-N3 bouts, use a length shared by both groups "
            "nearest their pooled median length, giving ties to the shorter length. "
            "Select the bout nearest its same-length group's median baseline margin, "
            "with ties broken by subject and start epoch."
        ),
    }


def main():
    if not (OUT / "FINDINGS.md").exists():
        raise FileNotFoundError(
            "Declare experiment 57 before profiling the development data"
        )
    frame = reference_and_error_runs(read_archives())
    sustained = frame.loc[frame.strict_sustained].copy()
    bouts, counts, chain = summarize_bouts(sustained)
    counts.to_csv(OUT / "sustained_summary.csv", index=False)
    chain.to_csv(OUT / "sustained_all_feature_models.csv", index=False)
    bouts.to_parquet(OUT / "sustained_bouts.parquet", index=False)
    evidence = load_sustained_patients(frame)
    new_features = list(evidence.columns.difference(frame.columns))
    for amplitude in (60, 75, 90):
        for channel in ("mean", "best_clean"):
            feature = f"slow_wave_{amplitude}_{channel}"
            threshold = f"{feature}_coverage_ge20"
            x = evidence[feature]
            evidence[threshold] = np.where(x.notna(), x.ge(0.20), np.nan)
            new_features.append(threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cells = pd.concat(
            [
                within_cells(evidence, ARCHIVED_FEATURES + new_features, pair)
                for pair in PAIRS
            ],
            ignore_index=True,
        )
    aggregate_cells(cells).to_csv(
        OUT / "sustained_matched_features.csv", index=False, float_format="%.6f"
    )
    cells.to_parquet(OUT / "sustained_matched_cells.parquet", index=False)
    evidence.loc[evidence.strict_sustained].to_parquet(
        OUT / "sustained_epoch_evidence.parquet", index=False
    )
    atomic_json(
        OUT / "sustained_scope.json",
        {
            "development_epochs": len(frame),
            "development_patients": int(frame.subject_id.nunique()),
            "same_rival_sustained_epochs": len(sustained),
            "same_rival_sustained_bouts": len(bouts),
            "same_rival_sustained_patients": int(bouts.subject_id.nunique()),
            "identical_5_feature_model_sustained_epochs": int(chain.epochs.sum()),
            "representative_trajectories": representative_trajectories(evidence, bouts),
            "stage_pairs": [[STAGES[truth], STAGES[rival]] for truth, rival in PAIRS],
            "evidence_features": ARCHIVED_FEATURES + new_features,
        },
    )
    print(
        f"{len(sustained):,} sustained epochs in {len(bouts):,} bouts, {len(cells):,} feature cells"
    )


if __name__ == "__main__":
    main()
