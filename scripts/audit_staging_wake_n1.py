"""Audit directional Wake/N1 errors using frozen Boston development predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit_staging_recipe_errors import EPOCH_KEYS, load_predictions, previous_link
from audit_staging_sustained_errors import placement
from physionet_sleep.io.cinc2018 import resolve_cinc2018_record
from physionet_sleep.labels.cinc2018 import EPOCH_SECONDS, load_cinc2018_annotations
from run_waveform_development import (
    CACHE,
    EXPLORATION,
    LABEL,
    PROBABILITIES,
    RAW,
    STAGES,
    atomic_json,
)


OUT = EXPLORATION / "cinc-wake-n1-data"
EXPANDED = EXPLORATION / "cinc-complementary-markers"
MARKERS = EXPLORATION / "cinc-cap-audit/feature_failure_epochs.parquet"
SUBJECT_AHI = EXPLORATION / "cinc2018/subject_summary.csv"
WAKE, N1 = 0, 1
PAIRS = ((N1, WAKE), (WAKE, N1))
COMPONENTS = [
    f"{part}_{stage}" for part in ("local", "past", "future") for stage in STAGES
]
EXISTING = [
    "occipital_rel_alpha",
    "o_rel_alpha",
    "c_rel_theta",
    "rel_theta",
    "rel_beta",
    "aperiodic_exponent",
    "eye_movement_activity",
    "eog_rem_vs_sem_db",
    "emg_floor_db",
]
DERIVED = ["alpha_late_minus_early", "alpha_epoch_change", "slow_eog_power", "chin_rms"]
FEATURES = EXISTING + DERIVED
PHASES = (
    "first_observed_sleep_onset",
    "later_return_to_sleep",
    "later_awakening",
    "other_sleep_stage_boundary",
    "stable_interior",
    "record_edge_or_gap",
    "other_scored",
)
EVENT_KINDS = {
    "arousal_spontaneous": "marked_spontaneous",
    "arousal_rera": "marked_rera",
}


def aligned(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    joined = base[EPOCH_KEYS].merge(
        other, on=EPOCH_KEYS, how="left", validate="one_to_one", indicator=True
    )
    if len(joined) != len(other) or not joined["_merge"].eq("both").all():
        raise ValueError("Archives differ on patient, epoch, reference stage or fold")
    return joined.drop(columns="_merge")


def run_bounds(start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.flatnonzero(start)
    last = np.r_[first[1:] - 1, len(start) - 1]
    run = np.cumsum(start) - 1
    i = np.arange(len(run))
    return i - first[run], last[run] - i, last[run] - first[run] + 1


def annotation_phases(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Use known adjacent labels only, with no imputed changes across gaps."""
    y = frame[LABEL].to_numpy()
    links = previous_link(frame)
    changes = np.r_[False, y[1:] != y[:-1]] & links
    from_start, to_end, _ = run_bounds(changes | ~links)
    recording = np.cumsum(~links)
    from_gap, to_gap, _ = run_bounds(~links)
    phase = np.full(len(frame), "other_scored", dtype=object)
    phase[(from_start >= 6) & (to_end >= 6)] = "stable_interior"
    phase[(from_gap < 2) | (to_gap < 2)] = "record_edge_or_gap"
    direct = np.zeros(len(frame), dtype=bool)
    nearest = np.full(len(frame), np.inf)
    first_sleep = ~frame.loc[frame[LABEL].ne(WAKE), "subject_id"].duplicated()
    first_sleep_idx = set(first_sleep.index[first_sleep])
    for boundary in np.flatnonzero(changes):
        old, new = y[boundary - 1], y[boundary]
        if old == WAKE:
            kind = (
                "first_observed_sleep_onset"
                if boundary in first_sleep_idx
                else "later_return_to_sleep"
            )
        elif new == WAKE:
            kind = "later_awakening"
        else:
            kind = "other_sleep_stage_boundary"
        neighbours = np.arange(max(0, boundary - 2), min(len(frame), boundary + 2))
        distance = np.abs(neighbours - (boundary - 0.5))
        keep = (recording[neighbours] == recording[boundary]) & (
            distance < nearest[neighbours]
        )
        phase[neighbours[keep]] = kind
        direct[neighbours[keep]] = {old, new} == {WAKE, N1}
        nearest[neighbours[keep]] = distance[keep]
    return phase, direct


def finite_mean(x: np.ndarray) -> np.ndarray:
    observed = np.isfinite(x)
    n = observed.sum(axis=1)
    return np.divide(
        np.where(observed, x, 0).sum(axis=1),
        n,
        out=np.full(len(n), np.nan),
        where=n >= x.shape[1] / 2,
    )


def read_night_features(
    subject: str, epochs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tracks = pd.read_parquet(
        CACHE / "tracks1s" / f"{subject}.parquet",
        columns=["second", "o_rel_alpha", "eog_low", "emg_log_rms"],
    )
    idx = epochs[:, None] * EPOCH_SECONDS + np.arange(EPOCH_SECONDS)
    if not np.array_equal(tracks.second, np.arange(len(tracks))):
        raise ValueError(f"{subject}: cached spectral clock is not contiguous")
    values = tracks[["o_rel_alpha", "eog_low", "emg_log_rms"]].to_numpy()
    # Padding treats a partial unobserved tail as unavailable rather than zero.
    if idx.max() >= len(values):
        values = np.pad(
            values, ((0, idx.max() + 1 - len(values)), (0, 0)), constant_values=np.nan
        )
    alpha, slow_eog, chin = (values[idx, k] for k in range(3))
    alpha_mean = finite_mean(alpha)
    delta = np.r_[np.nan, np.diff(alpha_mean)]
    delta[np.r_[True, np.diff(epochs) != 1]] = np.nan
    evidence = np.column_stack(
        (
            finite_mean(alpha[:, -10:]) - finite_mean(alpha[:, :10]),
            delta,
            finite_mean(slow_eog),
            finite_mean(chin),
        )
    )

    _, events, _ = load_cinc2018_annotations(resolve_cinc2018_record(RAW, subject))
    marks = np.column_stack(
        [
            np.isin(
                epochs,
                [int(t // EPOCH_SECONDS) for t, _, name in events if name == kind],
            )
            for kind in EVENT_KINDS
        ]
    )
    return evidence, marks


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    measured = aligned(
        frame, pd.read_parquet(MARKERS, columns=EPOCH_KEYS + ["quarter"] + EXISTING)
    )
    signal = np.full((len(frame), len(DERIVED)), np.nan, dtype=np.float32)
    marks = np.zeros((len(frame), len(EVENT_KINDS)), dtype=bool)
    y = frame[LABEL].to_numpy()
    for subject, rows in frame.groupby("subject_id", sort=False).indices.items():
        if not np.isin(y[rows], [WAKE, N1]).any():
            continue
        signal[rows], marks[rows] = read_night_features(
            subject, frame.epoch_index.to_numpy()[rows]
        )
    result = pd.concat((frame, measured[["quarter", *EXISTING]]), axis=1)
    return result.assign(
        **dict(
            zip(
                DERIVED + list(EVENT_KINDS.values()), [*signal.T, *marks.T], strict=True
            )
        )
    )


def directional(
    frame: pd.DataFrame, models: dict[str, np.ndarray], hard: np.ndarray
) -> tuple[list, list, dict]:
    y = frame[LABEL].to_numpy()
    stage_changed = np.r_[True, y[1:] != y[:-1]] | ~previous_link(frame)
    directions, phases, ranking = [], [], {}
    for name, p in models.items():
        pred = p.argmax(1)
        full_start = stage_changed | np.r_[True, pred[1:] != pred[:-1]]
        _, _, length = run_bounds(full_start)
        pair = np.isin(y, [WAKE, N1])
        contrast = np.log(np.maximum(p[:, N1], 1e-12)) - np.log(
            np.maximum(p[:, WAKE], 1e-12)
        )
        ranking[name] = {
            "all_pair_auc": float(roc_auc_score(y[pair] == N1, contrast[pair])),
            "hard_pair_auc": float(roc_auc_score(y[hard] == N1, contrast[hard])),
            "hard_epochs": int(hard.sum()),
            "phase_auc": {
                kind: float(roc_auc_score(y[sample] == N1, contrast[sample]))
                for kind in PHASES
                if (sample := hard & frame.phase.eq(kind).to_numpy()).sum() >= 20
                and len(np.unique(y[sample])) == 2
            },
        }
        for true, rival in PAIRS:
            missed = (y == true) & (pred == rival)
            long = missed & (length >= 6)
            record = {
                "model": name,
                "truth": STAGES[true],
                "rival": STAGES[rival],
                "epochs_truth": int((y == true).sum()),
                "errors": int(missed.sum()),
                "patients": int(frame.loc[missed, "subject_id"].nunique()),
                "isolated_errors": int((missed & (length == 1)).sum()),
                "sustained_errors": int(long.sum()),
                "sustained_runs": int((long & full_start).sum()),
                "marked_rera_on_error": int((missed & frame.marked_rera).sum()),
                "marked_spontaneous_on_error": int(
                    (missed & frame.marked_spontaneous).sum()
                ),
                "marked_rera_on_truth": int(((y == true) & frame.marked_rera).sum()),
                "marked_spontaneous_on_truth": int(
                    ((y == true) & frame.marked_spontaneous).sum()
                ),
            }
            directions.append(record)
            for kind in PHASES:
                phase = frame.phase.eq(kind).to_numpy()
                phases.append(
                    {
                        **{k: record[k] for k in ("model", "truth", "rival")},
                        "phase": kind,
                        "epochs_truth": int((phase & (y == true)).sum()),
                        "errors": int((phase & missed).sum()),
                        "errors_near_direct_wake_n1": int(
                            (phase & missed & frame.near_direct_wake_n1).sum()
                        ),
                        "sustained_errors": int((phase & long).sum()),
                    }
                )
    return directions, phases, ranking


def recorded_ahi_summary(
    frame: pd.DataFrame, models: dict, hard: np.ndarray
) -> list[dict]:
    subjects = pd.read_csv(SUBJECT_AHI, usecols=["record_id", "ahi"]).set_index(
        "record_id"
    )
    ahi = frame.subject_id.map(subjects["ahi"])
    bands = pd.cut(
        ahi,
        [-np.inf, 5, 15, 30, np.inf],
        labels=["<5", "5–<15", "15–<30", "≥30"],
        right=False,
    )
    if bands.isna().any():
        raise ValueError("Saved AHI summaries do not cover the development cohort")
    y = frame[LABEL].to_numpy()
    pair = np.isin(y, [N1, WAKE])
    rows = []
    for name, p in models.items():
        pred = p.argmax(axis=1)
        rank = p[:, N1] / np.maximum(p[:, N1] + p[:, WAKE], 1e-12)
        for band in bands.cat.categories:
            pop = bands.eq(band).to_numpy()
            row = {
                "model": name,
                "ahi_band": str(band),
                "patients": int(frame.loc[pop, "subject_id"].nunique()),
            }
            focus = pop & hard
            row["hard_pair_auc"] = float(roc_auc_score(y[focus] == N1, rank[focus]))
            row["hard_epochs"] = int(focus.sum())
            row["pair_epochs"] = int((pop & pair).sum())
            for true, rival in PAIRS:
                truth = pop & (y == true)
                missed = truth & (pred == rival)
                row[f"{STAGES[true]}_epochs"] = int(truth.sum())
                row[f"{STAGES[true]}_to_{STAGES[rival]}"] = int(missed.sum())
                row[f"{STAGES[true]}_to_{STAGES[rival]}_rate"] = float(
                    missed.sum() / truth.sum()
                )
            rows.append(row)
    return rows


def matched_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    correct = frame.compact.eq(frame[LABEL]) & frame.expanded.eq(frame[LABEL])
    keep = correct | (frame.expanded.ne(frame[LABEL]) & frame.expanded.isin([WAKE, N1]))
    groups = frame.loc[keep & frame[LABEL].isin([WAKE, N1])].groupby(
        ["subject_id", "quarter", "phase"], sort=False, observed=True
    )
    for (subject, _, phase), cell in groups:
        label = cell[LABEL].to_numpy()
        pred = cell.expanded.to_numpy()
        waveform = cell.waveform.to_numpy()
        controls = cell.compact.to_numpy() == label
        for true, rival in PAIRS:
            sampled = (label == true) & (pred == rival)
            if not sampled.any():
                continue
            mask = {
                "correct_true": (label == true) & (pred == true) & controls,
                "correct_rival": (label == rival) & (pred == rival) & controls,
                "errors": sampled,
                "waveform_rescue": sampled & (waveform == true),
                "waveform_also_wrong": sampled & (waveform != true),
            }
            if min(mask["correct_true"].sum(), mask["correct_rival"].sum()) < 2:
                continue
            for feature in FEATURES:
                signal = cell[feature].to_numpy()
                values = {
                    name: signal[use & np.isfinite(signal)]
                    for name, use in mask.items()
                }
                reference = np.sort(values["correct_rival"])
                if min(len(reference), len(values["correct_true"])) < 2 or not len(
                    values["errors"]
                ):
                    continue
                stats = {}
                for name, data in values.items():
                    stats[f"{name}_n"] = len(data)
                    stats[f"{name}_mean"] = (
                        float(np.mean(data)) if len(data) else np.nan
                    )
                    stats[f"{name}_auc"] = (
                        placement(data, reference) if len(data) else np.nan
                    )
                rows.append(
                    {
                        "subject_id": subject,
                        "phase": phase,
                        "feature": feature,
                        "truth": STAGES[true],
                        "rival": STAGES[rival],
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_matched(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["truth", "rival", "feature"]
    choices = [("all_phases", cells), *cells.groupby("phase", sort=False)]
    rng = np.random.default_rng(69)
    for phase, sample in choices:
        for key, group in sample.groupby(by, sort=False):
            for contrast, match, current, comparison in (
                ("errors_vs_correct", group, "errors", "correct_true"),
                (
                    "waveform_rescue_vs_wrong",
                    group.loc[
                        group.waveform_rescue_n.ge(2)
                        & group.waveform_also_wrong_n.ge(2)
                    ],
                    "waveform_rescue",
                    "waveform_also_wrong",
                ),
            ):
                patient = match.groupby("subject_id", sort=False).mean(
                    numeric_only=True
                )
                if len(patient) < 5:
                    continue
                sign = 1 if patient.correct_true_auc.mean() >= 0.5 else -1
                shift = (
                    sign
                    * (
                        patient[f"{current}_auc"] - patient[f"{comparison}_auc"]
                    ).to_numpy()
                )
                boot = shift[rng.integers(0, len(shift), size=(1000, len(shift)))].mean(
                    1
                )
                low, high = np.quantile(boot, (0.025, 0.975))
                record = dict(
                    zip(by, key, strict=True),
                    phase=phase,
                    contrast=contrast,
                    patients=len(patient),
                    cells=len(match),
                    auc_delta=float(shift.mean()),
                    low=float(low),
                    high=float(high),
                    truth_direction="higher" if sign > 0 else "lower",
                )
                for name in dict.fromkeys(
                    ("correct_true", "correct_rival", "errors", current, comparison)
                ):
                    mean_auc = patient[f"{name}_auc"].mean()
                    record[f"{name}_auc"] = mean_auc if sign > 0 else 1 - mean_auc
                    record[f"{name}_mean"] = patient[f"{name}_mean"].mean()
                    record[f"{name}_n"] = match[f"{name}_n"].sum()
                rows.append(record)
    return pd.DataFrame(rows)


def model_components(
    frame: pd.DataFrame, p: np.ndarray, terms: pd.DataFrame
) -> list[dict]:
    local, past, future = (
        terms[COMPONENTS].to_numpy().reshape(-1, 3, len(STAGES)).transpose(1, 0, 2)
    )
    full = local + past + future
    scaled = np.exp(full - full.max(1, keepdims=True))
    if not np.allclose(scaled / scaled.sum(1, keepdims=True), p, atol=2e-7, rtol=1e-6):
        raise ValueError(
            "Archived components do not reconstruct the expanded probabilities"
        )
    y, pred = frame[LABEL].to_numpy(), p.argmax(1)
    records = []
    for true, rival in PAIRS:
        deltas = [
            part[:, true] - part[:, rival] for part in (local, past, future, full)
        ]
        local_exact = local.argmax(1) == true
        other = np.full(len(y), True)
        for phase in ("all_phases", *PHASES):
            phase_set = (
                other if phase == "all_phases" else frame.phase.eq(phase).to_numpy()
            )
            for name, cohort in (
                ("pair_errors", (y == true) & (pred == rival)),
                ("correct_reference", (y == true) & (pred == true)),
            ):
                keep = cohort & phase_set
                if not keep.any():
                    continue
                records.append(
                    {
                        "truth": STAGES[true],
                        "rival": STAGES[rival],
                        "phase": phase,
                        "cohort": name,
                        "epochs": int(keep.sum()),
                        "local_argmax_was_correct": int((keep & local_exact).sum()),
                        "local_pair_favors_truth": int((keep & (deltas[0] > 0)).sum()),
                        **{
                            f"median_{part}_true_vs_rival": float(
                                np.median(delta[keep])
                            )
                            for part, delta in zip(
                                ("local", "past", "future", "full"), deltas, strict=True
                            )
                        },
                    }
                )
    return records


def main() -> None:
    if not (OUT / "FINDINGS.md").is_file():
        raise FileNotFoundError("Declare experiment 69 before running the audit")
    frame, archived, _ = load_predictions()
    expanded = aligned(
        frame, pd.read_parquet(EXPANDED / "expanded_predictions.parquet")
    )
    models = {
        "compact": archived["baseline"],
        "expanded": expanded[PROBABILITIES].to_numpy(),
        "blend": archived["blend"],
        "waveform": archived["waveform"],
    }
    for name, p in models.items():
        frame[name] = p.argmax(1)
    frame["phase"], frame["near_direct_wake_n1"] = annotation_phases(frame)
    frame = enrich(frame)
    top2 = np.argsort(-models["compact"], axis=1, kind="stable")[:, :2]
    sorted_p = np.take_along_axis(models["compact"], top2, axis=1)
    hard = (np.sort(top2, axis=1) == [WAKE, N1]).all(1) & (
        sorted_p[:, 0] <= 2 * sorted_p[:, 1]
    )
    hard &= frame[LABEL].isin([WAKE, N1]).to_numpy()
    directions, phases, ranking = directional(frame, models, hard)
    bands = recorded_ahi_summary(frame, models, hard)
    terms = pd.concat(
        [
            pd.read_parquet(
                EXPANDED / "model" / f"fold-{fold}" / "components.parquet",
                columns=EPOCH_KEYS + COMPONENTS,
            )
            for fold in range(5)
        ],
        ignore_index=True,
    )
    traces = model_components(frame, models["expanded"], aligned(frame, terms))
    features = aggregate_matched(matched_features(frame))

    overlaps = []
    for true, rival in PAIRS:
        truth = frame[LABEL].eq(true)
        before = frame.compact.eq(rival)
        now = frame.expanded.eq(rival)
        common = truth & before & now
        errors = truth & now
        overlap = {
            "truth": STAGES[true],
            "rival": STAGES[rival],
            "shared_compact_expanded": int(common.sum()),
            "corrected_to_truth_from_compact": int(
                (truth & before & frame.expanded.eq(true)).sum()
            ),
            "new_errors_from_compact_correct": int(
                (truth & frame.compact.eq(true) & now).sum()
            ),
            "waveform_agrees_with_reference_on_shared": int(
                (common & frame.waveform.eq(true)).sum()
            ),
            "waveform_agrees_with_reference_on_expanded": int(
                (errors & frame.waveform.eq(true)).sum()
            ),
            "expanded_error_all_models_same_rival": int(
                (common & frame.blend.eq(rival) & frame.waveform.eq(rival)).sum()
            ),
            "expanded_errors_top_10pct_patients": int(
                frame.loc[errors].groupby("subject_id").size().nlargest(32).sum()
            ),
        }
        overlaps.append(overlap)
    pd.DataFrame(directions).to_csv(OUT / "wake_n1_directional.csv", index=False)
    pd.DataFrame(phases).to_csv(OUT / "wake_n1_phases.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "wake_n1_ahi_bands.csv", index=False)
    pd.DataFrame(traces).to_csv(OUT / "wake_n1_components.csv", index=False)
    features.to_csv(
        OUT / "wake_n1_matched_markers.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(overlaps).to_csv(OUT / "wake_n1_model_overlap.csv", index=False)
    atomic_json(
        OUT / "wake_n1_summary.json",
        {
            "epochs": len(frame),
            "patients": int(frame.subject_id.nunique()),
            "ranking": ranking,
            "directions": directions,
            "overlap": overlaps,
            "ahi_bands": bands,
            "notes": {
                "hard": "compact top two Wake/N1, at most 2:1 odds, reference Wake/N1",
                "phase": "two epochs each side of the nearest adjacent marked stage change, then stable interior at least six epochs from either same-stage edge, then other",
                "sustained": "at least six consecutive 30-second epochs with unchanged reference and same wrong prediction",
                "events": "epoch contains a marked spontaneous or RERA onset; not a whole-night arousal classifier",
                "markers": "matched patient, night quarter and transition phase; two correctly labeled examples on each side minimum per cell; equal patient weighting",
                "components": "local, past, future logit contributions, relative to the true and rival stage",
            },
        },
    )
    print(
        f"Audited Wake/N1 on {len(frame):,} epochs, {len(features):,} matched marker comparisons"
    )


if __name__ == "__main__":
    main()
