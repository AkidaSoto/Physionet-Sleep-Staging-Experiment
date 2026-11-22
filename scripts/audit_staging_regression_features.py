"""Compare local engineered evidence in new versus recurring #54 errors."""


import numpy as np
import pandas as pd

from run_waveform_development import EXPLORATION, KEYS, LABEL, STAGES

OUT = EXPLORATION / "cinc-error-interactions"
INPUT = EXPLORATION / "cinc-cap-audit/feature_failure_epochs.parquet"
JOIN = [*KEYS, LABEL, "fold"]
FEATURES = [
    "c_rel_theta",
    "c_rel_alpha",
    "c_rel_beta",
    "c_log_power",
    "o_rel_alpha",
    "eog_high",
    "aperiodic_exponent",
    "delta_power",
    "rel_theta",
    "rel_beta",
    "eye_movement_activity",
    "eye_movement_baseline",
    "eog_rem_vs_noise_db",
    "eog_rem_vs_sem_db",
    "emg_floor_db",
    "frontal_slow_wave_fraction",
    "occipital_rel_alpha",
    "burst_seconds",
    "frontal_delta_control",
]
PAIRS = [(2, 1), (2, 3), (3, 2), (1, 0), (0, 1), (4, 1), (1, 2), (1, 4)]


def against_rival(sample: np.ndarray, comparators: np.ndarray) -> np.ndarray:
    auc = np.full(sample.shape[1], np.nan)
    for k, (a, b) in enumerate(zip(sample.T, comparators.T)):
        a = a[np.isfinite(a)]
        b = np.sort(b[np.isfinite(b)])
        if len(a) and len(b) >= 3:
            ranks = np.searchsorted(b, a, side="left") + np.searchsorted(
                b, a, side="right"
            )
            auc[k] = ranks.mean() / (2 * len(b))
    return auc


def matched_pair(frame: pd.DataFrame, true: int, rival: int) -> list[dict]:
    # Use the same contexts and same correctly scored comparators for both
    # cohorts. Comparing up to four directional slices in each stratum costs
    # little and avoids weight being driven by sample size in just one cohort.
    groups = ["subject_id", "quarter", "ref_position"]
    correct = frame.baseline.eq(frame[LABEL]) & frame.searched.eq(frame[LABEL])
    rivals = frame.loc[correct & frame[LABEL].eq(rival)]
    old_context = {
        k: g[FEATURES].to_numpy()
        for k, g in rivals.groupby(groups, sort=False)
        if len(g) >= 3
    }
    true_context = {
        k: g[FEATURES].to_numpy()
        for k, g in frame.loc[correct & frame[LABEL].eq(true)].groupby(
            groups, sort=False
        )
        if len(g) >= 3
    }
    cases = frame.loc[
        frame[LABEL].eq(true) & (frame.searched.eq(rival) | frame.baseline.eq(rival))
    ]
    rows = []
    for key, stratum in cases.groupby(
        groups + ["margin_band"], observed=True, sort=False
    ):
        use = key[:3]
        if use not in old_context or use not in true_context:
            continue
        outcomes = {
            "new": stratum.baseline.eq(true) & stratum.searched.eq(rival),
            "persistent": stratum.baseline.eq(rival) & stratum.searched.eq(rival),
            "repaired": stratum.baseline.eq(rival) & stratum.searched.eq(true),
        }
        if not outcomes["new"].any() or not outcomes["persistent"].any():
            continue
        ref = old_context[use]
        clean = against_rival(true_context[use], ref)
        arrays = {
            tag: against_rival(stratum.loc[mask, FEATURES].to_numpy(), ref)
            for tag, mask in outcomes.items()
        }
        for k, name in enumerate(FEATURES):
            if np.isfinite([clean[k], arrays["new"][k], arrays["persistent"][k]]).all():
                rows.append(
                    {
                        "person": key[0],
                        "feature": name,
                        "clean": clean[k],
                        "new": arrays["new"][k],
                        "old": arrays["persistent"][k],
                        "repaired": arrays["repaired"][k],
                        "matched_new": int(outcomes["new"].sum()),
                        "matched_persistent": int(outcomes["persistent"].sum()),
                        "matched_repaired": int(outcomes["repaired"].sum()),
                        "weight": min(
                            int(outcomes["new"].sum()),
                            int(outcomes["persistent"].sum()),
                        ),
                    }
                )
    if not rows:
        return []
    summary = []
    for name, g in pd.DataFrame(rows).groupby("feature", sort=False):
        x = {
            col: float(
                np.average(
                    g.loc[g[col].notna(), col], weights=g.loc[g[col].notna(), "weight"]
                )
            )
            if g[col].notna().any()
            else None
            for col in ("clean", "new", "old", "repaired")
        }
        oriented = {
            k: (v if x["clean"] >= 0.5 else 1 - v) if v is not None else None
            for k, v in x.items()
        }
        summary.append(
            {
                "truth": STAGES[true],
                "rival": STAGES[rival],
                "feature": name,
                "direction": "higher supports truth"
                if x["clean"] >= 0.5
                else "lower supports truth",
                "matched_patients": g.person.nunique(),
                "new": int(g.matched_new.sum()),
                "persistent": int(g.matched_persistent.sum()),
                **{f"{k}_auc": v for k, v in oriented.items()},
                "new_minus_old": oriented["new"] - oriented["old"],
            }
        )
    return summary


def main():
    features = pd.read_parquet(
        INPUT, columns=JOIN + ["quarter", "ref_position"] + FEATURES
    )
    results = pd.read_parquet(
        OUT / "archived_error_epochs.parquet",
        columns=JOIN + ["baseline", "searched", "baseline_margin"],
    )
    evidence = features.merge(results, on=JOIN, validate="one_to_one")
    if len(evidence) != len(features) or len(evidence) != len(results):
        raise ValueError(
            "Feature audit and archived error predictions differ in coverage"
        )
    evidence["margin_band"] = pd.cut(
        evidence.baseline_margin, [-1e-6, 0.1, 0.25, 0.5, 1.00001]
    )
    frame = pd.DataFrame(
        [row for pair in PAIRS for row in matched_pair(evidence, *pair)]
    )
    frame.to_csv(
        OUT / "matched_new_vs_persistent_features.csv", index=False, float_format="%.6f"
    )
    print(f"{len(frame)} stage-contrast feature summaries")


if __name__ == "__main__":
    main()
