from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _mean_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _run_ordinal_severity_diagnostics(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    norm_candidates: tuple[str, ...],
    subject_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from physionet_sleep.diagnostics.metrics import rank_auc
    from physionet_sleep.diagnostics.reference_runner import compute_group_effect
    from physionet_sleep.experiments.normalization import apply_normalization_single, resolve_normalization

    frame = table.dropna(subset=[target_column, subject_column]).copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column]).copy()
    frame[target_column] = frame[target_column].astype(int)

    candidate_outputs: list[dict[str, object]] = []
    for norm_name in norm_candidates:
        norm_frame = apply_normalization_single(
            frame,
            feature_columns=feature_columns,
            norm_name=norm_name,
            group_column=subject_column,
        )
        rows: list[dict[str, object]] = []
        for feature_name in feature_columns:
            subset = norm_frame[[feature_name, target_column, subject_column]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) < 10:
                continue
            x = subset[feature_name].to_numpy(dtype=float)
            y = subset[target_column].to_numpy(dtype=int)
            rho = float(spearmanr(x, y, nan_policy="omit").statistic)
            auc_ge_hyp = float("nan")
            auc_ge_apn = float("nan")
            pos = x[y >= 1]
            neg = x[y == 0]
            if pos.size >= 5 and neg.size >= 5:
                auc = rank_auc(pos, neg)
                auc_ge_hyp = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
            pos = x[y >= 2]
            neg = x[y <= 1]
            if pos.size >= 5 and neg.size >= 5:
                auc = rank_auc(pos, neg)
                auc_ge_apn = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
            subject_effect = float(
                compute_group_effect(
                    x,
                    subset[subject_column].astype(str).to_numpy(),
                    method="kw",
                )
            )
            mean_threshold_auc = _mean_finite([auc_ge_hyp, auc_ge_apn])
            rows.append(
                {
                    "feature_name": feature_name,
                    "selected_norm_name": resolve_normalization(norm_name).name,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else float("nan"),
                    "auc_ge_hypopnea": auc_ge_hyp,
                    "auc_ge_apnea": auc_ge_apn,
                    "mean_threshold_auc": mean_threshold_auc,
                    "subject_effect_score": subject_effect,
                    "selection_score": mean_threshold_auc - subject_effect if np.isfinite(mean_threshold_auc) and np.isfinite(subject_effect) else float("nan"),
                    "n_obs": int(len(subset)),
                }
            )
        candidate_outputs.append(
            {
                "norm_name": resolve_normalization(norm_name).name,
                "rows": pd.DataFrame(rows),
            }
        )

    best_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    feature_names = list(dict.fromkeys(feature_columns))
    for feature_name in feature_names:
        candidates: list[pd.Series] = []
        for candidate in candidate_outputs:
            rows = candidate["rows"]
            if isinstance(rows, pd.DataFrame):
                match = rows.loc[rows["feature_name"] == feature_name]
                if not match.empty:
                    row = match.iloc[0]
                    candidates.append(row)
                    all_rows.append(row.to_dict())
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: row["selection_score"] if np.isfinite(row["selection_score"]) else -np.inf,
        )
        best_rows.append(best.to_dict())

    best_df = pd.DataFrame(best_rows).sort_values("selection_score", ascending=False, na_position="last")
    all_df = pd.DataFrame(all_rows).sort_values(
        ["selection_score", "mean_threshold_auc", "abs_spearman_rho"],
        ascending=[False, False, False],
        na_position="last",
    )
    return best_df.reset_index(drop=True), all_df.reset_index(drop=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.diagnostics.reference_runner import DEFAULT_REFERENCE_NORMS, run_reference_diagnostics
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        print("no_rows", flush=True)
        return 1

    specs = build_default_experiment_specs(table)
    apnea_features = specs["apnea_plus_shared"].feature_columns

    out_root = repo_root / "artifacts" / "reference_diagnostics_by_task"
    out_root.mkdir(parents=True, exist_ok=True)

    print("respiratory_severity_start", flush=True)
    severity_best, severity_all = _run_ordinal_severity_diagnostics(
        table,
        feature_columns=apnea_features,
        norm_candidates=DEFAULT_REFERENCE_NORMS,
        subject_column="subject_id",
        target_column="label.respiratory_apnea",
    )
    severity_dir = out_root / "respiratory_severity_ordinal"
    severity_dir.mkdir(parents=True, exist_ok=True)
    severity_best.to_parquet(severity_dir / "best_feature_norms.parquet", index=False)
    severity_all.to_parquet(severity_dir / "all_norm_scores.parquet", index=False)
    pd.DataFrame(
        [
            {
                "target_column": "label.respiratory_apnea",
                "feature_count": int(len(apnea_features)),
                "row_count": int(len(table)),
                "severity_levels": [0, 1, 2],
            }
        ]
    ).to_json(severity_dir / "metadata.json", orient="records", indent=2)
    print(f"respiratory_severity_done features={len(severity_best)}", flush=True)

    print("respiratory_obstruction_start", flush=True)
    obstruction_frame = table.loc[table["label.respiratory_apnea"] > 0].copy()
    obstruction_result = run_reference_diagnostics(
        obstruction_frame,
        target_column="label.respiratory_obstruction",
        subject_column="subject_id",
        feature_columns=apnea_features,
        enable_feature_selection=False,
    )
    obstruction_dir = out_root / "respiratory_obstruction"
    obstruction_result.save(obstruction_dir)
    print(
        f"respiratory_obstruction_done classes={obstruction_result.metadata['class_ids']} features={obstruction_result.metadata['feature_count']}",
        flush=True,
    )
    print(f"saved={out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
