from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from physionet_sleep.diagnostics.metrics import cohens_d, rank_auc
from physionet_sleep.experiments.normalization import apply_normalization_single, resolve_normalization


DEFAULT_REFERENCE_NORMS = ("standard_global", "robust_global", "robust_by_subject", "none")


@dataclass(slots=True)
class ReferenceDiagnosticsResult:
    best_feature_norms: pd.DataFrame
    feat_score_tab: pd.DataFrame
    top_per_class: pd.DataFrame
    cohen_d_heatmap: pd.DataFrame
    per_subject_cohend: pd.DataFrame
    per_class_cohend: pd.DataFrame
    pairwise_cohend: pd.DataFrame
    feature_auc: pd.DataFrame
    class_auc_signal: pd.DataFrame
    hypothesis_tests: pd.DataFrame
    feature_selection_auc_rank: pd.DataFrame
    feature_selection_greedy_forward: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.best_feature_norms.to_parquet(root / "best_feature_norms.parquet", index=False)
        self.feat_score_tab.to_parquet(root / "feat_score_tab.parquet", index=False)
        self.top_per_class.to_parquet(root / "top_per_class.parquet", index=False)
        self.cohen_d_heatmap.to_parquet(root / "cohen_d_heatmap.parquet", index=False)
        self.per_subject_cohend.to_parquet(root / "per_subject_cohend.parquet", index=False)
        self.per_class_cohend.to_parquet(root / "per_class_cohend.parquet", index=False)
        self.pairwise_cohend.to_parquet(root / "pairwise_cohend.parquet", index=False)
        self.feature_auc.to_parquet(root / "feature_auc.parquet", index=False)
        self.class_auc_signal.to_parquet(root / "class_auc_signal.parquet", index=False)
        self.hypothesis_tests.to_parquet(root / "hypothesis_tests.parquet", index=False)
        self.feature_selection_auc_rank.to_parquet(root / "feature_selection_auc_rank.parquet", index=False)
        self.feature_selection_greedy_forward.to_parquet(root / "feature_selection_greedy_forward.parquet", index=False)
        pd.DataFrame([self.metadata]).to_json(root / "metadata.json", orient="records", indent=2)


def run_reference_diagnostics(
    feature_table: pd.DataFrame,
    *,
    target_column: str = "label.stage_seconds",
    subject_column: str = "subject_id",
    norm_candidates: tuple[str, ...] = DEFAULT_REFERENCE_NORMS,
    feat_norm_metric: str = "stage_minus_subject_eta2",
    feat_group_effect_method: str = "kw",
    top_n: int = 20,
    hypothesis_tests: list[dict[str, Any]] | None = None,
    auc_rank_model_family: str = "tree",
    greedy_model_family: str = "tree",
    feature_selection_progress: bool = False,
    feature_columns: list[str] | None = None,
    enable_feature_selection: bool = True,
) -> ReferenceDiagnosticsResult:
    feature_columns = feature_columns or _feature_columns(feature_table)
    stage_frame = feature_table.dropna(subset=[target_column, subject_column]).copy()
    stage_frame[target_column] = pd.to_numeric(stage_frame[target_column], errors="coerce")
    stage_frame = stage_frame.dropna(subset=[target_column]).copy()
    stage_frame[target_column] = stage_frame[target_column].astype(int)
    if target_column == "label.stage_seconds":
        stage_frame = stage_frame.loc[stage_frame[target_column] >= 0].copy()
    class_ids = sorted(int(value) for value in stage_frame[target_column].dropna().unique().tolist())

    candidate_outputs: list[dict[str, Any]] = []
    for norm_name in norm_candidates:
        norm_frame = apply_normalization_single(
            stage_frame,
            feature_columns=feature_columns,
            norm_name=norm_name,
            group_column=subject_column,
        )
        feat_screen = build_feat_score_tab(norm_frame, feature_columns=feature_columns, stage_column=target_column)
        candidate_outputs.append(
            {
                "norm_name": resolve_normalization(norm_name).name,
                "frame": norm_frame,
                "feat_score_tab": feat_screen,
            }
        )

    best_feature_norms = build_best_feature_norms(
        candidate_outputs,
        feature_columns=feature_columns,
        stage_column=target_column,
        subject_column=subject_column,
        metric=feat_norm_metric,
        effect_method=feat_group_effect_method,
    )
    feat_score_tab = assemble_selected_feat_score_tab(candidate_outputs, best_feature_norms, class_ids=class_ids)
    top_per_class = build_top_per_class(feat_score_tab, top_n=5)
    cohen_d_heatmap = build_cohen_d_heatmap(feat_score_tab)
    per_subject_cohend = compute_per_subject_cohend(
        candidate_outputs,
        best_feature_norms,
        stage_column=target_column,
        subject_column=subject_column,
        top_n=top_n,
    )
    per_class_cohend = compute_per_class_cohend(feat_score_tab, top_n=top_n)
    pairwise_cohend = compute_pairwise_cohend(
        candidate_outputs,
        best_feature_norms,
        stage_column=target_column,
        top_n=top_n,
    )
    feature_auc = compute_feature_auc(feat_score_tab)
    class_auc_signal = compute_class_auc_signal(feat_score_tab)
    hyp_df = run_hypothesis_tests(
        candidate_outputs,
        best_feature_norms,
        stage_column=target_column,
        tests=hypothesis_tests or [],
    )
    if enable_feature_selection:
        feature_selection_auc_rank = run_feature_selection_search(
            stage_frame,
            feat_score_tab,
            stage_column=target_column,
            subject_column=subject_column,
            mode="auc_rank",
            model_family=auc_rank_model_family,
            max_features=15,
            candidate_pool_size=20,
            progress=feature_selection_progress,
        )
        feature_selection_greedy = run_feature_selection_search(
            stage_frame,
            feat_score_tab,
            stage_column=target_column,
            subject_column=subject_column,
            mode="greedy_forward",
            model_family=greedy_model_family,
            max_features=10,
            candidate_pool_size=20,
            progress=feature_selection_progress,
        )
    else:
        feature_selection_auc_rank = pd.DataFrame()
        feature_selection_greedy = pd.DataFrame()

    return ReferenceDiagnosticsResult(
        best_feature_norms=best_feature_norms,
        feat_score_tab=feat_score_tab,
        top_per_class=top_per_class,
        cohen_d_heatmap=cohen_d_heatmap,
        per_subject_cohend=per_subject_cohend,
        per_class_cohend=per_class_cohend,
        pairwise_cohend=pairwise_cohend,
        feature_auc=feature_auc,
        class_auc_signal=class_auc_signal,
        hypothesis_tests=hyp_df,
        feature_selection_auc_rank=feature_selection_auc_rank,
        feature_selection_greedy_forward=feature_selection_greedy,
        metadata={
            "norm_candidates": list(norm_candidates),
            "feat_norm_metric": feat_norm_metric,
            "feat_group_effect_method": feat_group_effect_method,
            "target_column": target_column,
            "class_ids": class_ids,
            "row_count": int(len(stage_frame)),
            "feature_count": int(len(feature_columns)),
            "auc_rank_model_family": auc_rank_model_family,
            "greedy_model_family": greedy_model_family,
        },
    )


def build_best_feature_norms(
    candidate_outputs: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    stage_column: str,
    subject_column: str,
    metric: str,
    effect_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in feature_columns:
        best_row: dict[str, Any] | None = None
        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidate_outputs:
            frame = candidate["frame"]
            score, stage_effect, subject_effect, n_obs = compute_norm_selection_score(
                frame,
                feature_name=feature_name,
                stage_column=stage_column,
                subject_column=subject_column,
                metric=metric,
                effect_method=effect_method,
            )
            row = {
                "feature_name": feature_name,
                "selected_norm_name": candidate["norm_name"],
                "selection_score": score,
                "stage_effect_score": stage_effect,
                "subject_effect_score": subject_effect,
                "n_obs": n_obs,
            }
            candidate_rows.append(row)
            if best_row is None or _better_score(row["selection_score"], best_row["selection_score"]):
                best_row = row
        if best_row is None:
            continue
        for row in candidate_rows:
            norm_name = row["selected_norm_name"]
            row[f"score__{norm_name}"] = row["selection_score"]
            row[f"stage_effect__{norm_name}"] = row["stage_effect_score"]
            row[f"subject_effect__{norm_name}"] = row["subject_effect_score"]
        base = best_row.copy()
        for candidate in candidate_outputs:
            match = next(item for item in candidate_rows if item["selected_norm_name"] == candidate["norm_name"])
            base[f"score__{candidate['norm_name']}"] = match["selection_score"]
            base[f"stage_effect__{candidate['norm_name']}"] = match["stage_effect_score"]
            base[f"subject_effect__{candidate['norm_name']}"] = match["subject_effect_score"]
        rows.append(base)

    out = pd.DataFrame(rows).sort_values("selection_score", ascending=False, na_position="last")
    if not out.empty:
        out["selection_metric"] = metric
        out["effect_method"] = effect_method
    return out


def compute_norm_selection_score(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    stage_column: str,
    subject_column: str,
    metric: str,
    effect_method: str,
) -> tuple[float, float, float, int]:
    subset = frame[[feature_name, stage_column, subject_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if subset.empty:
        return float("nan"), float("nan"), float("nan"), 0
    x = subset[feature_name].to_numpy(dtype=float)
    stage_labels = subset[stage_column].astype(str).to_numpy()
    subject_labels = subset[subject_column].astype(str).to_numpy()
    stage_effect = compute_group_effect(x, stage_labels, method=effect_method)
    subject_effect = compute_group_effect(x, subject_labels, method=effect_method)
    if metric == "neg_subject_eta2":
        score = -subject_effect if np.isfinite(subject_effect) else float("nan")
    else:
        score = (stage_effect - subject_effect) if np.isfinite(stage_effect) and np.isfinite(subject_effect) else float("nan")
    return float(score), float(stage_effect), float(subject_effect), int(len(subset))


def build_feat_score_tab(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    stage_column: str,
) -> pd.DataFrame:
    y = frame[stage_column].astype(int).to_numpy()
    class_ids = sorted(int(value) for value in np.unique(y))
    rows: list[dict[str, Any]] = []
    for feature_name in feature_columns:
        x = frame[feature_name].to_numpy(dtype=float, copy=False)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 10:
            continue
        x_use = x[finite]
        y_use = y[finite]
        row: dict[str, Any] = {
            "feat_name": feature_name,
            "channel_name": feature_name,
            "part_offset": 0,
        }
        auc_scores: list[float] = []
        d_scores: list[float] = []
        for class_id in class_ids:
            pos = x_use[y_use == class_id]
            neg = x_use[y_use != class_id]
            if pos.size < 5 or neg.size < 5:
                auc_sep = float("nan")
                d_val = float("nan")
            else:
                auc = rank_auc(pos, neg)
                auc_sep = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
                d_val = float(cohens_d(pos, neg))
            row[f"auc_sep_{class_id}"] = auc_sep
            row[f"cohen_d_{class_id}"] = d_val
            auc_scores.append(auc_sep)
            d_scores.append(abs(d_val) if np.isfinite(d_val) else np.nan)
        row["effect_size"] = float(np.sqrt(np.nansum(np.square(auc_scores))))
        row["std_effect_size"] = float(np.nanstd(auc_scores))
        row["mean_abs_cohen_d"] = float(np.nanmean(d_scores))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("effect_size", ascending=False, na_position="last").reset_index(drop=True)
    return out


def assemble_selected_feat_score_tab(
    candidate_outputs: list[dict[str, Any]],
    best_feature_norms: pd.DataFrame,
    *,
    class_ids: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, norm_row in best_feature_norms.iterrows():
        feature_name = norm_row["feature_name"]
        selected_norm = norm_row["selected_norm_name"]
        candidate = next(item for item in candidate_outputs if item["norm_name"] == selected_norm)
        feat_tab = candidate["feat_score_tab"]
        match = feat_tab.loc[feat_tab["feat_name"] == feature_name]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["selected_norm_name"] = selected_norm
        row["selection_score"] = float(norm_row["selection_score"])
        row["stage_effect_score"] = float(norm_row["stage_effect_score"])
        row["subject_effect_score"] = float(norm_row["subject_effect_score"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    auc_cols = [f"auc_sep_{class_id}" for class_id in class_ids if f"auc_sep_{class_id}" in out.columns]
    if auc_cols:
        out["effect_size"] = np.sqrt(np.nansum(np.square(out[auc_cols].to_numpy(dtype=float)), axis=1))
        out["std_effect_size"] = np.nanstd(out[auc_cols].to_numpy(dtype=float), axis=1)
    return out.sort_values("effect_size", ascending=False, na_position="last").reset_index(drop=True)


def build_cohen_d_heatmap(feat_score_tab: pd.DataFrame) -> pd.DataFrame:
    d_cols = [col for col in feat_score_tab.columns if col.startswith("cohen_d_")]
    if feat_score_tab.empty or not d_cols:
        return pd.DataFrame()
    heatmap = feat_score_tab[["channel_name", *d_cols]].copy()
    return heatmap


def build_top_per_class(feat_score_tab: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    auc_cols = [col for col in feat_score_tab.columns if col.startswith("auc_sep_")]
    for col in auc_cols:
        class_id = int(col.replace("auc_sep_", ""))
        ranked = feat_score_tab[
            ["feat_name", "channel_name", "selected_norm_name", col, f"cohen_d_{class_id}"]
        ].copy()
        ranked = ranked.sort_values(col, ascending=False, na_position="last").head(top_n)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "class_id": class_id,
                    "rank": rank,
                    "feature_name": row["feat_name"],
                    "channel_name": row["channel_name"],
                    "selected_norm_name": row["selected_norm_name"],
                    "auc_sep": float(row[col]),
                    "cohen_d": float(row[f"cohen_d_{class_id}"]),
                }
            )
    return pd.DataFrame(rows)


def compute_per_subject_cohend(
    candidate_outputs: list[dict[str, Any]],
    best_feature_norms: pd.DataFrame,
    *,
    stage_column: str,
    subject_column: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_features = best_feature_norms.head(top_n)
    for _, norm_row in top_features.iterrows():
        feature_name = norm_row["feature_name"]
        selected_norm = norm_row["selected_norm_name"]
        frame = next(item["frame"] for item in candidate_outputs if item["norm_name"] == selected_norm)
        for subject_id, subject_frame in frame.groupby(subject_column, sort=False):
            y = subject_frame[stage_column].dropna().astype(int)
            if y.nunique() < 2:
                continue
            x = subject_frame.loc[y.index, feature_name].to_numpy(dtype=float)
            class_ids = sorted(int(value) for value in np.unique(y))
            class_effects = []
            for class_id in class_ids:
                pos = x[y.to_numpy() == class_id]
                neg = x[y.to_numpy() != class_id]
                if pos.size < 3 or neg.size < 3:
                    continue
                class_effects.append(abs(cohens_d(pos, neg)))
            if not class_effects:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "feature_name": feature_name,
                    "selected_norm_name": selected_norm,
                    "mean_abs_cohen_d": float(np.nanmean(class_effects)),
                }
            )
    return pd.DataFrame(rows)


def compute_per_class_cohend(feat_score_tab: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    d_cols = [col for col in feat_score_tab.columns if col.startswith("cohen_d_")]
    for col in d_cols:
        class_id = int(col.replace("cohen_d_", ""))
        ranked = feat_score_tab[["channel_name", "selected_norm_name", col]].copy().sort_values(
            col,
            ascending=False,
            key=lambda s: s.abs(),
            na_position="last",
        )
        for rank, (_, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
            rows.append(
                {
                    "class_id": class_id,
                    "rank": rank,
                    "feature_name": row["channel_name"],
                    "selected_norm_name": row.get("selected_norm_name"),
                    "cohen_d": float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def compute_pairwise_cohend(
    candidate_outputs: list[dict[str, Any]],
    best_feature_norms: pd.DataFrame,
    *,
    stage_column: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_features = best_feature_norms.head(top_n)
    for _, norm_row in top_features.iterrows():
        feature_name = norm_row["feature_name"]
        selected_norm = norm_row["selected_norm_name"]
        frame = next(item["frame"] for item in candidate_outputs if item["norm_name"] == selected_norm)
        subset = frame[[feature_name, stage_column]].replace([np.inf, -np.inf], np.nan).dropna()
        y = subset[stage_column].astype(int).to_numpy()
        x = subset[feature_name].to_numpy(dtype=float)
        class_ids = sorted(int(value) for value in np.unique(y))
        for i, class_a in enumerate(class_ids):
            for class_b in class_ids[i + 1 :]:
                xa = x[y == class_a]
                xb = x[y == class_b]
                if xa.size < 5 or xb.size < 5:
                    continue
                auc = rank_auc(xa, xb)
                rows.append(
                    {
                        "feature_name": feature_name,
                        "selected_norm_name": selected_norm,
                        "class_a": class_a,
                        "class_b": class_b,
                        "cohen_d": float(abs(cohens_d(xa, xb))),
                        "auc_sep": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def compute_feature_auc(feat_score_tab: pd.DataFrame) -> pd.DataFrame:
    auc_cols = [col for col in feat_score_tab.columns if col.startswith("auc_sep_")]
    rows: list[dict[str, Any]] = []
    for _, row in feat_score_tab.iterrows():
        for col in auc_cols:
            rows.append(
                {
                    "feature_name": row["channel_name"],
                    "selected_norm_name": row.get("selected_norm_name"),
                    "class_id": int(col.replace("auc_sep_", "")),
                    "auc_sep": float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def compute_class_auc_signal(feat_score_tab: pd.DataFrame) -> pd.DataFrame:
    auc_cols = [col for col in feat_score_tab.columns if col.startswith("auc_sep_")]
    rows: list[dict[str, Any]] = []
    for col in auc_cols:
        class_id = int(col.replace("auc_sep_", ""))
        rows.append(
            {
                "class_id": class_id,
                "mean_auc_sep": float(feat_score_tab[col].mean()),
                "median_auc_sep": float(feat_score_tab[col].median()),
                "n_features": int(feat_score_tab[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def run_hypothesis_tests(
    candidate_outputs: list[dict[str, Any]],
    best_feature_norms: pd.DataFrame,
    *,
    stage_column: str,
    tests: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not tests:
        return pd.DataFrame(rows)
    for test in tests:
        feature_name = test.get("feature_name")
        class_a = test.get("class_a")
        class_b = test.get("class_b")
        if feature_name is None or class_a is None or class_b is None:
            continue
        match = best_feature_norms.loc[best_feature_norms["feature_name"] == feature_name]
        if match.empty:
            continue
        selected_norm = match.iloc[0]["selected_norm_name"]
        frame = next(item["frame"] for item in candidate_outputs if item["norm_name"] == selected_norm)
        subset = frame[[feature_name, stage_column]].replace([np.inf, -np.inf], np.nan).dropna()
        y = subset[stage_column].astype(int).to_numpy()
        x = subset[feature_name].to_numpy(dtype=float)
        xa = x[y == int(class_a)]
        xb = x[y == int(class_b)]
        if xa.size < 5 or xb.size < 5:
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "selected_norm_name": selected_norm,
                "class_a": int(class_a),
                "class_b": int(class_b),
                "cohen_d": float(cohens_d(xa, xb)),
                "auc_sep": float(max(rank_auc(xa, xb), 1.0 - rank_auc(xa, xb))),
            }
        )
    return pd.DataFrame(rows)


def run_feature_selection_search(
    frame: pd.DataFrame,
    feat_score_tab: pd.DataFrame,
    *,
    stage_column: str,
    subject_column: str,
    mode: str,
    model_family: str,
    max_features: int = 20,
    min_delta_pct: float = 0.0,
    candidate_pool_size: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    if frame.empty or feat_score_tab.empty:
        return pd.DataFrame()
    ordered_features = feat_score_tab.sort_values("effect_size", ascending=False)["channel_name"].tolist()
    ordered_features = [feature for feature in ordered_features if feature in frame.columns]
    remaining = list(dict.fromkeys(ordered_features))
    if candidate_pool_size is not None:
        remaining = remaining[:candidate_pool_size]
    if not remaining:
        return pd.DataFrame()

    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[stage_column].astype(int).to_numpy()
    majority_pred = np.repeat(pd.Series(y).mode().iloc[0], len(y))
    baseline_f1 = float(f1_score(y, majority_pred, average="macro", zero_division=0))

    active_features: list[str] = []
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "added_feature": "__baseline__",
            "cumulative_macro_f1": baseline_f1,
            "delta_f1": np.nan,
            "delta_f1_pct": np.nan,
            "mode": mode,
            "model_family": model_family,
        }
    ]
    if progress:
        print(
            f"[feature_selection][{mode}][{model_family}] step=0 feature=__baseline__ macro_f1={baseline_f1:.6f}",
            flush=True,
        )
    prev_f1 = baseline_f1

    for step in range(1, min(max_features, len(remaining)) + 1):
        if mode == "auc_rank":
            chosen = remaining[0]
            candidate_features = active_features + [chosen]
            candidate_f1 = evaluate_group_loso_macro_f1(
                frame,
                feature_columns=candidate_features,
                target_column=stage_column,
                subject_column=subject_column,
                model_family=model_family,
            )
        else:
            best_feature = None
            best_f1 = -np.inf
            for candidate in remaining:
                candidate_features = active_features + [candidate]
                score = evaluate_group_loso_macro_f1(
                    frame,
                    feature_columns=candidate_features,
                    target_column=stage_column,
                    subject_column=subject_column,
                    model_family=model_family,
                )
                if score > best_f1:
                    best_f1 = score
                    best_feature = candidate
            chosen = best_feature
            candidate_f1 = best_f1

        if chosen is None:
            break
        active_features.append(chosen)
        remaining.remove(chosen)
        delta = float(candidate_f1 - prev_f1)
        delta_pct = float((delta / prev_f1) * 100.0) if prev_f1 > 0 else np.nan
        rows.append(
            {
                "step": step,
                "added_feature": chosen,
                "cumulative_macro_f1": float(candidate_f1),
                "delta_f1": delta,
                "delta_f1_pct": delta_pct,
                "mode": mode,
                "model_family": model_family,
            }
        )
        if progress:
            print(
                f"[feature_selection][{mode}][{model_family}] step={step} feature={chosen} "
                f"macro_f1={float(candidate_f1):.6f} delta_f1={delta:.6f} delta_pct={delta_pct:.3f}",
                flush=True,
            )
        prev_f1 = float(candidate_f1)
        if np.isfinite(delta_pct) and delta_pct < min_delta_pct:
            break
    return pd.DataFrame(rows)


def evaluate_group_loso_macro_f1(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    subject_column: str,
    model_family: str,
) -> float:
    fold_scores: list[float] = []
    groups = frame[subject_column].astype(str).to_numpy()
    y = frame[target_column].astype(int).to_numpy()
    unique_groups = np.unique(groups)
    for held_out in unique_groups:
        train_mask = groups != held_out
        val_mask = groups == held_out
        if train_mask.sum() < 10 or val_mask.sum() < 2:
            continue
        train_df = frame.loc[train_mask, [*feature_columns, target_column, subject_column]].copy()
        val_df = frame.loc[val_mask, [*feature_columns, target_column, subject_column]].copy()
        from physionet_sleep.experiments.normalization import apply_normalization
        from physionet_sleep.experiments.tabular_runner import build_model

        train_df, val_df, _ = apply_normalization(
            train_df,
            val_df,
            feature_columns=feature_columns,
            norm_name="robust_by_subject",
            group_column=subject_column,
        )
        usable_columns = [col for col in feature_columns if train_df[col].notna().any()]
        if not usable_columns:
            continue
        estimator = build_model(model_family=model_family, task="classification", random_state=7)
        estimator.fit(train_df[usable_columns], train_df[target_column])
        pred = estimator.predict(val_df[usable_columns])
        fold_scores.append(float(f1_score(val_df[target_column], pred, average="macro", zero_division=0)))
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def compute_group_effect(x: np.ndarray, group_labels: np.ndarray, *, method: str) -> float:
    x, labels = _prep_group_data(x, group_labels)
    if x.size < 5 or np.unique(labels).size < 2:
        return float("nan")
    method_key = method.lower()
    if method_key == "icc":
        return _compute_group_icc(x, labels)
    if method_key == "kw":
        return _compute_group_kw(x, labels)
    return _compute_group_eta2(x, labels)


def _compute_group_eta2(x: np.ndarray, labels: np.ndarray) -> float:
    grand = float(np.mean(x))
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 0:
        return float("nan")
    group_means = pd.Series(x).groupby(labels).transform("mean").to_numpy(dtype=float)
    ss_between = float(np.sum((group_means - grand) ** 2))
    return max(0.0, ss_between / ss_total)


def _compute_group_icc(x: np.ndarray, labels: np.ndarray) -> float:
    frame = pd.DataFrame({"x": x, "group": labels})
    group_sizes = frame.groupby("group").size().to_numpy(dtype=float)
    group_means = frame.groupby("group")["x"].mean()
    grand = float(frame["x"].mean())
    ss_between = float(np.sum(group_sizes * np.square(group_means.to_numpy(dtype=float) - grand)))
    ss_total = float(np.sum(np.square(frame["x"].to_numpy(dtype=float) - grand)))
    ss_within = max(0.0, ss_total - ss_between)
    n = float(len(frame))
    k = float(group_sizes.size)
    if k <= 1 or n <= k:
        return float("nan")
    ms_between = ss_between / (k - 1.0)
    ms_within = ss_within / max(n - k, 1.0)
    k0 = (n - np.sum(np.square(group_sizes)) / n) / (k - 1.0)
    if k0 <= 0:
        return float("nan")
    icc = (ms_between - ms_within) / (ms_between + (k0 - 1.0) * ms_within)
    return max(0.0, float(icc))


def _compute_group_kw(x: np.ndarray, labels: np.ndarray) -> float:
    ranks = rankdata(x, method="average")
    frame = pd.DataFrame({"rank": ranks, "group": labels})
    stats = frame.groupby("group").agg(n=("rank", "size"), rank_sum=("rank", "sum"))
    n = float(len(frame))
    k = float(len(stats))
    if n <= k:
        return float("nan")
    h_stat = (12.0 / (n * (n + 1.0))) * float(np.sum(np.square(stats["rank_sum"]) / stats["n"])) - 3.0 * (n + 1.0)
    eps2 = (h_stat - k + 1.0) / (n - k)
    return max(0.0, float(eps2))


def _prep_group_data(x: np.ndarray, group_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    labels = np.asarray(group_labels).reshape(-1)
    mask = np.isfinite(arr) & np.asarray(pd.notna(labels), dtype=bool)
    return arr[mask], labels[mask]


def _feature_columns(feature_table: pd.DataFrame) -> list[str]:
    protected_prefixes = ("label.", "truth.", "aligned.", "pred.")
    return [
        col
        for col in feature_table.columns
        if col not in {"subject_id", "time_seconds"}
        and not col.startswith(protected_prefixes)
        and pd.api.types.is_numeric_dtype(feature_table[col])
    ]


def _better_score(new_score: float, old_score: float) -> bool:
    if not np.isfinite(old_score):
        return True
    if not np.isfinite(new_score):
        return False
    return bool(new_score > old_score)
