from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.diagnostics.reference_runner import run_feature_selection_search

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        print("no_rows", flush=True)
        return 1

    feat_score_path = repo_root / "artifacts" / "reference_diagnostics" / "feat_score_tab.parquet"
    if not feat_score_path.exists():
        print("missing_feat_score_tab", flush=True)
        return 1
    feat_score_tab = pd.read_parquet(feat_score_path)

    rows = run_feature_selection_search(
        table,
        feat_score_tab,
        stage_column="label.stage_seconds",
        subject_column="subject_id",
        mode="greedy_forward",
        model_family="linear",
        max_features=10,
        candidate_pool_size=20,
        progress=True,
    )
    out_path = repo_root / "artifacts" / "reference_diagnostics" / "feature_selection_greedy_forward_linear.parquet"
    rows.to_parquet(out_path, index=False)
    print(f"saved={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
