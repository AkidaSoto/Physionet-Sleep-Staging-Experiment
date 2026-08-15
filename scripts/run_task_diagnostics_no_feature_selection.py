from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.diagnostics.reference_runner import run_reference_diagnostics
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        print("no_rows", flush=True)
        return 1

    specs = build_default_experiment_specs(table)
    selected_specs = {
        "staging": specs["staging_plus_shared"],
        "apnea": specs["apnea_plus_shared"],
    }

    out_root = repo_root / "artifacts" / "reference_diagnostics_by_task"
    out_root.mkdir(parents=True, exist_ok=True)

    for task_name, spec in selected_specs.items():
        print(
            f"task_start name={task_name} target={spec.target_column} features={len(spec.feature_columns)} rows={len(table)}",
            flush=True,
        )
        result = run_reference_diagnostics(
            table,
            target_column=spec.target_column,
            subject_column="subject_id",
            feature_columns=spec.feature_columns,
            enable_feature_selection=False,
        )
        task_dir = out_root / task_name
        result.save(task_dir)
        print(
            f"task_done name={task_name} classes={result.metadata['class_ids']} features={result.metadata['feature_count']}",
            flush=True,
        )

    print(f"saved={out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
