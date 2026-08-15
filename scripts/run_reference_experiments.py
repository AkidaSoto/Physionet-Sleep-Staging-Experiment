from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.experiments.tabular_runner import run_tabular_experiments
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        print("no_rows")
        return 1

    specs = build_default_experiment_specs(table)
    for spec in specs.values():
        spec.model_families = ("tree", "ensemble")
    out_dir = repo_root / "artifacts" / "reference_experiments"
    results = run_tabular_experiments(table, specs=specs, output_dir=out_dir)
    print(f"saved={out_dir}")
    print(f"experiments={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
