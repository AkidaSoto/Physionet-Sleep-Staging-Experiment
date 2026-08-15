from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.diagnostics.reference_runner import run_reference_diagnostics

    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        print("no_rows")
        return 1

    result = run_reference_diagnostics(table)
    out_dir = repo_root / "artifacts" / "reference_diagnostics"
    result.save(out_dir)
    print(f"saved={out_dir}")
    print(f"rows={len(table)}")
    print(f"features={result.metadata['feature_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
