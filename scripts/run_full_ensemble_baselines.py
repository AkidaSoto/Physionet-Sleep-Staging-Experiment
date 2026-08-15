from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _log(message: str, *, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.analysis import load_ucddb_reference_dataset
    from physionet_sleep.experiments.tabular_runner import run_tabular_experiments
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    log_dir = repo_root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "full_ensemble_baselines.log"
    status_path = log_dir / "full_ensemble_baselines_status.json"

    _log("baseline_run_start", log_path=log_path)
    table = load_ucddb_reference_dataset(
        cache_dir=repo_root / "artifacts" / "ucddb_recipe_cache",
        root_dir=repo_root / "data" / "raw" / "ucddb",
    )
    if table.empty:
        _log("no_rows", log_path=log_path)
        status_path.write_text(json.dumps({"status": "failed", "reason": "no_rows"}, indent=2))
        return 1

    specs = build_default_experiment_specs(table)
    selected_names = ("staging_plus_shared", "apnea_plus_shared")
    selected_specs = {name: specs[name] for name in selected_names if name in specs}
    for spec in selected_specs.values():
        spec.model_families = ("ensemble",)
        spec.normalization_candidates = ("robust_by_subject",)
    for name, spec in selected_specs.items():
        _log(
            f"task_start name={name} target={spec.target_column} features={len(spec.feature_columns)}",
            log_path=log_path,
        )

    out_dir = repo_root / "artifacts" / "reference_experiments_full_ensemble"
    results = run_tabular_experiments(
        table,
        specs=selected_specs,
        output_dir=out_dir,
        validation="loso",
    )
    summary = {}
    for name, result in results.items():
        top = result.summary.head(1)
        if top.empty:
            continue
        row = top.iloc[0].to_dict()
        summary[name] = row
        _log(
            f"task_done name={name} macro_f1={row.get('macro_f1_mean')} "
            f"kappa={row.get('cohen_kappa_mean')} bal_acc={row.get('balanced_accuracy_mean')}",
            log_path=log_path,
        )

    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(out_dir),
                "tasks": list(results.keys()),
                "summary": summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _log(f"baseline_run_done output_dir={out_dir}", log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
