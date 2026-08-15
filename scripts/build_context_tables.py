from __future__ import annotations

from pathlib import Path


def main() -> None:
    from physionet_sleep.analysis.reference_data import load_ucddb_reference_dataset
    from physionet_sleep.experiments.sequence_data import build_flat_context_frame
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    output_root = Path("artifacts/context_tables")
    output_root.mkdir(parents=True, exist_ok=True)

    table = load_ucddb_reference_dataset()
    specs = build_default_experiment_specs(table)

    for spec_name in ("staging_plus_shared", "apnea_plus_shared"):
        spec = specs.get(spec_name)
        if spec is None:
            continue
        context_frame = build_flat_context_frame(
            table,
            feature_columns=spec.feature_columns,
            target_column=spec.target_column,
            context_radius=2,
        )
        out_path = output_root / f"{spec_name}_ctx5.parquet"
        context_frame.to_parquet(out_path, index=False)
        print(f"{spec_name}: {context_frame.shape} -> {out_path}")


if __name__ == "__main__":
    main()
