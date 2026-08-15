from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    from physionet_sleep.analysis.reference_data import load_ucddb_reference_dataset
    from physionet_sleep.experiments.sequence_cnn import SequenceExperimentSpec, run_sequence_cnn_experiment
    from physionet_sleep.experiments.tabular_specs import build_default_experiment_specs

    output_root = Path("artifacts/sequence_cnn_baseline")
    output_root.mkdir(parents=True, exist_ok=True)

    table = load_ucddb_reference_dataset()
    specs = build_default_experiment_specs(table)
    base_spec = specs["staging_plus_shared"]

    sequence_spec = SequenceExperimentSpec(
        name="staging_plus_shared_cnn_ctx5",
        target_column=base_spec.target_column,
        feature_columns=base_spec.feature_columns,
        context_radius=2,
        normalization="medianiqr",
        cv_folds=5,
        epochs=20,
        batch_size=256,
        learning_rate=1e-3,
        hidden_channels=32,
        kernel_size=3,
    )

    result = run_sequence_cnn_experiment(
        table,
        spec=sequence_spec,
        validation="kfold",
        random_state=7,
    )

    result.summary.to_parquet(output_root / "summary.parquet", index=False)
    result.fold_metrics.to_parquet(output_root / "fold_metrics.parquet", index=False)
    result.predictions.to_parquet(output_root / "predictions.parquet", index=False)
    with (output_root / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(result.metadata, fh, indent=2)

    if not result.summary.empty:
        print(result.summary.to_string(index=False))
    else:
        print("No sequence CNN results produced.")


if __name__ == "__main__":
    main()
