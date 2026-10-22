# Model Experiment Map

## Purpose

This is the code-facing map for the model layer.

It answers:

- where reusable model code should live
- which scripts are canonical entry points versus analysis wrappers
- which modules define model families versus runner mechanics
- how to keep future model work from turning into script sprawl again

## Separation of concerns

### `src/physionet_sleep/experiments/`

This is the reusable model library.

Put code here when it is any of the following:

- a model definition
- a dataset builder
- a cross-validation runner
- a shared experiment utility
- a reusable spec or result type

Current core modules:

- `tabular_specs.py`: named tabular experiment specs and feature-set definitions
- `tabular_runner.py`: grouped tabular training, validation splitting, scoring, and result packaging
- `sequence_data.py`: sequence dataset builders for centered, epoch-aligned, and hierarchical contexts
- `sequence_cnn.py`: 30-second CNN and 6-second TCN-style sequence training
- `stage_expert_temporal.py`: feature-expert branch and temporal combiner
- `stage_signal_expert_temporal.py`: signal-expert branch and temporal combiner
- `runner_utils.py`: shared runner mechanics used by scripts

### `scripts/`

This is the orchestration layer.

Scripts should stay thin. Their job is:

- parse arguments
- load datasets
- choose specs or model variants
- call reusable experiment code
- save artifacts and logs

They should not re-implement:

- split-tag logic
- frame cleaning
- featurewise norm-map application
- probability alignment
- context-window feature assembly
- fold-summary aggregation
- prediction-update merges

Those now belong in `src/physionet_sleep/experiments/runner_utils.py`.

## Shared runner utility surface

`runner_utils.py` is now the common home for the repeated model-launch plumbing:

- `build_split_tag`
- `load_selected_norm_map`
- `apply_featurewise_norm_map`
- `prepare_model_frame`
- `ordered_prediction_sequences`
- `summarize_numeric_folds`
- `align_probability_matrix`
- `attach_aligned_probabilities`
- `build_probability_context_frame`
- `apply_prediction_updates`

This is the right place for shared mechanics that are not themselves a model.

## Canonical model families

Preferred launcher:

- `scripts/run_modeling.py`

This is the common entrypoint for the cleaner model surface. Specialized research wrappers still exist, but this is the intended front door.

### 1. Tabular models

Core implementation:

- `src/physionet_sleep/experiments/tabular_specs.py`
- `src/physionet_sleep/experiments/tabular_runner.py`

Canonical entry points:

- `scripts/run_reference_experiments.py`
- `scripts/run_staging_sweep.py`
- `scripts/run_paper_matched_benchmarks_5fold_hsmm.py`

Use when:

- the main object is interpretable feature-based modeling
- you want grouped CV comparisons across feature sets and model families

### 2. Sequence feature models

Core implementation:

- `src/physionet_sleep/experiments/sequence_data.py`
- `src/physionet_sleep/experiments/sequence_cnn.py`

Canonical entry points:

- `scripts/run_sequence_cnn_baseline.py`
- `scripts/run_sequence_cnn_6s_epoch20.py`
- `scripts/run_sequence_cnn_context_linear.py`
- `scripts/run_hierarchical_epoch_staging.py`

Use when:

- the inputs are time-ordered feature sequences
- the comparison is about raw temporal modeling versus post-hoc smoothing

### 3. Expert models

Core implementation:

- `src/physionet_sleep/experiments/stage_expert_temporal.py`
- `src/physionet_sleep/experiments/stage_signal_expert_temporal.py`

Canonical entry points:

- `scripts/run_stage_expert_temporal.py`
- `scripts/run_stage_signal_expert_temporal.py`

Use when:

- the model story is decomposed by physiology or stage-specific expert
- the combiner is part of the comparison story

### 4. Diagnostics and comparison wrappers

Canonical entry points:

- `scripts/run_staging_diagnostics.py`
- `scripts/run_reference_diagnostics.py`
- `scripts/run_reference_diagnostics_linear_greedy.py`
- `scripts/run_task_diagnostics_no_feature_selection.py`

These should stay analysis-oriented. They are not the place to define new reusable model logic.

## Cleanup rules going forward

- New reusable model code goes in `src/physionet_sleep/experiments/`.
- New scripts should be thin wrappers around existing experiment modules.
- If two scripts need the same helper twice, move it into `runner_utils.py` or another `src` module immediately.
- Queue helpers, launch wrappers, and one-off sweep scripts should stay disposable.
- Before adding a new model family, add it to this file and note its canonical runner.

## Current cleanup result

The main model-runner duplication that used to exist across the smoother, hierarchical, CNN-context, ensemble, and benchmark scripts is now centralized through `runner_utils.py`.

That means the next cleanup pass should focus on:

- trimming obsolete exploratory scripts
- reducing stale artifact-specific wrappers
- tightening the distinction between headline model runners and diagnostics
