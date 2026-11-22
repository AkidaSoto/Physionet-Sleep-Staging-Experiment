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

### Raw-waveform development

Run `python scripts/run_waveform_development.py reduction` for the cached 18-signal HGB comparison. Run `python scripts/run_waveform_development.py profile --device cpu --batch-size 16` to measure the residual waveform model on training patients. Use the same command with `xpu` if the installed PyTorch supports the Intel device. The Python dependencies are pinned in `requirements-waveform.txt` and the device-specific PyTorch wheel is installed separately.

Run `python scripts/run_waveform_development.py scratch --device cpu --batch-size 16` with the device and batch size chosen during profiling. The launcher checks that the saved Model C predictions reproduce all five development folds and stores subject partitions, per-epoch probabilities, resources, and paired results. It reads only development waveforms. The waveform loader memory-maps the existing WFDB files and the trainer encodes each epoch once per patient at inference.

To pause, create a file named `PAUSE` in the variant directory under `artifacts/exploration/cinc-waveform` or `artifacts/exploration/cinc-reduction18`. The neural run checkpoints before exiting. HGB exits between folds. Remove the pause file and repeat the same command to resume. Completed folds are reused. The waveform and feature-reduction jobs share a 24-hour active budget. Run them sequentially.

### Feature tuning and reduction

Run `python scripts/cache_staging_tuning_waveforms.py` to prepare the calibrated slow-wave and EOG candidates, then run `python scripts/run_staging_feature_tuning.py`. Experiments 44–46 use the same Boston development folds. Marker selection uses only training patients. A patient-separated inner comparison chooses the feature bundle before the outer fit. The reduced recipe keeps the selected settings and drops two inputs.

Fold recipes, probabilities, paired results, and the active-time ledger live in `artifacts/exploration/cinc-feature-tuning`. Create `PAUSE` in that directory to stop at a fit boundary. Remove it and repeat the command to resume from saved fits.

### Compact feature context

Run `python scripts/run_staging_feature_context.py --smoke` once for the training-only shape and checkpoint check. Run `python scripts/run_staging_feature_context.py` for the two matched temporal models and their fixed blends with HGB. The runner uses the original patient folds, with separate inner validation patients, and reads cached engineered features. It writes predictions and pass checkpoints under `artifacts/exploration/cinc-feature-context`. Create a `PAUSE` file there to pause at the next pass boundary, then remove it and rerun to resume.

### Feature recipe and context search

Install the optional `feature-search` dependencies and run `python scripts/run_staging_recipe_search.py --smoke`. The check exercises two feature families on training patients, compares the original window with the longer input, and reopens the saved trial study. Run `python scripts/run_staging_recipe_search.py` to start or resume the registered search. Set `--trials` before creating a study to choose the first-rung ceiling. The default is 20 attempts per fold.

The runner stores search proposals, resolved feature settings, marker checks, patient partitions, and probabilities under `artifacts/exploration/cinc-recipe-search`. The first rung uses one validation patient group, and the confirmation fit scores two disjoint groups before the frozen recipe reaches the outer fold. A matched control uses the selected features with the original 150-second window. Create `PAUSE` in the experiment directory to stop between fits and remove it before rerunning.

### Error turnover and interpretable interactions

Experiments 54–56 use the archived Boston development folds. Run `python scripts/audit_staging_recipe_errors.py` followed by `python scripts/audit_staging_regression_features.py` to compare the new and recurring errors. Run `python scripts/audit_staging_tree_interactions.py` to reconstruct the trees in the specified diagnostic fold and measure their response to feature-group interventions and a tempered class-weight refit.

Install the optional `glassbox` dependency and run `python scripts/run_staging_glassbox.py --smoke` to verify exact term decomposition on inner patients. Run the same command without `--smoke` to compare matched additive and pairwise boosting models across all five folds. The outputs under `artifacts/exploration/cinc-error-interactions` include stage contributions, keyed predictions, paired scores, and representative decision traces. Create `PAUSE_GLASSBOX` in that directory to stop at a fit boundary. Remove it and rerun the command to resume completed folds.

### Sustained errors and directed temporal staging

Run `python scripts/audit_staging_sustained_errors.py` to reconstruct the shared sustained errors from archived predictions and compare the cached measurements with matched N2, N1 and N3 control epochs. The audit tables are under `artifacts/exploration/cinc-gated-temporal`.

Run `python scripts/run_staging_gated_temporal.py --smoke` to check interrupted and uninterrupted training on inner patients. Run without `--smoke` to fit the three matched temporal variants. The directory contains keyed stage probabilities, per-epoch local, past and future logits, gate values, and fit checkpoints. Create `PAUSE` there to stop after a saved training pass. Remove it and rerun to resume.

### Complementary sleep-stage markers

Run `python scripts/run_staging_complementary_markers.py --smoke` to verify the detector clocks and calibrated slow-wave reference on two inner training patients. Run without `--smoke` to fill the one-recording-at-a-time cache and measure each marker family in the matched pair probes. Create `PAUSE_CACHE` under `artifacts/exploration/cinc-complementary-markers` to stop at a saved recording boundary. The probe tables, marker profiles and extraction cost are in that directory.

With the marker cache complete, run `python scripts/run_staging_complementary_model.py --smoke` to verify the original prediction path and exact training resume. Run without `--smoke` to fit the expanded recipe across the five development folds. The archived gated network supplies the comparison. The output includes keyed predictions, stage-score contributions, and matched patient and error-run results.

Run `python scripts/audit_staging_targeted_refinement.py` to replay the frozen expanded checkpoints, audit nominal kappa, compare two N1/N2 decision adjustments on the original folds and measure separate K-like and recent-event sensitivities. The script chooses adjustments on the existing inner patients. Outputs under `artifacts/exploration/cinc-targeted-refinement` include keyed fold scores and patient-paired kappa intervals.

### Stage-pair head and compact measurement revisit

Run `python scripts/audit_staging_n2_interruptions.py` to screen the fixed interruption cue against the Boston annotations. Run `python scripts/run_staging_pair_head_revisit.py` for the three fitted N1/N2 comparisons using frozen checkpoints and inner validation patients. The archived interruption screen enables the split event histories in the registered comparison. The DREAMS event audit is replayable with `python scripts/audit_staging_marked_k_events.py` and its data preparation notes in `artifacts/exploration/cinc-stage-pair-revisit/k_events/PROTOCOL.md`.

Run `python scripts/run_staging_compact_marker_revisit.py --smoke` before the five reduced-information fits, then repeat without `--smoke`. Use the pause file documented in the runner to stop at a saved pass boundary. The summary under `artifacts/exploration/cinc-stage-pair-revisit/reduction` logs all fold, kappa and stage-specific guards.

### Wake/N1 failure audit and fitting-patient budget

Run `python scripts/audit_staging_wake_n1.py` and `python scripts/audit_staging_alpha_occupancy.py` for the archived directional, timing and feature checks. The 128-patient diagnostic in `scripts/run_staging_wake_n1_data_curve.py` compares with all 230–231 fitting patients per fold in the original 320-patient cohort. Use the original five folds and full fitting partitions for subsequent model comparisons.

Run `python scripts/run_staging_wake_n1_followup.py` for experiments 71–72. It fits the theta probe on inner fitting patients, replays saved inner checkpoints for the Wake/N1 adjustments, and reports both error directions on the original five outer folds.

Experiment 73 adds the 474 subjects listed in `artifacts/exploration/cinc-all-additional-training/subjects.txt` only to fitting. Run `python scripts/cache_staging_expansion.py --smoke DEV_SUBJECT` to compare the compact extraction with the archived model inputs for an original development subject. Run without `--smoke` to process the added nights as their waveforms arrive. The script writes experiment-local caches and publishes `additional_epochs.parquet` after all nights pass. `PAUSE_CACHE` requests a stop at a recording boundary.

Run `python scripts/run_staging_all_additional_training.py --smoke` for the inner-patient encoding and resume check. After the caches are ready, run without `--smoke` to fit all five comparisons on the original folds. The runner matches the archived optimizer update counts and keeps each added patient out of validation and evaluation. `PAUSE` saves a checkpoint and stops. Remove the corresponding pause file and rerun to resume at the cached records or saved fitting state.

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
