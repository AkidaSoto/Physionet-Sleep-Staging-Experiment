# Pipeline separation plan

Keep the project split into four layers.

| Layer | Purpose | Repo path | Imported reference |
| --- | --- | --- | --- |
| Loaders | waveform loading, label loading, channel normalization | `src/physionet_sleep/io`, `src/physionet_sleep/labels`, `src/physionet_sleep/alignment` | `python_infra/connectors/*.py` |
| Algo execution | feature/event extraction and recipe-driven execution | `src/physionet_sleep/algorithms`, `src/physionet_sleep/pipelines`, `src/physionet_sleep/recipe`, `src/physionet_sleep/studies` | `python_infra/core/run_recipe.py`, `algo_def.py`, `study_def.py` |
| Diagnostics | feature screening, subject review, label/source sanity checks | `src/physionet_sleep/diagnostics` | `adda_diagnostics/*.m` |
| Experiments | parity checks, metric runs, subset studies, model comparisons | `src/physionet_sleep/experiments`, later `src/physionet_sleep/evaluation` | `python_infra/parity_check.py`, `run_single_algo.py`, `run_algo_subset.py` |

Current workflow entry:

- `src/physionet_sleep/runners/ucddb.py`
- `src/physionet_sleep/studies/ucddb.py`

That runner now separates:

1. `load_ucddb_inputs(...)`
2. recipe or direct algorithm execution
3. label-to-output alignment
4. diagnostics
5. experiments

Current default:

- recipe runner is the default path for `run_ucddb_pipeline(...)`
- direct `AlgorithmPipeline` execution is still available when explicitly passed in

First diagnostics to port from the imported files:

- feature effect screening
  - Cohen's d one-vs-rest
  - AUC-style separability per class
  - top-feature ranking
- subject diagnostics
  - worst/best subject review
  - per-subject effect summaries
- label diagnostics
  - label/source timelines
  - pairwise label sanity plots

First experiment utilities to port from the imported files:

- parity checks
- single-algo runs
- subset runs
- second-wise apnea metric checks

Immediate rule for this repo:

- algorithms should not load labels
- label loaders should not run algorithms
- diagnostics should read `DatasetRun`
- experiments should read `DatasetRun`
- the runner coordinates those stages

MATLAB infra check-in:

| Imported piece | What it means for us | Status |
| --- | --- | --- |
| `AlgoDef`, `StudyDef`, `RecipeBranch`, `AlgoRecipe`, `run_recipe` | keep the recipe/study runner as the main execution path | already mirrored in `src/physionet_sleep/recipe` and `src/physionet_sleep/studies` |
| `AlgoStore`, parquet stores | keep cache/persistence pluggable, not hard-wired into algorithms | base interface already mirrored; parquet persistence is the right default |
| `load_labels`, `LabelDef` | keep labels separate from algo execution | direction already matches current runner split |
| `apply_hmm_viterbi`, `learn_hmm_params`, HSMM helpers | decoder/post-processing should be its own small utility layer, not part of core orchestration | next useful port |
| `f1score_multiclass`, `score_stage_fit` | scoring helpers belong in diagnostics/evaluation, not the runner | next useful port |

What not to copy:

- no giant training orchestrator
- no MATLAB-style all-in-one build framework
- no tight coupling between cache, labels, diagnostics, and experiments

Near-term next ports:

1. scoring helpers for stage/apnea evaluation
2. HMM-style decoding utilities for context smoothing
3. parity helpers where we already have Python-side references
