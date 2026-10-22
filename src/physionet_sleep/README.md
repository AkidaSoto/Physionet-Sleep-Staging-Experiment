# PhysioNet Sleep Code Map

This package is organized around four layers:

- `io`, `labels`, `alignment`: dataset loading and label preparation
- `algorithms`, `pipelines`, `recipe`, `studies`: feature and event extraction
- `diagnostics`: analysis utilities that inspect data, labels, and model behavior
- `experiments`: model definitions, datasets, specs, and evaluation runs

## Runner split

`runners/ucddb.py` is the build-facing runner.

It handles:

- loading a record
- running the recipe or direct pipeline
- aligning labels
- attaching diagnostics
- attaching experiment outputs

`runners/modeling.py` is the model-facing runner.

It handles:

- resolving repo and artifact paths
- loading the reference feature tables
- preparing standard staging inputs
- saving experiment and diagnostic bundles
- launching the main model families through a shared interface

## Preferred entrypoints

For the model layer, use:

- `scripts/run_modeling.py`

Supported jobs:

- `sequence-baseline`
- `sequence-epoch`
- `stage-expert`
- `signal-expert`
- `reference-diagnostics`

The older `scripts/run_*.py` files are still available, but the preferred pattern is now:

1. put reusable logic in `src/physionet_sleep/experiments/`
2. put orchestration in `src/physionet_sleep/runners/modeling.py`
3. keep script files thin

## Design intent

The point of this repo is not only to run models. It should also show:

- clean separation between build, diagnostics, and modeling
- reusable experiment code instead of copy-pasted launch code
- explicit model families and entrypoints
- code that is easy to navigate and explain
