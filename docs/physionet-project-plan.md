# PhysioNet Sleep PSG Project Plan

## Working title

PhysioNet Sleep Explorer

## Project goal

Build a polished data science project around open PhysioNet sleep PSG data that:

- shows signal-processing and modeling skill clearly
- stays clinically interpretable
- uses one coherent dataset story
- is strong enough for portfolio use without turning into a huge platform build

## Current working dataset

Primary v1 dataset:

- St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (`ucddb`)

Why this dataset first:

- one dataset supports both sleep staging and respiratory-event work
- includes sleep stage labels and respiratory-event labels on the same records
- open access
- manageable enough for local development

Potential later reference datasets:

- Sleep-EDF Expanded for staging comparison
- Apnea-ECG for ECG-only apnea comparison

## Project principles

- clinical interpretability over black box
- physiology first
- simple, defensible baselines before complex models
- clean presentation over unnecessary scope

## Scope

### In scope for v1

- local UCDDB download and organization
- parsing waveform metadata and annotation files
- clinically interpretable feature extraction
- baseline modeling for sleep staging and apnea-related tasks
- feature diagnostics and evaluation
- polished repo presentation

### Nice-to-have, not core

- lightweight waveform viewing
- preview plots for records and channels
- limited web/demo layer if it helps presentation

### Out of scope for v1

- a heavy production website
- a full hosted data platform
- black-box-first modeling
- lots of side quests or dataset expansions

## Repo structure

```text
docs/
scripts/
data/
notebooks/
src/        (only if we actually need app code)
python/     (only if we split scripts into modules)
```

## Core workstreams

### 1. Data foundation

- keep a clean local mirror of UCDDB
- inspect record/file structure
- parse stage and respiratory-event annotations
- create small metadata summaries

### 2. Feature engineering

- implement core clinically meaningful features
- keep secondary and exploratory features separate
- support explicit context features without hiding everything in the model

### 3. Baseline modeling

- sleep staging baseline
- apnea / respiratory-event baseline
- interpretable models first

### 4. Diagnostics and presentation

- per-feature diagnostics
- subject-wise evaluation
- clear plots, tables, and writeups

## Deliverables for v1

- clean repo structure
- working local UCDDB pipeline
- feature extraction code for core features
- baseline evaluation notebooks or scripts
- benchmark comparison writeup
- polished README / docs set

## Current docs

- `docs/physionet-project-plan.md`
- `docs/physionet-streams-and-benchmarks.md`
- `docs/clinical-feature-plan.md`

## Immediate next step

Start implementation from the smallest useful path:

1. inspect downloaded UCDDB files
2. define parsers for stage and respiratory-event labels
3. build core feature extraction
4. run diagnostics before expanding scope
