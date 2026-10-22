# Sleep Staging Model Map

## Purpose

This is the clean staging index for the current codebase.

It answers four practical questions:

- which staging model families are still worth keeping visible
- which runners produce the main comparison tables
- which branches are exploratory and should not be treated as headline results
- what each branch taught us

## Current headline results

All numbers below are from grouped subject-wise runs already saved in `artifacts/`.

| Role | Model | Macro-F1 | Accuracy | Kappa | Keep? |
| --- | --- | ---: | ---: | ---: | --- |
| Best overall | 30-second ensemble + learned Stage 2 smoother, `ctx9`, logit+confidence | 0.6826 | 0.6755 | 0.5796 | Yes |
| Best simpler smoother anchor | 30-second ensemble + learned Stage 2 smoother, `ctx5`, probability context only | 0.6681 | 0.6623 | 0.5628 | Yes |
| Best feature-sequence baseline | 6-second feature TCN | 0.6655 | 0.6494 | 0.5529 | Yes |
| Best connected hierarchical model | hierarchical epoch model, `eh1_ec1_ef1_s5`, raw output | 0.6714 | 0.6548 | 0.5611 | Yes |
| Best signal-expert branch | signal expert temporal combiner, `wp96 ctx5` | 0.6562 | 0.6540 | 0.5571 | No headline, keep as exploratory |
| 30-second feature CNN raw | centered feature CNN | 0.6591 | 0.6413 | 0.5440 | Secondary baseline |

## Recommended active staging surface

These are the staging entry points that still make sense to keep visible.

### 1. Paper-matched tabular + learned smoother

Primary runner:

- `scripts/run_paper_matched_stage2_smoothers.py`

Primary artifact anchor:

- `artifacts/paper_matched_stage2_smoothers/kfold5__none/staging_30s`

Why it stays:

- It is still the best-performing staging path in the repo.
- It is the cleanest comparison against the classical feature-based story.
- It gives both a stronger headline result and a simpler `ctx5` interpretability anchor.

How to read it:

- `ctx5/context_linear` is the cleaner, lighter learned smoother.
- `ctx9/context_linear_logit_conf` is the strongest score-focused variant.

### 2. Feature sequence baselines

Primary runners:

- `scripts/run_sequence_cnn_baseline.py`
- `scripts/run_sequence_cnn_6s_epoch20.py`
- `scripts/run_sequence_cnn_context_linear.py`

Primary artifact anchors:

- `artifacts/sequence_cnn_context_linear/kfold5__none`
- `artifacts/sequence_cnn_6s_epoch/kfold5__h10_c5_f5`
- `artifacts/sequence_cnn_6s_epoch/kfold5__h10_c5_f5__dilated_tcn`

Why they stay:

- They are the clean temporal baselines built from the same physiological feature set.
- The 6-second TCN is still the strongest non-tabular baseline worth comparing against.

How to read them:

- 30-second CNN raw is a baseline.
- 6-second TCN is the stronger temporal baseline.
- CNN + extra learned smoother did not become a primary path.

### 3. Connected hierarchical epoch model

Primary runner:

- `scripts/run_hierarchical_epoch_staging.py`

Primary artifact anchors:

- `artifacts/hierarchical_epoch_staging/kfold5__eh0_ec1_ef0_s5__hierarchical_epoch`
- `artifacts/hierarchical_epoch_staging/kfold5__eh1_ec1_ef1_s5__hierarchical_epoch`
- `artifacts/hierarchical_epoch_staging/kfold5__eh2_ec1_ef2_s5__hierarchical_epoch`

Why it stays:

- It is the cleanest connected alternative to the explicit Stage 2 smoother.
- It learned enough temporal structure that a separate Stage 3 smoother was usually redundant or harmful.

How to read it:

- `eh0_ec1_ef0_s5` shows that no cross-epoch context is worse.
- `eh1_ec1_ef1_s5` is the best hierarchical setting so far.
- `eh2_ec1_ef2_s5` did not improve further over `1/1`.
- For this family, use the raw model output as the result. Do not treat Stage 3 smoothing as the default.

### 4. Diagnostics and broad comparison utilities

Keep:

- `scripts/run_staging_diagnostics.py`
- `scripts/run_staging_sweep.py`

Why they stay:

- These are analysis tools, not headline models.
- `run_staging_diagnostics.py` is the clean path for confusion, subject-level errors, and variant comparison.
- `run_staging_sweep.py` is the broad comparison utility for feature-set and model-family ablations.

## Exploratory branches to keep but de-emphasize

These are worth preserving for reference, but they should not be described as the main staging path.

### Feature expert branch

- runner: `scripts/run_stage_expert_temporal.py`
- implementation: `src/physionet_sleep/experiments/stage_expert_temporal.py`

Why it is secondary:

- It was useful conceptually for expert decomposition.
- It did not become a competitive top-tier model.

### Signal expert branch

- runner: `scripts/run_stage_signal_expert_temporal.py`
- implementation: `src/physionet_sleep/experiments/stage_signal_expert_temporal.py`

Why it is secondary:

- It was the main attempt at a more paper-like signal expert design.
- Even after fixing the Stage 2 training bug and scaling it up, it remained below the best feature-based smoother path.

## Practical interpretation

What the staging work now says:

- The repo already covers enough model breadth for a strong project.
- The best score still comes from the feature-based ensemble plus learned 30-second smoother.
- The connected hierarchical model is the cleanest "more neural, still structured" alternative, but it has not beaten the smoother path yet.
- The expert branches were informative, but they are not the strongest end state.

## Suggested default staging comparison set

If we need a compact comparison table, use these four rows:

- 30-second ensemble raw
- 30-second ensemble + `ctx5` learned smoother
- 30-second ensemble + `ctx9` logit+confidence learned smoother
- 6-second feature TCN
- hierarchical epoch raw `eh1_ec1_ef1_s5`

If we need a trimmed interpretability-first table, use these three:

- 30-second ensemble raw
- 30-second ensemble + `ctx5` learned smoother
- hierarchical epoch raw `eh1_ec1_ef1_s5`

## Code cleanup rule

Going forward:

- keep the runners above as the visible staging surface
- treat queue helpers and one-off launch wrappers as disposable
- document new staging branches in this file before they become part of the comparison story
