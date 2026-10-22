# Dashboard fresh review — October 22, 2026

Two independent reviews examined presentation flow and physiological claims. These are recommendations, not an authorization to rewrite the dashboard. The requested §1.2 terminology changes have been applied separately.

## Correct before presenting

- **N3 scoring:** Remove the claim that older adults can qualify with fewer waves above the amplitude threshold. Age affects slow-wave amplitude; that does not establish an alternative adult scoring threshold. Describe the implemented feature as AASM-inspired slow-wave coverage rather than an exact implementation of visual scoring.
- **Feature calculations and provenance:** The delta feature uses spectral magnitude, not the displayed log-power units. The aperiodic feature fits magnitude over different frequency ranges from the cited Lendner method; call it an adaptation rather than a replication with standard parameters. These are description corrections and do not require rerunning the model.
- **Spindle proxy:** The calculation counts seconds above a relative-sigma threshold, not validated individual spindle events. Its N2 association supports the rationale; it does not establish spindle-detection accuracy.
- **Breathing expectations:** A signed respiratory-effort waveform has no general “higher in REM” interpretation. Remove directional success/failure judgments for that raw waveform. Belt-envelope amplitude also needs a stronger justification before presenting a stage-specific expectation.
- **Eight-leaf next step:** The complexity sweep already uses the final Model C inputs. Replace “test on final Model C” with assessment of stage and patient trade-offs, followed by independent confirmation if the smaller model is selected.

## Improve the narrative

- Start interpretation with one concrete example: spindle-related activity in N2. Separate the physiological rationale, observed stage association, and prediction-reliance result.
- Keep low chin activity as a proxy for muscle atonia, and EOG spectral contrasts as proxies for eye activity. Neither establishes REM sleep alone.
- Make the discussion answer the three objectives: what is supported, what remains uncertain, and what decision follows. Avoid repeating metric values already visible in results.
- Qualify published-model comparisons as comparisons across studies and datasets. Human-scoring agreement is context, not a universal performance ceiling.
- Keep the main presentation focused on the retained EEG, eye and chin inputs; exploratory heart and breathing features can remain available for questions.

## Preserve

The patient distributions, separate Coimbra scorer comparisons, and explanation that agreement-only epochs form a selected subset are useful. The feature explorer and explicit distinction between physiological association and model reliance are worth keeping.

## Study sheet

The updated standalone copy is `docs/presentation-study-sheet.md`. It follows the current dashboard order and includes feature sources, physiological rationale, measurement caveats, current validation results and short answers for questions. The original Claude artifact remains at https://claude.ai/artifact/D4P264NnThVc9WNSCeu8ex.

## Sources for physiological wording

- Adult visual-scoring recommendations: https://link.springer.com/article/10.5664/jcsm.26814
- Aperiodic activity and arousal, including the original calculation: https://elifesciences.org/articles/55092
- REM physiology: https://pubmed.ncbi.nlm.nih.gov/29161567/
- Thalamocortical oscillations: https://pubmed.ncbi.nlm.nih.gov/8235588/

Implementation checked against `src/physionet_sleep/algorithms/staging.py`, `src/physionet_sleep/epoch_windows.py`, `scripts/run_window_hmm_staging.py`, and `scripts/run_window_complexity.py`.
