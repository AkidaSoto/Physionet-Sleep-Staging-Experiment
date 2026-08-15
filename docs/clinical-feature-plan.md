# Clinical Feature Plan for UCDDB Modeling

## Purpose

This is the practical feature plan for modeling on UCDDB.

It is intentionally separate from the research review.

The point is to decide:

- which features belong in the core model story
- which features are allowed but secondary
- which features are exploratory only
- how we want to handle context without turning the project into a black box

## Core principle

The project should stay clinically interpretable.

The algorithm should be built to notice the same kinds of signal patterns a trained human would notice, then combine them consistently and at scale.

It should not depend on hidden or abstract signal summaries as the main story.

In plain language:

- the model should help structure clinical reading
- not replace it with magic

## Guiding philosophy

- clinical interpretability over black box
- physiology first
- models should support reading, not replace it
- simplicity earns the first shot

## Feature policy

We will use four buckets.

### 1. Core

These are directly clinically meaningful and should define the project.

### 2. Secondary

These are still interpretable, but more derived or contextual.

### 3. Exploratory

These may be useful, but they are not the main clinical story.

### 4. Not allowed for v1 core modeling

These are not banned forever, but they should not drive the first version of the project.

## Implementation organization

This is the main organization we should use when building the pipeline.

### Implementation types

| Type | Meaning | Examples |
| --- | --- | --- |
| Basic math | Straightforward signal summary with clear formula | band power, RMS, mean, variance |
| Known implementation | Standard established algorithm we can implement or reuse | Pan-Tompkins for R-peaks / HR |
| Derived / external algorithm | Higher-level feature that likely needs custom logic or an external method | spindle detection, REM burst detection, respiratory-pattern derivation |

This matters because two features can both be clinically important, but have very different implementation effort.

## Sleep staging feature plan

### Sleep staging feature table

| Bucket | Feature | Signal(s) | Clinical meaning | Implementation type | v1 note |
| --- | --- | --- | --- | --- | --- |
| Core | Delta band power | EEG | slow-wave / deep-sleep content | Basic math | include |
| Core | Theta band power | EEG | lighter sleep / transition-related content | Basic math | include |
| Core | Alpha band power | EEG | wakefulness / relaxed wake tendency | Basic math | include |
| Core | Sigma band power | EEG | spindle-related sleep content | Basic math | include |
| Core | Beta band power | EEG | higher-frequency activation / arousal-adjacent content | Basic math | include |
| Core | Spindle feature | EEG | discrete spindle-like sleep marker | Derived / external algorithm | include when detector is ready |
| Core | Slow-wave feature | EEG | direct slow-wave prominence beyond raw delta power | Derived / external algorithm | include if derivation stays simple and readable |
| Core | Eye-movement activity | EOG | visible eye activity level | Basic math | include |
| Core | REM-like eye-movement burst feature | EOG | REM-associated eye bursts | Derived / external algorithm | include later or with external method |
| Core | Chin EMG tone | Chin EMG | muscle tone level | Basic math | include |
| Core | Chin EMG suppression | Chin EMG | REM-like tone drop | Derived / rule-based logic | include |
| Secondary | Delta/alpha ratio | EEG | interpretable stage contrast summary | Basic math | include if useful |
| Secondary | Theta/alpha ratio | EEG | interpretable transition summary | Basic math | include if useful |
| Secondary | Sigma/beta ratio | EEG | spindle-related vs activation balance | Basic math | include if useful |
| Secondary | Previous-epoch core features | EEG / EOG / EMG | explicit local context | Basic math over extracted features | include |
| Secondary | Next-epoch core features | EEG / EOG / EMG | offline-only local context | Basic math over extracted features | offline analysis only |
| Secondary | Adjacent-epoch deltas | EEG / EOG / EMG | change across epochs | Basic math over extracted features | limited use |
| Secondary | Simple readable interactions | mixed | combinations a human would recognize | Derived / rule-style | limited use |
| Exploratory | Hjorth parameters | EEG | abstract signal-shape summary | Basic math | later only |
| Exploratory | Spectral entropy | EEG | abstract irregularity summary | Basic math | later only |
| Exploratory | Sample entropy | EEG | abstract complexity summary | Known implementation | later only |
| Exploratory | Wavelet summaries | EEG / EOG / EMG | transformed multiscale summary | Known implementation | later only |
| Exploratory | Generic complexity measures | mixed | mathematically interesting but less clinical | Derived | later only |

### Sleep staging not allowed for v1 core

| Not allowed as core story | Why |
| --- | --- |
| Large automated feature dumps | too easy to lose the clinical thread |
| Opaque latent features as the main explanation | breaks interpretability goal |
| Hard-to-explain feature families | weakens public project story |
| Dozens of tiny derivatives of the same EEG bands | adds noise without adding clinical meaning |

### Current EEG retention rule

For this repo, EEG reduction should follow a simple rule:

- keep anything explicitly used by the reference implementation
- keep residual-band features
- keep aperiodic features
- reduce only extra overlapping summaries that are not needed once the reference set is present

That means the reduction target is not the core band stack itself.

It is the extra overlap around it.

### EEG keep-first set

This is the set we should preserve unless diagnostics clearly justify a later change.

| Group | Keep-first features |
| --- | --- |
| Canonical bands | delta, theta, alpha, sigma, beta, gamma |
| Residual bands | delta_residual, theta_residual, alpha_residual, sigma_residual, beta_residual, gamma_residual |
| Aperiodic | aperiodic_exponent, aperiodic_intercept |
| Reference spindle / slow-wave summaries | splindex, sigma_beta_ratio, slow_wave_envelope, slow_wave_index, slow_wave_ratio_power |

### EEG reduction target

If we reduce EEG features later, cut order should be:

1. extra summaries that restate the same distinction
2. non-reference overlap inside the same family
3. anything that is weak in diagnostics and redundant with a stronger retained feature

We should not cut a reference-used feature just because another feature looks cleaner on paper.

## Apnea / respiratory-event feature plan

### Apnea feature table

| Bucket | Feature | Signal(s) | Clinical meaning | Implementation type | v1 note |
| --- | --- | --- | --- | --- | --- |
| Core | Airflow amplitude | Airflow | visible breathing magnitude | Basic math | include |
| Core | Airflow reduction | Airflow | hypopnea/apnea-style reduction | Basic math / rule-based thresholding | include |
| Core | Near-flat airflow pattern | Airflow | near-cessation of airflow | Derived / rule-based logic | include |
| Core | Thoracic effort amplitude | Thoracic effort | chest effort level | Basic math | include |
| Core | Abdominal effort amplitude | Abdominal effort | abdominal effort level | Basic math | include |
| Core | Preserved-versus-reduced effort pattern | Thoracic + abdominal effort | helps distinguish event behavior | Derived / rule-based logic | include |
| Core | Thoracoabdominal synchrony / paradox pattern | Thoracic + abdominal effort | mismatch or paradoxical effort pattern | Derived / external algorithm | include later or with external method |
| Core | Oxygen saturation level | SpO2 | oxygenation state | Basic math | include |
| Core | Oxygen desaturation presence | SpO2 | clinically recognized desaturation event | Derived / rule-based logic | include |
| Core | Heart rate | ECG | cardiac response to disturbance | Known implementation | Pan-Tompkins or equivalent R-peak path |
| Core | Mean RR interval | ECG | direct beat-timing summary | Known implementation | derived from R-peaks |
| Core | SDNN | ECG | basic clinically legible HRV | Known implementation | derived from R-peaks |
| Core | RMSSD | ECG | basic clinically legible HRV | Known implementation | derived from R-peaks |
| Secondary | Airflow + effort interaction | Airflow + effort | obstruction-style joint pattern | Derived / rule-style | include if helpful |
| Secondary | Airflow + SpO2 interaction | Airflow + SpO2 | respiratory change with desaturation | Derived / rule-style | include if helpful |
| Secondary | Limited context windows | mixed | explicit short-term event context | Basic math over extracted features | include |
| Secondary | Limited additional HRV summaries | ECG | small extension beyond core HRV | Known implementation | only if useful |
| Exploratory | EDR | ECG | respiration proxy from ECG | Known implementation / external algorithm | later only |
| Exploratory | LF/HF | ECG | more abstract HRV decomposition | Known implementation | later only |
| Exploratory | Nonlinear HRV | ECG | mathematically richer HRV summary | Known implementation | later only |
| Exploratory | Entropy | mixed | abstract irregularity summary | Known implementation | later only |
| Exploratory | Wavelet summaries | mixed | transformed time-frequency summary | Known implementation | later only |
| Exploratory | Morphology-heavy ECG summaries | ECG | detailed ECG-shape descriptors | Derived | later only |
| Exploratory | Slope / recovery / curvature families | Airflow / SpO2 / ECG | extra derivative-heavy summaries | Derived | later only |

### Apnea not allowed for v1 core

| Not allowed as core story | Why |
| --- | --- |
| Huge derivative families around airflow, HR, or SpO2 | too far from the clean clinical story |
| ECG-only framing as the main project story | ignores richer PSG physiology available in UCDDB |
| Black-box feature embeddings treated as core evidence | breaks interpretability goal |

## External algorithm slots

These are the places where your later external algorithms should plug in.

| Feature area | Likely source | Why it belongs here |
| --- | --- | --- |
| Spindle feature | external detector or custom rule set | clinically important but not just a raw summary |
| Slow-wave feature | external detector or custom rule set | more readable than a pile of abstract EEG summaries |
| REM-like eye bursts | external detector or custom rule set | clinically meaningful event-style feature |
| Thoracoabdominal paradox / synchrony | external detector or custom rule set | useful apnea physiology if implementation is solid |
| EDR or derived respiration proxy | external detector or custom rule set | interesting, but should stay secondary or exploratory unless it proves especially valuable |

## Context policy

Context is allowed, but it should be explicit and controlled.

We do not want context handling to quietly become the whole model.

### Allowed context for v1

- previous-window core feature values
- next-window core feature values for offline analysis only
- short rolling summaries of core features
- simple interaction terms between clinically meaningful features
- HMM or similar transparent sequence smoothing

### Context that should stay secondary or exploratory

- large stacked multi-window feature banks
- complicated temporal embeddings
- abstract sequence representations that are hard to explain clinically

## Baseline model plan

The first model family should stay interpretable.

### Preferred v1 models

- decision tree as a sanity-check baseline
- random forest
- gradient-boosted trees
- HMM or simple transition smoothing on top of per-window predictions

### Why this is the right first step

- we can inspect feature importance
- we can test whether context helps without hiding everything in a deep model
- we can compare physiologic features directly against one another

## Feature diagnostics plan

This part is a core deliverable, not a side exercise.

We want to know which features are actually useful before we overbuild the modeling stack.

### Per-feature review

For each candidate feature:

- clinical meaning
- signal source
- expected behavior
- missingness / artifact sensitivity
- redundancy with other features

### Discrimination diagnostics

For binary tasks or one-vs-rest framing:

- AUC
- **Cohen's d**

For sleep staging:

- one-vs-rest AUC by stage
- Cohen's d for selected clinically meaningful contrasts
  - Wake vs N2
  - N2 vs REM
  - N2 vs N3

For apnea:

- apnea-event vs non-event AUC
- Cohen's d for major core features
  - airflow reduction
  - effort pattern
  - SpO2 change
  - HR / HRV response

### Redundancy diagnostics

- correlation matrix
- rank-correlation where needed
- feature clustering

### Model-level diagnostics

These are not per-feature diagnostics, but they still matter:

- AUC
- F1
- Cohen's kappa where appropriate
- confusion matrix
- subject-wise spread

## Approved v1 feature shortlist

### Build first

| Task | Feature set |
| --- | --- |
| Sleep staging | delta, theta, alpha, sigma, beta band powers |
| Sleep staging | eye-movement activity |
| Sleep staging | chin EMG tone |
| Sleep staging | chin EMG suppression |
| Apnea | airflow amplitude and airflow reduction |
| Apnea | thoracic and abdominal effort amplitude |
| Apnea | oxygen saturation level and desaturation presence |
| Apnea | heart rate, mean RR, SDNN, RMSSD |
| Both | previous-window versions of core features for explicit context |

### Build after core extraction is stable

| Task | Feature set |
| --- | --- |
| Sleep staging | spindle feature |
| Sleep staging | slow-wave feature |
| Sleep staging | REM-like eye-movement burst feature |
| Sleep staging | limited band ratios |
| Apnea | preserved-versus-reduced effort pattern |
| Apnea | thoracoabdominal synchrony / paradox pattern |
| Apnea | airflow + effort interaction |
| Apnea | airflow + SpO2 interaction |

### Hold for later unless clearly needed

| Task | Feature set |
| --- | --- |
| Sleep staging | Hjorth, entropy, wavelet summaries, generic complexity families |
| Apnea | EDR, LF/HF, nonlinear HRV, wavelet summaries, morphology-heavy ECG features |
| Apnea | large slope / recovery / curvature families |

## What success looks like

This plan is working if:

- the core features can be explained in clinical language
- the diagnostics clearly show which core features separate classes best
- the model gains from context are measurable and explicit
- we can defend the feature set to a technically strong outside reviewer without appealing to black-box behavior

## Recommended next step

Before writing extraction code, we should draft a small feature specification sheet with:

- feature_name
- task
- bucket
- signal_source
- clinical_meaning
- window_length
- context_allowed
- expected_direction
- diagnostics

That sheet should stay small and disciplined for v1.
