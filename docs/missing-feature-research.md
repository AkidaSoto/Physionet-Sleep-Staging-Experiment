# Missing feature research

This is the short research note for the remaining clinically readable features that are still missing or only partially implemented.

The goal here is not to collect every possible method.

It is to identify:

- what we can directly reuse
- what we can derive cleanly from the signals we already have
- what has a defendable external implementation path

## Bottom line

The cleanest path is:

| Feature | Best near-term path | Why |
| --- | --- | --- |
| REM-like eye-movement bursts | reuse bilateral EOG and copy a threshold detector style like YASA | interpretable, clinically legible, event-level |
| Preserved-vs-reduced effort pattern | derive from event-level effort-to-flow behavior relative to baseline | uses signals we already have and matches obstructive vs central logic |
| Thoracoabdominal synchrony / paradox | start with rolling thorax-abdomen correlation and cross-correlation lag, not full fancy phase-angle math | simpler and more robust for v1 |
| Flow flattening / flow limitation | derive a simple flatness index on inspiratory airflow shape | gives a concrete obstruction feature without black-box modeling |

## 1. REM-like eye-movement burst feature

## What we already have

- local imported reference: [external/st_vincent_analysis/staging/algo_sleep_eog.m](C:/Users/johnn/OneDrive/Documents/GitHub/Physionet/external/st_vincent_analysis/staging/algo_sleep_eog.m)
- current repo output:
  - REM-band EOG power
  - REM-band variance
  - basic eye-movement activity

That means we already have the coarse REM-related EOG summary layer.

What we do not yet have is a true event detector for rapid eye movements or REM bursts.

## Best external implementation to copy

The best direct implementation candidate is YASA's REM detector:

- [YASA REM detector docs](https://yasa-sleep.org/generated/yasa.rem_detect.html)
- [YASA detection source](https://github.com/raphaelvallat/yasa/blob/master/src/yasa/detection.py)

Why it fits:

- requires left and right EOG, which we already preserve as `left_eye` and `right_eye`
- explicitly detects REM events, not just epoch power
- uses interpretable thresholds:
  - REM band roughly 0.5 to 5 Hz
  - amplitude bounds
  - duration bounds
  - prominence threshold
- restricted to REM sleep if hypnogram is available

The YASA docs state the detector is based on amplitude thresholding of the negative product of filtered LOC and ROC, with defaults:

- amplitude: 50 to 325 µV
- duration: 0.3 to 1.2 s
- REM band: 0.5 to 5 Hz

## Recommended repo implementation

Do not replace the current `EOGRemFeatureGenerator`.

Instead add a second layer:

- `REMBurstFeatureGenerator`

Suggested outputs:

- `rem_event_mask`
- `rem_event_density_epoch`
- `rem_burst_mask`
- `rem_burst_count_epoch`
- `mean_rem_event_amplitude`
- `mean_rem_event_duration`

Suggested logic:

1. use `left_eye` and `right_eye` if both exist
2. bandpass 0.5 to 5 Hz
3. compute a conjugate eye-movement signal
4. detect candidate REM events using amplitude, duration, and prominence rules
5. define a burst as at least 2 to 3 REM events within a short window like 5 to 10 s
6. summarize burst density per 30 s epoch

## Recommendation

This is the best missing feature to copy from an external implementation first.

It is clinically legible and low-risk.

## 2. Preserved-versus-reduced effort pattern

## Key clinical point

This feature is not just “is effort present.”

It is about whether airflow reduction is disproportionately larger than effort reduction.

That is one of the clinically meaningful signatures of obstruction.

Relevant references:

- [AASM respiratory event scoring update](https://pmc.ncbi.nlm.nih.gov/articles/PMC3459210/)
- [Randerath et al. 2013: noninvasive obstructive vs central hypopnea algorithm](https://pubmed.ncbi.nlm.nih.gov/23450252/)
- [Berry et al. 2018: effort-to-flow resistance surrogate](https://pubmed.ncbi.nlm.nih.gov/29734977/)

The AASM update is especially useful conceptually:

- obstructive hypopneas often show flattening, snoring, and sometimes thoracoabdominal paradox
- central hypopneas show flow reduction more in parallel with effort reduction

Berry et al. is useful because it operationalizes this with an effort / flow style surrogate.

## What we already have

- airflow envelopes
- apnea / hypopnea events
- snore signal
- thoracic effort amplitude
- abdominal effort amplitude
- summed effort amplitude

So this feature should be built on top of current outputs, not from scratch.

## Recommended repo implementation

Add an event-level feature generator after airflow + effort extraction:

- `EffortFlowPatternGenerator`

For each candidate event:

1. estimate pre-event baseline for:
   - airflow amplitude
   - thoracic amplitude
   - abdominal amplitude
2. compute event amplitudes
3. normalize event amplitudes to baseline
4. derive:
   - `flow_fraction`
   - `thoracic_fraction`
   - `abdominal_fraction`
   - `effort_fraction = max(thoracic_fraction, abdominal_fraction)` or mean of the two
   - `effort_flow_ratio = effort_fraction / max(flow_fraction, eps)`
5. mark:
   - `preserved_effort_flag` if effort stays materially higher than flow
   - `reduced_effort_flag` if effort and flow fall together

Useful secondary summaries:

- first-half vs second-half event ratio
- slope of effort / flow ratio across the event
- snore present / absent

## Recommendation

This should be implemented as a rule-based derived feature, not a learned model.

That matches the project philosophy and the literature better.

## 3. Thoracoabdominal synchrony / paradox

## What the literature says

Thoracoabdominal asynchrony is traditionally measured with RIP chest and abdomen signals.

The classic metric is phase angle from Konno-Mead / Lissajous style analysis, but there are two important caveats:

- newer technical work says the classic phase-angle method has limitations
- older comparison work found cross-correlation or maximum linear correlation more robust than loop analysis

Useful references:

- [Prisk et al. / NASA summary: cross-correlation and maximum linear correlation performed best](https://ntrs.nasa.gov/citations/20040088004)
- [Recent RIP ventilation paper: correlation between abdomen and thorax used directly, with -1 meaning paradox and +1 meaning synchrony](https://pubmed.ncbi.nlm.nih.gov/41056175/)
- [Recent technical note on limits of phase angle and the idea of global phase delay](https://pubmed.ncbi.nlm.nih.gov/38829281/)

## Best v1 choice

For this repo, the best v1 choice is not full phase-angle math.

It is:

- rolling Pearson correlation between thoracic and abdominal effort
- rolling lag at maximum cross-correlation

Why:

- easier to explain
- less fragile to weird breath shapes
- aligns with the literature saying correlation-based approaches are robust

## Recommended repo implementation

Add:

- `ThoracoAbdominalSynchronyGenerator`

Suggested outputs:

- `thor_abd_corr`
- `thor_abd_xcorr_lag_sec`
- `paradox_mask`
- `paradox_fraction_epoch`

Suggested logic:

1. split into breath-scale or short rolling windows
2. normalize thorax and abdomen locally
3. compute:
   - Pearson correlation
   - cross-correlation lag
4. define:
   - synchrony when correlation is strongly positive and lag is near zero
   - paradox when correlation is negative or lag indicates near-opposite motion

Possible later extension:

- optional phase-angle / global-phase-delay output for comparison only

## Recommendation

Use correlation first.

Only add phase-angle style metrics later if we want a literature comparison metric.

## 4. Flow flattening / inspiratory flow limitation

This is not one of the originally named missing features, but it is the most useful companion feature for preserved effort.

Relevant sources:

- [AASM update: flattening of inspiratory flow waveform is evidence of airflow limitation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3459210/)
- [ATS workshop report on noninvasive inspiratory flow limitation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5566295/)
- [Weighted polynomial approximation paper with explicit flatness-index framing](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467386/)
- [Nox support note describing the ResMed-style flattening index in plain language](https://support.noxmedical.com/hc/en-us/articles/205185506-How-does-the-flow-limitation-quantification-work)

The most practical v1 route is a simple flatness index on inspiratory breaths.

## Recommended repo implementation

Add:

- `FlowLimitationFeatureGenerator`

Suggested outputs:

- `flattening_index`
- `flow_limited_breath_mask`
- `flow_limitation_fraction_epoch`

Suggested logic:

1. detect inspiratory breaths from airflow
2. normalize each inspiratory segment
3. discard the first and last 25% of the inspiratory trace
4. compare the middle 50% against a rounded baseline shape or average inspiratory level
5. higher flatness / lower curvature => more flow limitation

## Recommendation

This is a very good partner feature for:

- preserved-vs-reduced effort
- obstructive-vs-central event interpretation

## How this changes our implementation plan

## Highest-value next ports

1. REM burst detector from bilateral EOG
2. flow flattening / flow limitation
3. effort-vs-flow event pattern
4. thorax-abdomen synchrony / paradox

## What should stay simple

- no deep REM microstate model
- no complicated phase-angle-only framework for v1
- no giant hypopnea subtype classifier yet

## What we can leverage immediately from our current repo

Already usable as inputs:

- bilateral eye channels
- EOG power / variance
- eye activity
- EMG tone / suppression
- airflow envelopes and event masks
- snore power
- thoracic / abdominal effort amplitudes
- SpO2 drop events

So the missing layer is mainly:

- event logic
- breath-shape logic
- cross-signal interaction logic

not major new raw preprocessing.

## Recommendation summary

If we want the cleanest next implementation sequence:

| Priority | Feature | Approach |
| --- | --- | --- |
| P1 | REM bursts | copy YASA-style bilateral EOG detector |
| P1 | Flow limitation | implement simple inspiratory flattening index |
| P2 | Preserved vs reduced effort | derive normalized effort-to-flow ratio per event |
| P2 | Thoracoabdominal paradox | rolling thorax-abdomen correlation + lag |

That sequence stays readable, clinically grounded, and reuses the code we already built.
