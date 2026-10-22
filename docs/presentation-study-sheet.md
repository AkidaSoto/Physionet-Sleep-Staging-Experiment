# Sleep Staging Study Sheet

Updated October 22, 2026 · @Akida

Updated standalone copy of the Claude presentation sheet supplied in this chat. Read before presenting; use the dashboard for selected examples rather than reading these notes aloud.

## The story to remember

**Question:** How far can explicit physiological measurements take automatic sleep staging when interpreted in context?

**Answer:** Twenty EEG, EOG and chin-EMG measurements support useful offline staging with a default tree ensemble. Context provides the largest gain. The measurements are inspectable, but individual explanations and reliability on difficult nights still need validation.

**Opening:** “I started from what changes physiologically across sleep, translated those cues into measurements, and tested which measurements actually helped on other patients and at another hospital.”

## Five physiology highlights

| State | What to say | Connection to my measurements |
| --- | --- | --- |
| Wake | Posterior alpha is especially visible during relaxed, eyes-closed wake. Its attenuation helps identify sleep onset; eyes-open wake may have little alpha. | Occipital alpha share is spatially relevant to that cue. Central alpha is a related spectral measurement, not a direct posterior-alpha measurement. |
| N1 | Alpha activity gives way to low-amplitude mixed-frequency activity, often theta, with slow eye movements (SEMs). It is a transition, so the cues overlap other states. | Theta shares and eye-channel contrasts quantify parts of this change. No single one is an N1-specific marker. |
| N2 | Spindles reflect thalamocortical activity; K-complexes are another defining event. Sustained sigma-band activity and a brief spindle are different things. | My spindle proxy emphasizes unusually strong sigma relative to adjacent alpha and beta bands. It behaves as expected for N2, but is not yet validated against annotated spindle events. |
| N3 | Large slow waves reflect synchronized cortical activity. Frequency content, absolute amplitude, scalp location and time occupied all matter. | Frontal slow-wave coverage is inspired by the amplitude/frequency/coverage scoring cue. Relative delta alone cannot establish the absolute-amplitude criterion. |
| REM sleep | The combination is crucial: relatively activated EEG, low tonic chin EMG consistent with muscle atonia, and possible rapid eye movements (REMs). REMs need not appear in every REM-sleep epoch; brief muscle twitches can occur. | EOG spectral activity plus relative chin tone are complementary. EOG also contains frontal EEG, and low chin tone alone does not establish REM sleep. |

Sources: [adult visual-scoring recommendations](https://link.springer.com/article/10.5664/jcsm.26814), [Steriade, McCormick & Sejnowski on thalamocortical oscillations](https://pubmed.ncbi.nlm.nih.gov/8235588/), and [Peever & Fuller on REM physiology](https://pubmed.ncbi.nlm.nih.gov/29161567/).

**Useful distinction:** SEMs and REMs describe eye movements; “REM sleep” names a sleep stage. The same fast eye activity can occur during wakefulness.

## Feature sources and rationale: all 20 inputs

The source of the physiological idea and the exact implementation are separate. “AASM-inspired” means I quantified a scoring cue; it does not mean I implemented the complete manual scoring rule.

| Measurements (number of inputs) | Rationale and provenance | Important qualification |
| --- | --- | --- |
| Spectral-background exponent (1) | Research associates spectral-background shape with arousal. My implementation fits spectral magnitude over 2–7 and 14–24 Hz. | Adapted, not a replication of Lendner’s signed 30–45 Hz power-spectrum slope. A larger exponent in my code means a steeper decline. Do not claim it directly measures excitation/inhibition balance. |
| Central delta feature (1) | Low-frequency central EEG activity is relevant to deeper NREM. | The current implementation computes median spectral magnitude at 1–4 Hz, despite the stored name `delta_power`. It is neither absolute slow-wave coverage nor integrated delta power. |
| Theta shares (2) | Slowing around sleep onset motivates theta measurements calculated over different windows. | Theta is also present outside N1. |
| Beta shares (2) | Faster activity can accompany wakefulness and arousals. | Scalp muscle activity contaminates faster EEG frequencies, especially above about 20 Hz. |
| Central alpha share (1) | Captures alpha-range central activity. | Do not equate a central derivation with posterior alpha. |
| Sigma share and sigma-burst proxy (2) | Sigma is associated with spindles; the adapted proxy counts seconds with a high sigma-to-alpha-plus-beta ratio. | Counts threshold-exceeding seconds, not discrete spindles. The stage association is encouraging, not event-detection validation. |
| Central total EEG power (1) | Provides broadband signal-level context alongside relative band shares. | Slow waves, movement and recording artifacts can all affect it. |
| Frontal slow-wave coverage and frontal delta share (2) | Coverage captures calibrated high-amplitude slow activity; relative delta captures its share of the spectrum. | The coverage algorithm filters and averages derivations. It approximates the visual cue rather than reproducing the entire N3 rule. |
| Occipital alpha shares (2) | Eyes-closed posterior alpha motivates epoch-spectrum and shorter-window measurements. | These are two implementations of a related cue, not independent physiological processes. |
| Eye power, fast EOG power, fast/slow ratio, eye activity and eye baseline (5) | Quantify EOG spectral contrasts and envelope activity relevant to eye movements and sleep state. | Project adaptations. None independently confirms REMs; blinks and frontal EEG can contribute. |
| Relative chin tone (1) | REM-related reduction of tonic skeletal-muscle activity motivates comparison with the night’s low-tone baseline. | My envelope-based log index is not a clinical diagnosis of atonia or conventional EMG power in dB. |

Sources to recognize: [Lendner et al., 2020](https://elifesciences.org/articles/55092) for the spectral-background rationale; [Whitham et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17574912/) for muscle contamination of faster EEG; adult scoring recommendations and REM physiology above for the stage cues. The precise bands, normalization, ratios and burst threshold are implementation choices in this project.

**If asked where the features came from:** “The physiological cues came from sleep-scoring recommendations and established sleep research. I used spectral and envelope calculations to quantify them, with explicit choices of bands, windows and normalization. The tests establish staging utility and stage association; waveform-level event validity is a separate question.”

## Dashboard walkthrough

### 1 Introduction

Use the EEG/EOG/chin-EMG sequence to establish the task, then show the stage table. Land one physiological point: REM sleep requires a combination of signals, while N3 introduces an absolute-amplitude cue that relative power alone cannot capture.

Scorer disagreement is relevant, especially around transitions. The published 83% reference is agreement with a majority score in the AASM program, not a universal human accuracy ceiling or a direct pairwise comparison with my model.

### 2.1 Data and testing

Dublin was early development. Boston is the main dataset: 320 development patients and 200 held-out patients. The held-out cohort has two recorded evaluation looks. Coimbra is the external dataset: 99 evaluated nights, two scorers, one recorded evaluation look for final Model C. Night 8 lacked frontal EEG and was excluded before scoring.

### 2.2 Features

Show **N2 → spindle bursts** first. N2 versus the other stages: effect size about **+0.90**, versus **+0.02** for broad spindle-band share. N2 and N3 have tied median burst-proxy values, so this is an N2-associated feature, not exclusive to N2.

Then show **N3 → frontal slow-wave fraction**, explaining frequency versus amplitude and electrode location. For REM, use **eye-movement power + chin tone**, emphasizing that the combination is more specific than either alone.

Do not claim that a feature’s name proves what it detects. EOG is the best example of why: a channel over the eye can carry both ocular and cortical information.

### 2.3 Model

Model C receives 20 measurements every six seconds across 150 seconds. It combines the scored 30-second epoch with two preceding and two following epochs. That fits the temporal continuity of sleep and the context scorers use. It is offline because future epochs and whole-night normalization are used.

The CNN comparison learns combinations of derived signals; it is **not** a raw-EEG end-to-end comparison. Named inputs make assumptions inspectable, but 500 trees do not become one readable clinical rule.

### 3.1 Performance

Start with **Model comparisons**, then show stage errors if needed. Context produces the main gain. N1 is still the weakest stage. The published-model table is context, not a head-to-head trial on identical test patients.

### 3.2 Reliance

“I swap one measurement’s values between test examples while leaving the labels, other inputs and model fixed. If performance falls, the model depended on that information.”

A small bar means little **additional** reliance; correlated measurements can substitute. The “Other” bars sum individual drops and are not joint-shuffle importance. Importance is not physiological causation or an individual explanation.

For timing, measurements that vary within an epoch are swapped at one six-second position. Measurements calculated once per epoch have five repeated values, so all five are swapped together. Both plots show **drop in macro-F1**, not significance or a p-value.

### 3.3 Validation

Use the graph to distinguish the headline result from variation between patients. Box: middle 50%; line: patient median; whiskers: patient min–max. The dot is the cohort-level score, not the average of the patient scores, so it need not match the median.

Coimbra compares the same predictions separately with each scorer. Scorer-to-scorer agreement supplies a within-cohort reference. Performance where the scorers agree is a selected, easier subset and must not be presented as the full-cohort result.

### 4 Discussion and conclusion

“Explicit physiological measurements plus temporal context support useful staging. The strongest interpretability result is that I can inspect and test the measurement assumptions. The remaining work is to verify explanations in difficult nights and establish reliability across more patients and recording systems.”

Next: reduce complexity and redundant inputs while checking stage-specific and patient-level costs. Review the eight-leaf result below before describing it as an experiment not yet run.

## Numbers to remember

| Result | Macro-F1 | κ | Accuracy |
| --- | ---: | ---: | ---: |
| Boston cross-validation, Model C | 0.773 | 0.716 | 78.6% |
| Boston held-out, 200 patients | 0.729 | 0.653 | 73.8% |
| Coimbra, 99 nights, scorer 1 | 0.761 | 0.709 | 77.4% |
| Coimbra, scorer 2 | 0.751 | 0.704 | 77.2% |
| Coimbra scorer 1 versus scorer 2 | 0.802 | 0.767 | 82.0% |

Context: single epoch **0.721** → full context model **0.777** → simplified Model C **0.773**.

The current saved complexity sweep already uses Model C’s adopted inputs: **eight leaves 0.763 versus default 0.773**, with **7,335 versus 30,134 learned values** in that sweep (about 76% fewer). The appropriate next step is to inspect its stage/patient trade-offs and decide whether to lock it for a future independent evaluation. These counts describe that complexity experiment, not every fitted fold.

## Answers that preserve the physiology

- **Why not use relative delta alone for N3?** A relative share can rise when other bands fall. The scoring cue also depends on absolute amplitude, frequency, coverage and frontal location.
- **Why not interpret EOG importance as eye movements?** The electrode measures a mixture. Waveform review or additional channels are needed to distinguish ocular events from EEG leakage.
- **Did the spindle proxy work?** It was associated with N2 in the expected direction and contributes to staging. Whether it detects annotated spindle events accurately is not established.
- **Can you explain a mistaken epoch?** I can inspect named measurements and their failure modes. I still need to verify an individual explanation against the waveform and scorer judgement.
- **Why is the patient median different from the headline score?** Patient medians summarize night-to-night variation; headline macro-F1 pools epochs or averages folds. They summarize different things.

## Backup navigation

S1: patient variation and validation details. S2: model development, simplification and parameter counts. S3: signal reliance and sleep structure. S4: alternative heart/breathing models. S5: history, collaboration and implementation. Keep these for questions.
