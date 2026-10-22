# Presentation study sheet

Review before the talk; let the dashboard carry the detail.

## Opening sequence

**My background → why this project → dashboard → sleep/PSG → my approach.** Skip reading the abstract. These are speaking prompts, not a script to memorize.

1. **After my background — why this project:** “This project brings together my interest in physiology and building tools that interpret physiological signals. I wanted to explore automatic sleep staging using measurements I could connect back to the physiology.”
2. **Introduce the dashboard:** “I'll show how I developed the measurements, what worked, and how the model performed on other patients and at another hospital. You're welcome to follow along, explore the tabs, or interrupt—we can dive deeper wherever you're interested. The supplementary sections have more detail for later.”
3. **§1.1 — sleep and PSG background:** “First, a little background. Sleep moves through different stages overnight. Their timing and organization help clinicians assess sleep and diagnose disorders. A polysomnogram, or PSG, records brain activity, eye movements, muscle tone, breathing and other signals together. Staging assigns a label to each 30-second segment, mainly using the brain, eye and chin signals.”
4. **§1.2 — show the physiology:** “These stages have recognizable patterns.” Point to **N2 spindles** and **REM eye movements plus reduced muscle tone**. Don't explain every cell or discuss feature calculations yet.
5. **Bridge into §2:** “Those are the patterns a scorer looks for. My task was to translate them into measurements and test whether a simple model could use them.”

## From methods to discussion

- **2.1 Data and testing:** Mention **38 numbered development/history entries**, covering features, models, audits and testing—not 37 independent validation studies. “There was a lot of iteration; I'll show the decisions that mattered.” Boston: 320 development patients, 200 held out. Coimbra: 99 nights, two scorers. Boston held-out has two recorded evaluation looks; Coimbra has one (#38).
- **2.2 Features:** **Don't tour every feature.** Start with **N2 spindle proxy**: physiological idea → calculation → observed behaviour. If useful, contrast **N3 slow-wave coverage** with relative delta: amplitude and scalp location matter. Keep REM eyes plus chin for questions. “Which would you like to look at more closely?”
- **2.3 Model:** Explain the diagram once: 20 measurements, 150-second window, default tree ensemble. Explicit inputs and a straightforward baseline. Future context makes it offline.
- **3 Results:** Model comparison → N1 remains difficult → reliance → validation. “I swap one measurement between examples, keep the labels fixed, and see how much performance falls.” Small drop means little *additional* reliance. Use the validation graph to discuss patient variation. Don't open every tab.
- **4 Close:** “The contribution is physiologically motivated staging whose measurement assumptions and reliance can be examined. Next I'd assess the smaller model's trade-offs and review difficult nights with a scorer.” Pause for discussion.

**Keep for questions:** supplementary details, every metric and individual formulas. Answer the question, then return to the main thread.

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

## Feature sources: the short answer

“The cues come from sleep-scoring recommendations and established physiology. My bands, windows, normalization and detectors translate those cues into measurements; I checked their stage associations and contribution to prediction.”

- **AASM-inspired:** posterior alpha, frontal slow waves, spindle-related activity and reduced chin tone. Measuring a cue isn't implementing the whole scoring rule.
- **Research adapted:** spectral-background shape and arousal ([Lendner et al.](https://elifesciences.org/articles/55092)). My bands and magnitude-based calculation differ; avoid claiming a direct excitation/inhibition measure.
- **My proxies:** sigma-threshold seconds, eye spectral contrasts and relative chin tone. Staging utility doesn't automatically validate individual event detection.

## Only a few numbers

**Macro-F1:** Boston cross-validation **0.773**; held-out Boston **0.729**; Coimbra scorer 1 **0.761**.

**Eight-leaf model:** **0.763** versus default **0.773**, about **76% fewer learned values** in the complexity sweep. Already tested with Model C inputs; next assess stage/patient trade-offs.

**Remember:** stage association asks whether a measurement behaves as expected; reliance asks whether the model uses it. Neither alone proves a physiological mechanism.

