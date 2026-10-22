# Dashboard audit for the October 22 interview

The analysis is strongest when presented as a test of explicit physiological measurements, with context providing most of the performance gain. Lead with that finding. Treat interpretability as an outcome to examine, not something guaranteed by using named inputs or decision trees.

## Changes made

| Area | Finding | Change |
| --- | --- | --- |
| Abstract and objectives | Independence and explainability were stronger claims than the evidence supports. | Emphasized physiological interpretation, patient-separated testing, scoring history and inspectable inputs. |
| 2.1 Data | Coimbra was absent from the dataset table. “Scored once” obscured the earlier held-out model-family evaluation. | Added Coimbra and distinguished the adopted-model look from the two recorded cohort looks. |
| 2.2 Features | The expected-versus-observed feature view is a strong interpretability result, especially the N2 counterexample. | Preserved it and connected it explicitly to Results and Discussion. |
| 2.3 Model | Absolute language about normalization and fixed stage order was misleading. | Described reduced scale differences and recurring transition patterns. |
| 3.1 Signal evidence | EEG spectrum, frontal slow waves and occipital alpha sounded like competing explanations; network agreement sounded like physiological validation. | Consolidated the physiological reading, retained measured family reliance, kept timing separate, moved the network comparison to S7. |
| 3.2 Performance | Headline metrics were scattered. Recall was compared directly with a published scorer-agreement percentage. | Added accuracy, macro-F1, balanced accuracy and kappa together; lead with stage errors, precision, sensitivity, specificity and F1. Explained pooled stage metrics versus fold-mean headline scores. |
| 3.3 Transfer | Dublin was already present but its cohort and input differences were easy to miss. Coimbra’s full-cohort analysis was incomplete. | Named Dublin explicitly and added a separate Coimbra processing/final-result panel. |
| Published comparison | Parameter ratios and a small F1 gap could imply a controlled comparison. | Stated that U-Sleep used different test patients; the comparison is a reference, not head-to-head evidence. |
| Discussion | Too many model variants and repeated scores obscured what was learned about sleep. | Reframed around physiological cues, temporal continuity, and transfer limits. Removed claims that an error's cause is established by named inputs. |
| Preparation aid | A dense live cheat sheet competes with speaking and listening. | Keep preparation notes outside the dashboard. The unrequested brief page and dashboard link were removed. |

## Suggested presentation route

1. **Question:** How far can measurements tied to sleep physiology take staging?
2. **One physiological example:** N3 slow waves are direct; N2 band power is a useful failure of an initially plausible proxy. Show the expected-versus-observed feature view.
3. **One modeling decision:** A brief segment makes more sense beside its neighbours. Show the window figure without enumerating every model variant.
4. **Three results:** approximately 0.773 macro-F1 in cross-validation, 0.729 on held-out Boston patients, 0.754 for shared-input transfer to Dublin. Name the setting each time. Add Coimbra only when its final status is available.
5. **Interpretation:** The inputs are inspectable; N1, difficult individual nights and external validation remain limitations. Explain what you would validate next.

Keep implementation details for questions. Use the dashboard as visual evidence, not a script to read from top to bottom. In a short presentation, skip experiment history, the full feature catalog, parameter-count plots and network attribution unless asked.

## What still needs more analysis

- **No retraining is required for the current consolidation.** Existing family importance and feature distributions support the revised descriptive presentation. No new importance numbers were manufactured.
- **A single overall EEG importance number would require a new joint perturbation analysis.** Do not sum separately shuffled spectrum, spindle and topography drops: correlated inputs and interactions make those drops non-additive. If this is pursued later, shuffle all EEG families together on the same held-out folds and preserve the existing model.
- **Individual explanations need individual checks.** A useful next interpretability result would pair a correct prediction and an error with the measured waveform cues, tree attribution and a clinician's assessment. Global importance alone does not establish why one epoch was misclassified.
- **N2 deserves further event-level validation.** The detector is more directly connected to spindles than broad band power, but that does not establish detection accuracy against annotated spindle events. K-complexes remain another useful target.
- **Validation is not untouched simply because patients differ from training.** The development folds have supported many model decisions; Dublin informed early development; the Boston held-out cohort has two recorded looks across model families. Preserve that history. Future confirmation should use a prospectively untouched cohort.
- **Clinical-use performance is a separate question.** The method uses future epochs and whole-night normalization. It supports offline scoring; real-time claims would require a causal window, causal normalization and new evaluation.

## Further simplifications worth considering after the interview

| Priority | Suggestion | Purpose |
| --- | --- | --- |
| High | Keep only one feature example open during the talk. | The stage tabs, feature list, distributions, trace examples and calculation tabs otherwise create too many navigation decisions. |
| High | Give each Results subsection one spoken claim before showing detail. | Stops the audience from comparing multiple tables while trying to follow the argument. |
| Medium | Move the full A–D model catalog in Methods into an expandable panel. | The window diagram and one sentence about the adopted model are enough for the main route. |
| Medium | Move the parameter-count scatterplot below a disclosure. | It is secondary to physiological evidence and not a controlled performance comparison. |
| Medium | Add patient-level uncertainty intervals to external results in a later analysis. | A pooled score hides between-night variation, especially in a 25-night cohort. Resample patients, not individual epochs. |
| Medium | Prospectively define performance and interpretability success criteria together. | A feature should improve a meaningful prediction or explanation claim, not merely sound physiological. |

## Data and analysis handoff

The four missing ISRUC subjects were 6–9. Their 20 files were downloaded using the existing downloader, with zero reported failures. The remote listing check found those to be the only missing or wrong-size files before the download.

Processing uses the existing `build_isruc_cache.py` recipe for subjects 59, 62, 6, 7, 8 and 9. The follow-on `finish_isruc_analysis.py` waits for all 100 processed nights, calls the existing guarded transfer analysis once for the adopted model, exports the status/results, then builds the dashboard. It skips redundant Boston cross-validation because the adopted model already has that result.

Read `showcase/content/generated/isruc-status.json` for the latest exported counts and result availability. Progress is in `logs/isruc_interview_build.log` and `logs/isruc_interview_analysis.log`; errors have matching `.err.log` files. The status is a timestamped file snapshot, not a live network monitor. Local preview updates from it; a remotely hosted site requires publishing the updated build.

Production checks use a separate `.next-production` directory so a build can run alongside the active local preview without corrupting its output.
