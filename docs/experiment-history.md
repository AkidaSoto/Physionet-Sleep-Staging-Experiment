# Sleep-staging experiment history

How the current model was reached, what was tried and dropped, and why. Read this before starting a new experiment, and add a row when one finishes.
Numbers are 5-class macro-F1 averaged over 5 subject-grouped folds (seed 7) unless stated. Dates come from git commits and artifact file times; detailed write-ups live in `artifacts/exploration/<folder>/FINDINGS.md`.


## Working rules

1. **Number every experiment when it is declared.** Add its row here and to the dashboard's Supplementary S1 straight away, not at the end.
2. **Declare variants and adoption rules before running.** Write them in the experiment's FINDINGS.md first.
3. **Validate a new feature before any model uses it.** It must separate its intended stage in the predicted direction, at least as well as the input it would replace (Cohen's d). Record the check as its own experiment.
4. **Protect the test sets.** The held-out set and ISRUC are scored only by the adopted model, every look is logged, and only headline numbers are reported.

## Where we are now

| | |
| --- | --- |
| Dataset | CinC 2018 (MGH), 320 of 994 labelled records, apnea-stratified; 288,923 epochs |
| Inputs | 11 one-second tracks + 16 features from EEG, EOG and chin EMG, over the epoch ±2 epochs (150 s), averaged to 6-s steps: 675 named inputs (no heart or breathing) |
| Model | Library-default HistGradientBoosting, class-balanced, no tuning, no decoder |
| Score | **0.775** macro-F1, κ 0.719 (folds 0.780 0.770 0.777 0.768 0.782) |
| Code | `scripts/run_window_hmm_staging.py` → `scripts/export_staging_dashboard_data.py`; modules `algorithms/eeg_tracks.py`, `analysis/epoch_windows.py` |

## Chronology

| # | Date | Question | What we found | Decision | Where |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-08-13 | Can a feature pipeline on UCDDB (25 apnea patients) stage sleep? | Feature recipe for EEG, EOG, chin, ECG, breathing built; reference diagnostics per task | Build on hand-made, named features | initial commit (08-15), `artifacts/reference_diagnostics*` |
| 2 | 2026-08-14 | How good is a paper-matched epoch classifier? | Bagged trees on epoch features 0.608; + HSMM smoothing 0.629 | Keep a temporal second stage | `paper_matched_benchmarks_5fold_hsmm` |
| 3 | 2026-08-21 | Do sequence networks beat smoothing? | 30-s CNN and 6-s TCN ~0.58–0.60 (validation); feature and signal "experts" 0.39–0.47 | Smoothing stays ahead on 25 subjects | `sequence_cnn_*`, `stage_expert_temporal` |
| 4 | 2026-08-22 | Does a hierarchical epoch network help? | Raw output 0.40 (2-fold); with a linear head 0.62 | No gain over smoothing | `hierarchical_epoch_staging` |
| 5 | 2026-09-09 | Publish the first dashboard | GitHub Pages site on UCDDB, bagged trees + smoothers | Dashboard v1 | commits 9c671ea–f8bfe65 |
| 6 | 2026-10-16 | Audit the dashboard for the Ripple talk | Best smoother 0.67; copy overstated results | Audit models and features in parallel | `staging_diagnostics` |
| 7 | 2026-10-16 | Why is UCDDB stuck at ~0.66? | Stage codes mapped wrong (UCDDB 1 = REM, 2 = S1) | Fixed `labels/ucddb.py` | coordinator |
| 8 | 2026-10-17 | Is the ceiling physiology or a bug? | 18 of 29 features drifted against labels: 6-s rows binned by sample count, not time (EEG ~6.5 min ahead by morning) | Bin by time | model-audit |
| 9 | 2026-10-17 | Is the learner the bottleneck? | 21-leaf bagged trees 0.609 vs default HistGradientBoosting 0.680 on the same features | Adopt default HGB, never tune | coordinator |
| 10 | 2026-10-17 | Is a second stage leaking? | Smoother fit on in-sample stage-1 probabilities; a real bug once the learner is strong | Out-of-fold probabilities for any second stage | model-audit |
| 11 | 2026-10-17 | Does amplifier gain limit us? | Per-subject median/IQR already cancels gain; oracle normalisation ≤ +0.018 | Add gain-free ratio features (+0.028), un-normalised | ratio-normalization |
| 12 | 2026-10-17 | Why do HRV features show no effect? | Peaks are fine (detector F1 ≥ 0.965); cleaned NN intervals and structural measures separate N3 from REM | Add cleaned HRV, breathing regularity, heart–breathing coupling | hrv-features |
| 13 | 2026-10-17 | Why do some subjects fail? | Flat dropouts scored as Wake, dead leads, mains noise | Mask flat-signal dropouts | subject-failures |
| 14 | 2026-10-17 | Result of the UCDDB fixes | 0.669 → 0.744 | UCDDB too small and R&K-scored to go further | coordinator |
| 15 | 2026-10-17 | Move to a bigger, cleaner dataset? | CinC 2018 (120 records): 0.749; µV-calibrated, F/C/O leads, AASM. Train UCDDB → test CinC: 0.679 | Switch to CinC 2018 | cinc2018 |
| 16 | 2026-10-17 | Better model or more data? | Learning curve still rising at 96 subjects; tuning no gain; HMM decoder +0.008 over smoother | Download to 320; HMM replaces smoother | cinc-models |
| 17 | 2026-10-17 | Can we cut features? | 33 → 16 costs 0.007 alone, 0.000 with HMM; event features +0.006 (below the 0.01 bar) | Rule: a feature must add ≥ 0.01 or replace ≥ 2 | cinc-models, cinc-features |
| 18 | 2026-10-17 | Why do CinC subjects fail? | Dead C4 lead, ECG artefact in EEG, few-epoch stages distort per-subject F1 | Report stage counts with per-subject scores | cinc-subjects |
| 19 | 2026-10-18 | Does real context beat smoothing? | 6-s features over ±2 epochs +0.02; 1-s tracks add more; 320 subjects: window + HMM 0.779 vs 16 + HMM 0.751 | Adopt the window model; retire smoother, TCN, hierarchical | cinc-context |
| 20 | 2026-10-18 | What does a network on the same signals do? | Attention CNN 0.769 (0.775 with HMM); saliency agrees with the trees | Keep as the black-box comparison | cinc-context |
| 21 | 2026-10-19 | Is "the first 6 s matters" real? | No: epoch-level features are identical across the 5 steps and trees split on the first copy. Per-epoch shuffling: fine evidence peaks 6–18 s in; previous epoch 3.8× the next | Report the corrected profile | `run_window_time_importance.py` |
| 22 | 2026-10-19 | How do we compare with published models? | U-Sleep 0.79 on CinC (~3.1M weights, ~15,600 people); human scorers 82.6% (N1 63%, N3 67%) | Comparison section on the dashboard | dashboard |
| 23 | 2026-10-21 | Is the decoder still needed? | With the window it adds 0.002 (3 folds up, 1 down) and lowers N1 F1 0.580 → 0.558 | Drop it; keep the transition table as a description | — |
| 24 | 2026-10-21 | Do heart and breathing earn their place? | Without them 0.7755 vs 0.7775. Heart + breathing only: 0.441 (2.5 min), 0.464 (5.5 min); with decoder 0.476 / 0.485; 4-class κ 0.38 | Open: kept for now; worth 0.002 with EEG present | `ablations/` |
| 25 | 2026-10-21 | How much does simplifying the learner cost? | Default 0.777 (30.5k values); 8-leaf 0.766 (7.2k); 4-leaf 0.758 (3.5k); boosted single splits 0.751 (3k, additive); one tree depth 8/6/4: 0.659/0.634/0.596. The depth-4 tree's splits are textbook cues (beta, occipital alpha, slow waves, chin atonia, 1/f slope) | Keep default boosting; show the sweep and the readable tree on the dashboard | `complexity/`, `run_window_complexity.py` |
| 26 | 2026-10-21 | Can heart and breathing alone do better? (literature: whole-night context, more cardiorespiratory features) | All 9 heart/breathing features + 3 tracks: 0.503 over 5.5 min, 0.524 over 10.5 min, 0.533 with the decoder; 4-class κ 0.46 (published 0.61–0.68 with whole-night networks on thousands of nights). Equal class weights: same κ, N1 F1 collapses to 0.12–0.23 | Longer context and more heart features help; the remaining gap needs whole-night sequence models and more data | `cardio_full*/` |
| 27 | 2026-10-21 | Does the window model transfer to another lab? | Shared signals only (no F/O leads, no 10–80 Hz chin track). With heart/breathing: CinC 0.769 → UCDDB 0.754. 27-signal version (8 tracks + 14 features): CinC 0.767 → UCDDB 0.750, κ 0.68 (best UCDDB-trained model ever 0.744; earlier epoch pipeline lost ~0.07 between these labs) | Transfer holds; per-night scaling makes the signals portable | `transfer/`, `run_window_transfer.py` |
| 28 | 2026-10-21 | Keep heart and breathing in the final model? (user decision) | They add 0.002 with EEG, eyes and chin present | Drop them: adopted model = 11 tracks + 16 features (27 signals), 0.775; heart/breathing stay in the explorer as tested-not-adopted, with their stand-alone result | `ablations/nocardio27` |
| 29 | 2026-10-21 | Does a cleaner EEG lead or excluding bad segments help? | Lead switching touches 2.4% of seconds; clean-lead tracks 0.7756 vs 0.7755. Excluding 1.6–3.3% of epochs (mostly movement in Wake, which the model already gets right) leaves the retained score at 0.776; training without them changes nothing; neighbour fallback hurts | Keep averaging and all epochs; the model is already robust to the defects it can detect | `exploration/signal-quality` |
| 30 | 2026-10-21 | Final held-out test (look #1, one batch of 4 models) | 200 new CinC patients (apnea-heavier): adopted 0.730 (κ 0.653), with heart/breathing 0.731, single-epoch reference 0.673, with decoder 0.740. Every model drops ~0.045–0.05 vs CV by the same amount, consistent with a harder set. First run had 199 (tr11-0452 failed to build under memory pressure); rebuilt and the same batch rerun, moved < 0.001 — a bug fix, counted within look #1 | Report as is; no model choice made from it | `holdout/LEDGER.csv` |
| 31 | 2026-10-22 | No-EEG model pushed further (heart, breathing, oxygen only) | Whole-night bidirectional GRU + TCN (seed-averaged) with breath-by-breath breathing and SaO2 summaries: 0.649, 4-class κ 0.602 (from 0.533 / 0.46). Single sensors: ECG + SpO2 0.604, belts 0.589, ECG 0.549. HMM no longer helps; recomputed HRV adds nothing | Wearable outlook only; final model stays EEG-based | `exploration/no-eeg/FINDINGS.md` |
| 32 | 2026-10-22 | Do the features behave as designed? | Each feature's observed stage separation vs its intended stage: N3/wake hold; spindle-band share d 0.02 for N2; single EOG channel carries frontal slow waves | Keep published designs; show expected vs observed in 2.2 | dashboard 2.2 |
| 33 | 2026-10-22 | Feature reduction, round 1 | prune6 (−3 residual powers, −emg_log_rms, −eog_low, −c_rel_delta): 0.7754 vs 0.7755 (21 signals); spindle additions ≤ +0.0006 | Keep the 21-signal set (F) | `feature_reduction/FINDINGS.md` |
| 34 | 2026-10-22 | Swap round 2 | swapA/B/AB pass the accuracy rule (−0.0014 to −0.0029) but kc_transients runs the wrong way (N2 < Wake) and eog_fast_resid d 0.18 | Validate a feature's meaning before testing | `feature_reduction/FINDINGS.md` |
| 35 | 2026-10-22 | Candidate marker validation (round 4 step A) | Pass: sawtooth_theta (REM vs rest d 0.94), alpha_dropout (ties o_rel_alpha, relative gate added post hoc); fail: rem_eye_bursts, kcomplex_events, slow_eye_movements | Only validated markers enter swaps | `feature_reduction/round4_validation.csv` |
| 36 | 2026-10-22 | Validated swaps (rounds 3–4) | Passed: swapS2 (spindle_bursts for aperiodic_intercept + gamma_residual, −0.0026), swap4_saw (−0.0019), swap4_alpha (−0.0012); failed: swapS1 (by 0.00002), swapS3, combo. Only spindle_bursts is used by the model (3rd-largest N2 importance); sawtooth/alpha rank last | Adopt swapS2 as model C (20 signals, config g20) | `feature_reduction/FINDINGS.md` |
| 37 | 2026-10-22 | Held-out look #2: model C (g20) | 0.7291, κ 0.653, accuracy 0.738 on the 200 held-out patients (look #1, 27 signals: 0.7296) | Reported as is; no model choice made from it | `holdout/LEDGER.csv` |

## Tried and rejected (don't repeat without a new reason)

- **Learners:** multinomial logistic (0.458), bagged trees (0.609), tuned HGB (no gain over defaults).
- **Temporal models on probabilities:** 5- and 9-epoch logistic smoothers (hard to explain), HSMM and time-binned HMM (no gain over plain HMM), the HMM on top of the window (+0.002, costs N1).
- **Sequence networks on 25–96 training subjects:** 30-s CNN, 6-s TCN, hierarchical epoch network, feature/signal experts.
- **Rolling averages as "context"** (not what context means; uninterpretable).
- **Percentile or label-informed normalisation** (≤ 0.018 even as an oracle).
- **Redundancy pruning by correlation** (hurt both learners).
- **Event-count features** (alpha fraction, K-complex count, SEM, REM count): +0.006, below the 0.01 bar.
- **Raw per-epoch saliency heatmaps** on the dashboard (too noisy; replaced by the body-system × stage agreement grid).

## Reference facts used on the dashboard

- U-Sleep 0.79 on held-out CinC 2018; U-Time 0.77; SLEEPNET (MGH) 86% accuracy, κ 0.79.
- Human scorers (AASM program): 82.6% agreement; W 84.1, N1 63.0, N2 85.2, N3 67.4, REM 90.5%.
- Published cardiorespiratory staging, 4-class: Sridhar 2020 κ 0.66 (heart rate only, 10k+ nights, whole-night network, validated on MGH); Bakker 2021 CReSS κ 0.64–0.68; Radha 2019 κ 0.61.

## Open questions

- Probability-based grading (hypnodensity, calibration); a real test needs multi-scorer data.
