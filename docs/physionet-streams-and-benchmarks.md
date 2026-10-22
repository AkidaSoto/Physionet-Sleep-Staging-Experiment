# PhysioNet Data Streams and Benchmark Targets

## Purpose

This doc is the research/reference note for:

- what UCDDB contains
- what tasks it supports
- what benchmark numbers are worth knowing
- how results should be evaluated and compared

It is not the implementation plan.

## Primary dataset

- **St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (`ucddb`)**

Primary source:

- [UCD Sleep Apnea Database on PhysioNet](https://physionet.org/content/ucddb/1.0.0/)

## Bottom line

UCDDB is the right primary dataset for this project because it combines:

- overnight PSG recordings
- sleep stage labels
- respiratory-event annotations
- ECG plus multiple sleep-related physiological channels

That makes it stronger for a unified project than splitting staging and apnea work across different datasets.

## Why UCDDB fits

From the official PhysioNet page, UCDDB contains:

- **25 full overnight polysomnograms**
- **simultaneous three-channel Holter ECG**
- adult subjects with suspected sleep-disordered breathing

Per-subject files include:

- PSG waveform files
- Holter ECG EDF files
- sleep stage annotation text files
- respiratory event annotation text files

Examples:

- `ucddb002.rec`
- `ucddb002_lifecard.edf`
- `ucddb002_stage.txt`
- `ucddb002_respevt.txt`

## Stream review

### Signals available

UCDDB includes the core PSG streams we care about:

- ECG
- EEG
- EOG
- chin EMG
- nasal airflow
- thoracic / abdominal respiratory effort
- oxygen saturation
- snoring
- body position

### Labels available

From the official UCDDB page:

- sleep stages are coded in the stage files
- respiratory events include:
  - obstructive apnea
  - central apnea
  - mixed apnea
  - hypopnea
  - periodic breathing / Cheyne-Stokes related markers

## Recommended task mapping inside UCDDB

| Task | Primary labels | Primary streams |
| --- | --- | --- |
| Sleep staging | `*_stage.txt` | EEG, EOG, EMG |
| Sleep apnea detection | `*_respevt.txt` | airflow, respiratory effort, SpO2, ECG |
| Multimodal modeling | both | EEG + respiratory + ECG combinations |

## Dataset strengths

- one dataset supports both core tasks
- clinically relevant sleep-disordered-breathing cohort
- full overnight PSG
- open access on PhysioNet
- manageable size for local development

## Dataset limitations

- only 25 subjects
- thinner benchmark literature than Sleep-EDF or Apnea-ECG
- results vary a lot with split protocol, segmentation, and channel selection

## Sleep staging benchmark targets on UCDDB

Representative sources:

- [Multi-Branch CNN with stage refinement and attention fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC7698838/)
- [Automatic method using MFCC features for sleep stage classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC10858857/)
- [RAPIDEST framework snippet](https://www.researchgate.net/publication/365726448_RAPIDEST_A_Framework_for_Obstructive_Sleep_Apnea_Detection)

Representative reported results:

| Method | ACC | Cohen's kappa | Macro-F1 |
| --- | ---: | ---: | ---: |
| Multi-Branch CNN + refinement | 79.4% | 0.73 | 78.8% |
| MFCC-based staging method | 73.07% | 0.63 | not reported in snippet |
| RAPIDEST snippet | ~82% | 0.764 | 74.79% |

### Plain-language paper comparison

This is the practical reading of the UCDDB sleep-staging papers, not just the headline score table.

| Paper | Reported UCDDB score | What the model is | Complexity | Interpretability | Comparison note |
| --- | --- | --- | --- | --- | --- |
| Multi-Branch CNN + refinement | ACC 79.4%, macro-F1 78.8%, kappa 0.73 | Four-channel deep model with per-channel CNN branches, attention-based fusion, and a learned stage-refinement block | High | Low | Strongest clean staging benchmark in the tracked papers, but also the most black-box |
| MFCC + CNN-LSTM | ACC 73.07%, kappa 0.63 | Hand-designed MFCC representation from EEG and EMG, then four conv layers, one LSTM layer, and a classifier using three epochs as input | Medium-high | Medium-low | Easier to compare with a feature pipeline than the end-to-end CNN paper, but still mainly a deep model |
| RAPIDEST | About ACC 82%, macro-F1 74.79%, kappa 0.764 | EEG-only sleep-stage classifier whose transition patterns are turned into a rarity score for downstream apnea / OSA detection | Medium | Medium | Less apples-to-apples for pure sleep staging because the paper's main goal is apnea severity detection through staging dynamics |

### How to read these papers against our project

- As of **August 22, 2026**, the current internal best staging anchor is the 30-second ensemble plus learned Stage 2 smoother at **macro-F1 68.26%** and **kappa 0.580**, versus the strongest tracked paper anchor at **macro-F1 78.8%** and **kappa 0.73**.
- The cleanest simpler internal anchor is the lighter `ctx5` learned smoother at **macro-F1 66.81%** and **kappa 0.563**.
- The strongest connected neural alternative so far is the hierarchical epoch model at **macro-F1 67.14%** and **kappa 0.561**; it did not beat the feature-based smoother path.
- The highest reported UCDDB staging scores still come from models that are more complex and less interpretable than our current feature-grounded work.
- The Multi-Branch CNN paper is the strongest benchmark target for raw staging score, but it is a substantially more black-box architecture.
- The MFCC paper is the easiest "middle ground" comparison because it still uses explicit signal transforms before deep learning.
- RAPIDEST should be treated carefully in staging comparisons because it is not just a sleep-staging paper; it uses sleep-stage patterns to support apnea detection.
- If our project goal is to demonstrate interpretable physiological modeling, the fair claim is not "beat every deep model on F1." The fair claim is "trade some top-end score for much smaller, clearer, and more explainable physiology-grounded models."

### Validation caveat

- The MFCC paper states a 70/20/10 train/validation/test split rather than strict subject-wise cross-validation.
- RAPIDEST reports 25 leave-one-out runs on UCDDB for sleep-stage classification.
- Published UCDDB scores are therefore not always under the same validation protocol, so direct score comparisons should always mention the split protocol.

### Practical staging target

- respectable: **kappa >= 0.70**, **macro-F1 near 0.75**
- strong v1: **kappa >= 0.74**, **macro-F1 >= 0.78**

## Apnea benchmark targets on UCDDB

Representative sources:

- [Frontiers 2025: SpiTranNet comparison table on UCDDB](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1716204/full)
- [A spatio-temporal learning-based model for sleep apnea detection using single-lead ECG signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC9389170/)
- [Modified GoogLeNet comparison snippet](https://www.researchgate.net/publication/388849009_Explainable_AI-driven_scalogram_analysis_and_optimized_transfer_learning_for_sleep_apnea_detection_with_single-lead_electrocardiograms)
- [TASPPNet abstract snippet](https://pubmed.ncbi.nlm.nih.gov/41337251/)

Representative reported results:

| Method | Evaluation style | ACC | F1 | Kappa / similar |
| --- | --- | ---: | ---: | ---: |
| SpiTranNet (2025) | per-segment binary apnea classification | 99.4% | 0.899 | 0.896 |
| Spatio-temporal ECG model | per-minute detection | 92.3% | 76.0% | not shown in snippet |
| Modified GoogLeNet snippet | per-segment cross-validation style table | ~87.0% | ~86.4% | MCC instead of kappa |
| TASPPNet | UCDDB evaluation | 0.783 | 0.799 | not shown |

### Practical apnea target

- respectable: **F1 >= 0.76**
- strong v1: **F1 >= 0.80**
- very strong: **F1 >= 0.85**

## External reference comparisons

These are comparison anchors, not primary training targets.

### Sleep-EDF staging reference band

Source:

- [Frontiers 2024: Multimodal sleep staging network based on obstructive sleep apnea](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1505746/full)

| Model | ACC | Cohen's kappa | Macro-F1 |
| --- | ---: | ---: | ---: |
| SeqSleepNet | 85.2% | 0.790 | 79.6% |
| MultiChannelSleepNet | 86.5% | 0.816 | 80.3% |
| SleepViTransformer | 87.8% | 0.834 | 81.5% |
| MSDC-SSRNet | 88.7% | 0.846 | 83.5% |

### Apnea-ECG reference band

Sources:

- [Apnea Detection from the ECG](https://archive.physionet.org/physiotools/apdet/)
- [Efficient sleep apnea detection using single-lead ECG: A CNN-Transformer-LSTM approach](https://www.sciencedirect.com/science/article/abs/pii/S0010482525010066)
- [Frontiers 2025: SpiTranNet results on Apnea-ECG](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1716204/full)
- [A leakage-free analytics framework for subject-wise electrocardiogram-based sleep apnea detection using machine learning](https://www.sciencedirect.com/science/article/pii/S2772442526000316)

| Dataset / method | ACC | F1 | Cohen's kappa |
| --- | ---: | ---: | ---: |
| Apnea-ECG `apdet` minute classification | 84.5% test accuracy | not reported | not reported |
| Apnea-ECG CNN-Transformer-LSTM | 91.6% | 0.890 | 0.822 |
| Apnea-ECG SpiTranNet | 95.0% | 0.935 | 0.894 |
| Apnea-ECG leakage-free subject-wise SVM | 0.72 | not exposed in snippet | 0.39 |

## Recommended evaluation strategy

### Sleep staging

Report:

- accuracy
- macro-F1
- Cohen's kappa
- per-stage F1

### Apnea detection

Report:

- accuracy
- F1
- sensitivity
- specificity
- precision
- Cohen's kappa if computed

### Split protocol

Because there are only 25 subjects, strongly prefer:

- subject-wise cross-validation
- leave-one-subject-out or grouped folds

## Design implications

The data structure supports three aligned layers per record:

- waveform channels
- sleep stage timeline
- respiratory event timeline

## Sources

- [St. Vincent's University Hospital / University College Dublin Sleep Apnea Database](https://physionet.org/content/ucddb/1.0.0/)
- [Archived UCDDB page](https://archive.physionet.org/physiobank/database/ucddb/?C=S)
- [Multi-Branch CNN sleep staging results](https://pmc.ncbi.nlm.nih.gov/articles/PMC7698838/)
- [MFCC-based sleep stage classification on UCDDB](https://pmc.ncbi.nlm.nih.gov/articles/PMC10858857/)
- [RAPIDEST snippet](https://www.researchgate.net/publication/365726448_RAPIDEST_A_Framework_for_Obstructive_Sleep_Apnea_Detection)
- [Spatio-temporal ECG apnea detection on UCDDB](https://pmc.ncbi.nlm.nih.gov/articles/PMC9389170/)
- [Modified GoogLeNet comparison snippet](https://www.researchgate.net/publication/388849009_Explainable_AI-driven_scalogram_analysis_and_optimized_transfer_learning_for_sleep_apnea_detection_with_single-lead_electrocardiograms)
- [TASPPNet abstract snippet](https://pubmed.ncbi.nlm.nih.gov/41337251/)
- [Sleep-EDF staging reference table](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1505746/full)
- [Apnea-ECG historical baseline](https://archive.physionet.org/physiotools/apdet/)
- [Apnea-ECG CNN-Transformer-LSTM](https://www.sciencedirect.com/science/article/abs/pii/S0010482525010066)
- [Leakage-free Apnea-ECG subject-wise analysis](https://www.sciencedirect.com/science/article/pii/S2772442526000316)
