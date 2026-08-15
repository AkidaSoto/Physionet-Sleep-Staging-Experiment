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
