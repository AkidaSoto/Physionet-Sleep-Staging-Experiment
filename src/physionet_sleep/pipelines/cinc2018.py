from physionet_sleep.algorithms.ecg import HRVFeatureGenerator, PanTompkinsDetector
from physionet_sleep.algorithms.feature_table import FeatureTableCompiler
from physionet_sleep.algorithms.respiration import BreathRegularityFeatureGenerator, CardioRespiratoryCouplingGenerator
from physionet_sleep.algorithms.signal_quality import FlatDropoutDetector
from physionet_sleep.algorithms.staging import (
    EMGToneFeatureGenerator,
    EpochRatioFeatureGenerator,
    EyeMovementActivityGenerator,
    SlowWaveFeatureGenerator,
    SpindleFeatureGenerator,
)
from physionet_sleep.algorithms.topography import CalibratedTopographyGenerator
from physionet_sleep.core.pipeline import AlgorithmPipeline


def build_cinc2018_staging_pipeline(*, ecg_detector_backend: str = "wfdb_xqrs") -> AlgorithmPipeline:
    """Staging-only recipe for CinC 2018: the shared staging features on the central pair (C3/C4), plus features
    that need this dataset's calibrated microvolts and frontal/occipital derivations. No apnea algorithms."""
    return AlgorithmPipeline(
        algorithms=[
            SpindleFeatureGenerator(signal_key="eeg"),
            SlowWaveFeatureGenerator(signal_key="eeg"),
            EyeMovementActivityGenerator(signal_key="eog"),
            EMGToneFeatureGenerator(signal_key="emg"),
            EpochRatioFeatureGenerator(eeg_key="eeg", eog_key="eog", emg_key="emg"),
            CalibratedTopographyGenerator(frontal_key="eeg_frontal", occipital_key="eeg_occipital"),
            FlatDropoutDetector(channel_keys=("eeg", "eeg_frontal", "eeg_occipital", "eog", "emg", "ecg")),
            PanTompkinsDetector(signal_key="ecg", backend=ecg_detector_backend),
            HRVFeatureGenerator(signal_key="ecg", peaks_source="pan_tompkins"),
            BreathRegularityFeatureGenerator(signal_key="effort_sum"),
            CardioRespiratoryCouplingGenerator(signal_key="effort_sum", nn_source="hrv_features"),
            FeatureTableCompiler(),
        ]
    )
