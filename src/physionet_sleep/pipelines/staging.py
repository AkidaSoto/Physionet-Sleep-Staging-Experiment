from physionet_sleep.algorithms.staging import (
    EOGRemFeatureGenerator,
    EMGToneFeatureGenerator,
    EEGArousalFeatureGenerator,
    EyeMovementActivityGenerator,
    SlowWaveFeatureGenerator,
    SpindleFeatureGenerator,
)
from physionet_sleep.algorithms.feature_table import FeatureTableCompiler
from physionet_sleep.core.pipeline import AlgorithmPipeline


def build_staging_pipeline() -> AlgorithmPipeline:
    return AlgorithmPipeline(
        algorithms=[
            SpindleFeatureGenerator(signal_key="eeg"),
            SlowWaveFeatureGenerator(signal_key="eeg"),
            EyeMovementActivityGenerator(signal_key="eog"),
            EMGToneFeatureGenerator(signal_key="emg"),
            EEGArousalFeatureGenerator(eeg_key="eeg", emg_key="emg"),
            EOGRemFeatureGenerator(signal_key="eog"),
            FeatureTableCompiler(),
        ]
    )
