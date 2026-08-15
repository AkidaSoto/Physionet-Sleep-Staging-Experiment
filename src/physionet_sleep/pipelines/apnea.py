from physionet_sleep.algorithms.apnea import (
    AirflowMorphologyFeatureGenerator,
    ApneaEventDetector,
    EffortSnoreFeatureGenerator,
    ParadoxicalBreathingFeatureGenerator,
)
from physionet_sleep.algorithms.ecg import PanTompkinsDetector
from physionet_sleep.algorithms.feature_table import FeatureTableCompiler
from physionet_sleep.algorithms.respiration import RespirationDetector
from physionet_sleep.algorithms.spo2 import SpO2DropDetector
from physionet_sleep.algorithms.staging import (
    EOGRemFeatureGenerator,
    EMGToneFeatureGenerator,
    EEGArousalFeatureGenerator,
    EyeMovementActivityGenerator,
    SlowWaveFeatureGenerator,
    SpindleFeatureGenerator,
)
from physionet_sleep.core.pipeline import AlgorithmPipeline


def build_apnea_pipeline() -> AlgorithmPipeline:
    return AlgorithmPipeline(
        algorithms=[
            SpindleFeatureGenerator(signal_key="eeg"),
            SlowWaveFeatureGenerator(signal_key="eeg"),
            EyeMovementActivityGenerator(signal_key="eog"),
            EMGToneFeatureGenerator(signal_key="emg"),
            EEGArousalFeatureGenerator(eeg_key="eeg", emg_key="emg"),
            EOGRemFeatureGenerator(signal_key="eog"),
            PanTompkinsDetector(signal_key="ecg"),
            RespirationDetector(signal_keys=("airflow_thermal", "airflow_pressure", "flow")),
            SpO2DropDetector(signal_key="spo2"),
            EffortSnoreFeatureGenerator(),
            AirflowMorphologyFeatureGenerator(signal_keys=("airflow_thermal", "airflow_pressure", "flow")),
            ParadoxicalBreathingFeatureGenerator(
                thoracic_key="effort_thoracic",
                abdominal_key="effort_abdominal",
            ),
            ApneaEventDetector(
                thermal_key="airflow_thermal",
                pressure_key="airflow_pressure",
                respiration_result_key="respiration",
                spo2_result_key="spo2_drop",
            ),
            FeatureTableCompiler(),
        ]
    )
