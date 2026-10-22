from physionet_sleep.algorithms.apnea import (
    AirflowMorphologyFeatureGenerator,
    ApneaEventDetector,
    EffortSnoreFeatureGenerator,
    ParadoxicalBreathingFeatureGenerator,
)
from physionet_sleep.algorithms.ecg import HRVFeatureGenerator, PanTompkinsDetector
from physionet_sleep.algorithms.feature_table import FeatureTableCompiler
from physionet_sleep.algorithms.signal_quality import FlatDropoutDetector
from physionet_sleep.algorithms.respiration import (
    BreathRegularityFeatureGenerator,
    CardioRespiratoryCouplingGenerator,
    RespirationDetector,
)
from physionet_sleep.algorithms.spo2 import SpO2DropDetector
from physionet_sleep.algorithms.staging import (
    EOGRemFeatureGenerator,
    EMGToneFeatureGenerator,
    EpochRatioFeatureGenerator,
    EEGArousalFeatureGenerator,
    EyeMovementActivityGenerator,
    SlowWaveFeatureGenerator,
    SpindleFeatureGenerator,
)
from physionet_sleep.core.pipeline import AlgorithmPipeline


def build_apnea_pipeline(*, ecg_detector_backend: str = "wfdb_xqrs") -> AlgorithmPipeline:
    return AlgorithmPipeline(
        algorithms=[
            SpindleFeatureGenerator(signal_key="eeg"),
            SlowWaveFeatureGenerator(signal_key="eeg"),
            EyeMovementActivityGenerator(signal_key="eog"),
            EMGToneFeatureGenerator(signal_key="emg"),
            EEGArousalFeatureGenerator(eeg_key="eeg", emg_key="emg"),
            EOGRemFeatureGenerator(signal_key="eog"),
            EpochRatioFeatureGenerator(eeg_key="eeg", eog_key="eog", emg_key="emg"),
            FlatDropoutDetector(),
            PanTompkinsDetector(signal_key="ecg", backend=ecg_detector_backend),
            # Cleaned, literature-window HRV and cardio-respiratory features (need pan_tompkins peaks first).
            HRVFeatureGenerator(signal_key="ecg", peaks_source="pan_tompkins"),
            BreathRegularityFeatureGenerator(signal_key="effort_sum"),
            CardioRespiratoryCouplingGenerator(signal_key="effort_sum", nn_source="hrv_features"),
            RespirationDetector(signal_keys=("flow", "airflow_thermal", "airflow_pressure")),
            SpO2DropDetector(signal_key="spo2"),
            EffortSnoreFeatureGenerator(snore_key=None),
            AirflowMorphologyFeatureGenerator(signal_keys=("flow", "airflow_thermal", "airflow_pressure")),
            ParadoxicalBreathingFeatureGenerator(
                thoracic_key="effort_thoracic",
                abdominal_key="effort_abdominal",
            ),
            ApneaEventDetector(
                thermal_key="flow",
                pressure_key=None,
                respiration_result_key="respiration",
                spo2_result_key="spo2_drop",
            ),
            FeatureTableCompiler(),
            FeatureTableCompiler(
                window_seconds=1.0,
                name="feature_table_apnea_1s",
            ),
        ]
    )
