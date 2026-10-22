import shutil
from pathlib import Path

import numpy as np
import pytest

from physionet_sleep.analysis.quality import flat_dropout_epochs, flat_dropout_epochs_from_record
from physionet_sleep.core.types import SignalRecord

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def _noise(seconds: float, fs: float) -> np.ndarray:
    return RNG.normal(size=int(seconds * fs))


def _channels(eeg: np.ndarray, eog: np.ndarray) -> dict:
    return {"C3A2": (eeg, 128.0), "Lefteye": (eog, 64.0)}


def test_all_channels_flat_epoch_is_flagged():
    eeg, eog = _noise(120, 128), _noise(120, 64)
    eeg[30 * 128 : 60 * 128] = 0.25
    eog[30 * 64 : 60 * 64] = -3.0
    flags = flat_dropout_epochs(_channels(eeg, eog), [0, 30, 60, 90])
    assert flags.tolist() == [False, True, False, False]


def test_one_live_channel_prevents_flag():
    eeg, eog = _noise(60, 128), _noise(60, 64)
    eeg[30 * 128 :] = 0.0
    assert flat_dropout_epochs(_channels(eeg, eog), [0, 30]).tolist() == [False, False]


def test_half_epoch_threshold_is_inclusive():
    eeg, eog = _noise(60, 128), _noise(60, 64)
    eeg[:15 * 128] = 1.0  # exactly 15 of 30 one-second windows flat
    eog[:15 * 64] = 1.0
    eeg[30 * 128 : 44 * 128] = 1.0  # 14 of 30
    eog[30 * 64 : 44 * 64] = 1.0
    assert flat_dropout_epochs(_channels(eeg, eog), [0, 30]).tolist() == [True, False]


def test_low_amplitude_signal_is_not_flat():
    eeg, eog = 1e-9 * _noise(30, 128), 1e-9 * _noise(30, 64)
    assert not flat_dropout_epochs(_channels(eeg, eog), [0])[0]


def test_epoch_past_end_of_signal_is_not_flagged():
    eeg, eog = np.zeros(30 * 128), np.zeros(30 * 64)
    assert flat_dropout_epochs(_channels(eeg, eog), [0, 30]).tolist() == [True, False]


def test_record_wrapper_matches_array_function():
    eeg, eog = _noise(60, 128), _noise(60, 64)
    eeg[:30 * 128] = 2.0
    eog[:30 * 64] = 2.0
    record = SignalRecord()
    record.add_channel("C3A2", eeg, 128.0)
    record.add_channel("Lefteye", eog, 64.0)
    expected = flat_dropout_epochs(_channels(eeg, eog), [0, 30])
    assert flat_dropout_epochs_from_record(record, [0, 30]).tolist() == expected.tolist() == [True, False]


# Real-data check: counts validated in artifacts/exploration/subject-failures/FINDINGS.md (scored epochs only).
EXPECTED_DROPOUT_EPOCHS = {"ucddb008": 37, "ucddb006": 22}  # 006: 21 strictly above 50 %, plus one epoch at exactly 50 %
CHANNELS = ["C3A2", "C4A1", "Lefteye", "RightEye", "EMG", "ECG"]


@pytest.mark.parametrize("subject_id,expected", sorted(EXPECTED_DROPOUT_EPOCHS.items()))
def test_ucddb_dropout_counts(subject_id, expected, tmp_path):
    pyedflib = pytest.importorskip("pyedflib")
    pd = pytest.importorskip("pandas")
    rec = REPO / "data/raw/ucddb" / f"{subject_id}.rec"
    stages = REPO / "data/raw/ucddb" / f"{subject_id}_stage.txt"
    if not rec.exists() or not stages.exists():
        pytest.skip("UCDDB raw data not available")
    edf = tmp_path / f"{subject_id}.edf"
    shutil.copy(rec, edf)
    reader = pyedflib.EdfReader(str(edf))
    try:
        labels = reader.getSignalLabels()
        channels = {c: (reader.readSignal(labels.index(c)), reader.getSampleFrequency(labels.index(c))) for c in CHANNELS}
    finally:
        reader.close()
    # Scored annotation grid (codes 0-5). Model prediction files can't be used: they now exclude dropout epochs.
    codes = [int(line) for line in stages.read_text().split() if line.strip()]
    starts = [30.0 * k for k, code in enumerate(codes) if 0 <= code <= 5]
    assert int(flat_dropout_epochs(channels, starts).sum()) == expected
