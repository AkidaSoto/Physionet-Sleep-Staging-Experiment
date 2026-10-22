import numpy as np
import pytest

from physionet_sleep.algorithms.staging import EpochRatioFeatureGenerator
from physionet_sleep.core.types import SignalRecord

EEG_FS, EOG_FS, EMG_FS = 128.0, 64.0, 64.0


def _tone(freq, amp, seconds, fs):
    t = np.arange(int(round(seconds * fs))) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def _record(n_epochs=4, eeg_amps=(2.0, 1.0, 0.5, 0.25), eog_rem_amp=None, emg_amp=None, gain=1.0, extra_seconds=0.0):
    seconds = n_epochs * 30.0 + extra_seconds
    theta, sigma, beta, muscle = eeg_amps
    # Tones sit on Welch bin centres (0.25 Hz resolution) well inside their bands.
    eeg_one = _tone(6.0, theta, seconds, EEG_FS) + _tone(13.0, sigma, seconds, EEG_FS) + _tone(20.0, beta, seconds, EEG_FS) + _tone(40.0, muscle, seconds, EEG_FS)
    eeg = np.column_stack([eeg_one, 0.5 * eeg_one])  # second channel at a different gain: ratios must agree
    if eog_rem_amp is None:
        eog_rem_amp = np.ones(n_epochs)
    eog = np.zeros(int(round(seconds * EOG_FS)))
    n = int(30 * EOG_FS)
    for k, a in enumerate(eog_rem_amp):
        eog[k * n:(k + 1) * n] = _tone(2.0, a, 30.0, EOG_FS) + _tone(12.0, 1.0, 30.0, EOG_FS)
    if emg_amp is None:
        emg_amp = np.ones(n_epochs)
    emg = np.zeros(int(round(seconds * EMG_FS)))
    m = int(30 * EMG_FS)
    for k, a in enumerate(emg_amp):
        emg[k * m:(k + 1) * m] = _tone(20.0, a, 30.0, EMG_FS)
    record = SignalRecord()
    record.add_channel("eeg", gain * eeg, EEG_FS)
    record.add_channel("eog", gain * eog, EOG_FS)
    record.add_channel("emg", gain * emg, EMG_FS)
    return record


def _epochs(result, name):
    return result.signals[name].values[::5]


def test_eeg_band_ratios_match_known_tone_powers():
    result = EpochRatioFeatureGenerator().run(_record())
    p = {"theta": 2.0**2, "sigma": 1.0**2, "beta": 0.5**2, "muscle": 0.25**2}
    total = p["theta"] + p["sigma"] + p["beta"]
    np.testing.assert_allclose(_epochs(result, "rel_theta"), p["theta"] / total, rtol=1e-3)
    np.testing.assert_allclose(_epochs(result, "rel_sigma"), p["sigma"] / total, rtol=1e-3)
    np.testing.assert_allclose(_epochs(result, "rel_beta"), p["beta"] / total, rtol=1e-3)
    np.testing.assert_allclose(_epochs(result, "rel_alpha"), 0.0, atol=1e-4)
    np.testing.assert_allclose(_epochs(result, "spindle_rel"), p["sigma"] / (p["theta"] + p["beta"]), rtol=1e-3)
    np.testing.assert_allclose(_epochs(result, "muscle_frac_db"), 10 * np.log10(p["muscle"] / total), atol=0.01)


def test_outputs_are_gain_invariant():
    a = EpochRatioFeatureGenerator().run(_record(emg_amp=[1, 2, 3, 4]))
    b = EpochRatioFeatureGenerator().run(_record(emg_amp=[1, 2, 3, 4], gain=1e-3))
    for name, series in a.signals.items():
        np.testing.assert_allclose(series.values, b.signals[name].values, rtol=1e-6, atol=1e-6, err_msg=name)


def test_eog_ratio_tracks_rapid_eye_movement_amplitude_per_epoch():
    result = EpochRatioFeatureGenerator().run(_record(eog_rem_amp=[1.0, 2.0, 1.0, 4.0]))
    rem_vs_noise = _epochs(result, "eog_rem_vs_noise_db")
    # Mean spectral magnitude is linear in amplitude, so doubling the 2 Hz tone adds 10*log10(2) dB.
    np.testing.assert_allclose(rem_vs_noise[1] - rem_vs_noise[0], 10 * np.log10(2.0), atol=0.05)
    np.testing.assert_allclose(rem_vs_noise[3] - rem_vs_noise[2], 10 * np.log10(4.0), atol=0.05)
    np.testing.assert_allclose(rem_vs_noise[0], rem_vs_noise[2], atol=1e-9)
    assert np.all(np.isfinite(_epochs(result, "eog_rem_vs_sem_db")))


def test_emg_floor_is_relative_to_quietest_epochs():
    amps = [1.0] * 10 + [4.0] * 10
    result = EpochRatioFeatureGenerator().run(_record(n_epochs=20, emg_amp=amps))
    floor_db = _epochs(result, "emg_floor_db")
    np.testing.assert_allclose(floor_db[:10], 0.0, atol=0.05)
    # Envelope is amplitude, so a 4x louder chin reads 10*log10(4) dB above the floor (epoch 10 straddles the step).
    np.testing.assert_allclose(floor_db[11:], 10 * np.log10(4.0), atol=0.05)
    assert np.all(np.isfinite(_epochs(result, "emg_baseline_floor_db")))


def test_epoch_grid_time_stamps_and_incomplete_tail():
    result = EpochRatioFeatureGenerator().run(_record(eog_rem_amp=[1.0, 3.0, 1.0, 1.0], extra_seconds=12.0))
    series = result.signals["eog_rem_vs_noise_db"]
    assert series.sample_rate == pytest.approx(1 / 6)
    assert series.values.size == 5 * 5  # 4 full epochs + 1 incomplete, 5 rows of 6 s each
    # Rows 5..9 (t = 30..54 s) carry epoch 1, which has the louder 2 Hz tone.
    assert np.all(series.values[5:10] == series.values[5])
    np.testing.assert_allclose(series.values[5] - series.values[4], 10 * np.log10(3.0), atol=0.05)
    assert np.all(np.isnan(series.values[20:]))


def test_spectrogram_tracks_carry_first_frame_time():
    from physionet_sleep.algorithms.staging import EEGArousalFeatureGenerator, EOGRemFeatureGenerator, SpindleFeatureGenerator

    record = _record(n_epochs=4)
    spindle = SpindleFeatureGenerator().run(record)
    assert all(s.metadata["start_seconds"] == pytest.approx(1.0) for s in spindle.signals.values())  # 2-s frames, centre 1 s
    arousal = EEGArousalFeatureGenerator().run(record)
    assert all(s.metadata["start_seconds"] == pytest.approx(1.0) for s in arousal.signals.values())
    eog = EOGRemFeatureGenerator().run(record)
    for name in ("rem_band_power_fast", "sem_band_power_fast", "noise_band_power_fast"):
        assert eog.signals[name].metadata["start_seconds"] == pytest.approx(15.0)  # 30-s frames, centre 15 s
    fast, epoch = eog.signals["rem_band_power_fast"], eog.signals["rem_power_epoch"]
    bins = int(round(fast.sample_rate / epoch.sample_rate))
    assert epoch.sample_rate == pytest.approx(fast.sample_rate / bins)  # true block rate, not a nominal 1/30 Hz
    assert epoch.metadata["start_seconds"] == pytest.approx(15.0 + 0.5 * (bins - 1) / fast.sample_rate)
    ratios = EpochRatioFeatureGenerator().run(record)
    assert all(s.metadata["start_seconds"] == 0.0 for s in ratios.signals.values())
