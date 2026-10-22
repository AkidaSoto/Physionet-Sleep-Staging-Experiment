import numpy as np

from physionet_sleep.algorithms.ecg import (
    HRVFeatureGenerator,
    clean_nn_mask,
    compute_hrv_tracks,
    dfa_alpha,
    refine_r_peaks,
)
from physionet_sleep.algorithms.respiration import (
    BreathRegularityFeatureGenerator,
    CardioRespiratoryCouplingGenerator,
    detect_breaths,
    hr_resp_coherence_track,
)
from physionet_sleep.core.types import AlgorithmResult, SignalRecord

RNG = np.random.default_rng(7)


def _beats_from_rr(rr_of_t, duration_s: float) -> np.ndarray:
    """Beat times (s) from an RR function of time (ms)."""
    times = [0.5]
    while times[-1] < duration_s:
        times.append(times[-1] + rr_of_t(times[-1]) / 1000.0)
    return np.asarray(times[:-1])


def test_clean_nn_rejects_missed_and_extra_beats_only():
    rr = 1000.0 + RNG.normal(0, 15, 400)
    rr[100] = 2000.0  # missed beat: two intervals merged
    rr[200], rr[201] = 420.0, 580.0  # extra beat: one interval split in two
    rr[300] = 250.0  # non-physiological
    ok = clean_nn_mask(rr)
    assert set(np.flatnonzero(~ok)) == {100, 200, 201, 300}


def test_sdnn_and_spectral_shape_for_known_rhythm():
    # Pure respiratory sinus arrhythmia: RR = 1000 + 50 sin(2 pi 0.25 t) ms -> SDNN = 50 / sqrt(2), HF-dominated spectrum.
    beats = _beats_from_rr(lambda t: 1000.0 + 50.0 * np.sin(2 * np.pi * 0.25 * t), 1200.0)
    tracks = compute_hrv_tracks(beats, duration_seconds=1200.0)
    mid = slice(300, 900)
    assert np.allclose(np.nanmedian(tracks["sdnn_150s"][mid]), 50.0 / np.sqrt(2), rtol=0.05)
    assert abs(np.nanmedian(tracks["hr_bpm_30s"][mid]) - 60.0) < 1.5
    assert np.nanmedian(tracks["ln_lf_hf_300s"][mid]) < -3.0  # LF << HF
    assert np.nanmedian(tracks["hr_slow_sd_300s"][mid]) < 0.5  # 10-s bins average out the 4-s RSA cycle
    assert np.all(np.isfinite(tracks["dfa_a2_300s"][mid]))


def test_cleaning_keeps_sdnn_stable_under_artifacts():
    beats = _beats_from_rr(lambda t: 1000.0 + 50.0 * np.sin(2 * np.pi * 0.25 * t), 1200.0)
    clean = compute_hrv_tracks(beats, duration_seconds=1200.0)["sdnn_150s"]
    corrupted = np.sort(np.concatenate([np.delete(beats, [300, 600]), [beats[450] + 0.3, beats[800] + 0.25]]))
    dirty = compute_hrv_tracks(corrupted, duration_seconds=1200.0)["sdnn_150s"]
    mid = slice(300, 900)
    assert np.nanmax(np.abs(dirty[mid] - clean[mid])) < 5.0  # raw SDNN would jump by hundreds of ms


def test_dfa_alpha_white_vs_brown_noise():
    white = RNG.normal(0, 1, 4000)
    brown = np.cumsum(RNG.normal(0, 1, 4000))
    scales = np.array([16, 20, 25, 32, 40, 50, 64])
    assert abs(dfa_alpha(white, scales) - 0.5) < 0.12
    assert abs(dfa_alpha(brown, scales) - 1.5) < 0.15
    assert np.isnan(dfa_alpha(white[:100], scales))


def test_refine_r_peaks_recovers_subsample_timing():
    fs = 128.0
    true_t = np.arange(1.0, 60.0, 0.9137)
    t = np.arange(int(62 * fs)) / fs
    ecg = np.zeros_like(t)
    for tp in true_t:
        ecg += np.exp(-0.5 * ((t - tp) / 0.012) ** 2)
    ecg += RNG.normal(0, 0.01, t.size)
    detected = np.round(true_t * fs).astype(int)
    refined = refine_r_peaks(ecg, fs, detected)
    assert np.max(np.abs(refined - true_t)) < 0.0015
    assert np.max(np.abs(detected / fs - true_t)) > 0.003


def test_coherence_high_when_heart_follows_breathing():
    duration, fs_resp = 900.0, 8.0
    f_breath = 0.25
    beats = _beats_from_rr(lambda t: 1000.0 + 40.0 * np.sin(2 * np.pi * f_breath * t), duration)
    rr = np.diff(beats) * 1000.0
    effort = np.sin(2 * np.pi * f_breath * np.arange(int(duration * fs_resp)) / fs_resp)
    coupled = hr_resp_coherence_track(beats[1:], rr, np.ones(rr.size, bool), effort, fs_resp, duration_seconds=duration)
    # Same breathing, but RR varies independently (smoothed noise, no respiratory locking).
    noise = np.convolve(RNG.normal(0, 40, 2000), np.ones(3) / 3, mode="same")
    beats_u = 0.5 + np.cumsum(np.r_[0, (1000.0 + noise) / 1000.0])
    beats_u = beats_u[beats_u < duration]
    rr_u = np.diff(beats_u) * 1000.0
    uncoupled = hr_resp_coherence_track(beats_u[1:], rr_u, np.ones(rr_u.size, bool), effort, fs_resp, duration_seconds=duration)
    mid = slice(200, 700)
    assert np.nanmedian(coupled[mid]) > 0.9
    assert np.nanmedian(uncoupled[mid]) < 0.5


def _effort(duration: float, fs: float, amps: np.ndarray, period: float) -> np.ndarray:
    t = np.arange(int(duration * fs)) / fs
    k = np.minimum((t // period).astype(int), amps.size - 1)
    return amps[k] * np.sin(2 * np.pi * t / period - np.pi / 2) + RNG.normal(0, 0.01, t.size)


def test_breath_regularity_regular_vs_irregular():
    fs, duration, period = 8.0, 600.0, 4.0
    n_breaths = int(duration / period) + 1
    regular = _effort(duration, fs, np.ones(n_breaths), period)
    irregular = _effort(duration, fs, RNG.uniform(0.4, 1.6, n_breaths), period)
    times, _ = detect_breaths(regular, fs)
    assert abs(times.size - duration / period) <= 3

    def run(x):
        record = SignalRecord()
        record.add_channel("effort_sum", x, fs)
        return BreathRegularityFeatureGenerator().run(record)

    reg, irr = run(regular), run(irregular)
    mid = slice(100, 500)
    assert reg.signals["breath_interval_cv_60s"].sample_rate == 1.0
    assert reg.signals["breath_interval_cv_60s"].values.size == int(duration)
    assert np.nanmedian(reg.signals["breath_interval_cv_60s"].values[mid]) < 0.05
    assert np.nanmedian(reg.signals["breath_amp_cv_60s"].values[mid]) < 0.05
    assert np.nanmedian(irr.signals["breath_amp_cv_60s"].values[mid]) > 0.2


def test_algorithm_classes_end_to_end():
    fs, duration = 128.0, 600.0
    true_t = _beats_from_rr(lambda t: 1000.0 + 40.0 * np.sin(2 * np.pi * 0.25 * t), duration - 1.0)
    t = np.arange(int(duration * fs)) / fs
    ecg = np.zeros_like(t)
    for tp in true_t:
        ecg += np.exp(-0.5 * ((t - tp) / 0.012) ** 2)
    record = SignalRecord()
    record.add_channel("ecg", ecg, fs)
    record.add_channel("effort_sum", np.sin(2 * np.pi * 0.25 * np.arange(int(duration * 8)) / 8), 8.0)
    prior = {"pan_tompkins": AlgorithmResult(name="pan_tompkins", features={"r_peak_indices": np.round(true_t * fs).astype(int)})}
    hrv = HRVFeatureGenerator().run(record, prior=prior)
    expected = {"hr_bpm_30s", "sdnn_150s", "hr_slow_sd_300s", "dfa_a2_300s", "ln_lf_hf_300s", "hr_surge_30s", "nn_clean_frac_300s"}
    assert set(hrv.signals) == expected
    assert all(s.values.size == int(duration) and s.sample_rate == 1.0 for s in hrv.signals.values())
    coupling = CardioRespiratoryCouplingGenerator().run(record, prior={"hrv_features": hrv})
    assert np.nanmedian(coupling.signals["hr_resp_coherence_120s"].values[100:500]) > 0.9
