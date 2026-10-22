"""One-second interpretable tracks for the window staging model (CinC 2018 montage).

Each track is one number per second from a 2-s Hann-windowed spectrum (1-s hop, frame k centred at k + 0.5 s):
central relative band powers and total power, frontal delta share, occipital alpha share, slow (0.5-2 Hz) and fast
(2-8 Hz) EOG power, chin EMG RMS (10-80 Hz), instantaneous heart rate from R-peaks, and band-passed respiratory
effort (0.1-0.7 Hz) with its envelope. These are the raw material the AASM rules describe (alpha, spindles, slow
waves, eye movements, chin tone), kept at 1-s resolution so the model can see when inside the epoch they happen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FS = 200
WIN = 2 * FS
FREQS = np.fft.rfftfreq(WIN, 1 / FS)
SPECTRAL_CHANNELS = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "Chin1-Chin2"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (11, 16), "beta": (16, 30)}
TRACKS = [
    "c_rel_delta", "c_rel_theta", "c_rel_alpha", "c_rel_sigma", "c_rel_beta", "c_log_power",
    "f_rel_delta", "o_rel_alpha", "eog_low", "eog_high", "emg_log_rms", "hr", "resp_effort", "resp_env",
]
EPS = 1e-6


def power_spectra(x: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_frames, WIN // 2 + 1) periodogram per 1-s frame, chunked to bound memory."""
    x = np.concatenate([np.zeros(FS // 2), np.asarray(x, np.float32), np.zeros(WIN)])
    win = np.hanning(WIN).astype(np.float32)
    out = np.empty((n_frames, WIN // 2 + 1), np.float32)
    for a in range(0, n_frames, 4000):
        b = min(n_frames, a + 4000)
        idx = (np.arange(a, b) * FS)[:, None] + np.arange(WIN)[None, :]
        seg = x[idx]
        seg = seg - seg.mean(1, keepdims=True)
        out[a:b] = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2
    return out


def band_power(P: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return P[:, (FREQS >= lo) & (FREQS < hi)].sum(1)


def heart_rate_track(r_peak_seconds: np.ndarray, n_seconds: int) -> np.ndarray:
    """Beats per minute at each second centre, interpolated between plausible beats (RR 0.3-2.0 s)."""
    hr = np.full(n_seconds, np.nan)
    r = np.asarray(r_peak_seconds, float)
    if r.size < 2:
        return hr
    rr = np.diff(r)
    ok = (rr > 0.3) & (rr < 2.0)
    if ok.sum() > 10:
        mid = (r[1:] + r[:-1])[ok] / 2
        hr = np.interp(np.arange(n_seconds) + 0.5, mid, 60 / rr[ok], left=np.nan, right=np.nan)
    return hr


def respiratory_tracks(effort: np.ndarray, n_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Band-passed (0.1-0.7 Hz) summed effort and log envelope, filtered at 10 Hz so the 0.1-Hz edge stays stable."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    eff10 = np.asarray(effort[: n_seconds * FS], float).reshape(n_seconds * 10, FS // 10).mean(1)
    bp = sosfiltfilt(butter(4, [0.1, 0.7], btype="band", fs=10, output="sos"), eff10)
    env = np.abs(hilbert(bp))
    return bp.reshape(n_seconds, 10).mean(1), np.log10(env.reshape(n_seconds, 10).mean(1) + EPS)


def compute_tracks(signals: dict[str, np.ndarray], r_peak_seconds: np.ndarray | None) -> pd.DataFrame:
    """signals: raw CinC channel name -> samples at 200 Hz (EEG/EOG/EMG in µV, plus ABD and CHEST)."""
    n = len(signals["C3-M2"]) // FS
    P = {c: power_spectra(signals[c], n) for c in SPECTRAL_CHANNELS}
    C = (P["C3-M2"] + P["C4-M1"]) / 2
    F = (P["F3-M2"] + P["F4-M1"]) / 2
    O = (P["O1-M2"] + P["O2-M1"]) / 2
    total = band_power(C, 0.5, 30) + EPS
    t = {f"c_rel_{k}": band_power(C, *v) / total for k, v in BANDS.items()}
    t["c_log_power"] = np.log10(total)
    t["f_rel_delta"] = band_power(F, 0.5, 4) / (band_power(F, 0.5, 30) + EPS)
    t["o_rel_alpha"] = band_power(O, 8, 12) / (band_power(O, 0.5, 30) + EPS)
    t["eog_low"] = np.log10(band_power(P["E1-M2"], 0.5, 2) + EPS)
    t["eog_high"] = np.log10(band_power(P["E1-M2"], 2, 8) + EPS)
    t["emg_log_rms"] = 0.5 * np.log10(band_power(P["Chin1-Chin2"], 10, 80) + EPS)
    t["hr"] = heart_rate_track(r_peak_seconds if r_peak_seconds is not None else np.empty(0), n)
    t["resp_effort"], t["resp_env"] = respiratory_tracks(signals["ABD"] + signals["CHEST"], n)
    df = pd.DataFrame({k: np.asarray(t[k], np.float32) for k in TRACKS})
    df.insert(0, "second", np.arange(n))
    return df


def cinc2018_subject_tracks(raw_dir: Path, cache_dir: Path, subject_id: str) -> pd.DataFrame:
    """Tracks for one CinC record; heart rate uses the recipe cache's Pan-Tompkins R-peaks."""
    import wfdb

    channels = SPECTRAL_CHANNELS + ["ABD", "CHEST"]
    rec = wfdb.rdrecord(str(Path(raw_dir) / subject_id / subject_id), channel_names=channels, physical=True)
    signals = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}
    peaks_path = Path(cache_dir) / subject_id / "pan_tompkins_py/events/r_peaks.parquet"
    r_peaks = pd.read_parquet(peaks_path)["start"].to_numpy() / FS if peaks_path.exists() else None
    return compute_tracks(signals, r_peaks)


# ---------------------------------------------------------------- any sample rate (for other datasets, e.g. UCDDB)
TRANSFER_TRACKS = ["c_rel_delta", "c_rel_theta", "c_rel_alpha", "c_rel_sigma", "c_rel_beta", "c_log_power",
                   "eog_low", "eog_high", "hr", "resp_effort", "resp_env"]


def spectra_at(x: np.ndarray, fs: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Same 2-s Hann, 1-s hop periodogram as power_spectra, for any integer sample rate; returns (P, freqs)."""
    win_len = 2 * fs
    x = np.concatenate([np.zeros(fs // 2), np.asarray(x, np.float32), np.zeros(win_len)])
    win = np.hanning(win_len).astype(np.float32)
    out = np.empty((n_frames, win_len // 2 + 1), np.float32)
    for a in range(0, n_frames, 4000):
        b = min(n_frames, a + 4000)
        idx = (np.arange(a, b) * fs)[:, None] + np.arange(win_len)[None, :]
        seg = x[idx]
        seg = seg - seg.mean(1, keepdims=True)
        out[a:b] = np.abs(np.fft.rfft(seg * win, axis=1)) ** 2
    return out, np.fft.rfftfreq(win_len, 1 / fs)


def _band_at(P: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return P[:, (freqs >= lo) & (freqs < hi)].sum(1)


def transfer_tracks(eeg: list[tuple[np.ndarray, int]], eog: tuple[np.ndarray, int], effort: tuple[np.ndarray, int],
                    r_peak_seconds: np.ndarray | None) -> pd.DataFrame:
    """The 11 tracks shared by CinC 2018 and UCDDB (no frontal/occipital leads, no 10-80 Hz chin EMG).

    eeg: central channels as (samples, fs); eog: one EOG derivation; effort: summed effort belts.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    n = min(len(x) // fs for x, fs in eeg)
    spectra = [spectra_at(x, fs, n) for x, fs in eeg]
    freqs = spectra[0][1]
    C = sum(P for P, _ in spectra) / len(spectra)
    total = _band_at(C, freqs, 0.5, 30) + EPS
    t = {f"c_rel_{k}": _band_at(C, freqs, *v) / total for k, v in BANDS.items()}
    t["c_log_power"] = np.log10(total)
    Pe, fe = spectra_at(eog[0], eog[1], n)
    t["eog_low"] = np.log10(_band_at(Pe, fe, 0.5, 2) + EPS)
    t["eog_high"] = np.log10(_band_at(Pe, fe, 2, 8) + EPS)
    t["hr"] = heart_rate_track(r_peak_seconds if r_peak_seconds is not None else np.empty(0), n)
    x, fs = effort
    x = np.asarray(x[: n * fs], float)
    bp = sosfiltfilt(butter(4, [0.1, 0.7], btype="band", fs=fs, output="sos"), x)
    env = np.abs(hilbert(bp))
    t["resp_effort"] = bp.reshape(n, fs).mean(1)
    t["resp_env"] = np.log10(env.reshape(n, fs).mean(1) + EPS)
    df = pd.DataFrame({k: np.asarray(t[k], np.float32) for k in TRANSFER_TRACKS})
    df.insert(0, "second", np.arange(n))
    return df


def ucddb_subject_tracks(raw_dir: Path, cache_dir: Path, subject_id: str) -> pd.DataFrame:
    """Transfer tracks for one UCDDB record: C3A2/C4A1 at 128 Hz, left-right EOG at 64 Hz, effort sum at 8 Hz."""
    import pyedflib

    reader = pyedflib.EdfReader(str(Path(raw_dir) / f"{subject_id}.rec"))
    try:
        idx = {label: i for i, label in enumerate(reader.getSignalLabels())}
        get = lambda name: (reader.readSignal(idx[name]), int(reader.getSampleFrequency(idx[name])))  # noqa: E731
        c3, c4 = get("C3A2"), get("C4A1")
        (left, fs_eog), (right, _) = get("Lefteye"), get("RightEye")
        effort = get("Sum")
        ecg_fs = reader.getSampleFrequency(idx["ECG"])
    finally:
        reader.close()
    peaks_path = Path(cache_dir) / subject_id / "pan_tompkins_py/events/r_peaks.parquet"
    r_peaks = pd.read_parquet(peaks_path)["start"].to_numpy() / ecg_fs if peaks_path.exists() else None
    return transfer_tracks([c3, c4], (left - right, fs_eog), effort, r_peaks)


def isruc_subject_tracks(raw_dir: Path, cache_dir: Path, subject_id: str) -> pd.DataFrame:
    """All 14 tracks for one ISRUC-Sleep subgroup I night: same 200 Hz montage as CinC, so compute_tracks applies.

    Channels come from io.isruc (header gain fix applied); the 25-Hz effort belts are upsampled to 200 Hz for the
    respiratory tracks (not used by the adopted model). Heart rate uses the recipe cache's R-peaks.
    """
    from scipy.signal import resample_poly

    from physionet_sleep.io.isruc import CINC_NAMES, read_isruc_channels, resolve_isruc_record

    channels, _ = read_isruc_channels(resolve_isruc_record(raw_dir, subject_id))
    signals = {CINC_NAMES[k]: x for k, (x, fs) in channels.items() if k in CINC_NAMES and fs == FS}
    n = min(len(x) for x in signals.values())
    signals = {k: x[:n] for k, x in signals.items()}
    for raw, key in (("ABD", "effort_abdominal"), ("CHEST", "effort_thoracic")):
        x, fs = channels[key]
        up = resample_poly(x, FS, int(fs)) if fs != FS else x
        signals[raw] = np.pad(up, (0, max(0, n - len(up))))[:n]
    peaks_path = Path(cache_dir) / subject_id / "pan_tompkins_py/events/r_peaks.parquet"
    r_peaks = pd.read_parquet(peaks_path)["start"].to_numpy() / FS if peaks_path.exists() else None
    return compute_tracks(signals, r_peaks)
