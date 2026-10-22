"""Per-channel, per-second signal quality for the CinC 2018 montage, and tracks computed from the cleaner lead.

Quality is decided from the signals only (never labels or model errors). For each channel and each second (the same
2-s Hann frames as eeg_tracks, frame k centred at k + 0.5 s):
  log_power   log10 power in the channel's working band (EEG/EOG 0.5-30 Hz, chin 10-80 Hz excluding 58-62 Hz)
  line        mains share: power 58-62 Hz / working-band power (60 Hz mains at MGH)
  flat        peak-to-peak of the second's samples == 0 (dropout)
  ptp         peak-to-peak amplitude of the second in µV (movement / electrode pop)
  ecg_corr    |corr| of the 5-40 Hz channel with the 5-40 Hz ECG over the 2-s frame (cardiac field)
A second is bad for a channel when any rule in QUALITY_RULES fires; `bad_seconds` returns the flags.

quality_weighted_tracks computes the same tracks as eeg_tracks.compute_tracks, but each pair (C3/C4, F3/F4, O1/O2)
averages only the leads that are good over the surrounding 30-s epoch; when both or neither are good it averages both,
as the adopted tracks do. per_second_inputs caches per-lead band powers so the weighting can change without re-reading
raw signals. The existing compute_tracks and the tracks1s cache are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physionet_sleep.algorithms.eeg_tracks import (
    BANDS,
    EPS,
    FREQS,
    FS,
    SPECTRAL_CHANNELS,
    band_power,
    heart_rate_track,
    power_spectra,
    respiratory_tracks,
)

PAIRS = {"c": ("C3-M2", "C4-M1"), "f": ("F3-M2", "F4-M1"), "o": ("O1-M2", "O2-M1")}
SHORT = {"F3-M2": "F3", "F4-M1": "F4", "C3-M2": "C3", "C4-M1": "C4", "O1-M2": "O1", "O2-M1": "O2",
         "E1-M2": "EOG", "Chin1-Chin2": "EMG"}
EPOCH = 30
# Thresholds fixed from label-free signal distributions (artifacts/exploration/signal-quality/calibrate_rules.py)
# before any quality-aware model was trained or scored.
# Rules apply to EEG and EOG; chin EMG is judged on flatness only, because 60 Hz on the chin is common (a third of all
# seconds) and is handled by notching the chin track instead (quality_weighted_tracks(emg_notch=True)).
QUALITY_RULES = {
    "line": 0.5,        # 58-62 Hz power at least half the 0.5-30 Hz power (a clean lead sits near 0.001)
    "ecg_corr": 0.9,    # 5-40 Hz signal almost entirely cardiac (|corr| 0.4-0.7 is common through the mastoid reference)
    "ptp": 800.0,       # µV peak-to-peak within one second; movement, electrode pop (clean 99th percentile 300-460)
    "flat": 0.5,        # flat second (peak-to-peak 0)
}
LINE_BAND = (FREQS >= 58) & (FREQS <= 62)


def _working_power(P: np.ndarray, channel: str) -> np.ndarray:
    if channel == "Chin1-Chin2":
        return P[:, (FREQS >= 10) & (FREQS < 80) & ~LINE_BAND].sum(1)
    return band_power(P, 0.5, 30)


def _frame_corr(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """|corr| of two equally long signals over 2-s frames centred like power_spectra (frame k covers k-0.5 .. k+1.5 s)."""
    pad = lambda x: np.concatenate([np.zeros(FS // 2), x, np.zeros(2 * FS)])  # noqa: E731
    a, b = pad(np.asarray(a, np.float32)), pad(np.asarray(b, np.float32))
    out = np.empty(n, np.float32)
    for s in range(0, n, 4000):
        e = min(n, s + 4000)
        idx = (np.arange(s, e) * FS)[:, None] + np.arange(2 * FS)[None, :]
        x, y = a[idx], b[idx]
        x = x - x.mean(1, keepdims=True)
        y = y - y.mean(1, keepdims=True)
        out[s:e] = np.abs((x * y).sum(1)) / np.sqrt((x * x).sum(1) * (y * y).sum(1) + 1e-12)
    return out


def channel_quality(signals: dict[str, np.ndarray], spectra: dict[str, np.ndarray], ecg: np.ndarray | None,
                    n: int) -> pd.DataFrame:
    """One row per second, columns '<short channel>_<metric>' for the 8 spectral channels."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [5, 40], btype="band", fs=FS, output="sos")
    ecg_bp = sosfiltfilt(sos, np.nan_to_num(np.asarray(ecg, float))) if ecg is not None else None
    q = {"second": np.arange(n)}
    for ch in SPECTRAL_CHANNELS:
        P, sh = spectra[ch], SHORT[ch]
        work = _working_power(P, ch)
        q[f"{sh}_log_power"] = np.log10(work + EPS)
        q[f"{sh}_line"] = P[:, LINE_BAND].sum(1) / (work + EPS)
        x = np.asarray(signals[ch][: n * FS], np.float32)
        x = np.concatenate([x, np.zeros(n * FS - len(x), np.float32)]).reshape(n, FS)
        ptp = np.ptp(x, axis=1)
        q[f"{sh}_flat"] = (ptp == 0).astype(np.float32)
        q[f"{sh}_ptp"] = ptp
        if ecg_bp is not None and ch != "Chin1-Chin2":
            q[f"{sh}_ecg_corr"] = _frame_corr(sosfiltfilt(sos, np.nan_to_num(signals[ch].astype(float))), ecg_bp, n)
        else:
            q[f"{sh}_ecg_corr"] = np.zeros(n, np.float32)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in q.items()})


def bad_seconds(quality: pd.DataFrame, channel_short: str, rules: dict[str, float] | None = None) -> np.ndarray:
    """Boolean per second: the channel breaks any quality rule (chin EMG: flat only)."""
    r = {**QUALITY_RULES, **(rules or {})}
    c = channel_short
    bad = quality[f"{c}_flat"].to_numpy() >= r["flat"]
    if c != "EMG":
        bad |= quality[f"{c}_line"].to_numpy() >= r["line"]
        bad |= quality[f"{c}_ecg_corr"].to_numpy() >= r["ecg_corr"]
        bad |= quality[f"{c}_ptp"].to_numpy() >= r["ptp"]
    return bad


def epoch_bad_fraction(bad: np.ndarray, epoch_seconds: int = EPOCH) -> np.ndarray:
    """Per second, the share of bad seconds in the 30-s scoring epoch containing it."""
    n = len(bad)
    ne = -(-n // epoch_seconds)
    b = np.concatenate([bad.astype(float), np.full(ne * epoch_seconds - n, np.nan)]).reshape(ne, epoch_seconds)
    return np.repeat(np.nanmean(b, 1), epoch_seconds)[:n]


def pair_weights(quality: pd.DataFrame, max_bad_fraction: float = 0.2) -> dict[str, np.ndarray]:
    """Per pair, (n_seconds, 2) weights: a lead is used when its epoch has <= max_bad_fraction bad seconds;
    if both or neither pass, both are used equally (the adopted average)."""
    w = {}
    for key, (a, b) in PAIRS.items():
        ga = epoch_bad_fraction(bad_seconds(quality, SHORT[a])) <= max_bad_fraction
        gb = epoch_bad_fraction(bad_seconds(quality, SHORT[b])) <= max_bad_fraction
        one = ga ^ gb
        wa = np.where(one, ga.astype(float), 0.5)
        w[key] = np.stack([wa, 1 - wa], 1).astype(np.float32)
    return w


EEG_BANDS = {**BANDS, "total": (0.5, 30)}
QUALITY_TRACKS = ["c_rel_delta", "c_rel_theta", "c_rel_alpha", "c_rel_sigma", "c_rel_beta", "c_log_power",
                  "f_rel_delta", "o_rel_alpha", "eog_low", "eog_high", "emg_log_rms", "hr", "resp_effort", "resp_env"]
SINGLE_TRACKS = ["eog_low", "eog_high", "emg_log_rms", "hr", "resp_effort", "resp_env"]


def per_second_inputs(signals: dict[str, np.ndarray], r_peak_seconds: np.ndarray | None,
                      ecg: np.ndarray | None) -> pd.DataFrame:
    """Everything quality_weighted_tracks needs, one row per second: band powers of each EEG lead
    ('<lead>_<band>'), the single-channel tracks (EOG, chin, heart, breathing) and channel_quality's metrics.
    Band powers are linear in the spectrum, so a weighted lead average can be formed from them afterwards."""
    n = len(signals["C3-M2"]) // FS
    P = {c: power_spectra(signals[c], n) for c in SPECTRAL_CHANNELS}
    out = channel_quality(signals, P, ecg, n)
    for a, b in PAIRS.values():
        for lead in (a, b):
            for k, v in EEG_BANDS.items():
                out[f"{SHORT[lead]}_{k}"] = band_power(P[lead], *v).astype(np.float32)
    out["eog_low"] = np.log10(band_power(P["E1-M2"], 0.5, 2) + EPS)
    out["eog_high"] = np.log10(band_power(P["E1-M2"], 2, 8) + EPS)
    out["emg_log_rms"] = 0.5 * np.log10(band_power(P["Chin1-Chin2"], 10, 80) + EPS)
    out["hr"] = heart_rate_track(r_peak_seconds if r_peak_seconds is not None else np.empty(0), n)
    out["resp_effort"], out["resp_env"] = respiratory_tracks(signals["ABD"] + signals["CHEST"], n)
    return out.astype({c: np.float32 for c in out.columns if c != "second"})


def quality_weighted_tracks(inputs: pd.DataFrame, weights: dict[str, np.ndarray] | None = None,
                            emg_notch: bool = False) -> pd.DataFrame:
    """eeg_tracks.TRACKS from per_second_inputs, each lead pair averaged with `weights` ((n, 2) per pair key
    'c', 'f', 'o'; default pair_weights(inputs)). Equal weights reproduce eeg_tracks.compute_tracks.
    emg_notch: chin RMS over 10-80 Hz without 58-62 Hz, so mains on the chin cannot mask atonia.
    Also returns '<pair>_lead' (0 = both leads, 1 = left lead only, 2 = right lead only)."""
    w = weights if weights is not None else pair_weights(inputs)
    mix = {key: {k: w[key][:, 0] * inputs[f"{SHORT[a]}_{k}"].to_numpy() + w[key][:, 1] * inputs[f"{SHORT[b]}_{k}"].to_numpy()
                 for k in EEG_BANDS} for key, (a, b) in PAIRS.items()}
    C, F, O = mix["c"], mix["f"], mix["o"]
    total = C["total"] + EPS
    t = {f"c_rel_{k}": C[k] / total for k in BANDS}
    t["c_log_power"] = np.log10(total)
    t["f_rel_delta"] = F["delta"] / (F["total"] + EPS)
    t["o_rel_alpha"] = O["alpha"] / (O["total"] + EPS)
    for k in SINGLE_TRACKS:
        t[k] = inputs[k].to_numpy()
    if emg_notch:
        t["emg_log_rms"] = 0.5 * inputs["EMG_log_power"].to_numpy()
    df = pd.DataFrame({k: np.asarray(t[k], np.float32) for k in QUALITY_TRACKS})
    df.insert(0, "second", inputs["second"].to_numpy().astype(int))
    for key in PAIRS:
        df[f"{key}_lead"] = np.where(w[key][:, 0] == 1, 1, np.where(w[key][:, 1] == 1, 2, 0)).astype(np.int8)
    return df


def cinc2018_per_second_inputs(raw_dir: Path, cache_dir: Path, subject_id: str) -> pd.DataFrame:
    """per_second_inputs for one CinC record (R-peaks from the recipe cache, as cinc2018_subject_tracks)."""
    import wfdb

    channels = SPECTRAL_CHANNELS + ["ABD", "CHEST", "ECG"]
    rec = wfdb.rdrecord(str(Path(raw_dir) / subject_id / subject_id), channel_names=channels, physical=True)
    signals = {name: rec.p_signal[:, i].astype(np.float32) for i, name in enumerate(rec.sig_name)}
    del rec
    peaks_path = Path(cache_dir) / subject_id / "pan_tompkins_py/events/r_peaks.parquet"
    r_peaks = pd.read_parquet(peaks_path)["start"].to_numpy() / FS if peaks_path.exists() else None
    return per_second_inputs(signals, r_peaks, signals.get("ECG"))
