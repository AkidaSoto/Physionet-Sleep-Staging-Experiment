"""Validated physiological cue extraction and staged ConceptBiGRU variants (#80)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from physionet_sleep.algorithms.ecg import _detect_r_peaks
from physionet_sleep.algorithms.eeg_tracks import (
    BANDS,
    EPS,
    FS,
    band_power,
    heart_rate_track,
    power_spectra,
    respiratory_tracks,
)
from physionet_sleep.experiments.concept_bigru_staging import (
    CONCEPT_EPOCH_SIGNALS,
    CONCEPT_NAMES,
    CONCEPT_STEP_SIGNALS,
    build_concept_layout,
)
from physionet_sleep.experiments.waveform_data import (
    WaveformStore,
    resolve_cinc2018_record,
)

CUE_COLUMNS: tuple[str, ...] = (
    "c_rel_delta",
    "f_rel_delta",
    "eog_low",
    "emg_log_rms",
    "eog_burst_guarded",
    "hr",
    "resp_env",
)

_EOG_SOS = butter(2, [1.5, 6.0], btype="bandpass", fs=FS, output="sos")
_FRONT_SOS = butter(2, [0.5, 4.0], btype="bandpass", fs=FS, output="sos")


def compute_guarded_eog_burst(
    eog_uv: np.ndarray,
    f3_uv: np.ndarray,
    f4_uv: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """Detect rapid E1-M2 bursts (1.5-6 Hz) guarded against frontal (F3/F4) slow waves."""
    n_sec = min(len(eog_uv), len(f3_uv), len(f4_uv)) // fs
    if n_sec == 0:
        return np.empty(0, dtype=np.float32)
    n_samp = n_sec * fs
    e = np.asarray(eog_uv[:n_samp], dtype=np.float32)
    f3 = np.asarray(f3_uv[:n_samp], dtype=np.float32)
    f4 = np.asarray(f4_uv[:n_samp], dtype=np.float32)

    valid_sec = (
        np.isfinite(e).reshape(n_sec, fs).all(axis=1)
        & np.isfinite(f3).reshape(n_sec, fs).all(axis=1)
        & np.isfinite(f4).reshape(n_sec, fs).all(axis=1)
    )
    e_clean = np.where(np.isfinite(e), e, 0.0)
    f_clean = 0.5 * (
        np.where(np.isfinite(f3), f3, 0.0) + np.where(np.isfinite(f4), f4, 0.0)
    )

    sos_e = _EOG_SOS if fs == FS else butter(2, [1.5, 6.0], btype="bandpass", fs=fs, output="sos")
    sos_f = _FRONT_SOS if fs == FS else butter(2, [0.5, 4.0], btype="bandpass", fs=fs, output="sos")
    e_bp = np.abs(sosfiltfilt(sos_e, e_clean))
    f_bp = np.abs(sosfiltfilt(sos_f, f_clean))

    e_1s = e_bp.reshape(n_sec, fs).max(axis=1)
    f_1s = f_bp.reshape(n_sec, fs).max(axis=1)
    out = np.full(n_sec, np.nan, dtype=np.float32)
    if valid_sec.any():
        e_med = float(np.median(e_1s[valid_sec])) + 1e-6
        f_med = float(np.median(f_1s[valid_sec])) + 1e-6
        burst = ((e_1s / e_med > 2.5) & (f_1s / f_med < 2.2)).astype(np.float32)
        out[valid_sec] = burst[valid_sec]
    return out


def extract_missing_cues_for_subject(
    subject: str,
    raw_root: Path,
    dev_tracks_path: Path | None,
    permitted_subjects: Iterable[str],
) -> pd.DataFrame:
    """Extract the 7 validated 1-Hz cues for one subject, reusing Dev tracks1s where available."""
    permitted = frozenset(permitted_subjects)
    if subject not in permitted:
        raise ValueError(f"{subject!r} is outside permitted manifest")

    store = WaveformStore(raw_root, permitted_subjects=permitted, max_open_nights=1)
    try:
        eog = store.channel_uv(subject, "E1-M2")
        f3 = store.channel_uv(subject, "F3-M2")
        f4 = store.channel_uv(subject, "F4-M1")
        burst = compute_guarded_eog_burst(eog, f3, f4, fs=FS)
        n_sec = len(burst)

        if dev_tracks_path is not None and dev_tracks_path.exists():
            cached = pd.read_parquet(
                dev_tracks_path,
                columns=["c_rel_delta", "f_rel_delta", "eog_low", "emg_log_rms", "hr", "resp_env"],
            )
            n_out = min(n_sec, len(cached))
            data = {
                col: cached[col].to_numpy(dtype=np.float32)[:n_out]
                for col in cached.columns
            }
            data["eog_burst_guarded"] = burst[:n_out]
            df = pd.DataFrame({col: np.asarray(data[col], dtype=np.float32) for col in CUE_COLUMNS})
            df.insert(0, "second", np.arange(n_out, dtype=np.int64))
            return df

        c3 = store.channel_uv(subject, "C3-M2")
        c4 = store.channel_uv(subject, "C4-M1")
        chin = store.channel_uv(subject, "Chin1-Chin2")
        n_sec = min(n_sec, len(c3) // FS, len(c4) // FS, len(chin) // FS)
    finally:
        store._nights.clear()

    P_c3 = power_spectra(c3, n_sec)
    P_c4 = power_spectra(c4, n_sec)
    P_f3 = power_spectra(f3, n_sec)
    P_f4 = power_spectra(f4, n_sec)
    P_eog = power_spectra(eog, n_sec)
    P_emg = power_spectra(chin, n_sec)

    C = 0.5 * (P_c3 + P_c4)
    F = 0.5 * (P_f3 + P_f4)
    c_rel_delta = (band_power(C, *BANDS["delta"]) / (band_power(C, 0.5, 30) + EPS)).astype(np.float32)
    f_rel_delta = (band_power(F, *BANDS["delta"]) / (band_power(F, 0.5, 30) + EPS)).astype(np.float32)
    eog_low = np.log10(band_power(P_eog, 0.5, 2) + EPS).astype(np.float32)
    emg_log_rms = (0.5 * np.log10(band_power(P_emg, 10, 80) + EPS)).astype(np.float32)

    import wfdb

    paths = resolve_cinc2018_record(raw_root, subject)
    if paths.header_path is None:
        raise FileNotFoundError(f"Missing header for {subject}")
    sig, _ = wfdb.rdsamp(
        str(paths.header_path.with_suffix("")),
        channel_names=["ABD", "CHEST", "ECG"],
    )
    abd = np.asarray(sig[:, 0], dtype=np.float32)
    chest = np.asarray(sig[:, 1], dtype=np.float32)
    ecg = np.asarray(sig[:, 2], dtype=np.float64)

    _, resp_env = respiratory_tracks(abd + chest, n_sec)
    _, r_peaks_idx, _ = _detect_r_peaks(ecg, FS, backend="wfdb_xqrs")
    hr = heart_rate_track(r_peaks_idx / FS, n_sec)

    df = pd.DataFrame(
        {
            "c_rel_delta": c_rel_delta,
            "f_rel_delta": f_rel_delta,
            "eog_low": eog_low,
            "emg_log_rms": emg_log_rms,
            "eog_burst_guarded": burst[:n_sec],
            "hr": np.asarray(hr, dtype=np.float32),
            "resp_env": np.asarray(resp_env, dtype=np.float32),
        }
    )
    df.insert(0, "second", np.arange(n_sec, dtype=np.int64))
    return df


def get_variant_layout(variant: str) -> dict:
    """Return concept layout metadata for a staged Experiment 80 ablation variant."""
    step = {k: tuple(v) for k, v in CONCEPT_STEP_SIGNALS.items()}
    ep = {k: tuple(v) for k, v in CONCEPT_EPOCH_SIGNALS.items()}
    names = list(CONCEPT_NAMES)
    widths = [16, 16, 16, 16]

    if variant not in {"delta", "rem_emg", "autonomic", "rem_auto", "all"}:
        raise ValueError(f"Unknown Experiment 80 variant: {variant!r}")

    if variant in {"delta", "all"}:
        step["slow_wave"] = (*step["slow_wave"], "f_rel_delta", "c_rel_delta")
    if variant in {"rem_emg", "rem_auto", "all"}:
        step["rem_tone"] = (*step["rem_tone"], "emg_log_rms", "eog_low", "eog_burst_guarded")
    if variant in {"autonomic", "rem_auto", "all"}:
        names.append("autonomic")
        step["autonomic"] = ("hr", "resp_env")
        ep["autonomic"] = ()
        widths.append(8)

    concept_names = tuple(names)
    feature_names, concept_slices, concept_dims = build_concept_layout(
        step_signals=step,
        epoch_signals=ep,
        concept_names=concept_names,
    )
    return {
        "variant": variant,
        "concept_names": concept_names,
        "step_signals": step,
        "epoch_signals": ep,
        "feature_names": feature_names,
        "concept_slices": concept_slices,
        "concept_dims": concept_dims,
        "concept_widths": tuple(widths),
        "input_dim": len(feature_names),
    }


__all__ = [
    "CUE_COLUMNS",
    "compute_guarded_eog_burst",
    "extract_missing_cues_for_subject",
    "get_variant_layout",
]
