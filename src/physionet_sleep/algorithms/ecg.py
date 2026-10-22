from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import CubicSpline
from scipy.signal import sosfiltfilt, welch

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, Event, SignalRecord, SignalSeries


class PanTompkinsDetector(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "ecg", backend: str = "wfdb_xqrs") -> None:
        super().__init__(
            name="pan_tompkins",
            references=("external/st_vincent_analysis/external/pan_tompkins.m",),
        )
        self.signal_key = signal_key
        self.backend = _normalize_backend_name(backend)

    def required_signals(self) -> tuple[str, ...]:
        return (self.signal_key,)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        signal = record.get(self.signal_key)
        ecg = np.asarray(signal.values, dtype=float).squeeze()
        fs = float(signal.sample_rate)
        filtered, r_peak_idx, detector_meta = _detect_r_peaks(ecg, fs, backend=self.backend)

        rr_intervals_sec = np.diff(r_peak_idx) / fs if r_peak_idx.size >= 2 else np.array([])
        heart_rate_bpm = 60.0 / rr_intervals_sec if rr_intervals_sec.size else np.array([])
        beat_ts_ms = (r_peak_idx / fs) * 1000.0 if r_peak_idx.size else np.array([])
        hrv = _beat_domain_hrv_metrics(
            heart_rate_bpm,
            beat_ts_ms[1:] if beat_ts_ms.size >= 2 else np.array([]),
            hrv_win_ms=30_000.0,
            minmax_win_ms=30_000.0,
        )
        heartbeat_mask = np.zeros(ecg.shape[0], dtype=bool)
        heartbeat_mask[r_peak_idx] = True
        heart_rate_track = _scatter_forward_fill(ecg.shape[0], r_peak_idx[1:], heart_rate_bpm)
        rmssd_track = _scatter_forward_fill(ecg.shape[0], r_peak_idx[1:], hrv["rmssd"])
        sdnn_track = _scatter_forward_fill(ecg.shape[0], r_peak_idx[1:], hrv["sdnn"])
        hr_min_track = _scatter_forward_fill(ecg.shape[0], r_peak_idx[1:], hrv["hr_min"])
        hr_range_track = _scatter_forward_fill(ecg.shape[0], r_peak_idx[1:], hrv["hr_range"])

        r_peak_events = [
            Event(
                start=int(idx),
                end=int(idx),
                label="r_peak",
                kind="point",
                metadata={"start_sec": float(idx / fs), "sample_rate_hz": fs},
            )
            for idx in r_peak_idx
        ]

        return AlgorithmResult(
            name=self.name,
            signals={
                "ecg_filtered": SignalSeries(filtered, sample_rate=fs, units=signal.units),
                "heartbeat_mask": SignalSeries(
                    heartbeat_mask.astype(float),
                    sample_rate=fs,
                    units="binary",
                ),
                "heart_rate_bpm_track": SignalSeries(heart_rate_track, sample_rate=fs, units="bpm"),
                "hrv_rmssd_track": SignalSeries(rmssd_track, sample_rate=fs, units="ms"),
                "hrv_sdnn_track": SignalSeries(sdnn_track, sample_rate=fs, units="ms"),
                "heart_rate_min_track": SignalSeries(hr_min_track, sample_rate=fs, units="bpm"),
                "heart_rate_range_track": SignalSeries(hr_range_track, sample_rate=fs, units="bpm"),
            },
            events={"r_peaks": r_peak_events},
            features={
                "input_signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "num_r_peaks": int(r_peak_idx.size),
                "r_peak_indices": r_peak_idx,
                "rr_intervals_sec": rr_intervals_sec,
                "heart_rate_bpm": heart_rate_bpm,
                "mean_heart_rate_bpm": float(np.nanmean(heart_rate_bpm))
                if heart_rate_bpm.size
                else np.nan,
                "mean_hrv_rmssd_ms": float(np.nanmean(hrv["rmssd"])) if hrv["rmssd"].size else np.nan,
                "mean_hrv_sdnn_ms": float(np.nanmean(hrv["sdnn"])) if hrv["sdnn"].size else np.nan,
                "mean_heart_rate_range_bpm": float(np.nanmean(hrv["hr_range"]))
                if hrv["hr_range"].size
                else np.nan,
            },
            metadata={
                "status": "ported_feature_extraction",
                "detector": self.backend,
                "hrv_window_ms": 30_000.0,
                **detector_meta,
            },
        )


def _detect_r_peaks(ecg: np.ndarray, fs: float, *, backend: str) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    backend_name = _normalize_backend_name(backend)
    if backend_name == "adaptive":
        return _detect_r_peaks_pan_tompkins(ecg, fs)
    if backend_name == "wfdb_xqrs":
        return _detect_r_peaks_wfdb_xqrs(ecg, fs)
    if backend_name == "sleepecg":
        return _detect_r_peaks_sleepecg(ecg, fs)
    raise ValueError(f"Unsupported ECG detector backend: {backend}")


def _detect_r_peaks_pan_tompkins(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    ecg = np.asarray(ecg, dtype=float).squeeze()
    if ecg.ndim != 1:
        ecg = ecg.reshape(-1)
    if ecg.size == 0 or fs <= 0:
        return np.asarray(ecg, dtype=float), np.array([], dtype=int), {
            "detector_polarity_flipped": False,
            "detector_num_mvi_peaks": 0,
            "detector_num_bandpassed_peaks": 0,
        }

    centered = ecg - np.nanmean(ecg)
    filtered = _pan_tompkins_bandpass(centered, fs)
    polarity_flipped = False
    if np.nanmax(filtered) < abs(np.nanmin(filtered)):
        filtered = -filtered
        polarity_flipped = True
    filtered = _safe_normalize(filtered)

    derivative = _pan_tompkins_derivative(filtered, fs)
    derivative = _safe_normalize(derivative)
    squared = derivative**2
    window = max(int(round(0.150 * fs)), 1)
    integrated = np.convolve(squared, np.ones(window, dtype=float) / float(window), mode="full")

    candidate_peaks, _ = find_peaks(
        integrated,
        distance=max(int(round(0.2 * fs)), 1),
    )
    if candidate_peaks.size == 0:
        return filtered, np.array([], dtype=int), {
            "detector_polarity_flipped": polarity_flipped,
            "detector_num_mvi_peaks": 0,
            "detector_num_bandpassed_peaks": 0,
        }

    init_len_mvi = min(integrated.size, max(int(round(2.0 * fs)), 1))
    init_len_band = min(filtered.size, max(int(round(2.0 * fs)), 1))
    thr_sig = float(np.nanmax(integrated[:init_len_mvi]) / 3.0) if init_len_mvi > 0 else 0.0
    thr_noise = float(np.nanmean(integrated[:init_len_mvi]) / 2.0) if init_len_mvi > 0 else 0.0
    sig_lev = thr_sig
    noise_lev = thr_noise

    thr_sig1 = float(np.nanmax(filtered[:init_len_band]) / 3.0) if init_len_band > 0 else 0.0
    thr_noise1 = float(np.nanmean(filtered[:init_len_band]) / 2.0) if init_len_band > 0 else 0.0
    sig_lev1 = thr_sig1
    noise_lev1 = thr_noise1

    qrs_i: list[int] = []
    qrs_i_raw: list[int] = []
    qrs_amp_raw: list[float] = []
    m_selected_rr = 0.0
    mean_rr = 0.0
    search_back = max(int(round(0.150 * fs)), 1)
    rr_guard = max(int(round(0.2 * fs)), 1)
    t_wave_guard = max(int(round(0.360 * fs)), 1)
    slope_guard = max(int(round(0.075 * fs)), 1)

    for loc in candidate_peaks.tolist():
        y_i, x_i, ser_back = _filtered_peak_near(filtered, loc, search_back)

        if len(qrs_i) >= 9:
            diff_rr = np.diff(np.asarray(qrs_i[-9:], dtype=float))
            mean_rr = float(np.mean(diff_rr)) if diff_rr.size else 0.0
            comp = float(qrs_i[-1] - qrs_i[-2])
            if comp <= 0.92 * mean_rr or comp >= 1.16 * mean_rr:
                thr_sig *= 0.5
                thr_sig1 *= 0.5
            else:
                m_selected_rr = mean_rr

        test_m = m_selected_rr if m_selected_rr > 0 else mean_rr
        if test_m > 0 and qrs_i and (loc - qrs_i[-1]) >= int(round(1.66 * test_m)):
            sb_start = qrs_i[-1] + rr_guard
            sb_end = loc - rr_guard
            if sb_end > sb_start:
                pks_temp, locs_temp = _peak_in_window(integrated, sb_start, sb_end)
                if pks_temp > thr_noise:
                    qrs_i.append(locs_temp)
                    y_i_t, x_i_t, ser_back_t = _filtered_peak_near(filtered, locs_temp, search_back)
                    if y_i_t > thr_noise1:
                        raw_idx = x_i_t if ser_back_t else locs_temp - search_back + x_i_t
                        raw_idx = int(np.clip(raw_idx, 0, filtered.size - 1))
                        qrs_i_raw.append(raw_idx)
                        qrs_amp_raw.append(float(y_i_t))
                        sig_lev1 = (0.25 * float(y_i_t)) + (0.75 * sig_lev1)
                    sig_lev = (0.25 * float(pks_temp)) + (0.75 * sig_lev)

        peak_value = float(integrated[loc])
        skip = False
        if peak_value >= thr_sig:
            if len(qrs_i) >= 3 and (loc - qrs_i[-1]) <= t_wave_guard:
                slope1 = _mean_slope(integrated, loc, slope_guard)
                slope2 = _mean_slope(integrated, qrs_i[-1], slope_guard)
                if abs(slope1) <= abs(0.5 * slope2):
                    noise_lev1 = (0.125 * float(y_i)) + (0.875 * noise_lev1)
                    noise_lev = (0.125 * peak_value) + (0.875 * noise_lev)
                    skip = True

            if not skip:
                qrs_i.append(loc)
                if y_i >= thr_sig1:
                    raw_idx = x_i if ser_back else loc - search_back + x_i
                    raw_idx = int(np.clip(raw_idx, 0, filtered.size - 1))
                    qrs_i_raw.append(raw_idx)
                    qrs_amp_raw.append(float(y_i))
                    sig_lev1 = (0.125 * float(y_i)) + (0.875 * sig_lev1)
                sig_lev = (0.125 * peak_value) + (0.875 * sig_lev)
        elif thr_noise <= peak_value < thr_sig:
            noise_lev1 = (0.125 * float(y_i)) + (0.875 * noise_lev1)
            noise_lev = (0.125 * peak_value) + (0.875 * noise_lev)
        else:
            noise_lev1 = (0.125 * float(y_i)) + (0.875 * noise_lev1)
            noise_lev = (0.125 * peak_value) + (0.875 * noise_lev)

        if noise_lev != 0.0 or sig_lev != 0.0:
            thr_sig = noise_lev + (0.25 * abs(sig_lev - noise_lev))
            thr_noise = 0.5 * thr_sig
        if noise_lev1 != 0.0 or sig_lev1 != 0.0:
            thr_sig1 = noise_lev1 + (0.25 * abs(sig_lev1 - noise_lev1))
            thr_noise1 = 0.5 * thr_sig1

    if qrs_i_raw:
        r_peak_idx = np.unique(np.asarray(qrs_i_raw, dtype=int))
    elif qrs_i:
        r_peak_idx = np.unique(np.clip(np.asarray(qrs_i, dtype=int) - search_back, 0, filtered.size - 1))
    else:
        r_peak_idx = np.array([], dtype=int)

    return filtered, r_peak_idx, {
        "detector_polarity_flipped": polarity_flipped,
        "detector_num_mvi_peaks": int(len(qrs_i)),
        "detector_num_bandpassed_peaks": int(r_peak_idx.size),
    }


def _detect_r_peaks_wfdb_xqrs(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    try:
        from wfdb import processing
    except ImportError as exc:
        raise ImportError("wfdb is required for backend='wfdb_xqrs'.") from exc

    centered = np.asarray(ecg, dtype=float).squeeze()
    centered = centered - np.nanmean(centered)
    filtered = _pan_tompkins_bandpass(centered, fs)
    polarity_flipped = False
    if np.nanmax(filtered) < abs(np.nanmin(filtered)):
        filtered = -filtered
        polarity_flipped = True
    filtered = _safe_normalize(filtered)

    r_peak_idx = np.asarray(
        processing.xqrs_detect(
            sig=centered,
            fs=fs,
            learn=True,
            verbose=False,
        ),
        dtype=int,
    )
    r_peak_idx = np.unique(np.clip(r_peak_idx, 0, max(filtered.size - 1, 0)))
    return filtered, r_peak_idx, {
        "detector_polarity_flipped": polarity_flipped,
        "detector_num_mvi_peaks": int(r_peak_idx.size),
        "detector_num_bandpassed_peaks": int(r_peak_idx.size),
        "detector_source": "wfdb.processing.xqrs_detect",
    }


def _detect_r_peaks_sleepecg(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    try:
        from sleepecg import detect_heartbeats
    except ImportError as exc:
        raise ImportError("sleepecg is required for backend='sleepecg'.") from exc

    centered = np.asarray(ecg, dtype=float).squeeze()
    centered = centered - np.nanmean(centered)
    filtered = _pan_tompkins_bandpass(centered, fs)
    polarity_flipped = False
    if np.nanmax(filtered) < abs(np.nanmin(filtered)):
        filtered = -filtered
        polarity_flipped = True
    filtered = _safe_normalize(filtered)

    r_peak_idx = np.asarray(detect_heartbeats(centered, fs), dtype=int)
    r_peak_idx = np.unique(np.clip(r_peak_idx, 0, max(filtered.size - 1, 0)))
    return filtered, r_peak_idx, {
        "detector_polarity_flipped": polarity_flipped,
        "detector_num_mvi_peaks": int(r_peak_idx.size),
        "detector_num_bandpassed_peaks": int(r_peak_idx.size),
        "detector_source": "sleepecg.detect_heartbeats",
    }


def _pan_tompkins_bandpass(values: np.ndarray, fs: float) -> np.ndarray:
    if values.size < 8:
        return values.astype(float, copy=True)
    if int(round(fs)) == 200:
        low = _safe_filtfilt(*butter(3, 12.0 * 2.0 / fs, btype="low"), values)
        high = _safe_filtfilt(*butter(3, 5.0 * 2.0 / fs, btype="high"), low)
        return high
    wn = np.asarray([5.0, 15.0], dtype=float) * 2.0 / float(fs)
    wn[1] = min(wn[1], 0.99)
    wn[0] = min(wn[0], wn[1] * 0.8)
    b, a = butter(3, wn, btype="bandpass")
    return _safe_filtfilt(b, a, values)


def _pan_tompkins_derivative(values: np.ndarray, fs: float) -> np.ndarray:
    if values.size == 0:
        return values.astype(float, copy=True)
    if int(round(fs)) == 200:
        kernel = np.asarray([1.0, 2.0, 0.0, -2.0, -1.0], dtype=float) * ((1.0 / 8.0) * fs)
    else:
        step = (5.0 - 1.0) / (fs * (1.0 / 40.0))
        step = step if step > 0 else 1.0
        sample_points = np.arange(1.0, 5.0 + 1e-9, step, dtype=float)
        kernel = np.interp(
            sample_points,
            np.arange(1.0, 6.0, dtype=float),
            np.asarray([1.0, 2.0, 0.0, -2.0, -1.0], dtype=float) * ((1.0 / 8.0) * fs),
        )
    return _safe_filtfilt(kernel, np.asarray([1.0], dtype=float), values)


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, values: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size <= max(len(a), len(b)) * 3:
        return np.convolve(values, b, mode="same") if a.size == 1 else values.copy()
    return filtfilt(b, a, values)


def _safe_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scale = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return arr.copy()
    return arr / scale


def _filtered_peak_near(filtered: np.ndarray, loc: int, search_back: int) -> tuple[float, int, bool]:
    start = max(loc - search_back, 0)
    end = min(loc + 1, filtered.size)
    if end <= start:
        return 0.0, 0, False
    local = filtered[start:end]
    x_i = int(np.argmax(local))
    return float(local[x_i]), x_i, start == 0


def _peak_in_window(values: np.ndarray, start: int, end: int) -> tuple[float, int]:
    start = max(int(start), 0)
    end = min(int(end), values.size)
    if end <= start:
        return 0.0, start
    window = values[start:end]
    local_idx = int(np.argmax(window))
    return float(window[local_idx]), start + local_idx


def _mean_slope(values: np.ndarray, loc: int, width: int) -> float:
    start = max(int(loc) - int(width), 0)
    end = min(int(loc) + 1, values.size)
    if end - start < 2:
        return 0.0
    return float(np.mean(np.diff(values[start:end])))


def _normalize_backend_name(backend: str) -> str:
    value = str(backend).strip().lower().replace("-", "_")
    aliases = {
        "adaptive": "adaptive",
        "pan_tompkins": "adaptive",
        "pan_tompkins_adaptive": "adaptive",
        "wfdb": "wfdb_xqrs",
        "xqrs": "wfdb_xqrs",
        "wfdb_xqrs": "wfdb_xqrs",
        "sleepecg": "sleepecg",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported ECG detector backend: {backend}")
    return aliases[value]


def _scatter_forward_fill(length: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(length, np.nan, dtype=float)
    if len(indices) == 0 or len(values) == 0:
        return out
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(values, dtype=float)
    n = min(indices.size, values.size)
    out[np.clip(indices[:n], 0, length - 1)] = values[:n]
    valid = np.flatnonzero(np.isfinite(out))
    if valid.size == 0:
        return out
    out[: valid[0]] = out[valid[0]]
    for i in range(valid[0] + 1, out.size):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out


def _beat_domain_hrv_metrics(
    heart_rate_bpm: np.ndarray,
    beat_ts_ms: np.ndarray,
    *,
    hrv_win_ms: float,
    minmax_win_ms: float,
) -> dict[str, np.ndarray]:
    hr_bpm = np.asarray(heart_rate_bpm, dtype=float).ravel()
    ts = np.asarray(beat_ts_ms, dtype=float).ravel()
    n = hr_bpm.size

    if n < 2 or ts.size != n:
        nan_arr = np.full(n, np.nan, dtype=float)
        return {
            "rmssd": nan_arr,
            "sdnn": nan_arr,
            "hr_min": hr_bpm.copy(),
            "hr_range": np.zeros(n, dtype=float),
        }

    rr_ms = np.diff(ts)
    rr_valid = np.isfinite(rr_ms) & (rr_ms > 0)
    trr = ts[1:]
    lo, hi = _window_bounds(trr, hrv_win_ms)

    rrz = rr_ms.copy()
    rrz[~rr_valid] = 0.0
    cs = np.concatenate(([0.0], np.cumsum(rrz)))
    cs2 = np.concatenate(([0.0], np.cumsum(rrz**2)))
    cn = np.concatenate(([0.0], np.cumsum(rr_valid.astype(float))))

    cnt = cn[hi + 1] - cn[lo]
    mu = (cs[hi + 1] - cs[lo]) / np.maximum(cnt, 1.0)
    ss = (cs2[hi + 1] - cs2[lo]) - cnt * mu**2
    sdnn = np.sqrt(np.maximum(ss, 0.0) / np.maximum(cnt - 1.0, 1.0))
    sdnn[cnt < 2] = np.nan

    dok = rr_valid[:-1] & rr_valid[1:]
    d2 = np.diff(rr_ms) ** 2
    d2[~dok] = 0.0
    csd = np.concatenate(([0.0], np.cumsum(d2)))
    cdn = np.concatenate(([0.0], np.cumsum(dok.astype(float))))
    nd = cdn[hi] - cdn[lo]
    rmssd = np.sqrt((csd[hi] - csd[lo]) / np.maximum(nd, 1.0))
    rmssd[nd < 1] = np.nan

    ts_mono = ts.copy()
    for i in range(1, ts_mono.size):
        if ts_mono[i] <= ts_mono[i - 1]:
            ts_mono[i] = ts_mono[i - 1] + 1e-3
    lo2, hi2 = _window_bounds(ts_mono, minmax_win_ms)
    hr_min = np.array([np.min(hr_bpm[lo2[i] : hi2[i] + 1]) for i in range(n)], dtype=float)
    hr_max = np.array([np.max(hr_bpm[lo2[i] : hi2[i] + 1]) for i in range(n)], dtype=float)

    return {
        "rmssd": np.concatenate(([np.nan], rmssd)),
        "sdnn": np.concatenate(([np.nan], sdnn)),
        "hr_min": hr_min,
        "hr_range": hr_max - hr_min,
    }


def _window_bounds(timestamps_ms: np.ndarray, win_ms: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(timestamps_ms)
    half = win_ms / 2.0
    lo = np.zeros(n, dtype=np.int64)
    hi = np.zeros(n, dtype=np.int64)
    p = 0
    q = 0
    for i in range(n):
        while p < i and timestamps_ms[p] < timestamps_ms[i] - half:
            p += 1
        while q < n - 1 and timestamps_ms[q + 1] <= timestamps_ms[i] + half:
            q += 1
        if q < i:
            q = i
        lo[i] = p
        hi[i] = q
    return lo, hi


# --- Cleaned-NN heart-rate variability on literature-length windows -------------------------------------------------
# Exploration and evidence: artifacts/exploration/hrv-features/FINDINGS.md. Every track is sampled on a 1-s grid; the value
# at sample k describes the window centred on t = (k + 0.5) * step_seconds.

NN_MIN_MS = 300.0
NN_MAX_MS = 2000.0
NN_LOCAL_BEATS = 9
NN_LOCAL_TOLERANCE = 0.2
NN_GRID_HZ = 4.0


class HRVFeatureGenerator(BaseAlgorithm):
    """Sleep-stage HRV features from cleaned, sub-sample-refined NN intervals.

    Reads R-peak sample indices from a prior ``PanTompkinsDetector`` result and the raw ECG (for peak refinement).
    """

    def __init__(
        self,
        *,
        signal_key: str = "ecg",
        peaks_source: str = "pan_tompkins",
        step_seconds: float = 1.0,
        refine_peaks: bool = True,
    ) -> None:
        super().__init__(name="hrv_features", references=("artifacts/exploration/hrv-features/FINDINGS.md",))
        self.signal_key = signal_key
        self.peaks_source = peaks_source
        self.step_seconds = float(step_seconds)
        self.refine_peaks = refine_peaks

    def required_signals(self) -> tuple[str, ...]:
        return (self.signal_key,)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        if self.peaks_source not in prior:
            raise ValueError(f"{self.name} requires a prior '{self.peaks_source}' result with r_peak_indices")
        signal = record.get(self.signal_key)
        ecg = np.asarray(signal.values, dtype=float).squeeze()
        fs = float(signal.sample_rate)
        peak_idx = np.asarray(prior[self.peaks_source].features.get("r_peak_indices", []), dtype=int)
        if self.refine_peaks:
            peak_times = refine_r_peaks(ecg, fs, peak_idx)
        else:
            peak_times = peak_idx / fs
        duration = ecg.size / fs if fs > 0 else 0.0
        tracks = compute_hrv_tracks(peak_times, duration_seconds=duration, step_seconds=self.step_seconds)
        nn = tracks.pop("_nn")
        rate = 1.0 / self.step_seconds
        units = {
            "hr_bpm_30s": "bpm",
            "sdnn_150s": "ms",
            "hr_slow_sd_300s": "bpm",
            "dfa_a2_300s": "exponent",
            "ln_lf_hf_300s": "ln_ratio",
            "hr_surge_30s": "bpm",
            "nn_clean_frac_300s": "fraction",
        }
        shift_ms = (peak_times - peak_idx / fs) * 1000.0 if peak_idx.size else np.array([])
        return AlgorithmResult(
            name=self.name,
            signals={key: SignalSeries(values, sample_rate=rate, units=units[key]) for key, values in tracks.items()},
            features={
                "nn_times_sec": nn["t_end"],
                "nn_ms": nn["rr_ms"],
                "nn_clean": nn["clean"],
                "refined_peak_times_sec": peak_times,
                "nn_clean_fraction": float(nn["clean"].mean()) if nn["clean"].size else np.nan,
                "refine_shift_abs_ms_mean": float(np.mean(np.abs(shift_ms))) if shift_ms.size else np.nan,
            },
            metadata={
                "status": "hrv_features_v1",
                "peaks_source": self.peaks_source,
                "refine_peaks": self.refine_peaks,
                "step_seconds": self.step_seconds,
                "track_time_reference": "window centre = (sample + 0.5) * step_seconds",
            },
        )


def clean_nn_mask(
    rr_ms: np.ndarray,
    *,
    min_ms: float = NN_MIN_MS,
    max_ms: float = NN_MAX_MS,
    local_beats: int = NN_LOCAL_BEATS,
    tolerance: float = NN_LOCAL_TOLERANCE,
) -> np.ndarray:
    """True for RR intervals that are physiological and within ``tolerance`` of the centred local median."""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size == 0:
        return np.zeros(0, dtype=bool)
    local = _rolling_centered_median(rr, local_beats, min_count=3)
    return np.isfinite(rr) & (rr >= min_ms) & (rr <= max_ms) & (np.abs(rr - local) <= tolerance * local)


def refine_r_peaks(
    ecg: np.ndarray,
    fs: float,
    peak_idx: np.ndarray,
    *,
    half_window_seconds: float = 0.05,
    upsample_hz: float = 1000.0,
) -> np.ndarray:
    """Sub-sample R-peak times (seconds): cubic-spline upsampling of the 0.5-40 Hz ECG around each detected peak.

    The extremum is taken in the record's R polarity (sign of the median filtered value at the detected peaks), which
    also realigns detectors that mark the QRS upstroke inconsistently.
    """
    ecg = np.asarray(ecg, dtype=float).squeeze()
    peak_idx = np.asarray(peak_idx, dtype=int)
    if peak_idx.size == 0 or ecg.size < 16 or fs <= 0:
        return peak_idx / fs if fs > 0 else peak_idx.astype(float)
    high = min(40.0, 0.45 * fs)
    sos = butter(2, [0.5, high], btype="bandpass", fs=fs, output="sos")
    filt = sosfiltfilt(sos, ecg - np.nanmean(ecg))
    sign = 1.0 if np.median(filt[np.clip(peak_idx, 0, filt.size - 1)]) >= 0 else -1.0
    half = int(round(half_window_seconds * fs))
    fine = np.arange(-half, half + 1e-9, fs / upsample_hz)
    out = peak_idx.astype(float) / fs
    for k, i in enumerate(peak_idx):
        a, b = i - half - 2, i + half + 3
        if a < 0 or b > filt.size:
            continue
        spline = CubicSpline(np.arange(a, b), sign * filt[a:b])
        out[k] = (i + fine[int(np.argmax(spline(i + fine)))]) / fs
    return np.sort(out)


def dfa_alpha(x: np.ndarray, scales: np.ndarray) -> float:
    """Detrended fluctuation analysis exponent (linear detrending) over the given box sizes in beats."""
    x = np.asarray(x, dtype=float)
    scales = np.asarray(scales, dtype=int)
    if x.size < 4 * scales.max():
        return np.nan
    y = np.cumsum(x - x.mean())
    flucts = []
    for n in scales:
        m = y.size // n
        seg = y[: m * n].reshape(m, n)
        t = np.arange(n, dtype=float) - (n - 1) / 2.0
        centred = seg - seg.mean(axis=1, keepdims=True)
        slope = centred @ t / (t @ t)
        resid = centred - slope[:, None] * t
        flucts.append(np.sqrt(np.mean(resid**2)))
    flucts = np.asarray(flucts)
    if np.any(flucts <= 0):
        return np.nan
    return float(np.polyfit(np.log(scales), np.log(flucts), 1)[0])


DFA_A2_SCALES = np.unique(np.round(np.geomspace(16, 64, 7)).astype(int))


def nn_intervals(peak_times_sec: np.ndarray) -> dict[str, np.ndarray]:
    """RR intervals (ms), the time of the beat closing each interval (s), and the clean-NN mask."""
    t = np.sort(np.asarray(peak_times_sec, dtype=float))
    rr = np.diff(t) * 1000.0
    return {"rr_ms": rr, "t_end": t[1:] if t.size else t, "clean": clean_nn_mask(rr)}


def nn_grid(nn: Mapping[str, np.ndarray], duration_seconds: float, grid_hz: float = NN_GRID_HZ) -> np.ndarray:
    """Clean NN (ms) linearly interpolated onto a uniform grid starting at t=0 (sample k at k / grid_hz)."""
    grid = np.arange(0.0, duration_seconds, 1.0 / grid_hz)
    ok = nn["clean"]
    if ok.sum() < 2:
        return np.full(grid.size, np.nan)
    return np.interp(grid, nn["t_end"][ok], nn["rr_ms"][ok])


def compute_hrv_tracks(
    peak_times_sec: np.ndarray,
    *,
    duration_seconds: float,
    step_seconds: float = 1.0,
    min_clean_fraction: float = 0.8,
) -> dict[str, np.ndarray]:
    """Centred-window HRV tracks on a ``step_seconds`` grid. Windows with < 80% clean beats or < 60% of the expected
    beat count are NaN. Also returns the NN intervals under the private key ``_nn``."""
    nn = nn_intervals(peak_times_sec)
    rr, t, ok = nn["rr_ms"], nn["t_end"], nn["clean"]
    n_out = max(int(np.ceil(duration_seconds / step_seconds)), 0)
    centres = (np.arange(n_out) + 0.5) * step_seconds
    names = ["hr_bpm_30s", "sdnn_150s", "hr_slow_sd_300s", "dfa_a2_300s", "ln_lf_hf_300s", "hr_surge_30s", "nn_clean_frac_300s"]
    out = {name: np.full(n_out, np.nan) for name in names}
    out["_nn"] = nn
    if ok.sum() < 10 or n_out == 0:
        return out
    hr = 60000.0 / rr
    mean_rr = float(np.median(rr[ok]))
    ok_cum = np.concatenate(([0], np.cumsum(ok)))
    # 5-beat rolling mean HR over clean beats (>= 4 of 5) for arousal surges.
    hr_clean = np.where(ok, hr, np.nan)
    hr_roll = _rolling_trailing_nanmean(hr_clean, 5, min_count=4)
    # 10-s bin means of clean HR on an absolute grid, for slow HR fluctuation.
    n_bins = int(np.ceil(duration_seconds / 10.0)) + 1
    bins = (t[ok] // 10.0).astype(int)
    sums = np.bincount(bins, weights=hr[ok], minlength=n_bins)[:n_bins]
    counts = np.bincount(bins, minlength=n_bins)[:n_bins]
    hr_bins = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    grid = nn_grid(nn, duration_seconds)
    g_hz = NN_GRID_HZ

    def bounds(half: float) -> tuple[np.ndarray, np.ndarray]:
        return np.searchsorted(t, centres - half), np.searchsorted(t, centres + half)

    def quality(a: np.ndarray, b: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
        n = b - a
        frac = np.where(n > 0, (ok_cum[b] - ok_cum[a]) / np.maximum(n, 1), np.nan)
        good = (n >= 0.6 * width * 1000.0 / mean_rr) & (frac >= min_clean_fraction)
        return frac, good

    a30, b30 = bounds(15.0)
    _, good30 = quality(a30, b30, 30.0)
    a150, b150 = bounds(75.0)
    _, good150 = quality(a150, b150, 150.0)
    a300, b300 = bounds(150.0)
    frac300, good300 = quality(a300, b300, 300.0)
    out["nn_clean_frac_300s"] = frac300
    for k in range(n_out):
        c = centres[k]
        if good30[k]:
            sel = slice(a30[k], b30[k])
            out["hr_bpm_30s"][k] = float(np.median(hr[sel][ok[sel]]))
        if good150[k]:
            sel = slice(a150[k], b150[k])
            out["sdnn_150s"][k] = float(np.std(rr[sel][ok[sel]], ddof=1))
        if good300[k]:
            sel = slice(a300[k], b300[k])
            out["dfa_a2_300s"][k] = dfa_alpha(rr[sel][ok[sel]], DFA_A2_SCALES)
            g0, g1 = int(round((c - 150.0) * g_hz)), int(round((c + 150.0) * g_hz))
            if g0 >= 0 and g1 <= grid.size:
                out["ln_lf_hf_300s"][k] = _ln_lf_hf(grid[g0:g1], g_hz)
        k0 = int((c - 150.0) // 10.0)
        window = hr_bins[max(k0, 0) : max(k0 + 30, 0)]
        if np.isfinite(window).sum() >= 20:
            out["hr_slow_sd_300s"][k] = float(np.nanstd(window, ddof=1))
        # Surge: peak 5-beat HR inside the 30-s window minus the median clean HR of the preceding 300 s.
        e0, e1 = a30[k], b30[k]
        p0 = int(np.searchsorted(t, c - 15.0 - 300.0))
        base = hr[p0:e0][ok[p0:e0]]
        if e1 > e0 and base.size >= 100 and np.isfinite(hr_roll[e0:e1]).any():
            out["hr_surge_30s"][k] = float(np.nanmax(hr_roll[e0:e1]) - np.median(base))
    return out


def _ln_lf_hf(segment_ms: np.ndarray, grid_hz: float) -> float:
    if segment_ms.size < 0.95 * 300.0 * grid_hz or not np.all(np.isfinite(segment_ms)):
        return np.nan
    x = np.arange(segment_ms.size)
    detrended = segment_ms - np.polyval(np.polyfit(x, segment_ms, 1), x)
    nper = int(128 * grid_hz)
    f, p = welch(detrended, fs=grid_hz, nperseg=nper, noverlap=nper // 2, window="hann")
    lf = p[(f >= 0.04) & (f < 0.15)].sum()
    hf = p[(f >= 0.15) & (f <= 0.40)].sum()
    if lf <= 0 or hf <= 0:
        return np.nan
    return float(np.log(lf / hf))


def _rolling_centered_median(values: np.ndarray, window: int, *, min_count: int) -> np.ndarray:
    n = values.size
    half = window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = values[max(i - half, 0) : i + half + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= min_count:
            out[i] = np.median(seg)
    return out


def _rolling_trailing_nanmean(values: np.ndarray, window: int, *, min_count: int) -> np.ndarray:
    finite = np.isfinite(values)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(finite)))
    idx = np.arange(values.size)
    lo = np.maximum(idx - window + 1, 0)
    cnt = ccnt[idx + 1] - ccnt[lo]
    total = csum[idx + 1] - csum[lo]
    return np.where(cnt >= min_count, total / np.maximum(cnt, 1), np.nan)
