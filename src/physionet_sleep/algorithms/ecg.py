from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.signal import find_peaks

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, Event, SignalRecord, SignalSeries
from physionet_sleep.utils.signal import bandpass_filter


class PanTompkinsDetector(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "ecg") -> None:
        super().__init__(
            name="pan_tompkins",
            references=("external/st_vincent_analysis/external/pan_tompkins.m",),
        )
        self.signal_key = signal_key

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

        filtered = bandpass_filter(ecg - np.mean(ecg), fs, 5.0, 15.0, order=3)
        derivative = np.gradient(filtered) * fs
        squared = derivative**2
        window = max(int(round(0.150 * fs)), 1)
        integrated = np.convolve(squared, np.ones(window) / window, mode="same")

        threshold = float(np.median(integrated) + 0.5 * np.std(integrated))
        prominence = max(float(np.std(integrated) * 0.25), 1e-8)
        candidate_peaks, _ = find_peaks(
            integrated,
            distance=max(int(round(0.2 * fs)), 1),
            height=threshold,
            prominence=prominence,
        )

        search_back = max(int(round(0.15 * fs)), 1)
        r_peaks: list[int] = []
        for peak in candidate_peaks:
            start = max(peak - search_back, 0)
            end = min(peak + 1, ecg.shape[0])
            local = ecg[start:end]
            if local.size == 0:
                continue
            local_idx = int(np.argmax(np.abs(local)))
            r_peaks.append(start + local_idx)

        if r_peaks:
            r_peak_idx = np.unique(np.asarray(r_peaks, dtype=int))
        else:
            r_peak_idx = np.array([], dtype=int)

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
                "hrv_window_ms": 30_000.0,
            },
        )


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
