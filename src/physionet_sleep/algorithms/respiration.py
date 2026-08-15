from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, hilbert, peak_widths

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, Event, SignalRecord, SignalSeries
from physionet_sleep.utils.signal import bandpass_filter


class RespirationDetector(BaseAlgorithm):
    def __init__(
        self,
        *,
        signal_keys: tuple[str, ...] = ("flow",),
        cutoff: float = 5.0,
    ) -> None:
        super().__init__(
            name="respiration",
            references=(
                "external/st_vincent_analysis/apnea/algo_sleep_respiration.m",
                "external/st_vincent_analysis/apnea/algo_ppg_peakagreement.m",
            ),
        )
        self.signal_keys = signal_keys
        self.cutoff = cutoff

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def validate(self, record: SignalRecord) -> None:
        if not any(key in record.channels for key in self.signal_keys):
            raise ValueError(f"{self.name} requires at least one of: {', '.join(self.signal_keys)}")

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        available_keys = [key for key in self.signal_keys if key in record.channels]
        rr_peak_tracks: list[np.ndarray] = []
        rr_trough_tracks: list[np.ndarray] = []
        peak_lists: list[np.ndarray] = []
        trough_lists: list[np.ndarray] = []
        peak_width_lists: list[np.ndarray] = []
        trough_width_lists: list[np.ndarray] = []
        filtered_signals: dict[str, np.ndarray] = {}
        phase_signals: dict[str, np.ndarray] = {}
        envelopes: dict[str, np.ndarray] = {}
        per_signal: dict[str, dict[str, object]] = {}
        fs_by_key: dict[str, float] = {}

        for key in available_keys:
            series = record.get(key)
            signal = np.asarray(series.values, dtype=float).squeeze()
            fs = float(series.sample_rate)
            fs_by_key[key] = fs

            filtered = bandpass_filter(signal, fs, 0.1, 0.5, order=3)
            analytic = hilbert(filtered)
            phase_cos = np.cos(np.angle(analytic))
            envelope = np.abs(analytic)

            pos_idx, pos_width, pos_prom = _find_phase_extrema(phase_cos, positive=True)
            neg_idx, neg_width, neg_prom = _find_phase_extrema(phase_cos, positive=False)

            pos_keep = (pos_prom >= 1.9) & (signal[np.clip(pos_idx, 0, signal.size - 1)] >= self.cutoff)
            neg_keep = (neg_prom >= 1.9) & (signal[np.clip(neg_idx, 0, signal.size - 1)] <= -self.cutoff)
            pos_idx, pos_width = pos_idx[pos_keep], pos_width[pos_keep]
            neg_idx, neg_width = neg_idx[neg_keep], neg_width[neg_keep]

            pos_idx = _shift_to_local_extrema(signal, pos_idx, pos_width, positive=True)
            neg_idx = _shift_to_local_extrema(signal, neg_idx, neg_width, positive=False)
            pos_idx, pos_width = _stable_unique_with_width(pos_idx, pos_width)
            neg_idx, neg_width = _stable_unique_with_width(neg_idx, neg_width)

            rr_peak, rr_peak_time = _instant_resp_rate(pos_idx, fs)
            rr_trough, rr_trough_time = _instant_resp_rate(neg_idx, fs)
            rr_peak_full = _interp_nearest(rr_peak_time, rr_peak, signal.size)
            rr_trough_full = _interp_nearest(rr_trough_time, rr_trough, signal.size)

            rr_peak_tracks.append(rr_peak_full)
            rr_trough_tracks.append(rr_trough_full)
            peak_lists.append(pos_idx)
            trough_lists.append(neg_idx)
            peak_width_lists.append(pos_width)
            trough_width_lists.append(neg_width)
            filtered_signals[key] = filtered
            phase_signals[key] = phase_cos
            envelopes[key] = envelope

            per_signal[key] = {
                "sample_rate_hz": fs,
                "peak_indices": pos_idx,
                "trough_indices": neg_idx,
                "peak_widths": pos_width,
                "trough_widths": neg_width,
                "peak_rr_bpm": rr_peak,
                "trough_rr_bpm": rr_trough,
                "breath_count": int(pos_idx.size),
            }

        peak_stack = np.column_stack(rr_peak_tracks) if rr_peak_tracks else np.empty((0, 0))
        trough_stack = np.column_stack(rr_trough_tracks) if rr_trough_tracks else np.empty((0, 0))
        med_rr = _nanmedian_across_last_axes(peak_stack, trough_stack)
        mse_scores = []
        for i, key in enumerate(available_keys):
            diff = (rr_peak_tracks[i] - med_rr) ** 2 + (rr_trough_tracks[i] - med_rr) ** 2
            mse = _nanmean_or_inf(diff)
            if np.isnan(mse):
                mse = float("inf")
            mse_scores.append(mse)
            per_signal[key]["agreement_mse"] = mse

        sortidx = np.argsort(np.asarray(mse_scores))
        sorted_keys = [available_keys[i] for i in sortidx]
        sorted_peak_lists = [peak_lists[i] for i in sortidx]
        sorted_trough_lists = [trough_lists[i] for i in sortidx]
        sorted_peak_widths = [peak_width_lists[i] for i in sortidx]
        sorted_trough_widths = [trough_width_lists[i] for i in sortidx]

        consensus_peaks, peak_agree = _peak_agreement(sorted_peak_lists, sorted_peak_widths)
        consensus_troughs, trough_agree = _peak_agreement(sorted_trough_lists, sorted_trough_widths)

        selected_key = sorted_keys[0]
        selected_fs = fs_by_key[selected_key]
        consensus_peak_rr, consensus_peak_time = _instant_resp_rate(consensus_peaks, selected_fs)
        consensus_trough_rr, consensus_trough_time = _instant_resp_rate(consensus_troughs, selected_fs)
        rr_peak_full = _interp_nearest(consensus_peak_time, consensus_peak_rr, filtered_signals[selected_key].size)
        rr_trough_full = _interp_nearest(consensus_trough_time, consensus_trough_rr, filtered_signals[selected_key].size)
        rr_consensus = _nanmedian_pair(rr_peak_full, rr_trough_full)

        breath_events = _build_point_events(consensus_peaks, "breath_peak", selected_fs)
        trough_events = _build_point_events(consensus_troughs, "breath_trough", selected_fs)

        return AlgorithmResult(
            name=self.name,
            signals={
                "respiration_filtered": SignalSeries(filtered_signals[selected_key], sample_rate=selected_fs),
                "respiration_phase_cos": SignalSeries(phase_signals[selected_key], sample_rate=selected_fs),
                "respiration_envelope": SignalSeries(envelopes[selected_key], sample_rate=selected_fs),
                "respiration_rate_peak_bpm": SignalSeries(rr_peak_full, sample_rate=selected_fs, units="bpm"),
                "respiration_rate_trough_bpm": SignalSeries(rr_trough_full, sample_rate=selected_fs, units="bpm"),
                "respiration_rate_bpm": SignalSeries(rr_consensus, sample_rate=selected_fs, units="bpm"),
            },
            events={"breaths": breath_events, "troughs": trough_events},
            features={
                "input_signal_keys": tuple(available_keys),
                "selected_signal_key": selected_key,
                "signal_rank_order": sorted_keys,
                "sample_rates_hz": fs_by_key,
                "consensus_peak_indices": consensus_peaks,
                "consensus_trough_indices": consensus_troughs,
                "consensus_peak_agreement": peak_agree,
                "consensus_trough_agreement": trough_agree,
                "per_signal": per_signal,
                "mean_respiration_rate_bpm": _nanmean_or_nan(rr_consensus),
            },
            metadata={"status": "ported_feature_extraction"},
        )


def _find_phase_extrema(phase_cos: np.ndarray, *, positive: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = phase_cos if positive else -phase_cos
    idx, props = find_peaks(work)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    widths = peak_widths(work, idx, rel_height=0.5)[0]
    prominences = props.get("prominences")
    if prominences is None:
        prominences = peak_widths(work, idx, rel_height=1.0)[1]
    return idx.astype(int), widths.astype(float), np.asarray(prominences, dtype=float)


def _shift_to_local_extrema(signal: np.ndarray, idx: np.ndarray, widths: np.ndarray, *, positive: bool) -> np.ndarray:
    out = idx.astype(int).copy()
    for i, peak in enumerate(out):
        half_width = max(int(round(widths[i] / 2.0)), 1)
        start = max(peak - half_width, 0)
        end = min(peak + half_width + 1, signal.size)
        if positive:
            local = np.argmax(signal[start:end])
        else:
            local = np.argmin(signal[start:end])
        out[i] = start + int(local)
    return out


def _stable_unique_with_width(idx: np.ndarray, widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        return idx.astype(int), widths.astype(float)
    seen = {}
    keep_idx = []
    keep_width = []
    for peak, width in zip(idx.tolist(), widths.tolist(), strict=False):
        if peak in seen:
            continue
        seen[peak] = True
        keep_idx.append(int(peak))
        keep_width.append(float(width))
    return np.asarray(keep_idx, dtype=int), np.asarray(keep_width, dtype=float)


def _instant_resp_rate(peak_idx: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if peak_idx.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    rr = 60.0 / (np.diff(peak_idx) / sample_rate_hz)
    rr = np.concatenate([rr, rr[-1:]])
    return rr.astype(float), peak_idx.astype(float)


def _interp_nearest(x: np.ndarray, y: np.ndarray, output_length: int) -> np.ndarray:
    if x.size == 0 or y.size == 0:
        return np.full(output_length, np.nan, dtype=float)
    f = interp1d(x, y, kind="nearest", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    return np.asarray(f(np.arange(output_length)), dtype=float)


def _peak_agreement(peak_lists: list[np.ndarray], width_lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    remaining_peaks = [np.asarray(p, dtype=int).copy() for p in peak_lists]
    remaining_widths = [np.asarray(w, dtype=float).copy() for w in width_lists]
    beats: list[int] = []
    num_agree: list[int] = []

    for i in range(len(remaining_peaks)):
        ref = remaining_peaks[i]
        ref_widths = remaining_widths[i]
        for k, peak in enumerate(ref.tolist()):
            group = [peak]
            count = 1
            width = max(int(round(ref_widths[k])), 1) if k < ref_widths.size else 1
            for j in range(i + 1, len(remaining_peaks)):
                test = remaining_peaks[j]
                test_widths = remaining_widths[j]
                if test.size == 0:
                    continue
                matches = np.flatnonzero(np.abs(test - peak) <= width)
                if matches.size == 0:
                    continue
                group.extend(test[matches].tolist())
                count += 1
                remaining_peaks[j] = np.delete(test, matches)
                if test_widths.size:
                    remaining_widths[j] = np.delete(test_widths, matches)
            beats.append(int(round(float(np.mean(group)))))
            num_agree.append(count)

    if not beats:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(np.asarray(beats))
    return np.asarray(beats, dtype=int)[order], np.asarray(num_agree, dtype=int)[order]


def _build_point_events(indices: np.ndarray, label: str, sample_rate_hz: float) -> list[Event]:
    return [
        Event(
            start=int(idx),
            end=int(idx),
            label=label,
            kind="point",
            metadata={"start_sec": float(idx / sample_rate_hz), "sample_rate_hz": sample_rate_hz},
        )
        for idx in indices.tolist()
    ]


def _nanmedian_across_last_axes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 and b.size == 0:
        return np.array([], dtype=float)
    stacked = np.concatenate([a[:, :, None], b[:, :, None]], axis=2)
    out = np.full(stacked.shape[0], np.nan, dtype=float)
    for i in range(stacked.shape[0]):
        vals = stacked[i].ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = float(np.median(vals))
    return out


def _nanmedian_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan, dtype=float)
    for i in range(a.shape[0]):
        vals = np.array([a[i], b[i]], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = float(np.median(vals))
    return out


def _nanmean_or_inf(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.mean(finite))


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))
