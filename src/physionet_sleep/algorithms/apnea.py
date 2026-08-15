from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.ndimage import binary_closing
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, hilbert, peak_widths, spectrogram

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, Event, SignalRecord, SignalSeries
from physionet_sleep.utils.overlap import has_interval_overlap
from physionet_sleep.utils.runs import count_true_runs, expand_runs
from physionet_sleep.utils.signal import bandpass_filter, rolling_mean, rolling_median


class EffortSnoreFeatureGenerator(BaseAlgorithm):
    def __init__(
        self,
        *,
        snore_key: str = "snore",
        effort_keys: tuple[str, ...] = ("effort_sum", "effort_thoracic", "effort_abdominal"),
    ) -> None:
        super().__init__(
            name="effort_snore_features",
            references=("external/st_vincent_analysis/apnea/algo_sleep_effort_events.m",),
        )
        self.snore_key = snore_key
        self.effort_keys = effort_keys

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def validate(self, record: SignalRecord) -> None:
        if self.snore_key not in record.channels and not any(key in record.channels for key in self.effort_keys):
            raise ValueError(f"{self.name} requires snore and/or effort channels")

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        signals: dict[str, SignalSeries] = {}
        events: dict[str, list[Event]] = {}
        features: dict[str, object] = {}

        if self.snore_key in record.channels:
            snore_series = record.get(self.snore_key)
            snore = np.asarray(snore_series.values, dtype=float).squeeze()
            snore_fs = float(snore_series.sample_rate)
            snore_band = _safe_bandpass(snore, snore_fs, 40.0, 100.0)
            snore_pow = 10.0 * np.log10(np.maximum(np.abs(hilbert(snore_band)) ** 2, np.finfo(float).tiny))
            snore_pow = _fill_nan_linear(snore_pow)
            snore_pow = snore_pow - rolling_median(snore_pow, max(int(round(snore_fs * 20.0)), 1))
            snore_mask = snore_pow > 10.0
            snore_starts, snore_lengths = count_true_runs(snore_mask)
            signals["snore_power"] = SignalSeries(snore_pow, sample_rate=snore_fs)
            signals["snore_mask"] = SignalSeries(snore_mask.astype(float), sample_rate=snore_fs, units="binary")
            events["snore"] = _build_period_events(snore_starts, snore_lengths, "snore_event", snore_fs)
            features["num_snore_events"] = len(events["snore"])
            features["snore_key"] = self.snore_key

        preferred_effort_key = next((key for key in self.effort_keys if key in record.channels), None)
        for effort_key in self.effort_keys:
            if effort_key not in record.channels:
                continue
            effort_series = record.get(effort_key)
            effort = np.asarray(effort_series.values, dtype=float).squeeze()
            effort_fs = float(effort_series.sample_rate)
            effort_power = np.abs(hilbert(effort))
            effort_power = effort_power - rolling_median(effort_power, max(int(round(effort_fs * 20.0)), 1))
            effort_mask = effort_power > 100.0
            starts, lengths = count_true_runs(effort_mask)
            keep = lengths > 15
            starts = starts[keep]
            lengths = lengths[keep]
            effort_mask = expand_runs(starts, lengths, effort.size)
            suffix = effort_key.removeprefix("effort_")
            signals[f"{suffix}_effort_amplitude"] = SignalSeries(effort_power, sample_rate=effort_fs)
            signals[f"{suffix}_effort_mask"] = SignalSeries(
                effort_mask.astype(float),
                sample_rate=effort_fs,
                units="binary",
            )
            events[f"{suffix}_effort"] = _build_period_events(starts, lengths, f"{suffix}_effort_event", effort_fs)
            features[f"num_{suffix}_effort_events"] = len(events[f"{suffix}_effort"])

            if effort_key == preferred_effort_key:
                signals["effort_power"] = SignalSeries(effort_power, sample_rate=effort_fs)
                signals["effort_mask"] = SignalSeries(
                    effort_mask.astype(float),
                    sample_rate=effort_fs,
                    units="binary",
                )
                events["effort"] = list(events[f"{suffix}_effort"])
                features["num_effort_events"] = len(events["effort"])
                features["effort_key"] = effort_key

        return AlgorithmResult(
            name=self.name,
            signals=signals,
            events=events,
            features=features,
            metadata={"status": "ported_feature_extraction"},
        )


class AirflowMorphologyFeatureGenerator(BaseAlgorithm):
    def __init__(
        self,
        *,
        signal_keys: tuple[str, ...] = ("airflow_thermal", "airflow_pressure", "flow"),
        band_hz: tuple[float, float] = (0.1, 0.5),
        smooth_seconds: float = 6.0,
        baseline_seconds: float = 30.0,
    ) -> None:
        super().__init__(name="airflow_morphology")
        self.signal_keys = signal_keys
        self.band_hz = band_hz
        self.smooth_seconds = smooth_seconds
        self.baseline_seconds = baseline_seconds

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
        selected_key = None
        respiration_result = prior.get("respiration")
        if respiration_result is not None:
            candidate = respiration_result.features.get("selected_signal_key")
            if isinstance(candidate, str) and candidate in record.channels:
                selected_key = candidate
        if selected_key is None:
            selected_key = next(key for key in self.signal_keys if key in record.channels)

        series = record.get(selected_key)
        airflow = np.asarray(series.values, dtype=float).squeeze()
        fs = float(series.sample_rate)

        filtered = bandpass_filter(airflow, fs, self.band_hz[0], self.band_hz[1], order=3)
        band_log_power = np.log10(np.maximum(filtered**2, np.finfo(float).tiny))
        smooth = rolling_mean(band_log_power, max(int(round(fs * self.smooth_seconds)), 1))
        baseline = rolling_median(smooth, max(int(round(fs * self.baseline_seconds)), 1))
        relative_power = smooth - baseline

        return AlgorithmResult(
            name=self.name,
            signals={
                "resp_band_log_power": SignalSeries(smooth, sample_rate=fs),
                "resp_band_relative_power": SignalSeries(relative_power, sample_rate=fs),
            },
            features={
                "selected_signal_key": selected_key,
                "sample_rate_hz": fs,
                "band_hz": self.band_hz,
                "smooth_seconds": float(self.smooth_seconds),
                "baseline_seconds": float(self.baseline_seconds),
                "mean_relative_power": float(np.nanmean(relative_power)),
            },
            metadata={"status": "ported_feature_extraction"},
        )


class ParadoxicalBreathingFeatureGenerator(BaseAlgorithm):
    def __init__(
        self,
        *,
        thoracic_key: str = "effort_thoracic",
        abdominal_key: str = "effort_abdominal",
        band_hz: tuple[float, float] = (0.1, 0.5),
        window_seconds: float = 30.0,
    ) -> None:
        super().__init__(name="paradoxical_breathing")
        self.thoracic_key = thoracic_key
        self.abdominal_key = abdominal_key
        self.band_hz = band_hz
        self.window_seconds = window_seconds

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        if self.thoracic_key not in record.channels or self.abdominal_key not in record.channels:
            return AlgorithmResult(
                name=self.name,
                metadata={"status": "skipped", "reason": "missing thoracic or abdominal effort"},
            )

        thoracic_series = record.get(self.thoracic_key)
        abdominal_series = record.get(self.abdominal_key)
        thoracic = np.asarray(thoracic_series.values, dtype=float).squeeze()
        abdominal = np.asarray(abdominal_series.values, dtype=float).squeeze()
        thoracic_fs = float(thoracic_series.sample_rate)
        abdominal_fs = float(abdominal_series.sample_rate)

        if thoracic_fs != abdominal_fs or thoracic.size != abdominal.size:
            target_len = min(thoracic.size, abdominal.size)
            target_fs = min(thoracic_fs, abdominal_fs)
            thoracic = _resample_numeric(thoracic, source_fs=thoracic_fs, target_fs=target_fs, target_length=target_len)
            abdominal = _resample_numeric(abdominal, source_fs=abdominal_fs, target_fs=target_fs, target_length=target_len)
            fs = target_fs
        else:
            fs = thoracic_fs

        thoracic_f = bandpass_filter(thoracic, fs, self.band_hz[0], self.band_hz[1], order=3)
        abdominal_f = bandpass_filter(abdominal, fs, self.band_hz[0], self.band_hz[1], order=3)
        corr = _rolling_corr(thoracic_f, abdominal_f, max(int(round(fs * self.window_seconds)), 2))
        paradox_mask = corr < 0.0

        return AlgorithmResult(
            name=self.name,
            signals={
                "thor_abd_corr": SignalSeries(corr, sample_rate=fs),
                "paradox_mask": SignalSeries(paradox_mask.astype(float), sample_rate=fs, units="binary"),
            },
            features={
                "thoracic_key": self.thoracic_key,
                "abdominal_key": self.abdominal_key,
                "sample_rate_hz": fs,
                "window_seconds": float(self.window_seconds),
                "mean_corr": float(np.nanmean(corr)),
                "paradox_fraction": float(np.nanmean(paradox_mask.astype(float))),
            },
            metadata={"status": "ported_feature_extraction"},
        )


class ApneaEventDetector(BaseAlgorithm):
    def __init__(
        self,
        *,
        thermal_key: str = "airflow_thermal",
        pressure_key: str = "airflow_pressure",
        respiration_result_key: str = "respiration",
        spo2_result_key: str = "spo2_drop",
    ) -> None:
        super().__init__(
            name="apnea_events",
            references=(
                "external/st_vincent_analysis/apnea/detectApneaFromPeakDrop.m",
                "external/st_vincent_analysis/apnea/algo_sleep_apnea_events.m",
            ),
        )
        self.thermal_key = thermal_key
        self.pressure_key = pressure_key
        self.respiration_result_key = respiration_result_key
        self.spo2_result_key = spo2_result_key

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def validate(self, record: SignalRecord) -> None:
        if self.thermal_key not in record.channels and self.pressure_key not in record.channels:
            raise ValueError(f"{self.name} requires at least one of: {self.thermal_key}, {self.pressure_key}")

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        thermal_series = record.channels.get(self.thermal_key)
        pressure_series = record.channels.get(self.pressure_key)
        if thermal_series is None and pressure_series is None:
            raise ValueError("No airflow channels available for apnea event detection")

        thermal = np.asarray(thermal_series.values, dtype=float).squeeze() if thermal_series is not None else None
        pressure = np.asarray(pressure_series.values, dtype=float).squeeze() if pressure_series is not None else None
        thermal_fs = float(thermal_series.sample_rate) if thermal_series is not None else None
        pressure_fs = float(pressure_series.sample_rate) if pressure_series is not None else None

        thermal_env = np.abs(hilbert(thermal)) if thermal is not None else np.array([], dtype=float)
        pressure_env = np.abs(hilbert(pressure)) if pressure is not None else np.array([], dtype=float)

        thermal_peaks, thermal_troughs = _extract_resp_extrema(thermal, thermal_fs, cutoff=0.0) if thermal is not None and thermal_fs is not None else (np.array([], dtype=int), np.array([], dtype=int))
        pressure_peaks, pressure_troughs = _extract_resp_extrema(pressure, pressure_fs, cutoff=0.0) if pressure is not None and pressure_fs is not None else (np.array([], dtype=int), np.array([], dtype=int))

        thermal_bad_starts, thermal_bad_lengths = _airflow_quality_bad_runs(thermal, thermal_fs) if thermal is not None and thermal_fs is not None else (np.array([], dtype=int), np.array([], dtype=int))
        pressure_bad_starts, pressure_bad_lengths = _airflow_quality_bad_runs(pressure, pressure_fs) if pressure is not None and pressure_fs is not None else (np.array([], dtype=int), np.array([], dtype=int))

        if thermal is not None and thermal_fs is not None and thermal_peaks.size:
            apnea_mask, apnea_baseline = _detect_peak_drop_events(
                thermal_env,
                thermal_fs,
                thermal_peaks,
                threshold_ratio=0.20,
                hyp_flag=False,
                anchor_troughs=thermal_troughs,
            )
            apn_starts, apn_lengths = count_true_runs(apnea_mask)
        else:
            apnea_mask = np.zeros(0, dtype=bool)
            apnea_baseline = np.array([], dtype=float)
            apn_starts = np.array([], dtype=int)
            apn_lengths = np.array([], dtype=int)

        if pressure is not None and pressure_fs is not None and pressure_peaks.size:
            hyp_mask_pressure, hyp_baseline_pressure = _detect_peak_drop_events(
                pressure_env,
                pressure_fs,
                pressure_peaks,
                threshold_ratio=0.70,
                hyp_flag=True,
                anchor_troughs=pressure_troughs,
            )
            hyp_starts, hyp_lengths = count_true_runs(hyp_mask_pressure)
        else:
            hyp_mask_pressure = np.zeros(0, dtype=bool)
            hyp_baseline_pressure = np.array([], dtype=float)
            hyp_starts = np.array([], dtype=int)
            hyp_lengths = np.array([], dtype=int)

        hyp_mask_thermal = np.zeros_like(hyp_mask_pressure, dtype=bool)
        hyp_baseline_thermal = np.array([], dtype=float)
        if thermal is not None and thermal_fs is not None and thermal_peaks.size:
            hyp_mask_thermal_native, hyp_baseline_thermal = _detect_peak_drop_events(
                thermal_env,
                thermal_fs,
                thermal_peaks,
                threshold_ratio=0.70,
                hyp_flag=True,
                anchor_troughs=thermal_troughs,
            )
            if pressure is not None and pressure_fs is not None and hyp_mask_pressure.size:
                hyp_mask_thermal = _resample_bool_mask(
                    hyp_mask_thermal_native,
                    source_fs=thermal_fs,
                    target_fs=pressure_fs,
                    target_length=pressure.size,
                )

        hyp_mask = hyp_mask_pressure | hyp_mask_thermal
        hyp_starts, hyp_lengths = count_true_runs(hyp_mask)

        if apn_starts.size and hyp_starts.size and pressure is not None:
            overlap = has_interval_overlap(hyp_starts, hyp_lengths, apn_starts * int(round(pressure_fs / thermal_fs)), apn_lengths * int(round(pressure_fs / thermal_fs)))
            hyp_starts = hyp_starts[~overlap]
            hyp_lengths = hyp_lengths[~overlap]

        if thermal_bad_starts.size and apn_starts.size:
            keep = ~has_interval_overlap(apn_starts, apn_lengths, thermal_bad_starts, thermal_bad_lengths)
            apn_starts = apn_starts[keep]
            apn_lengths = apn_lengths[keep]
        if pressure_bad_starts.size and hyp_starts.size:
            keep = ~has_interval_overlap(hyp_starts, hyp_lengths, pressure_bad_starts, pressure_bad_lengths)
            hyp_starts = hyp_starts[keep]
            hyp_lengths = hyp_lengths[keep]

        spo2_bad = _event_seconds(prior.get(self.spo2_result_key), {"bad"})
        spo2_drop = _event_seconds(prior.get(self.spo2_result_key), {"drop_3", "drop_4"})

        if spo2_bad and apn_starts.size and thermal_fs is not None:
            apn_events_sec = _starts_lengths_to_windows(apn_starts, apn_lengths, thermal_fs)
            keep = np.array([not _window_overlaps(window, spo2_bad) for window in apn_events_sec], dtype=bool)
            apn_starts = apn_starts[keep]
            apn_lengths = apn_lengths[keep]
        if spo2_bad and hyp_starts.size and pressure_fs is not None:
            hyp_events_sec = _starts_lengths_to_windows(hyp_starts, hyp_lengths, pressure_fs)
            keep = np.array([not _window_overlaps(window, spo2_bad) for window in hyp_events_sec], dtype=bool)
            hyp_starts = hyp_starts[keep]
            hyp_lengths = hyp_lengths[keep]

        if spo2_drop and apn_starts.size and thermal_fs is not None:
            apn_events_sec = _starts_lengths_to_windows(apn_starts, apn_lengths, thermal_fs, extend_seconds=30.0)
            keep = np.array([_window_overlaps(window, spo2_drop) for window in apn_events_sec], dtype=bool)
            apn_starts = apn_starts[keep]
            apn_lengths = apn_lengths[keep]
        if spo2_drop and hyp_starts.size and pressure_fs is not None:
            hyp_events_sec = _starts_lengths_to_windows(hyp_starts, hyp_lengths, pressure_fs, extend_seconds=30.0)
            keep = np.array([_window_overlaps(window, spo2_drop) for window in hyp_events_sec], dtype=bool)
            hyp_starts = hyp_starts[keep]
            hyp_lengths = hyp_lengths[keep]

        if thermal_fs is not None and apn_starts.size:
            keep = apn_lengths > int(round(8.0 * thermal_fs))
            apn_starts = apn_starts[keep]
            apn_lengths = apn_lengths[keep]
            apn_mask = expand_runs(apn_starts, apn_lengths, thermal.size)
            apn_mask = binary_closing(apn_mask, structure=np.r_[1, np.zeros(int(round(8.0 * thermal_fs))), 1].astype(bool))
            apn_starts, apn_lengths = count_true_runs(apn_mask)
        if pressure_fs is not None and hyp_starts.size:
            keep = hyp_lengths > int(round(8.0 * pressure_fs))
            hyp_starts = hyp_starts[keep]
            hyp_lengths = hyp_lengths[keep]
            hyp_mask = expand_runs(hyp_starts, hyp_lengths, pressure.size)
            hyp_mask = binary_closing(hyp_mask, structure=np.r_[1, np.zeros(int(round(8.0 * pressure_fs))), 1].astype(bool))
            hyp_starts, hyp_lengths = count_true_runs(hyp_mask)

        thermal_len = thermal.size if thermal is not None else 0
        thermal_rate = thermal_fs if thermal_fs is not None else 1.0
        pressure_rate = pressure_fs if pressure_fs is not None else thermal_rate

        apnea_mask = expand_runs(apn_starts, apn_lengths, thermal_len) if thermal_len else np.zeros(0, dtype=bool)
        hyp_mask_thermal_grid = (
            _resample_bool_mask(expand_runs(hyp_starts, hyp_lengths, pressure.size), source_fs=pressure_rate, target_fs=thermal_rate, target_length=thermal_len)
            if thermal_len and pressure is not None and hyp_starts.size
            else np.zeros(thermal_len, dtype=bool)
        )
        event_vector = np.maximum(hyp_mask_thermal_grid.astype(int), apnea_mask.astype(int) * 2)

        apnea_events = _build_period_events(apn_starts, apn_lengths, "apnea", thermal_rate)
        hypopnea_events = _build_period_events(hyp_starts, hyp_lengths, "hypopnea", pressure_rate)
        thermal_bad_events = _build_period_events(thermal_bad_starts, thermal_bad_lengths, "poor_thermal_airflow", thermal_rate)
        pressure_bad_events = _build_period_events(pressure_bad_starts, pressure_bad_lengths, "poor_pressure_airflow", pressure_rate)

        return AlgorithmResult(
            name=self.name,
            signals={
                "apnea_event_vector": SignalSeries(event_vector, sample_rate=thermal_rate, units="class"),
                "thermal_envelope": SignalSeries(thermal_env if thermal_env.size else np.zeros(thermal_len), sample_rate=thermal_rate),
                "thermal_baseline": SignalSeries(apnea_baseline if apnea_baseline.size else np.zeros(thermal_len), sample_rate=thermal_rate),
                "pressure_envelope": SignalSeries(
                    pressure_env if pressure_env.size else np.zeros(thermal_len),
                    sample_rate=pressure_rate,
                ),
                "pressure_baseline": SignalSeries(
                    hyp_baseline_pressure if hyp_baseline_pressure.size else np.zeros(pressure.size if pressure is not None else 0),
                    sample_rate=pressure_rate,
                ),
                "thermal_hypopnea_baseline": SignalSeries(
                    hyp_baseline_thermal if hyp_baseline_thermal.size else np.zeros(thermal_len),
                    sample_rate=thermal_rate,
                ),
            },
            events={
                "apnea": apnea_events,
                "hypopnea": hypopnea_events,
                "poor_thermal_airflow": thermal_bad_events,
                "poor_pressure_airflow": pressure_bad_events,
            },
            features={
                "thermal_signal_key": self.thermal_key,
                "pressure_signal_key": self.pressure_key,
                "respiration_dependency": self.respiration_result_key,
                "spo2_dependency": self.spo2_result_key,
                "num_apnea_events": len(apnea_events),
                "num_hypopnea_events": len(hypopnea_events),
                "num_poor_thermal_segments": len(thermal_bad_events),
                "num_poor_pressure_segments": len(pressure_bad_events),
                "thermal_peak_count": int(thermal_peaks.size),
                "pressure_peak_count": int(pressure_peaks.size),
            },
            metadata={"status": "ported_feature_extraction", "depends_on": [self.respiration_result_key, self.spo2_result_key]},
        )


def _extract_resp_extrema(signal: np.ndarray, sample_rate_hz: float, *, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    filtered = bandpass_filter(signal, sample_rate_hz, 0.1, 0.5, order=3)
    analytic = hilbert(filtered)
    phase_cos = np.cos(np.angle(analytic))
    pos_idx, pos_width, pos_prom = _phase_extrema(phase_cos, positive=True)
    neg_idx, neg_width, neg_prom = _phase_extrema(phase_cos, positive=False)

    pos_keep = pos_prom >= 1.9
    neg_keep = neg_prom >= 1.9
    if cutoff > 0:
        pos_keep &= signal[np.clip(pos_idx, 0, signal.size - 1)] >= cutoff
        neg_keep &= signal[np.clip(neg_idx, 0, signal.size - 1)] <= -cutoff

    pos_idx, pos_width = pos_idx[pos_keep], pos_width[pos_keep]
    neg_idx, neg_width = neg_idx[neg_keep], neg_width[neg_keep]
    pos_idx = _shift_local_extrema(signal, pos_idx, pos_width, positive=True)
    neg_idx = _shift_local_extrema(signal, neg_idx, neg_width, positive=False)
    pos_idx = np.unique(pos_idx.astype(int))
    neg_idx = np.unique(neg_idx.astype(int))
    return pos_idx, neg_idx


def _phase_extrema(phase_cos: np.ndarray, *, positive: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = phase_cos if positive else -phase_cos
    idx, _ = find_peaks(work)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    widths, prominences = peak_widths(work, idx, rel_height=0.5)[0], peak_widths(work, idx, rel_height=1.0)[1]
    return idx.astype(int), widths.astype(float), np.asarray(prominences, dtype=float)


def _shift_local_extrema(signal: np.ndarray, idx: np.ndarray, widths: np.ndarray, *, positive: bool) -> np.ndarray:
    out = idx.astype(int).copy()
    for i, peak in enumerate(out):
        half_width = max(int(round(widths[i] / 2.0)), 1)
        start = max(peak - half_width, 0)
        end = min(peak + half_width + 1, signal.size)
        out[i] = start + int(np.argmax(signal[start:end]) if positive else np.argmin(signal[start:end]))
    return out


def _airflow_quality_bad_runs(signal: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if signal.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    win = max(int(round(30.0 * sample_rate_hz)), 16)
    overlap = min(int(round(win * 0.95)), win - 1)
    freqs, times, spec = spectrogram(signal, fs=sample_rate_hz, nperseg=win, noverlap=overlap, scaling="density", mode="magnitude")
    if spec.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    dominant_idx = np.argmax(spec, axis=0)
    dominant_freq = freqs[dominant_idx]
    smooth_bins = max(int(round(120.0 / max(np.median(np.diff(times)) if times.size > 1 else 1.0, 1e-6))), 1)
    dominant_freq = rolling_median(dominant_freq, smooth_bins)
    bad_lowres = dominant_freq > 0.6
    bad_full = _interp_bool_track(times, bad_lowres, signal.size, sample_rate_hz)
    closing = max(int(round(5.0 * 60.0 * sample_rate_hz)), 1)
    bad_full = binary_closing(bad_full, structure=np.r_[1, np.zeros(closing), 1].astype(bool))
    starts, lengths = count_true_runs(bad_full)
    keep = lengths > int(round(2.0 * 60.0 * sample_rate_hz))
    starts = starts[keep]
    lengths = lengths[keep]
    return starts, lengths


def _detect_peak_drop_events(
    envelope: np.ndarray,
    sample_rate_hz: float,
    peak_idx: np.ndarray,
    *,
    threshold_ratio: float,
    hyp_flag: bool,
    anchor_troughs: np.ndarray,
    window_seconds: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    if peak_idx.size == 0:
        return np.zeros(envelope.size, dtype=bool), np.zeros(envelope.size, dtype=float)

    peak_amp = envelope[np.clip(peak_idx, 0, envelope.size - 1)]
    peak_t = peak_idx.astype(float) / sample_rate_hz
    is_drop, baseline_used = _detect_apnea_from_peak_drop(
        peak_amp,
        peak_t,
        threshold_ratio=threshold_ratio,
        hyp_flag=hyp_flag,
        window_seconds=window_seconds,
    )
    baseline_interp = _interp_linear(peak_idx.astype(float), baseline_used, envelope.size)
    rolling_env = rolling_median(envelope, max(int(round(sample_rate_hz)), 1))
    mask = rolling_env < (baseline_interp * threshold_ratio)
    starts, lengths = count_true_runs(mask)
    if starts.size and anchor_troughs.size:
        starts, lengths = _anchor_runs_to_troughs(starts, lengths, peak_idx, anchor_troughs)
        mask = expand_runs(starts, lengths, envelope.size)
    mask = binary_closing(mask, structure=np.ones(max(int(round(sample_rate_hz / 2.0)), 1), dtype=bool))
    return mask, baseline_interp


def _detect_apnea_from_peak_drop(
    peak_amp: np.ndarray,
    peak_t: np.ndarray,
    *,
    threshold_ratio: float,
    hyp_flag: bool,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = peak_amp.size
    is_drop = np.zeros(n, dtype=bool)
    baseline = np.full(n, np.nan, dtype=float)

    all_vals: list[float] = []
    all_times: list[float] = []
    non_vals: list[float] = []
    non_times: list[float] = []
    in_drop = False

    for i in range(n):
        t_now = float(peak_t[i])
        a_now = float(peak_amp[i])
        t_cut = t_now - window_seconds

        while all_times and all_times[0] < t_cut:
            all_times.pop(0)
            all_vals.pop(0)
        while non_times and non_times[0] < t_cut:
            non_times.pop(0)
            non_vals.pop(0)

        work_vals = non_vals if non_vals else all_vals
        if work_vals:
            base = float(np.mean(work_vals))
            baseline[i] = base
            if not in_drop:
                if a_now < threshold_ratio * base:
                    in_drop = True
                    is_drop[i] = True
            else:
                if a_now >= threshold_ratio * base:
                    in_drop = False
                    is_drop[i] = False
                else:
                    is_drop[i] = True

        all_times.append(t_now)
        all_vals.append(a_now)
        if (not is_drop[i]) or hyp_flag:
            non_times.append(t_now)
            non_vals.append(a_now)

    return is_drop, baseline


def _anchor_runs_to_troughs(
    starts: np.ndarray,
    lengths: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchored_starts = []
    anchored_lengths = []
    for start, length in zip(starts, lengths, strict=False):
        prev_peak = peaks[peaks <= start]
        prev_trough = troughs[troughs <= start]
        if prev_trough.size:
            trough = int(prev_trough[-1])
            if prev_peak.size and trough < int(prev_peak[-1]):
                trough = int(start)
        else:
            trough = int(start)
        anchored_starts.append(trough)
        anchored_lengths.append(int(length + (start - trough)))
    return np.asarray(anchored_starts, dtype=int), np.asarray(anchored_lengths, dtype=int)


def _interp_linear(x: np.ndarray, y: np.ndarray, output_length: int) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return np.zeros(output_length, dtype=float)
    f = interp1d(x[valid], y[valid], kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    return np.asarray(f(np.arange(output_length)), dtype=float)


def _interp_bool_track(times: np.ndarray, values: np.ndarray, output_length: int, sample_rate_hz: float) -> np.ndarray:
    if times.size == 0:
        return np.zeros(output_length, dtype=bool)
    idx = np.clip(np.round(times * sample_rate_hz).astype(int), 0, output_length - 1)
    valid = np.zeros(output_length, dtype=bool)
    valid[idx] = values.astype(bool)
    return valid


def _resample_bool_mask(mask: np.ndarray, *, source_fs: float, target_fs: float, target_length: int) -> np.ndarray:
    if mask.size == 0 or source_fs <= 0 or target_fs <= 0 or target_length <= 0:
        return np.zeros(target_length, dtype=bool)
    source_t = np.arange(mask.size, dtype=float) / source_fs
    target_t = np.arange(target_length, dtype=float) / target_fs
    f = interp1d(source_t, mask.astype(float), kind="nearest", bounds_error=False, fill_value=0.0, assume_sorted=True)
    return np.asarray(f(target_t) > 0.5, dtype=bool)


def _starts_lengths_to_windows(starts: np.ndarray, lengths: np.ndarray, sample_rate_hz: float, *, extend_seconds: float = 0.0) -> list[tuple[float, float]]:
    return [
        (float(start / sample_rate_hz), float((start + length - 1) / sample_rate_hz + extend_seconds))
        for start, length in zip(starts, lengths, strict=False)
    ]


def _window_overlaps(window: tuple[float, float], others: list[tuple[float, float]]) -> bool:
    start, end = window
    return any(start <= other_end and end >= other_start for other_start, other_end in others)


def _event_seconds(result: AlgorithmResult | None, keys: set[str]) -> list[tuple[float, float]]:
    if result is None:
        return []
    windows: list[tuple[float, float]] = []
    for key, events in result.events.items():
        if key not in keys:
            continue
        for event in events:
            start_sec = float(event.metadata.get("start_sec", 0.0))
            end_sec = float(event.metadata.get("end_sec", start_sec))
            windows.append((start_sec, end_sec))
    return windows


def _build_period_events(
    starts: np.ndarray,
    lengths: np.ndarray,
    label: str,
    sample_rate_hz: float,
) -> list[Event]:
    return [
        Event(
            start=int(start),
            end=int(start + length - 1),
            label=label,
            kind="period",
            metadata={
                "sample_rate_hz": sample_rate_hz,
                "start_sec": float(start / sample_rate_hz),
                "end_sec": float((start + length - 1) / sample_rate_hz),
            },
        )
        for start, length in zip(starts, lengths, strict=False)
    ]


def _safe_bandpass(signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    try:
        return bandpass_filter(signal, sample_rate_hz, low_hz, high_hz, order=3)
    except Exception:
        return np.asarray(signal, dtype=float)


def _fill_nan_linear(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    idx = np.flatnonzero(np.isfinite(arr))
    if idx.size == 0:
        return np.zeros_like(arr)
    if idx.size == 1:
        arr[:] = arr[idx[0]]
        return arr
    missing = ~np.isfinite(arr)
    arr[missing] = np.interp(np.flatnonzero(missing), idx, arr[idx])
    return arr


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x = rolling_mean(x, window)
    mean_y = rolling_mean(y, window)
    mean_xy = rolling_mean(x * y, window)
    mean_x2 = rolling_mean(x * x, window)
    mean_y2 = rolling_mean(y * y, window)
    cov = mean_xy - (mean_x * mean_y)
    var_x = np.maximum(mean_x2 - (mean_x * mean_x), 0.0)
    var_y = np.maximum(mean_y2 - (mean_y * mean_y), 0.0)
    denom = np.sqrt(var_x * var_y)
    corr = np.full(x.shape, np.nan, dtype=float)
    good = denom > 1e-12
    corr[good] = cov[good] / denom[good]
    return np.clip(corr, -1.0, 1.0)


def _resample_numeric(values: np.ndarray, *, source_fs: float, target_fs: float, target_length: int) -> np.ndarray:
    if values.size == 0 or source_fs <= 0 or target_fs <= 0 or target_length <= 0:
        return np.zeros(max(target_length, 0), dtype=float)
    source_t = np.arange(values.size, dtype=float) / source_fs
    target_t = np.arange(target_length, dtype=float) / target_fs
    f = interp1d(source_t, values.astype(float), kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    return np.asarray(f(target_t), dtype=float)
