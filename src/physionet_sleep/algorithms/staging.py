from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.ndimage import binary_closing
from scipy.signal import hilbert, spectrogram

from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, Event, SignalRecord, SignalSeries
from physionet_sleep.utils.runs import count_true_runs, expand_runs
from physionet_sleep.utils.signal import bandpass_filter, rolling_mean, rolling_median


class SpindleFeatureGenerator(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "eeg") -> None:
        super().__init__(
            name="spindle_features",
            references=("external/st_vincent_analysis/staging/algo_sleep_splindex.m",),
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
        series = record.get(self.signal_key)
        eeg = _as_2d_signal(series.values)
        fs = float(series.sample_rate)

        bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 11.0), (11.0, 16.0), (16.0, 30.0), (30.0, 100.0)]
        band_names = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]
        per_channel_power = []
        per_channel_residual = []
        exponents = []
        intercepts = []
        time_sec_ref = None

        for ch in range(eeg.shape[1]):
            freqs, time_sec, ps = _spectrogram_abs(eeg[:, ch], fs, window_seconds=2.0, overlap_fraction=0.9)
            if time_sec_ref is None:
                time_sec_ref = time_sec
            residual_logp, exponent, intercept = _remove_one_over_f(ps, freqs)
            band_power = np.column_stack([_band_median(ps, freqs, lo, hi) for lo, hi in bands])
            band_residual = np.column_stack([_band_median(residual_logp, freqs, lo, hi) for lo, hi in bands])
            per_channel_power.append(band_power)
            per_channel_residual.append(band_residual)
            exponents.append(exponent)
            intercepts.append(intercept)

        if time_sec_ref is None:
            raise ValueError("No spindle spectrogram windows could be computed")

        power = np.stack(per_channel_power, axis=2)
        residual = np.stack(per_channel_residual, axis=2)
        exponent = np.nanmean(np.stack(exponents, axis=1), axis=1)
        intercept = np.nanmean(np.stack(intercepts, axis=1), axis=1)
        sigma_beta_ratio = np.nansum(power[:, 3, :] / np.maximum(power[:, 4, :], 1e-12), axis=1)
        sigma_beta_residual_diff = np.nansum(residual[:, 3, :] - residual[:, 4, :], axis=1)
        splindex = sigma_beta_ratio * sigma_beta_residual_diff
        out_fs = _time_series_rate(time_sec_ref)

        signals = {
            "splindex": SignalSeries(splindex, sample_rate=out_fs),
            "sigma_beta_ratio": SignalSeries(sigma_beta_ratio, sample_rate=out_fs),
            "sigma_beta_residual_diff": SignalSeries(sigma_beta_residual_diff, sample_rate=out_fs),
            "aperiodic_exponent": SignalSeries(exponent, sample_rate=out_fs),
            "aperiodic_intercept": SignalSeries(intercept, sample_rate=out_fs),
        }
        for i, band_name in enumerate(band_names):
            signals[f"{band_name}_power"] = SignalSeries(np.nanmean(power[:, i, :], axis=1), sample_rate=out_fs)
            signals[f"{band_name}_residual_power"] = SignalSeries(
                np.nanmean(residual[:, i, :], axis=1),
                sample_rate=out_fs,
            )

        return AlgorithmResult(
            name=self.name,
            signals=signals,
            features={
                "signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "n_channels": int(eeg.shape[1]),
                "band_names": band_names,
                "window_seconds": 2.0,
                "overlap_fraction": 0.9,
                "mean_splindex": float(np.nanmean(splindex)),
            },
            metadata={"status": "ported_feature_extraction"},
        )


class SlowWaveFeatureGenerator(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "eeg") -> None:
        super().__init__(
            name="slow_wave_features",
            references=("external/st_vincent_analysis/staging/algo_sleep_swindex.m",),
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
        series = record.get(self.signal_key)
        eeg = _as_2d_signal(series.values)
        fs = float(series.sample_rate)

        low_env = []
        high_env = []
        raw_pow = []
        ratio_pow = []
        for ch in range(eeg.shape[1]):
            slow = _band_envelope(eeg[:, ch], fs, 0.3, 1.5)
            high = _band_envelope(eeg[:, ch], fs, 30.0, 100.0)
            raw = slow - rolling_median(slow, max(int(round(fs * 30.0)), 1))
            ratio = (raw / np.maximum(high, 1e-12)) * (raw - high)
            low_env.append(slow)
            high_env.append(high)
            raw_pow.append(raw)
            ratio_pow.append(ratio)

        low_env_mean = np.nanmean(np.column_stack(low_env), axis=1)
        high_env_mean = np.nanmean(np.column_stack(high_env), axis=1)
        raw_pow_mean = np.nanmean(np.column_stack(raw_pow), axis=1)
        ratio_pow_mean = np.nanmean(np.column_stack(ratio_pow), axis=1)

        return AlgorithmResult(
            name=self.name,
            signals={
                "slow_wave_envelope": SignalSeries(low_env_mean, sample_rate=fs),
                "high_frequency_envelope": SignalSeries(high_env_mean, sample_rate=fs),
                "slow_wave_raw_power": SignalSeries(raw_pow_mean, sample_rate=fs),
                "slow_wave_ratio_power": SignalSeries(ratio_pow_mean, sample_rate=fs),
                "slow_wave_index": SignalSeries(ratio_pow_mean, sample_rate=fs),
            },
            features={
                "signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "n_channels": int(eeg.shape[1]),
                "mean_slow_wave_raw_power": float(np.nanmean(raw_pow_mean)),
                "mean_slow_wave_ratio_power": float(np.nanmean(ratio_pow_mean)),
            },
            metadata={"status": "ported_feature_extraction"},
        )


class EEGArousalFeatureGenerator(BaseAlgorithm):
    def __init__(self, *, eeg_key: str = "eeg", emg_key: str = "emg") -> None:
        super().__init__(
            name="eeg_arousal_features",
            references=("external/st_vincent_analysis/staging/algo_sleep_arousal.m",),
        )
        self.eeg_key = eeg_key
        self.emg_key = emg_key

    def required_signals(self) -> tuple[str, ...]:
        return (self.eeg_key,)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        series = record.get(self.eeg_key)
        eeg = _as_2d_signal(series.values)
        fs = float(series.sample_rate)

        bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 11.0), (11.0, 16.0), (16.0, 100.0)]
        channel_signals = []
        time_sec_ref = None

        for ch in range(eeg.shape[1]):
            freqs, time_sec, ps = _spectrogram_abs(eeg[:, ch], fs, window_seconds=2.0, overlap_fraction=0.9)
            if time_sec_ref is None:
                time_sec_ref = time_sec
            band_power = np.column_stack([_band_median(ps, freqs, lo, hi) for lo, hi in bands])
            arousal_signal = np.nanmean(band_power[:, [1, 2, 4]], axis=1)
            channel_signals.append(arousal_signal)

        if time_sec_ref is None:
            raise ValueError("No EEG arousal spectrogram windows could be computed")

        out_fs = _time_series_rate(time_sec_ref)
        eeg_signal = np.nanmean(np.column_stack(channel_signals), axis=1)
        eeg_signal = eeg_signal - rolling_median(eeg_signal, max(int(round(out_fs * 10.0)), 1))
        eeg_signal = rolling_mean(eeg_signal, max(int(round(out_fs * 3.0)), 1))
        mask = eeg_signal > 0.3
        mask = binary_closing(mask, structure=np.ones(5, dtype=bool))
        starts, lengths = count_true_runs(mask)
        keep = lengths >= max(int(round(out_fs * 3.0)), 1)
        starts = starts[keep]
        lengths = lengths[keep]
        arousal_mask = expand_runs(starts, lengths, eeg_signal.size)

        events = _build_timebin_events(starts, lengths, "eeg_arousal", out_fs)
        return AlgorithmResult(
            name=self.name,
            signals={
                "eeg_arousal_signal": SignalSeries(eeg_signal, sample_rate=out_fs),
                "eeg_arousal_mask": SignalSeries(arousal_mask.astype(float), sample_rate=out_fs, units="binary"),
            },
            events={"eeg_arousal": events},
            features={
                "eeg_key": self.eeg_key,
                "emg_key": self.emg_key,
                "sample_rate_hz": fs,
                "n_channels": int(eeg.shape[1]),
                "num_arousals": len(events),
            },
            metadata={"status": "ported_feature_extraction", "emg_used": self.emg_key in record.channels},
        )


class EyeMovementActivityGenerator(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "eog") -> None:
        super().__init__(name="eye_movement_activity")
        self.signal_key = signal_key

    def required_signals(self) -> tuple[str, ...]:
        return (self.signal_key,)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        series = record.get(self.signal_key)
        eog = np.asarray(series.values, dtype=float).squeeze()
        fs = float(series.sample_rate)

        eye_env = _band_envelope(eog, fs, 0.1, 5.0)
        eye_baseline = rolling_median(eye_env, max(int(round(fs * 30.0)), 1))
        eye_activity = np.maximum(eye_env - eye_baseline, 0.0)
        eye_activity = rolling_mean(eye_activity, max(int(round(fs * 1.0)), 1))

        return AlgorithmResult(
            name=self.name,
            signals={
                "eye_movement_activity": SignalSeries(eye_activity, sample_rate=fs),
                "eye_movement_baseline": SignalSeries(eye_baseline, sample_rate=fs),
            },
            features={
                "signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "mean_eye_movement_activity": float(np.nanmean(eye_activity)),
            },
            metadata={"status": "ported_feature_extraction"},
        )


class EMGToneFeatureGenerator(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "emg", suppression_ratio: float = 0.5) -> None:
        super().__init__(name="emg_tone_features")
        self.signal_key = signal_key
        self.suppression_ratio = suppression_ratio

    def required_signals(self) -> tuple[str, ...]:
        return (self.signal_key,)

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        series = record.get(self.signal_key)
        emg = np.asarray(series.values, dtype=float).squeeze()
        fs = float(series.sample_rate)

        tone = _band_envelope(emg, fs, 10.0, 100.0)
        tone = rolling_mean(tone, max(int(round(fs * 1.0)), 1))
        tone_baseline = rolling_median(tone, max(int(round(fs * 60.0)), 1))
        suppression_mask = tone < (np.maximum(tone_baseline, 1e-12) * self.suppression_ratio)
        suppression_mask = binary_closing(suppression_mask, structure=np.ones(max(int(round(fs * 2.0)), 1), dtype=bool))
        starts, lengths = count_true_runs(suppression_mask)
        keep = lengths >= max(int(round(fs * 3.0)), 1)
        starts = starts[keep]
        lengths = lengths[keep]
        suppression_mask = expand_runs(starts, lengths, tone.size)
        suppression_events = _build_timebin_events(starts, lengths, "emg_suppression", fs)

        return AlgorithmResult(
            name=self.name,
            signals={
                "emg_tone": SignalSeries(tone, sample_rate=fs),
                "emg_tone_baseline": SignalSeries(tone_baseline, sample_rate=fs),
                "emg_suppression_mask": SignalSeries(
                    suppression_mask.astype(float),
                    sample_rate=fs,
                    units="binary",
                ),
            },
            events={"emg_suppression": suppression_events},
            features={
                "signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "suppression_ratio": float(self.suppression_ratio),
                "mean_emg_tone": float(np.nanmean(tone)),
                "num_emg_suppression_events": len(suppression_events),
            },
            metadata={"status": "ported_feature_extraction"},
        )


class EOGRemFeatureGenerator(BaseAlgorithm):
    def __init__(self, *, signal_key: str = "eog") -> None:
        super().__init__(
            name="eog_rem_features",
            references=("external/st_vincent_analysis/staging/algo_sleep_eog.m",),
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
        series = record.get(self.signal_key)
        eog = np.asarray(series.values, dtype=float).squeeze()
        fs = float(series.sample_rate)

        freqs, time_sec, ps = _spectrogram_abs(eog, fs, window_seconds=30.0, overlap_fraction=0.99)
        rem_fast = _safe_log10(_band_mean(ps, freqs, 0.5, 5.0))
        sem_fast = _safe_log10(_band_mean(ps, freqs, 0.1, 0.5))
        noise_fast = _safe_log10(_band_mean(ps, freqs, 5.0, 30.0))
        fast_fs = _time_series_rate(time_sec)
        bins_per_epoch = max(int(round(30.0 * fast_fs)), 1)

        rem_epoch = _aggregate_blocks(rem_fast, bins_per_epoch, agg="mean")
        rem_var_epoch = _aggregate_blocks(rem_fast, bins_per_epoch, agg="var")
        sem_epoch = _aggregate_blocks(sem_fast, bins_per_epoch, agg="mean")
        noise_epoch = _aggregate_blocks(noise_fast, bins_per_epoch, agg="mean")

        return AlgorithmResult(
            name=self.name,
            signals={
                "rem_band_power_fast": SignalSeries(rem_fast, sample_rate=fast_fs),
                "sem_band_power_fast": SignalSeries(sem_fast, sample_rate=fast_fs),
                "noise_band_power_fast": SignalSeries(noise_fast, sample_rate=fast_fs),
                "rem_power_epoch": SignalSeries(rem_epoch, sample_rate=1.0 / 30.0),
                "rem_variance_epoch": SignalSeries(rem_var_epoch, sample_rate=1.0 / 30.0),
                "sem_power_epoch": SignalSeries(sem_epoch, sample_rate=1.0 / 30.0),
                "noise_power_epoch": SignalSeries(noise_epoch, sample_rate=1.0 / 30.0),
            },
            features={
                "signal_key": self.signal_key,
                "sample_rate_hz": fs,
                "mean_rem_power_epoch": float(np.nanmean(rem_epoch)),
                "mean_rem_variance_epoch": float(np.nanmean(rem_var_epoch)),
            },
            metadata={"status": "ported_feature_extraction"},
        )


def _as_2d_signal(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Expected 1D or 2D signal array")
    return arr




def _spectrogram_abs(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    window_seconds: float,
    overlap_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = max(int(round(sample_rate_hz * window_seconds)), 8)
    noverlap = min(int(round(nperseg * overlap_fraction)), nperseg - 1)
    freqs, time_sec, sp = spectrogram(
        np.asarray(values, dtype=float),
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        scaling="density",
        mode="complex",
        detrend=False,
    )
    return freqs, time_sec, np.abs(sp)


def _remove_one_over_f(ps: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_mask = ((freqs >= 2.0) & (freqs <= 7.0)) | ((freqs >= 14.0) & (freqs <= 24.0))
    log_f_all = np.log10(np.maximum(freqs, np.finfo(float).eps))
    log_f_fit = log_f_all[fit_mask]
    n_times = ps.shape[1]
    slope = np.full(n_times, np.nan, dtype=float)
    offset = np.full(n_times, np.nan, dtype=float)
    residual_logp = np.full(ps.shape, np.nan, dtype=float)

    for t in range(n_times):
        y = ps[fit_mask, t]
        good = np.isfinite(y) & (y > 0)
        if int(np.sum(good)) < 5:
            continue
        coeff = np.polyfit(log_f_fit[good], np.log10(y[good]), 1)
        slope[t] = coeff[0]
        offset[t] = coeff[1]
        bg = offset[t] + slope[t] * log_f_all
        residual_logp[:, t] = np.log10(np.maximum(ps[:, t], np.finfo(float).tiny)) - bg

    return residual_logp, -slope, offset


def _band_median(values: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return np.full(values.shape[1], np.nan, dtype=float)
    return np.nanmedian(values[mask, :], axis=0)


def _band_mean(values: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return np.full(values.shape[1], np.nan, dtype=float)
    return np.nanmean(values[mask, :], axis=0)


def _band_envelope(values: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = 0.5 * sample_rate_hz
    if nyquist <= low_hz:
        return np.zeros_like(values, dtype=float)
    high = min(high_hz, nyquist * 0.99)
    if high <= low_hz:
        return np.zeros_like(values, dtype=float)
    filtered = bandpass_filter(values, sample_rate_hz, low_hz, high, order=3)
    return np.abs(hilbert(filtered))


def _time_series_rate(time_sec: np.ndarray) -> float:
    if time_sec.size < 2:
        return 1.0
    step = float(np.median(np.diff(time_sec)))
    if step <= 0:
        return 1.0
    return 1.0 / step


def _build_timebin_events(starts: np.ndarray, lengths: np.ndarray, label: str, sample_rate_hz: float) -> list[Event]:
    events: list[Event] = []
    for start, length in zip(starts, lengths, strict=False):
        end = int(start + length - 1)
        events.append(
            Event(
                start=int(start),
                end=end,
                label=label,
                kind="period",
                metadata={
                    "sample_rate_hz": sample_rate_hz,
                    "start_sec": float(start / sample_rate_hz),
                    "end_sec": float(end / sample_rate_hz),
                },
            )
        )
    return events


def _aggregate_blocks(values: np.ndarray, block_size: int, *, agg: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    usable = (values.size // block_size) * block_size
    if usable == 0:
        return np.array([], dtype=float)
    blocks = values[:usable].reshape(-1, block_size)
    if agg == "mean":
        return np.nanmean(blocks, axis=1)
    if agg == "var":
        return np.nanvar(blocks, axis=1)
    raise ValueError(f"Unknown aggregation: {agg}")


def _safe_log10(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, np.finfo(float).tiny))
