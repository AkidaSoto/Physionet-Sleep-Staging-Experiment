from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    return (
        pd.Series(np.asarray(values))
        .rolling(window=window, min_periods=1)
        .median()
        .to_numpy()
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    return (
        pd.Series(np.asarray(values))
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )


def future_rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    arr = np.asarray(values)
    return (
        pd.Series(arr[::-1]).rolling(window=window, min_periods=1).min().to_numpy()[::-1]
    )


def bandpass_filter(
    values: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 3,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    nyquist = 0.5 * sample_rate
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError("Cutoffs must be positive")
    if high_hz >= nyquist:
        high_hz = nyquist * 0.99
    if low_hz >= high_hz:
        raise ValueError("low_hz must be less than high_hz")

    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, values)
