from __future__ import annotations

import numpy as np


def segment_strides(length: int, window_length: int, overlap_pct: float) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if not 0 <= overlap_pct < 100:
        raise ValueError("overlap_pct must be in [0, 100)")
    if window_length > length:
        return np.empty((0, window_length), dtype=int)

    stride = round(window_length - (overlap_pct / 100.0 * window_length))
    stride = max(stride, 1)
    starts = np.arange(0, length - window_length + 1, stride, dtype=int)[:, None]
    offsets = np.arange(window_length, dtype=int)[None, :]
    return starts + offsets
