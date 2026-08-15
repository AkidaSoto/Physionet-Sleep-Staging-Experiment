from __future__ import annotations

import numpy as np


def count_true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    padded = np.pad(mask.astype(int), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    return starts.astype(int), lengths.astype(int)


def expand_runs(starts: np.ndarray, lengths: np.ndarray, total_length: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=int).ravel()
    lengths = np.asarray(lengths, dtype=int).ravel()
    out = np.zeros(total_length, dtype=bool)
    for start, length in zip(starts, lengths, strict=False):
        if length <= 0:
            continue
        end = min(start + length, total_length)
        out[max(start, 0) : end] = True
    return out
