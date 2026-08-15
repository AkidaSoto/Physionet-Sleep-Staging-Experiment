from __future__ import annotations

import numpy as np


def has_interval_overlap(
    starts: np.ndarray,
    lengths: np.ndarray,
    other_starts: np.ndarray,
    other_lengths: np.ndarray,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=int).ravel()
    lengths = np.asarray(lengths, dtype=int).ravel()
    other_starts = np.asarray(other_starts, dtype=int).ravel()
    other_lengths = np.asarray(other_lengths, dtype=int).ravel()

    overlaps = np.zeros(starts.shape[0], dtype=bool)
    if starts.size == 0 or other_starts.size == 0:
        return overlaps

    ends = starts + lengths - 1
    other_ends = other_starts + other_lengths - 1

    for idx, (start, end) in enumerate(zip(starts, ends, strict=False)):
        overlaps[idx] = np.any((start <= other_ends) & (end >= other_starts))
    return overlaps
