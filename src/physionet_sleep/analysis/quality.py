from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from physionet_sleep.core.types import SignalRecord

# Recording dropouts in UCDDB: stretches where every channel holds one constant value (e.g. ucddb008, 37 epochs,
# all scored Wake). A flat line looks like low-EMG, slow-wave-free sleep and is predicted mostly as REM.
DEFAULT_EPOCH_SECONDS = 30.0
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_MIN_FLAT_FRACTION = 0.5


def flat_dropout_epochs(
    channels: Mapping[str, tuple[np.ndarray, float]],
    epoch_starts: Iterable[float],
    *,
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_flat_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
) -> np.ndarray:
    """Flag epochs where every channel is exactly flat for at least `min_flat_fraction` of the epoch.

    `channels` maps a name to (values, sample_rate_hz); `epoch_starts` are epoch start times in seconds from
    recording start. Each epoch is split into `window_seconds` windows; a window is flat when all its samples are
    identical. A channel's flat fraction is flat windows / windows that lie inside the recording. Epochs with no
    complete window in some channel (past the end of the signal) are never flagged.
    """
    starts = np.asarray(list(epoch_starts), dtype=float)
    if not channels:
        raise ValueError("flat_dropout_epochs needs at least one channel")
    flagged = np.ones(starts.size, dtype=bool)
    for name, (values, sample_rate) in channels.items():
        flagged &= _channel_flat_fraction(
            np.asarray(values),
            float(sample_rate),
            starts,
            epoch_seconds=epoch_seconds,
            window_seconds=window_seconds,
        ) >= min_flat_fraction
    return flagged


def flat_dropout_epochs_from_record(
    record: SignalRecord,
    epoch_starts: Iterable[float],
    *,
    channel_names: Iterable[str] | None = None,
    **kwargs: float,
) -> np.ndarray:
    """`flat_dropout_epochs` over channels of a SignalRecord (all channels when `channel_names` is None)."""
    names = list(channel_names) if channel_names is not None else list(record.channels)
    channels = {name: (record.get(name).values, record.get(name).sample_rate) for name in names}
    return flat_dropout_epochs(channels, epoch_starts, **kwargs)


def _channel_flat_fraction(
    values: np.ndarray,
    sample_rate: float,
    starts: np.ndarray,
    *,
    epoch_seconds: float,
    window_seconds: float,
) -> np.ndarray:
    window = int(round(window_seconds * sample_rate))
    if window < 2:
        raise ValueError(f"window of {window_seconds} s at {sample_rate} Hz has fewer than 2 samples")
    n_windows = int(round(epoch_seconds / window_seconds))
    out = np.zeros(starts.size, dtype=float)
    for i, start in enumerate(starts):
        a = int(round(start * sample_rate))
        seg = values[a : a + n_windows * window]
        complete = seg.size // window
        if a < 0 or complete == 0:
            continue
        blocks = seg[: complete * window].reshape(complete, window)
        out[i] = float(np.mean(np.ptp(blocks, axis=1) == 0))
    return out
