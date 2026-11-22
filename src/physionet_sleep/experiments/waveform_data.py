"""Read aligned 200 Hz CinC windows directly from the memory-mapped recordings."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from physionet_sleep.io.cinc2018 import CHANNEL_MAP, resolve_cinc2018_record
from physionet_sleep.labels.cinc2018 import EPOCH_SECONDS

CHANNELS = tuple(CHANNEL_MAP)[:8]
SAMPLE_RATE = 200
EPOCH_SAMPLES = EPOCH_SECONDS * SAMPLE_RATE
CONTEXT_EPOCHS = 2
# Fixed units across subjects, folds and model variants. 100 µV maps to 1.
MICROVOLTS_PER_UNIT = 100.0
_MISSING_SAMPLE = np.iinfo(np.int16).min


@dataclass
class _Night:
    samples: np.memmap
    columns: tuple[int, ...]
    gain: np.ndarray
    baseline: np.ndarray


class WaveformStore:
    """Bounded pool of raw record mappings. Each read copies only the requested epochs."""

    def __init__(self, raw_root: Path, *, permitted_subjects: Iterable[str], max_open_nights: int = 8):
        self.raw_root = Path(raw_root)
        self.permitted_subjects = frozenset(permitted_subjects)
        if any(not s or Path(s).name != s or s in {".", ".."} for s in self.permitted_subjects):
            raise ValueError("expected plain CinC record IDs")
        if max_open_nights < 1:
            raise ValueError("max_open_nights must be positive")
        self.max_open_nights = max_open_nights
        self._nights: OrderedDict[str, _Night] = OrderedDict()

    def _night(self, subject_id: str) -> _Night:
        if subject_id not in self.permitted_subjects:
            raise ValueError(f"{subject_id!r} is outside the permitted waveform manifest")
        if subject_id in self._nights:
            self._nights.move_to_end(subject_id)
            return self._nights[subject_id]

        import wfdb

        paths = resolve_cinc2018_record(self.raw_root, subject_id)
        if paths.signal_path is None:
            raise FileNotFoundError(f"waveform is missing for {subject_id}")
        header = wfdb.rdheader(str(paths.header_path.with_suffix("")))
        if header.fs != SAMPLE_RATE or header.sig_len <= 0:
            raise ValueError(f"{subject_id}: expected a non-empty {SAMPLE_RATE} Hz waveform")
        if not (
            all(name == paths.signal_path.name for name in header.file_name)
            and all(fmt == "16" for fmt in header.fmt)
            and set(header.byte_offset) == {24}
            and all(n == 1 for n in header.samps_per_frame)
            and all(n in (None, 0) for n in header.skew)
        ):
            raise ValueError(f"{subject_id}: expected the interleaved CinC waveform format")
        if paths.signal_path.stat().st_size != 24 + 2 * header.sig_len * header.n_sig:
            raise ValueError(f"{subject_id}: waveform length differs from its header")

        columns = tuple(header.sig_name.index(ch) if ch in header.sig_name else -1 for ch in CHANNELS)
        if all(i < 0 for i in columns):
            raise ValueError(f"{subject_id}: none of the expected waveform channels are present")
        indices = [i for i in columns if i >= 0]
        units_to_uv = {"uV": 1.0, "mV": 1000.0, "V": 1_000_000.0}
        if any(header.units[i] not in units_to_uv for i in indices):
            raise ValueError(f"{subject_id}: expected voltage units for all selected channels")
        gain = np.asarray([
            header.adc_gain[i] * MICROVOLTS_PER_UNIT / units_to_uv[header.units[i]] for i in indices
        ], dtype=np.float32)
        if not np.isfinite(gain).all() or (gain <= 0).any():
            raise ValueError(f"{subject_id}: invalid waveform calibration gain")
        baseline = np.asarray([header.baseline[i] for i in indices], dtype=np.float32)
        if not np.isfinite(baseline).all():
            raise ValueError(f"{subject_id}: invalid waveform calibration baseline")
        samples = np.memmap(paths.signal_path, mode="r", dtype="<i2", offset=24,
                            shape=(header.sig_len, header.n_sig))
        night = _Night(samples=samples, columns=columns, gain=gain, baseline=baseline)
        self._nights[subject_id] = night
        if len(self._nights) > self.max_open_nights:
            self._nights.popitem(last=False)
        return night

    def channel_uv(self, subject_id: str, channel_name: str) -> np.ndarray:
        """Copy one full lead in physical microvolts, with missing samples set to NaN."""
        night = self._night(subject_id)
        index = CHANNELS.index(channel_name)
        column = night.columns[index]
        if column < 0:
            raise ValueError(f"{subject_id}: waveform has no {channel_name} lead")
        calibration_index = sum(column >= 0 for column in night.columns[:index])
        digital = night.samples[:, column]
        gain_uv = float(night.gain[calibration_index]) / MICROVOLTS_PER_UNIT
        physical = (digital.astype(np.float64) - night.baseline[calibration_index]) / gain_uv
        physical[digital == _MISSING_SAMPLE] = np.nan
        return physical

    def epochs(self, subject_id: str, epoch_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned waveform values, a sample-time mask, and an epoch-level lead mask."""
        night = self._night(subject_id)
        epochs = tuple(epoch_indices)
        data = np.zeros((len(epochs), len(CHANNELS), EPOCH_SAMPLES), np.float32)
        valid = np.zeros((len(epochs), EPOCH_SAMPLES), dtype=bool)
        channels = np.zeros((len(epochs), len(CHANNELS)), dtype=bool)
        present = np.flatnonzero(np.asarray(night.columns) >= 0)
        selected = np.asarray(night.columns)[present]
        for row, epoch in enumerate(epochs):
            start = int(epoch) * EPOCH_SAMPLES
            end = min(start + EPOCH_SAMPLES, len(night.samples))
            if start < 0 or start >= end:
                continue
            # Advanced indexing creates only a bounded per-epoch copy.
            digital = night.samples[start:end, selected]
            observed = digital != _MISSING_SAMPLE
            physical = (digital.astype(np.float32) - night.baseline) / night.gain
            physical[~observed] = 0
            data[row, present, :end - start] = physical.T
            valid[row, :end - start] = True
            channels[row, present] = observed.any(axis=0)
        return data, valid, channels

    def window(self, subject_id: str, epoch_index: int) -> dict[str, np.ndarray]:
        context = range(epoch_index - CONTEXT_EPOCHS, epoch_index + CONTEXT_EPOCHS + 1)
        data, valid, channels = self.epochs(subject_id, context)
        return {"x": data, "valid": valid, "channels": channels}
