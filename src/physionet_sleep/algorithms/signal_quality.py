from __future__ import annotations

from typing import Mapping

import numpy as np

from physionet_sleep.analysis.quality import flat_dropout_epochs
from physionet_sleep.core.base import BaseAlgorithm
from physionet_sleep.core.types import AlgorithmResult, SignalRecord, SignalSeries


class FlatDropoutDetector(BaseAlgorithm):
    """Flags 30-s epochs where every electrophysiology channel is flat (a recording dropout, not sleep).

    Emitted on the 6-s row grid like EpochRatioFeatureGenerator (sample i starts at i * row_seconds and carries the
    flag of the epoch containing it), so the feature table and the 30-s benchmark windows pick it up without drift.
    """

    def __init__(
        self,
        *,
        channel_keys: tuple[str, ...] = ("eeg", "left_eye", "right_eye", "emg", "ecg"),
        epoch_seconds: float = 30.0,
        row_seconds: float = 6.0,
    ) -> None:
        super().__init__(name="signal_quality")
        self.channel_keys = channel_keys
        self.epoch_seconds = float(epoch_seconds)
        self.row_seconds = float(row_seconds)
        self.rows_per_epoch = int(round(self.epoch_seconds / self.row_seconds))

    def required_signals(self) -> tuple[str, ...]:
        return ()

    def _run(
        self,
        record: SignalRecord,
        *,
        prior: Mapping[str, AlgorithmResult],
    ) -> AlgorithmResult:
        channels: dict[str, tuple[np.ndarray, float]] = {}
        for key in self.channel_keys:
            if key not in record.channels:
                continue
            series = record.get(key)
            values = np.asarray(series.values, dtype=float)
            columns = values.reshape(values.shape[0], -1) if values.ndim > 1 else values[:, None]
            for index in range(columns.shape[1]):
                channels[f"{key}_{index}"] = (columns[:, index], float(series.sample_rate))
        if not channels:
            raise ValueError("FlatDropoutDetector found none of its channels in the record")
        duration = max(values.size / rate for values, rate in channels.values())
        n_epochs = int(np.ceil(duration / self.epoch_seconds))
        flags = flat_dropout_epochs(channels, np.arange(n_epochs) * self.epoch_seconds, epoch_seconds=self.epoch_seconds)
        return AlgorithmResult(
            name=self.name,
            signals={
                "flat_dropout": SignalSeries(
                    np.repeat(flags.astype(float), self.rows_per_epoch),
                    sample_rate=1.0 / self.row_seconds,
                    units="flag",
                    metadata={"start_seconds": 0.0},
                )
            },
            features={"flat_dropout_epochs": int(flags.sum()), "channels": sorted(channels)},
        )
