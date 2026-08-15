from __future__ import annotations

import re
from typing import Any

import numpy as np

from physionet_sleep.core.types import SignalRecord


CHANNEL_ALIASES: dict[str, str] = {
    "ecg": "ecg",
    "spo2": "spo2",
    "flow": "flow",
    "sound": "snore",
    "ribcage": "effort_thoracic",
    "abdo": "effort_abdominal",
    "sum": "effort_sum",
    "pulse": "pulse",
    "bodypos": "body_position",
    "lefteye": "left_eye",
    "righteye": "right_eye",
    "emg": "emg",
    "c3a2": "eeg_c3a2",
    "c4a1": "eeg_c4a1",
}


def load_edf_record(path: str) -> SignalRecord:
    try:
        import pyedflib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pyedflib is required for EDF loading. Install it in the repo environment "
            "to use load_edf_record()."
        ) from exc

    record = SignalRecord(metadata={"edf_path": path})
    reader = pyedflib.EdfReader(path)
    try:
        signal_labels = list(reader.getSignalLabels())
        for idx, raw_label in enumerate(signal_labels):
            canonical = _canonicalize_label(raw_label)
            signal = np.asarray(reader.readSignal(idx), dtype=float)
            sample_rate = float(reader.getSampleFrequency(idx))
            signal_header = _safe_signal_header(reader, idx)
            units = signal_header.get("dimension")
            record.add_channel(
                canonical,
                signal,
                sample_rate,
                units=units,
                metadata={"source_label": raw_label, "signal_header": signal_header},
            )

        _add_ucddb_alias_channels(record)
        file_header = _safe_file_header(reader)
        record.metadata["edf_header"] = file_header
        return record
    finally:
        reader.close()


def _safe_signal_header(reader: Any, idx: int) -> dict[str, Any]:
    try:
        header = reader.getSignalHeader(idx)
        return dict(header) if header is not None else {}
    except Exception:
        return {}


def _safe_file_header(reader: Any) -> dict[str, Any]:
    try:
        header = reader.getHeader()
        return dict(header) if header is not None else {}
    except Exception:
        return {}


def _canonicalize_label(label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", label.lower())
    return CHANNEL_ALIASES.get(key, key)


def _add_ucddb_alias_channels(record: SignalRecord) -> None:
    if "flow" in record.channels:
        flow = record.get("flow")
        for alias in ("airflow_thermal", "airflow_pressure"):
            if alias not in record.channels:
                record.add_channel(
                    alias,
                    flow.values.copy(),
                    flow.sample_rate,
                    units=flow.units,
                    metadata={"alias_of": "flow"},
                )

    eeg_keys = [key for key in ("eeg_c3a2", "eeg_c4a1") if key in record.channels]
    if eeg_keys and "eeg" not in record.channels:
        eeg_channels = [record.get(key) for key in eeg_keys]
        fs = eeg_channels[0].sample_rate
        stacked = np.column_stack([series.values for series in eeg_channels])
        record.add_channel(
            "eeg",
            stacked,
            fs,
            units=eeg_channels[0].units,
            metadata={"source_channels": eeg_keys},
        )

    if "left_eye" in record.channels and "right_eye" in record.channels and "eog" not in record.channels:
        left = record.get("left_eye")
        right = record.get("right_eye")
        if left.sample_rate == right.sample_rate:
            diff = np.asarray(left.values) - np.asarray(right.values)
            record.add_channel(
                "eog",
                diff,
                left.sample_rate,
                units=left.units,
                metadata={"source_channels": ["left_eye", "right_eye"], "mode": "difference"},
            )
