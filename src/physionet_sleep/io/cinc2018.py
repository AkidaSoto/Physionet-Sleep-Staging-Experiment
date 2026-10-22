"""PhysioNet/CinC Challenge 2018 ("You Snooze You Win") training records.

Each record directory holds `<id>.hea` / `<id>.mat` (13 channels at 200 Hz, EEG/EOG/EMG in microvolts) and
`<id>.arousal` (WFDB annotations: sleep stages at transitions plus "(event" / "event)" pairs for arousals and
respiratory events). Montage: F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, E1-M2, Chin1-Chin2, ABD, CHEST, AIRFLOW,
SaO2, ECG. There is a single EOG derivation (no right eye).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physionet_sleep.core.types import SignalRecord

CHANNEL_MAP = {
    "F3-M2": "eeg_f3",
    "F4-M1": "eeg_f4",
    "C3-M2": "eeg_c3",
    "C4-M1": "eeg_c4",
    "O1-M2": "eeg_o1",
    "O2-M1": "eeg_o2",
    "E1-M2": "eog",
    "Chin1-Chin2": "emg",
    "ABD": "effort_abdominal",
    "CHEST": "effort_thoracic",
    "AIRFLOW": "flow",
    "SaO2": "spo2",
    "ECG": "ecg",
}

# Stacked EEG views. "eeg" keeps the central pair so the shared staging algorithms see the same derivations as UCDDB.
EEG_STACKS = {
    "eeg": ("eeg_c3", "eeg_c4"),
    "eeg_frontal": ("eeg_f3", "eeg_f4"),
    "eeg_occipital": ("eeg_o1", "eeg_o2"),
}


@dataclass(slots=True)
class CinC2018RecordPaths:
    record_id: str
    root_dir: Path
    header_path: Path
    signal_path: Path | None
    annotation_path: Path | None


def discover_cinc2018_records(root_dir: str | Path, *, require_signals: bool = True) -> list[CinC2018RecordPaths]:
    root = Path(root_dir)
    records = []
    for header in sorted(root.glob("tr*/*.hea")):
        paths = resolve_cinc2018_record(root, header.stem)
        if require_signals and paths.signal_path is None:
            continue
        records.append(paths)
    return records


def resolve_cinc2018_record(root_dir: str | Path, record_id: str) -> CinC2018RecordPaths:
    folder = Path(root_dir) / record_id
    header = folder / f"{record_id}.hea"
    if not header.exists():
        raise FileNotFoundError(f"CinC 2018 header not found: {header}")
    signal = folder / f"{record_id}.mat"
    annotation = folder / f"{record_id}.arousal"
    return CinC2018RecordPaths(
        record_id=record_id,
        root_dir=Path(root_dir),
        header_path=header,
        signal_path=signal if signal.exists() else None,
        annotation_path=annotation if annotation.exists() else None,
    )


def load_cinc2018_waveforms(paths: CinC2018RecordPaths) -> SignalRecord:
    import wfdb

    if paths.signal_path is None:
        raise FileNotFoundError(f"Signals not downloaded for {paths.record_id}")
    raw = wfdb.rdrecord(str(paths.header_path.with_suffix("")))
    record = SignalRecord(metadata={"dataset": "cinc2018", "record_id": paths.record_id, "units": dict(zip(raw.sig_name, raw.units))})
    fs = float(raw.fs)
    for index, name in enumerate(raw.sig_name):
        key = CHANNEL_MAP.get(name)
        if key is None:
            continue
        record.add_channel(key, raw.p_signal[:, index].astype(float), fs, units=raw.units[index], metadata={"source_label": name})
    if "eog" in record.channels:
        eog = record.get("eog")
        record.add_channel("left_eye", eog.values, eog.sample_rate, units=eog.units, metadata={"alias_of": "eog"})
    for stack_name, members in EEG_STACKS.items():
        if all(member in record.channels for member in members):
            series = [record.get(member) for member in members]
            record.add_channel(
                stack_name,
                np.column_stack([s.values for s in series]),
                series[0].sample_rate,
                units=series[0].units,
                metadata={"source_channels": list(members)},
            )
    if "effort_abdominal" in record.channels and "effort_thoracic" in record.channels:
        abd, chest = record.get("effort_abdominal"), record.get("effort_thoracic")
        record.add_channel(
            "effort_sum",
            np.asarray(abd.values) + np.asarray(chest.values),
            abd.sample_rate,
            units=abd.units,
            metadata={"source_channels": ["effort_abdominal", "effort_thoracic"], "mode": "sum"},
        )
    if "flow" in record.channels:
        flow = record.get("flow")
        for alias in ("airflow_thermal", "airflow_pressure"):
            record.add_channel(alias, flow.values, flow.sample_rate, units=flow.units, metadata={"alias_of": "flow"})
    return record


def record_length_seconds(paths: CinC2018RecordPaths) -> float:
    first = paths.header_path.read_text().splitlines()[0].split()
    return float(first[3]) / float(first[2])
