"""ISRUC-Sleep subgroup I (Coimbra): 100 one-night clinical PSGs, two independent AASM scorers.

Each subject folder `<n>/` holds `<n>.rec` (EDF) and `<n>_1.txt`, `<n>_2.txt` (one stage code per 30-s epoch from the
two scorers). EEG/EOG/chin/ECG/legs at 200 Hz, effort belts at 25 Hz, flow and SpO2 at 12.5 Hz. Two label styles
exist: older files say `F3-A2`, `LOC-A2`, newer ones `F3-M2`, `E1-M2`; both are contralateral-mastoid derivations,
so they map onto the CinC 2018 montage one to one. `X1` is chin EMG, `X2` ECG, `X7`/`X8` abdominal/thoracic effort;
some newer files number these channels instead (`24` chin, `25` ECG, `30`/`31` effort: same filters, same order).

Calibration: some files declare a +-25 uV physical range for EEG/EOG and their signals come out about ten times
smaller than in the files that declare +-2490 uV (median N3 frontal slow-wave amplitude ~8 vs ~75 uV). The EDF
headers of those files are taken to be off by a factor of ten and are rescaled (see `EEG_GAIN_FIX`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physionet_sleep.core.types import SignalRecord

# Both label styles -> the CinC 2018 channel keys used by the shared staging pipeline.
CHANNEL_MAP = {
    "F3-A2": "eeg_f3", "F3-M2": "eeg_f3",
    "F4-A1": "eeg_f4", "F4-M1": "eeg_f4",
    "C3-A2": "eeg_c3", "C3-M2": "eeg_c3",
    "C4-A1": "eeg_c4", "C4-M1": "eeg_c4",
    "O1-A2": "eeg_o1", "O1-M2": "eeg_o1",
    "O2-A1": "eeg_o2", "O2-M1": "eeg_o2",
    "LOC-A2": "eog", "E1-M2": "eog",
    "ROC-A1": "right_eye", "E2-M1": "right_eye",
    "X1": "emg", "24": "emg",
    "X2": "ecg", "25": "ecg",
    "X7": "effort_abdominal", "30": "effort_abdominal",
    "X8": "effort_thoracic", "31": "effort_thoracic",
}
# CinC 2018 raw names of the same derivations (for algorithms.eeg_tracks.compute_tracks).
CINC_NAMES = {"eeg_f3": "F3-M2", "eeg_f4": "F4-M1", "eeg_c3": "C3-M2", "eeg_c4": "C4-M1", "eeg_o1": "O1-M2",
              "eeg_o2": "O2-M1", "eog": "E1-M2", "emg": "Chin1-Chin2"}
EEG_STACKS = {
    "eeg": ("eeg_c3", "eeg_c4"),
    "eeg_frontal": ("eeg_f3", "eeg_f4"),
    "eeg_occipital": ("eeg_o1", "eeg_o2"),
}
# A few files (e.g. subject 40) store single electrodes against the amplifier reference; derive the same pairs.
REFERENTIAL = {
    "eeg_f3": ("F3", "A2"), "eeg_f4": ("F4", "A1"), "eeg_c3": ("C3", "A2"), "eeg_c4": ("C4", "A1"),
    "eeg_o1": ("O1", "A2"), "eeg_o2": ("O2", "A1"), "eog": ("LOC", "A2"), "right_eye": ("ROC", "A1"),
}
EXG_KEYS = ("eeg_f3", "eeg_f4", "eeg_c3", "eeg_c4", "eeg_o1", "eeg_o2", "eog", "right_eye")
# Files whose EEG/EOG header range is below this (uV) are the +-25 uV ones; their samples are multiplied by EEG_GAIN_FIX.
LOW_RANGE_UV = 100.0
EEG_GAIN_FIX = 10.0


@dataclass(slots=True)
class ISRUCRecordPaths:
    record_id: str
    root_dir: Path
    rec_path: Path
    scorer_paths: tuple[Path | None, Path | None]


def discover_isruc_records(root_dir: str | Path) -> list[ISRUCRecordPaths]:
    root = Path(root_dir)
    ids = sorted((p.parent.name for p in root.glob("*/*.rec") if p.stem == p.parent.name), key=int)
    return [resolve_isruc_record(root, i) for i in ids]


def resolve_isruc_record(root_dir: str | Path, record_id: str) -> ISRUCRecordPaths:
    folder = Path(root_dir) / str(record_id)
    rec = folder / f"{record_id}.rec"
    if not rec.exists():
        raise FileNotFoundError(f"ISRUC recording not found: {rec}")
    scorers = tuple(p if p.exists() else None for p in (folder / f"{record_id}_1.txt", folder / f"{record_id}_2.txt"))
    return ISRUCRecordPaths(record_id=str(record_id), root_dir=Path(root_dir), rec_path=rec, scorer_paths=scorers)


def read_isruc_channels(paths: ISRUCRecordPaths) -> tuple[dict[str, tuple[np.ndarray, float]], dict]:
    """Channel key -> (samples in uV, fs), with the +-25 uV header fix applied; plus calibration metadata."""
    import pyedflib

    reader = pyedflib.EdfReader(str(paths.rec_path))
    try:
        out, ranges = {}, {}
        labels = [label.strip() for label in reader.getSignalLabels()]
        for i, label in enumerate(labels):
            key = CHANNEL_MAP.get(label)
            if key is None or key in out:
                continue
            out[key] = (reader.readSignal(i).astype(np.float64), float(reader.getSampleFrequency(i)))
            ranges[key] = float(reader.getPhysicalMaximum(i))
        for key, (active, ref) in REFERENTIAL.items():
            if key in out or active not in labels or ref not in labels:
                continue
            a, r = labels.index(active), labels.index(ref)
            out[key] = (reader.readSignal(a) - reader.readSignal(r), float(reader.getSampleFrequency(a)))
            ranges[key] = float(reader.getPhysicalMaximum(a))
    finally:
        reader.close()
    low = [k for k in EXG_KEYS if k in ranges and ranges[k] < LOW_RANGE_UV]
    for k in low:
        x, fs = out[k]
        out[k] = (x * EEG_GAIN_FIX, fs)
    return out, {"exg_gain_fixed": bool(low), "exg_header_range_uv": {k: ranges[k] for k in EXG_KEYS if k in ranges}}


def load_isruc_waveforms(paths: ISRUCRecordPaths) -> SignalRecord:
    """SignalRecord with the CinC 2018 channel keys, so the CinC staging recipe runs unchanged."""
    channels, calib = read_isruc_channels(paths)
    record = SignalRecord(metadata={"dataset": "isruc", "record_id": paths.record_id, **calib})
    for key, (x, fs) in channels.items():
        record.add_channel(key, x, fs, units="uV")
    if "eog" in record.channels:
        eog = record.get("eog")
        record.add_channel("left_eye", eog.values, eog.sample_rate, units=eog.units, metadata={"alias_of": "eog"})
    for stack_name, members in EEG_STACKS.items():
        if all(m in record.channels for m in members):
            series = [record.get(m) for m in members]
            n = min(len(s.values) for s in series)
            record.add_channel(stack_name, np.column_stack([s.values[:n] for s in series]), series[0].sample_rate,
                               units="uV", metadata={"source_channels": list(members)})
    if "effort_abdominal" in record.channels and "effort_thoracic" in record.channels:
        abd, chest = record.get("effort_abdominal"), record.get("effort_thoracic")
        n = min(len(abd.values), len(chest.values))
        record.add_channel("effort_sum", np.asarray(abd.values[:n]) + np.asarray(chest.values[:n]), abd.sample_rate,
                           units=abd.units, metadata={"source_channels": ["effort_abdominal", "effort_thoracic"], "mode": "sum"})
    return record


def record_length_seconds(paths: ISRUCRecordPaths) -> float:
    import pyedflib

    reader = pyedflib.EdfReader(str(paths.rec_path))
    try:
        return float(reader.getFileDuration())
    finally:
        reader.close()
