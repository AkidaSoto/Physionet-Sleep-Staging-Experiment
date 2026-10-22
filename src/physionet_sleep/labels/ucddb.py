from __future__ import annotations

from pathlib import Path

import numpy as np

from physionet_sleep.core.types import Event, LabelTrack, RecordLabels
from physionet_sleep.io.ucddb import UCDDBRecordPaths


# PhysioNet UCDDB stage codes: 0 Wake, 1 REM, 2 S1, 3 S2, 4 S3, 5 S4,
# 6 artifact, 7 indeterminate. R&K S3 and S4 are merged into AASM N3.
UCDDB_STAGE_CLASS_MAP = {
    0: "wake",
    1: "rem",
    2: "n1",
    3: "n2",
    4: "n3",
    5: "n3",
}

# Collapsed integer codes used by downstream tracks and feature tables.
COLLAPSED_STAGE_CLASS_MAP = {0: "wake", 1: "n1", 2: "n2", 3: "n3", 4: "rem"}
_RAW_TO_COLLAPSED = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3, 5: 3}

RESP_EVENT_LABELS = {
    "HYP-C": "hypopnea_central",
    "HYP-O": "hypopnea_obstructive",
    "HYP-M": "hypopnea_mixed",
    "APNEA-C": "apnea_central",
    "APNEA-O": "apnea_obstructive",
    "APNEA-M": "apnea_mixed",
}


def load_ucddb_labels(paths: UCDDBRecordPaths) -> RecordLabels:
    labels = RecordLabels(metadata={"dataset": "ucddb", "record_id": paths.record_id})

    if paths.stage_path is not None:
        raw_stage = _load_stage_codes(paths.stage_path)
        collapsed = np.vectorize(lambda x: UCDDB_STAGE_CLASS_MAP.get(int(x), "unknown"))(raw_stage)
        second_track = np.repeat(raw_stage, 30)
        second_collapsed = np.repeat(
            np.array([_collapse_stage_code(code) for code in raw_stage], dtype=int), 30
        )
        labels.tracks["stage_epochs_raw"] = LabelTrack(
            values=raw_stage,
            step_seconds=30.0,
            class_map=UCDDB_STAGE_CLASS_MAP,
            metadata={"representation": "raw_epoch_codes"},
        )
        labels.tracks["stage_seconds_collapsed"] = LabelTrack(
            values=second_collapsed,
            step_seconds=1.0,
            class_map=COLLAPSED_STAGE_CLASS_MAP,
            metadata={"representation": "collapsed_second_codes"},
        )
        labels.metadata["stage_labels"] = collapsed.tolist()

    if paths.respevt_path is not None:
        labels.events["respiratory"] = _load_respiratory_events(paths.respevt_path)

    return labels


def _load_stage_codes(path: Path) -> np.ndarray:
    values = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return np.asarray(values, dtype=int)


def _load_respiratory_events(path: Path) -> list[Event]:
    lines = path.read_text().splitlines()[3:]
    events: list[Event] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue

        time_token = tokens[0]
        type_token = tokens[1]

        duration_idx = 2
        if not _is_numeric_token(tokens[duration_idx]) and len(tokens) > 3:
            duration_idx = 3

        if duration_idx >= len(tokens) or not _is_numeric_token(tokens[duration_idx]):
            continue

        start_sec = _hhmmss_to_seconds(time_token)
        duration_sec = int(round(float(tokens[duration_idx])))
        end_sec = start_sec + max(duration_sec - 1, 0)

        events.append(
            Event(
                start=start_sec,
                end=end_sec,
                label=RESP_EVENT_LABELS.get(type_token, type_token.lower()),
                kind="period",
                metadata={
                    "time_hhmmss": time_token,
                    "type_raw": type_token,
                    "duration_seconds": duration_sec,
                },
            )
        )
    return events


def _hhmmss_to_seconds(value: str) -> int:
    hh, mm, ss = value.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def _is_numeric_token(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _collapse_stage_code(code: int) -> int:
    return _RAW_TO_COLLAPSED.get(int(code), -1)
