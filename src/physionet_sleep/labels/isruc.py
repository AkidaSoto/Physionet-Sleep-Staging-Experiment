"""Sleep stages from ISRUC-Sleep hypnograms (`<n>_1.txt`, `<n>_2.txt`: one code per 30-s epoch from recording start).

Codes: 0 W, 1 N1, 2 N2, 3 N3, 5 REM (4, an R&K S4 code, is merged into N3 if it appears). Scorer 1 is the reference
label everywhere unless a scorer is named; scorer 2 is kept for inter-scorer agreement.
"""
from __future__ import annotations

import numpy as np

from physionet_sleep.core.types import LabelTrack, RecordLabels
from physionet_sleep.io.isruc import ISRUCRecordPaths, record_length_seconds
from physionet_sleep.labels.ucddb import COLLAPSED_STAGE_CLASS_MAP

RAW_TO_COLLAPSED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
EPOCH_SECONDS = 30


def read_isruc_hypnogram(paths: ISRUCRecordPaths, scorer: int = 1) -> np.ndarray:
    """Collapsed stage code per 30-s epoch (W 0, N1 1, N2 2, N3 3, R 4), cut to the recording; -1 if unknown."""
    path = paths.scorer_paths[scorer - 1]
    if path is None:
        raise FileNotFoundError(f"ISRUC scorer {scorer} hypnogram missing for subject {paths.record_id}")
    raw = [int(v) for v in path.read_text().split() if v.strip().lstrip("-").isdigit()]
    codes = np.array([RAW_TO_COLLAPSED.get(v, -1) for v in raw], dtype=int)
    n_epochs = int(record_length_seconds(paths) // EPOCH_SECONDS)
    out = np.full(n_epochs, -1, dtype=int)
    out[: min(n_epochs, len(codes))] = codes[:n_epochs]
    return out


def load_isruc_labels(paths: ISRUCRecordPaths, scorer: int = 1) -> RecordLabels:
    epochs = read_isruc_hypnogram(paths, scorer)
    labels = RecordLabels(metadata={"dataset": "isruc", "record_id": paths.record_id, "scorer": scorer})
    labels.tracks["stage_epochs_raw"] = LabelTrack(
        values=epochs,
        step_seconds=float(EPOCH_SECONDS),
        class_map=dict(COLLAPSED_STAGE_CLASS_MAP),
        metadata={"representation": "collapsed_epoch_codes"},
    )
    labels.tracks["stage_seconds_collapsed"] = LabelTrack(
        values=np.repeat(epochs, EPOCH_SECONDS),
        step_seconds=1.0,
        class_map=dict(COLLAPSED_STAGE_CLASS_MAP),
        metadata={"representation": "collapsed_second_codes"},
    )
    labels.metadata["stage_labels"] = [COLLAPSED_STAGE_CLASS_MAP.get(int(c), "unknown") for c in epochs]
    labels.events["respiratory"] = []
    return labels
