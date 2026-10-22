"""Sleep stages and respiratory events from CinC 2018 `.arousal` annotations.

Stage labels (W, N1, N2, N3, R) are written at transitions and hold until the next stage label; samples before the
first stage label, and any other stage code, are unscored. Events are "(name" ... "name)" pairs.
"""
from __future__ import annotations

import numpy as np

from physionet_sleep.core.types import Event, LabelTrack, RecordLabels
from physionet_sleep.io.cinc2018 import CinC2018RecordPaths, record_length_seconds
from physionet_sleep.labels.ucddb import COLLAPSED_STAGE_CLASS_MAP

STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
# UCDDB-style type codes so the shared respiratory-truth tracks interpret them the same way.
RESPIRATORY_TYPES = {
    "resp_obstructiveapnea": ("apnea_obstructive", "APNEA-O"),
    "resp_centralapnea": ("apnea_central", "APNEA-C"),
    "resp_mixedapnea": ("apnea_mixed", "APNEA-M"),
    "resp_hypopnea": ("hypopnea", "HYP"),
}
EPOCH_SECONDS = 30


def load_cinc2018_annotations(paths: CinC2018RecordPaths) -> tuple[list[tuple[float, str]], list[tuple[float, float, str]], float]:
    """Return (stage changes [(t, code)], events [(start, end, name)], record length in seconds)."""
    import wfdb

    if paths.annotation_path is None:
        raise FileNotFoundError(f"Annotations not downloaded for {paths.record_id}")
    ann = wfdb.rdann(str(paths.annotation_path.with_suffix("")), "arousal")
    fs = float(ann.fs or 200.0)
    stages: list[tuple[float, str]] = []
    events: list[tuple[float, float, str]] = []
    open_events: dict[str, float] = {}
    for sample, note in zip(ann.sample, ann.aux_note):
        note = (note or "").strip("\x00 ")
        t = float(sample) / fs
        if note.startswith("("):
            open_events[note[1:]] = t
        elif note.endswith(")"):
            name = note[:-1]
            if name in open_events:
                events.append((open_events.pop(name), t, name))
        elif note:
            stages.append((t, note))
    return stages, events, record_length_seconds(paths)


def stage_epochs(stage_changes: list[tuple[float, str]], length_seconds: float) -> np.ndarray:
    """Collapsed stage code per 30-s epoch from record start; -1 where unscored."""
    n_epochs = int(length_seconds // EPOCH_SECONDS)
    codes = np.full(n_epochs, -1, dtype=int)
    for index, (start, note) in enumerate(stage_changes):
        end = stage_changes[index + 1][0] if index + 1 < len(stage_changes) else length_seconds
        first = int(np.ceil(start / EPOCH_SECONDS - 1e-9))
        last = int(np.floor(end / EPOCH_SECONDS + 1e-9))
        codes[first:min(last, n_epochs)] = STAGE_CODES.get(note, -1)
    return codes


def load_cinc2018_labels(paths: CinC2018RecordPaths) -> RecordLabels:
    stage_changes, events, length = load_cinc2018_annotations(paths)
    epochs = stage_epochs(stage_changes, length)
    labels = RecordLabels(metadata={"dataset": "cinc2018", "record_id": paths.record_id})
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
    labels.events["respiratory"] = [
        Event(
            start=int(np.floor(start)),
            end=int(np.ceil(end)) - 1,
            label=RESPIRATORY_TYPES[name][0],
            kind="period",
            metadata={"type_raw": RESPIRATORY_TYPES[name][1], "duration_seconds": end - start},
        )
        for start, end, name in events
        if name in RESPIRATORY_TYPES
    ]
    return labels


def subject_summary(paths: CinC2018RecordPaths) -> dict:
    """AHI and stage composition from annotations only (no signals needed)."""
    stage_changes, events, length = load_cinc2018_annotations(paths)
    epochs = stage_epochs(stage_changes, length)
    sleep_hours = float(np.isin(epochs, [1, 2, 3, 4]).sum() * EPOCH_SECONDS / 3600.0)
    counts = {name: sum(1 for *_, n in events if n == name) for name in RESPIRATORY_TYPES}
    apneas = counts["resp_obstructiveapnea"] + counts["resp_centralapnea"] + counts["resp_mixedapnea"]
    return {
        "record_id": paths.record_id,
        "hours_recorded": length / 3600.0,
        "sleep_hours": sleep_hours,
        "ahi": (apneas + counts["resp_hypopnea"]) / sleep_hours if sleep_hours > 0 else np.nan,
        "central_fraction": counts["resp_centralapnea"] / apneas if apneas else 0.0,
        **{f"epochs_{name}": int((epochs == code).sum()) for name, code in STAGE_CODES.items()},
        "epochs_unscored": int((epochs == -1).sum()),
    }
