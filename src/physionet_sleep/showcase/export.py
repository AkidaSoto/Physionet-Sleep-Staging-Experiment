from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from physionet_sleep.core.types import DatasetRun, Event, SignalSeries
from physionet_sleep.runners.ucddb import run_ucddb_pipeline


APNEA_SIGNAL_SPECS = (
    ("airflow_pressure", "Airflow"),
    ("spo2", "SpO2"),
    ("effort_thoracic", "Thoracic effort"),
    ("effort_abdominal", "Abdominal effort"),
)

APNEA_FEATURE_COLUMNS = (
    "respiration.respiration_rate_bpm",
    "spo2_drop.spo2_drop3_mask",
    "spo2_drop.spo2_drop4_mask",
    "effort_snore_features.thoracic_effort_amplitude",
    "effort_snore_features.abdominal_effort_amplitude",
    "airflow_morphology.resp_band_relative_power",
    "paradoxical_breathing.thor_abd_corr",
    "paradoxical_breathing.paradox_mask",
    "apnea_events.apnea_event_vector",
)

STAGE_SIGNAL_SPECS = (
    ("eeg_c3a2", "EEG"),
    ("eog", "EOG"),
    ("emg", "EMG"),
)

STAGING_FEATURE_COLUMNS = (
    "spindle_features.splindex",
    "slow_wave_features.slow_wave_index",
    "eog_rem_features.rem_power_epoch",
    "eog_rem_features.sem_power_epoch",
    "emg_tone_features.emg_tone",
    "emg_tone_features.emg_suppression_mask",
    "eeg_arousal_features.eeg_arousal_signal",
)


@dataclass(slots=True)
class ShowcaseExportConfig:
    record_id: str
    root_dir: str | Path = "data/raw/ucddb"
    engine: str = "direct"
    apnea_event_type: str = "APNEA-O"
    apnea_event_index: int = 0
    apnea_example_count: int = 3
    apnea_manual_starts_sec: tuple[float, ...] = ()
    apnea_duration_sec: float = 180.0
    apnea_context_before_sec: float = 60.0
    stage_excerpt_seconds: float = 90.0
    stage_examples_per_class: int = 2
    max_points: int = 1200
    max_feature_rows: int = 120


def export_showcase_payload(config: ShowcaseExportConfig) -> dict[str, Any]:
    dataset_run = run_ucddb_pipeline(
        config.record_id,
        root_dir=config.root_dir,
        diagnostics=[],
        experiments=[],
        engine=config.engine,
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "ucddb",
        "record_id": config.record_id,
        "apnea": _build_apnea_examples(
            dataset_run,
            event_type=config.apnea_event_type,
            event_index=config.apnea_event_index,
            example_count=config.apnea_example_count,
            manual_starts_sec=config.apnea_manual_starts_sec,
            duration_sec=config.apnea_duration_sec,
            context_before_sec=config.apnea_context_before_sec,
            max_points=config.max_points,
            max_feature_rows=config.max_feature_rows,
        ),
        "staging": _build_staging_examples(
            dataset_run,
            stage_excerpt_seconds=config.stage_excerpt_seconds,
            examples_per_class=config.stage_examples_per_class,
            max_points=config.max_points,
            max_feature_rows=config.max_feature_rows,
        ),
    }


def write_showcase_payload(config: ShowcaseExportConfig, output_path: str | Path) -> Path:
    payload = export_showcase_payload(config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_to_builtin(payload), indent=2))
    return output


def _build_apnea_examples(
    dataset_run: DatasetRun,
    *,
    event_type: str,
    event_index: int,
    example_count: int,
    manual_starts_sec: tuple[float, ...],
    duration_sec: float,
    context_before_sec: float,
    max_points: int,
    max_feature_rows: int,
) -> dict[str, Any]:
    duration_sec = max(float(duration_sec), 1.0)
    total_duration = _record_duration_seconds(dataset_run)
    anchors = _choose_respiratory_events(
        dataset_run,
        event_type=event_type,
        event_index=event_index,
        example_count=example_count,
    )

    examples = []
    starts = [float(value) for value in manual_starts_sec]
    if not starts:
      starts = [
          max(float(event.start) - float(context_before_sec), 0.0)
          for event in anchors
      ]
    if not starts:
      starts = [0.0]

    for index, start_sec in enumerate(starts):
        end_sec = min(start_sec + duration_sec, total_duration)
        anchor = anchors[index] if index < len(anchors) else None
        signals = []
        for channel_name, label in APNEA_SIGNAL_SPECS:
            if channel_name not in dataset_run.record.channels:
                continue
            signals.append(
                _slice_signal(
                    signal_id=channel_name,
                    label=label,
                    series=dataset_run.record.get(channel_name),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    max_points=max_points,
                )
            )

        paradox_result = dataset_run.results.get("paradoxical_breathing")
        if paradox_result is not None:
            for signal_name, label in (
                ("thor_abd_corr", "Paradox correlation"),
                ("paradox_mask", "Paradox mask"),
            ):
                series = paradox_result.signals.get(signal_name)
                if series is None:
                    continue
                signals.append(
                    _slice_signal(
                        signal_id=signal_name,
                        label=label,
                        series=series,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        max_points=max_points,
                    )
                )

        tracks = []
        for track_name, label in (
            ("respiratory_event_truth", "Respiratory event truth"),
            ("respiratory_obstruction_truth", "Respiratory obstruction truth"),
            ("apnea_event_prediction_sec", "Predicted apnea event"),
        ):
            values = dataset_run.alignment.tracks.get(track_name)
            if values is None:
                continue
            tracks.append(
                _slice_track(
                    track_id=track_name,
                    label=label,
                    values=np.asarray(values),
                    step_seconds=1.0,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    max_points=max_points,
                )
            )

        examples.append(
            {
                "id": f"apnea-example-{index + 1}",
                "segment": {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                },
                "anchor_event": _serialize_event(anchor, window_start_sec=start_sec) if anchor is not None else None,
                "signals": signals,
                "tracks": tracks,
                "overlapping_events": _events_in_window(dataset_run, start_sec=start_sec, end_sec=end_sec),
                "feature_window": _slice_feature_window(
                    dataset_run,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    columns=APNEA_FEATURE_COLUMNS,
                    max_rows=max_feature_rows,
                ),
            }
        )

    return {
        "default_example_index": 0,
        "examples": examples,
    }


def _build_staging_examples(
    dataset_run: DatasetRun,
    *,
    stage_excerpt_seconds: float,
    examples_per_class: int,
    max_points: int,
    max_feature_rows: int,
) -> dict[str, Any]:
    stage_track = dataset_run.labels.tracks.get("stage_seconds_collapsed")
    if stage_track is None:
        return {"status": "missing_stage_labels", "examples": []}

    epoch_track = dataset_run.labels.tracks.get("stage_epochs_raw")
    if epoch_track is not None:
        hypnogram = _slice_track(
            track_id="hypnogram",
            label="Night hypnogram",
            values=np.asarray(epoch_track.values),
            step_seconds=float(epoch_track.step_seconds),
            start_sec=0.0,
            end_sec=float(epoch_track.values.size * epoch_track.step_seconds),
            max_points=max_points,
            class_map=epoch_track.class_map,
        )
    else:
        hypnogram = _slice_track(
            track_id="hypnogram",
            label="Night hypnogram",
            values=np.asarray(stage_track.values),
            step_seconds=float(stage_track.step_seconds),
            start_sec=0.0,
            end_sec=float(stage_track.values.size * stage_track.step_seconds),
            max_points=max_points,
            class_map=stage_track.class_map,
        )

    examples: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    next_index = 0

    for class_id, class_label in sorted(stage_track.class_map.items()):
        if class_id < 0:
            continue
        runs = _top_class_runs(np.asarray(stage_track.values), class_id, examples_per_class)
        for run_index, (run_start, run_end) in enumerate(runs):
            center_sec = (run_start + run_end) / 2.0
            excerpt_start = max(center_sec - (stage_excerpt_seconds / 2.0), 0.0)
            excerpt_end = excerpt_start + stage_excerpt_seconds
            excerpt_signals = []
            for channel_name, label in STAGE_SIGNAL_SPECS:
                if channel_name not in dataset_run.record.channels:
                    continue
                excerpt_signals.append(
                    _slice_signal(
                        signal_id=channel_name,
                        label=label,
                        series=dataset_run.record.get(channel_name),
                        start_sec=excerpt_start,
                        end_sec=excerpt_end,
                        max_points=max_points,
                    )
                )
            example_id = f"stage-{class_label}-{run_index + 1}"
            examples.append(
                {
                    "id": example_id,
                    "class_id": class_id,
                    "class_label": class_label,
                    "run_start_sec": float(run_start),
                    "run_end_sec": float(run_end),
                    "excerpt_start_sec": excerpt_start,
                    "excerpt_end_sec": excerpt_end,
                    "signals": excerpt_signals,
                    "feature_window": _slice_feature_window(
                        dataset_run,
                        start_sec=excerpt_start,
                        end_sec=excerpt_end,
                        columns=STAGING_FEATURE_COLUMNS,
                        max_rows=max_feature_rows,
                    ),
                }
            )
            anchors.append(
                {
                    "example_index": next_index,
                    "class_id": class_id,
                    "class_label": class_label,
                    "time_sec": center_sec,
                }
            )
            next_index += 1

    return {
        "record_duration_sec": _record_duration_seconds(dataset_run),
        "hypnogram": hypnogram,
        "anchors": anchors,
        "examples": examples,
    }


def _slice_feature_window(
    dataset_run: DatasetRun,
    *,
    start_sec: float,
    end_sec: float,
    columns: Iterable[str],
    max_rows: int,
) -> dict[str, Any] | None:
    feature_result = dataset_run.results.get("feature_table")
    if feature_result is None:
        return None
    table = feature_result.artifacts.get("feature_table")
    if not isinstance(table, pd.DataFrame) or table.empty or "time_seconds" not in table.columns:
        return None

    window = table[(table["time_seconds"] >= start_sec) & (table["time_seconds"] <= end_sec)].copy()
    if window.empty:
        return None

    stride = max(int(math.ceil(len(window) / max_rows)), 1) if max_rows > 0 else 1
    window = window.iloc[::stride].reset_index(drop=True)

    exported_columns = []
    for column_name in columns:
        if column_name not in window.columns:
            continue
        series = pd.to_numeric(window[column_name], errors="coerce")
        exported_columns.append(
            {
                "id": column_name,
                "label": _feature_label(column_name),
                "values": [
                    None if not np.isfinite(value) else float(value)
                    for value in series.to_numpy(dtype=float, copy=False)
                ],
            }
        )

    return {
        "time_seconds": [float(value - start_sec) for value in window["time_seconds"].to_list()],
        "absolute_time_seconds": [float(value) for value in window["time_seconds"].to_list()],
        "columns": exported_columns,
    }


def _feature_label(column_name: str) -> str:
    name = column_name.split(".", 1)[-1]
    return name.replace("_", " ")


def _choose_respiratory_events(
    dataset_run: DatasetRun,
    *,
    event_type: str,
    event_index: int,
    example_count: int,
) -> list[Event]:
    events = dataset_run.labels.events.get("respiratory", [])
    matching = [event for event in events if str(event.metadata.get("type_raw", "")).upper() == event_type.upper()]
    if not matching:
        matching = [event for event in events if str(event.metadata.get("type_raw", "")).upper().endswith("-O")]
    if not matching:
        matching = list(events)
    if not matching:
        return []
    start = min(max(int(event_index), 0), len(matching) - 1)
    return matching[start : start + max(int(example_count), 1)]


def _events_in_window(dataset_run: DatasetRun, *, start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    events = dataset_run.labels.events.get("respiratory", [])
    return [
        _serialize_event(event, window_start_sec=start_sec)
        for event in events
        if event.end >= start_sec and event.start <= end_sec
    ]


def _slice_signal(
    *,
    signal_id: str,
    label: str,
    series: SignalSeries,
    start_sec: float,
    end_sec: float,
    max_points: int,
) -> dict[str, Any]:
    values = np.asarray(series.values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D series for showcase export: {signal_id}")
    sample_rate_hz = float(series.sample_rate)
    start_idx = max(int(math.floor(start_sec * sample_rate_hz)), 0)
    end_idx = min(int(math.ceil(end_sec * sample_rate_hz)), values.size)
    segment = values[start_idx:end_idx]
    stride = max(int(math.ceil(segment.size / max_points)), 1) if max_points > 0 else 1
    sampled = segment[::stride]
    relative_time = (np.arange(sampled.size, dtype=float) * stride / sample_rate_hz).tolist()
    return {
        "id": signal_id,
        "label": label,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "effective_sample_rate_hz": sample_rate_hz / stride,
        "time_seconds": relative_time,
        "values": sampled.tolist(),
    }


def _slice_track(
    *,
    track_id: str,
    label: str,
    values: np.ndarray,
    step_seconds: float,
    start_sec: float,
    end_sec: float,
    max_points: int,
    class_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    arr = np.asarray(values)
    start_idx = max(int(math.floor(start_sec / step_seconds)), 0)
    end_idx = min(int(math.ceil(end_sec / step_seconds)), arr.size)
    segment = arr[start_idx:end_idx]
    stride = max(int(math.ceil(segment.size / max_points)), 1) if max_points > 0 else 1
    sampled = segment[::stride]
    relative_time = (np.arange(sampled.size, dtype=float) * stride * step_seconds).tolist()
    payload = {
        "id": track_id,
        "label": label,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "step_seconds": step_seconds * stride,
        "time_seconds": relative_time,
        "values": sampled.tolist(),
    }
    if class_map:
        payload["class_map"] = class_map
    return payload


def _top_class_runs(values: np.ndarray, class_id: int, count: int) -> list[tuple[int, int]]:
    match = np.asarray(values) == class_id
    runs: list[tuple[int, int]] = []
    run_start = None
    for index, active in enumerate(match.tolist()):
        if active and run_start is None:
            run_start = index
        elif not active and run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, int(match.size)))
    runs.sort(key=lambda item: item[1] - item[0], reverse=True)
    return runs[: max(int(count), 0)]


def _serialize_event(event: Event, *, window_start_sec: float) -> dict[str, Any]:
    return {
        "label": event.label,
        "kind": event.kind,
        "absolute_start_sec": event.start,
        "absolute_end_sec": event.end,
        "relative_start_sec": float(event.start - window_start_sec),
        "relative_end_sec": float(event.end - window_start_sec),
        "metadata": _to_builtin(event.metadata),
    }


def _record_duration_seconds(dataset_run: DatasetRun) -> float:
    if not dataset_run.record.channels:
        return 0.0
    return max(series.duration_seconds for series in dataset_run.record.channels.values())


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
