from __future__ import annotations

import numpy as np

from physionet_sleep.core.types import AlgorithmResult, Event, LabelAlignment, RecordLabels, SignalRecord


def consolidate_algorithm_outputs(results: dict[str, AlgorithmResult]) -> dict[str, object]:
    return {
        "features": {name: result.features for name, result in results.items()},
        "events": {name: result.events for name, result in results.items()},
        "signals": {name: list(result.signals.keys()) for name, result in results.items()},
    }


def align_ucddb_labels_to_results(
    record: SignalRecord,
    labels: RecordLabels,
    results: dict[str, AlgorithmResult],
) -> LabelAlignment:
    alignment = LabelAlignment(metadata={"record_id": record.metadata.get("record_id")})

    if "stage_seconds_collapsed" in labels.tracks:
        alignment.tracks["stage_seconds"] = labels.tracks["stage_seconds_collapsed"].values.copy()

    if "respiratory" in labels.events:
        total_seconds = _infer_total_seconds(record)
        truth_vector = np.zeros(total_seconds, dtype=int)
        apnea_truth = np.zeros(total_seconds, dtype=int)
        obstruction_truth = np.zeros(total_seconds, dtype=int)
        for event in labels.events["respiratory"]:
            start = max(int(event.start), 0)
            end = min(int(event.end) + 1, total_seconds)
            truth_vector[start:end] = 1
            type_raw = str(event.metadata.get("type_raw", "")).upper()
            if type_raw.startswith("HYP"):
                apnea_truth[start:end] = 1
            elif type_raw.startswith("APNEA"):
                apnea_truth[start:end] = 2
            if type_raw.endswith("-C"):
                obstruction_truth[start:end] = 1
            elif type_raw.endswith("-O"):
                obstruction_truth[start:end] = 2
            elif type_raw.endswith("-M"):
                obstruction_truth[start:end] = 3
        alignment.tracks["respiratory_event_truth"] = truth_vector
        alignment.tracks["respiratory_apnea_truth"] = apnea_truth
        alignment.tracks["respiratory_obstruction_truth"] = obstruction_truth
        alignment.events["respiratory_truth"] = labels.events["respiratory"]

    apnea_result = results.get("apnea_events")
    if apnea_result is not None and "apnea_event_vector" in apnea_result.signals:
        signal = apnea_result.signals["apnea_event_vector"]
        fs = float(signal.sample_rate)
        values = np.asarray(signal.values)
        alignment.tracks["apnea_event_prediction_sec"] = _resample_to_seconds(values, fs)
        alignment.events["apnea_prediction"] = apnea_result.events.get("apnea", [])
        alignment.events["hypopnea_prediction"] = apnea_result.events.get("hypopnea", [])

    alignment.metadata["consolidated"] = consolidate_algorithm_outputs(results)
    return alignment


def _infer_total_seconds(record: SignalRecord) -> int:
    if not record.channels:
        return 0
    max_duration = max(series.duration_seconds for series in record.channels.values())
    return int(np.ceil(max_duration))


def _resample_to_seconds(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    step = max(int(round(sample_rate_hz)), 1)
    usable = (values.shape[0] // step) * step
    if usable == 0:
        return np.array([], dtype=int)
    reshaped = values[:usable].reshape(-1, step)
    return np.max(reshaped, axis=1).astype(int)
