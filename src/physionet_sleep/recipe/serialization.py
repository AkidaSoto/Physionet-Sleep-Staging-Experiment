from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physionet_sleep.core.types import AlgorithmResult, Event, SignalSeries


def save_algorithm_result(base_dir: str, result: AlgorithmResult) -> None:
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    manifest: dict[str, Any] = {
        "type": "AlgorithmResult",
        "name": result.name,
        "signals": [],
        "events": [],
        "artifacts": [],
    }

    for signal_name, signal in result.signals.items():
        safe_name = _safe_name(signal_name)
        rel_path = os.path.join("signals", f"{safe_name}.parquet")
        meta_rel_path = os.path.join("signals", f"{safe_name}.meta.json")
        _write_signal_series(os.path.join(base_dir, rel_path), signal)
        _write_json(
            os.path.join(base_dir, meta_rel_path),
            {
                "name": signal_name,
                "sample_rate": float(signal.sample_rate),
                "units": signal.units,
                "metadata": _encode_jsonable(signal.metadata),
            },
        )
        manifest["signals"].append(
            {
                "name": signal_name,
                "path": rel_path,
                "meta_path": meta_rel_path,
            }
        )

    for event_name, events in result.events.items():
        safe_name = _safe_name(event_name)
        rel_path = os.path.join("events", f"{safe_name}.parquet")
        _write_event_table(os.path.join(base_dir, rel_path), events)
        manifest["events"].append({"name": event_name, "path": rel_path})

    _write_json(os.path.join(base_dir, "features.json"), _encode_jsonable(result.features))
    _write_json(os.path.join(base_dir, "metadata.json"), _encode_jsonable(result.metadata))

    for artifact_name, artifact in result.artifacts.items():
        safe_name = _safe_name(artifact_name)
        if isinstance(artifact, pd.DataFrame):
            rel_path = os.path.join("artifacts", f"{safe_name}.parquet")
            _ensure_parent(os.path.join(base_dir, rel_path))
            artifact.to_parquet(os.path.join(base_dir, rel_path), index=False)
            manifest["artifacts"].append(
                {"name": artifact_name, "kind": "dataframe", "path": rel_path}
            )
        elif isinstance(artifact, np.ndarray):
            rel_path = os.path.join("artifacts", f"{safe_name}.parquet")
            meta_rel_path = os.path.join("artifacts", f"{safe_name}.meta.json")
            _write_array_table(os.path.join(base_dir, rel_path), artifact)
            _write_json(
                os.path.join(base_dir, meta_rel_path),
                {"shape": list(np.asarray(artifact).shape), "dtype": str(np.asarray(artifact).dtype)},
            )
            manifest["artifacts"].append(
                {
                    "name": artifact_name,
                    "kind": "ndarray",
                    "path": rel_path,
                    "meta_path": meta_rel_path,
                }
            )
        else:
            rel_path = os.path.join("artifacts", f"{safe_name}.json")
            _write_json(os.path.join(base_dir, rel_path), _encode_jsonable(artifact))
            manifest["artifacts"].append({"name": artifact_name, "kind": "json", "path": rel_path})

    _write_json(os.path.join(base_dir, "manifest.json"), manifest)


def load_algorithm_result(base_dir: str) -> AlgorithmResult | None:
    manifest_path = os.path.join(base_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("type") != "AlgorithmResult":
        return None

    signals: dict[str, SignalSeries] = {}
    for entry in manifest.get("signals", []):
        meta = _read_json(os.path.join(base_dir, entry["meta_path"]))
        signal = _read_signal_series(
            os.path.join(base_dir, entry["path"]),
            sample_rate=float(meta["sample_rate"]),
            units=meta.get("units"),
            metadata=_decode_jsonable(meta.get("metadata")),
        )
        signals[entry["name"]] = signal

    events: dict[str, list[Event]] = {}
    for entry in manifest.get("events", []):
        events[entry["name"]] = _read_event_table(os.path.join(base_dir, entry["path"]))

    artifacts: dict[str, Any] = {}
    for entry in manifest.get("artifacts", []):
        kind = entry.get("kind")
        artifact_path = os.path.join(base_dir, entry["path"])
        if kind == "dataframe":
            artifacts[entry["name"]] = pd.read_parquet(artifact_path)
        elif kind == "ndarray":
            meta = _read_json(os.path.join(base_dir, entry["meta_path"]))
            artifacts[entry["name"]] = _read_array_table(
                artifact_path,
                shape=tuple(int(v) for v in meta.get("shape", [])),
            )
        else:
            artifacts[entry["name"]] = _decode_jsonable(_read_json(artifact_path))

    return AlgorithmResult(
        name=str(manifest.get("name", "")),
        signals=signals,
        events=events,
        features=_decode_jsonable(_read_json(os.path.join(base_dir, "features.json"))),
        artifacts=artifacts,
        metadata=_decode_jsonable(_read_json(os.path.join(base_dir, "metadata.json"))),
    )


def _write_signal_series(path: str, signal: SignalSeries) -> None:
    values = np.asarray(signal.values)
    _write_array_table(path, values)


def _read_signal_series(
    path: str,
    *,
    sample_rate: float,
    units: str | None,
    metadata: dict[str, Any] | None,
) -> SignalSeries:
    values = _read_array_table(path)
    return SignalSeries(
        values=values,
        sample_rate=sample_rate,
        units=units,
        metadata=metadata or {},
    )


def _write_event_table(path: str, events: list[Event]) -> None:
    _ensure_parent(path)
    rows = [
        {
            "start": int(event.start),
            "end": int(event.end),
            "label": event.label,
            "kind": event.kind,
            "metadata_json": json.dumps(_encode_jsonable(event.metadata)),
        }
        for event in events
    ]
    pd.DataFrame(rows, columns=["start", "end", "label", "kind", "metadata_json"]).to_parquet(
        path,
        index=False,
    )


def _read_event_table(path: str) -> list[Event]:
    if not os.path.isfile(path):
        return []
    table = pd.read_parquet(path)
    events: list[Event] = []
    for row in table.to_dict(orient="records"):
        metadata = _decode_jsonable(json.loads(row.get("metadata_json") or "{}"))
        events.append(
            Event(
                start=int(row["start"]),
                end=int(row["end"]),
                label=str(row["label"]),
                kind=str(row.get("kind", "period")),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return events


def _write_array_table(path: str, values: np.ndarray) -> None:
    _ensure_parent(path)
    arr = np.asarray(values)
    if arr.ndim == 1:
        df = pd.DataFrame({"value": arr})
    else:
        flat = arr.reshape(arr.shape[0], -1)
        df = pd.DataFrame(flat, columns=[f"value_{i}" for i in range(flat.shape[1])])
    df.to_parquet(path, index=False)


def _read_array_table(path: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    frame = pd.read_parquet(path)
    cols = [col for col in frame.columns if col == "value" or col.startswith("value_")]
    if not cols:
        return np.array([], dtype=float)
    values = frame[cols].to_numpy()
    if len(cols) == 1:
        arr = values[:, 0]
    else:
        arr = values
    if shape:
        return np.asarray(arr).reshape(shape)
    return np.asarray(arr)


def _write_json(path: str, value: Any) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", value)
    return cleaned.strip("._") or "item"


def _encode_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _encode_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_encode_jsonable(v) for v in value]}
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return {"__kind__": "path", "value": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"__kind__": "repr", "value": repr(value)}


def _decode_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind == "tuple":
            return tuple(_decode_jsonable(v) for v in value.get("items", []))
        if kind == "ndarray":
            return np.asarray(value.get("data", []), dtype=value.get("dtype"))
        if kind == "path":
            return Path(str(value.get("value", "")))
        if kind == "repr":
            return value.get("value")
        return {k: _decode_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_jsonable(v) for v in value]
    return value
