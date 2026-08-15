from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from physionet_sleep.core.types import SignalRecord
from physionet_sleep.io.edf import load_edf_record


@dataclass(slots=True)
class UCDDBRecordPaths:
    record_id: str
    root_dir: Path
    edf_path: Path
    stage_path: Path | None
    respevt_path: Path | None
    rec_path: Path | None


def discover_ucddb_records(root_dir: str | Path) -> list[UCDDBRecordPaths]:
    root = Path(root_dir)
    paths: list[UCDDBRecordPaths] = []
    record_ids = {
        path.name.replace("_lifecard.edf", "") for path in root.glob("ucddb*_lifecard.edf")
    }
    record_ids.update(path.stem for path in root.glob("ucddb*.rec"))
    for record_id in sorted(record_ids):
        paths.append(
            UCDDBRecordPaths(
                record_id=record_id,
                root_dir=root,
                edf_path=root / f"{record_id}_lifecard.edf",
                stage_path=_optional_path(root / f"{record_id}_stage.txt"),
                respevt_path=_optional_path(root / f"{record_id}_respevt.txt"),
                rec_path=_optional_path(root / f"{record_id}.rec"),
            )
        )
    return paths


def resolve_ucddb_record(root_dir: str | Path, record_id: str) -> UCDDBRecordPaths:
    root = Path(root_dir)
    normalized = record_id.replace("_lifecard.edf", "").replace(".edf", "")
    edf_path = root / f"{normalized}_lifecard.edf"
    rec_path = _optional_path(root / f"{normalized}.rec")
    if not edf_path.exists() and rec_path is None:
        raise FileNotFoundError(f"UCDDB waveform file not found for: {normalized}")
    return UCDDBRecordPaths(
        record_id=normalized,
        root_dir=root,
        edf_path=edf_path,
        stage_path=_optional_path(root / f"{normalized}_stage.txt"),
        respevt_path=_optional_path(root / f"{normalized}_respevt.txt"),
        rec_path=rec_path,
    )


def load_ucddb_waveforms(paths: UCDDBRecordPaths) -> SignalRecord:
    source_path = paths.rec_path if paths.rec_path is not None else paths.edf_path
    record = load_edf_record(str(source_path))
    record.metadata.update(
        {
            "dataset": "ucddb",
            "record_id": paths.record_id,
            "root_dir": str(paths.root_dir),
            "waveform_path": str(source_path),
            "lifecard_edf_path": str(paths.edf_path) if paths.edf_path.exists() else None,
            "rec_path": str(paths.rec_path) if paths.rec_path else None,
        }
    )
    return record


def _optional_path(path: Path) -> Path | None:
    return path if path.exists() else None
