from pathlib import Path

import numpy as np

from physionet_sleep.io.ucddb import UCDDBRecordPaths
from physionet_sleep.labels.ucddb import (
    COLLAPSED_STAGE_CLASS_MAP,
    UCDDB_STAGE_CLASS_MAP,
    _collapse_stage_code,
    load_ucddb_labels,
)

# PhysioNet UCDDB annotation coding (https://physionet.org/content/ucddb/1.0.0/).
PHYSIONET_CODES = {0: "wake", 1: "rem", 2: "n1", 3: "n2", 4: "n3", 5: "n3"}


def test_raw_codes_follow_physionet_definition():
    assert UCDDB_STAGE_CLASS_MAP == PHYSIONET_CODES


def test_collapsed_codes_agree_with_raw_names():
    for raw, name in PHYSIONET_CODES.items():
        assert COLLAPSED_STAGE_CLASS_MAP[_collapse_stage_code(raw)] == name
    for raw in (6, 7, 8):
        assert _collapse_stage_code(raw) == -1


def test_loader_expands_epochs_to_seconds(tmp_path: Path):
    stage_file = tmp_path / "ucddb999_stage.txt"
    stage_file.write_text("0\n1\n2\n3\n4\n5\n8\n")
    paths = UCDDBRecordPaths(
        record_id="ucddb999",
        root_dir=tmp_path,
        edf_path=tmp_path / "ucddb999.rec",
        stage_path=stage_file,
        respevt_path=None,
        rec_path=None,
    )

    labels = load_ucddb_labels(paths)

    seconds = labels.tracks["stage_seconds_collapsed"].values
    names = [COLLAPSED_STAGE_CLASS_MAP.get(int(v), "unknown") for v in seconds[::30]]
    assert names == ["wake", "rem", "n1", "n2", "n3", "n3", "unknown"]
    assert np.all(seconds[30:60] == 4)
    assert labels.metadata["stage_labels"][:6] == ["wake", "rem", "n1", "n2", "n3", "n3"]
