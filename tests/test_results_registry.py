"""Guards for the results registry (scripts/build_results_registry.py).

1. The dashboard holds no hand-typed results and every registry id it references exists
   (showcase/scripts/check-results.mjs, which also runs before `npm run build`).
2. Measured entries still match their artifacts (checked when the artifact is present; artifacts are not in git).
3. Derived entries equal the arithmetic on the entries they come from.
4. Recorded and protocol entries appear verbatim in their cited source file, or in their row of
   docs/experiment-history.md when that file is not present; literature entries carry a reference number.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "showcase" / "content" / "generated" / "results-registry.json"


def _entries() -> dict[str, dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["entries"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_results_registry", ROOT / "scripts" / "build_results_registry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_dashboard_has_no_hand_typed_results():
    run = subprocess.run(["node", "scripts/check-results.mjs"], cwd=ROOT / "showcase", capture_output=True, text=True)
    assert run.returncode == 0, run.stderr or run.stdout


def test_measured_entries_match_artifacts():
    builder, entries, checked = _builder(), _entries(), 0
    for source in builder.MEASURED:
        if source.path.endswith(".npz") or not (ROOT / source.path).exists():
            continue  # the no-EEG ensemble is recomputed from predictions by the builder itself (slow)
        for entry_id, value in builder.read_source(source).items():
            assert entries[entry_id]["value"] == pytest.approx(value, abs=1e-12), entry_id
            checked += 1
    if not checked:
        pytest.skip("no artifacts present")


def test_registry_files_unchanged_since_build():
    files = json.loads(REGISTRY.read_text(encoding="utf-8"))["files"]
    builder = _builder()
    stale = [name for name, digest in files.items() if (ROOT / name).exists() and builder.sha256(ROOT / name) != digest]
    assert not stale, "sources changed since the registry was built; rerun scripts/build_results_registry.py:\n" + "\n".join(stale)


def test_derived_entries_recompute():
    entries = _entries()
    values = {k: e["value"] for k, e in entries.items()}
    for entry_id, (value, inputs, _) in _builder().derive(values).items():
        assert entries[entry_id]["kind"] == "derived" and entries[entry_id]["from"] == inputs
        assert entries[entry_id]["value"] == pytest.approx(value, abs=1e-12), entry_id


def _history_rows() -> dict[int, str]:
    rows = {}
    for line in (ROOT / "docs" / "experiment-history.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if m:
            rows[int(m.group(1))] = line
    return rows


def test_recorded_entries_match_their_source():
    rows, bad = _history_rows(), []
    for entry_id, entry in _entries().items():
        if entry["kind"] == "literature":
            if not isinstance(entry.get("cite"), int):
                bad.append(f"{entry_id}: literature without a reference number")
            continue
        if entry["kind"] not in ("recorded", "protocol"):
            continue
        shown = f"{entry['value']:.{entry['digits']}f}"
        source = ROOT / entry["source"]
        if source.exists() and source.name != "experiment-history.md":
            text, where = source.read_text(encoding="utf-8"), entry["source"]
        elif source.name == "experiment-history.md":
            text, where = source.read_text(encoding="utf-8") if entry_id.startswith("lit.") else rows.get(entry["exp"], ""), f"experiment #{entry['exp']}"
        else:
            text, where = rows.get(entry["exp"], ""), f"experiment #{entry['exp']} (source file absent)"
        if shown not in text.replace("−", "-"):
            bad.append(f"{entry_id} = {shown} not found in {where}")
    assert not bad, "\n".join(bad)
