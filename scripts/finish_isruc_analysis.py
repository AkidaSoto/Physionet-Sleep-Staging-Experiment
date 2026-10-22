"""Wait for complete features, run the existing guarded external analysis, and refresh the dashboard."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    deadline = time.monotonic() + 3 * 3600
    status_path = ROOT / "showcase/content/generated/isruc-status.json"
    while True:
        subprocess.run([sys.executable, "scripts/export_isruc_status.py"], cwd=ROOT, check=True)
        status = json.loads(status_path.read_text())
        if status["final_complete"] or status["processed"] == 100:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("Timed out waiting for complete processing; no external scoring performed.")
        time.sleep(30)
    if not status["final_complete"]:
        # Existing script checks all 100 nights and refuses to repeat a recorded look.
        subprocess.run([sys.executable, "-u", "scripts/run_window_transfer_isruc.py", "--skip-cv"],
                       cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/export_isruc_status.py"], cwd=ROOT, check=True)
    env = {**os.environ, "NEXT_DIST_DIR": ".next-production"}
    subprocess.run(["npm.cmd", "run", "build"], cwd=ROOT / "showcase", check=True, env=env)


if __name__ == "__main__":
    main()
