from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.runners import build_modeling_context, run_reference_diagnostics_job


def main() -> int:
    outcome = run_reference_diagnostics_job(build_modeling_context(REPO_ROOT))
    if not outcome.summary.empty:
        print(outcome.summary.to_string(index=False))
    print(outcome.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
