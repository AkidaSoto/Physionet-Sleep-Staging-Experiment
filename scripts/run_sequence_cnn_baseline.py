from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.runners import build_modeling_context, run_sequence_baseline_job


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outcome = run_sequence_baseline_job(
        build_modeling_context(REPO_ROOT),
        context_radius=args.context_radius,
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        random_state=args.random_state,
    )
    if not outcome.summary.empty:
        print(outcome.summary.to_string(index=False))
    else:
        print("No sequence CNN results produced.")
    print(outcome.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
