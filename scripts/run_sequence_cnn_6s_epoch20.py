from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.runners import build_modeling_context, run_sequence_epoch_job


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--current-steps", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=5)
    parser.add_argument("--model-arch", choices=("simple_cnn", "dilated_tcn"), default="dilated_tcn")
    parser.add_argument("--tcn-levels", type=int, default=4)
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucddb"],
        help="Datasets from physionet_sleep.analysis.datasets to use; non-default runs get their own output folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outcome = run_sequence_epoch_job(
        build_modeling_context(REPO_ROOT, datasets=args.datasets),
        history_steps=args.history_steps,
        current_steps=args.current_steps,
        future_steps=args.future_steps,
        model_arch=args.model_arch,
        tcn_levels=args.tcn_levels,
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        random_state=args.random_state,
    )
    if not outcome.summary.empty:
        print(outcome.summary.to_string(index=False))
    else:
        print("No 6s epoch-aligned CNN results produced.")
    print(outcome.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
