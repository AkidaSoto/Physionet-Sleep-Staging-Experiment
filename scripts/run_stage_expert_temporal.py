from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.runners import build_modeling_context, run_stage_expert_temporal_job


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-model-family", choices=("cnn", "linear", "tree", "ensemble"), default="cnn")
    parser.add_argument("--stage2-model-family", choices=("linear", "tree", "ensemble", "tcn"), default="linear")
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-validation", choices=("single_subject", "none"), default="single_subject")
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--current-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/stage_expert_temporal")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outcome = run_stage_expert_temporal_job(
        build_modeling_context(REPO_ROOT),
        expert_model_family=args.expert_model_family,
        stage2_model_family=args.stage2_model_family,
        outer_validation=args.outer_validation,
        outer_folds=args.outer_folds,
        inner_validation=args.inner_validation,
        context_radius=args.context_radius,
        current_steps=args.current_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        min_epochs=args.min_epochs,
        patience=args.patience,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
    if not outcome.summary.empty:
        print(outcome.summary.to_string(index=False))
    else:
        print("No stage expert temporal results produced.")
    print(outcome.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
