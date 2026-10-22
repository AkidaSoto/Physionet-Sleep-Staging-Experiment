from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from physionet_sleep.runners import (
    build_modeling_context,
    run_reference_diagnostics_job,
    run_sequence_baseline_job,
    run_sequence_epoch_job,
    run_stage_expert_temporal_job,
    run_stage_signal_expert_temporal_job,
)


def _add_outer_cv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outer-validation", choices=("kfold", "loso"), default="kfold")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=7)


def _add_inner_cv_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inner-validation", choices=("single_subject", "none"), default="single_subject")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("sequence-baseline")
    baseline.add_argument("--context-radius", type=int, default=2)
    _add_outer_cv_args(baseline)

    epoch = subparsers.add_parser("sequence-epoch")
    epoch.add_argument("--history-steps", type=int, default=10)
    epoch.add_argument("--current-steps", type=int, default=5)
    epoch.add_argument("--future-steps", type=int, default=5)
    epoch.add_argument("--model-arch", choices=("simple_cnn", "dilated_tcn"), default="dilated_tcn")
    epoch.add_argument("--tcn-levels", type=int, default=4)
    _add_outer_cv_args(epoch)

    expert = subparsers.add_parser("stage-expert")
    expert.add_argument("--expert-model-family", choices=("cnn", "linear", "tree", "ensemble"), default="cnn")
    expert.add_argument("--stage2-model-family", choices=("linear", "tree", "ensemble", "tcn"), default="linear")
    expert.add_argument("--context-radius", type=int, default=2)
    expert.add_argument("--current-steps", type=int, default=5)
    expert.add_argument("--epochs", type=int, default=40)
    expert.add_argument("--batch-size", type=int, default=256)
    expert.add_argument("--learning-rate", type=float, default=1e-3)
    expert.add_argument("--hidden-channels", type=int, default=32)
    expert.add_argument("--kernel-size", type=int, default=3)
    expert.add_argument("--dropout", type=float, default=0.1)
    expert.add_argument("--min-epochs", type=int, default=10)
    expert.add_argument("--patience", type=int, default=8)
    expert.add_argument("--output-dir", default="artifacts/stage_expert_temporal")
    _add_outer_cv_args(expert)
    _add_inner_cv_arg(expert)

    signal = subparsers.add_parser("signal-expert")
    signal.add_argument("--stage1-model-arch", choices=("simple_cnn", "dilated_tcn", "multi_branch_tcn"), default="dilated_tcn")
    signal.add_argument("--stage2-model-family", choices=("linear", "tree", "ensemble", "tcn"), default="tcn")
    signal.add_argument("--window-points", type=int, default=64)
    signal.add_argument("--context-radius", type=int, default=2)
    signal.add_argument("--epochs", type=int, default=40)
    signal.add_argument("--batch-size", type=int, default=256)
    signal.add_argument("--learning-rate", type=float, default=1e-3)
    signal.add_argument("--hidden-channels", type=int, default=32)
    signal.add_argument("--kernel-size", type=int, default=5)
    signal.add_argument("--dropout", type=float, default=0.1)
    signal.add_argument("--min-epochs", type=int, default=8)
    signal.add_argument("--patience", type=int, default=6)
    signal.add_argument("--tcn-levels", type=int, default=3)
    signal.add_argument("--output-dir", default="artifacts/stage_signal_expert_temporal")
    _add_outer_cv_args(signal)
    _add_inner_cv_arg(signal)

    subparsers.add_parser("reference-diagnostics")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    context = build_modeling_context(REPO_ROOT)

    if args.command == "sequence-baseline":
        outcome = run_sequence_baseline_job(
            context,
            context_radius=args.context_radius,
            outer_validation=args.outer_validation,
            outer_folds=args.outer_folds,
            random_state=args.random_state,
        )
    elif args.command == "sequence-epoch":
        outcome = run_sequence_epoch_job(
            context,
            history_steps=args.history_steps,
            current_steps=args.current_steps,
            future_steps=args.future_steps,
            model_arch=args.model_arch,
            tcn_levels=args.tcn_levels,
            outer_validation=args.outer_validation,
            outer_folds=args.outer_folds,
            random_state=args.random_state,
        )
    elif args.command == "stage-expert":
        outcome = run_stage_expert_temporal_job(
            context,
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
    elif args.command == "signal-expert":
        outcome = run_stage_signal_expert_temporal_job(
            context,
            stage1_model_arch=args.stage1_model_arch,
            stage2_model_family=args.stage2_model_family,
            outer_validation=args.outer_validation,
            outer_folds=args.outer_folds,
            inner_validation=args.inner_validation,
            window_points=args.window_points,
            context_radius=args.context_radius,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            min_epochs=args.min_epochs,
            patience=args.patience,
            tcn_levels=args.tcn_levels,
            random_state=args.random_state,
            output_dir=args.output_dir,
        )
    else:
        outcome = run_reference_diagnostics_job(context)

    if not outcome.summary.empty:
        print(outcome.summary.to_string(index=False))
    print(outcome.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
