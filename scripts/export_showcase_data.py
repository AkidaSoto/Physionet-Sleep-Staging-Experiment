from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export real UCDDB-driven showcase data for the site.")
    parser.add_argument("--record-id", required=True, help="UCDDB record id, for example ucddb023")
    parser.add_argument(
        "--output",
        default="artifacts/showcase/site-data.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--root-dir",
        default="data/raw/ucddb",
        help="Root directory containing raw UCDDB files",
    )
    parser.add_argument(
        "--engine",
        default="direct",
        choices=("direct", "recipe"),
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--apnea-event-type",
        default="APNEA-O",
        help="Respiratory event type to anchor the apnea strip, for example APNEA-O or HYP-O",
    )
    parser.add_argument(
        "--apnea-event-index",
        type=int,
        default=0,
        help="Which matching event to use when multiple exist",
    )
    parser.add_argument(
        "--apnea-start-sec",
        type=float,
        action="append",
        default=None,
        help="Manual segment start time in seconds; repeat to export multiple fixed examples",
    )
    parser.add_argument(
        "--apnea-example-count",
        type=int,
        default=3,
        help="How many apnea examples to export when using automatic event selection",
    )
    parser.add_argument(
        "--apnea-duration-sec",
        type=float,
        default=180.0,
        help="Duration of the exported apnea strip",
    )
    parser.add_argument(
        "--apnea-context-before-sec",
        type=float,
        default=60.0,
        help="How much context to keep before the chosen apnea event when auto-selecting",
    )
    parser.add_argument(
        "--stage-excerpt-seconds",
        type=float,
        default=90.0,
        help="Duration of each per-stage example excerpt",
    )
    parser.add_argument(
        "--stage-examples-per-class",
        type=int,
        default=2,
        help="How many examples to export for each sleep stage class",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1200,
        help="Maximum sampled points per exported series",
    )
    parser.add_argument(
        "--max-feature-rows",
        type=int,
        default=120,
        help="Maximum rows to keep in each exported feature window",
    )
    return parser


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.showcase.export import ShowcaseExportConfig, write_showcase_payload

    args = build_parser().parse_args()
    output_path = repo_root / args.output
    config = ShowcaseExportConfig(
        record_id=args.record_id,
        root_dir=repo_root / args.root_dir,
        engine=args.engine,
        apnea_event_type=args.apnea_event_type,
        apnea_event_index=args.apnea_event_index,
        apnea_example_count=args.apnea_example_count,
        apnea_manual_starts_sec=tuple(args.apnea_start_sec or []),
        apnea_duration_sec=args.apnea_duration_sec,
        apnea_context_before_sec=args.apnea_context_before_sec,
        stage_excerpt_seconds=args.stage_excerpt_seconds,
        stage_examples_per_class=args.stage_examples_per_class,
        max_points=args.max_points,
        max_feature_rows=args.max_feature_rows,
    )
    written = write_showcase_payload(config, output_path)
    print(f"wrote_showcase_data={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
