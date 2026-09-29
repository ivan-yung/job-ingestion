"""CLI entry point for the resumable end-to-end ranking pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.config import PipelineConfig
from src.orchestration.pipeline import Pipeline
from src.reporting.report_builder import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the resumable resume-ranking pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--apify-run-id", help="Completed Apify actor run ID")
    source.add_argument("--run-dir", help="Existing local run directory to resume")
    parser.add_argument("--resume", required=True, help="Path to the candidate resume")
    parser.add_argument("--config", default="config.toml", help="Path to TOML pipeline config")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse a valid ingestion artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(dotenv_path=Path(".env"))
    run_dir = Path(args.run_dir) if args.run_dir else None
    run_id = run_dir.name if run_dir else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    config = PipelineConfig.from_toml(args.config, require_apify_token=not args.skip_ingest)
    pipeline = Pipeline(config)
    payload = pipeline.run(
        run_id=run_id,
        run_dir=run_dir,
        resume_path=args.resume,
        apify_run_id=args.apify_run_id,
        skip_ingest=args.skip_ingest,
    )
    output_dir = run_dir or Path(config.runs_dir) / run_id
    build_report(payload, output_dir, config)
    print(f"Report: {output_dir / 'report.html'}")
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())