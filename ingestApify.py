"""Backward-compatible entrypoint for the job ingestion script."""

from ingest_jobs import main


if __name__ == "__main__":
    raise SystemExit(main())