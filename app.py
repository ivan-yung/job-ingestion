"""Main pipeline entry point for job ingestion and filtering."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tim
from pathlib import Path
from typing import Any

# Allow `python -m app.retrieve` to coexist with this legacy top-level module.
__path__ = [os.path.join(os.path.dirname(__file__), "app")]

from src.filtering.hard_filters import apply_hard_filters
from src.ingestion.deduplicate import deduplicate
from src.ingestion.normalize import normalize_apify_record
from src.ingestion.requirement_classifier import classify_job_record
from src.ingestion.resume_parser import parse_resume
from src.models.candidate import CandidateProfile
from src.models.job import ClassifiedJobRecord, JobRecord
from src.storage.storage import CandidateStorage, JobStorage


def load_jobs_from_file(path: str | Path) -> list[dict[str, Any]]:
    """Load job records from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of job records.

    Raises:
        ValueError: If file doesn't exist or isn't valid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Jobs file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "items" in data:
        return data["items"]

    raise ValueError(f"Unexpected JSON structure in {path}")


def normalize_and_deduplicate(
    raw_jobs: list[dict[str, Any]],
    source: str = "apify",
) -> tuple[list[JobRecord], list[str]]:
    """Normalize and deduplicate jobs.

    Args:
        raw_jobs: List of raw job records.
        source: Data source identifier.

    Returns:
        Tuple of (unique_jobs, failed_job_ids).
    """
    failed_ids = []
    normalized_jobs = []

    for i, job_record in enumerate(raw_jobs):
        try:
            job = normalize_apify_record(job_record)
            if job:
                normalized_jobs.append(job)
            else:
                failed_ids.append(str(job_record.get("id", f"job_{i}")))
        except Exception as e:
            print(f"Error normalizing job: {e}", file=sys.stderr)
            failed_ids.append(str(job_record.get("id", f"job_{i}")))

    unique_jobs = deduplicate(normalized_jobs)

    return unique_jobs, failed_ids


def classify_and_filter(
    jobs: list[JobRecord],
    candidate: CandidateProfile | None = None,
) -> tuple[list[ClassifiedJobRecord], list[ClassifiedJobRecord]]:
    """Classify requirements and apply hard filters.

    Args:
        jobs: List of normalized jobs.
        candidate: Candidate profile for filtering, or None to skip filtering.

    Returns:
        Tuple of (retained_jobs, filtered_out_jobs).
    """
    classified_jobs = []
    for job in jobs:
        try:
            classified = classify_job_record(job)
            classified_jobs.append(classified)
        except Exception as e:
            print(f"Error classifying job {job.job_id}: {e}", file=sys.stderr)

    if not candidate:
        return classified_jobs, []

    retained, decisions = apply_hard_filters(candidate, classified_jobs)

    filtered_out = [job for job in classified_jobs if job.job_id not in {j.job_id for j in retained}]

    print(f"\nFiltering Summary:")
    print(f"  Total jobs: {len(classified_jobs)}")
    print(f"  Retained: {len(retained)}")
    print(f"  Filtered out: {len(filtered_out)}")

    for decision in decisions:
        if decision.rejected:
            print(f"  Job {decision.job_id}: REJECTED")
            for reason in decision.rejection_reasons:
                print(f"    - {reason}")

    return retained, filtered_out


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Resume-ranker: Job normalization and candidate matching"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jobs JSON file",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Path to candidate resume file (PDF or TXT)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Path to output file for matched jobs",
    )

    parser.add_argument(
        "--db",
        type=str,
        default="resume_ranker.db",
        help="Path to SQLite database",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="apify",
        help="Data source identifier",
    )

    args = parser.parse_args()

    print(f"Resume Ranker v1.0")
    print(f"=" * 50)

    # Load jobs
    print(f"\n1. Loading jobs from {args.input}...")
    try:
        raw_jobs = load_jobs_from_file(args.input)
        print(f"   Loaded {len(raw_jobs)} jobs")
    except Exception as e:
        print(f"   Error: {e}", file=sys.stderr)
        return 1

    # Normalize and deduplicate
    print(f"\n2. Normalizing and deduplicating jobs...")
    start = time.time()
    normalized_jobs, failed_ids = normalize_and_deduplicate(raw_jobs, source=args.source)
    elapsed = time.time() - start
    print(f"   Normalized: {len(normalized_jobs)} unique jobs in {elapsed:.2f}s")
    if failed_ids:
        print(f"   Failed: {len(failed_ids)} jobs")

    # Store jobs
    print(f"\n3. Storing jobs in database...")
    job_storage = JobStorage(args.db)
    job_storage.store_jobs(normalized_jobs)
    print(f"   Stored {job_storage.count_jobs()} jobs total")

    # Load and parse resume if provided
    candidate = None
    if args.resume:
        print(f"\n4. Parsing resume from {args.resume}...")
        try:
            candidate = parse_resume(args.resume)
            print(f"   Parsed resume for {candidate.candidate_id}")
            print(f"   Experience level: {candidate.experience_level} ({candidate.years_experience} years)")
            print(f"   Skills: {len(candidate.skills.languages)} languages, {len(candidate.skills.frameworks)} frameworks")

            # Store candidate
            candidate_storage = CandidateStorage(args.db)
            candidate_storage.store_candidate(candidate)

        except Exception as e:
            print(f"   Error parsing resume: {e}", file=sys.stderr)
            candidate = None

    # Classify and filter
    print(f"\n5. Classifying requirements and applying filters...")
    start = time.time()
    retained_jobs, filtered_jobs = classify_and_filter(normalized_jobs, candidate)
    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.2f}s")

    # Save output
    print(f"\n6. Saving results to {args.output}...")
    output = {
        "metadata": {
            "total_jobs": len(normalized_jobs),
            "retained_jobs": len(retained_jobs),
            "filtered_jobs": len(filtered_jobs),
            "candidate_id": candidate.candidate_id if candidate else None,
        },
        "jobs": [job.model_dump() for job in retained_jobs],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n" + "=" * 50)
    print(f"Pipeline completed successfully!")
    print(f"  Output: {args.output}")
    print(f"  Database: {args.db}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
