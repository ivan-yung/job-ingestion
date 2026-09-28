"""Job deduplication logic."""

from __future__ import annotations

from typing import NamedTuple

from src.models.job import JobRecord
from src.utils.hashing import content_fingerprint


class DuplicateGroup(NamedTuple):
    """A group of duplicate job records."""

    primary: JobRecord
    duplicates: list[JobRecord]
    reason: str


def is_duplicate(job1: JobRecord, job2: JobRecord) -> bool:
    """Check if two jobs are duplicates.

    Uses exact match on:
    1. source + job_id
    2. content_hash

    Args:
        job1: First job record.
        job2: Second job record.

    Returns:
        True if jobs are duplicates.
    """
    # Exact source + ID match
    if (
        job1.source == job2.source
        and job1.job_id == job2.job_id
    ):
        return True

    # Content hash match (fingerprints are identical)
    if job1.content_hash == job2.content_hash:
        return True

    return False


def find_duplicates(jobs: list[JobRecord]) -> list[DuplicateGroup]:
    """Find all duplicate groups in a job list.

    Args:
        jobs: List of job records.

    Returns:
        List of duplicate groups.
    """
    seen = {}
    groups = []

    for job in jobs:
        key = (job.source, job.job_id)

        if key in seen:
            # Found duplicate by source + ID
            primary = seen[key][0]
            seen[key].append(job)
            continue

        # Check content hash
        hash_key = job.content_hash
        found_hash_match = False

        for existing_key, existing_jobs in seen.items():
            if existing_jobs[0].content_hash == hash_key:
                # Found duplicate by content hash
                existing_jobs.append(job)
                found_hash_match = True
                break

        if not found_hash_match:
            # New unique job
            seen[key] = [job]

    # Create duplicate groups
    for jobs_list in seen.values():
        if len(jobs_list) > 1:
            groups.append(
                DuplicateGroup(
                    primary=jobs_list[0],
                    duplicates=jobs_list[1:],
                    reason="source_id" if jobs_list[0].source == jobs_list[1].source else "content_hash",
                )
            )

    return groups


def deduplicate(jobs: list[JobRecord]) -> list[JobRecord]:
    """Remove duplicate jobs, keeping only the first occurrence.

    Args:
        jobs: List of job records.

    Returns:
        Deduplicated list of unique jobs.
    """
    seen_sources = set()
    seen_hashes = set()
    unique_jobs = []

    for job in jobs:
        source_key = (job.source, job.job_id)

        if source_key in seen_sources:
            continue

        if job.content_hash in seen_hashes:
            continue

        seen_sources.add(source_key)
        seen_hashes.add(job.content_hash)
        unique_jobs.append(job)

    return unique_jobs


def merge_duplicates(primary: JobRecord, duplicate: JobRecord) -> JobRecord:
    """Merge a duplicate into primary job record.

    Prefers non-empty fields from primary, falls back to duplicate.

    Args:
        primary: Primary job record.
        duplicate: Duplicate job record to merge from.

    Returns:
        Merged job record.
    """
    # Merge locations
    merged_locations = list(set(primary.location + duplicate.location))

    # Merge URLs (prefer apply_url over source_url)
    merged_apply_url = primary.apply_url or duplicate.apply_url
    merged_source_url = primary.source_url or duplicate.source_url

    # Merge salary (prefer ranges from both)
    merged_salary_min = primary.salary_min or duplicate.salary_min
    merged_salary_max = primary.salary_max or duplicate.salary_max

    # Merge field lists (prefer primary, add unique from duplicate)
    def merge_lists(*lists):
        return list(dict.fromkeys(item for lst in lists for item in lst))

    merged = JobRecord(
        schema_version=primary.schema_version,
        job_id=primary.job_id,
        source=primary.source,
        source_url=merged_source_url,
        apply_url=merged_apply_url,
        title=primary.title,
        normalized_title=primary.normalized_title,
        company=primary.company,
        normalized_company=primary.normalized_company,
        location=merged_locations,
        remote_type=primary.remote_type,
        employment_type=primary.employment_type or duplicate.employment_type,
        seniority=primary.seniority or duplicate.seniority,
        description_raw=primary.description_raw,
        summary=primary.summary or duplicate.summary,
        responsibilities=merge_lists(primary.responsibilities, duplicate.responsibilities),
        required_qualifications=merge_lists(
            primary.required_qualifications, duplicate.required_qualifications
        ),
        preferred_qualifications=merge_lists(
            primary.preferred_qualifications, duplicate.preferred_qualifications
        ),
        technologies=merge_lists(primary.technologies, duplicate.technologies),
        education_requirements=merge_lists(
            primary.education_requirements, duplicate.education_requirements
        ),
        experience_requirements=merge_lists(
            primary.experience_requirements, duplicate.experience_requirements
        ),
        salary_min=merged_salary_min,
        salary_max=merged_salary_max,
        salary_currency=primary.salary_currency or duplicate.salary_currency,
        posted_at=primary.posted_at or duplicate.posted_at,
        searchable_text=primary.searchable_text,
        content_hash=primary.content_hash,
    )

    return merged
