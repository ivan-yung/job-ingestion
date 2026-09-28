"""Hard filtering logic for candidate-job compatibility."""

from __future__ import annotations

from typing import NamedTuple

from src.models.candidate import CandidateProfile
from src.models.job import ClassifiedJobRecord


class FilterResult(NamedTuple):
    """Result of a hard filter check."""

    passed: bool
    reason: str | None


class HardFilterDecision(NamedTuple):
    """Final hard filter decision for a job."""

    job_id: str
    rejected: bool
    rejection_reasons: list[str]
    retention_reasons: list[str]


def filter_by_work_authorization(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> FilterResult:
    """Check work authorization compatibility.

    Only rejects if:
    - Job explicitly requires US work authorization
    - Candidate explicitly states they need visa sponsorship

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Filter result.
    """
    if not candidate.work_authorization or not job.description_raw:
        return FilterResult(passed=True, reason=None)

    # Only hard-reject if explicitly incompatible
    if (
        candidate.work_authorization == "VISA_SPONSOR_NEEDED"
        and "no visa sponsorship" in job.description_raw.lower()
    ):
        return FilterResult(
            passed=False,
            reason="Candidate needs visa sponsorship but job does not offer it",
        )

    return FilterResult(passed=True, reason=None)


def filter_by_remote_preference(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> FilterResult:
    """Check remote work preference compatibility.

    Only rejects if:
    - Candidate only wants remote
    - Job is only onsite

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Filter result.
    """
    if not candidate.remote_preference or candidate.remote_preference == "onsite":
        # Candidate is flexible or prefers onsite
        return FilterResult(passed=True, reason=None)

    if candidate.remote_preference == "remote":
        if job.remote_type == "onsite":
            return FilterResult(
                passed=False,
                reason="Candidate requires remote but job is onsite-only",
            )

    if candidate.remote_preference == "hybrid":
        if job.remote_type == "onsite":
            return FilterResult(
                passed=False,
                reason="Candidate requires hybrid/remote but job is onsite-only",
            )

    return FilterResult(passed=True, reason=None)


def filter_by_location(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> FilterResult:
    """Check location compatibility.

    Only rejects if:
    - Job requires specific location
    - Candidate has different specific location
    - Both cannot work (onsite job, candidate elsewhere)

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Filter result.
    """
    if (
        not candidate.location
        or not job.location
        or job.remote_type != "onsite"
    ):
        return FilterResult(passed=True, reason=None)

    # Check if candidate location overlaps with job location
    candidate_locations = [loc.lower() for loc in candidate.location]
    job_locations = [loc.lower() for loc in job.location]

    for cand_loc in candidate_locations:
        for job_loc in job_locations:
            if cand_loc in job_loc or job_loc in cand_loc:
                return FilterResult(passed=True, reason=None)

    # No location overlap and job is onsite
    return FilterResult(
        passed=False,
        reason=f"Job is onsite in {', '.join(job.location)} but candidate is in {', '.join(candidate.location)}",
    )


def filter_by_employment_type(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> FilterResult:
    """Check employment type compatibility.

    Only rejects if:
    - Job specifies a type
    - Candidate cannot work that type

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Filter result.
    """
    if not job.employment_type:
        return FilterResult(passed=True, reason=None)

    # For now, only allow employment type to be informational
    # Do not hard-reject based on employment type
    return FilterResult(passed=True, reason=None)


def filter_by_education(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> FilterResult:
    """Check education requirements.

    Only rejects if:
    - Job explicitly requires a specific degree
    - Candidate cannot satisfy it

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Filter result.
    """
    if not job.education_requirements:
        return FilterResult(passed=True, reason=None)

    # Check if candidate has any education
    if candidate.education:
        return FilterResult(passed=True, reason=None)

    # If job requires CS degree and candidate has no education, might reject
    for req in job.education_requirements:
        if "bachelor" in req.lower() and "computer science" in req.lower():
            if not candidate.education:
                return FilterResult(
                    passed=False,
                    reason="Job requires Bachelor's in Computer Science but candidate has no degree listed",
                )

    return FilterResult(passed=True, reason=None)


def filter_by_experience_years(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> FilterResult:
    """Check experience requirement compatibility.

    Only rejects if:
    - Job explicitly requires N+ years
    - Candidate has significantly less

    Conservative: only reject if clearly impossible.

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Filter result.
    """
    if not job.experience_requirements:
        return FilterResult(passed=True, reason=None)

    # Check for explicit requirement (e.g., "10+ years required")
    import re
    for req in job.experience_requirements:
        match = re.search(r"(\d+)\+?\s*years?\s*required", req, re.IGNORECASE)
        if match:
            required_years = int(match.group(1))

            # Only hard reject if candidate has <20% of required years
            if candidate.years_experience < required_years * 0.2:
                return FilterResult(
                    passed=False,
                    reason=f"Job requires {required_years}+ years but candidate has {candidate.years_experience}",
                )

    return FilterResult(passed=True, reason=None)


def check_hard_filters(
    candidate: CandidateProfile,
    job: ClassifiedJobRecord,
) -> HardFilterDecision:
    """Check all hard filters for a job.

    Args:
        candidate: Candidate profile.
        job: Job record.

    Returns:
        Hard filter decision.
    """
    rejection_reasons = []
    retention_reasons = []

    # Apply all filters
    filters = [
        ("Work Authorization", filter_by_work_authorization),
        ("Remote Preference", filter_by_remote_preference),
        ("Location", filter_by_location),
        ("Employment Type", filter_by_employment_type),
        ("Education", filter_by_education),
        ("Experience", filter_by_experience_years),
    ]

    for filter_name, filter_func in filters:
        result = filter_func(candidate, job)

        if not result.passed and result.reason:
            rejection_reasons.append(result.reason)
        elif result.passed and result.reason:
            retention_reasons.append(f"{filter_name}: {result.reason}")

    rejected = len(rejection_reasons) > 0

    return HardFilterDecision(
        job_id=job.job_id,
        rejected=rejected,
        rejection_reasons=rejection_reasons,
        retention_reasons=retention_reasons,
    )


def apply_hard_filters(
    candidate: CandidateProfile,
    jobs: list[ClassifiedJobRecord],
) -> tuple[list[ClassifiedJobRecord], list[HardFilterDecision]]:
    """Apply hard filters to a list of jobs.

    Args:
        candidate: Candidate profile.
        jobs: List of jobs to filter.

    Returns:
        Tuple of (retained_jobs, all_decisions).
    """
    retained = []
    decisions = []

    for job in jobs:
        decision = check_hard_filters(candidate, job)
        decisions.append(decision)

        if not decision.rejected:
            retained.append(job)

    return retained, decisions
