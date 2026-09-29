"""Hard filtering logic for jobs."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.candidate import CandidateProfile
from src.models.job import ClassifiedJobRecord


@dataclass
class FilterResult:
    passed: bool
    reason: str | None = None


@dataclass
class FilterDecision:
    job_id: str
    rejected: bool
    rejection_reasons: list[str]


def filter_by_experience_years(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterResult:
    """Hard filter jobs only when the experience gap is severe and explicitly required."""
    if getattr(candidate, 'years_experience', None) is None:
        return FilterResult(passed=True)
        
    req_years = None
    
    if job.experience_requirements:
        for req in job.experience_requirements:
            match = re.search(r'(\d+)', req)
            if match:
                req_years = int(match.group(1))
                break
                
    if req_years is None and job.description_raw:
        match = re.search(r'(\d+)\+?\s*years.*required', job.description_raw, re.IGNORECASE)
        if match:
            req_years = int(match.group(1))
            
    if req_years is not None:
        if candidate.years_experience < (req_years - 3) or candidate.years_experience < (req_years / 2):
            return FilterResult(
                passed=False, 
                reason=f"Severe experience mismatch: Candidate has {candidate.years_experience}y, job strictly requires {req_years}y+"
            )
            
    return FilterResult(passed=True)


def filter_by_education(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterResult:
    """Filter by strict education requirements."""
    return FilterResult(passed=True)


def filter_by_employment_type(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterResult:
    """Filter by incompatible employment type."""
    if not job.employment_type:
        return FilterResult(passed=True)
        
    cand_prefs = getattr(candidate, 'employment_preferences', [])
    if cand_prefs and "Contract" not in cand_prefs and "contract" in job.employment_type.lower():
        return FilterResult(passed=False, reason=f"Incompatible employment type: {job.employment_type}")
        
    return FilterResult(passed=True)


def filter_by_location(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterResult:
    """Filter by impossible location constraints."""
    if not job.location or not getattr(candidate, 'location', None):
        return FilterResult(passed=True)
        
    if getattr(job, 'remote_type', '') == "remote":
        return FilterResult(passed=True)
        
    job_locs = {loc.lower().strip() for loc in job.location}
    cand_locs = {loc.lower().strip() for loc in candidate.location}
    
    for c_loc in cand_locs:
        for j_loc in job_locs:
            if c_loc in j_loc or j_loc in c_loc:
                return FilterResult(passed=True)
                
    return FilterResult(passed=False, reason="Location mismatch for non-remote job")


def filter_by_work_authorization(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterResult:
    """Filter by work authorization conflicts."""
    auth = getattr(candidate, 'work_authorization', None)
    if auth == "VISA_SPONSOR_NEEDED":
        desc = (job.description_raw or "").lower()
        if "no visa sponsorship" in desc or "does not sponsor" in desc or "cannot sponsor" in desc:
            return FilterResult(passed=False, reason="Job does not offer required visa sponsorship")
            
    return FilterResult(passed=True)


def filter_by_remote_preference(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterResult:
    """Filter by incompatible remote preference."""
    pref = getattr(candidate, 'remote_preference', None)
    if not pref:
        return FilterResult(passed=True)
        
    if pref.lower() == "remote":
        if getattr(job, 'remote_type', '') == "onsite":
            return FilterResult(passed=False, reason="Candidate requires remote, job is onsite")
            
        desc = (job.description_raw or "").lower()
        if "onsite position" in desc or "100% onsite" in desc:
            return FilterResult(passed=False, reason="Candidate requires remote, job is onsite")
            
    return FilterResult(passed=True)


def apply_hard_filters(
    candidate: CandidateProfile, 
    jobs: list[ClassifiedJobRecord]
) -> tuple[list[ClassifiedJobRecord], list[FilterDecision]]:
    """Apply all hard filters and return retained jobs + decisions."""
    retained = []
    decisions = []
    
    for job in jobs:
        reasons = []
        
        for filter_func in [
            filter_by_experience_years, 
            filter_by_education, 
            filter_by_employment_type,
            filter_by_location,
            filter_by_work_authorization,
            filter_by_remote_preference
        ]:
            result = filter_func(candidate, job)
            if not result.passed:
                reasons.append(result.reason or f"{filter_func.__name__} failed")

        is_rejected = len(reasons) > 0
        decisions.append(FilterDecision(
            job_id=job.job_id,
            rejected=is_rejected,
            rejection_reasons=reasons
        ))
        
        if not is_rejected:
            retained.append(job)
            
    return retained, decisions


def check_hard_filters(candidate: CandidateProfile, job: ClassifiedJobRecord) -> FilterDecision:
    """Helper to check a single job."""
    _, decisions = apply_hard_filters(candidate, [job])
    return decisions[0]