"""Feature extraction and deterministic scoring for reranking."""

import re
from typing import Any

from pydantic import BaseModel, Field

from src.models.candidate import CandidateProfile
from src.models.job import JobRecord
from src.reranking.normalizers import normalize_skill, title_similarity


class RerankFeatures(BaseModel):
    semantic_score: float = Field(ge=0, le=1)
    bm25_signal: float = Field(ge=0, le=1)
    title_similarity: float = Field(ge=0, le=1)
    required_skill_coverage: float = Field(ge=0, le=1)
    preferred_skill_coverage: float = Field(ge=0, le=1)
    required_technology_coverage: float = Field(ge=0, le=1)
    preferred_technology_coverage: float = Field(ge=0, le=1)
    experience_fit: float = Field(ge=0, le=1)
    seniority_fit: float = Field(ge=0, le=1)
    education_fit: float = Field(ge=0, le=1)
    location_fit: float = Field(ge=0, le=1)
    hard_requirement_status: str


# Stage 3 Scoring Weights
WEIGHTS = {
    "required_skill_coverage": 0.30,
    "required_technology_coverage": 0.15,
    "title_similarity": 0.15,
    "experience_fit": 0.15,
    "semantic_score": 0.10,
    "preferred_skill_coverage": 0.05,
    "preferred_technology_coverage": 0.05,
    "education_fit": 0.03,
    "location_fit": 0.02,
}
FORMULA_VERSION = "1.1"


class HardRequirementError(ValueError):
    """Raised internally when an explicit hard requirement is unmet."""


def _calculate_coverage(required: list[str], possessed: set[str]) -> float:
    if not required:
        return 1.0
    normalized = {normalize_skill(value) for value in required}
    return len(normalized & possessed) / len(normalized)


def _extract_years(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _is_preferred(text: str) -> bool:
    return bool(re.search(r"\b(preferred|desired|bonus|nice to have|plus|optional)\b", text, re.IGNORECASE))


def _candidate_skills(candidate: CandidateProfile) -> set[str]:
    skills = getattr(candidate, "skills", None)
    values: list[str] = []
    for category in ("languages", "frameworks", "databases", "cloud", "tools", "other"):
        category_values = getattr(skills, category, [])
        if isinstance(category_values, (list, tuple, set)):
            values.extend(str(value) for value in category_values)
    for experience in getattr(candidate, "professional_experience", []) or []:
        technologies = getattr(experience, "technologies", [])
        if isinstance(technologies, (list, tuple, set)):
            values.extend(str(value) for value in technologies)
    return {normalize_skill(value) for value in values}


def _requirement_matches(requirement: str, possessed: set[str]) -> bool:
    normalized_requirement = normalize_skill(requirement)
    if normalized_requirement in possessed:
        return True
    return any(re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", normalized_requirement) for skill in possessed)


def _education_fit(candidate: CandidateProfile, requirements: list[str]) -> tuple[float, bool]:
    if not requirements:
        return 1.0, False
    education = getattr(candidate, "education", [])
    if not isinstance(education, list):
        education = []
    candidate_text = " ".join(
        f"{getattr(education, 'degree', '')} {getattr(education, 'field', '')}".lower()
        for education in education
    )
    exact_or_related = ("computer science", "computer engineering", "software engineering", "electrical engineering", "information technology", "technical")
    matched = any(
        "related" in requirement.lower() and any(field in candidate_text for field in exact_or_related)
        or any(token in candidate_text for token in re.findall(r"[a-z]+", requirement.lower()) if len(token) > 3)
        for requirement in requirements
    )
    return (1.0 if matched else 0.0), not matched


def _experience_features(candidate: CandidateProfile, job: JobRecord) -> tuple[float, bool]:
    candidate_years = float(getattr(candidate, "years_experience", 0) or 0)
    requirements = list(getattr(job, "experience_requirements", []) or [])
    if not requirements:
        requirements = re.findall(r"\d+(?:\.\d+)?\s*\+?\s*years?[^.\n]*", job.description_raw, re.IGNORECASE)
    parsed = [(years, _is_preferred(text)) for text in requirements if (years := _extract_years(text)) is not None]
    if not parsed:
        return 1.0, False
    hard_years = [years for years, preferred in parsed if not preferred]
    preferred_years = [years for years, preferred in parsed if preferred]
    target = max(hard_years + preferred_years)
    fit = min(1.0, candidate_years / target) if target else 1.0
    hard_failed = bool(hard_years and candidate_years < max(hard_years))
    return fit, hard_failed


def extract_features(candidate: CandidateProfile, job: JobRecord, retrieval_data: dict[str, Any]) -> RerankFeatures:
    """Extract 0.0-1.0 bounded features for scoring."""
    
    cand_skills = _candidate_skills(candidate)

    # Skill/Tech Coverage
    req_skills_cov = _calculate_coverage(job.required_qualifications, cand_skills)
    pref_skills_cov = _calculate_coverage(job.preferred_qualifications, cand_skills)
    preferred_technologies = getattr(job, "preferred_technologies", []) or []
    req_tech_cov = _calculate_coverage(job.technologies, cand_skills)
    pref_tech_cov = _calculate_coverage(preferred_technologies, cand_skills)

    target_roles = getattr(candidate, "target_roles", [])
    if not isinstance(target_roles, list):
        target_roles = []
    title_sim = title_similarity(target_roles, job.title)
    exp_fit, experience_failed = _experience_features(candidate, job)
    education_fit, education_failed = _education_fit(candidate, job.education_requirements)

    missing_required_skills = any(not _requirement_matches(req, cand_skills) for req in job.required_qualifications if not _extract_years(req) and "degree" not in req.lower() and "bachelor" not in req.lower() and "master" not in req.lower())
    technology_failed = req_tech_cov < 1.0 and bool(job.technologies)
    hard_status = "failed" if experience_failed or education_failed or missing_required_skills or technology_failed else "passed"

    # Normalize retrieval signals
    semantic_score = max(0.0, min(1.0, retrieval_data.get("semantic_score", 0.0)))
    
    # Simple BM25 normalization (bounded to 1.0 for scoring feature)
    raw_bm25 = retrieval_data.get("bm25_score", 0.0)
    bm25_signal = max(0.0, min(1.0, raw_bm25 / 50.0))

    return RerankFeatures(
        semantic_score=semantic_score,
        bm25_signal=bm25_signal,
        title_similarity=title_sim,
        required_skill_coverage=req_skills_cov,
        preferred_skill_coverage=pref_skills_cov,
        required_technology_coverage=req_tech_cov,
        preferred_technology_coverage=pref_tech_cov,
        experience_fit=exp_fit,
        seniority_fit=1.0,
        education_fit=education_fit,
        location_fit=_location_fit(candidate, job),
        hard_requirement_status=hard_status
    )


def _location_fit(candidate: CandidateProfile, job: JobRecord) -> float:
    if getattr(candidate, "remote_preference", None) == "remote":
        return 1.0 if job.remote_type == "remote" or any("remote" in place.lower() for place in job.location) else 0.0
    candidate_location = getattr(candidate, "location", [])
    if not isinstance(candidate_location, list):
        candidate_location = []
    if not candidate_location or not job.location:
        return 1.0
    candidate_places = {place.lower() for place in candidate_location}
    return 1.0 if any(place.lower() in candidate_places for place in job.location) else 0.5


def calculate_score(features: RerankFeatures) -> float:
    """Compute the weighted deterministic score."""
    if features.hard_requirement_status == "failed":
        return 0.0
        
    score = (
        (features.required_skill_coverage * WEIGHTS["required_skill_coverage"]) +
        (features.required_technology_coverage * WEIGHTS["required_technology_coverage"]) +
        (features.title_similarity * WEIGHTS["title_similarity"]) +
        (features.experience_fit * WEIGHTS["experience_fit"]) +
        (features.semantic_score * WEIGHTS["semantic_score"]) +
        (features.preferred_skill_coverage * WEIGHTS["preferred_skill_coverage"]) +
        (features.preferred_technology_coverage * WEIGHTS["preferred_technology_coverage"]) +
        (features.education_fit * WEIGHTS["education_fit"]) +
        (features.location_fit * WEIGHTS["location_fit"])
    )
    return round(max(0.0, min(1.0, score)), 4)
