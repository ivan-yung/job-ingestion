"""Requirement classification logic."""

from __future__ import annotations

import re
from typing import Literal

from src.models.job import ClassifiedJobRecord, JobRecord, RequirementWithEvidence


# Keywords indicating hard requirements
HARD_REQUIREMENT_KEYWORDS = [
    r"\bmust\b",
    r"\brequired\b",
    r"\brequire\b",
    r"\bessential\b",
    r"\bobligatory\b",
    r"\bcompulsory\b",
    r"\bmandatory\b",
]

# Keywords indicating preferred requirements
PREFERRED_REQUIREMENT_KEYWORDS = [
    r"\bpreferred\b",
    r"\bnice to have\b",
    r"\ba plus\b",
    r"\badditional\b",
    r"\bdesirable\b",
    r"\bbeneficial\b",
]

# Keywords indicating unknown/ambiguous requirements
AMBIGUOUS_REQUIREMENT_KEYWORDS = [
    r"\bshould\b",
    r"\bcan\b",
    r"\bmight\b",
    r"\bmaybe\b",
]


def classify_requirement(text: str) -> Literal["hard", "preferred", "unknown"]:
    """Classify a requirement statement.

    Args:
        text: Requirement text.

    Returns:
        Classification: "hard", "preferred", or "unknown".
    """
    if not text:
        return "unknown"

    text_lower = text.lower()

    # Check hard requirement indicators
    for pattern in HARD_REQUIREMENT_KEYWORDS:
        if re.search(pattern, text_lower):
            # Check for negation of hard requirement (e.g., "not required")
            if re.search(r"\b(?:no|not).*?\b(?:required|must)\b", text_lower):
                return "preferred"
            return "hard"

    # Check preferred requirement indicators
    for pattern in PREFERRED_REQUIREMENT_KEYWORDS:
        if re.search(pattern, text_lower):
            return "preferred"

    # Check ambiguous indicators
    for pattern in AMBIGUOUS_REQUIREMENT_KEYWORDS:
        if re.search(pattern, text_lower):
            return "unknown"

    # Default to preferred if unclear (better than hard filtering)
    return "unknown"


def extract_and_classify_requirements(
    description: str,
    requirements: list[str],
) -> list[RequirementWithEvidence]:
    """Extract requirements and classify them with evidence.

    Args:
        description: Full job description for finding evidence.
        requirements: List of extracted requirement texts.

    Returns:
        List of requirements with classification and evidence.
    """
    classified = []

    for req in requirements:
        if not req.strip():
            continue

        classification = classify_requirement(req)

        # Find evidence in description
        evidence = find_evidence_sentence(req, description)

        classified.append(
            RequirementWithEvidence(
                text=req,
                classification=classification,
                evidence_sentence=evidence,
            )
        )

    return classified


def find_evidence_sentence(requirement: str, description: str) -> str:
    """Find the original sentence containing the requirement in the description.

    Args:
        requirement: Requirement text.
        description: Full job description.

    Returns:
        The sentence containing the requirement, or the requirement itself if not found.
    """
    if not description:
        return requirement

    # Split description into sentences
    sentences = re.split(r"(?<=[.!?])\s+", description)

    # Find the sentence containing the requirement
    req_lower = requirement.lower()
    for sentence in sentences:
        if req_lower in sentence.lower():
            return sentence.strip()

    # Return first sentence containing key words from requirement
    words = requirement.split()[:3]  # First 3 words
    for word in words:
        if len(word) > 3:  # Skip short words
            for sentence in sentences:
                if word.lower() in sentence.lower():
                    return sentence.strip()

    return requirement


def extract_seniority(description: str) -> str | None:
    """Extract seniority level from job description.

    Args:
        description: Job description.

    Returns:
        Seniority level or None.
    """
    if not description:
        return None

    description_lower = description.lower()

    # Check for explicit seniority indicators
    seniority_patterns = [
        (r"\bentry\s*-?\s*level\b", "junior"),
        (r"\bjunior\b", "junior"),
        (r"\bmid\s*-?\s*level\b", "mid"),
        (r"\bmid\b", "mid"),
        (r"\bsenior\b", "senior"),
        (r"\blead\b", "lead"),
        (r"\bstaff\b", "senior"),
        (r"\bexecutive\b", "director"),
        (r"\bdirector\b", "director"),
    ]

    for pattern, level in seniority_patterns:
        if re.search(pattern, description_lower):
            return level

    # Check for years of experience as seniority indicator
    years_pattern = r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)"
    match = re.search(years_pattern, description_lower)
    if match:
        years = int(match.group(1))
        if years == 0:
            return "new_grad"
        elif years < 2:
            return "junior"
        elif years < 5:
            return "mid"
        elif years < 10:
            return "senior"
        else:
            return "director"

    return None


def classify_job_record(job: JobRecord) -> ClassifiedJobRecord:
    """Classify all requirements in a job record.

    Args:
        job: Job record.

    Returns:
        Classified job record with evidence.
    """
    # Classify requirements
    required_with_evidence = extract_and_classify_requirements(
        job.description_raw, job.required_qualifications
    )
    preferred_with_evidence = extract_and_classify_requirements(
        job.description_raw, job.preferred_qualifications
    )

    # Extract seniority
    seniority = job.seniority or extract_seniority(job.description_raw)

    # Create classified record
    classified = ClassifiedJobRecord(
        schema_version=job.schema_version,
        job_id=job.job_id,
        source=job.source,
        source_url=job.source_url,
        apply_url=job.apply_url,
        title=job.title,
        normalized_title=job.normalized_title,
        company=job.company,
        normalized_company=job.normalized_company,
        location=job.location,
        remote_type=job.remote_type,
        employment_type=job.employment_type,
        seniority=seniority,
        description_raw=job.description_raw,
        summary=job.summary,
        responsibilities=job.responsibilities,
        required_qualifications=job.required_qualifications,
        preferred_qualifications=job.preferred_qualifications,
        technologies=job.technologies,
        education_requirements=job.education_requirements,
        experience_requirements=job.experience_requirements,
        salary_min=job.salary_min,
        salary_max=job.salary_max,
        salary_currency=job.salary_currency,
        posted_at=job.posted_at,
        searchable_text=job.searchable_text,
        content_hash=job.content_hash,
        required_qualifications_with_evidence=required_with_evidence,
        preferred_qualifications_with_evidence=preferred_with_evidence,
    )

    return classified
