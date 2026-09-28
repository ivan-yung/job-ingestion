"""Pydantic models for normalized job records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class JobRecord(BaseModel):
    """Normalized, validated job record schema."""

    schema_version: str = Field(default="1.0", description="Version of the schema")
    job_id: str = Field(description="Unique job identifier")
    source: str = Field(description="Source platform (e.g., 'apify', 'linkedin')")
    source_url: str | None = Field(default=None, description="URL where job was found")
    apply_url: str | None = Field(default=None, description="Direct application URL")

    title: str = Field(description="Original job title")
    normalized_title: str = Field(description="Normalized job title")
    company: str = Field(description="Original company name")
    normalized_company: str = Field(description="Normalized company name")

    location: list[str] = Field(default_factory=list, description="Job locations (cities/regions)")
    remote_type: Literal["remote", "hybrid", "onsite", "unknown"] = Field(
        default="unknown", description="Remote work status"
    )
    employment_type: str | None = Field(
        default=None, description="Full-time, Part-time, Contract, etc."
    )
    seniority: str | None = Field(
        default=None, description="Junior, Mid, Senior, Lead, etc."
    )

    description_raw: str = Field(description="Original job description (unchanged)")
    summary: str = Field(description="Brief summary of the role")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities")
    required_qualifications: list[str] = Field(
        default_factory=list, description="Hard requirements"
    )
    preferred_qualifications: list[str] = Field(
        default_factory=list, description="Nice-to-have qualifications"
    )
    technologies: list[str] = Field(default_factory=list, description="Technologies/languages")
    education_requirements: list[str] = Field(
        default_factory=list, description="Education/degree requirements"
    )
    experience_requirements: list[str] = Field(
        default_factory=list, description="Years/type of experience"
    )

    salary_min: float | None = Field(default=None, description="Minimum salary")
    salary_max: float | None = Field(default=None, description="Maximum salary")
    salary_currency: str | None = Field(default=None, description="Currency code (USD, EUR, etc.)")

    posted_at: datetime | None = Field(default=None, description="When the job was posted")

    searchable_text: str = Field(
        description="Concatenated searchable text (for full-text search)"
    )
    content_hash: str = Field(
        description="SHA-256 hash of normalized content for deduplication"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "schema_version": "1.0",
                "job_id": "apify_12345",
                "source": "apify",
                "source_url": "https://example.com/jobs/12345",
                "apply_url": "https://example.com/apply/12345",
                "title": "Senior Python Developer",
                "normalized_title": "senior python developer",
                "company": "Tech Corp Inc.",
                "normalized_company": "tech corp",
                "location": ["San Francisco, CA", "Remote"],
                "remote_type": "hybrid",
                "employment_type": "Full-time",
                "seniority": "Senior",
                "description_raw": "...",
                "summary": "We are looking for...",
                "responsibilities": ["Design and build scalable systems..."],
                "required_qualifications": ["5+ years Python experience"],
                "preferred_qualifications": ["Kubernetes experience"],
                "technologies": ["Python", "FastAPI", "PostgreSQL"],
                "education_requirements": ["Bachelor's in CS or related field"],
                "experience_requirements": ["5+ years"],
                "salary_min": 150000.0,
                "salary_max": 200000.0,
                "salary_currency": "USD",
                "posted_at": "2024-01-15T10:30:00",
                "searchable_text": "...",
                "content_hash": "abc123def456",
            }
        }


class RequirementWithEvidence(BaseModel):
    """A requirement with its source evidence and classification."""

    text: str = Field(description="The requirement text")
    classification: Literal["hard", "preferred", "unknown"] = Field(
        description="Whether this is a hard requirement, preferred, or unknown"
    )
    evidence_sentence: str = Field(
        description="The original sentence from the job description"
    )


class ClassifiedJobRecord(JobRecord):
    """Extended job record with classified requirements."""

    required_qualifications_with_evidence: list[RequirementWithEvidence] = Field(
        default_factory=list, description="Hard requirements with source evidence"
    )
    preferred_qualifications_with_evidence: list[RequirementWithEvidence] = Field(
        default_factory=list, description="Preferred requirements with source evidence"
    )
