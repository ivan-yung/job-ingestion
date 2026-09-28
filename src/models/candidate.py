"""Pydantic models for candidate profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Skills(BaseModel):
    """Categorized technical and professional skills."""

    languages: list[str] = Field(default_factory=list, description="Programming languages")
    frameworks: list[str] = Field(default_factory=list, description="Frameworks and libraries")
    databases: list[str] = Field(default_factory=list, description="Database systems")
    cloud: list[str] = Field(default_factory=list, description="Cloud platforms")
    tools: list[str] = Field(default_factory=list, description="Tools and utilities")
    other: list[str] = Field(default_factory=list, description="Other skills")


class ProfessionalExperience(BaseModel):
    """A job/role in candidate's professional history."""

    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    duration: str | None = Field(default=None, description="Duration (e.g., '2020-2023')")
    years: float | None = Field(default=None, description="Estimated years in role")
    description: str = Field(default="", description="Role description")
    technologies: list[str] = Field(default_factory=list, description="Technologies used")


class Project(BaseModel):
    """A project from candidate's portfolio."""

    name: str = Field(description="Project name")
    description: str = Field(default="", description="Project description")
    technologies: list[str] = Field(default_factory=list, description="Technologies used")
    url: str | None = Field(default=None, description="Project URL/link")


class Education(BaseModel):
    """Educational background."""

    institution: str = Field(description="School/university name")
    degree: str = Field(description="Degree type (e.g., 'Bachelor of Science')")
    field: str | None = Field(default=None, description="Field of study")
    graduation_year: int | None = Field(default=None, description="Year of graduation")


class Certification(BaseModel):
    """Professional certification."""

    name: str = Field(description="Certification name")
    issuer: str = Field(description="Issuing organization")
    date: str | None = Field(default=None, description="Date obtained")


class CandidateProfile(BaseModel):
    """Parsed, structured candidate profile from resume."""

    schema_version: str = Field(default="1.0", description="Version of the schema")
    candidate_id: str = Field(default="local-user", description="Unique candidate identifier")

    # Target information
    target_roles: list[str] = Field(
        default_factory=list, description="Target job titles/roles"
    )

    # Skills
    skills: Skills = Field(default_factory=Skills, description="Categorized skills")

    # Experience
    professional_experience: list[ProfessionalExperience] = Field(
        default_factory=list, description="Professional work history"
    )
    projects: list[Project] = Field(default_factory=list, description="Portfolio projects")

    # Education
    education: list[Education] = Field(
        default_factory=list, description="Educational background"
    )
    certifications: list[Certification] = Field(
        default_factory=list, description="Professional certifications"
    )

    # Location and preferences
    location: list[str] = Field(default_factory=list, description="Current/preferred locations")
    remote_preference: Literal["remote", "hybrid", "onsite", None] = Field(
        default=None, description="Remote work preference"
    )

    # Experience level
    experience_level: Literal[
        "new_grad", "junior", "mid", "senior", "lead", "manager", "director", "executive"
    ] = Field(default="new_grad", description="Overall experience level")
    years_experience: float = Field(
        default=0, description="Total years of professional experience"
    )

    # Work authorization
    work_authorization: str | None = Field(
        default=None, description="Work authorization status (e.g., 'US_CITIZEN', 'VISA_SPONSOR_NEEDED')"
    )

    # Source tracking
    resume_path: str | None = Field(default=None, description="Path to source resume file")
    resume_raw_text: str | None = Field(default=None, description="Raw extracted resume text")

    class Config:
        json_schema_extra = {
            "example": {
                "schema_version": "1.0",
                "candidate_id": "local-user",
                "target_roles": ["Senior Python Developer", "Tech Lead"],
                "skills": {
                    "languages": ["Python", "JavaScript", "Rust"],
                    "frameworks": ["FastAPI", "React", "Django"],
                    "databases": ["PostgreSQL", "MongoDB"],
                    "cloud": ["AWS", "GCP"],
                    "tools": ["Docker", "Kubernetes", "Git"],
                },
                "professional_experience": [
                    {
                        "title": "Senior Engineer",
                        "company": "TechCorp",
                        "duration": "2021-2024",
                        "years": 3,
                        "description": "Led backend team...",
                        "technologies": ["Python", "FastAPI", "PostgreSQL"],
                    }
                ],
                "projects": [],
                "education": [
                    {
                        "institution": "University of California",
                        "degree": "Bachelor of Science",
                        "field": "Computer Science",
                        "graduation_year": 2018,
                    }
                ],
                "certifications": [],
                "location": ["San Francisco, CA"],
                "remote_preference": "hybrid",
                "experience_level": "senior",
                "years_experience": 6,
                "work_authorization": None,
                "resume_path": "/path/to/resume.pdf",
                "resume_raw_text": "...",
            }
        }
