"""Resume parsing and candidate profile extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.ingestion.normalize import clean_text, split_items
from src.models.candidate import CandidateProfile, Education, Skills


def extract_text_from_file(path: str | Path) -> str:
    """Extract text from a resume file.

    Supports .txt and .pdf files.

    Args:
        path: Path to resume file.

    Returns:
        Extracted text.

    Raises:
        ValueError: If file format is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Resume file not found: {path}")

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)

    raise ValueError(f"Unsupported resume format: {path.suffix}")


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file.

    Args:
        path: Path to PDF file.

    Returns:
        Extracted text.

    Raises:
        ImportError: If PyMuPDF is not installed.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PDF support requires PyMuPDF. Install with: pip install PyMuPDF"
        ) from exc

    text_parts = []

    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def extract_email(text: str) -> str | None:
    """Extract email address from text.

    Args:
        text: Text to search.

    Returns:
        First email found or None.
    """
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    """Extract phone number from text.

    Args:
        text: Text to search.

    Returns:
        First phone number found or None.
    """
    # Pattern for common US/international phone formats
    match = re.search(
        r"(?:\+1[-.\s]?)?\(?[\d]{3}\)?[-.\s]?[\d]{3}[-.\s]?[\d]{4}",
        text
    )
    return match.group(0) if match else None


def extract_links(text: str) -> list[str]:
    """Extract URLs from text.

    Args:
        text: Text to search.

    Returns:
        List of URLs found.
    """
    pattern = r"https?://[^\s]+"
    return re.findall(pattern, text)


def extract_years_of_experience(text: str) -> float:
    """Extract estimated years of professional experience.

    Args:
        text: Resume text.

    Returns:
        Estimated years of experience.
    """
    # Look for explicit years mentioned
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience",
        r"(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:software|tech|it|industry)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass

    # Fall back to counting dates in work experience section
    date_pattern = r"(\d{4})\s*-\s*(?:present|current|now|(\d{4}))"
    matches = re.findall(date_pattern, text, re.IGNORECASE)

    if matches:
        current_year = 2024
        total_years = 0

        for start, end in matches:
            start_year = int(start)
            if end:
                end_year = int(end)
            else:
                end_year = current_year

            total_years += end_year - start_year

        return float(total_years)

    return 0.0


def extract_skills(text: str) -> Skills:
    """Extract technical skills from resume text.

    Args:
        text: Resume text.

    Returns:
        Skills object.
    """
    # Common programming languages
    languages = [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "groovy", "sql", "r", "matlab",
        "perl", "bash", "shell", "lua", "erlang", "elixir", "clojure"
    ]

    # Common frameworks
    frameworks = [
        "django", "fastapi", "flask", "react", "vue", "angular", "svelte",
        "express", "nest", "spring", "spring boot", "rails", "laravel", "asp.net",
        "dot net", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    ]

    # Common databases
    databases = [
        "postgresql", "mysql", "mongodb", "cassandra", "redis", "dynamodb",
        "elasticsearch", "neo4j", "firestore", "snowflake", "bigquery",
    ]

    # Cloud platforms
    cloud = [
        "aws", "amazon", "gcp", "google cloud", "azure", "heroku", "digital ocean",
        "linode", "kubernetes", "docker", "ec2", "s3", "rds", "lambda",
    ]

    # Tools
    tools = [
        "git", "docker", "kubernetes", "jenkins", "gitlab", "github", "bitbucket",
        "jira", "confluence", "slack", "terraform", "ansible", "grafana", "prometheus",
        "datadog", "splunk", "elk", "linux", "windows", "macos",
    ]

    def find_items(item_list: list[str]) -> list[str]:
        found = []
        text_lower = text.lower()
        for item in item_list:
            # Use word boundaries to avoid false matches
            if re.search(rf"\b{re.escape(item)}\b", text_lower):
                found.append(item.capitalize())
        return found

    return Skills(
        languages=find_items(languages),
        frameworks=find_items(frameworks),
        databases=find_items(databases),
        cloud=find_items(cloud),
        tools=find_items(tools),
    )


def extract_education(text: str) -> list[Education]:
    """Extract education history from resume.

    Args:
        text: Resume text.

    Returns:
        List of education records.
    """
    education_list = []

    # Pattern: Degree, School, Year
    degree_pattern = r"(?:Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|MBA|B\.E\.|M\.E\.)\s*(?:of\s*)?(?:Science|Arts|Engineering)?\s*(?:in\s*)?([^,\n]+?)(?:,|\s+from\s+|\s+at\s+)([^,\n]+?)(?:,|\s+)?(\d{4})?"

    for match in re.finditer(degree_pattern, text, re.IGNORECASE):
        field = match.group(1).strip() if match.group(1) else None
        institution = match.group(2).strip() if match.group(2) else None
        year = int(match.group(3)) if match.group(3) else None

        if institution:
            education_list.append(
                Education(
                    institution=institution,
                    degree=match.group(0).split()[0],
                    field=field,
                    graduation_year=year,
                )
            )

    return education_list


def parse_resume(path: str | Path) -> CandidateProfile:
    """Parse a resume file and extract candidate profile.

    Args:
        path: Path to resume file.

    Returns:
        Parsed candidate profile.
    """
    path = Path(path)

    # Extract text
    resume_text = extract_text_from_file(path)

    # Extract basic info
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    links = extract_links(resume_text)

    # Extract experience info
    years_experience = extract_years_of_experience(resume_text)
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)

    # Determine experience level
    if years_experience == 0:
        experience_level = "new_grad"
    elif years_experience < 2:
        experience_level = "junior"
    elif years_experience < 5:
        experience_level = "mid"
    elif years_experience < 10:
        experience_level = "senior"
    elif years_experience < 15:
        experience_level = "lead"
    else:
        experience_level = "director"

    # Create candidate profile
    profile = CandidateProfile(
        schema_version="1.0",
        candidate_id="local-user",
        target_roles=[],
        skills=skills,
        professional_experience=[],
        projects=[],
        education=education,
        certifications=[],
        location=[],
        remote_preference=None,
        experience_level=experience_level,  # type: ignore
        years_experience=years_experience,
        work_authorization=None,
        resume_path=str(path),
        resume_raw_text=resume_text,
    )

    return profile
