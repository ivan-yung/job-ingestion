"""Job record normalization and cleaning."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from src.models.job import JobRecord
from src.utils.hashing import sha256_hash


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    - Normalize unicode
    - Unescape HTML entities
    - Replace multiple spaces with single space
    - Replace multiple newlines with double newlines
    - Remove leading/trailing whitespace

    Args:
        text: Text to normalize.

    Returns:
        Normalized text.
    """
    if not text:
        return ""

    # Normalize unicode composition
    text = unicodedata.normalize("NFKC", text)

    # Unescape HTML entities
    text = html.unescape(text)

    # Normalize line endings
    text = re.sub(r"\r\n?", "\n", text)

    # Normalize tabs and other whitespace
    text = re.sub(r"[\t\f\v]+", " ", text)

    # Remove trailing spaces on each line
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)

    # Replace 3+ newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with single space (but preserve newlines)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags and preserve text content.

    Args:
        text: Text potentially containing HTML.

    Returns:
        Text with HTML tags removed.
    """
    if not text:
        return ""

    # Simple HTML tag removal
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_text(value: Any) -> str | None:
    """Clean and normalize text value.

    Args:
        value: Value to clean (may be None or non-string).

    Returns:
        Cleaned text or None if empty.
    """
    if value is None:
        return None

    text = str(value)
    text = normalize_whitespace(text)
    text = remove_html_tags(text)

    return text or None


def normalize_title(title: str) -> str:
    """Normalize job title for deduplication and comparison.

    - Lowercase
    - Remove special characters
    - Remove common suffixes
    - Collapse whitespace

    Args:
        title: Raw job title.

    Returns:
        Normalized title.
    """
    if not title:
        return ""

    # Lowercase
    title = title.lower().strip()

    # Remove common role suffixes
    title = re.sub(r"\s*\(.*?\)\s*", " ", title)

    # Remove special characters (keep alphanumeric, spaces, plus, sharp)
    title = re.sub(r"[^\w\s+#]", " ", title)

    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()

    return title


def normalize_company(company: str) -> str:
    """Normalize company name for deduplication and comparison.

    - Lowercase
    - Remove common suffixes (Inc., LLC, Ltd., etc.)
    - Remove special characters
    - Collapse whitespace

    Args:
        company: Raw company name.

    Returns:
        Normalized company name.
    """
    if not company:
        return ""

    # Lowercase
    company = company.lower().strip()

    # Remove common suffixes
    suffixes = [
        r"\binc\.?$",
        r"\bllc\.?$",
        r"\bltd\.?$",
        r"\bcorp\.?$",
        r"\bco\.?$",
        r"\binc\. \& co\.?$",
        r"\bplc\.?$",
    ]

    for suffix in suffixes:
        company = re.sub(suffix, "", company)

    # Remove special characters (keep alphanumeric, spaces, hyphens)
    company = re.sub(r"[^\w\s\-]", " ", company)

    # Collapse whitespace
    company = re.sub(r"\s+", " ", company).strip()

    return company


def normalize_location(location: str) -> str:
    """Normalize location string.

    Args:
        location: Raw location string.

    Returns:
        Normalized location.
    """
    if not location:
        return ""

    # Normalize whitespace
    location = normalize_whitespace(location).lower()

    # Common location normalizations
    location = location.replace("san francisco", "san francisco").replace("sf", "san francisco")

    return location


def extract_sections(description: str) -> dict[str, str]:
    """Extract common sections from job description.

    Attempts to identify sections like:
    - Summary/Overview
    - Responsibilities
    - Required Qualifications
    - Preferred Qualifications
    - Technologies/Stack
    - Education
    - Experience

    Args:
        description: Full job description text.

    Returns:
        Dictionary with extracted sections.
    """
    sections = {
        "summary": "",
        "responsibilities": "",
        "required_qualifications": "",
        "preferred_qualifications": "",
        "technologies": "",
        "education": "",
        "experience": "",
    }

    if not description:
        return sections

    text = description.lower()
    lines = description.split("\n")

    # Define section markers
    markers = {
        "summary": [
            r"^\s*(summary|overview|about the role|role description)",
        ],
        "responsibilities": [
            r"^\s*(responsibilities|what you'll do|what you will do|key responsibilities)",
        ],
        "required_qualifications": [
            r"^\s*(required|must have|requirements|required qualifications|essential)",
        ],
        "preferred_qualifications": [
            r"^\s*(preferred|nice to have|preferred qualifications)",
        ],
        "technologies": [
            r"^\s*(tech stack|technologies|tools|stack)",
        ],
        "education": [
            r"^\s*(education|degree required|educational requirements)",
        ],
        "experience": [
            r"^\s*(experience|years of experience)",
        ],
    }

    # Simple section detection
    current_section = None
    current_content = []

    for line in lines:
        line_lower = line.lower()

        # Check if this line starts a new section
        found_section = False
        for section_name, patterns in markers.items():
            for pattern in patterns:
                if re.match(pattern, line_lower):
                    # Save previous section
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = section_name
                    current_content = []
                    found_section = True
                    break
            if found_section:
                break

        # Add line to current section
        if not found_section and current_section and line.strip():
            current_content.append(line)

    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def remove_boilerplate(text: str) -> str:
    """Remove or minimize common boilerplate text.

    Args:
        text: Text that may contain boilerplate.

    Returns:
        Text with boilerplate minimized.
    """
    if not text:
        return ""

    # Remove common EEO/legal statements
    eeo_patterns = [
        r"(?i)equal opportunity employer.*?(?:\n|$)",
        r"(?i)we are an equal opportunity employer.*?(?:\n|$)",
        r"(?i)affirmative action.*?(?:\n|$)",
        r"(?i)disability accommodation.*?(?:\n|$)",
        r"(?i)if you need an accommodation.*?(?:\n|$)",
    ]

    for pattern in eeo_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def create_searchable_text(job: JobRecord) -> str:
    """Create concatenated searchable text from job record.

    Args:
        job: Job record.

    Returns:
        Concatenated searchable text.
    """
    parts = [
        job.title,
        job.company,
        " ".join(job.location),
        job.summary,
        " ".join(job.responsibilities),
        " ".join(job.required_qualifications),
        " ".join(job.preferred_qualifications),
        " ".join(job.technologies),
        " ".join(job.education_requirements),
        " ".join(job.experience_requirements),
    ]

    text = " ".join(p for p in parts if p)
    text = remove_boilerplate(text)

    return text


def normalize_apify_record(raw_record: dict[str, Any]) -> JobRecord | None:
    """Normalize an Apify raw job record to JobRecord schema.

    This is the primary adapter for the Apify format. Field names follow Apify's
    typical output schema.

    Args:
        raw_record: Raw job record from Apify dataset.

    Returns:
        Normalized JobRecord or None if required fields are missing.
    """
    # Try multiple field name variants common in Apify output
    title = clean_text(
        raw_record.get("title")
        or raw_record.get("jobTitle")
        or raw_record.get("positionName")
    )
    description = clean_text(
        raw_record.get("description")
        or raw_record.get("jobDescription")
        or raw_record.get("description_raw")
    )
    company = clean_text(
        raw_record.get("company")
        or raw_record.get("companyName")
        or raw_record.get("company_name")
    )
    source_url = clean_text(raw_record.get("url") or raw_record.get("jobUrl"))
    apply_url = clean_text(
        raw_record.get("apply_url")
        or raw_record.get("applyUrl")
        or raw_record.get("applicationUrl")
    )

    # Validate required fields
    if not title or not description or not company:
        return None

    # Extract job_id (use Apify's id field if available)
    job_id = raw_record.get("id") or raw_record.get("job_id") or f"apify_{hash(title + company) % 10**9}"

    # Rest of normalization follows standard path
    return _complete_normalization(
        title=title,
        description=description,
        company=company,
        source_url=source_url,
        apply_url=apply_url,
        job_id=str(job_id),
        raw_record=raw_record,
        source="apify",
    )


def _complete_normalization(
    title: str,
    description: str,
    company: str,
    source_url: str | None,
    apply_url: str | None,
    job_id: str,
    raw_record: dict[str, Any],
    source: str = "apify",
) -> JobRecord:
    """Complete the normalization process for extracted fields.

    Args:
        title: Cleaned job title.
        description: Cleaned job description.
        company: Cleaned company name.
        source_url: URL where job was found.
        apply_url: Direct application URL.
        job_id: Job identifier.
        raw_record: Original raw record (for extracting other fields).
        source: Data source identifier.

    Returns:
        Normalized JobRecord.
    """
    # Extract sections
    sections = extract_sections(description)

    # Create normalized record
    normalized_title = normalize_title(title)
    normalized_company = normalize_company(company)

    # Parse remote type
    remote_text = (description or "").lower()
    if "remote" in remote_text and "hybrid" in remote_text:
        remote_type = "hybrid"
    elif "remote" in remote_text or "wfh" in remote_text or "work from home" in remote_text:
        remote_type = "remote"
    elif "onsite" in remote_text or "on-site" in remote_text or "in-office" in remote_text:
        remote_type = "onsite"
    else:
        remote_type = "unknown"

    # Extract salary if present
    salary_min = salary_max = salary_currency = None
    salary_pattern = r"\$?([\d,]+)\s*(?:k|K)?"
    salary_matches = re.findall(salary_pattern, description)
    if salary_matches:
        try:
            values = [float(m.replace(",", "")) for m in salary_matches]
            if values:
                salary_min = min(values) * 1000 if min(values) < 500 else min(values)
                salary_max = max(values) * 1000 if max(values) < 500 else max(values)
                salary_currency = "USD"
        except (ValueError, TypeError):
            pass

    # Create searchable text and hash
    location = clean_text(raw_record.get("location", "")) or "Unknown"
    location_list = [location]
    
    searchable_parts = [
        title,
        company,
        description,
        sections.get("responsibilities", ""),
        sections.get("required_qualifications", ""),
        sections.get("technologies", ""),
    ]
    searchable_text = " ".join(p for p in searchable_parts if p)

    # Create fingerprint for deduplication
    fingerprint_input = f"{normalized_company}||{normalized_title}||{location}||{description[:500]}"
    content_hash = sha256_hash(fingerprint_input)

    job = JobRecord(
        schema_version="1.0",
        job_id=job_id,
        source=source,
        source_url=source_url,
        apply_url=apply_url,
        title=title,
        normalized_title=normalized_title,
        company=company,
        normalized_company=normalized_company,
        location=location_list,
        remote_type=remote_type,  # type: ignore
        employment_type=clean_text(raw_record.get("employmentType")) or None,
        seniority=None,  # Will be extracted during requirement classification
        description_raw=description,
        summary=sections.get("summary", description[:200]),
        responsibilities=split_items(sections.get("responsibilities", "")),
        required_qualifications=split_items(sections.get("required_qualifications", "")),
        preferred_qualifications=split_items(sections.get("preferred_qualifications", "")),
        technologies=split_items(sections.get("technologies", "")),
        education_requirements=split_items(sections.get("education", "")),
        experience_requirements=split_items(sections.get("experience", "")),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        posted_at=None,  # Would parse from raw_record if date field exists
        searchable_text=searchable_text,
        content_hash=content_hash,
    )

    return job


def normalize_job_record(
    raw_record: dict[str, Any],
    source: str = "apify",
) -> JobRecord | None:
    """Normalize and validate a raw job record.

    This is a generic entry point that delegates to source-specific normalizers.

    Args:
        raw_record: Raw job record from source.
        source: Data source identifier.

    Returns:
        Normalized JobRecord or None if validation fails.
    """
    if source == "apify":
        return normalize_apify_record(raw_record)

    return None


def split_items(text: str, delimiter: str = r"[\n\-•*]") -> list[str]:
    """Split text into list items by common delimiters.

    Args:
        text: Text to split.
        delimiter: Regex pattern for delimiters.

    Returns:
        List of cleaned items.
    """
    if not text:
        return []

    items = re.split(delimiter, text)
    items = [clean_text(item) for item in items]
    items = [item for item in items if item]

    return items
