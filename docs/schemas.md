# Schema Documentation

## Overview

This document describes the normalized schemas used throughout the resume-ranker system.

## Job Record Schema

The `JobRecord` schema represents a normalized, validated job posting.

### Fields

#### Identification
- `schema_version` (string): Version of this schema (default: "1.0")
- `job_id` (string): Unique identifier for the job (required)
- `source` (string): Data source platform (e.g., "apify", "linkedin")
- `source_url` (string, optional): URL where job was found
- `apply_url` (string, optional): Direct application URL

#### Job Details
- `title` (string): Original job title (required)
- `normalized_title` (string): Normalized job title for matching (required)
- `company` (string): Original company name (required)
- `normalized_company` (string): Normalized company name (required)
- `location` (array of strings): Job locations/cities
- `remote_type` (enum): Type of remote work
  - "remote": Fully remote
  - "hybrid": Mix of remote and onsite
  - "onsite": In-office only
  - "unknown": Not specified
- `employment_type` (string, optional): "Full-time", "Part-time", "Contract", etc.
- `seniority` (string, optional): "junior", "mid", "senior", "lead", etc.

#### Job Description
- `description_raw` (string): Original job description (unchanged)
- `summary` (string): Brief summary of the role
- `responsibilities` (array): List of key responsibilities
- `required_qualifications` (array): Hard requirements
- `preferred_qualifications` (array): Nice-to-have qualifications
- `technologies` (array): Technologies/languages mentioned
- `education_requirements` (array): Education/degree requirements
- `experience_requirements` (array): Years/type of experience needed

#### Compensation
- `salary_min` (number, optional): Minimum salary
- `salary_max` (number, optional): Maximum salary
- `salary_currency` (string, optional): Currency code (e.g., "USD", "EUR")

#### Metadata
- `posted_at` (datetime, optional): When the job was posted
- `searchable_text` (string): Concatenated text for full-text search
- `content_hash` (string): SHA-256 hash for deduplication

### Example

```json
{
  "schema_version": "1.0",
  "job_id": "apify_12345",
  "source": "apify",
  "source_url": "https://example.com/jobs/12345",
  "apply_url": "https://example.com/apply/12345",
  "title": "Senior Python Developer",
  "normalized_title": "senior python developer",
  "company": "Tech Corp Inc.",
  "normalized_company": "tech corp",
  "location": ["San Francisco, CA"],
  "remote_type": "hybrid",
  "employment_type": "Full-time",
  "seniority": "senior",
  "description_raw": "...",
  "summary": "We are looking for a senior Python developer...",
  "responsibilities": [
    "Design and build scalable systems",
    "Lead technical discussions",
    "Mentor junior developers"
  ],
  "required_qualifications": [
    "5+ years Python experience",
    "Experience with FastAPI or Django"
  ],
  "preferred_qualifications": [
    "Kubernetes experience",
    "AWS expertise"
  ],
  "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker"],
  "education_requirements": ["Bachelor's in CS or related field"],
  "experience_requirements": ["5+ years"],
  "salary_min": 150000.0,
  "salary_max": 200000.0,
  "salary_currency": "USD",
  "posted_at": "2024-01-15T10:30:00",
  "searchable_text": "...",
  "content_hash": "abc123def456..."
}
```

## Classified Job Record Schema

The `ClassifiedJobRecord` extends `JobRecord` with classified requirements and evidence.

### Additional Fields

- `required_qualifications_with_evidence` (array): Hard requirements with source evidence
- `preferred_qualifications_with_evidence` (array): Preferred requirements with source evidence

### RequirementWithEvidence Structure

Each requirement includes:
- `text` (string): The requirement text
- `classification` (enum): "hard", "preferred", or "unknown"
- `evidence_sentence` (string): The original sentence from the description

### Example

```json
{
  "text": "5+ years Python experience",
  "classification": "hard",
  "evidence_sentence": "Must have 5+ years of Python experience with production systems."
}
```

## Candidate Profile Schema

The `CandidateProfile` schema represents a parsed resume and candidate information.

### Fields

#### Identification & Metadata
- `schema_version` (string): Version of this schema (default: "1.0")
- `candidate_id` (string): Unique identifier (default: "local-user")
- `resume_path` (string, optional): Path to source resume file
- `resume_raw_text` (string, optional): Raw extracted resume text

#### Career Goals
- `target_roles` (array): Target job titles/roles

#### Skills
- `skills` (Skills object):
  - `languages` (array): Programming languages
  - `frameworks` (array): Frameworks and libraries
  - `databases` (array): Database systems
  - `cloud` (array): Cloud platforms
  - `tools` (array): Tools and utilities
  - `other` (array): Other skills

#### Experience
- `professional_experience` (array): Work history
  - Each entry includes: title, company, duration, years, description, technologies
- `projects` (array): Portfolio projects
  - Each includes: name, description, technologies, url

#### Education
- `education` (array): Educational background
  - Each includes: institution, degree, field, graduation_year
- `certifications` (array): Professional certifications
  - Each includes: name, issuer, date

#### Location & Preferences
- `location` (array): Current/preferred locations
- `remote_preference` (enum, optional): "remote", "hybrid", "onsite", or null
- `work_authorization` (string, optional): Work authorization status

#### Experience Level
- `experience_level` (enum): Overall experience level
  - "new_grad", "junior", "mid", "senior", "lead", "manager", "director", "executive"
- `years_experience` (number): Total years of professional experience

### Example

```json
{
  "schema_version": "1.0",
  "candidate_id": "local-user",
  "target_roles": ["Senior Python Developer", "Tech Lead"],
  "skills": {
    "languages": ["Python", "JavaScript", "Rust"],
    "frameworks": ["FastAPI", "React", "Django"],
    "databases": ["PostgreSQL", "MongoDB"],
    "cloud": ["AWS", "GCP"],
    "tools": ["Docker", "Kubernetes", "Git"]
  },
  "professional_experience": [
    {
      "title": "Senior Engineer",
      "company": "TechCorp",
      "duration": "2021-2024",
      "years": 3,
      "description": "Led backend team...",
      "technologies": ["Python", "FastAPI", "PostgreSQL"]
    }
  ],
  "projects": [],
  "education": [
    {
      "institution": "University of California, Berkeley",
      "degree": "Bachelor of Science",
      "field": "Computer Science",
      "graduation_year": 2018
    }
  ],
  "certifications": [],
  "location": ["San Francisco, CA"],
  "remote_preference": "hybrid",
  "work_authorization": null,
  "experience_level": "senior",
  "years_experience": 6,
  "resume_path": "/path/to/resume.pdf",
  "resume_raw_text": "..."
}
```

## Requirement Classification

Requirements are classified into three categories:

### Hard Requirements
Must be explicitly stated with keywords like:
- "must", "required", "essential", "mandatory"

Examples:
- "Must have 5+ years of Python"
- "Required: Bachelor's degree in CS"
- "Essential: US work authorization"

### Preferred Requirements
Indicated by keywords:
- "preferred", "nice to have", "a plus", "desirable", "beneficial"

Examples:
- "Preferred: Kubernetes experience"
- "A plus: Open source contributions"
- "Nice to have: AWS expertise"

### Unknown/Ambiguous
Indicated by ambiguous language:
- "should", "can", "might", "may"

Or when classification is unclear.

## Normalization Rules

### Title Normalization
- Lowercase all characters
- Remove parenthetical content
- Remove special characters (except +, #)
- Collapse whitespace

### Company Normalization
- Lowercase all characters
- Remove common suffixes: Inc, LLC, Ltd, Corp, Co, PLC
- Remove special characters
- Collapse whitespace

### Content Hash Generation
- Combine: normalized_company + "|" + normalized_title + "|" + location + "|" + first 500 chars of description
- SHA-256 hash of combined string
- Used for deduplication

### Searchable Text
- Concatenate: title, company, summary, responsibilities, requirements, technologies
- Remove boilerplate (EEO statements, etc.)
- Used for full-text search

## Database Schema

### jobs table
```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_url TEXT,
    apply_url TEXT,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    company TEXT NOT NULL,
    normalized_company TEXT NOT NULL,
    remote_type TEXT,
    employment_type TEXT,
    seniority TEXT,
    content_hash TEXT NOT NULL,
    schema_version TEXT,
    description_raw TEXT,
    summary TEXT,
    responsibilities TEXT,
    required_qualifications TEXT,
    preferred_qualifications TEXT,
    technologies TEXT,
    education_requirements TEXT,
    experience_requirements TEXT,
    salary_min REAL,
    salary_max REAL,
    salary_currency TEXT,
    posted_at TEXT,
    location TEXT,
    searchable_text TEXT,
    raw_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### candidates table
```sql
CREATE TABLE candidates (
    candidate_id TEXT PRIMARY KEY,
    schema_version TEXT,
    resume_path TEXT,
    experience_level TEXT,
    years_experience REAL,
    work_authorization TEXT,
    remote_preference TEXT,
    raw_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### job_deduplication_log table
```sql
CREATE TABLE job_deduplication_log (
    primary_job_id TEXT NOT NULL,
    duplicate_job_id TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (primary_job_id, duplicate_job_id)
)
```
