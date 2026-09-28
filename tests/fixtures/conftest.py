"""Test fixtures for job normalization and filtering."""

import json
from pathlib import Path

# Sample raw jobs for testing
SAMPLE_RAW_JOBS = [
    {
        "id": "job_001",
        "title": "Senior Python Developer",
        "company": "Tech Corp Inc.",
        "description": """
        We are looking for a Senior Python Developer to join our team.
        
        Responsibilities:
        - Design and build scalable systems
        - Mentor junior developers
        - Code review and architecture discussions
        
        Required Qualifications:
        - 5+ years of Python experience
        - Experience with FastAPI or Django
        - Bachelor's degree in Computer Science
        
        Preferred Qualifications:
        - Kubernetes experience
        - AWS expertise
        - Open source contributions
        
        Tech Stack: Python, FastAPI, PostgreSQL, Docker, AWS
        """,
        "location": "San Francisco, CA",
        "url": "https://example.com/jobs/001",
    },
    {
        "id": "job_002",
        "jobTitle": "Python Developer",
        "companyName": "Tech Corp Inc",
        "jobDescription": """
        Senior Python Developer position.
        Requirements: 5+ years Python, FastAPI, Bachelor's in CS
        Preferred: Kubernetes, AWS
        Tech: Python, FastAPI, PostgreSQL, Docker, AWS
        Location: San Francisco, CA
        """,
        "url": "https://example.com/jobs/002",
    },
    {
        "id": "job_003",
        "title": "Junior Python Developer",
        "company": "StartUp Co",
        "description": """
        Looking for a Junior Python Developer (entry-level).
        
        Responsibilities:
        - Develop features for our web application
        - Write unit tests
        - Participate in code reviews
        
        Required:
        - 0-2 years Python experience
        - Knowledge of web frameworks
        - Willing to learn
        
        Nice to have:
        - JavaScript experience
        - SQL knowledge
        
        Tech: Python, Django, PostgreSQL
        """,
        "location": "Remote",
    },
    {
        "id": "job_004",
        "title": "Frontend React Developer",
        "company": "WebCorp Ltd",
        "description": """
        Join our frontend team!
        Requirements: 3+ years React, TypeScript required
        Preferred: Next.js, GraphQL
        """,
    },
]

# Sample resumes for testing
SAMPLE_RESUME_TEXT = """
John Doe
john.doe@example.com | (555) 123-4567 | linkedin.com/in/johndoe | github.com/johndoe

PROFESSIONAL SUMMARY
Senior full-stack engineer with 6+ years of experience building scalable web applications and leading technical teams.

PROFESSIONAL EXPERIENCE

Senior Software Engineer | TechCorp Inc. | San Francisco, CA | 2021 - Present (3 years)
- Led development of microservices architecture serving 1M+ users
- Mentored team of 4 junior engineers
- Technologies: Python, FastAPI, PostgreSQL, Docker, Kubernetes, AWS

Software Engineer | StartupXYZ | San Francisco, CA | 2019 - 2021 (2 years)
- Built REST APIs using Django and FastAPI
- Implemented automated testing and CI/CD pipelines
- Technologies: Python, Django, React, PostgreSQL, Docker

Junior Developer | WebDev Co | 2017 - 2019 (2 years)
- Developed and maintained web applications
- Fixed bugs and optimized performance
- Technologies: Python, JavaScript, HTML/CSS

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2017

TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, SQL
Frameworks: FastAPI, Django, React, Next.js
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS (EC2, S3, RDS, Lambda), GCP
Tools: Docker, Kubernetes, Git, Jenkins, GitHub

CERTIFICATIONS
AWS Certified Solutions Architect (2022)
"""


def create_test_fixtures():
    """Create fixture files for testing."""
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Create jobs fixtures
    jobs_dir = fixtures_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(SAMPLE_RAW_JOBS, 1):
        with open(jobs_dir / f"job_{i:03d}.json", "w") as f:
            json.dump(job, f, indent=2)

    # Create batch jobs file
    with open(jobs_dir / "batch_jobs.json", "w") as f:
        json.dump(SAMPLE_RAW_JOBS, f, indent=2)

    # Create resume fixtures
    resumes_dir = fixtures_dir / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    with open(resumes_dir / "sample_resume.txt", "w") as f:
        f.write(SAMPLE_RESUME_TEXT)


if __name__ == "__main__":
    create_test_fixtures()
