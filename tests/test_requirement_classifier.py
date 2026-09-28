"""Tests for requirement classification."""

import pytest

from src.ingestion.requirement_classifier import (
    classify_job_record,
    classify_requirement,
    extract_seniority,
)
from src.ingestion.normalize import normalize_job_record


class TestRequirementClassification:
    """Tests for requirement classification."""

    def test_classify_requirement_hard(self):
        """Test hard requirement detection."""
        cases = [
            "Must have 5+ years of Python",
            "Required: Bachelor's degree",
            "Essential: experience with Docker",
            "Mandatory: US work authorization",
        ]

        for req in cases:
            classification = classify_requirement(req)
            assert classification == "hard", f"Failed for: {req}"

    def test_classify_requirement_preferred(self):
        """Test preferred requirement detection."""
        cases = [
            "Preferred: Kubernetes experience",
            "Nice to have: AWS knowledge",
            "A plus: Open source contributions",
            "Desirable: MBA degree",
        ]

        for req in cases:
            classification = classify_requirement(req)
            assert classification == "preferred", f"Failed for: {req}"

    def test_classify_requirement_unknown(self):
        """Test unknown/ambiguous requirement detection."""
        cases = [
            "Should have experience with React",
            "Can work independently",
            "Might need to travel occasionally",
        ]

        for req in cases:
            classification = classify_requirement(req)
            assert classification == "unknown", f"Failed for: {req}"

    def test_classify_requirement_negation(self):
        """Test negated requirements."""
        req = "Not required: experience with C++"
        classification = classify_requirement(req)
        # Should not classify as hard if negated
        assert classification != "hard"


class TestSeniorityExtraction:
    """Tests for seniority level extraction."""

    def test_extract_seniority_explicit(self):
        """Test explicit seniority indicators."""
        cases = [
            ("Entry-level Python Developer", "junior"),
            ("Senior Software Engineer", "senior"),
            ("Mid-level Developer", "mid"),
            ("Lead Engineer", "lead"),
            ("Director of Engineering", "director"),
        ]

        for description, expected in cases:
            seniority = extract_seniority(description)
            assert seniority == expected, f"Failed for: {description}"

    def test_extract_seniority_by_years(self):
        """Test seniority extraction from years of experience."""
        cases = [
            ("0-1 years of experience", "new_grad"),
            ("1-2 years of experience", "junior"),
            ("3-5 years of experience", "mid"),
            ("7+ years of experience", "senior"),
            ("12+ years in software engineering", "director"),
        ]

        for description, expected in cases:
            seniority = extract_seniority(description)
            assert seniority == expected, f"Failed for: {description}"

    def test_extract_seniority_none(self):
        """Test no seniority found."""
        description = "General software developer position"
        seniority = extract_seniority(description)
        assert seniority is None


class TestJobRecordClassification:
    """Tests for complete job record classification."""

    def test_classify_job_record_basic(self):
        """Test basic job classification."""
        raw_job = {
            "title": "Senior Python Developer",
            "company": "TechCorp",
            "description": """
            Required: 5+ years Python
            Preferred: Kubernetes experience
            """,
        }

        job = normalize_job_record(raw_job)
        assert job is not None

        classified = classify_job_record(job)

        assert classified.seniority == "senior"

    def test_classify_job_record_extracts_seniority(self):
        """Test seniority extraction in classification."""
        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "We are looking for a developer with 8+ years of experience",
        }

        job = normalize_job_record(raw_job)
        assert job is not None

        classified = classify_job_record(job)

        assert classified.seniority == "senior"
