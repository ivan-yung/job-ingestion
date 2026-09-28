"""Tests for hard filtering logic."""

import pytest

from src.filtering.hard_filters import (
    apply_hard_filters,
    check_hard_filters,
    filter_by_education,
    filter_by_employment_type,
    filter_by_experience_years,
    filter_by_location,
    filter_by_remote_preference,
    filter_by_work_authorization,
)
from src.ingestion.normalize import normalize_job_record
from src.ingestion.requirement_classifier import classify_job_record
from src.ingestion.resume_parser import CandidateProfile, Skills
from src.models.candidate import CandidateProfile


class TestWorkAuthorizationFilter:
    """Tests for work authorization filtering."""

    def test_passes_when_candidate_auth_known(self):
        """Test that jobs pass when candidate work auth is known and compatible."""
        candidate = CandidateProfile(work_authorization=None)

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "Requires US work authorization",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_work_authorization(candidate, classified_job)

        assert result.passed

    def test_rejects_visa_without_sponsorship(self):
        """Test rejection when visa needed but not offered."""
        candidate = CandidateProfile(work_authorization="VISA_SPONSOR_NEEDED")

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "No visa sponsorship available",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_work_authorization(candidate, classified_job)

        assert not result.passed


class TestRemotePreferenceFilter:
    """Tests for remote preference filtering."""

    def test_passes_flexible_candidate(self):
        """Test that flexible candidates pass all remote types."""
        candidate = CandidateProfile(remote_preference=None)

        for remote_type in ["remote", "hybrid", "onsite"]:
            raw_job = {
                "title": "Developer",
                "company": "TechCorp",
                "description": f"This is a {remote_type} position",
            }

            job = normalize_job_record(raw_job)
            classified_job = classify_job_record(job)

            result = filter_by_remote_preference(candidate, classified_job)

            assert result.passed

    def test_rejects_remote_only_candidate_for_onsite(self):
        """Test rejection when candidate wants remote but job is onsite."""
        candidate = CandidateProfile(remote_preference="remote")

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "This is an onsite position in our office",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_remote_preference(candidate, classified_job)

        assert not result.passed

    def test_accepts_hybrid_for_remote_candidate(self):
        """Test that hybrid jobs can accommodate remote-seeking candidates."""
        candidate = CandidateProfile(remote_preference="hybrid")

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "Hybrid: 3 days remote, 2 in office",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_remote_preference(candidate, classified_job)

        assert result.passed


class TestLocationFilter:
    """Tests for location filtering."""

    def test_passes_no_location_specified(self):
        """Test that jobs pass when no location filtering needed."""
        candidate = CandidateProfile(location=[])

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "Remote position",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_location(candidate, classified_job)

        assert result.passed

    def test_rejects_location_mismatch_onsite(self):
        """Test rejection when onsite job doesn't match candidate location."""
        candidate = CandidateProfile(
            location=["New York, NY"],
        )

        raw_job = {
            "id": "test_job",
            "title": "Developer",
            "company": "TechCorp",
            "description": "Onsite position in San Francisco",
            "location": "San Francisco, CA",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_location(candidate, classified_job)

        assert not result.passed


class TestEducationFilter:
    """Tests for education filtering."""

    def test_passes_no_education_requirement(self):
        """Test that jobs without education requirements pass."""
        candidate = CandidateProfile(education=[])

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "No specific education required",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_education(candidate, classified_job)

        assert result.passed

    def test_passes_candidate_with_education(self):
        """Test that candidates with education pass education requirements."""
        from src.models.candidate import Education

        candidate = CandidateProfile(
            education=[
                Education(
                    institution="UC Berkeley",
                    degree="B.S.",
                    field="Computer Science",
                )
            ]
        )

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "Bachelor's degree in Computer Science required",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_education(candidate, classified_job)

        assert result.passed


class TestExperienceYearsFilter:
    """Tests for years of experience filtering."""

    def test_passes_sufficient_experience(self):
        """Test that candidates with sufficient experience pass."""
        candidate = CandidateProfile(years_experience=6)

        raw_job = {
            "title": "Senior Developer",
            "company": "TechCorp",
            "description": "5+ years of experience required",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_experience_years(candidate, classified_job)

        assert result.passed

    def test_passes_junior_for_flexible_requirement(self):
        """Test that junior candidates pass non-strict requirements."""
        candidate = CandidateProfile(years_experience=1)

        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "5+ years preferred",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_experience_years(candidate, classified_job)

        # Preferred requirements should not hard reject
        assert result.passed

    def test_rejects_severe_experience_mismatch(self):
        """Test rejection when experience is severely insufficient."""
        candidate = CandidateProfile(years_experience=0.5)

        raw_job = {
            "title": "Senior Developer",
            "company": "TechCorp",
            "description": "15+ years of experience required",
        }

        job = normalize_job_record(raw_job)
        classified_job = classify_job_record(job)

        result = filter_by_experience_years(candidate, classified_job)

        assert not result.passed


class TestHardFilterIntegration:
    """Integration tests for hard filtering."""

    def test_apply_hard_filters_retains_good_match(self):
        """Test that a good match is retained."""
        candidate = CandidateProfile(
            years_experience=6,
            remote_preference="hybrid",
            location=["San Francisco, CA"],
        )

        raw_jobs = [
            {
                "id": "job_001",
                "title": "Senior Python Developer",
                "company": "TechCorp",
                "description": """
                We are hiring a Senior Python Developer.
                5+ years of experience required.
                Hybrid role in San Francisco.
                """,
                "location": "San Francisco, CA",
            }
        ]

        jobs = [normalize_job_record(job) for job in raw_jobs]
        jobs = [classify_job_record(job) for job in jobs if job]

        retained, decisions = apply_hard_filters(candidate, jobs)

        assert len(retained) == 1
        assert len(decisions) == 1
        assert not decisions[0].rejected

    def test_apply_hard_filters_rejects_bad_match(self):
        """Test that a bad match is rejected."""
        candidate = CandidateProfile(
            years_experience=1,
            remote_preference="remote",
            location=["New York, NY"],
        )

        raw_jobs = [
            {
                "id": "job_001",
                "title": "Senior Python Developer",
                "company": "TechCorp",
                "description": """
                Senior Python Developer - 10+ years required.
                Onsite in San Francisco.
                """,
                "location": "San Francisco, CA",
            }
        ]

        jobs = [normalize_job_record(job) for job in raw_jobs]
        jobs = [classify_job_record(job) for job in jobs if job]

        retained, decisions = apply_hard_filters(candidate, jobs)

        assert len(retained) == 0
        assert len(decisions) == 1
        assert decisions[0].rejected
        assert len(decisions[0].rejection_reasons) > 0
