"""Tests for job deduplication."""

import pytest

from src.ingestion.deduplicate import (
    deduplicate,
    find_duplicates,
    is_duplicate,
    merge_duplicates,
)
from src.ingestion.normalize import normalize_job_record


class TestDuplicateDetection:
    """Tests for duplicate detection."""

    def test_is_duplicate_same_source_and_id(self):
        """Test duplicate detection by source and ID."""
        raw_job = {
            "id": "job_001",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "A great job",
        }

        job1 = normalize_job_record(raw_job, source="apify")
        job2 = normalize_job_record(raw_job, source="apify")

        assert job1 is not None
        assert job2 is not None
        assert is_duplicate(job1, job2)

    def test_is_duplicate_same_content_hash(self):
        """Test duplicate detection by content hash."""
        raw_job1 = {
            "id": "job_001",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "A great job with specific details.",
        }

        raw_job2 = {
            "id": "job_002",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "A great job with specific details.",
        }

        job1 = normalize_job_record(raw_job1, source="apify")
        job2 = normalize_job_record(raw_job2, source="apify")

        assert job1 is not None
        assert job2 is not None
        # Same content should produce same hash
        assert job1.content_hash == job2.content_hash
        assert is_duplicate(job1, job2)

    def test_is_not_duplicate_different_jobs(self):
        """Test that different jobs are not marked as duplicates."""
        raw_job1 = {
            "id": "job_001",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "Python job 1",
        }

        raw_job2 = {
            "id": "job_002",
            "title": "JavaScript Developer",
            "company": "WebCorp",
            "description": "JavaScript job 2",
        }

        job1 = normalize_job_record(raw_job1)
        job2 = normalize_job_record(raw_job2)

        assert job1 is not None
        assert job2 is not None
        assert not is_duplicate(job1, job2)


class TestFindDuplicates:
    """Tests for finding duplicate groups."""

    def test_find_duplicates_basic(self):
        """Test basic duplicate group detection."""
        raw_jobs = [
            {
                "id": "job_001",
                "title": "Python Developer",
                "company": "TechCorp",
                "description": "Position A",
            },
            {
                "id": "job_001",
                "title": "Python Developer",
                "company": "TechCorp",
                "description": "Position A",
            },
        ]

        jobs = [normalize_job_record(job) for job in raw_jobs]
        jobs = [j for j in jobs if j is not None]

        groups = find_duplicates(jobs)

        assert len(groups) == 1
        assert len(groups[0].duplicates) == 1

    def test_find_duplicates_none(self):
        """Test no duplicates found."""
        raw_jobs = [
            {
                "id": "job_001",
                "title": "Python Developer",
                "company": "TechCorp",
                "description": "Position A",
            },
            {
                "id": "job_002",
                "title": "JavaScript Developer",
                "company": "WebCorp",
                "description": "Position B",
            },
        ]

        jobs = [normalize_job_record(job) for job in raw_jobs]
        jobs = [j for j in jobs if j is not None]

        groups = find_duplicates(jobs)

        assert len(groups) == 0


class TestDeduplication:
    """Tests for deduplication."""

    def test_deduplicate_basic(self):
        """Test basic deduplication."""
        raw_jobs = [
            {
                "id": "job_001",
                "title": "Python Developer",
                "company": "TechCorp",
                "description": "Position A",
            },
            {
                "id": "job_001",
                "title": "Python Developer",
                "company": "TechCorp",
                "description": "Position A",
            },
            {
                "id": "job_002",
                "title": "JavaScript Developer",
                "company": "WebCorp",
                "description": "Position B",
            },
        ]

        jobs = [normalize_job_record(job) for job in raw_jobs]
        jobs = [j for j in jobs if j is not None]

        unique_jobs = deduplicate(jobs)

        # Should have 2 unique jobs (first Python job kept, second removed, JS job kept)
        assert len(unique_jobs) == 2

    def test_deduplicate_preserves_order(self):
        """Test that deduplication preserves first occurrence."""
        raw_job = {
            "id": "job_001",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "Position A",
        }

        job1 = normalize_job_record(raw_job)
        job2 = normalize_job_record(raw_job)

        assert job1 is not None
        assert job2 is not None

        # Change job_id for second instance to test content hash match
        job2.job_id = "job_002"

        unique = deduplicate([job1, job2])

        assert len(unique) == 1
        assert unique[0].job_id == "job_001"  # First one preserved


class TestMergeDuplicates:
    """Tests for merging duplicate records."""

    def test_merge_duplicates_complete_data(self):
        """Test merging fills in missing data."""
        raw_job1 = {
            "id": "job_001",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "Position A",
            "location": "San Francisco, CA",
        }

        raw_job2 = {
            "id": "job_002",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "Position A",
            "location": "New York, NY",
        }

        job1 = normalize_job_record(raw_job1)
        job2 = normalize_job_record(raw_job2)

        assert job1 is not None
        assert job2 is not None

        merged = merge_duplicates(job1, job2)

        # Should have both locations
        assert "San Francisco, CA" in merged.location
        assert "New York, NY" in merged.location

    def test_merge_duplicates_prefers_primary(self):
        """Test that merge prefers primary job data."""
        raw_job1 = {
            "id": "job_001",
            "title": "Senior Python Developer",
            "company": "TechCorp",
            "description": "Position A",
            "url": "https://example.com/job1",
        }

        raw_job2 = {
            "id": "job_002",
            "title": "Python Developer",
            "company": "TechCorp",
            "description": "Position A",
        }

        job1 = normalize_job_record(raw_job1)
        job2 = normalize_job_record(raw_job2)

        assert job1 is not None
        assert job2 is not None

        merged = merge_duplicates(job1, job2)

        # Should keep primary job's title
        assert merged.title == job1.title
