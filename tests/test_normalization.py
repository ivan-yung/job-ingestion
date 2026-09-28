"""Tests for job normalization."""

import pytest

from src.ingestion.normalize import (
    clean_text,
    extract_sections,
    normalize_company,
    normalize_job_record,
    normalize_title,
    normalize_whitespace,
    remove_boilerplate,
    remove_html_tags,
    split_items,
)


class TestWhitespaceNormalization:
    """Tests for whitespace normalization."""

    def test_normalize_whitespace_basic(self):
        """Test basic whitespace normalization."""
        text = "This   has   multiple   spaces"
        result = normalize_whitespace(text)
        assert result == "This has multiple spaces"

    def test_normalize_whitespace_newlines(self):
        """Test newline normalization."""
        text = "Line 1\n\n\nLine 2\n\n\n\nLine 3"
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_normalize_whitespace_html_entities(self):
        """Test HTML entity unescaping."""
        text = "Salary: &euro;50,000 &ndash; &euro;100,000"
        result = normalize_whitespace(text)
        assert "€" in result

    def test_normalize_whitespace_empty(self):
        """Test empty string normalization."""
        assert normalize_whitespace("") == ""
        assert normalize_whitespace("   ") == ""


class TestHTMLRemoval:
    """Tests for HTML tag removal."""

    def test_remove_html_tags_simple(self):
        """Test simple HTML tag removal."""
        text = "<p>Hello <b>World</b></p>"
        result = remove_html_tags(text)
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result

    def test_remove_html_tags_complex(self):
        """Test complex HTML removal."""
        text = "<div><strong>Requirements:</strong><ul><li>Python</li><li>Django</li></ul></div>"
        result = remove_html_tags(text)
        assert "<" not in result
        assert "Requirements" in result
        assert "Python" in result


class TestCleanText:
    """Tests for text cleaning."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "  Hello   World  "
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_text_none(self):
        """Test None handling."""
        assert clean_text(None) is None

    def test_clean_text_empty_string(self):
        """Test empty string returns None."""
        assert clean_text("") is None
        assert clean_text("   ") is None


class TestTitleNormalization:
    """Tests for job title normalization."""

    def test_normalize_title_lowercase(self):
        """Test title is lowercased."""
        title = "Senior Python Developer"
        result = normalize_title(title)
        assert result == result.lower()

    def test_normalize_title_removes_parentheses(self):
        """Test parenthetical content removal."""
        title = "Python Developer (Remote)"
        result = normalize_title(title)
        assert "(Remote)" not in result
        assert "python" in result
        assert "developer" in result

    def test_normalize_title_special_chars(self):
        """Test special character removal."""
        title = "C++ / C# Developer"
        result = normalize_title(title)
        assert "/" not in result
        assert "developer" in result


class TestCompanyNormalization:
    """Tests for company name normalization."""

    def test_normalize_company_lowercase(self):
        """Test company is lowercased."""
        company = "GOOGLE Inc."
        result = normalize_company(company)
        assert result == result.lower()

    def test_normalize_company_removes_suffixes(self):
        """Test common suffix removal."""
        cases = [
            ("Google Inc.", "google"),
            ("Facebook LLC", "facebook"),
            ("Tesla Ltd.", "tesla"),
            ("IBM Corp.", "ibm"),
        ]

        for company, expected_substring in cases:
            result = normalize_company(company)
            assert expected_substring in result
            assert "inc" not in result
            assert "llc" not in result

    def test_normalize_company_removes_special_chars(self):
        """Test special character removal."""
        company = "Tech @ Co. Inc."
        result = normalize_company(company)
        assert "@" not in result
        assert "tech" in result


class TestSectionExtraction:
    """Tests for section extraction from job descriptions."""

    def test_extract_sections_basic(self):
        """Test basic section extraction."""
        description = """
        Summary: We are looking for a developer.
        
        Responsibilities:
        - Build systems
        - Code review
        
        Required Qualifications:
        - 5+ years Python
        - Bachelor's degree
        """

        sections = extract_sections(description)

        assert "summary" in sections
        assert "responsibilities" in sections
        assert "required_qualifications" in sections

    def test_extract_sections_empty_description(self):
        """Test empty description handling."""
        sections = extract_sections("")
        assert all(v == "" for v in sections.values())


class TestBoilerplateRemoval:
    """Tests for boilerplate removal."""

    def test_remove_boilerplate_eeo(self):
        """Test EEO statement removal."""
        text = "Join our team. Equal opportunity employer."
        result = remove_boilerplate(text)
        assert "Equal opportunity employer" not in result
        assert "Join our team" in result

    def test_remove_boilerplate_disability(self):
        """Test disability accommodation statement removal."""
        text = "Great job! If you need an accommodation, please contact us."
        result = remove_boilerplate(text)
        assert "accommodation" not in result.lower()


class TestSplitItems:
    """Tests for splitting items by delimiters."""

    def test_split_items_newline(self):
        """Test newline-delimited splitting."""
        text = "Python\nJavaScript\nGo"
        result = split_items(text)
        assert len(result) == 3
        assert "Python" in result

    def test_split_items_bullet_points(self):
        """Test bullet point splitting."""
        text = "• Python\n• JavaScript\n• Go"
        result = split_items(text)
        assert len(result) == 3

    def test_split_items_empty(self):
        """Test empty text returns empty list."""
        assert split_items("") == []
        assert split_items(None) == []


class TestJobNormalization:
    """Tests for complete job normalization."""

    def test_normalize_job_record_basic(self):
        """Test basic job normalization."""
        raw_job = {
            "title": "Senior Python Developer",
            "company": "Tech Corp Inc.",
            "description": "We are hiring a senior Python developer.",
            "location": "San Francisco, CA",
        }

        job = normalize_job_record(raw_job)

        assert job is not None
        assert job.title == "Senior Python Developer"
        assert job.normalized_title == "senior python developer"
        assert "tech corp" in job.normalized_company
        assert job.job_id is not None
        assert job.content_hash is not None

    def test_normalize_job_record_missing_required_fields(self):
        """Test job with missing required fields returns None."""
        raw_job = {
            "title": "Python Developer",
            # Missing company and description
        }

        job = normalize_job_record(raw_job)
        assert job is None

    def test_normalize_job_record_remote_type_detection(self):
        """Test remote type detection."""
        cases = [
            ("This is a remote position", "remote"),
            ("Work from home opportunity", "remote"),
            ("Hybrid: 3 days remote, 2 in office", "hybrid"),
            ("On-site only in our San Francisco office", "onsite"),
        ]

        for description_fragment, expected_remote_type in cases:
            raw_job = {
                "title": "Developer",
                "company": "TechCorp",
                "description": description_fragment,
            }

            job = normalize_job_record(raw_job)
            assert job is not None
            assert job.remote_type == expected_remote_type

    def test_normalize_job_record_content_hash_deterministic(self):
        """Test that content hash is deterministic."""
        raw_job = {
            "title": "Senior Python Developer",
            "company": "Tech Corp Inc.",
            "description": "We are hiring a senior Python developer with 5+ years.",
            "location": "San Francisco, CA",
        }

        job1 = normalize_job_record(raw_job)
        job2 = normalize_job_record(raw_job)

        assert job1 is not None
        assert job2 is not None
        assert job1.content_hash == job2.content_hash

    def test_normalize_job_record_salary_extraction(self):
        """Test salary extraction."""
        raw_job = {
            "title": "Developer",
            "company": "TechCorp",
            "description": "Salary: $150,000 - $200,000 per year",
        }

        job = normalize_job_record(raw_job)
        assert job is not None
        assert job.salary_min is not None
        assert job.salary_max is not None
        assert job.salary_currency == "USD"
