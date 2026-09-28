import pytest
from pathlib import Path
from src.models.job import JobRecord, ClassifiedJobRecord
from src.models.candidate import CandidateProfile
from src.ingestion.normalize import normalize_apify_record
from src.ingestion.resume_parser import parse_resume
from src.ingestion.requirement_classifier import classify_job_record
from src.ingestion.deduplicate import deduplicate
from src.filtering.hard_filters import apply_hard_filters

# --- TEST 1: Schema Roundtrip ---
def test_job_schema_roundtrip():
    valid_raw_job = {
        "id": "apify_123",
        "url": "https://example.com/job",
        "title": "Software Engineer",
        "company": "Sinclair Digital",
        "description": "Building full-stack apps in Go and React."
    }
    
    job = normalize_apify_record(valid_raw_job)
    assert isinstance(job, JobRecord)
    assert job.normalized_title == "software engineer"
    
    malformed_job = {"company": "No Title Tech"} 
    result = normalize_apify_record(malformed_job)
    assert result is None

# --- TEST 2: Real Resume Parsing ---
def test_resume_parsing_real_file():
    resume_path = Path("tests/fixtures/resumes/resume.pdf")
    
    if not resume_path.exists():
        pytest.skip(f"Real resume not found at {resume_path}")
        
    candidate = parse_resume(str(resume_path))
    assert isinstance(candidate, CandidateProfile)
    assert candidate.years_experience == 1 
    
    extracted_skills = [skill.lower() for skill in candidate.skills.languages + candidate.skills.frameworks]
    for expected_skill in ["c++", "python", "go", "react"]:
        assert expected_skill in extracted_skills
        
    degrees = [edu.degree.lower() for edu in candidate.education]
    assert any("computer engineering" in d for d in degrees)
    
    institutions = [edu.institution.lower() for edu in candidate.education]
    assert any("davis" in i for i in institutions)

# --- TEST 3: Requirement Classification ---
@pytest.mark.parametrize("sentence, expected_classification", [
    ("Required: Bachelor's degree in Computer Science or related technical field.", "hard"),
    ("Must have valid work authorization.", "hard"),
    ("3+ years of experience preferred.", "preferred"),
    ("Nice to have: Experience with AWS or GCP.", "preferred"),
    ("Experience building scalable backend APIs.", "unknown"),
    ("Familiarity with Agile methodologies.", "unknown"),
])
def test_requirement_classification(sentence, expected_classification):
    dummy_job = JobRecord(
        schema_version="1.0",
        job_id="test_req",
        source="test",
        source_url="http://test.com",
        apply_url=None,
        title="Test",
        normalized_title="test",
        company="Test",
        normalized_company="test",
        location=["Remote"],
        remote_type="remote",
        employment_type="full-time",
        seniority=None,
        description_raw=sentence,
        summary="",
        responsibilities=[],
        required_qualifications=[sentence],
        preferred_qualifications=[],
        technologies=[],
        education_requirements=[],
        experience_requirements=[],
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        posted_at=None,
        searchable_text=sentence,
        content_hash="hash"
    )
    
    classified = classify_job_record(dummy_job)
    all_reqs = classified.required_qualifications_with_evidence + classified.preferred_qualifications_with_evidence
    
    status_matches = [
        req.classification == expected_classification 
        for req in all_reqs 
        if getattr(req, "text", getattr(req, "evidence_sentence", None)) == sentence
    ]
    assert any(status_matches), f"Expected '{sentence}' to be classified as {expected_classification}"

# --- TEST 4: Deduplication ---
def test_deduplication():
    base_job = JobRecord(
        schema_version="1.0", job_id="id_1", source="apify", source_url="http://a.com", apply_url=None,
        title="Backend Dev", normalized_title="backend dev", company="Tech Corp", normalized_company="tech corp",
        location=["Seattle, WA"], remote_type="hybrid", employment_type="full-time", seniority="junior",
        description_raw="Desc 1", summary="", responsibilities=[], required_qualifications=[],
        preferred_qualifications=[], technologies=[], education_requirements=[], experience_requirements=[],
        salary_min=None, salary_max=None, salary_currency=None, posted_at=None, searchable_text="text",
        content_hash="hash_1"
    )
    
    clone_pk = base_job.model_copy(deep=True)
    
    clone_hash = base_job.model_copy(deep=True)
    clone_hash.job_id = "id_2" 
    
    distinct_job = base_job.model_copy(deep=True)
    distinct_job.job_id = "id_3"
    distinct_job.title = "Frontend Dev"
    distinct_job.content_hash = "hash_3"
    
    job_list = [base_job, clone_pk, clone_hash, distinct_job]
    unique_jobs = deduplicate(job_list)
    
    assert len(unique_jobs) == 2
    assert {j.job_id for j in unique_jobs} == {"id_1", "id_3"}

# --- TEST 5: Hard Filters Calibration ---
def test_hard_filters_calibration():
    candidate = CandidateProfile(
        candidate_id="ivan_yung",
        skills={"languages": ["Python", "Go"], "frameworks": ["React"]},
        professional_experience=[{"title": "Associate Engineer", "company": "Sinclair"}],
        education=[{"degree": "B.S. Computer Engineering", "institution": "UC Davis"}],
        location=["Seattle, WA"],
        remote_preference="onsite",
        experience_level="junior",
        years_experience=1.0,
        work_authorization="authorized"
    )
    
    def make_mock_job(job_id, location, reqs, pref_reqs, remote_type="onsite"):
        return ClassifiedJobRecord(
            schema_version="1.0", job_id=job_id, source="apify", source_url="http://a.com", apply_url=None,
            title="Dev", normalized_title="dev", company="Co", normalized_company="co",
            location=[location], remote_type=remote_type, employment_type="full-time", seniority=None,
            description_raw="", summary="", responsibilities=[], required_qualifications=[],
            preferred_qualifications=[], technologies=[], education_requirements=[], experience_requirements=[],
            salary_min=None, salary_max=None, salary_currency=None, posted_at=None, searchable_text="",
            content_hash=job_id,
            required_qualifications_with_evidence=reqs,
            preferred_qualifications_with_evidence=pref_reqs
        )

    jobs = [
        make_mock_job("job_perfect", "Seattle, WA", [], []),
        make_mock_job("job_related_degree", "Seattle, WA", 
                      [{"classification": "hard", "text": "B.S. in Computer Science or related", "evidence_sentence": "B.S. in Computer Science or related"}], []),
        make_mock_job("job_pref_senior", "Seattle, WA", [], 
                      [{"classification": "preferred", "text": "5+ years of experience preferred", "evidence_sentence": "5+ years of experience preferred"}]),
        make_mock_job("job_bad_location", "New York, NY", [], [], remote_type="onsite"),
        make_mock_job("job_hard_senior", "Seattle, WA", 
                      [{"classification": "hard", "text": "Minimum 5 years of mandatory experience", "evidence_sentence": "Minimum 5 years of mandatory experience"}], [])
    ]
    
    retained, filtered = apply_hard_filters(candidate, jobs)
    
    retained_ids = {j.job_id for j in retained}
    filtered_ids = {j.job_id for j in filtered}
    
    assert "job_perfect" in retained_ids
    assert "job_related_degree" in retained_ids
    assert "job_pref_senior" in retained_ids
    
    assert "job_bad_location" in filtered_ids
    assert "job_hard_senior" in filtered_ids