from unittest.mock import Mock

import pytest

from src.models.candidate import CandidateProfile, Education, Skills
from src.models.job import JobRecord
from src.reranking.normalizers import normalize_skill, normalize_title, title_similarity
from src.reranking.rerank import rerank_jobs
from src.reranking.scoring import RerankFeatures, calculate_score, extract_features


def make_candidate(**overrides):
    values = {
        "target_roles": ["Software Engineer"],
        "skills": Skills(languages=["Python", "JavaScript"], frameworks=["React"]),
        "years_experience": 3,
        "experience_level": "mid",
        "education": [],
        "location": [],
    }
    values.update(overrides)
    return CandidateProfile(**values)


def make_job(job_id="job", title="Software Engineer", **overrides):
    values = {
        "schema_version": "1.0",
        "job_id": job_id,
        "source": "test",
        "title": title,
        "normalized_title": title.lower(),
        "company": "TechCorp",
        "normalized_company": "techcorp",
        "location": ["Remote"],
        "remote_type": "remote",
        "description_raw": "",
        "summary": "",
        "required_qualifications": [],
        "preferred_qualifications": [],
        "technologies": [],
        "education_requirements": [],
        "experience_requirements": [],
        "searchable_text": "",
        "content_hash": job_id,
    }
    values.update(overrides)
    return JobRecord(**values)


def test_skill_and_title_normalization():
    cases = [
        ("JS", "javascript"),
        ("Postgres", "postgresql"),
        ("Node", "node.js"),
        ("React Native", "react native"),
        ("C++", "c++"),
        ("C#", "c#"),
        ("AWS", "aws"),
        ("Azure", "azure"),
    ]
    for value, expected in cases:
        assert normalize_skill(value) == expected
    assert normalize_skill("React") != normalize_skill("React Native")
    assert normalize_skill("SQL") != normalize_skill("PostgreSQL")

    assert normalize_title("Senior Backend Engineer") == "backend"
    assert normalize_title("Embedded Software Engineer") == "embedded"
    assert normalize_title("Machine Learning Engineer") == "ml_ai"
    assert title_similarity(["Software Engineer"], "Senior Marketing Manager") == 0
    assert title_similarity(["Backend Engineer"], "Senior Backend Engineer") == 1


def test_feature_extraction():
    candidate = make_candidate(target_roles=["Backend Engineer"])
    job = make_job(
        title="Backend Engineer",
        required_qualifications=["Python"],
        preferred_qualifications=["Docker"],
        technologies=["Python"],
        experience_requirements=["2+ years required"],
    )
    features = extract_features(candidate, job, {"semantic_score": 0.8, "bm25_score": 10})

    assert features.required_skill_coverage == 1
    assert features.required_technology_coverage == 1
    assert features.preferred_skill_coverage == 0
    assert features.experience_fit == 1
    assert features.hard_requirement_status == "passed"
    for value in features.model_dump().values():
        if isinstance(value, (int, float)):
            assert 0 <= value <= 1


@pytest.mark.parametrize(
    ("candidate_years", "requirement", "status", "expected_fit"),
    [
        (0, "5+ years required", "failed", 0),
        (0, "2+ years preferred", "passed", 0),
    ],
)
def test_experience_and_education_fit(candidate_years, requirement, status, expected_fit):
    candidate = make_candidate(years_experience=candidate_years)
    job = make_job(experience_requirements=[requirement])
    features = extract_features(candidate, job, {})
    assert features.hard_requirement_status == status
    assert features.experience_fit == expected_fit

    educated = make_candidate(
        education=[Education(institution="State U", degree="Bachelor of Science", field="Computer Engineering")]
    )
    degree_job = make_job(education_requirements=["Bachelor's in Computer Science or related technical field"])
    assert extract_features(educated, degree_job, {}).education_fit == 1


def test_hard_requirement_gating():
    candidate = make_candidate()
    jobs = [
        make_job("missing-skill", required_qualifications=["Rust"]),
        make_job("missing-tech", technologies=["PostgreSQL"]),
        make_job("missing-degree", education_requirements=["Bachelor's in Computer Science"]),
        make_job("qualified", required_qualifications=["Python"], technologies=["Python"]),
    ]
    results = rerank_jobs(
        candidate,
        [{"job_id": job.job_id, "semantic_score": 1, "bm25_score": 1} for job in jobs],
        jobs,
    )
    assert [result["job_id"] for result in results] == ["qualified"]

    failed = RerankFeatures(
        semantic_score=1, bm25_signal=1, title_similarity=1,
        required_skill_coverage=1, preferred_skill_coverage=1,
        required_technology_coverage=1, preferred_technology_coverage=1,
        experience_fit=1, seniority_fit=1, education_fit=1, location_fit=1,
        hard_requirement_status="failed",
    )
    assert calculate_score(failed) == 0


def test_rerank_end_to_end():
    candidate = make_candidate(target_roles=["Full Stack Engineer"], years_experience=5)
    jobs = [
        make_job("best", "Full Stack Engineer", required_qualifications=["Python", "JavaScript"], technologies=["Python", "JavaScript"], experience_requirements=["3 years required"]),
        make_job("partial", "Backend Engineer", required_qualifications=["Python"], technologies=["Python"], experience_requirements=["5 years required"]),
        *[make_job(f"other-{index}", "Data Engineer", required_qualifications=["Python"], technologies=["Python"]) for index in range(8)],
    ]
    retrieval = [{"job_id": job.job_id, "semantic_score": 0.9, "bm25_score": 5} for job in jobs]
    ranked = rerank_jobs(candidate, retrieval, jobs, top_k=10)

    assert ranked[0]["job_id"] == "best"
    assert all("feature_breakdown" in result for result in ranked)
    assert all("qualification_score" in result for result in ranked)
    assert all(result["formula_version"] == "1.1" for result in ranked)
    assert all(set(result["feature_breakdown"]) == set(RerankFeatures.model_fields) for result in ranked)
    assert rerank_jobs(candidate, jobs, top_k=1)[0]["job_id"] == "best"