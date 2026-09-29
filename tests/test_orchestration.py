import pytest

from src.config import PipelineConfig
from src.llm.judge import JudgeMetrics, JudgeResult
from src.llm.schemas import FinalScore, JudgeConfig, Recommendation
from src.models.candidate import CandidateProfile
from src.models.job import JobRecord
from src.orchestration.application_priority import ApplicationPriorityConfig, application_priority
from src.orchestration.pipeline import Pipeline
from src.reporting.report_builder import build_report
from src.reranking.rerank import RerankOutput


def make_config(tmp_path):
    return PipelineConfig(
        config_version="1",
        cache_db_path=str(tmp_path / "cache.sqlite"),
        judge=JudgeConfig(
            model_id="test-model",
            pricing={"test-model": {"input_per_mtok": 1, "output_per_mtok": 1}},
        ),
    )


def make_job(job_id="job-1", title="Engineer", company="Example"):
    return JobRecord(
        job_id=job_id,
        source="test",
        title=title,
        normalized_title=title.lower(),
        company=company,
        normalized_company=company.lower(),
        description_raw="Python role",
        summary="Python role",
        location=["Remote"],
        remote_type="remote",
        responsibilities=[],
        required_qualifications=[],
        preferred_qualifications=[],
        technologies=[],
        education_requirements=[],
        experience_requirements=[],
        searchable_text="Python role",
        content_hash=job_id,
    )


def make_candidate():
    return CandidateProfile(candidate_id="candidate", experience_level="junior", years_experience=1)


def test_application_priority_preserves_needs_review():
    final_score = FinalScore(
        qualification_score=35,
        recommendation=Recommendation.DO_NOT_APPLY,
        needs_review=True,
        required_requirements_score=0.2,
        preferred_requirements_score=0.5,
    )
    result = application_priority(
        final_score,
        location_fit=1,
        remote_fit=1,
        company="Example",
        config=ApplicationPriorityConfig(),
    )
    assert result.recommendation == Recommendation.DO_NOT_APPLY
    assert result.needs_review is True
    assert result.application_priority_score < 1


def test_config_fail_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_APIFY_TOKEN", raising=False)
    with pytest.raises(ValueError, match="No pricing configured"):
        PipelineConfig(
            config_version="1",
            judge=JudgeConfig(model_id="unpriced", pricing={}),
        )
    with pytest.raises(ValueError, match="weights must sum"):
        ApplicationPriorityConfig(weights={
            "qualification": 1,
            "location_fit": 0,
            "remote_fit": 0,
            "company_interest": 1,
        })
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """config_version = \"1\"\napify_api_token_env = \"MISSING_APIFY_TOKEN\"\n\n[judge]\nmodel_id = \"test-model\"\n\n[judge.pricing.test-model]\ninput_per_mtok = 1\noutput_per_mtok = 1\n""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing MISSING_APIFY_TOKEN"):
        PipelineConfig.from_toml(config_path, require_apify_token=True)


def test_report_groups_and_escapes(tmp_path):
    job = make_job(title="<Engineer>", company="Example & Co")
    payload = {
        "manifest": {"run_id": "run-1"},
        "candidate": make_candidate().model_dump(),
        "ingest": {"retained": [{"job": job.model_dump()}], "filtered": [{"job": make_job("filtered").model_dump(), "reasons": ["location"]}]},
        "rerank": {"scored": [{"job_id": "job-1", "feature_breakdown": {"location_fit": 1}}], "gated": [{"job_id": "gated", "failed_requirement": "Kubernetes", "feature_snapshot": {}}]},
        "llm_judge": {"results": [{"job_id": "job-1", "judge_status": "accepted", "response": {"overall_fit": 80, "skills_fit": 80, "experience_fit": 80, "role_alignment": 80, "education_fit": 80, "required_requirements": [], "preferred_requirements": [], "strengths": [], "concerns": [], "uncertainties": [], "recommendation": "apply"}, "score": {"qualification_score": 80, "recommendation": "apply", "needs_review": False, "required_requirements_score": 0.5, "preferred_requirements_score": 0.5}}]},
    }
    envelope = build_report(payload, tmp_path, make_config(tmp_path))
    html = (tmp_path / "report.html").read_text()
    assert len(envelope["ranked"]) == 1
    assert len(envelope["filtered"]) == 1
    assert len(envelope["gated"]) == 1
    assert "&lt;Engineer&gt;" in html
    assert "Example &amp; Co" in html


def test_pipeline_resumes_prefix_after_judge_failure(tmp_path, monkeypatch):
    import src.orchestration.pipeline as pipeline_module

    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("resume", encoding="utf-8")
    candidate = make_candidate()
    job = make_job()
    calls = {"retrieve": 0, "rerank": 0, "judge": 0}
    monkeypatch.setattr(pipeline_module, "parse_resume", lambda _: candidate)
    monkeypatch.setattr(pipeline_module, "fetch_apify_dataset", lambda *args, **kwargs: [job.model_dump()])
    monkeypatch.setattr(pipeline_module, "normalize_apify_record", lambda _: job)
    monkeypatch.setattr(pipeline_module, "retrieve_jobs", lambda *args, **kwargs: calls.__setitem__("retrieve", calls["retrieve"] + 1) or [{"job_id": "job-1", "semantic_score": 1, "bm25_score": 1}])
    monkeypatch.setattr(pipeline_module, "rerank_jobs", lambda *args, **kwargs: calls.__setitem__("rerank", calls["rerank"] + 1) or RerankOutput(scored=[{"job_id": "job-1", "qualification_score": 1, "feature_breakdown": {"location_fit": 1}}], gated=[]))

    def judge(*args, **kwargs):
        calls["judge"] += 1
        if calls["judge"] == 1:
            raise RuntimeError("simulated interruption")
        return [], JudgeMetrics()

    monkeypatch.setattr(pipeline_module, "judge_jobs", judge)
    pipeline = Pipeline(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run(run_id="run-1", run_dir=tmp_path / "run-1", resume_path=resume_path, apify_run_id="apify-1")
    pipeline.run(run_id="run-1", run_dir=tmp_path / "run-1", resume_path=resume_path, apify_run_id="apify-1")
    assert calls["retrieve"] == 1
    assert calls["rerank"] == 1
    assert calls["judge"] == 2

    resume_path.write_text("changed resume", encoding="utf-8")
    pipeline.run(run_id="run-1", run_dir=tmp_path / "run-1", resume_path=resume_path, apify_run_id="apify-1")
    assert calls["retrieve"] == 2
    assert calls["rerank"] == 2