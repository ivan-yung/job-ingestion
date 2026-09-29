import asyncio
import json

import pytest

from src.llm.judge import RawCompletion, judge_jobs
from src.llm.schemas import (
    JudgeConfig,
    JudgeResponse,
    Recommendation,
    RequirementEvaluation,
    RequirementInput,
    RequirementStatus,
)
from src.models.candidate import CandidateProfile, Skills
from src.models.job import JobRecord
from src.scoring import final_qualification_score


class FakeTransport:
    def __init__(self, payloads, delay=0):
        self.payloads = list(payloads)
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.delay = delay

    async def complete(self, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
            if isinstance(payload, Exception):
                raise payload
            return RawCompletion(
                payload=payload,
                model_id=kwargs["model_id"],
                input_tokens=100,
                output_tokens=50,
                latency_seconds=0.01,
                transport_retries=0,
                raw_response=json.dumps(payload),
            )
        finally:
            self.active -= 1


def make_candidate():
    return CandidateProfile(
        candidate_id="candidate",
        target_roles=["Backend Engineer"],
        skills=Skills(languages=["Python"]),
        resume_raw_text="Professional experience with Python.",
    )


def make_job(job_id="job", required=None, preferred=None):
    return JobRecord(
        job_id=job_id,
        source="test",
        title="Backend Engineer",
        normalized_title="backend engineer",
        company="Example",
        normalized_company="example",
        description_raw="Python backend role.",
        summary="Python backend role.",
        required_qualifications=required or [],
        preferred_qualifications=preferred or [],
        technologies=[],
        education_requirements=[],
        experience_requirements=[],
        responsibilities=[],
        searchable_text="Python backend role.",
        content_hash=f"hash-{job_id}",
    )


def valid_payload(requirement="Python", status="matched"):
    return {
        "overall_fit": 90,
        "skills_fit": 90,
        "experience_fit": 80,
        "role_alignment": 90,
        "education_fit": 70,
        "required_requirements": [
            {
                "requirement": requirement,
                "status": status,
                "candidate_evidence": "Professional experience with Python." if status == "matched" else None,
                "job_evidence": "Python" if status == "matched" else None,
            }
        ],
        "preferred_requirements": [],
        "strengths": [],
        "concerns": [],
        "uncertainties": [],
        "recommendation": "strong_apply",
    }


def config(tmp_path, **overrides):
    values = {
        "cache_db_path": str(tmp_path / "judge.sqlite"),
        "pricing": {"test-model": {"input_per_mtok": 1, "output_per_mtok": 1}},
        "model_id": "test-model",
    }
    values.update(overrides)
    return JudgeConfig(**values)


def test_output_schema_validation(tmp_path):
    malformed = {**valid_payload(), "skills_fit": 101}
    transport = FakeTransport([malformed, valid_payload()])
    results, metrics = judge_jobs(make_candidate(), [make_job(required=["Python"])], config(tmp_path), transport)

    assert results[0].judge_status == "accepted"
    assert transport.calls == 2
    assert metrics.semantic_retries == 1


def test_cache_key_invalidation(tmp_path, monkeypatch):
    candidate = make_candidate()
    job = make_job(required=["Python"])
    transport = FakeTransport([valid_payload()])
    cfg = config(tmp_path)

    first, _ = judge_jobs(candidate, [job], cfg, transport)
    second, metrics = judge_jobs(candidate, [job], cfg, transport)
    assert first[0].response == second[0].response
    assert metrics.cache_hits == 1
    assert transport.calls == 1

    for changed_job, changed_candidate, changed_model in [
        (make_job(required=["Python"], job_id="changed"), candidate, "test-model"),
        (job, candidate.model_copy(update={"candidate_id": "changed"}), "test-model"),
        (job, candidate, "other-model"),
    ]:
        changed_cfg = config(tmp_path, model_id=changed_model, pricing={
            "test-model": {"input_per_mtok": 1, "output_per_mtok": 1},
            "other-model": {"input_per_mtok": 1, "output_per_mtok": 1},
        })
        changed_transport = FakeTransport([valid_payload()])
        judge_jobs(changed_candidate, [changed_job], changed_cfg, changed_transport)
        assert changed_transport.calls == 1

    import src.llm.schemas as schemas
    old_prompt = schemas.PROMPT_TEMPLATE
    schemas.PROMPT_TEMPLATE = old_prompt + " changed"
    try:
        changed_transport = FakeTransport([valid_payload()])
        judge_jobs(candidate, [job], cfg, changed_transport)
        assert changed_transport.calls == 1
    finally:
        schemas.PROMPT_TEMPLATE = old_prompt


def test_evidence_grounding(tmp_path):
    invalid = valid_payload()
    invalid["required_requirements"][0]["candidate_evidence"] = "Invented experience"
    corrected = valid_payload()
    transport = FakeTransport([invalid, corrected])
    results, metrics = judge_jobs(make_candidate(), [make_job(required=["Python"])], config(tmp_path), transport)

    assert results[0].judge_status == "accepted"
    assert metrics.semantic_retries == 1


def test_final_score_hard_requirement_override(tmp_path):
    missing = JudgeResponse(
        overall_fit=100,
        skills_fit=100,
        experience_fit=100,
        role_alignment=100,
        education_fit=100,
        required_requirements=[RequirementEvaluation(requirement="Python", status=RequirementStatus.MISSING)],
        preferred_requirements=[],
        strengths=[],
        concerns=[],
        uncertainties=[],
        recommendation=Recommendation.STRONG_APPLY,
    )
    score = final_qualification_score(missing, [RequirementInput(text="Python", classification="hard")])
    assert score.qualification_score <= 39
    assert score.recommendation == Recommendation.DO_NOT_APPLY

    uncertain = missing.model_copy(update={
        "required_requirements": [RequirementEvaluation(requirement="Python", status=RequirementStatus.UNCERTAIN)],
    })
    uncertain_score = final_qualification_score(uncertain, [RequirementInput(text="Python", classification="hard")])
    assert uncertain_score.needs_review is True
    assert uncertain_score.qualification_score > 39

    omitted = valid_payload()
    omitted["required_requirements"] = []
    transport = FakeTransport([omitted, valid_payload()])
    results, metrics = judge_jobs(make_candidate(), [make_job(required=["Python"])], config(tmp_path), transport)
    assert results[0].judge_status == "accepted"
    assert metrics.semantic_retries == 1


def test_concurrency_and_cost_budget(tmp_path):
    jobs = [make_job(f"job-{index}") for index in range(6)]
    transport = FakeTransport([valid_payload(requirement="unused")], delay=0.01)
    cfg = config(
        tmp_path,
        max_concurrency=2,
        max_total_cost_usd=0.0011,
        assumed_output_tokens=1000,
    )
    results, metrics = judge_jobs(make_candidate(), jobs, cfg, transport)

    assert transport.max_active <= 2
    assert len(results) == len(jobs)
    assert sum(result.judge_status == "accepted" for result in results) < len(jobs)
    assert metrics.budget_stop_reason == "cost_reservation_exceeded"
    assert metrics.estimated_cost_usd >= 0
