"""Evidence validation, caching, bounded judging, and run accounting."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from src.llm.client import AttemptUsage, OpenAIResponsesTransport, RawCompletion, ResponsesTransport
from src.llm.schemas import (
    JudgeConfig,
    JudgeResponse,
    Recommendation,
    RequirementEvaluation,
    RequirementInput,
    RequirementStatus,
    prompt_version,
)
from src.models.job import JobRecord
from src.scoring import final_qualification_score
from src.utils.hashing import deterministic_json_hash


@dataclass
class JudgeMetrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    transport_retries: int = 0
    semantic_retries: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    total_accounted_cost_usd: float = 0.0
    latencies: list[float] = field(default_factory=list)
    budget_stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        ordered = sorted(self.latencies)
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "transport_retries": self.transport_retries,
            "semantic_retries": self.semantic_retries,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "total_accounted_cost_usd": self.total_accounted_cost_usd,
            "latency_p50": _percentile(ordered, 0.50),
            "latency_p95": _percentile(ordered, 0.95),
            "budget_stop_reason": self.budget_stop_reason,
        }


@dataclass
class JudgeResult:
    job_id: str
    judge_status: str
    response: JudgeResponse | None = None
    score: Any | None = None
    error: str | None = None
    cache_hit: bool = False
    model_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgableJob:
    """Validated handoff from deterministic reranking to LLM judging."""

    job: JobRecord
    rerank_result: dict[str, Any] | None = None

    @property
    def job_id(self) -> str:
        return self.job.job_id


def resolve_reranked_jobs(
    reranked_jobs: list[JobRecord | dict[str, Any] | JudgableJob],
    jobs_by_id: dict[str, JobRecord] | None = None,
) -> list[JudgableJob]:
    """Resolve reranker results to validated jobs with an explicit contract."""
    resolved: list[JudgableJob] = []
    for item in reranked_jobs:
        if isinstance(item, JudgableJob):
            resolved.append(item)
            continue
        if isinstance(item, JobRecord):
            resolved.append(JudgableJob(job=item))
            continue
        if not isinstance(item, dict):
            raise TypeError("each judged item must be a JobRecord, JudgableJob, or reranker dictionary")

        embedded_job = item.get("job")
        if isinstance(embedded_job, JobRecord):
            resolved.append(JudgableJob(job=embedded_job, rerank_result=item))
            continue
        if isinstance(embedded_job, dict):
            try:
                resolved.append(JudgableJob(job=JobRecord.model_validate(embedded_job), rerank_result=item))
            except Exception as exc:
                raise ValueError("reranker dictionary contains an invalid 'job' payload") from exc
            continue

        job_id = item.get("job_id")
        if not isinstance(job_id, str) or jobs_by_id is None or job_id not in jobs_by_id:
            raise ValueError(
                "reranker result requires a JobRecord in 'job' or a matching jobs_by_id entry"
            )
        resolved.append(JudgableJob(job=jobs_by_id[job_id], rerank_result=item))
    return resolved


@dataclass
class _Reservation:
    amount: float
    released: bool = False


class SQLiteJudgeCache:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_judgments (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    def get(self, cache_key: str) -> tuple[JudgeResponse, dict[str, Any]] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT result_json, usage_json FROM llm_judgments WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return JudgeResponse.model_validate_json(row[0]), json.loads(row[1])

    def put(self, cache_key: str, response: JudgeResponse, raw_response: str, usage: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_judgments
                   (cache_key, result_json, raw_response, usage_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (cache_key, response.model_dump_json(), raw_response, json.dumps(usage), time.time()),
            )


def evidence_is_grounded(evidence: str | None, source_text: str) -> bool:
    if not evidence:
        return False

    def normalize(value: str) -> str:
        value = value.lower().replace("\u2018", "'").replace("\u2019", "'")
        value = value.replace("\u201c", '"').replace("\u201d", '"')
        return re.sub(r"\s+", " ", value).strip()

    return normalize(evidence) in normalize(source_text)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return values[index]


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    payload = candidate.model_dump() if hasattr(candidate, "model_dump") else dict(candidate)
    payload.pop("resume_path", None)
    return payload


def _job_payload(job: Any) -> dict[str, Any]:
    payload = job.model_dump() if hasattr(job, "model_dump") else dict(job)
    return {
        key: payload.get(key, [])
        for key in (
            "title", "seniority", "required_qualifications", "preferred_qualifications",
            "technologies", "responsibilities", "education_requirements", "experience_requirements",
            "summary",
        )
    }


def _requirements(job: Any) -> list[RequirementInput]:
    evidence_groups = (
        getattr(job, "required_qualifications_with_evidence", []),
        getattr(job, "preferred_qualifications_with_evidence", []),
    )
    inputs: list[RequirementInput] = []
    for group in evidence_groups:
        for item in group:
            text = getattr(item, "text", None) or item.get("text")
            classification = getattr(item, "classification", None) or item.get("classification")
            if text and classification in {"hard", "preferred", "unknown"}:
                inputs.append(RequirementInput(text=text, classification=classification))
    if inputs:
        return inputs
    inputs.extend(RequirementInput(text=text, classification="hard") for text in getattr(job, "required_qualifications", []))
    inputs.extend(RequirementInput(text=text, classification="preferred") for text in getattr(job, "preferred_qualifications", []))
    return inputs


def _prompt(candidate: Any, job: Any, requirements: list[RequirementInput], correction: str | None = None) -> str:
    payload = {
        "job": _job_payload(job),
        "requirements": [item.model_dump() for item in requirements],
        "candidate": _candidate_payload(candidate),
    }
    suffix = f"\nCORRECTION REQUIRED: {correction}" if correction else ""
    return (
        "Evaluate the structured data below. Every requirement must appear exactly once, in order, "
        "in the list matching its classification. Quote evidence verbatim from the supplied data."
        "\n<job_data>" + json.dumps(payload["job"], ensure_ascii=True) + "</job_data>"
        "\n<requirements>" + json.dumps(payload["requirements"], ensure_ascii=True) + "</requirements>"
        "\n<candidate_data>" + json.dumps(payload["candidate"], ensure_ascii=True) + "</candidate_data>"
        + suffix
    )


def _cache_key(job: Any, candidate: Any, model_id: str) -> str:
    job_hash = getattr(job, "content_hash", None) or deterministic_json_hash(_job_payload(job))
    candidate_hash = deterministic_json_hash(_candidate_payload(candidate))
    return deterministic_json_hash({
        "job_content_hash": job_hash,
        "candidate_profile_hash": candidate_hash,
        "prompt_version": prompt_version(),
        "model_id": model_id,
    })


def _validate_completeness(response: JudgeResponse, requirements: list[RequirementInput]) -> None:
    expected_required = [item.text.casefold() for item in requirements if item.classification in {"hard", "unknown"}]
    expected_preferred = [item.text.casefold() for item in requirements if item.classification == "preferred"]
    actual_required = [item.requirement.casefold() for item in response.required_requirements]
    actual_preferred = [item.requirement.casefold() for item in response.preferred_requirements]
    if actual_required != expected_required or actual_preferred != expected_preferred:
        raise ValueError("response requirements are incomplete or out of order")


def _validate_evidence(response: JudgeResponse, candidate_source: str, job_source: str) -> None:
    evaluations = response.required_requirements + response.preferred_requirements
    for evaluation in evaluations:
        if evaluation.status in {RequirementStatus.MATCHED, RequirementStatus.PARTIALLY_MATCHED}:
            if not evidence_is_grounded(evaluation.candidate_evidence, candidate_source):
                raise ValueError(f"ungrounded candidate evidence for {evaluation.requirement}")
            if evaluation.job_evidence is not None and not evidence_is_grounded(evaluation.job_evidence, job_source):
                raise ValueError(f"ungrounded job evidence for {evaluation.requirement}")
    for strength in response.strengths:
        if not evidence_is_grounded(strength.candidate_evidence, candidate_source):
            raise ValueError(f"ungrounded strength evidence for {strength.claim}")


def _estimate_cost(prompt: str, config: JudgeConfig, model_id: str) -> float:
    pricing = config.pricing[model_id]
    input_tokens = max(1, len(prompt) // 4)
    return (input_tokens * pricing["input_per_mtok"] + config.max_output_tokens * pricing["output_per_mtok"]) / 1_000_000


class _Budget:
    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.committed = 0.0
        self.reserved = 0.0
        self.lock = asyncio.Lock()

    async def reserve(self, amount: float) -> _Reservation | None:
        async with self.lock:
            if self.committed + self.reserved + amount > self.limit:
                return None
            self.reserved += amount
            return _Reservation(amount)

    async def settle(self, reservation: _Reservation, actual_cost: float) -> None:
        async with self.lock:
            if reservation.released:
                return
            self.reserved -= reservation.amount
            self.committed += max(0.0, actual_cost)
            reservation.released = True


def _record_attempts(
    metrics: JudgeMetrics,
    attempts: list[AttemptUsage],
    config: JudgeConfig,
    model_id: str,
    prompt: str,
    default_outcome: str = "transport_error",
) -> float:
    """Record each attempt once and return its accounted cost."""
    if not attempts:
        attempts = [AttemptUsage(None, None, 0.0, default_outcome, default_outcome != "success")]
    total = 0.0
    for attempt in attempts:
        input_tokens = attempt.input_tokens or max(1, len(prompt) // 4)
        output_tokens = attempt.output_tokens or config.assumed_output_tokens
        cost = (
            input_tokens * config.pricing[model_id]["input_per_mtok"]
            + output_tokens * config.pricing[model_id]["output_per_mtok"]
        ) / 1_000_000
        metrics.requests += 1
        metrics.latencies.append(attempt.latency_seconds)
        metrics.total_accounted_cost_usd += cost
        total += cost
        if attempt.estimated:
            metrics.estimated_cost_usd += cost
        else:
            metrics.actual_cost_usd += cost
            if attempt.input_tokens is not None:
                metrics.input_tokens += attempt.input_tokens
            if attempt.output_tokens is not None:
                metrics.output_tokens += attempt.output_tokens
    metrics.transport_retries += max(0, len(attempts) - 1)
    return total


async def judge_jobs_async(
    candidate_profile: Any,
    reranked_jobs: list[Any],
    config: JudgeConfig,
    transport: ResponsesTransport | None = None,
    jobs_by_id: dict[str, JobRecord] | None = None,
) -> tuple[list[JudgeResult], JudgeMetrics]:
    """Judge reranked jobs independently with bounded concurrency."""
    config.validate_pricing()
    transport = transport or OpenAIResponsesTransport()
    cache = SQLiteJudgeCache(config.cache_db_path)
    metrics = JudgeMetrics()
    budget = _Budget(config.max_total_cost_usd)
    semaphore = asyncio.Semaphore(config.max_concurrency)
    jobs = resolve_reranked_jobs(
        reranked_jobs[: config.max_llm_jobs_per_run],
        jobs_by_id=jobs_by_id,
    )

    async def judge_one(judgable_job: JudgableJob) -> JudgeResult:
        job = judgable_job.job
        model_id = config.model_id
        key = _cache_key(job, candidate_profile, model_id)
        cached = cache.get(key)
        if cached is not None:
            response, usage = cached
            metrics.cache_hits += 1
            return JudgeResult(job_id=job.job_id, judge_status="accepted", response=response, score=final_qualification_score(response, _requirements(job)), cache_hit=True, model_id=model_id, usage=usage)

        requirements = _requirements(job)
        candidate_source = json.dumps(_candidate_payload(candidate_profile), ensure_ascii=True)
        job_source = json.dumps(_job_payload(job), ensure_ascii=True)
        semantic_retry = 0
        api_attempts = 0
        max_attempts = max(1, config.max_transport_retries + config.max_semantic_retries)
        correction = None
        async with semaphore:
            while True:
                attempt_prompt = _prompt(candidate_profile, job, requirements, correction)
                reservation = await budget.reserve(_estimate_cost(attempt_prompt, config, model_id))
                if reservation is None:
                    metrics.budget_stop_reason = "cost_reservation_exceeded"
                    return JudgeResult(job_id=job.job_id, judge_status="budget_skipped", error="cost budget exhausted", model_id=model_id)
                completion: RawCompletion | None = None
                try:
                    completion = await transport.complete(
                        prompt=attempt_prompt,
                        model_id=model_id,
                        timeout_seconds=config.request_timeout_s,
                        max_transport_retries=min(
                            config.max_transport_retries,
                            max(0, max_attempts - api_attempts - 1),
                        ),
                        max_output_tokens=config.max_output_tokens,
                        retry_base_backoff_seconds=config.retry_base_backoff_seconds,
                        retry_max_backoff_seconds=config.retry_max_backoff_seconds,
                    )
                    completion_attempts = completion.attempts or [
                        AttemptUsage(
                            completion.input_tokens,
                            completion.output_tokens,
                            completion.latency_seconds,
                            "success",
                            False,
                        )
                    ]
                    api_attempts += len(completion_attempts)
                    attempt_cost = _record_attempts(metrics, completion_attempts, config, model_id, attempt_prompt, "success")
                    await budget.settle(reservation, attempt_cost)
                    response = JudgeResponse.model_validate(completion.payload)
                    _validate_completeness(response, requirements)
                    _validate_evidence(response, candidate_source, job_source)
                    usage = {"input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens, "latency_seconds": completion.latency_seconds, "transport_retries": completion.transport_retries}
                    cache.put(key, response, completion.raw_response, usage)
                    metrics.successes += 1
                    return JudgeResult(job_id=job.job_id, judge_status="accepted", response=response, score=final_qualification_score(response, requirements), model_id=model_id, usage=usage)
                except Exception as exc:
                    if completion is None:
                        attempt_cost = _record_attempts(
                            metrics,
                            getattr(exc, "attempts", []),
                            config,
                            model_id,
                            attempt_prompt,
                        )
                        api_attempts += max(1, len(getattr(exc, "attempts", [])))
                    else:
                        attempt_cost = 0.0
                    await budget.settle(reservation, attempt_cost)
                    if semantic_retry >= config.max_semantic_retries or api_attempts >= max_attempts:
                        metrics.failures += 1
                        return JudgeResult(job_id=job.job_id, judge_status="failed", error=str(exc), model_id=model_id)
                    semantic_retry += 1
                    metrics.semantic_retries += 1
                    correction = str(exc)

    results = await asyncio.gather(*(judge_one(job) for job in jobs))
    return results, metrics


def judge_jobs(
    candidate_profile: Any,
    reranked_jobs: list[Any],
    config: JudgeConfig,
    transport: ResponsesTransport | None = None,
    jobs_by_id: dict[str, JobRecord] | None = None,
) -> tuple[list[JudgeResult], JudgeMetrics]:
    """Synchronous wrapper; async callers must use ``judge_jobs_async``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            judge_jobs_async(
                candidate_profile,
                reranked_jobs,
                config,
                transport,
                jobs_by_id,
            )
        )
    raise RuntimeError("judge_jobs() called inside a running event loop; await judge_jobs_async() instead")
