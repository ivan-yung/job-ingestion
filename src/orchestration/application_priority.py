"""Application-priority scoring layered on top of validated judge results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.llm.schemas import FinalScore, JudgeResponse, Recommendation


DEFAULT_PRIORITY_WEIGHTS = {
    "qualification": 0.60,
    "location_fit": 0.15,
    "remote_fit": 0.15,
    "company_interest": 0.10,
}


class ApplicationPriorityConfig(BaseModel):
    """Validated weights and company preferences for final application ordering."""

    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_PRIORITY_WEIGHTS))
    company_allowlist: list[str] = Field(default_factory=list)
    company_denylist: list[str] = Field(default_factory=list)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        expected = set(DEFAULT_PRIORITY_WEIGHTS)
        if set(weights) != expected:
            raise ValueError(f"weights must contain exactly: {sorted(expected)}")
        if any(value < 0 for value in weights.values()):
            raise ValueError("application priority weights must be non-negative")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("application priority weights must sum to 1.0")
        return weights


class ApplicationPriority(BaseModel):
    """A final priority score with the judge's decision carried through."""

    job_id: str | None = None
    application_priority_score: float = Field(ge=0, le=1)
    recommendation: Recommendation
    needs_review: bool
    final_score: FinalScore
    judge_response: JudgeResponse | None = None
    location_fit: float = Field(ge=0, le=1)
    remote_fit: float = Field(ge=0, le=1)
    company_interest: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _company_interest(company: str, config: ApplicationPriorityConfig) -> float:
    normalized = company.casefold().strip()
    allowlist = {item.casefold().strip() for item in config.company_allowlist}
    denylist = {item.casefold().strip() for item in config.company_denylist}
    if normalized in allowlist:
        return 1.0
    if normalized in denylist:
        return 0.0
    return 0.5


def application_priority(
    final_score: FinalScore,
    location_fit: float,
    remote_fit: float,
    company: str,
    config: ApplicationPriorityConfig,
    *,
    judge_response: JudgeResponse | None = None,
    job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ApplicationPriority:
    """Combine existing qualification and convenience signals without overrides."""
    company_interest = _company_interest(company, config)
    weights = config.weights
    priority_score = (
        weights["qualification"] * final_score.qualification_score / 100
        + weights["location_fit"] * location_fit
        + weights["remote_fit"] * remote_fit
        + weights["company_interest"] * company_interest
    )

    return ApplicationPriority(
        job_id=job_id,
        application_priority_score=priority_score,
        recommendation=final_score.recommendation,
        needs_review=final_score.needs_review,
        final_score=final_score,
        judge_response=judge_response,
        location_fit=location_fit,
        remote_fit=remote_fit,
        company_interest=company_interest,
        metadata=metadata or {},
    )