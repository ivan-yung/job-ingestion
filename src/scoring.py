"""Deterministic final qualification scoring for validated LLM judgments."""

from __future__ import annotations

from src.llm.schemas import (
    FinalScore,
    JudgeResponse,
    Recommendation,
    RequirementInput,
    RequirementStatus,
    requirement_score,
)


def _matches_hard(requirement: str, hard_requirements: list[RequirementInput]) -> bool:
    normalized = requirement.strip().casefold()
    return any(
        item.classification == "hard" and item.text.strip().casefold() == normalized
        for item in hard_requirements
    )


def final_qualification_score(
    response: JudgeResponse,
    hard_requirements: list[RequirementInput],
) -> FinalScore:
    """Calculate the final ordinal score and apply the single hard-failure override."""
    required_score = requirement_score(response.required_requirements)
    preferred_score = requirement_score(response.preferred_requirements)
    raw_score = (
        0.35 * required_score * 100
        + 0.20 * response.skills_fit
        + 0.20 * response.experience_fit
        + 0.15 * response.role_alignment
        + 0.05 * response.education_fit
        + 0.05 * preferred_score * 100
    )

    hard_missing = any(
        evaluation.status == RequirementStatus.MISSING
        and _matches_hard(evaluation.requirement, hard_requirements)
        for evaluation in response.required_requirements
    )
    hard_uncertain = any(
        evaluation.status == RequirementStatus.UNCERTAIN
        and _matches_hard(evaluation.requirement, hard_requirements)
        for evaluation in response.required_requirements
    )

    if hard_missing:
        return FinalScore(
            qualification_score=min(raw_score, 39.0),
            recommendation=Recommendation.DO_NOT_APPLY,
            needs_review=False,
            required_requirements_score=required_score,
            preferred_requirements_score=preferred_score,
        )

    return FinalScore(
        qualification_score=min(100.0, max(0.0, raw_score)),
        recommendation=response.recommendation,
        needs_review=hard_uncertain,
        required_requirements_score=required_score,
        preferred_requirements_score=preferred_score,
    )
