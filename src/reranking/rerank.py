"""Orchestration for Stage 3 deterministic reranking."""

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from src.models.candidate import CandidateProfile
from src.models.job import JobRecord
from src.reranking.scoring import (
    FORMULA_VERSION,
    RerankFeatures,
    calculate_score,
    extract_features,
)


class GatedJob(BaseModel):
    """A job excluded before scoring because a hard requirement failed."""

    job_id: str
    failed_requirement: str
    feature_snapshot: RerankFeatures


class RerankOutput(BaseModel):
    """Inspectable Stage 3 output split into scored and gated jobs."""

    scored: list[dict[str, Any]]
    gated: list[GatedJob]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Keep legacy callers iterating over the scored result list working."""
        return iter(self.scored)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Keep legacy positional access focused on scored jobs."""
        return self.scored[index]

    def __len__(self) -> int:
        return len(self.scored)


def _failed_requirement(job: JobRecord, features: RerankFeatures) -> str:
    """Return the most specific hard requirement available for a gated job."""
    evidence = getattr(job, "required_qualifications_with_evidence", [])
    for item in evidence:
        classification = getattr(item, "classification", None) or item.get("classification")
        text = getattr(item, "text", None) or item.get("text")
        if classification in {"hard", "unknown"} and text:
            return text

    if features.experience_fit < 1 and job.experience_requirements:
        return job.experience_requirements[0]
    if features.education_fit < 1 and job.education_requirements:
        return job.education_requirements[0]
    if features.required_skill_coverage < 1 and job.required_qualifications:
        return job.required_qualifications[0]
    if features.required_technology_coverage < 1 and job.technologies:
        return job.technologies[0]
    return "Unspecified hard requirement"

def rerank_jobs(
    candidate_profile: CandidateProfile,
    retrieved_jobs: list[dict[str, Any]] | list[JobRecord],
    jobs: list[JobRecord] | None = None,
    top_k: int = 50,
    score_weights: dict[str, float] | None = None,
) -> RerankOutput:
    """Rerank retrieved jobs, gating hard requirements before scoring."""

    if jobs is None:
        if all(isinstance(item, JobRecord) for item in retrieved_jobs):
            jobs = [item for item in retrieved_jobs if isinstance(item, JobRecord)]
            retrieved_results = [
                {"job_id": job.job_id, "semantic_score": 0.0, "bm25_score": 0.0}
                for job in jobs
            ]
        else:
            embedded_jobs = [item.get("job") for item in retrieved_jobs if isinstance(item, dict)]
            if not embedded_jobs or not all(isinstance(job, JobRecord) for job in embedded_jobs):
                raise ValueError("jobs are required unless retrieved_jobs contains JobRecord objects or a 'job' field")
            jobs = [job for job in embedded_jobs if isinstance(job, JobRecord)]
            retrieved_results = [item for item in retrieved_jobs if isinstance(item, dict)]
    else:
        retrieved_results = [item for item in retrieved_jobs if isinstance(item, dict)]

    job_map = {job.job_id: job for job in jobs}
    scored_results = []
    gated_results = []
    
    for result in retrieved_results:
        job_id = result["job_id"]
        if job_id not in job_map:
            continue
            
        job = job_map[job_id]
        
        # Extract and Score
        features = extract_features(candidate_profile, job, result)
        
        # Hard Requirement Gating
        if features.hard_requirement_status == "failed":
            gated_results.append(
                GatedJob(
                    job_id=job_id,
                    failed_requirement=_failed_requirement(job, features),
                    feature_snapshot=features,
                )
            )
            continue
            
        qualification_score = calculate_score(features, score_weights)
        
        scored_results.append({
            "job_id": job_id,
            "qualification_score": qualification_score,
            "rerank_score": qualification_score,
            "formula_version": FORMULA_VERSION,
            "feature_breakdown": features.model_dump(),
            "retrieval_data": result
        })
        
    # Sort by the new deterministic score
    scored_results.sort(key=lambda x: (-x["qualification_score"], x["job_id"]))
    gated_results.sort(key=lambda item: item.job_id)
    return RerankOutput(scored=scored_results[:top_k], gated=gated_results)
