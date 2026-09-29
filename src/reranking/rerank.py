"""Orchestration for Stage 3 deterministic reranking."""

from typing import Any
from src.models.candidate import CandidateProfile
from src.models.job import JobRecord
from src.reranking.scoring import FORMULA_VERSION, calculate_score, extract_features

def rerank_jobs(
    candidate_profile: CandidateProfile,
    retrieved_jobs: list[dict[str, Any]] | list[JobRecord],
    jobs: list[JobRecord] | None = None,
    top_k: int = 50
) -> list[dict[str, Any]]:
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
    
    for result in retrieved_results:
        job_id = result["job_id"]
        if job_id not in job_map:
            continue
            
        job = job_map[job_id]
        
        # Extract and Score
        features = extract_features(candidate_profile, job, result)
        
        # Hard Requirement Gating
        if features.hard_requirement_status == "failed":
            continue
            
        qualification_score = calculate_score(features)
        
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
    return scored_results[:top_k]
