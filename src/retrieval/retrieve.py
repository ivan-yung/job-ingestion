"""Hybrid Semantic and Lexical Retrieval."""

from typing import Any
import sys
import json

from src.models.job import JobRecord
from src.models.candidate import CandidateProfile
from src.retrieval.embeddings import (
    OpenAIEmbeddingProvider,
    SQLiteEmbeddingCache,
    embed_with_cache,
    extract_job_embedding_items,
    extract_candidate_embedding_items,
    cosine_similarity,
)
from src.retrieval.lexical import BM25SearchIndex

# Configurable semantic weights
WEIGHT_OVERALL = 0.50
WEIGHT_SKILLS = 0.30
WEIGHT_ROLE = 0.20

RRF_K = 60


def compute_rrf(rank_1: int, rank_2: int, k: int = RRF_K) -> float:
    """Computes Reciprocal Rank Fusion score."""
    return (1.0 / (k + rank_1)) + (1.0 / (k + rank_2))


def retrieve_jobs(
    candidate: CandidateProfile,
    jobs: list[JobRecord],
    top_k: int = 150,
    db_path: str = "resume_ranker.db",
    semantic_weights: dict[str, float] | None = None,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Retrieve top jobs using Hybrid Search (Semantic + BM25) and RRF."""
    if not jobs:
        return []

    weights = semantic_weights or {
        "overall": WEIGHT_OVERALL,
        "skills": WEIGHT_SKILLS,
        "role": WEIGHT_ROLE,
    }

    cache = SQLiteEmbeddingCache(db_path)
    provider = OpenAIEmbeddingProvider()

    # 1. Prepare text items for batch embedding
    job_items_map = {}
    all_embedding_items = []
    
    for job in jobs:
        items = extract_job_embedding_items(job)
        job_items_map[job.job_id] = items
        all_embedding_items.extend(items.values())

    candidate_items = extract_candidate_embedding_items(candidate)
    all_embedding_items.extend(candidate_items.values())

    # 2. Fetch or retrieve all embeddings in one batched flow
    vectors, _ = embed_with_cache(
        all_embedding_items,
        cache=cache,
        provider=provider
    )

    # Map content hashes to their vectors for O(1) lookup
    vector_map = {item.content_hash: vec for item, vec in zip(all_embedding_items, vectors)}

    cand_overall = vector_map.get(candidate_items["overall"].content_hash)
    cand_skills = vector_map.get(candidate_items["skills"].content_hash)
    cand_role = vector_map.get(candidate_items["role"].content_hash)

    # 3. Compute Semantic Scores and Ranks
    semantic_scores = []
    for job in jobs:
        j_items = job_items_map[job.job_id]
        sim_overall = cosine_similarity(cand_overall, vector_map.get(j_items["overall"].content_hash))
        sim_skills = cosine_similarity(cand_skills, vector_map.get(j_items["skills"].content_hash))
        sim_role = cosine_similarity(cand_role, vector_map.get(j_items["role"].content_hash))
        
        score = (weights["overall"] * sim_overall) + (weights["skills"] * sim_skills) + (weights["role"] * sim_role)
        semantic_scores.append((job.job_id, score))

    semantic_scores.sort(key=lambda x: x[1], reverse=True)
    semantic_ranks = {job_id: rank + 1 for rank, (job_id, _) in enumerate(semantic_scores)}
    semantic_score_map = {job_id: score for job_id, score in semantic_scores}

    # 4. Compute BM25 Scores and Ranks
    exp_list = [exp.description for exp in getattr(candidate, 'professional_experience', []) if hasattr(exp, 'description')]
    candidate_search_text = f"{getattr(candidate, 'summary', '')} {' '.join(exp_list)} {candidate_items['skills'].text}"
    
    bm25_index = BM25SearchIndex.from_jobs(jobs)
    raw_bm25_scores = bm25_index.score(candidate_search_text)
    
    bm25_scores = [(jobs[i].job_id, float(score)) for i, score in enumerate(raw_bm25_scores)]
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    bm25_ranks = {job_id: rank + 1 for rank, (job_id, _) in enumerate(bm25_scores)}
    bm25_score_map = {job_id: score for job_id, score in bm25_scores}

    # 5. Reciprocal Rank Fusion
    results = []
    for job in jobs:
        s_rank = semantic_ranks[job.job_id]
        b_rank = bm25_ranks[job.job_id]
        
        results.append({
            "job_id": job.job_id,
            "semantic_rank": s_rank,
            "bm25_rank": b_rank,
            "rrf_score": compute_rrf(s_rank, b_rank, rrf_k),
            "semantic_score": semantic_score_map[job.job_id],
            "bm25_score": bm25_score_map[job.job_id]
        })

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables (including OPENAI_API_KEY)
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=str, required=True, help="Path to normalized jobs JSON")
    parser.add_argument("--top_k", type=int, default=150, help="Number of results to retrieve")
    args = parser.parse_args()
    
    with open(args.jobs) as f:
        jobs_data = json.load(f)
        if isinstance(jobs_data, dict) and "jobs" in jobs_data:
            jobs_data = jobs_data["jobs"]
        jobs = [JobRecord(**j) for j in jobs_data]
        
    candidate = CandidateProfile(
        candidate_id="CLI_TEST",
        skills={"languages": ["Python", "Go"], "frameworks": ["React"]},
        professional_experience=[],
        summary="Experienced engineer.",
        experience_level="mid",
        years_experience=3.0
    )
    
    try:
        results = retrieve_jobs(candidate, jobs, top_k=args.top_k)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during retrieval: {e}", file=sys.stderr)