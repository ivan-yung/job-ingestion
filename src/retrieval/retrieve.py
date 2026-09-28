"""Hybrid retrieval: semantic + BM25 + reciprocal rank fusion."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.models.candidate import CandidateProfile
from src.models.job import JobRecord
from src.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_VERSION,
    OpenAIEmbeddingProvider,
    SQLiteEmbeddingCache,
    TextEmbeddingItem,
    embed_with_cache,
    hash_content,
)
from src.retrieval.lexical import BM25SearchIndex


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "retrieval": {
        "semantic_weights": {
            "overall": 0.50,
            "skills": 0.30,
            "role": 0.20,
        },
        "rrf_k": 60,
        "top_k": 150,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_version": DEFAULT_EMBEDDING_VERSION,
        "embedding_dimension": 1536,
        "embedding_batch_size": 64,
    }
}

EMBEDDING_COST_PER_1K_TOKENS = {
    "text-embedding-3-small": 0.00002,
}


@dataclass(frozen=True)
class SemanticParts:
    overall: str
    skills: str
    role: str


@dataclass
class RetrievalDiagnostics:
    embedding_api_calls: int
    embedding_cache_hits: int
    embedding_cache_misses: int
    estimated_embedding_tokens: int
    estimated_embedding_cost_usd: float
    retrieval_latency_ms: float


def _load_pipeline_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return DEFAULT_PIPELINE_CONFIG

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    merged = dict(DEFAULT_PIPELINE_CONFIG)
    merged.update(payload)
    if "retrieval" in payload:
        merged["retrieval"] = dict(DEFAULT_PIPELINE_CONFIG["retrieval"])
        merged["retrieval"].update(payload["retrieval"])
        if "semantic_weights" in payload["retrieval"]:
            merged["retrieval"]["semantic_weights"] = dict(
                DEFAULT_PIPELINE_CONFIG["retrieval"]["semantic_weights"]
            )
            merged["retrieval"]["semantic_weights"].update(
                payload["retrieval"]["semantic_weights"]
            )
    return merged


def _semantic_parts_for_candidate(candidate: CandidateProfile) -> SemanticParts:
    skills = []
    skills.extend(candidate.skills.languages)
    skills.extend(candidate.skills.frameworks)
    skills.extend(candidate.skills.databases)
    skills.extend(candidate.skills.cloud)
    skills.extend(candidate.skills.tools)
    skills.extend(candidate.skills.other)

    role_bits = []
    role_bits.extend(candidate.target_roles)
    role_bits.extend([exp.title for exp in candidate.professional_experience if exp.title])

    overall = [
        candidate.resume_raw_text or "",
        " ".join(role_bits),
        " ".join(skills),
    ]

    return SemanticParts(
        overall="\n".join(part for part in overall if part).strip(),
        skills=" ".join(skills).strip(),
        role=" ".join(role_bits).strip(),
    )


def _semantic_parts_for_job(job: Mapping[str, Any]) -> SemanticParts:
    technologies = [str(t) for t in job.get("technologies", [])]
    required = [str(t) for t in job.get("required_qualifications", [])]
    preferred = [str(t) for t in job.get("preferred_qualifications", [])]

    role_parts = [
        str(job.get("title", "")),
        str(job.get("normalized_title", "")),
        str(job.get("summary", "")),
        str(job.get("seniority", "")),
    ]

    return SemanticParts(
        overall=str(job.get("searchable_text", "")),
        skills=" ".join(technologies + required + preferred).strip(),
        role=" ".join([part for part in role_parts if part]).strip(),
    )


def _candidate_profile_hash(candidate: CandidateProfile) -> str:
    payload = candidate.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    denominator = np.clip(query_norm * matrix_norm, a_min=1e-12, a_max=None)
    return (matrix @ query) / denominator


def _ranking_from_scores(scores: np.ndarray) -> dict[int, int]:
    ordered = np.argsort(-scores)
    return {doc_index: rank for rank, doc_index in enumerate(ordered, start=1)}


def reciprocal_rank_fusion(
    rankings: list[dict[int, int]],
    *,
    k: int = 60,
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for doc_index, rank in ranking.items():
            fused[doc_index] = fused.get(doc_index, 0.0) + (1.0 / (k + rank))
    return fused


def retrieve_jobs(
    candidate_profile: CandidateProfile,
    jobs: list[JobRecord | Mapping[str, Any]],
    *,
    top_k: int | None = None,
    pipeline_config: Mapping[str, Any] | None = None,
    cache_db_path: str | Path = "embedding_cache.db",
    embedding_provider=None,
) -> dict[str, Any]:
    """Retrieve a high-recall hybrid candidate set with diagnostics."""
    started_at = time.perf_counter()

    config = dict(DEFAULT_PIPELINE_CONFIG)
    if pipeline_config:
        config.update(dict(pipeline_config))
        if "retrieval" in pipeline_config:
            config["retrieval"] = dict(DEFAULT_PIPELINE_CONFIG["retrieval"])
            config["retrieval"].update(dict(pipeline_config["retrieval"]))
            if "semantic_weights" in pipeline_config["retrieval"]:
                config["retrieval"]["semantic_weights"] = dict(
                    DEFAULT_PIPELINE_CONFIG["retrieval"]["semantic_weights"]
                )
                config["retrieval"]["semantic_weights"].update(
                    dict(pipeline_config["retrieval"]["semantic_weights"])
                )

    retrieval_cfg = config["retrieval"]
    weights = retrieval_cfg["semantic_weights"]

    top_k = top_k or int(retrieval_cfg.get("top_k", 150))
    rrf_k = int(retrieval_cfg.get("rrf_k", 60))
    embedding_model = str(retrieval_cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    embedding_version = str(retrieval_cfg.get("embedding_version", DEFAULT_EMBEDDING_VERSION))
    embedding_dimension = int(retrieval_cfg.get("embedding_dimension", 1536))
    embedding_batch_size = int(retrieval_cfg.get("embedding_batch_size", 64))

    provider = embedding_provider or OpenAIEmbeddingProvider()
    cache = SQLiteEmbeddingCache(cache_db_path)

    normalized_jobs: list[dict[str, Any]] = [
        job.model_dump() if isinstance(job, JobRecord) else dict(job)
        for job in jobs
    ]

    candidate_parts = _semantic_parts_for_candidate(candidate_profile)
    candidate_hash = _candidate_profile_hash(candidate_profile)
    candidate_items = [
        TextEmbeddingItem(
            text=candidate_parts.overall,
            content_hash=hash_content(f"candidate:{candidate_hash}:overall:{candidate_parts.overall}"),
        ),
        TextEmbeddingItem(
            text=candidate_parts.skills,
            content_hash=hash_content(f"candidate:{candidate_hash}:skills:{candidate_parts.skills}"),
        ),
        TextEmbeddingItem(
            text=candidate_parts.role,
            content_hash=hash_content(f"candidate:{candidate_hash}:role:{candidate_parts.role}"),
        ),
    ]

    candidate_vectors, candidate_stats = embed_with_cache(
        candidate_items,
        cache=cache,
        provider=provider,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        dimension=embedding_dimension,
        batch_size=embedding_batch_size,
    )
    candidate_overall, candidate_skills, candidate_role = candidate_vectors

    job_overall_items: list[TextEmbeddingItem] = []
    job_skills_items: list[TextEmbeddingItem] = []
    job_role_items: list[TextEmbeddingItem] = []

    for job in normalized_jobs:
        parts = _semantic_parts_for_job(job)
        job_overall_items.append(
            TextEmbeddingItem(text=parts.overall, content_hash=hash_content(parts.overall))
        )
        job_skills_items.append(
            TextEmbeddingItem(text=parts.skills, content_hash=hash_content(parts.skills))
        )
        job_role_items.append(
            TextEmbeddingItem(text=parts.role, content_hash=hash_content(parts.role))
        )

    job_overall_vectors, overall_stats = embed_with_cache(
        job_overall_items,
        cache=cache,
        provider=provider,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        dimension=embedding_dimension,
        batch_size=embedding_batch_size,
    )
    job_skills_vectors, skills_stats = embed_with_cache(
        job_skills_items,
        cache=cache,
        provider=provider,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        dimension=embedding_dimension,
        batch_size=embedding_batch_size,
    )
    job_role_vectors, role_stats = embed_with_cache(
        job_role_items,
        cache=cache,
        provider=provider,
        embedding_model=embedding_model,
        embedding_version=embedding_version,
        dimension=embedding_dimension,
        batch_size=embedding_batch_size,
    )

    overall_matrix = np.vstack(job_overall_vectors)
    skills_matrix = np.vstack(job_skills_vectors)
    role_matrix = np.vstack(job_role_vectors)

    overall_similarity = _cosine_similarity(candidate_overall, overall_matrix)
    skills_similarity = _cosine_similarity(candidate_skills, skills_matrix)
    role_similarity = _cosine_similarity(candidate_role, role_matrix)

    semantic_scores = (
        float(weights["overall"]) * overall_similarity
        + float(weights["skills"]) * skills_similarity
        + float(weights["role"]) * role_similarity
    )

    candidate_query = " ".join(
        [candidate_parts.overall, candidate_parts.skills, candidate_parts.role]
    )
    bm25_index = BM25SearchIndex.from_jobs(normalized_jobs)
    bm25_scores = bm25_index.score(candidate_query)

    semantic_ranks = _ranking_from_scores(semantic_scores)
    bm25_ranks = _ranking_from_scores(bm25_scores)
    fused_scores = reciprocal_rank_fusion([semantic_ranks, bm25_ranks], k=rrf_k)

    ordered_doc_indices = sorted(fused_scores, key=lambda idx: fused_scores[idx], reverse=True)[:top_k]

    results: list[dict[str, Any]] = []
    for doc_index in ordered_doc_indices:
        job = normalized_jobs[doc_index]
        results.append(
            {
                "job_id": str(job.get("job_id", "")),
                "job": job,
                "semantic_rank": semantic_ranks[doc_index],
                "bm25_rank": bm25_ranks[doc_index],
                "rrf_score": fused_scores[doc_index],
                "semantic_score": float(semantic_scores[doc_index]),
                "bm25_score": float(bm25_scores[doc_index]),
            }
        )

    embedding_stats = EmbeddingStatsAggregator.from_many(
        [candidate_stats, overall_stats, skills_stats, role_stats]
    )

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    cost_rate = EMBEDDING_COST_PER_1K_TOKENS.get(embedding_model, 0.0)
    diagnostics = RetrievalDiagnostics(
        embedding_api_calls=embedding_stats.api_calls,
        embedding_cache_hits=embedding_stats.cache_hits,
        embedding_cache_misses=embedding_stats.cache_misses,
        estimated_embedding_tokens=embedding_stats.estimated_tokens,
        estimated_embedding_cost_usd=(embedding_stats.estimated_tokens / 1000.0) * cost_rate,
        retrieval_latency_ms=latency_ms,
    )

    return {
        "results": results,
        "diagnostics": diagnostics.__dict__,
        "config": {
            "top_k": top_k,
            "rrf_k": rrf_k,
            "semantic_weights": weights,
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "embedding_dimension": embedding_dimension,
        },
    }


@dataclass
class EmbeddingStatsAggregator:
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_tokens: int = 0

    @classmethod
    def from_many(cls, stats_list: list[Any]) -> "EmbeddingStatsAggregator":
        acc = cls()
        for stats in stats_list:
            acc.api_calls += int(getattr(stats, "api_calls", 0))
            acc.cache_hits += int(getattr(stats, "cache_hits", 0))
            acc.cache_misses += int(getattr(stats, "cache_misses", 0))
            acc.estimated_tokens += int(getattr(stats, "estimated_tokens", 0))
        return acc


def _load_jobs_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and "jobs" in payload:
        return [dict(item) for item in payload["jobs"]]
    if isinstance(payload, dict) and "items" in payload:
        return [dict(item) for item in payload["items"]]
    raise ValueError(f"Unsupported jobs payload format at {path}")


def _load_candidate_profile(path: str | Path) -> CandidateProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CandidateProfile.model_validate(payload)


def _load_candidate_from_cleaned_resume(path: str | Path = "cleaned_resume.json") -> CandidateProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    resume_text = str(payload.get("embedding_input") or payload.get("cleaned_text") or "").strip()
    if not resume_text:
        raise ValueError("cleaned_resume.json is missing embedding_input/cleaned_text")

    return CandidateProfile(
        candidate_id="local-user",
        resume_raw_text=resume_text,
    )


def cli_main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid retrieval over normalized jobs")
    parser.add_argument("--jobs", required=True, help="Path to normalized jobs JSON")
    parser.add_argument(
        "--candidate",
        default=None,
        help="Path to candidate profile JSON in CandidateProfile schema",
    )
    parser.add_argument("--output", default="retrieved_jobs.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=None, choices=[50, 100, 150, 200])
    parser.add_argument("--cache-db", default="embedding_cache.db", help="Embedding cache SQLite path")
    parser.add_argument("--config", default=None, help="Optional pipeline config JSON path")
    args = parser.parse_args()

    pipeline_config = _load_pipeline_config(args.config)
    jobs = _load_jobs_json(args.jobs)
    if args.candidate:
        candidate = _load_candidate_profile(args.candidate)
    else:
        candidate = _load_candidate_from_cleaned_resume()

    retrieved = retrieve_jobs(
        candidate,
        jobs,
        top_k=args.top_k,
        pipeline_config=pipeline_config,
        cache_db_path=args.cache_db,
    )

    Path(args.output).write_text(json.dumps(retrieved, indent=2), encoding="utf-8")

    diagnostics = retrieved["diagnostics"]
    print(f"Retrieved {len(retrieved['results'])} jobs -> {args.output}")
    print(f"Latency: {diagnostics['retrieval_latency_ms']:.1f} ms")
    print(
        "Embedding: "
        f"calls={diagnostics['embedding_api_calls']} "
        f"hits={diagnostics['embedding_cache_hits']} "
        f"misses={diagnostics['embedding_cache_misses']}"
    )
    print(
        "Embedding cost estimate (USD): "
        f"{diagnostics['estimated_embedding_cost_usd']:.6f} "
        f"for {diagnostics['estimated_embedding_tokens']} est. tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
