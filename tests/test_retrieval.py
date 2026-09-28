"""Tests for hybrid retrieval (semantic + lexical + RRF)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.candidate import CandidateProfile, Skills
from src.retrieval.embeddings import SQLiteEmbeddingCache, TextEmbeddingItem, embed_with_cache, hash_content
from src.retrieval.lexical import BM25SearchIndex, tokenize_for_bm25
from src.retrieval.retrieve import reciprocal_rank_fusion, retrieve_jobs


@dataclass
class FakeEmbeddingProvider:
    dimensions: int = 1536

    def __post_init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            seed = abs(hash((text, model))) % 997
            vector = np.zeros(self.dimensions, dtype=np.float32)
            vector[seed % self.dimensions] = 1.0
            vectors.append(vector.tolist())
        return vectors


def test_embedding_cache_hit_and_invalidation() -> None:
    provider = FakeEmbeddingProvider(dimensions=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SQLiteEmbeddingCache(Path(tmpdir) / "cache.db")

        item = TextEmbeddingItem(text="python fastapi", content_hash=hash_content("python fastapi"))
        vectors, _ = embed_with_cache(
            [item],
            cache=cache,
            provider=provider,
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            dimension=4,
            batch_size=4,
        )
        assert len(vectors) == 1
        assert provider.calls == 1

        vectors_again, _ = embed_with_cache(
            [item],
            cache=cache,
            provider=provider,
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            dimension=4,
            batch_size=4,
        )
        assert len(vectors_again) == 1
        assert provider.calls == 1

        changed_item = TextEmbeddingItem(
            text="python fastapi postgresql",
            content_hash=hash_content("python fastapi postgresql"),
        )
        _, _ = embed_with_cache(
            [changed_item],
            cache=cache,
            provider=provider,
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            dimension=4,
            batch_size=4,
        )
        assert provider.calls == 2

        _, _ = embed_with_cache(
            [item],
            cache=cache,
            provider=provider,
            embedding_model="text-embedding-3-large",
            embedding_version="1",
            dimension=4,
            batch_size=4,
        )
        assert provider.calls == 3


def test_bm25_preserves_technology_tokens() -> None:
    text = "C++ and .NET with React Native experience"
    tokens = tokenize_for_bm25(text)

    assert "c++" in tokens
    assert ".net" in tokens
    assert "react_native" in tokens

    jobs = [
        {"job_id": "1", "searchable_text": "C++ backend engineer"},
        {"job_id": "2", "searchable_text": "React Native mobile engineer"},
        {"job_id": "3", "searchable_text": ".NET platform engineer"},
    ]
    index = BM25SearchIndex.from_jobs(jobs)

    cpp_scores = index.score("C++")
    dotnet_scores = index.score(".NET")
    rn_scores = index.score("React Native")

    assert int(np.argmax(cpp_scores)) == 0
    assert int(np.argmax(dotnet_scores)) == 2
    assert int(np.argmax(rn_scores)) == 1


def test_rrf_fusion_math() -> None:
    semantic_ranks = {0: 1, 1: 2, 2: 3}
    bm25_ranks = {1: 1, 2: 2, 0: 3}

    fused = reciprocal_rank_fusion([semantic_ranks, bm25_ranks], k=60)

    expected_doc0 = (1 / (60 + 1)) + (1 / (60 + 3))
    expected_doc1 = (1 / (60 + 2)) + (1 / (60 + 1))
    expected_doc2 = (1 / (60 + 3)) + (1 / (60 + 2))

    assert abs(fused[0] - expected_doc0) < 1e-12
    assert abs(fused[1] - expected_doc1) < 1e-12
    assert abs(fused[2] - expected_doc2) < 1e-12


def test_retrieve_jobs_end_to_end() -> None:
    candidate = CandidateProfile(
        candidate_id="candidate-1",
        target_roles=["Python Engineer"],
        skills=Skills(
            languages=["Python"],
            frameworks=["FastAPI"],
            databases=["PostgreSQL"],
        ),
        resume_raw_text="Built Python APIs with FastAPI and PostgreSQL",
    )

    jobs = [
        {
            "job_id": "job-1",
            "title": "Backend Engineer",
            "normalized_title": "backend engineer",
            "summary": "Build APIs",
            "seniority": "mid",
            "technologies": ["Python", "FastAPI"],
            "required_qualifications": ["Python"],
            "preferred_qualifications": ["PostgreSQL"],
            "searchable_text": "Backend engineer with Python and FastAPI",
        },
        {
            "job_id": "job-2",
            "title": "Mobile Engineer",
            "normalized_title": "mobile engineer",
            "summary": "Build mobile apps",
            "seniority": "mid",
            "technologies": ["React Native"],
            "required_qualifications": ["React Native"],
            "preferred_qualifications": [],
            "searchable_text": "React Native engineer",
        },
    ]

    provider = FakeEmbeddingProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        retrieved = retrieve_jobs(
            candidate,
            jobs,
            top_k=2,
            cache_db_path=Path(tmpdir) / "cache.db",
            embedding_provider=provider,
        )

    assert "results" in retrieved
    assert len(retrieved["results"]) == 2

    for result in retrieved["results"]:
        assert "job_id" in result
        assert "semantic_rank" in result
        assert "bm25_rank" in resultgit
        
        assert "rrf_score" in result
        assert "semantic_score" in result
        assert "bm25_score" in result
