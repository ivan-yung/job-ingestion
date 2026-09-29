import pytest
import numpy as np
from unittest.mock import Mock

from src.retrieval.lexical import tokenize_for_bm25
from src.retrieval.retrieve import compute_rrf, retrieve_jobs
from src.retrieval.embeddings import (
    TextEmbeddingItem, 
    embed_with_cache, 
    SQLiteEmbeddingCache
)

class MockCandidate:
    def __init__(self):
        self.summary = "Software Engineer"
        self.experience_level = "Mid"
        self.years_experience = 3.0
        self.skills = Mock(languages=["Python", "Go"], frameworks=["React"])
        self.professional_experience = []

class MockJob:
    def __init__(self, job_id, text=""):
        self.job_id = job_id
        self.required_qualifications = ["Python"]
        self.preferred_qualifications = []
        self.technologies = ["React"]
        self.seniority = "Mid"
        self.normalized_title = "Backend Engineer"
        self.summary = "Looking for an engineer."
        self.description_raw = text
        self.content_hash = f"hash_{job_id}"
        self.searchable_text = text


def test_embedding_cache_hit_and_invalidation(tmp_path):
    """Test that embedding cache reuses vectors and isolates by hash/model."""
    db_path = tmp_path / "test_cache.db"
    cache = SQLiteEmbeddingCache(str(db_path))
    
    mock_provider = Mock()
    mock_provider.embed_texts.return_value = [[0.1, 0.2]]
    
    item1 = TextEmbeddingItem(text="hello", content_hash="hash1")
    
    vecs1, stats1 = embed_with_cache([item1], cache=cache, provider=mock_provider, dimension=2)
    assert stats1.cache_misses == 1
    assert mock_provider.embed_texts.call_count == 1
    
    vecs2, stats2 = embed_with_cache([item1], cache=cache, provider=mock_provider, dimension=2)
    assert stats2.cache_hits == 1
    assert mock_provider.embed_texts.call_count == 1 
    assert np.array_equal(vecs1[0], vecs2[0])
    
    vecs3, stats3 = embed_with_cache([item1], cache=cache, provider=mock_provider, embedding_model="new-model", dimension=2)
    assert stats3.cache_misses == 1
    assert mock_provider.embed_texts.call_count == 2


def test_bm25_preserves_technology_tokens():
    """Test that punctuation is preserved for core tech tokens."""
    tokens = tokenize_for_bm25("Experience with C++, .NET, C#, and React Native.")
    
    assert "c++" in tokens
    assert ".net" in tokens
    assert "c#" in tokens
    assert "react_native" in tokens
    assert "react" not in tokens 


def test_rrf_fusion_math():
    """Test the raw Reciprocal Rank Fusion calculation directly."""
    k = 60
    perfect_score = compute_rrf(1, 1, k)
    assert np.isclose(perfect_score, (1 / 61) + (1 / 61))
    
    skewed_score = compute_rrf(1, 100, k)
    assert np.isclose(skewed_score, (1 / 61) + (1 / 160))
    
    even_score = compute_rrf(10, 10, k)
    assert np.isclose(even_score, (1 / 70) + (1 / 70))


def test_retrieve_jobs_end_to_end(tmp_path, monkeypatch):
    """Test full retrieval orchestration and diagnostic output structure."""
    candidate = MockCandidate()
    jobs = [MockJob("job_1", "Python backend"), MockJob("job_2", "Go backend"), MockJob("job_3", "C++ embedded")]

    class DummyProvider:
        def embed_texts(self, texts, model):
            return [[0.5] * 1536 for _ in texts]
            
    monkeypatch.setattr("src.retrieval.retrieve.OpenAIEmbeddingProvider", DummyProvider)

    results = retrieve_jobs(candidate, jobs, top_k=2, db_path=str(tmp_path / "test.db"))

    assert len(results) == 2
    
    top_result = results[0]
    expected_keys = {"job_id", "semantic_rank", "bm25_rank", "rrf_score", "semantic_score", "bm25_score"}
    assert set(top_result.keys()) == expected_keys
    assert 0.0 < top_result["rrf_score"] <= (2.0 / 61.0)