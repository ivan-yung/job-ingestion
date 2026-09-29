"""Embedding provider and SQLite-backed embedding cache."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np

from src.models.job import JobRecord
from src.models.candidate import CandidateProfile
from src.retry import is_transient_error, retry_call


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_VERSION = "1"


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """Identity fields that make an embedding reusable."""
    content_hash: str
    embedding_model: str
    embedding_version: str
    dimension: int


@dataclass(frozen=True)
class TextEmbeddingItem:
    """A piece of text that needs an embedding."""
    text: str
    content_hash: str


@dataclass
class EmbeddingStats:
    """Diagnostics for embedding work."""
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_texts: int = 0
    estimated_tokens: int = 0


class EmbeddingProvider(Protocol):
    """Pluggable embedding provider."""
    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        """Return vectors for each text in order."""


class OpenAIEmbeddingProvider:
    """OpenAI-backed embedding provider with retry support."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 5,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
    ) -> None:
        self._api_key = api_key
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client to keep tests lightweight."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []

        response = retry_call(
            lambda: self.client.embeddings.create(model=model, input=texts),
            max_retries=self.max_retries,
            base_backoff_seconds=self.base_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
        )
        return [row.embedding for row in response.data]

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        return is_transient_error(exc)


class SQLiteEmbeddingCache:
    """Simple SQLite cache for embeddings keyed by content+model."""

    def __init__(self, db_path: str | Path = "resume_ranker.db") -> None:
        self.db_path = str(db_path)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (content_hash, embedding_model, embedding_version)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_model
                ON embeddings (embedding_model)
                """
            )
            conn.commit()

    def get(
        self,
        *,
        content_hash: str,
        embedding_model: str,
        embedding_version: str,
        dimension: int,
    ) -> np.ndarray | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT vector_json, dimension
                FROM embeddings
                WHERE content_hash = ? AND embedding_model = ? AND embedding_version = ?
                """,
                (content_hash, embedding_model, embedding_version),
            ).fetchone()

        if row is None:
            return None

        vector_json, cached_dimension = row
        if cached_dimension != dimension:
            return None

        vector = np.asarray(json.loads(vector_json), dtype=np.float32)
        if vector.size != dimension:
            return None

        return vector

    def put(
        self,
        *,
        fingerprint: EmbeddingFingerprint,
        vector: np.ndarray,
    ) -> None:
        payload = json.dumps(vector.astype(float).tolist())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO embeddings (
                    content_hash,
                    embedding_model,
                    embedding_version,
                    dimension,
                    vector_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash, embedding_model, embedding_version)
                DO UPDATE SET
                    dimension = excluded.dimension,
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                (
                    fingerprint.content_hash,
                    fingerprint.embedding_model,
                    fingerprint.embedding_version,
                    fingerprint.dimension,
                    payload,
                    time.time(),
                ),
            )
            conn.commit()


def hash_content(text: str) -> str:
    """Deterministic content hash used as cache identity."""
    return sha256(text.encode("utf-8")).hexdigest()


def estimate_text_tokens(text: str) -> int:
    """Cheap token estimate used for diagnostics and rough cost reporting."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def embed_with_cache(
    items: list[TextEmbeddingItem],
    *,
    cache: SQLiteEmbeddingCache,
    provider: EmbeddingProvider,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_version: str = DEFAULT_EMBEDDING_VERSION,
    dimension: int = 1536,
    batch_size: int = 64,
) -> tuple[list[np.ndarray], EmbeddingStats]:
    """Embed texts with cache lookup and batched provider calls."""
    stats = EmbeddingStats(total_texts=len(items))

    if not items:
        return [], stats

    vectors: list[np.ndarray | None] = [None] * len(items)
    missing_indices: list[int] = []

    for idx, item in enumerate(items):
        stats.estimated_tokens += estimate_text_tokens(item.text)
        cached = cache.get(
            content_hash=item.content_hash,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            dimension=dimension,
        )
        if cached is not None:
            vectors[idx] = cached
            stats.cache_hits += 1
        else:
            missing_indices.append(idx)
            stats.cache_misses += 1

    for batch_start in range(0, len(missing_indices), batch_size):
        batch_indices = missing_indices[batch_start : batch_start + batch_size]
        batch_texts = [items[index].text for index in batch_indices]
        batch_vectors = provider.embed_texts(batch_texts, model=embedding_model)
        stats.api_calls += 1

        for item_index, batch_vector in zip(batch_indices, batch_vectors, strict=True):
            vector = np.asarray(batch_vector, dtype=np.float32)
            if vector.size != dimension:
                raise ValueError(
                    f"Unexpected embedding dimension {vector.size}, expected {dimension}"
                )

            vectors[item_index] = vector
            fingerprint = EmbeddingFingerprint(
                content_hash=items[item_index].content_hash,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
                dimension=dimension,
            )
            cache.put(fingerprint=fingerprint, vector=vector)

    return [vector for vector in vectors if vector is not None], stats


def cosine_similarity(v1: np.ndarray | None, v2: np.ndarray | None) -> float:
    """Calculates cosine similarity between two normalized vectors."""
    if v1 is None or v2 is None:
        return 0.0
    return float(np.dot(v1, v2))


def extract_job_embedding_items(job: JobRecord) -> dict[str, TextEmbeddingItem]:
    """Extracts the three text components for a job for semantic scoring with fallbacks."""
    skills = (job.required_qualifications or []) + (job.preferred_qualifications or []) + (job.technologies or [])
    skills_text = ", ".join(skills) if skills else "No specific skills listed"
    
    role_text = f"{job.seniority or ''} {job.normalized_title or ''}".strip()
    if not role_text:
        role_text = "Unknown role"
        
    overall_text = f"{job.summary or ''} {job.description_raw or ''}".strip()
    if not overall_text:
        overall_text = "No description provided"
    
    return {
        "overall": TextEmbeddingItem(text=overall_text, content_hash=hash_content(f"{job.content_hash}_overall")),
        "skills": TextEmbeddingItem(text=skills_text, content_hash=hash_content(f"{job.content_hash}_skills")),
        "role": TextEmbeddingItem(text=role_text, content_hash=hash_content(f"{job.content_hash}_role"))
    }


def extract_candidate_embedding_items(candidate: CandidateProfile) -> dict[str, TextEmbeddingItem]:
    """Extracts the three text components for a candidate for semantic scoring with fallbacks."""
    langs = getattr(candidate.skills, 'languages', []) if hasattr(candidate, 'skills') else []
    fwks = getattr(candidate.skills, 'frameworks', []) if hasattr(candidate, 'skills') else []
    all_skills = langs + fwks
    skills_text = ", ".join(all_skills) if all_skills else "No skills listed"
    
    level = getattr(candidate, 'experience_level', '') or ''
    years = getattr(candidate, 'years_experience', '') or ''
    role_text = f"{level} {years} years".strip()
    if not role_text or role_text == "years":
        role_text = "Unknown experience"
    
    experiences = getattr(candidate, 'professional_experience', []) or []
    exp_list = [getattr(exp, 'description', '') for exp in experiences if hasattr(exp, 'description')]
    summary = getattr(candidate, 'summary', '') or ''
    overall_text = f"{summary} {' '.join(exp_list)}".strip()
    
    if not overall_text:
        overall_text = "No summary or experience provided"
    
    return {
        "overall": TextEmbeddingItem(text=overall_text, content_hash=hash_content(f"candidate_overall_{overall_text}")),
        "skills": TextEmbeddingItem(text=skills_text, content_hash=hash_content(f"candidate_skills_{skills_text}")),
        "role": TextEmbeddingItem(text=role_text, content_hash=hash_content(f"candidate_role_{role_text}"))
    }