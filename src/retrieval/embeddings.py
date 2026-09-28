"""Embedding provider and SQLite-backed embedding cache."""

from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np


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

        attempt = 0
        while True:
            try:
                response = self.client.embeddings.create(model=model, input=texts)
                return [row.embedding for row in response.data]
            except Exception as exc:  # pragma: no cover - exercised by integration
                if not self._is_transient(exc) or attempt >= self.max_retries:
                    raise

                delay = min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** attempt))
                delay += random.uniform(0, delay * 0.25)
                time.sleep(delay)
                attempt += 1

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        status_code = getattr(exc, "status_code", None)

        transient_names = {
            "apiconnectionerror",
            "apitimeouterror",
            "ratelimiterror",
            "internalservererror",
            "serviceunavailableerror",
        }
        if name in transient_names:
            return True

        if isinstance(status_code, int) and status_code >= 500:
            return True

        return False


class SQLiteEmbeddingCache:
    """Simple SQLite cache for embeddings keyed by content+model."""

    def __init__(self, db_path: str | Path = "embedding_cache.db") -> None:
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
