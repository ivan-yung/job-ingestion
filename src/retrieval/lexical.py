"""BM25 lexical retrieval utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.models.job import JobRecord


PHRASE_TOKEN_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\w)react\s+native(?!\w)", re.IGNORECASE), "react_native"),
]


def normalize_search_text(text: str) -> str:
    """Normalize text while preserving technology tokens."""
    normalized = text.lower()

    for pattern, replacement in PHRASE_TOKEN_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)

    # Keep alnum plus selected symbols to preserve tokens like c++, .net and c#.
    normalized = re.sub(r"[^a-z0-9_\s\+\#\.]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize while keeping exact technology tokens queryable."""
    cleaned = normalize_search_text(text)
    if not cleaned:
        return []

    tokens = []
    for token in cleaned.split(" "):
        # Strip trailing periods to handle end-of-sentence punctuation 
        # while safely preserving leading periods (like .net)
        token = token.rstrip(".")
        if token:
            tokens.append(token)
            
    return tokens


@dataclass
class BM25SearchIndex:
    """In-memory BM25 index over normalized jobs."""

    jobs: list[JobRecord]
    tokenized_docs: list[list[str]]
    bm25: BM25Okapi

    @classmethod
    def from_jobs(cls, jobs: list[JobRecord]) -> "BM25SearchIndex":
        tokenized_docs = [tokenize_for_bm25(job.searchable_text or "") for job in jobs]
        return cls(jobs=jobs, tokenized_docs=tokenized_docs, bm25=BM25Okapi(tokenized_docs))

    def score(self, query: str) -> np.ndarray:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return np.zeros(len(self.jobs), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)