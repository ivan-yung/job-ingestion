"""Hashing and deterministic fingerprinting utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_hash(content: str) -> str:
    """Compute SHA-256 hash of content.

    Args:
        content: The content to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def deterministic_json_hash(data: dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary using deterministic JSON serialization.

    Args:
        data: Dictionary to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_hash(json_str)


def content_fingerprint(
    company: str,
    title: str,
    location: str,
    description: str,
) -> str:
    """Generate a deterministic fingerprint for deduplication.

    Args:
        company: Normalized company name.
        title: Normalized job title.
        location: Normalized location.
        description: Normalized description.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    combined = f"{company}|||{title}|||{location}|||{description}"
    return sha256_hash(combined)
