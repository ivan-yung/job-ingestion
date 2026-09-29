"""Shared retry and backoff helpers for external API calls."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """A transient operation failed after all configured retries."""

    def __init__(self, cause: Exception, retries: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.retries = retries


def is_transient_error(exc: Exception) -> bool:
    """Return whether an exception is likely safe to retry."""
    name = exc.__class__.__name__.lower()
    status_code = getattr(exc, "status_code", None)
    return name in {
        "apiconnectionerror",
        "apitimeouterror",
        "ratelimiterror",
        "internalservererror",
        "serviceunavailableerror",
        "timeouterror",
    } or (isinstance(status_code, int) and (status_code == 429 or status_code >= 500))


def retry_call(
    operation: Callable[[], T],
    *,
    max_retries: int,
    base_backoff_seconds: float = 0.5,
    max_backoff_seconds: float = 8.0,
    on_retry: Callable[[], None] | None = None,
) -> T:
    """Retry transient failures with exponential backoff and jitter."""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            if attempt >= max_retries:
                raise RetryExhaustedError(exc, attempt) from exc
            if on_retry is not None:
                on_retry()
            delay = min(max_backoff_seconds, base_backoff_seconds * (2**attempt))
            time.sleep(delay + random.uniform(0, delay * 0.25))
            attempt += 1
