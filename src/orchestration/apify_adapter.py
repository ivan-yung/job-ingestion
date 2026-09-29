"""Apify actor-run retrieval and raw dataset persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import requests

from src.retry import retry_call


APIFY_API_BASE = "https://api.apify.com/v2"


class ApifyTransport(Protocol):
    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> Any:
        """Return an HTTP-like response."""


class ApifyRequestError(RuntimeError):
    """An Apify request failed with a response status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _request_json(
    transport: ApifyTransport,
    url: str,
    api_token: str,
    *,
    timeout: float = 30.0,
) -> Any:
    def request() -> Any:
        try:
            response = transport.get(
                url,
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise ApifyRequestError(503, "Apify network request failed") from exc

        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise ApifyRequestError(status_code, f"Apify request failed with HTTP {status_code}")
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Apify response was not valid JSON") from exc

    return retry_call(request, max_retries=3)


def _payload_data(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise RuntimeError("Unexpected Apify response shape")
    return payload["data"]


def fetch_apify_dataset(
    actor_run_id: str,
    api_token: str,
    *,
    raw_output_path: str | Path | None = None,
    transport: ApifyTransport | None = None,
) -> list[dict[str, Any]]:
    """Fetch a completed Apify actor run's raw dataset items."""
    if not api_token.strip():
        raise ValueError("Apify API token is required")

    transport = transport or requests
    run_payload = _request_json(
        transport,
        f"{APIFY_API_BASE}/actor-runs/{actor_run_id}",
        api_token,
    )
    run_data = _payload_data(run_payload)
    status = run_data.get("status")
    if status != "SUCCEEDED":
        raise RuntimeError(f"Apify actor run {actor_run_id} has status {status!r}")

    dataset_id = run_data.get("defaultDatasetId") or run_data.get("datasetId")
    if not dataset_id:
        raise RuntimeError(f"Apify actor run {actor_run_id} has no dataset ID")

    dataset_payload = _request_json(
        transport,
        f"{APIFY_API_BASE}/datasets/{dataset_id}/items",
        api_token,
    )
    if isinstance(dataset_payload, list):
        items = dataset_payload
    elif isinstance(dataset_payload, dict) and isinstance(dataset_payload.get("items"), list):
        items = dataset_payload["items"]
    else:
        raise RuntimeError("Unexpected Apify dataset response shape")

    if not all(isinstance(item, dict) for item in items):
        raise RuntimeError("Apify dataset contains a non-object item")

    raw_items = [dict(item) for item in items]
    if raw_output_path is not None:
        output_path = Path(raw_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(raw_items, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return raw_items