"""Fetch, clean, and persist job postings from an Apify dataset."""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


OUTPUT_PATH = Path("cleaned_jobs.json")
APIFY_DATASET_ITEMS_URL = "https://api.apify.com/v2/datasets/{dataset_id}/items"


class _HTMLStripper(HTMLParser):
    _space_tags = {"br", "div", "li", "p", "td", "th", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._space_tags:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._space_tags:
            self._parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._space_tags:
            self._parts.append(" ")

    def get_text(self) -> str:
        return "".join(self._parts)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing {name} in .env")
    return value.strip()


def _build_dataset_url(dataset_id: str) -> str:
    return APIFY_DATASET_ITEMS_URL.format(dataset_id=dataset_id)


def _strip_html_tags(value: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(value)
    stripper.close()
    return stripper.get_text()


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = _strip_html_tags(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _coerce_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    raise ValueError("Unexpected Apify payload shape; expected a list of job records.")


def _normalize_job_record(record: dict[str, Any]) -> dict[str, Any] | None:
    title = _first_non_empty(record.get("title"), record.get("jobTitle"), record.get("positionName"))
    description = _first_non_empty(record.get("description"), record.get("jobDescription"))
    url = _first_non_empty(record.get("url"), record.get("jobUrl"), record.get("job_url"))
    company = _first_non_empty(record.get("company"), record.get("companyName"), record.get("company_name"))

    if not title or not description or not url:
        return None

    normalized: dict[str, Any] = {
        "title": title,
        "company": company,
        "url": url,
        "description": description,
    }

    optional_fields = (
        "location",
        "employmentType",
        "datePosted",
        "salary",
        "source",
        "jobId",
        "id",
    )
    for field in optional_fields:
        value = record.get(field)
        if value is None:
            continue
        cleaned_value = _clean_text(value)
        if cleaned_value is not None:
            normalized[field] = cleaned_value

    return normalized


def fetch_and_clean_jobs() -> list[dict[str, Any]]:
    load_dotenv()
    api_token = _get_required_env("APIFY_API_TOKEN")
    dataset_id = _get_required_env("APIFY_DATASET_ID")
    url = _build_dataset_url(dataset_id)

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error while fetching Apify dataset: {exc}") from exc

    if response.status_code == 401:
        raise RuntimeError("Apify request failed with 401 Unauthorized. Check APIFY_API_TOKEN.")
    if response.status_code == 404:
        raise RuntimeError("Apify request failed with 404 Not Found. Check APIFY_DATASET_ID.")
    if response.status_code == 429:
        raise RuntimeError("Apify request failed with 429 Too Many Requests. Retry later.")

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Apify request failed with HTTP {response.status_code}: {response.text}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Apify response was not valid JSON.") from exc

    records = _coerce_records(payload)
    cleaned_jobs = []
    for record in records:
        normalized = _normalize_job_record(record)
        if normalized is not None:
            cleaned_jobs.append(normalized)

    return cleaned_jobs


def save_cleaned_jobs(cleaned_jobs: list[dict[str, Any]], output_path: Path = OUTPUT_PATH) -> Path:
    output_path.write_text(json.dumps(cleaned_jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> int:
    try:
        cleaned_jobs = fetch_and_clean_jobs()
        output_path = save_cleaned_jobs(cleaned_jobs)
    except ValueError as exc:
        print(str(exc))
        return 1
    except RuntimeError as exc:
        print(str(exc))
        return 1

    print(f"Saved {len(cleaned_jobs)} cleaned jobs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())