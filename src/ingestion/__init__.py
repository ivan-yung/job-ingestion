"""Ingestion and normalization modules."""

from src.ingestion.deduplicate import deduplicate, find_duplicates
from src.ingestion.normalize import normalize_job_record
from src.ingestion.requirement_classifier import classify_job_record
from src.ingestion.resume_parser import parse_resume

__all__ = [
    "normalize_job_record",
    "deduplicate",
    "find_duplicates",
    "classify_job_record",
    "parse_resume",
]
