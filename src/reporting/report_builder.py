"""Build the self-contained HTML and structured JSON run reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import PipelineConfig
from src.llm.schemas import FinalScore, JudgeResponse
from src.models.candidate import CandidateProfile
from src.models.job import ClassifiedJobRecord
from src.orchestration.application_priority import application_priority


def _remote_fit(candidate: CandidateProfile, job: ClassifiedJobRecord) -> float:
    preference = candidate.remote_preference
    if preference is None:
        return 0.5
    if preference == "remote":
        return 1.0 if job.remote_type == "remote" else 0.0
    if preference == job.remote_type:
        return 1.0
    if preference == "hybrid" and job.remote_type in {"remote", "hybrid"}:
        return 1.0
    return 0.0


def _job_lookup(payload: dict[str, Any]) -> dict[str, ClassifiedJobRecord]:
    jobs = [ClassifiedJobRecord.model_validate(item["job"]) for item in payload["ingest"]["retained"]]
    return {job.job_id: job for job in jobs}


def build_report(
    pipeline_payload: dict[str, Any],
    output_dir: str | Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Write `report.html` and `results.json`, returning the report envelope."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = CandidateProfile.model_validate(pipeline_payload["candidate"])
    jobs = _job_lookup(pipeline_payload)
    scored_by_id = {item["job_id"]: item for item in pipeline_payload["rerank"]["scored"]}
    judge_by_id = {item["job_id"]: item for item in pipeline_payload["llm_judge"]["results"]}

    ranked: list[dict[str, Any]] = []
    not_recommended: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job_id, judge_item in judge_by_id.items():
        job = jobs.get(job_id)
        if job is None:
            continue
        if judge_item["judge_status"] != "accepted" or judge_item["score"] is None:
            failures.append(judge_item)
            continue
        final_score = FinalScore.model_validate(judge_item["score"])
        response = JudgeResponse.model_validate(judge_item["response"])
        feature_breakdown = scored_by_id.get(job_id, {}).get("feature_breakdown", {})
        priority = application_priority(
            final_score,
            float(feature_breakdown.get("location_fit", 0.0)),
            _remote_fit(candidate, job),
            job.company,
            config.application_priority,
            judge_response=response,
            job_id=job_id,
            metadata={"feature_breakdown": feature_breakdown, "job": job.model_dump()},
        )
        row = {
            "job": job.model_dump(),
            "priority": priority.model_dump(mode="json"),
            "feature_breakdown": feature_breakdown,
            "judge": judge_item,
        }
        if final_score.recommendation.value == "do_not_apply":
            not_recommended.append(row)
        else:
            ranked.append(row)

    ranked.sort(key=lambda row: (-row["priority"]["application_priority_score"], -row["priority"]["final_score"]["qualification_score"], row["job"]["job_id"]))
    not_recommended.sort(key=lambda row: (-row["priority"]["application_priority_score"], row["job"]["job_id"]))
    filtered = pipeline_payload["ingest"].get("filtered", [])
    gated = pipeline_payload["rerank"].get("gated", [])
    envelope = {
        "schema_version": "1",
        "manifest": pipeline_payload["manifest"],
        "ranked": ranked,
        "not_recommended": not_recommended,
        "filtered": filtered,
        "gated": gated,
        "judge_failures": failures,
    }
    (directory / "results.json").write_text(json.dumps(envelope, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    template_dir = Path(__file__).parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = environment.get_template("report.html.j2")
    html = template.render(**envelope)
    (directory / "report.html").write_text(html, encoding="utf-8")
    return envelope