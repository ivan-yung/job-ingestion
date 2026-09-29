"""Synchronous, resumable orchestration of the ranking stages."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.filtering.hard_filters import apply_hard_filters
from src.ingestion.deduplicate import deduplicate
from src.ingestion.normalize import normalize_apify_record
from src.ingestion.requirement_classifier import classify_job_record
from src.ingestion import resume_parser
from src.ingestion.resume_parser import parse_resume
from src.llm.judge import judge_jobs
from src.models.candidate import CandidateProfile
from src.models.job import ClassifiedJobRecord, JobRecord
from src.orchestration.apify_adapter import fetch_apify_dataset
from src.retrieval.retrieve import retrieve_jobs
from src.reranking.rerank import RerankOutput, rerank_jobs


STAGES = ("ingest", "retrieval", "rerank", "llm_judge")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resume_source_hash(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def candidate_profile_hash(candidate: CandidateProfile) -> str:
    payload = candidate.model_dump(exclude={"resume_raw_text"})
    return _sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode())


def resume_parser_fingerprint() -> str:
    source = inspect.getsource(resume_parser)
    return _sha256_bytes(source.encode())


def should_skip_stage(
    stage_name: str,
    manifest: dict[str, Any],
    current_config_version: str,
    current_resume_source_hash: str,
    current_candidate_profile_hash: str | None = None,
    current_resume_parser_fingerprint: str | None = None,
) -> bool:
    """Check whether one stage's artifact is valid for the current inputs."""
    stage_entry = manifest.get("stages", {}).get(stage_name)
    if stage_entry is None or stage_entry.get("status") != "ok":
        return False
    if manifest.get("config_version") != current_config_version:
        return False
    if manifest.get("resume_source_hash") != current_resume_source_hash:
        return False
    if current_candidate_profile_hash is not None and manifest.get("candidate_profile_hash") != current_candidate_profile_hash:
        return False
    if current_resume_parser_fingerprint is not None and manifest.get("resume_parser_fingerprint") != current_resume_parser_fingerprint:
        return False
    return True


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing pipeline artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupt pipeline artifact: {path}") from exc


def _new_manifest(
    run_id: str,
    apify_run_id: str | None,
    config: PipelineConfig,
    source_hash: str,
    profile_hash: str,
    parser_fingerprint: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "apify_run_id": apify_run_id,
        "config_version": config.config_version,
        "resume_source_hash": source_hash,
        "candidate_profile_hash": profile_hash,
        "resume_parser_fingerprint": parser_fingerprint,
        "stages": {stage: {"status": "pending"} for stage in STAGES},
        "errors": [],
    }


def _job_map(jobs: list[ClassifiedJobRecord]) -> dict[str, ClassifiedJobRecord]:
    return {job.job_id: job for job in jobs}


class Pipeline:
    """Run the existing stages and persist every boundary for resumption."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        run_id: str,
        resume_path: str | Path,
        apify_run_id: str | None = None,
        run_dir: str | Path | None = None,
        skip_ingest: bool = False,
    ) -> dict[str, Any]:
        candidate = parse_resume(resume_path)
        source_hash = resume_source_hash(resume_path)
        profile_hash = candidate_profile_hash(candidate)
        parser_fingerprint = resume_parser_fingerprint()
        directory = Path(run_dir or Path(self.config.runs_dir) / run_id)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        manifest_exists = manifest_path.exists()
        manifest = _read_json(manifest_path) if manifest_exists else _new_manifest(
            run_id, apify_run_id, self.config, source_hash, profile_hash, parser_fingerprint
        )
        resume_metadata_matches = not manifest_exists or (
            manifest.get("config_version") == self.config.config_version
            and manifest.get("resume_source_hash") == source_hash
            and manifest.get("candidate_profile_hash") == profile_hash
            and manifest.get("resume_parser_fingerprint") == parser_fingerprint
        )
        manifest.update(
            {
                "run_id": run_id,
                "apify_run_id": apify_run_id or manifest.get("apify_run_id"),
                "config_version": self.config.config_version,
                "resume_source_hash": source_hash,
                "candidate_profile_hash": profile_hash,
                "resume_parser_fingerprint": parser_fingerprint,
            }
        )
        _write_json(manifest_path, manifest)

        prefix_valid = True
        if skip_ingest:
            prefix_valid = resume_metadata_matches and manifest.get("stages", {}).get("ingest", {}).get("status") == "ok"
            if not prefix_valid:
                raise RuntimeError("Cannot skip ingestion: ingest artifact is stale or missing")
            ingest_payload = _read_json(directory / "ingest.json")
        else:
            prefix_valid = resume_metadata_matches and manifest.get("stages", {}).get("ingest", {}).get("status") == "ok"
            ingest_payload = _read_json(directory / "ingest.json") if prefix_valid else self._run_ingest(
                directory, manifest, apify_run_id, candidate
            )

        classified_jobs = [ClassifiedJobRecord.model_validate(item["job"]) for item in ingest_payload["retained"]]
        retrieval_payload = self._stage_or_load(
            "retrieval", directory, manifest, prefix_valid, lambda: self._run_retrieval(candidate, classified_jobs)
        )
        prefix_valid = prefix_valid and retrieval_payload["_skipped"]
        retrieval_payload.pop("_skipped", None)

        rerank_payload = self._stage_or_load(
            "rerank", directory, manifest, prefix_valid,
            lambda: self._run_rerank(candidate, classified_jobs, retrieval_payload),
        )
        prefix_valid = prefix_valid and rerank_payload["_skipped"]
        rerank_payload.pop("_skipped", None)

        judge_payload = self._stage_or_load(
            "llm_judge", directory, manifest, prefix_valid,
            lambda: self._run_judge(candidate, classified_jobs, rerank_payload),
        )
        judge_payload.pop("_skipped", None)
        _write_json(directory / "manifest.json", manifest)
        return {"directory": str(directory), "manifest": manifest, "candidate": candidate.model_dump(), "ingest": ingest_payload, "retrieval": retrieval_payload, "rerank": rerank_payload, "llm_judge": judge_payload}

    def _stage_or_load(self, stage: str, directory: Path, manifest: dict[str, Any], prefix_valid: bool, operation: Any) -> dict[str, Any]:
        can_skip = prefix_valid and manifest.get("stages", {}).get(stage, {}).get("status") == "ok"
        if can_skip:
            payload = _read_json(directory / f"{stage}.json")
            payload["_skipped"] = True
            return payload
        started = time.perf_counter()
        payload = operation()
        duration_s = time.perf_counter() - started
        _write_json(directory / f"{stage}.json", payload)
        manifest["stages"][stage] = {
            "status": "partial" if payload.get("partial") else "ok",
            "input_count": payload.get("input_count", 0),
            "output_count": payload.get("output_count", 0),
            "duration_s": round(duration_s, 3),
        }
        if stage == "llm_judge":
            metrics = payload.get("metrics", {})
            manifest["stages"][stage].update(
                {
                    "cost_usd": metrics.get("total_accounted_cost_usd", 0.0),
                    "cache_hits": metrics.get("cache_hits", 0),
                    "budget_stop_reason": metrics.get("budget_stop_reason"),
                    "failed_job_ids": [
                        item["job_id"]
                        for item in payload.get("results", [])
                        if item.get("judge_status") != "accepted"
                    ],
                }
            )
        _write_json(directory / "manifest.json", manifest)
        payload["_skipped"] = False
        return payload

    def _run_ingest(self, directory: Path, manifest: dict[str, Any], apify_run_id: str | None, candidate: CandidateProfile) -> dict[str, Any]:
        if not apify_run_id:
            raise ValueError("apify_run_id is required when ingestion is not skipped")
        token = __import__("os").getenv(self.config.apify_api_token_env)
        if not token:
            raise ValueError(f"Missing {self.config.apify_api_token_env} for Apify ingestion")
        started = time.perf_counter()
        raw_jobs = fetch_apify_dataset(apify_run_id, token, raw_output_path=directory / "raw_apify_dataset.json")
        normalized = [normalize_apify_record(raw) for raw in raw_jobs]
        jobs = deduplicate([job for job in normalized if job is not None])
        classified = [classify_job_record(job) for job in jobs]
        retained, decisions = apply_hard_filters(candidate, classified)
        decision_map = {decision.job_id: decision.rejection_reasons for decision in decisions if decision.rejected}
        filtered = [{"job": job.model_dump(), "reasons": decision_map.get(job.job_id, [])} for job in classified if job.job_id in decision_map]
        payload = {"input_count": len(raw_jobs), "output_count": len(retained), "retained": [{"job": job.model_dump()} for job in retained], "filtered": filtered}
        _write_json(directory / "ingest.json", payload)
        manifest["stages"]["ingest"] = {
            "status": "ok",
            "input_count": len(raw_jobs),
            "output_count": len(retained),
            "duration_s": round(time.perf_counter() - started, 3),
        }
        _write_json(directory / "manifest.json", manifest)
        return payload

    def _run_retrieval(self, candidate: CandidateProfile, jobs: list[ClassifiedJobRecord]) -> dict[str, Any]:
        results = retrieve_jobs(
            candidate,
            jobs,
            top_k=self.config.retrieval.top_k,
            db_path=self.config.cache_db_path,
            semantic_weights=self.config.retrieval.semantic_weights,
            rrf_k=self.config.retrieval.rrf_k,
        )
        return {"input_count": len(jobs), "output_count": len(results), "results": results}

    def _run_rerank(self, candidate: CandidateProfile, jobs: list[ClassifiedJobRecord], retrieval_payload: dict[str, Any]) -> dict[str, Any]:
        result: RerankOutput = rerank_jobs(
            candidate,
            retrieval_payload["results"],
            jobs,
            top_k=self.config.rerank.top_k,
            score_weights=self.config.rerank.feature_weights,
        )
        return {"input_count": len(retrieval_payload["results"]), "output_count": len(result.scored), "scored": result.scored, "gated": [item.model_dump() for item in result.gated]}

    def _run_judge(self, candidate: CandidateProfile, jobs: list[ClassifiedJobRecord], rerank_payload: dict[str, Any]) -> dict[str, Any]:
        jobs_by_id = _job_map(jobs)
        scored_jobs = [jobs_by_id[item["job_id"]] for item in rerank_payload["scored"] if item["job_id"] in jobs_by_id]
        judge_config = self.config.judge.model_copy(update={"cache_db_path": self.config.cache_db_path})
        results, metrics = judge_jobs(candidate, scored_jobs, judge_config)
        serialized = []
        for result in results:
            serialized.append({"job_id": result.job_id, "judge_status": result.judge_status, "response": result.response.model_dump() if result.response else None, "score": result.score.model_dump() if hasattr(result.score, "model_dump") else result.score, "error": result.error, "cache_hit": result.cache_hit, "model_id": result.model_id, "usage": result.usage})
        return {"input_count": len(scored_jobs), "output_count": len(serialized), "partial": any(item["judge_status"] != "accepted" for item in serialized), "results": serialized, "metrics": metrics.as_dict()}