"""Shared, versioned pipeline configuration loaded from TOML."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.llm.schemas import JudgeConfig
from src.orchestration.application_priority import ApplicationPriorityConfig
from src.reranking.scoring import WEIGHTS as DEFAULT_RERANK_WEIGHTS


class RetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval."""

    top_k: int = Field(default=150, gt=0)
    semantic_weights: dict[str, float] = Field(
        default_factory=lambda: {"overall": 0.50, "skills": 0.30, "role": 0.20}
    )
    rrf_k: int = Field(default=60, gt=0)

    @field_validator("semantic_weights")
    @classmethod
    def validate_semantic_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        if set(weights) != {"overall", "skills", "role"}:
            raise ValueError("semantic_weights must contain overall, skills, and role")
        if any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("semantic_weights must be non-negative and sum to 1.0")
        return weights


class RerankConfig(BaseModel):
    """Configuration for deterministic reranking."""

    top_k: int = Field(default=50, gt=0)
    feature_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_RERANK_WEIGHTS))

    @field_validator("feature_weights")
    @classmethod
    def validate_feature_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        if set(weights) != set(DEFAULT_RERANK_WEIGHTS):
            raise ValueError(f"feature_weights must contain exactly: {sorted(DEFAULT_RERANK_WEIGHTS)}")
        if any(value < 0 for value in weights.values()):
            raise ValueError("rerank feature weights must be non-negative")
        if weights and abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("rerank feature weights must sum to 1.0")
        return weights


class PipelineConfig(BaseModel):
    """All configuration required to execute a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    config_version: str = Field(min_length=1)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    judge: JudgeConfig
    application_priority: ApplicationPriorityConfig = Field(default_factory=ApplicationPriorityConfig)
    apify_api_token_env: str = "APIFY_API_TOKEN"
    cache_db_path: str = "cache.sqlite"
    runs_dir: str = "runs"

    @model_validator(mode="after")
    def validate_judge_pricing(self) -> "PipelineConfig":
        self.judge.validate_pricing()
        return self

    @classmethod
    def from_toml(cls, path: str | Path, *, require_apify_token: bool = False) -> "PipelineConfig":
        config_path = Path(path)
        with config_path.open("rb") as config_file:
            values = tomllib.load(config_file)
        config = cls.model_validate(values)
        if require_apify_token and not os.getenv(config.apify_api_token_env):
            raise ValueError(f"Missing {config.apify_api_token_env} for Apify ingestion")
        return config