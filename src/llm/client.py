"""OpenAI Responses transport with structured output and transport retries."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from src.llm.schemas import JudgeResponse
from src.retry import RetryExhaustedError, is_transient_error, retry_async_call


@dataclass
class AttemptUsage:
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    outcome: Literal["success", "transport_error", "validation_error"]
    estimated: bool


@dataclass
class RawCompletion:
    payload: dict[str, Any]
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    transport_retries: int
    raw_response: str
    attempts: list[AttemptUsage] = field(default_factory=list)


class ResponsesTransport(Protocol):
    async def complete(
        self,
        *,
        prompt: str,
        model_id: str,
        timeout_seconds: float,
        max_transport_retries: int,
        max_output_tokens: int,
        retry_base_backoff_seconds: float = 0.5,
        retry_max_backoff_seconds: float = 8.0,
    ) -> RawCompletion:
        """Return one structured completion."""


class OpenAIResponsesTransport:
    """Production Responses API transport; imports the SDK lazily for tests."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required to use the LLM judge")
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        *,
        prompt: str,
        model_id: str,
        timeout_seconds: float,
        max_transport_retries: int,
        max_output_tokens: int,
        retry_base_backoff_seconds: float = 0.5,
        retry_max_backoff_seconds: float = 8.0,
    ) -> RawCompletion:
        retries = 0
        attempts: list[AttemptUsage] = []

        def on_retry() -> None:
            nonlocal retries
            retries += 1

        started = time.perf_counter()

        async def operation() -> RawCompletion:
            attempt_started = time.perf_counter()
            response = await asyncio.wait_for(
                asyncio.to_thread(self._request, prompt, model_id, timeout_seconds, max_output_tokens),
                timeout=timeout_seconds,
            )
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            output_text = getattr(response, "output_text", "")
            try:
                payload = json.loads(output_text)
            except Exception:
                attempts.append(AttemptUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=time.perf_counter() - attempt_started,
                    outcome="validation_error",
                    estimated=False,
                ))
                raise
            attempts.append(AttemptUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - attempt_started,
                outcome="success",
                estimated=False,
            ))
            return RawCompletion(
                payload=payload,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - started,
                transport_retries=retries,
                raw_response=output_text,
                attempts=attempts,
            )

        def on_failure(exc: Exception, attempt: int) -> None:
            attempts.append(AttemptUsage(
                input_tokens=None,
                output_tokens=None,
                latency_seconds=0.0,
                outcome="transport_error",
                estimated=True,
            ))

        try:
            return await retry_async_call(
                operation,
                max_retries=max_transport_retries,
                base_backoff_seconds=retry_base_backoff_seconds,
                max_backoff_seconds=retry_max_backoff_seconds,
                on_retry=on_retry,
                on_failure=on_failure,
            )
        except Exception as exc:
            setattr(exc, "attempts", attempts)
            raise

    def _request(self, prompt: str, model_id: str, timeout_seconds: float, max_output_tokens: int) -> Any:
        schema = JudgeResponse.model_json_schema()
        validate_strict_schema(schema)
        return self.client.responses.create(
            model=model_id,
            input=prompt,
            timeout=timeout_seconds,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "job_judge",
                    "strict": True,
                    "schema": JudgeResponse.model_json_schema(),
                }
            },
            max_output_tokens=max_output_tokens,
        )


def validate_strict_schema(schema: dict[str, Any]) -> None:
    """Validate the JSON Schema subset required by strict Structured Outputs."""
    definitions = schema.get("$defs", {})

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                raise ValueError("strict schema objects require additionalProperties=false")
            if set(node.get("required", [])) != set(properties):
                raise ValueError("strict schema objects must require every property")
            for property_schema in properties.values():
                visit(property_schema)
        for option in node.get("anyOf", []):
            visit(option)
        visit(node.get("items"))
        if "$ref" in node:
            reference = node["$ref"].rsplit("/", 1)[-1]
            visit(definitions.get(reference))

    visit(schema)


def complete_sync(transport: ResponsesTransport, **kwargs: Any) -> RawCompletion:
    """Call the async transport from synchronous code outside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(transport.complete(**kwargs))
    raise RuntimeError("complete_sync() called inside a running event loop; await complete() instead")
