"""OpenAI Responses transport with structured output and transport retries."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from src.llm.schemas import JudgeResponse
from src.retry import RetryExhaustedError, is_transient_error


@dataclass
class RawCompletion:
    payload: dict[str, Any]
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    transport_retries: int
    raw_response: str


class ResponsesTransport(Protocol):
    async def complete(
        self,
        *,
        prompt: str,
        model_id: str,
        timeout_seconds: float,
        max_transport_retries: int,
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
    ) -> RawCompletion:
        retries = 0
        while True:
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._request, prompt, model_id, timeout_seconds),
                    timeout=timeout_seconds,
                )
                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                output_text = getattr(response, "output_text", "")
                payload = json.loads(output_text)
                return RawCompletion(
                    payload=payload,
                    model_id=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=time.perf_counter() - started,
                    transport_retries=retries,
                    raw_response=output_text,
                )
            except Exception as exc:
                if not is_transient_error(exc) or retries >= max_transport_retries:
                    if is_transient_error(exc):
                        raise RetryExhaustedError(exc, retries) from exc
                    raise
                retries += 1

    def _request(self, prompt: str, model_id: str, timeout_seconds: float) -> Any:
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
        )


def complete_sync(transport: ResponsesTransport, **kwargs: Any) -> RawCompletion:
    """Call the async transport from synchronous code outside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(transport.complete(**kwargs))
    raise RuntimeError("complete_sync() called inside a running event loop; await complete() instead")
