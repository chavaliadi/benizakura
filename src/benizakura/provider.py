from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

from benizakura.errors import JudgeProviderError


@dataclass
class JudgeRequest:
    """Prompt payload submitted to an LLM judge provider.

    Attributes:
        system_prompt: High-level instructions establishing the judge role and response format.
        user_prompt: Specific evaluation details (case question, presented outputs, rubric).
        metadata: Optional dictionary for tracking model parameters or execution context.
    """
    system_prompt: str
    user_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResponse:
    """Raw generation response returned by an LLM judge provider.

    Attributes:
        content: Raw output text emitted by the model.
        metadata: Optional auxiliary details from the provider (token counts, latency, model ID).
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class JudgeProvider(Protocol):
    """Protocol for generating raw model completions for a judge request."""

    def generate(self, request: JudgeRequest) -> JudgeResponse:
        """Execute model generation for the given judge request."""
        ...


class MockJudgeProvider:
    """Deterministic fake provider for local testing and CI without external API calls.

    Allows tests to configure predetermined raw responses through:
    1. A sequence/queue of responses
    2. A default fallback response
    """

    def __init__(
        self,
        default_response: Optional[Union[JudgeResponse, str]] = None,
        responses: Optional[Sequence[Union[JudgeResponse, str]]] = None,
    ):
        self.default_response = self._to_response(default_response) if default_response is not None else None
        self._queue: List[JudgeResponse] = (
            [self._to_response(r) for r in responses] if responses is not None else []
        )
        self._has_queue: bool = responses is not None
        self.calls: List[JudgeRequest] = []

    @staticmethod
    def _to_response(resp: Union[JudgeResponse, str]) -> JudgeResponse:
        if isinstance(resp, str):
            return JudgeResponse(content=resp)
        return resp

    def generate(self, request: JudgeRequest) -> JudgeResponse:
        self.calls.append(request)

        if self._has_queue:
            if self._queue:
                return self._queue.pop(0)
            if self.default_response is not None:
                return self.default_response
            raise JudgeProviderError("Configured mock judge responses are exhausted.")

        if self.default_response is not None:
            return self.default_response

        raise JudgeProviderError("No response configured for MockJudgeProvider.")
