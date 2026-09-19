from __future__ import annotations

from typing import Optional

from benizakura.models import EvaluationCase, PairwiseJudgment, Rubric
from benizakura.parser import JudgeResponseParser
from benizakura.prompt import JudgePromptBuilder
from benizakura.provider import JudgeProvider


class LLMPairwiseJudge:
    """Pairwise judge coordinating prompt generation, provider execution, and response parsing.

    Satisfies the PairwiseJudge protocol without depending on specific LLM vendor SDKs.
    """

    def __init__(
        self,
        provider: JudgeProvider,
        prompt_builder: Optional[JudgePromptBuilder] = None,
        parser: Optional[JudgeResponseParser] = None,
    ):
        self.provider = provider
        self.prompt_builder = prompt_builder or JudgePromptBuilder()
        self.parser = parser or JudgeResponseParser()

    def judge(
        self,
        case: EvaluationCase,
        output_a: str,
        output_b: str,
        rubric: Optional[Rubric] = None,
    ) -> PairwiseJudgment:
        """Execute the pairwise judgment pipeline: build request -> generate -> parse & validate."""
        request = self.prompt_builder.build(
            case=case,
            output_a=output_a,
            output_b=output_b,
            rubric=rubric,
        )

        raw_response = self.provider.generate(request)

        judgment = self.parser.parse(raw_response.content, rubric=rubric)

        # Merge provider execution metadata if present
        if raw_response.metadata:
            judgment.metadata.setdefault("provider_metadata", raw_response.metadata)

        return judgment
