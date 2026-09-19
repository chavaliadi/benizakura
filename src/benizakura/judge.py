from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from benizakura.llm_judge import LLMPairwiseJudge
from benizakura.models import (
    CriterionAssessment,
    EvaluationCase,
    PairwiseJudgment,
    PairwiseWinner,
    Rubric,
)
from benizakura.parser import JudgeResponseParser
from benizakura.prompt import JudgePromptBuilder
from benizakura.provider import JudgeProvider, JudgeRequest, JudgeResponse, MockJudgeProvider


__all__ = [
    "PairwiseJudge",
    "PairwiseJudgeCall",
    "MockPairwiseJudge",
    "LLMPairwiseJudge",
    "JudgeProvider",
    "JudgeRequest",
    "JudgeResponse",
    "MockJudgeProvider",
    "JudgePromptBuilder",
    "JudgeResponseParser",
]


@runtime_checkable
class PairwiseJudge(Protocol):
    """Protocol for comparing two presented outputs for an evaluation case."""

    def judge(
        self,
        case: EvaluationCase,
        output_a: str,
        output_b: str,
        rubric: Optional[Rubric] = None,
    ) -> PairwiseJudgment:
        """Compare output A and output B for the given evaluation case and rubric.

        Returns a raw PairwiseJudgment indicating whether position A or B won, or if it was a TIE.
        """
        ...


@dataclass
class PairwiseJudgeCall:
    """Record of an invocation of a pairwise judge for test verification.

    Attributes:
        case: The evaluation case passed to the judge.
        output_a: Text presented in position A.
        output_b: Text presented in position B.
        rubric: The rubric provided for evaluation, if any.
    """
    case: EvaluationCase
    output_a: str
    output_b: str
    rubric: Optional[Rubric] = None


class MockPairwiseJudge:
    """Deterministic mock pairwise judge for testing without external API calls.

    Allows tests to configure explicit raw judgments through:
    1. Per case overrides (mapped by case id)
    2. A sequence or queue of predetermined judgments
    3. A default judgment fallback
    """

    def __init__(
        self,
        default_judgment: Optional[PairwiseJudgment] = None,
        judgments: Optional[Sequence[PairwiseJudgment]] = None,
        case_overrides: Optional[Dict[str, PairwiseJudgment]] = None,
    ):
        self.default_judgment = default_judgment
        self._judgments_queue: List[PairwiseJudgment] = list(judgments) if judgments is not None else []
        self._has_queue: bool = judgments is not None
        self.case_overrides = case_overrides or {}
        self.calls: List[PairwiseJudgeCall] = []

    def judge(
        self,
        case: EvaluationCase,
        output_a: str,
        output_b: str,
        rubric: Optional[Rubric] = None,
    ) -> PairwiseJudgment:
        self.calls.append(
            PairwiseJudgeCall(
                case=case,
                output_a=output_a,
                output_b=output_b,
                rubric=rubric,
            )
        )

        if case.id in self.case_overrides:
            return self.case_overrides[case.id]

        if self._has_queue:
            if self._judgments_queue:
                return self._judgments_queue.pop(0)
            if self.default_judgment is not None:
                return self.default_judgment
            raise IndexError("Configured mock pairwise judgments are exhausted.")

        if self.default_judgment is not None:
            return self.default_judgment

        assessments: List[CriterionAssessment] = []
        if rubric is not None and rubric.criteria:
            assessments = [
                CriterionAssessment(
                    criterion_name=c.name,
                    winner=PairwiseWinner.TIE,
                    rationale=f"Default mock tie for criterion '{c.name}'.",
                )
                for c in rubric.criteria
            ]

        return PairwiseJudgment(
            winner=PairwiseWinner.TIE,
            rationale="Default mock pairwise judgment tie.",
            criterion_assessments=assessments,
        )
