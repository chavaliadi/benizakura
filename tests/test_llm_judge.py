import pytest

from benizakura.errors import JudgeParseError, JudgeProviderError, JudgeValidationError
from benizakura.judge import (
    JudgeResponse,
    LLMPairwiseJudge,
    MockJudgeProvider,
    PairwiseJudge,
)
from benizakura.models import (
    ConsistencyOutcome,
    EvaluationCase,
    NormalizedWinner,
    PairwiseWinner,
    Rubric,
    StandardCriteria,
    classify_consistency,
)
from benizakura.pairwise_runner import BidirectionalPairwiseRunner


def _make_sample_case(case_id: str = "case-eval-1") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        topic="Databases",
        question="Compare optimistic vs pessimistic concurrency control.",
        candidate_answer="Optimistic assumes low contention, validates at commit. Pessimistic locks upfront.",
    )


def test_llm_judge_protocol_compliance():
    provider = MockJudgeProvider(default_response='{"winner": "TIE", "rationale": "Tie."}')
    judge = LLMPairwiseJudge(provider=provider)
    assert isinstance(judge, PairwiseJudge)


def test_llm_judge_end_to_end_successful_judgment():
    valid_json = """
    {
        "winner": "B",
        "rationale": "Response B provided concrete lock granularity trade-offs.",
        "confidence": 0.91,
        "criterion_assessments": [
            {
                "criterion_name": "correctness",
                "winner": "B",
                "rationale": "B accurately explained two-phase locking.",
                "evidence": "2PL guarantees serializability."
            }
        ]
    }
    """
    provider = MockJudgeProvider(
        default_response=JudgeResponse(content=valid_json, metadata={"model": "fake-llm-v1"})
    )
    judge = LLMPairwiseJudge(provider=provider)
    case = _make_sample_case()
    rubric = Rubric(name="DB Rubric", criteria=[StandardCriteria.CORRECTNESS])

    judgment = judge.judge(
        case=case,
        output_a="Baseline answer",
        output_b="Candidate answer",
        rubric=rubric,
    )

    # 1. Verify provider received expected request
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call.metadata["case_id"] == "case-eval-1"
    assert "Response A:\nBaseline answer" in call.user_prompt
    assert "Response B:\nCandidate answer" in call.user_prompt

    # 2. Verify judgment structure
    assert judgment.winner == PairwiseWinner.B
    assert "Response B provided concrete lock granularity" in judgment.rationale
    assert judgment.confidence == 0.91
    assert len(judgment.criterion_assessments) == 1
    assert judgment.criterion_assessments[0].criterion_name == "correctness"
    assert judgment.criterion_assessments[0].winner == PairwiseWinner.B
    assert judgment.metadata["provider_metadata"]["model"] == "fake-llm-v1"


def test_llm_judge_provider_exhaustion_raises():
    provider = MockJudgeProvider(responses=[])
    judge = LLMPairwiseJudge(provider=provider)
    case = _make_sample_case()

    with pytest.raises(JudgeProviderError, match="exhausted"):
        judge.judge(case, "A", "B")


def test_llm_judge_malformed_json_raises_parse_error():
    provider = MockJudgeProvider(default_response="I think Model A is better because of reasons.")
    judge = LLMPairwiseJudge(provider=provider)
    case = _make_sample_case()

    # Never convert malformed model output to a tie!
    with pytest.raises(JudgeParseError, match="Failed to parse model output as JSON"):
        judge.judge(case, "A", "B")


def test_llm_judge_invalid_semantics_raises_validation_error():
    invalid_winner_json = '{"winner": "X", "rationale": "Unknown winner position."}'
    provider = MockJudgeProvider(default_response=invalid_winner_json)
    judge = LLMPairwiseJudge(provider=provider)
    case = _make_sample_case()

    with pytest.raises(JudgeValidationError, match="Invalid winner value 'X'"):
        judge.judge(case, "A", "B")


def test_bidirectional_evaluation_with_llm_judge():
    # Pass 1: A=Baseline, B=Candidate -> Provider returns B as winner
    # Pass 2: A=Candidate, B=Baseline -> Provider returns A as winner
    # This represents a consistent Candidate win.
    pass1_response = """
    {
        "winner": "B",
        "rationale": "Response B has clearer failure domain analysis."
    }
    """
    pass2_response = """
    {
        "winner": "A",
        "rationale": "Response A has clearer failure domain analysis."
    }
    """
    provider = MockJudgeProvider(responses=[pass1_response, pass2_response])
    llm_judge = LLMPairwiseJudge(provider=provider)
    runner = BidirectionalPairwiseRunner(judge=llm_judge)
    case = _make_sample_case()

    result = runner.evaluate(
        case=case,
        baseline_output="Baseline text",
        candidate_output="Candidate text",
    )

    # Verify normalization works identically with LLMPairwiseJudge
    assert result.pass1_judgment.winner == PairwiseWinner.B
    assert result.pass2_judgment.winner == PairwiseWinner.A
    assert result.pass1_normalized_winner == NormalizedWinner.CANDIDATE
    assert result.pass2_normalized_winner == NormalizedWinner.CANDIDATE

    outcome = classify_consistency(
        result.pass1_normalized_winner,
        result.pass2_normalized_winner,
    )
    assert outcome == ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN
    assert len(provider.calls) == 2


def test_bidirectional_position_bias_detection_with_llm_judge():
    # Simulated position bias: model always picks whichever response is in position A
    pass1_response = '{"winner": "A", "rationale": "A favored"}'
    pass2_response = '{"winner": "A", "rationale": "A favored"}'

    provider = MockJudgeProvider(responses=[pass1_response, pass2_response])
    llm_judge = LLMPairwiseJudge(provider=provider)
    runner = BidirectionalPairwiseRunner(judge=llm_judge)
    case = _make_sample_case()

    result = runner.evaluate(
        case=case,
        baseline_output="Baseline text",
        candidate_output="Candidate text",
    )

    # Position A wins in both passes -> Pass 1 Baseline, Pass 2 Candidate
    assert result.pass1_normalized_winner == NormalizedWinner.BASELINE
    assert result.pass2_normalized_winner == NormalizedWinner.CANDIDATE

    outcome = classify_consistency(
        result.pass1_normalized_winner,
        result.pass2_normalized_winner,
    )
    assert outcome == ConsistencyOutcome.POSITION_UNSTABLE
