import pytest

from benizakura.judge import MockPairwiseJudge
from benizakura.models import (
    BidirectionalEvaluationResult,
    Criterion,
    CriterionAssessment,
    EvaluationCase,
    NormalizedWinner,
    PairwiseJudgment,
    PairwiseWinner,
    Rubric,
    StandardCriteria,
    classify_consistency,
    ConsistencyOutcome,
)
from benizakura.pairwise_runner import BidirectionalPairwiseRunner


def _create_sample_case(case_id: str = "case-eval-1") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        topic="Architecture",
        question="Describe CQRS architecture and event sourcing.",
        candidate_answer="CQRS separates read and write models. Event sourcing persists state as sequence of events.",
    )


# ============================================================================
# Criterion Tests
# ============================================================================


def test_criterion_valid_construction():
    c = Criterion(
        name="correctness",
        description="Measures factual accuracy and logical soundness.",
        weight=1.5,
    )
    assert c.name == "correctness"
    assert c.description == "Measures factual accuracy and logical soundness."
    assert c.weight == 1.5


def test_criterion_default_weight():
    c = Criterion(name="clarity", description="Assesses readability.")
    assert c.weight == 1.0


def test_criterion_invalid_empty_name():
    with pytest.raises(ValueError, match="Criterion name must be a non-empty string"):
        Criterion(name="", description="Valid description")

    with pytest.raises(ValueError, match="Criterion name must be a non-empty string"):
        Criterion(name="   ", description="Valid description")


def test_criterion_invalid_empty_description():
    with pytest.raises(ValueError, match="Criterion description must be a non-empty string"):
        Criterion(name="correctness", description="")

    with pytest.raises(ValueError, match="Criterion description must be a non-empty string"):
        Criterion(name="correctness", description="   ")


def test_criterion_invalid_negative_weight():
    with pytest.raises(ValueError, match="Criterion weight must be non negative"):
        Criterion(name="correctness", description="Valid", weight=-1.0)


def test_criterion_invalid_non_numeric_weight():
    with pytest.raises(TypeError, match="Criterion weight must be a numeric value"):
        Criterion(name="correctness", description="Valid", weight="heavy")  # type: ignore


def test_criterion_serialization_round_trip():
    c = Criterion(name="depth", description="Technical depth", weight=2.5)
    data = c.to_dict()
    assert data == {
        "name": "depth",
        "description": "Technical depth",
        "weight": 2.5,
    }
    restored = Criterion.from_dict(data)
    assert restored.name == c.name
    assert restored.description == c.description
    assert restored.weight == c.weight


# ============================================================================
# StandardCriteria Tests
# ============================================================================


def test_standard_criteria_catalog():
    criteria = [
        StandardCriteria.CORRECTNESS,
        StandardCriteria.RELEVANCE,
        StandardCriteria.COMPLETENESS,
        StandardCriteria.TECHNICAL_DEPTH,
        StandardCriteria.CLARITY,
        StandardCriteria.GROUNDEDNESS,
    ]
    for c in criteria:
        assert isinstance(c, Criterion)
        assert len(c.name) > 0
        assert len(c.description) > 0
        assert c.weight >= 0.0


# ============================================================================
# Rubric Tests
# ============================================================================


def test_rubric_construction_with_instructions():
    r = Rubric(
        name="General Quality",
        criteria=[
            StandardCriteria.CORRECTNESS,
            StandardCriteria.TECHNICAL_DEPTH,
        ],
        instructions="Compare technical trade-offs objectively. Penalize unverified assertions.",
    )
    assert r.name == "General Quality"
    assert len(r.criteria) == 2
    assert "Compare technical trade-offs" in r.instructions
    assert r.total_weight == 2.0


def test_rubric_invalid_empty_name():
    with pytest.raises(ValueError, match="Rubric name must be a non-empty string"):
        Rubric(name="")


def test_rubric_duplicate_criteria_rejected():
    c1 = Criterion(name="accuracy", description="First")
    c2 = Criterion(name="Accuracy", description="Duplicate with different case")
    with pytest.raises(ValueError, match="Duplicate criterion name detected in rubric"):
        Rubric(name="Invalid Rubric", criteria=[c1, c2])


def test_rubric_get_criterion():
    r = Rubric(
        name="Test Rubric",
        criteria=[
            Criterion(name="Correctness", description="Accurate"),
            Criterion(name="Clarity", description="Clear"),
        ],
    )
    assert r.get_criterion("correctness") is not None
    assert r.get_criterion("CORRECTNESS") is not None
    assert r.get_criterion("clarity").description == "Clear"
    assert r.get_criterion("non-existent") is None


def test_rubric_serialization_round_trip():
    r = Rubric(
        name="Interview Rubric",
        criteria=[
            Criterion(name="Depth", description="Deep understanding", weight=2.0),
            Criterion(name="Clarity", description="Clear structure", weight=1.0),
        ],
        instructions="Focus on concrete examples over hand-waving.",
    )
    data = r.to_dict()
    assert data["name"] == "Interview Rubric"
    assert data["instructions"] == "Focus on concrete examples over hand-waving."
    assert len(data["criteria"]) == 2

    restored = Rubric.from_dict(data)
    assert restored.name == r.name
    assert restored.instructions == r.instructions
    assert len(restored.criteria) == 2
    assert restored.criteria[0].weight == 2.0
    assert restored.criteria[1].name == "Clarity"


# ============================================================================
# CriterionAssessment Tests
# ============================================================================


def test_criterion_assessment_construction():
    ca = CriterionAssessment(
        criterion_name="correctness",
        winner=PairwiseWinner.A,
        rationale="Model A correctly identified ACID properties, while B missed Isolation.",
        confidence=0.95,
        evidence="A explicitly quoted ANSI SQL-92 isolation levels.",
        metadata={"token_len": 45},
    )
    assert ca.criterion_name == "correctness"
    assert ca.winner == PairwiseWinner.A
    assert "ACID" in ca.rationale
    assert ca.reason == ca.rationale
    assert ca.confidence == 0.95
    assert "ANSI" in ca.evidence
    assert ca.metadata["token_len"] == 45


def test_criterion_assessment_backward_compatible_reason():
    ca = CriterionAssessment(
        criterion_name="clarity",
        winner=PairwiseWinner.B,
        reason="Answer B was structured with bullet points.",
    )
    assert ca.rationale == "Answer B was structured with bullet points."
    assert ca.reason == "Answer B was structured with bullet points."


def test_criterion_assessment_invalid_name():
    with pytest.raises(ValueError, match="criterion_name must be a non-empty string"):
        CriterionAssessment(
            criterion_name="",
            winner=PairwiseWinner.A,
            rationale="Valid rationale",
        )


def test_criterion_assessment_invalid_confidence():
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        CriterionAssessment(
            criterion_name="clarity",
            winner=PairwiseWinner.A,
            rationale="Valid",
            confidence=1.5,
        )


def test_criterion_assessment_serialization_round_trip():
    ca = CriterionAssessment(
        criterion_name="completeness",
        winner=PairwiseWinner.TIE,
        rationale="Both models covered all requirements equally.",
        confidence=0.88,
        evidence="Both listed 4 out of 4 criteria.",
        metadata={"annotator": "judge_v1"},
    )
    data = ca.to_dict()
    assert data["criterion_name"] == "completeness"
    assert data["winner"] == "TIE"
    assert data["rationale"] == "Both models covered all requirements equally."
    assert data["confidence"] == 0.88
    assert data["evidence"] == "Both listed 4 out of 4 criteria."

    restored = CriterionAssessment.from_dict(data)
    assert restored.criterion_name == ca.criterion_name
    assert restored.winner == PairwiseWinner.TIE
    assert restored.rationale == ca.rationale
    assert restored.confidence == 0.88
    assert restored.evidence == ca.evidence


# ============================================================================
# PairwiseJudgment Rich Structure Tests
# ============================================================================


def test_pairwise_judgment_with_criterion_assessments():
    ca1 = CriterionAssessment(
        criterion_name="correctness",
        winner=PairwiseWinner.A,
        rationale="A gave mathematically sound formulas.",
    )
    ca2 = CriterionAssessment(
        criterion_name="clarity",
        winner=PairwiseWinner.B,
        rationale="B was clearer and more concise.",
    )

    judgment = PairwiseJudgment(
        winner=PairwiseWinner.A,
        rationale="Overall, correctness in position A outweighs presentation clarity in B.",
        confidence=0.85,
        criterion_assessments=[ca1, ca2],
    )

    assert judgment.winner == PairwiseWinner.A
    assert "correctness in position A outweighs" in judgment.rationale
    assert judgment.reason == judgment.rationale
    assert len(judgment.criterion_assessments) == 2
    assert judgment.get_assessment("correctness") == ca1
    assert judgment.get_assessment("CLARITY") == ca2
    assert judgment.get_assessment("depth") is None


def test_pairwise_judgment_invalid_confidence():
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        PairwiseJudgment(
            winner=PairwiseWinner.A,
            rationale="Valid",
            confidence=-0.1,
        )


def test_pairwise_judgment_serialization_round_trip():
    ca = CriterionAssessment(
        criterion_name="technical_depth",
        winner=PairwiseWinner.B,
        rationale="Position B detailed Raft leader election edge cases.",
        evidence="Leader election timeout handling.",
    )
    judgment = PairwiseJudgment(
        winner=PairwiseWinner.B,
        rationale="Position B provided deeper architectural analysis.",
        confidence=0.92,
        criterion_assessments=[ca],
        metadata={"tokens": 300},
    )

    data = judgment.to_dict()
    assert data["winner"] == "B"
    assert data["confidence"] == 0.92
    assert len(data["criterion_assessments"]) == 1

    restored = PairwiseJudgment.from_dict(data)
    assert restored.winner == PairwiseWinner.B
    assert restored.rationale == judgment.rationale
    assert restored.confidence == 0.92
    assert len(restored.criterion_assessments) == 1
    assert restored.criterion_assessments[0].criterion_name == "technical_depth"
    assert restored.criterion_assessments[0].winner == PairwiseWinner.B


# ============================================================================
# Mock Judge with Rich Rubric Tests
# ============================================================================


def test_mock_judge_auto_generates_criterion_ties_when_rubric_provided():
    rubric = Rubric(
        name="Standard Rubric",
        criteria=[
            StandardCriteria.CORRECTNESS,
            StandardCriteria.CLARITY,
        ],
    )
    judge = MockPairwiseJudge()
    case = _create_sample_case()

    result = judge.judge(case, output_a="Ans A", output_b="Ans B", rubric=rubric)

    assert result.winner == PairwiseWinner.TIE
    assert len(result.criterion_assessments) == 2
    ca_correctness = result.get_assessment("correctness")
    assert ca_correctness is not None
    assert ca_correctness.winner == PairwiseWinner.TIE
    assert "Default mock tie" in ca_correctness.rationale


def test_mock_judge_returns_custom_criterion_assessments():
    ca = CriterionAssessment(
        criterion_name="correctness",
        winner=PairwiseWinner.A,
        rationale="Position A proved correctness via mathematical induction.",
    )
    custom_judgment = PairwiseJudgment(
        winner=PairwiseWinner.A,
        rationale="A is superior.",
        criterion_assessments=[ca],
    )
    judge = MockPairwiseJudge(default_judgment=custom_judgment)
    case = _create_sample_case()

    result = judge.judge(case, output_a="Ans A", output_b="Ans B")
    assert result == custom_judgment
    assert len(result.criterion_assessments) == 1
    assert result.criterion_assessments[0].criterion_name == "correctness"


# ============================================================================
# Bidirectional Runner with Structured Assessments Tests
# ============================================================================


def test_bidirectional_preserves_criterion_assessments_and_normalizes_criteria():
    # Pass 1 (A=Baseline, B=Candidate):
    # Candidate in position B wins overall and on both criteria
    pass1_ca_correctness = CriterionAssessment(
        criterion_name="correctness",
        winner=PairwiseWinner.B,
        rationale="Position B had accurate distributed consensus details.",
    )
    pass1_ca_clarity = CriterionAssessment(
        criterion_name="clarity",
        winner=PairwiseWinner.B,
        rationale="Position B was structured clearly.",
    )
    pass1 = PairwiseJudgment(
        winner=PairwiseWinner.B,
        rationale="Candidate in position B wins across all criteria.",
        criterion_assessments=[pass1_ca_correctness, pass1_ca_clarity],
    )

    # Pass 2 (A=Candidate, B=Baseline):
    # Candidate in position A wins overall and on both criteria
    pass2_ca_correctness = CriterionAssessment(
        criterion_name="correctness",
        winner=PairwiseWinner.A,
        rationale="Position A had accurate distributed consensus details.",
    )
    pass2_ca_clarity = CriterionAssessment(
        criterion_name="clarity",
        winner=PairwiseWinner.A,
        rationale="Position A was structured clearly.",
    )
    pass2 = PairwiseJudgment(
        winner=PairwiseWinner.A,
        rationale="Candidate in position A wins across all criteria.",
        criterion_assessments=[pass2_ca_correctness, pass2_ca_clarity],
    )

    mock_judge = MockPairwiseJudge(judgments=[pass1, pass2])
    runner = BidirectionalPairwiseRunner(mock_judge)
    case = _create_sample_case()

    rubric = Rubric(
        name="Consensus Rubric",
        criteria=[StandardCriteria.CORRECTNESS, StandardCriteria.CLARITY],
        instructions="Compare fault tolerance and consensus handling.",
    )

    result = runner.evaluate(
        case=case,
        baseline_output="Baseline answer",
        candidate_output="Candidate answer",
        rubric=rubric,
    )

    # Overall normalized winners
    assert result.pass1_normalized_winner == NormalizedWinner.CANDIDATE
    assert result.pass2_normalized_winner == NormalizedWinner.CANDIDATE
    assert classify_consistency(result.pass1_normalized_winner, result.pass2_normalized_winner) == ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN

    # Criterion-level normalized winners
    norm1_corr, norm2_corr = result.get_criterion_normalized_winners("correctness")
    assert norm1_corr == NormalizedWinner.CANDIDATE
    assert norm2_corr == NormalizedWinner.CANDIDATE

    norm1_clar, norm2_clar = result.get_criterion_normalized_winners("clarity")
    assert norm1_clar == NormalizedWinner.CANDIDATE
    assert norm2_clar == NormalizedWinner.CANDIDATE

    # Non-existent criterion returns (None, None)
    assert result.get_criterion_normalized_winners("non_existent") == (None, None)


def test_bidirectional_result_serialization_round_trip():
    ca = CriterionAssessment(
        criterion_name="technical_depth",
        winner=PairwiseWinner.A,
        rationale="Position A explained write amplification.",
    )
    j1 = PairwiseJudgment(
        winner=PairwiseWinner.A,
        rationale="Pass 1 judgment",
        criterion_assessments=[ca],
    )
    j2 = PairwiseJudgment(
        winner=PairwiseWinner.B,
        rationale="Pass 2 judgment",
    )
    res = BidirectionalEvaluationResult(
        pass1_judgment=j1,
        pass2_judgment=j2,
        pass1_normalized_winner=NormalizedWinner.BASELINE,
        pass2_normalized_winner=NormalizedWinner.BASELINE,
        metadata={"run": "test"},
    )

    data = res.to_dict()
    restored = BidirectionalEvaluationResult.from_dict(data)

    assert restored.pass1_normalized_winner == NormalizedWinner.BASELINE
    assert restored.pass2_normalized_winner == NormalizedWinner.BASELINE
    assert len(restored.pass1_judgment.criterion_assessments) == 1
    assert restored.pass1_judgment.criterion_assessments[0].criterion_name == "technical_depth"
    assert restored.metadata["run"] == "test"
