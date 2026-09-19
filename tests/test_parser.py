import pytest

from benizakura.errors import JudgeParseError, JudgeValidationError
from benizakura.models import PairwiseWinner, Rubric, StandardCriteria
from benizakura.parser import JudgeResponseParser


def test_parse_valid_raw_json():
    raw = """
    {
        "winner": "A",
        "rationale": "Position A provided comprehensive mathematical proofs.",
        "confidence": 0.88,
        "criterion_assessments": [
            {
                "criterion_name": "correctness",
                "winner": "A",
                "rationale": "Formulas in A were correct.",
                "evidence": "Quoted Theorem 1.1"
            }
        ],
        "metadata": {"tokens": 120}
    }
    """
    parser = JudgeResponseParser()
    judgment = parser.parse(raw)

    assert judgment.winner == PairwiseWinner.A
    assert judgment.rationale == "Position A provided comprehensive mathematical proofs."
    assert judgment.confidence == 0.88
    assert len(judgment.criterion_assessments) == 1
    ca = judgment.criterion_assessments[0]
    assert ca.criterion_name == "correctness"
    assert ca.winner == PairwiseWinner.A
    assert ca.evidence == "Quoted Theorem 1.1"
    assert judgment.metadata["tokens"] == 120


def test_parse_valid_markdown_fenced_json():
    raw = """
    Here is my evaluation of the two responses:
    ```json
    {
        "winner": "B",
        "rationale": "Position B was structured with concrete bullet points.",
        "confidence": 0.95,
        "criterion_assessments": []
    }
    ```
    Hope this helps!
    """
    parser = JudgeResponseParser()
    judgment = parser.parse(raw)

    assert judgment.winner == PairwiseWinner.B
    assert "concrete bullet points" in judgment.rationale
    assert judgment.confidence == 0.95
    assert judgment.criterion_assessments == []


def test_parse_reason_alias_accepted():
    raw = '{"winner": "TIE", "reason": "Both answers are identical."}'
    parser = JudgeResponseParser()
    judgment = parser.parse(raw)

    assert judgment.winner == PairwiseWinner.TIE
    assert judgment.rationale == "Both answers are identical."
    assert judgment.reason == "Both answers are identical."


def test_parse_rubric_matching_and_canonicalization():
    rubric = Rubric(
        name="Tech Rubric",
        criteria=[StandardCriteria.CORRECTNESS, StandardCriteria.TECHNICAL_DEPTH],
    )
    # Model returns "CORRECTNESS" in uppercase
    raw = """
    {
        "winner": "A",
        "rationale": "A is technically superior.",
        "criterion_assessments": [
            {
                "criterion_name": "CORRECTNESS",
                "winner": "A",
                "rationale": "A is accurate."
            }
        ]
    }
    """
    parser = JudgeResponseParser()
    judgment = parser.parse(raw, rubric=rubric)

    assert len(judgment.criterion_assessments) == 1
    # Canonicalized to rubric's name: "correctness"
    assert judgment.criterion_assessments[0].criterion_name == "correctness"


def test_parse_empty_or_whitespace_raises_parse_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeParseError, match="empty or not a string"):
        parser.parse("")

    with pytest.raises(JudgeParseError, match="empty or not a string"):
        parser.parse("   ")


def test_parse_malformed_json_raises_parse_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeParseError, match="Failed to parse model output as JSON"):
        parser.parse("Not a JSON string {winner: A}")


def test_parse_top_level_not_dict_raises_validation_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Expected JSON object at top level"):
        parser.parse('["winner", "A"]')


def test_parse_missing_winner_raises_validation_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Missing required field 'winner'"):
        parser.parse('{"rationale": "Some rationale without a winner."}')


def test_parse_invalid_winner_value_raises_validation_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Invalid winner value 'C'"):
        parser.parse('{"winner": "C", "rationale": "Valid rationale"}')

    with pytest.raises(JudgeValidationError, match="Field 'winner' must be a string"):
        parser.parse('{"winner": 1, "rationale": "Valid rationale"}')


def test_parse_missing_or_empty_rationale_raises_validation_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Missing or empty required field 'rationale'"):
        parser.parse('{"winner": "A"}')

    with pytest.raises(JudgeValidationError, match="Missing or empty required field 'rationale'"):
        parser.parse('{"winner": "A", "rationale": "   "}')


def test_parse_invalid_confidence_bounds_raises_validation_error():
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Field 'confidence' must be between 0.0 and 1.0"):
        parser.parse('{"winner": "A", "rationale": "OK", "confidence": 1.5}')

    with pytest.raises(JudgeValidationError, match="Field 'confidence' must be between 0.0 and 1.0"):
        parser.parse('{"winner": "A", "rationale": "OK", "confidence": -0.1}')

    with pytest.raises(JudgeValidationError, match="Field 'confidence' must be numeric"):
        parser.parse('{"winner": "A", "rationale": "OK", "confidence": "high"}')


def test_parse_unknown_criterion_raises_validation_error():
    rubric = Rubric(name="Simple", criteria=[StandardCriteria.CORRECTNESS])
    raw = """
    {
        "winner": "A",
        "rationale": "A is better.",
        "criterion_assessments": [
            {
                "criterion_name": "unknown_metric",
                "winner": "A",
                "rationale": "A had better vibe."
            }
        ]
    }
    """
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Unknown criterion 'unknown_metric' not defined in rubric"):
        parser.parse(raw, rubric=rubric)


def test_parse_criterion_assessment_invalid_winner_raises_validation_error():
    raw = """
    {
        "winner": "A",
        "rationale": "A is better.",
        "criterion_assessments": [
            {
                "criterion_name": "correctness",
                "winner": "C",
                "rationale": "Invalid position."
            }
        ]
    }
    """
    parser = JudgeResponseParser()
    with pytest.raises(JudgeValidationError, match="Invalid winner 'C' in criterion assessment"):
        parser.parse(raw)
