from benizakura.models import Criterion, EvaluationCase, Rubric, StandardCriteria
from benizakura.prompt import JudgePromptBuilder


def _make_sample_case(case_id: str = "case-sys-101") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        topic="Distributed Systems",
        question="Explain Paxos consensus algorithm phase 1 and phase 2.",
        candidate_answer="Phase 1 prepare/promise, Phase 2 accept/accepted.",
        mode="DEEP_DIVE",
    )


def test_prompt_builder_basic_case_without_rubric():
    case = _make_sample_case()
    builder = JudgePromptBuilder()
    output_a = "Answer A text explaining Paxos"
    output_b = "Answer B text explaining Paxos"

    request = builder.build(case=case, output_a=output_a, output_b=output_b)

    assert request.system_prompt == builder.system_prompt
    assert "Evaluation Case: case-sys-101" in request.user_prompt
    assert "Topic: Distributed Systems" in request.user_prompt
    assert "Question:\nExplain Paxos consensus algorithm phase 1 and phase 2." in request.user_prompt
    assert "# Response A:\nAnswer A text explaining Paxos" in request.user_prompt
    assert "# Response B:\nAnswer B text explaining Paxos" in request.user_prompt
    assert request.metadata["case_id"] == "case-sys-101"
    assert request.metadata["has_rubric"] is False
    assert request.metadata["rubric_name"] is None


def test_prompt_builder_with_rubric():
    case = _make_sample_case()
    rubric = Rubric(
        name="Consensus Evaluation Rubric",
        criteria=[
            StandardCriteria.CORRECTNESS,
            Criterion(name="Edge Cases", description="Handles network partitions.", weight=2.0),
        ],
        instructions="Evaluate handling of split votes and leader drops.",
    )
    builder = JudgePromptBuilder()
    request = builder.build(
        case=case,
        output_a="Ans A",
        output_b="Ans B",
        rubric=rubric,
    )

    assert "Evaluation Rubric: Consensus Evaluation Rubric" in request.user_prompt
    assert "Instructions:\nEvaluate handling of split votes and leader drops." in request.user_prompt
    assert "- **correctness** (weight: 1.0)" in request.user_prompt
    assert "- **Edge Cases** (weight: 2.0): Handles network partitions." in request.user_prompt
    assert request.metadata["has_rubric"] is True
    assert request.metadata["rubric_name"] == "Consensus Evaluation Rubric"


def test_prompt_builder_strict_positional_symmetry():
    # Verify that Candidate vs Baseline is never leaked in prompts
    case = _make_sample_case()
    builder = JudgePromptBuilder()
    request = builder.build(
        case=case,
        output_a="Some text A",
        output_b="Some text B",
    )

    full_prompt = request.system_prompt + "\n" + request.user_prompt
    assert "Candidate" not in full_prompt
    assert "Baseline" not in full_prompt
    assert "Conquer" not in full_prompt

    # Positions A and B must be present
    assert "Response A" in request.user_prompt
    assert "Response B" in request.user_prompt


def test_prompt_builder_deterministic():
    case = _make_sample_case()
    rubric = Rubric(
        name="Test",
        criteria=[StandardCriteria.CORRECTNESS],
        instructions="Instructions",
    )
    builder = JudgePromptBuilder()

    req1 = builder.build(case, "A", "B", rubric)
    req2 = builder.build(case, "A", "B", rubric)

    assert req1.system_prompt == req2.system_prompt
    assert req1.user_prompt == req2.user_prompt
    assert req1.metadata == req2.metadata
