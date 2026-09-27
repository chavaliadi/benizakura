from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import (
    REQUIRED_TOTAL_CASES,
    compute_benchmark_hash,
    load_benchmark_data,
)
from benizakura.errors import (
    CalibrationPrerequisiteError,
    JudgeParseError,
    JudgeProviderError,
    JudgeRateLimitError,
    JudgeTimeoutError,
    JudgeValidationError,
)
from benizakura.experiment import (
    AnthropicJudgeProvider,
    CallStatus,
    DEFAULT_JUDGE_A,
    DEFAULT_JUDGE_B,
    ExecutionStatus,
    ExperimentCaseRun,
    ExperimentPlan,
    ExperimentVariant,
    JudgeConfig,
    MockExperimentJudgeProvider,
    OpenAIJudgeProvider,
    check_calibration_prerequisites,
    execute_experiment,
    plan_experiment,
    sanitize_secrets,
    summarize_experiment,
)
from benizakura.models import EvaluationCase, PairwiseWinner
from benizakura.parser import JudgeResponseParser
from benizakura.prompt import JudgePromptBuilder
from benizakura.provider import JudgeRequest, JudgeResponse


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")
GROUND_TRUTH_PATH = Path("evals/conquer/ground_truth_v1.json")
EXPECTED_SHA = "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"


# ============================================================================
# 1. Prerequisite Gate Tests
# ============================================================================

def test_prerequisite_blocks_when_ground_truth_missing(tmp_path):
    fake_gt = tmp_path / "non_existent_gt.json"
    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=fake_gt,
        expected_benchmark_hash=EXPECTED_SHA,
    )
    assert not prereq.is_ready
    assert any("not found" in r for r in prereq.blocking_reasons)


def test_prerequisite_blocks_when_ground_truth_unfrozen():
    # Currently ground_truth_v1.json is PENDING_HUMAN_ANNOTATION
    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        expected_benchmark_hash=EXPECTED_SHA,
    )
    assert not prereq.is_ready
    assert prereq.benchmark_hash_valid is True
    assert prereq.ground_truth_frozen is False
    assert any("GROUND_TRUTH_FROZEN" in r for r in prereq.blocking_reasons)


def test_prerequisite_blocks_when_benchmark_hash_mismatched(tmp_path):
    # Altered expected hash
    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        expected_benchmark_hash="0000000000000000000000000000000000000000000000000000000000000000",
    )
    assert not prereq.is_ready
    assert prereq.benchmark_hash_valid is False
    assert any("SHA-256 mismatch" in r for r in prereq.blocking_reasons)


def test_prerequisite_blocks_when_kappa_below_threshold(tmp_path):
    low_kappa_gt = tmp_path / "low_kappa_gt.json"
    with open(low_kappa_gt, "w", encoding="utf-8") as f:
        json.dump({
            "status": "GROUND_TRUTH_FROZEN",
            "agreement_statistics": {"cohens_kappa": 0.42},
        }, f)

    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=low_kappa_gt,
        expected_benchmark_hash=EXPECTED_SHA,
    )
    assert not prereq.is_ready
    assert prereq.kappa_sufficient is False
    assert any("Cohen's kappa is 0.4200 < 0.60" in r for r in prereq.blocking_reasons)


def test_prerequisite_passes_when_all_conditions_satisfied(tmp_path):
    valid_gt = tmp_path / "valid_frozen_gt.json"
    with open(valid_gt, "w", encoding="utf-8") as f:
        json.dump({
            "status": "GROUND_TRUTH_FROZEN",
            "agreement_statistics": {"cohens_kappa": 0.78},
        }, f)

    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=valid_gt,
        expected_benchmark_hash=EXPECTED_SHA,
    )
    assert prereq.is_ready is True
    assert prereq.benchmark_hash_valid is True
    assert prereq.ground_truth_frozen is True
    assert prereq.kappa_sufficient is True
    assert len(prereq.blocking_reasons) == 0


# ============================================================================
# 2. Secret Sanitization Tests
# ============================================================================

def test_secret_sanitization_scrubs_keys_and_tokens():
    payload = {
        "api_key": "sk-ant-api03-abcdef1234567890",
        "authorization": "Bearer sk-openai-1234567890abcdef1234567890",
        "nested": {
            "token": "secret_token_val",
            "normal_field": "safe_value",
            "message": "Encountered sk-ant-secretkey in error trace.",
        },
    }
    cleaned = sanitize_secrets(payload)
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["authorization"] == "[REDACTED]"
    assert cleaned["nested"]["token"] == "[REDACTED]"
    assert cleaned["nested"]["normal_field"] == "safe_value"
    assert "sk-ant-secretkey" not in cleaned["nested"]["message"]
    assert "[REDACTED_ANTHROPIC_KEY]" in cleaned["nested"]["message"]


# ============================================================================
# 3. Provider Abstraction Tests
# ============================================================================

def test_anthropic_provider_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicJudgeProvider(api_key=None)
    req = JudgeRequest(system_prompt="sys", user_prompt="usr")
    with pytest.raises(JudgeProviderError, match="ANTHROPIC_API_KEY environment variable is not set"):
        provider.generate(req, DEFAULT_JUDGE_A)


def test_openai_provider_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIJudgeProvider(api_key=None)
    req = JudgeRequest(system_prompt="sys", user_prompt="usr")
    with pytest.raises(JudgeProviderError, match="OPENAI_API_KEY environment variable is not set"):
        provider.generate(req, DEFAULT_JUDGE_B)


def test_mock_provider_returns_configured_response():
    provider = MockExperimentJudgeProvider(default_response=json.dumps({
        "winner": "A",
        "rationale": "Clear algorithmic superiority.",
        "confidence": 0.95,
        "criterion_assessments": [],
    }))
    req = JudgeRequest(system_prompt="sys", user_prompt="usr", metadata={"case_id": "dsa-001"})
    resp = provider.generate(req, DEFAULT_JUDGE_A)
    assert resp.content is not None
    data = json.loads(resp.content)
    assert data["winner"] == "A"


# ============================================================================
# 4. Experiment Planning & Workload Calculation Tests
# ============================================================================

def test_plan_experiment_calculates_workload_correctly():
    plan = plan_experiment(
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=GROUND_TRUTH_PATH,
        experiment_id="test_plan_v1",
        judges=[DEFAULT_JUDGE_A, DEFAULT_JUDGE_B],
        variants=[
            ExperimentVariant.POINTWISE,        # 2 calls * 30 = 60
            ExperimentVariant.SINGLE_DIRECTION, # 1 call  * 30 = 30
            ExperimentVariant.BIDIRECTIONAL,    # 2 calls * 30 = 60
        ],
        repeat_count=1,
        is_dry_run=True,
    )
    # Total calls per judge = 60 + 30 + 60 = 150. Across 2 judges = 300.
    assert plan.total_requests == 300
    assert plan.case_count == REQUIRED_TOTAL_CASES
    assert plan.is_dry_run is True
    assert "judge_a_claude" in plan.summary()
    assert "judge_b_gpt4o" in plan.summary()


def test_plan_repeat_count_multiplies_workload():
    plan = plan_experiment(
        benchmark_source=BENCHMARK_PATH,
        experiment_id="test_repeats",
        judges=[DEFAULT_JUDGE_A],
        variants=[ExperimentVariant.SINGLE_DIRECTION],  # 1 * 30 = 30
        repeat_count=3,
        is_dry_run=True,
    )
    assert plan.total_requests == 90


# ============================================================================
# 5. Position Bias & Bidirectional Normalization Tests
# ============================================================================

def test_experiment_bidirectional_consistent_decision(tmp_path):
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    mini_case = bench_data["cases"][0]
    mini_bench = tmp_path / "mini_bench.json"
    with open(mini_bench, "w", encoding="utf-8") as f:
        json.dump({"cases": [mini_case], "benchmark_id": "test-bench"}, f)

    # Candidate wins in both forward (B) and reverse (A)
    fwd_resp = json.dumps({"winner": "B", "rationale": "Candidate B is better."})
    rev_resp = json.dumps({"winner": "A", "rationale": "Candidate A is better."})

    call_count = 0
    def mock_generate(req, config):
        nonlocal call_count
        call_count += 1
        return JudgeResponse(content=fwd_resp if call_count % 2 == 1 else rev_resp)

    class CustomMockProvider:
        def generate(self, req, cfg):
            return mock_generate(req, cfg)

    plan = ExperimentPlan(
        experiment_id="test_bidi_consistent",
        benchmark_id="test-bench",
        benchmark_sha256=compute_benchmark_hash(mini_bench),
        ground_truth_id="gt-test",
        ground_truth_sha256="gt-sha",
        judges=[DEFAULT_JUDGE_A],
        variants=[ExperimentVariant.BIDIRECTIONAL],
        case_count=1,
        repeat_count=1,
        total_requests=2,
        estimated_tokens=3800,
        is_dry_run=False,
    )

    status, info = execute_experiment(
        plan=plan,
        output_dir=tmp_path / "experiments",
        benchmark_source=mini_bench,
        ground_truth_source=GROUND_TRUTH_PATH,
        provider_factory=lambda cfg: CustomMockProvider(),
        allow_unfrozen=True,
        execute_live=True,
    )

    assert status == ExecutionStatus.COMPLETED
    norm_file = tmp_path / "experiments" / "test_bidi_consistent" / "normalized" / "normalized_decisions.json"
    with open(norm_file, "r", encoding="utf-8") as f:
        decisions = json.load(f)

    assert len(decisions) == 1
    assert decisions[0]["is_unstable"] is False
    assert decisions[0]["normalized_winner"] == "CANDIDATE"
    assert decisions[0]["outcome"] == "CONSISTENT_CANDIDATE_WIN"


def test_experiment_bidirectional_detects_unstable_flip(tmp_path):
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    mini_case = bench_data["cases"][0]
    mini_bench = tmp_path / "mini_bench.json"
    with open(mini_bench, "w", encoding="utf-8") as f:
        json.dump({"cases": [mini_case], "benchmark_id": "test-bench"}, f)

    # Absolute position bias: always picks Position A
    # Forward: A is Baseline -> normalized BASELINE
    # Reverse: A is Candidate -> normalized CANDIDATE
    # Normalized results disagree -> POSITION_UNSTABLE
    pos_a_resp = json.dumps({"winner": "A", "rationale": "Position A favored."})

    class PositionBiasProvider:
        def generate(self, req, cfg):
            return JudgeResponse(content=pos_a_resp)

    plan = ExperimentPlan(
        experiment_id="test_bidi_unstable",
        benchmark_id="test-bench",
        benchmark_sha256=compute_benchmark_hash(mini_bench),
        ground_truth_id="gt-test",
        ground_truth_sha256="gt-sha",
        judges=[DEFAULT_JUDGE_A],
        variants=[ExperimentVariant.BIDIRECTIONAL],
        case_count=1,
        repeat_count=1,
        total_requests=2,
        estimated_tokens=3800,
        is_dry_run=False,
    )

    status, info = execute_experiment(
        plan=plan,
        output_dir=tmp_path / "experiments",
        benchmark_source=mini_bench,
        ground_truth_source=GROUND_TRUTH_PATH,
        provider_factory=lambda cfg: PositionBiasProvider(),
        allow_unfrozen=True,
        execute_live=True,
    )

    norm_file = tmp_path / "experiments" / "test_bidi_unstable" / "normalized" / "normalized_decisions.json"
    with open(norm_file, "r", encoding="utf-8") as f:
        decisions = json.load(f)

    assert decisions[0]["is_unstable"] is True
    assert decisions[0]["outcome"] == "POSITION_UNSTABLE"


# ============================================================================
# 6. Result Integrity & Summarization Tests
# ============================================================================

def test_summarize_experiment_against_fixture_ground_truth(tmp_path):
    exp_dir = tmp_path / "experiments" / "summary_test"
    norm_dir = exp_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    decisions = [
        {
            "case_id": "dsa-001",
            "judge_id": "judge_a_claude",
            "variant": "C",
            "repeat_index": 0,
            "normalized_winner": "CANDIDATE",
            "is_unstable": False,
        },
        {
            "case_id": "dsa-002",
            "judge_id": "judge_a_claude",
            "variant": "C",
            "repeat_index": 0,
            "normalized_winner": "BASELINE",
            "is_unstable": False,
        },
        {
            "case_id": "dsa-003",
            "judge_id": "judge_a_claude",
            "variant": "C",
            "repeat_index": 0,
            "normalized_winner": "TIE",
            "is_unstable": True,
        },
    ]
    with open(norm_dir / "normalized_decisions.json", "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)

    # Fixture ground truth
    gt_file = tmp_path / "gt.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": "GROUND_TRUTH_FROZEN",
            "cases": [
                {"case_id": "dsa-001", "consensus": {"winner": "CANDIDATE"}},
                {"case_id": "dsa-002", "consensus": {"winner": "CANDIDATE"}},  # Judge chose BASELINE -> False regression
                {"case_id": "dsa-003", "consensus": {"winner": "TIE"}},
            ],
        }, f)

    summary = summarize_experiment(
        experiment_id="summary_test",
        output_dir=tmp_path / "experiments",
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=gt_file,
    )

    cfg_stat = summary["configurations"]["judge_a_claude_variant_C"]
    assert cfg_stat["total_decisions"] == 3
    assert cfg_stat["unstable_count"] == 1
    assert pytest.approx(cfg_stat["position_instability_rate"], 0.01) == 0.333
    assert cfg_stat["false_regression_count"] == 1
    assert cfg_stat["false_pass_count"] == 0
    assert summary["hypotheses_evidence"] is not None


# ============================================================================
# 7. Benchmark Immutability Verification
# ============================================================================

def test_benchmark_hash_strictly_unmodified():
    actual_hash = compute_benchmark_hash(BENCHMARK_PATH)
    assert actual_hash == EXPECTED_SHA, (
        f"CRITICAL: Benchmark hash mismatch! Expected {EXPECTED_SHA}, got {actual_hash}"
    )


def test_prerequisite_pending_human_annotation_semantic_message():
    """Verify that when human annotations are pending, it reports semantic status without corruption errors."""
    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        expected_benchmark_hash=EXPECTED_SHA,
    )
    assert prereq.is_ready is False
    assert prereq.ground_truth_status == "PENDING_HUMAN_ANNOTATION"
    assert "NOT AVAILABLE — human annotation pending" in prereq.summary()
    assert len(prereq.blocking_reasons) == 1
    assert "PENDING_HUMAN_ANNOTATION" in prereq.blocking_reasons[0]
    assert not any("missing 'agreement_statistics' container" in r for r in prereq.blocking_reasons)


def test_execute_live_blocked_with_allow_unfrozen(tmp_path):
    """Verify that live API calls are strictly blocked when --allow-unfrozen fixture mode is active."""
    plan = plan_experiment(
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=GROUND_TRUTH_PATH,
        repeat_count=1,
        is_dry_run=False,
    )
    with pytest.raises(CalibrationPrerequisiteError, match="CRITICAL SAFETY VIOLATION: Live provider execution"):
        execute_experiment(
            plan=plan,
            output_dir=tmp_path / "experiments",
            benchmark_source=BENCHMARK_PATH,
            ground_truth_source=GROUND_TRUTH_PATH,
            allow_unfrozen=True,
            execute_live=True,
            provider_factory=None,  # No mock factory -> live execution attempted
        )


def test_error_hierarchy_and_rate_limit_distinction():
    """Verify error classes and CallStatus distinctions."""
    assert issubclass(JudgeRateLimitError, JudgeProviderError)
    assert issubclass(JudgeTimeoutError, JudgeProviderError)
    assert issubclass(JudgeParseError, Exception)
    assert issubclass(JudgeValidationError, Exception)
    assert CallStatus.RATE_LIMIT == "RATE_LIMIT"
    assert CallStatus.TIMEOUT == "TIMEOUT"
    assert CallStatus.PARSE_ERROR == "PARSE_ERROR"
    assert CallStatus.VALIDATION_ERROR == "VALIDATION_ERROR"
    assert CallStatus.PROVIDER_ERROR == "PROVIDER_ERROR"
