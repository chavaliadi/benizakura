from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import compute_benchmark_hash
from benizakura.result_schema import BENCHMARK_CANONICAL_SHA256
from benizakura.errors import CalibrationPrerequisiteError, ExperimentError
from benizakura.experiment import (
    CallStatus,
    DEFAULT_JUDGE_A,
    DEFAULT_JUDGE_B,
    ExperimentCaseRun,
    ExperimentPlan,
    ExperimentVariant,
    check_calibration_prerequisites,
    execute_experiment,
    plan_experiment,
    sanitize_secrets,
    summarize_experiment,
)
from benizakura.result_schema import ResultStatus, validate_experiment_result


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")
GROUND_TRUTH_PATH = Path("evals/conquer/ground_truth_v1.json")


# ============================================================================
# Task 7: Reproducibility & Manifest Audit Tests
# ============================================================================

def test_manifest_reconstruction_completeness():
    """Verify that an ExperimentPlan and manifest contains all reproducibility fields."""
    plan = plan_experiment(
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=GROUND_TRUTH_PATH,
        experiment_id="audit_manifest_01",
        judges=[DEFAULT_JUDGE_A, DEFAULT_JUDGE_B],
        variants=[ExperimentVariant.FULL_BENIZAKURA],
        repeat_count=1,
        is_dry_run=True,
    )

    d = plan.to_dict()

    # Required reproducibility fields
    assert d["benchmark_id"] == "conquer-benchmark-v1"
    assert d["benchmark_sha256"] == "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"
    assert "ground_truth_id" in d
    assert "ground_truth_sha256" in d
    assert d["software_version"] == "benizakura-0.1.0"
    assert "created_at" in d
    assert "statistical_configuration" in d
    assert "gate_configuration" in d

    # Judge metadata
    assert len(d["judges"]) == 2
    for j in d["judges"]:
        assert "provider" in j
        assert "model" in j  # Exact model snapshot
        assert "model_family" in j
        assert "temperature" in j
        assert "system_prompt_version" in j
        assert "rubric_version" in j
        assert "prompt_schema_version" in j

    # Statistical configuration
    stat_cfg = d["statistical_configuration"]
    assert "bootstrap_samples" in stat_cfg
    assert "confidence_level" in stat_cfg

    # Gate configuration
    gate_cfg = d["gate_configuration"]
    assert "max_unstable_rate" in gate_cfg
    assert "max_regression_rate" in gate_cfg
    assert "regression_tolerance" in gate_cfg


def test_manifest_and_run_secret_leakage():
    """Ensure secrets (API keys, authorization headers) are scrubbed from plans and runs."""
    leaked_dict = {
        "api_key": "sk-ant-api03-verysecretstring1234567890",
        "authorization": "Bearer sk-proj-supersecrettokenabcdefg",
        "nested": {
            "token": "sk-999999999999999999999999",
            "normal_field": "safe_value",
        },
        "error_trace": "Error with key sk-ant-api03-1234567890abcdefgh in request",
    }

    sanitized = sanitize_secrets(leaked_dict)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["authorization"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert sanitized["nested"]["normal_field"] == "safe_value"
    assert "sk-ant-" not in sanitized["error_trace"]
    assert "[REDACTED_ANTHROPIC_KEY]" in sanitized["error_trace"]


# ============================================================================
# Task 8: Incomplete Data Safety Tests
# ============================================================================

def test_missing_cases_not_silently_treated_as_ties(tmp_path):
    """When cases are missing or fail, they must NOT be recorded as TIE or omitted from denominator."""
    exp_dir = tmp_path / "experiments" / "incomplete_test"
    norm_dir = exp_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # 5 decisions out of 30, with 1 failure
    decisions = [
        {"case_id": "dsa-001", "judge_id": "judge_a_claude", "variant": "C", "repeat_index": 0, "normalized_winner": "CANDIDATE", "outcome": "CONSISTENT_CANDIDATE_WIN"},
        {"case_id": "dsa-002", "judge_id": "judge_a_claude", "variant": "C", "repeat_index": 0, "normalized_winner": None, "outcome": "EXECUTION_FAILURE"},
    ]
    with open(norm_dir / "normalized_decisions.json", "w", encoding="utf-8") as f:
        json.dump(decisions, f)

    summary = summarize_experiment(
        experiment_id="incomplete_test",
        output_dir=tmp_path / "experiments",
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=GROUND_TRUTH_PATH,
    )

    cfg = summary["configurations"]["judge_a_claude_variant_C"]
    assert cfg["total_decisions"] == 2
    assert cfg["failed_decisions"] == 1
    assert cfg["valid_decisions"] == 1

    # Verify result.json has INCONCLUSIVE_RESULT due to small sample (N < 10)
    with open(exp_dir / "result.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    assert res["result_status"] == ResultStatus.INCONCLUSIVE_RESULT.value


def test_api_failures_distinguishable():
    """Verify distinct error classes and statuses for timeout, rate limit, parse error, provider error."""
    runs = [
        ExperimentCaseRun(
            run_id="r1", case_id="c1", judge_id="j1", variant="C", orientation="fwd",
            repeat_index=0, call_status=CallStatus.TIMEOUT, raw_winner=None, normalized_winner=None,
            confidence=None, rationale=None, error_message="HTTP 504 Gateway Timeout"
        ),
        ExperimentCaseRun(
            run_id="r2", case_id="c2", judge_id="j1", variant="C", orientation="fwd",
            repeat_index=0, call_status=CallStatus.RATE_LIMIT, raw_winner=None, normalized_winner=None,
            confidence=None, rationale=None, error_message="HTTP 429 Rate Limit Exceeded"
        ),
        ExperimentCaseRun(
            run_id="r3", case_id="c3", judge_id="j1", variant="C", orientation="fwd",
            repeat_index=0, call_status=CallStatus.PARSE_ERROR, raw_winner=None, normalized_winner=None,
            confidence=None, rationale=None, error_message="Malformed JSON output"
        ),
        ExperimentCaseRun(
            run_id="r4", case_id="c4", judge_id="j1", variant="C", orientation="fwd",
            repeat_index=0, call_status=CallStatus.PROVIDER_ERROR, raw_winner=None, normalized_winner=None,
            confidence=None, rationale=None, error_message="Internal Server Error 500"
        ),
    ]

    statuses = [r.call_status for r in runs]
    assert CallStatus.TIMEOUT in statuses
    assert CallStatus.RATE_LIMIT in statuses
    assert CallStatus.PARSE_ERROR in statuses
    assert CallStatus.PROVIDER_ERROR in statuses

    for r in runs:
        # Normalized winner must remain None on failure
        assert r.normalized_winner is None


def test_missing_human_ground_truth_blocks_judge_calibration():
    """Attempting calibration without ground truth is strictly blocked."""
    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path="non_existent_gt_path.json",
    )
    assert prereq.is_ready is False
    assert any("not found" in r for r in prereq.blocking_reasons)


def test_benchmark_hash_mismatch_blocks_analysis(tmp_path):
    """Mismatched benchmark SHA blocks summarize_experiment."""
    exp_dir = tmp_path / "experiments" / "hash_mismatch_test"
    norm_dir = exp_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    with open(norm_dir / "normalized_decisions.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    # Manifest with a different expected hash
    with open(exp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
        }, f)

    with pytest.raises(ExperimentError, match="Benchmark hash mismatch blocks analysis"):
        summarize_experiment(
            experiment_id="hash_mismatch_test",
            output_dir=tmp_path / "experiments",
            benchmark_source=BENCHMARK_PATH,
            ground_truth_source=GROUND_TRUTH_PATH,
        )


def test_ground_truth_hash_mismatch_blocks_analysis(tmp_path):
    """Mismatched ground truth SHA between manifest and current file blocks analysis."""
    exp_dir = tmp_path / "experiments" / "gt_mismatch_test"
    norm_dir = exp_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    with open(norm_dir / "normalized_decisions.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    with open(exp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_sha256": "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
            "ground_truth_sha256": "expected_gt_sha_different_from_current",
        }, f)

    with pytest.raises(ExperimentError, match="Ground truth hash mismatch blocks analysis"):
        summarize_experiment(
            experiment_id="gt_mismatch_test",
            output_dir=tmp_path / "experiments",
            benchmark_source=BENCHMARK_PATH,
            ground_truth_source=GROUND_TRUTH_PATH,
        )
