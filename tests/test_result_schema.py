from __future__ import annotations

import pytest

from benizakura.result_schema import (
    AggregateStatistics,
    CaseDecision,
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentResult,
    InstabilityStatus,
    ResultStatus,
    validate_experiment_result,
)


def test_experiment_result_valid_instantiation():
    case = CaseDecision(
        case_id="dsa-001",
        pass_1_decision="A",
        pass_2_decision="A",
        normalized_decision="BASELINE",
        is_unstable=False,
        instability_status=InstabilityStatus.STABLE,
        execution_status="SUCCESS",
    )
    agg = AggregateStatistics(
        total_cases=1,
        valid_decision_count=1,
        failed_decision_count=0,
        unstable_count=0,
        position_dependent_error_rate=0.0,
        candidate_win_rate=0.0,
        baseline_win_rate=1.0,
        tie_rate=0.0,
    )
    result = ExperimentResult(
        experiment_id="test_exp_01",
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256="aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        ground_truth_status="GROUND_TRUTH_FROZEN",
        ground_truth_sha256="fake_gt_sha",
        judge_provider="anthropic",
        model_snapshot="claude-3-5-sonnet-20241022",
        model_family="anthropic",
        temperature=0.0,
        prompt_version="pairwise_rubric_v1",
        rubric_version="standard_rubric_v1",
        experiment_variant="E",
        result_status=ResultStatus.VALID_RESULT,
        case_decisions=[case],
        aggregate_statistics=agg,
    )

    errors = validate_experiment_result(result)
    assert len(errors) == 0

    d = result.to_dict()
    assert d["experiment_schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert d["result_status"] == "VALID_RESULT"
    assert len(d["case_decisions"]) == 1

    # Roundtrip from dict
    restored = ExperimentResult.from_dict(d)
    assert restored.experiment_id == "test_exp_01"
    assert restored.model_snapshot == "claude-3-5-sonnet-20241022"


def test_failed_api_call_never_represented_as_valid_tie():
    """A failed API call must never be given a normalized_decision or tie."""
    # Instantiation should reject normalized_decision on failure
    with pytest.raises(ValueError, match="Failed calls must never be assigned a winner or tie"):
        CaseDecision(
            case_id="dsa-002",
            execution_status="TIMEOUT",
            normalized_decision="TIE",
        )

    # Validating dictionary with failed execution and assigned normalized_winner
    invalid_dict = {
        "experiment_id": "test_exp_fail",
        "experiment_schema_version": "1.0.0",
        "benchmark_id": "conquer-benchmark-v1",
        "benchmark_version": "1.0.0",
        "benchmark_sha256": "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        "ground_truth_status": "GROUND_TRUTH_FROZEN",
        "ground_truth_sha256": "fake_gt_sha",
        "judge_provider": "anthropic",
        "model_snapshot": "claude-3-5-sonnet-20241022",
        "model_family": "anthropic",
        "temperature": 0.0,
        "prompt_version": "pairwise_rubric_v1",
        "rubric_version": "standard_rubric_v1",
        "experiment_variant": "E",
        "result_status": "EXECUTION_FAILURE",
        "case_decisions": [
            {
                "case_id": "dsa-001",
                "execution_status": "PROVIDER_ERROR",
                "normalized_decision": "TIE",  # ILLEGAL
            }
        ],
        "aggregate_statistics": {
            "total_cases": 1,
            "valid_decision_count": 0,
            "failed_decision_count": 1,
            "unstable_count": 0,
            "position_dependent_error_rate": 0.0,
            "candidate_win_rate": 0.0,
            "baseline_win_rate": 0.0,
            "tie_rate": 0.0,
        },
        "started_at": "2026-10-02T12:00:00Z",
        "software_version": "benizakura-0.1.0",
    }

    errors = validate_experiment_result(invalid_dict)
    assert any("Failed decisions must NEVER be assigned a winner or tie" in e for e in errors)


def test_denominator_preservation_validation():
    """Valid + failed must equal total_cases."""
    invalid_stats = {
        "experiment_id": "test_exp_denom",
        "experiment_schema_version": "1.0.0",
        "benchmark_id": "conquer-benchmark-v1",
        "benchmark_version": "1.0.0",
        "benchmark_sha256": "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        "ground_truth_status": "GROUND_TRUTH_FROZEN",
        "ground_truth_sha256": "fake_gt_sha",
        "judge_provider": "anthropic",
        "model_snapshot": "claude-3-5-sonnet-20241022",
        "model_family": "anthropic",
        "temperature": 0.0,
        "prompt_version": "pairwise_rubric_v1",
        "rubric_version": "standard_rubric_v1",
        "experiment_variant": "E",
        "result_status": "VALID_RESULT",
        "case_decisions": [
            {
                "case_id": "dsa-001",
                "execution_status": "SUCCESS",
                "normalized_decision": "CANDIDATE",
            },
            {
                "case_id": "dsa-002",
                "execution_status": "SUCCESS",
                "normalized_decision": "BASELINE",
            },
        ],
        "aggregate_statistics": {
            "total_cases": 2,
            "valid_decision_count": 1,  # Inconsistent: 2 valid decisions exist
            "failed_decision_count": 0,
            "unstable_count": 0,
            "position_dependent_error_rate": 0.0,
            "candidate_win_rate": 0.5,
            "baseline_win_rate": 0.5,
            "tie_rate": 0.0,
        },
        "started_at": "2026-10-02T12:00:00Z",
        "software_version": "benizakura-0.1.0",
    }
    errors = validate_experiment_result(invalid_stats)
    assert any("does not match observed valid cases" in e for e in errors)
    assert any("Denominator inconsistency" in e for e in errors)


def test_benchmark_sha_mismatch_fails_validation():
    invalid_sha_dict = {
        "experiment_id": "test_sha_mismatch",
        "experiment_schema_version": "1.0.0",
        "benchmark_id": "conquer-benchmark-v1",
        "benchmark_version": "1.0.0",
        "benchmark_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "ground_truth_status": "GROUND_TRUTH_FROZEN",
        "ground_truth_sha256": "fake_gt_sha",
        "judge_provider": "anthropic",
        "model_snapshot": "claude-3-5-sonnet-20241022",
        "model_family": "anthropic",
        "temperature": 0.0,
        "prompt_version": "pairwise_rubric_v1",
        "rubric_version": "standard_rubric_v1",
        "experiment_variant": "E",
        "result_status": "VALID_RESULT",
        "case_decisions": [],
        "aggregate_statistics": {
            "total_cases": 0,
            "valid_decision_count": 0,
            "failed_decision_count": 0,
            "unstable_count": 0,
            "position_dependent_error_rate": 0.0,
            "candidate_win_rate": 0.0,
            "baseline_win_rate": 0.0,
            "tie_rate": 0.0,
        },
        "started_at": "2026-10-02T12:00:00Z",
        "software_version": "benizakura-0.1.0",
    }
    errors = validate_experiment_result(invalid_sha_dict)
    assert any("Canonical benchmark SHA mismatch" in e for e in errors)
