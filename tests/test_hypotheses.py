from __future__ import annotations

import pytest

from benizakura.hypotheses import HypothesisStatus, evaluate_hypotheses
from benizakura.result_schema import (
    AggregateStatistics,
    CaseDecision,
    ExperimentResult,
    ResultStatus,
)


def _create_mock_result(
    variant: str,
    error_rate: float = 0.0,
    kappa: float = 0.5,
    provider: str = "anthropic",
    snapshot: str = "claude-3-5-sonnet-20241022",
    false_pass_rate: float = 0.0,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=f"exp_{variant}",
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256="aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        ground_truth_status="GROUND_TRUTH_FROZEN",
        ground_truth_sha256="gt_sha",
        judge_provider=provider,
        model_snapshot=snapshot,
        model_family="anthropic" if provider == "anthropic" else "openai",
        temperature=0.0,
        prompt_version="pairwise_v1",
        rubric_version="standard_rubric_v1",
        experiment_variant=variant,
        result_status=ResultStatus.VALID_RESULT,
        case_decisions=[
            CaseDecision(case_id=f"case-{i}", execution_status="SUCCESS", normalized_decision="CANDIDATE")
            for i in range(30)
        ],
        aggregate_statistics=AggregateStatistics(
            total_cases=30,
            valid_decision_count=30,
            failed_decision_count=0,
            unstable_count=int(error_rate * 30),
            position_dependent_error_rate=error_rate,
            candidate_win_rate=1.0,
            baseline_win_rate=0.0,
            tie_rate=0.0,
            kappa=kappa,
            false_pass_rate=false_pass_rate,
        ),
    )


def test_hypotheses_not_evaluable_when_data_missing():
    """All hypotheses must report NOT_EVALUABLE if experiment data is absent."""
    h_eval = evaluate_hypotheses()
    for hid in ["H1", "H2", "H3", "H4", "H5"]:
        assert h_eval[hid].status == HypothesisStatus.NOT_EVALUABLE
        assert "proved" not in h_eval[hid].verdict_rationale.lower()


def test_hypothesis_h1_evaluation():
    # Variant B (single-direction) has 30% error; Variant C (bidirectional) has 10% error
    res_b = _create_mock_result("B", error_rate=0.30)
    res_c = _create_mock_result("C", error_rate=0.10)

    h_eval = evaluate_hypotheses(experiment_results=[res_b, res_c])
    h1 = h_eval["H1"]
    assert h1.status == HypothesisStatus.SUPPORTED
    assert "proved" not in h1.verdict_rationale.lower()
    assert h1.observed_metrics["variant_b_position_error_rate"] == 0.30
    assert h1.observed_metrics["variant_c_position_error_rate"] == 0.10


def test_hypothesis_h2_rubric_threshold():
    # H2 requires delta kappa >= 0.15 against frozen ground truth
    res_c = _create_mock_result("C", kappa=0.50)
    res_d = _create_mock_result("D", kappa=0.68)  # Delta kappa = +0.18 >= 0.15

    h_eval = evaluate_hypotheses(
        experiment_results=[res_c, res_d],
        ground_truth={"status": "GROUND_TRUTH_FROZEN", "cases": []},
    )
    h2 = h_eval["H2"]
    assert h2.status == HypothesisStatus.SUPPORTED
    assert pytest.approx(h2.observed_metrics["delta_kappa"], 0.01) == 0.18

    # Test when delta kappa is below threshold (+0.05 < 0.15)
    res_d_low = _create_mock_result("D", kappa=0.55)
    h_eval_low = evaluate_hypotheses(
        experiment_results=[res_c, res_d_low],
        ground_truth={"status": "GROUND_TRUTH_FROZEN", "cases": []},
    )
    assert h_eval_low["H2"].status == HypothesisStatus.NOT_SUPPORTED


def test_hypothesis_h3_rationale_first_threshold():
    # H3 requires relative reduction in instability >= 25%
    wf_res = _create_mock_result("C", error_rate=0.20)
    rf_res = _create_mock_result("C", error_rate=0.12)  # Relative reduction = (0.20 - 0.12) / 0.20 = 40% >= 25%

    h_eval = evaluate_hypotheses(
        winner_first_result=wf_res,
        rationale_first_result=rf_res,
    )
    h3 = h_eval["H3"]
    assert h3.status == HypothesisStatus.SUPPORTED
    assert pytest.approx(h3.observed_metrics["relative_reduction"], 0.01) == 0.40

    # Below 25% threshold
    rf_res_low = _create_mock_result("C", error_rate=0.18)  # Relative reduction = 10%
    h_eval_low = evaluate_hypotheses(
        winner_first_result=wf_res,
        rationale_first_result=rf_res_low,
    )
    assert h_eval_low["H3"].status == HypothesisStatus.NOT_SUPPORTED


def test_hypothesis_h4_evaluator_independence():
    claude = _create_mock_result("E", provider="anthropic", snapshot="claude-3-5-sonnet-20241022", false_pass_rate=0.0)
    gpt4o = _create_mock_result("E", provider="openai", snapshot="gpt-4o-2024-08-06", false_pass_rate=0.25)

    h_eval = evaluate_hypotheses(
        claude_result=claude,
        gpt4o_result=gpt4o,
    )
    h4 = h_eval["H4"]
    assert h4.status == HypothesisStatus.SUPPORTED
    # Ensure methodological limitation regarding small N=4 probe size is reported
    assert any("N=4" in lim for lim in h4.methodological_limitations)
    assert len(h4.gaps_reported) > 0


def test_hypothesis_h5_bootstrap_gating():
    # H5 requires relative false alarm reduction >= 50%
    naive = {"false_regression_alarms": 10}
    boot = {"false_regression_alarms": 3}  # 70% reduction >= 50%

    h_eval = evaluate_hypotheses(
        naive_gating_stats=naive,
        bootstrap_gating_stats=boot,
    )
    h5 = h_eval["H5"]
    assert h5.status == HypothesisStatus.SUPPORTED
    assert pytest.approx(h5.observed_metrics["relative_reduction"], 0.01) == 0.70
