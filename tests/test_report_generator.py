from __future__ import annotations

import pytest

from benizakura.bias_analysis import (
    AlternativeArchitectureReport,
    BiasAnalysisReport,
    CriterionTradeoffReport,
    HallucinationDetectionReport,
    NearTieBehaviorReport,
    PositionSensitivityReport,
    VerbositySensitivityReport,
)
from benizakura.comparison import ComparisonReport
from benizakura.hypotheses import HypothesisEvaluation, HypothesisStatus
from benizakura.report import generate_experiment_report
from benizakura.result_schema import (
    AggregateStatistics,
    CaseDecision,
    ExperimentResult,
    ResultStatus,
)


def test_report_generator_unexecuted_experiment():
    manifest = {
        "experiment_id": "dry_run_01",
        "benchmark_id": "conquer-benchmark-v1",
        "benchmark_sha256": "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        "ground_truth_id": "conquer-ground-truth-v1",
        "ground_truth_sha256": "unverified",
        "status": "PENDING_HUMAN_ANNOTATION",
        "is_dry_run": True,
    }

    report = generate_experiment_report(manifest_data=manifest)

    # Required sections
    assert "## 1. Experiment Configuration" in report
    assert "## 2. Benchmark Identity" in report
    assert "## 3. Ground-Truth Calibration Status" in report
    assert "## 4. Human Agreement" in report
    assert "## 5. Judge Agreement" in report
    assert "## 6. Aggregate Results" in report
    assert "## 7. Statistical Uncertainty" in report
    assert "## 8. Position / Order Analysis" in report
    assert "## 9. Bias-Probe Analysis" in report
    assert "## 10. Hypotheses Status (H1–H5)" in report
    assert "## 11. Failure & Incomplete Cases" in report
    assert "## 12. Reproducibility Metadata" in report
    assert "## 13. Final Interpretation" in report

    # Unexecuted declaration
    assert "DRY-RUN / PLANNED (Zero external API calls)" in report
    assert "Human annotations are currently pending" in report
    assert "No live LLM API calls have been executed" in report


def test_report_generator_completed_experiment_with_all_sections():
    exp_res = ExperimentResult(
        experiment_id="completed_test_run",
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256="aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        ground_truth_status="GROUND_TRUTH_FROZEN",
        ground_truth_sha256="gt_sha_123",
        judge_provider="anthropic",
        model_snapshot="claude-3-5-sonnet-20241022",
        model_family="anthropic",
        temperature=0.0,
        prompt_version="pairwise_rubric_v1",
        rubric_version="standard_rubric_v1",
        experiment_variant="E",
        result_status=ResultStatus.VALID_RESULT,
        case_decisions=[
            CaseDecision(case_id="dsa-001", execution_status="SUCCESS", normalized_decision="CANDIDATE"),
            CaseDecision(case_id="dsa-002", execution_status="SUCCESS", normalized_decision="BASELINE"),
        ],
        aggregate_statistics=AggregateStatistics(
            total_cases=2,
            valid_decision_count=2,
            failed_decision_count=0,
            unstable_count=0,
            position_dependent_error_rate=0.0,
            candidate_win_rate=0.5,
            baseline_win_rate=0.5,
            tie_rate=0.0,
            bootstrap_ci=(-0.10, 0.10),
            effect_size=0.15,
        ),
    )

    comp_rep = ComparisonReport(
        benchmark_id="conquer-benchmark-v1",
        benchmark_sha256="aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314",
        ground_truth_id="gt-v1",
        ground_truth_sha256="gt-sha",
        ground_truth_status="GROUND_TRUTH_FROZEN",
        total_benchmark_cases=30,
        total_evaluated_cases=2,
        failed_execution_cases=0,
        overall_agreement=0.90,
        cohens_kappa=0.82,
        kappa_se=0.05,
        kappa_ci=(0.72, 0.92),
        confusion_matrix={"CANDIDATE": {"CANDIDATE": 1, "BASELINE": 0, "TIE": 0}},
        candidate_win_agreement=1.0,
        baseline_win_agreement=1.0,
        tie_agreement=1.0,
        false_pass_count=0,
        false_pass_rate=0.0,
        false_regression_count=0,
        false_regression_rate=0.0,
        directional_agreement=0.95,
        category_breakdown={},
        probe_breakdown={},
    )

    bias_rep = BiasAnalysisReport(
        position_sensitivity=PositionSensitivityReport(
            total_bidirectional_cases=2,
            first_position_wins_pass1=1,
            first_position_wins_pass2=1,
            raw_first_position_win_rate=0.5,
            order_disagreement_count=0,
            order_disagreement_rate=0.0,
            order_instability_rate=0.0,
            position_dependent_errors=0,
        ),
        verbosity_sensitivity=VerbositySensitivityReport(
            total_probe_cases=5,
            concise_wins=5,
            verbose_wins=0,
            ties=0,
            concise_win_rate=1.0,
            verbose_win_rate=0.0,
            verbosity_favored=False,
        ),
        hallucination_detection=HallucinationDetectionReport(
            total_probe_cases=4,
            hallucination_detected_count=4,
            hallucination_detection_rate=1.0,
            false_pass_count=0,
            false_pass_rate=0.0,
        ),
        near_tie_behavior=NearTieBehaviorReport(
            total_probe_cases=7,
            ties_recognized=7,
            tie_recognition_rate=1.0,
            artificial_decisiveness_count=0,
            artificial_candidate_wins=0,
            artificial_baseline_wins=0,
        ),
        alternative_architecture=AlternativeArchitectureReport(
            total_probe_cases=4,
            valid_alternatives_accepted=4,
            acceptance_rate=1.0,
            alternatives_penalized=0,
        ),
        criterion_tradeoff=CriterionTradeoffReport(
            total_probe_cases=4,
            tradeoffs_evaluated=4,
            balanced_verdicts=4,
            one_sided_verdicts=0,
        ),
    )

    hyp_eval = {
        "H1": HypothesisEvaluation(
            hypothesis_id="H1",
            name="Bidirectional Debiasing",
            description="Reduces position error",
            status=HypothesisStatus.SUPPORTED,
            threshold_criteria="Error C < Error B",
            observed_metrics={},
            statistical_result={},
            verdict_rationale="Observed error was lower",
            methodological_limitations=[],
        )
    }

    report = generate_experiment_report(
        experiment_result=exp_res,
        comparison_report=comp_rep,
        bias_report=bias_rep,
        hypotheses_eval=hyp_eval,
    )

    # Verification of strict separation of reporting categories
    assert "### OBSERVED RESULT" in report
    assert "### STATISTICAL RESULT" in report
    assert "### INTERPRETATION" in report
    assert "### HYPOTHESIS STATUS: `H1`" in report

    # Verification of values in report
    assert "Cohen's Kappa (κ)" in report
    assert "0.820" in report
    assert "Paired Bootstrap 95% Confidence Interval" in report
    assert "[-0.1000, 0.1000]" in report
    assert "claude-3-5-sonnet-20241022" in report
    assert "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314" in report
