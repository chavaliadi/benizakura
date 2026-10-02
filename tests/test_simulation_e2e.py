from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import compute_benchmark_hash, load_benchmark_data
from benizakura.bias_analysis import analyze_bias_probes
from benizakura.calibration import compute_cohens_kappa
from benizakura.comparison import compare_judge_to_human
from benizakura.experiment import (
    DEFAULT_JUDGE_A,
    DEFAULT_JUDGE_B,
    ExecutionStatus,
    ExperimentPlan,
    ExperimentVariant,
    check_calibration_prerequisites,
    execute_experiment,
    plan_experiment,
    summarize_experiment,
)
from benizakura.hypotheses import evaluate_hypotheses
from benizakura.provider import JudgeRequest, JudgeResponse
from benizakura.report import generate_experiment_report
from benizakura.result_schema import (
    BENCHMARK_CANONICAL_SHA256,
    CaseDecision,
    ExperimentResult,
    ResultStatus,
    validate_experiment_result,
)


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")


class DeterministicMockProvider:
    """Mock judge provider that simulates deterministic pairwise decisions for integration testing."""

    def generate(self, request: JudgeRequest, config) -> JudgeResponse:
        # Generate predictable JSON matching schema
        content = json.dumps({
            "criterion_assessments": [
                {"criterion_name": "CORRECTNESS", "winner": "B", "rationale": "Clear algorithmic proof"},
                {"criterion_name": "CLARITY", "winner": "B", "rationale": "Clean explanation"},
            ],
            "overall_rationale": "Response B is technically superior across all criteria.",
            "winner": "B",
            "confidence": 0.95,
        })
        return JudgeResponse(
            content=content,
            metadata={"prompt_tokens": 1500, "completion_tokens": 300},
        )


def test_offline_end_to_end_simulation(tmp_path):
    """TASK 6 — Offline End-to-End Experiment Simulation.

    Flow:
    Frozen benchmark
          ↓
    Synthetic mock human ground truth (isolated in tmp_path)
          ↓
    Calibration gate
          ↓
    Mock judge
          ↓
    Experiment execution
          ↓
    Result validation
          ↓
    Human-vs-judge comparison
          ↓
    Bias analysis
          ↓
    H1-H5 evaluation
          ↓
    Statistical summary & Report generation
    """
    # 1. Verify frozen benchmark and canonical SHA
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    bench_hash = compute_benchmark_hash(BENCHMARK_PATH)
    assert bench_hash == BENCHMARK_CANONICAL_SHA256, "Benchmark canonical SHA must be strictly preserved"
    assert len(bench_data["cases"]) == 30

    # 2. Build isolated synthetic mock human ground truth in test directory
    synthetic_gt_file = tmp_path / "synthetic_ground_truth_v1.json"
    synthetic_cases = []
    for idx, c in enumerate(bench_data["cases"]):
        cid = c["case_id"]
        probes = c.get("bias_probes", [])
        if "near_tie" in probes:
            w = "TIE"
        elif "hallucination" in probes or "complexity_regression" in probes or "subtle_regression" in probes:
            w = "BASELINE"
        else:
            w = "CANDIDATE"

        synthetic_cases.append({
            "case_id": cid,
            "category": c["category"],
            "consensus": {
                "winner": w,
                "confidence": 1.0,
                "adjudicated": False,
                "criterion_assessments": [
                    {"criterion_name": "CORRECTNESS", "winner": w},
                    {"criterion_name": "CLARITY", "winner": w},
                ],
            },
        })

    synthetic_gt = {
        "ground_truth_id": "synthetic-sim-ground-truth-v1",
        "benchmark_id": "conquer-benchmark-v1",
        "status": "GROUND_TRUTH_FROZEN",
        "agreement_statistics": {
            "cohens_kappa": 0.78,
            "standard_error": 0.08,
            "raw_agreement": 0.85,
        },
        "cases": synthetic_cases,
    }
    with open(synthetic_gt_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_gt, f, indent=2)

    # 3. Calibration gate check
    prereq = check_calibration_prerequisites(
        benchmark_path=BENCHMARK_PATH,
        ground_truth_path=synthetic_gt_file,
        expected_benchmark_hash=BENCHMARK_CANONICAL_SHA256,
        allow_unfrozen=False,
    )
    assert prereq.is_ready is True
    assert prereq.ground_truth_frozen is True
    assert prereq.human_kappa == 0.78
    assert prereq.kappa_sufficient is True

    # 4. Plan experiment
    exp_id = "sim_e2e_01"
    plan = plan_experiment(
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=synthetic_gt_file,
        experiment_id=exp_id,
        judges=[DEFAULT_JUDGE_A],
        variants=[ExperimentVariant.SINGLE_DIRECTION, ExperimentVariant.BIDIRECTIONAL],
        repeat_count=1,
        is_dry_run=False,
    )

    # 5. Execute experiment with mock provider
    status, info = execute_experiment(
        plan=plan,
        output_dir=tmp_path / "experiments",
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=synthetic_gt_file,
        provider_factory=lambda cfg: DeterministicMockProvider(),
        allow_unfrozen=False,
        execute_live=True,
    )
    assert status == ExecutionStatus.COMPLETED
    assert info["runs_recorded"] > 0

    # 6. Statistical summary & Result validation
    summary = summarize_experiment(
        experiment_id=exp_id,
        output_dir=tmp_path / "experiments",
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=synthetic_gt_file,
    )
    assert "configurations" in summary
    assert "hypotheses_evidence" in summary

    # Verify result.json artifact
    result_path = tmp_path / "experiments" / exp_id / "result.json"
    assert result_path.is_file()
    with open(result_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    v_errs = validate_experiment_result(res_data)
    assert len(v_errs) == 0, f"Result schema validation errors: {v_errs}"
    exp_result = ExperimentResult.from_dict(res_data)
    assert exp_result.result_status in [ResultStatus.VALID_RESULT, ResultStatus.INCONCLUSIVE_RESULT]

    # 7. Human vs Judge comparison layer
    comp_report = compare_judge_to_human(
        judge_decisions=exp_result.case_decisions,
        ground_truth_source=synthetic_gt_file,
        benchmark_source=BENCHMARK_PATH,
    )
    assert comp_report.total_benchmark_cases == 30
    assert comp_report.total_evaluated_cases == 30
    assert comp_report.failed_execution_cases == 0
    assert 0.0 <= comp_report.overall_agreement <= 1.0
    assert -1.0 <= comp_report.cohens_kappa <= 1.0

    # 8. Bias probe analysis
    bias_report = analyze_bias_probes(
        case_decisions=exp_result.case_decisions,
        benchmark_source=BENCHMARK_PATH,
        ground_truth_source=synthetic_gt_file,
    )
    assert bias_report.position_sensitivity.total_bidirectional_cases == 30
    assert bias_report.verbosity_sensitivity.total_probe_cases == 5
    assert bias_report.hallucination_detection.total_probe_cases == 4
    assert bias_report.near_tie_behavior.total_probe_cases == 7
    assert bias_report.alternative_architecture.total_probe_cases == 4
    assert bias_report.criterion_tradeoff.total_probe_cases == 4

    # 9. Hypotheses evaluation
    hyp_eval = evaluate_hypotheses(
        experiment_results=[exp_result],
        ground_truth=synthetic_gt,
    )
    assert "H1" in hyp_eval
    assert "H2" in hyp_eval
    assert "H3" in hyp_eval
    assert "H4" in hyp_eval
    assert "H5" in hyp_eval

    # 10. Final research report generation
    report_text = generate_experiment_report(
        experiment_result=exp_result,
        comparison_report=comp_report,
        bias_report=bias_report,
        hypotheses_eval=hyp_eval,
    )
    assert "## 1. Experiment Configuration" in report_text
    assert "## 2. Benchmark Identity" in report_text
    assert "## 3. Ground-Truth Calibration Status" in report_text
    assert "## 4. Human Agreement" in report_text
    assert "## 5. Judge Agreement" in report_text
    assert "## 6. Aggregate Results" in report_text
    assert "## 7. Statistical Uncertainty" in report_text
    assert "## 8. Position / Order Analysis" in report_text
    assert "## 9. Bias-Probe Analysis" in report_text
    assert "## 10. Hypotheses Status (H1–H5)" in report_text
    assert "## 11. Failure & Incomplete Cases" in report_text
    assert "## 12. Reproducibility Metadata" in report_text
    assert "## 13. Final Interpretation" in report_text

    # Write report artifact to experiment directory
    with open(tmp_path / "experiments" / exp_id / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    assert (tmp_path / "experiments" / exp_id / "REPORT.md").is_file()
