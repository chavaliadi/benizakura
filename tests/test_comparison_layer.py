from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import load_benchmark_data
from benizakura.result_schema import BENCHMARK_CANONICAL_SHA256
from benizakura.comparison import compare_judge_to_human
from benizakura.result_schema import CaseDecision


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")


def _generate_synthetic_ground_truth(benchmark_cases: list[dict], candidate_ratio: float = 0.5) -> dict:
    """Generate isolated synthetic ground truth fixture for testing the comparison layer."""
    cases = []
    for idx, c in enumerate(benchmark_cases):
        cid = c["case_id"]
        probes = c.get("bias_probes", [])
        if "near_tie" in probes:
            winner = "TIE"
        elif "hallucination" in probes or "complexity_regression" in probes or "subtle_regression" in probes:
            winner = "BASELINE"
        elif "clearly_better" in probes or idx % 2 == 0:
            winner = "CANDIDATE"
        else:
            winner = "BASELINE"

        cases.append({
            "case_id": cid,
            "category": c["category"],
            "consensus": {
                "winner": winner,
                "confidence": 1.0,
                "adjudicated": False,
                "criterion_assessments": [
                    {"criterion_name": "CORRECTNESS", "winner": winner},
                    {"criterion_name": "CLARITY", "winner": winner},
                ],
            },
        })

    return {
        "ground_truth_id": "synthetic-test-gt-v1",
        "benchmark_id": "conquer-benchmark-v1",
        "status": "GROUND_TRUTH_FROZEN",
        "cases": cases,
    }


def test_comparison_layer_perfect_agreement():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]
    synthetic_gt = _generate_synthetic_ground_truth(cases)

    # Build judge decisions matching synthetic ground truth exactly
    judge_decisions = []
    for c in synthetic_gt["cases"]:
        winner = c["consensus"]["winner"]
        judge_decisions.append(
            CaseDecision(
                case_id=c["case_id"],
                execution_status="SUCCESS",
                normalized_decision=winner,
                criterion_assessments={"CORRECTNESS": winner, "CLARITY": winner},
            )
        )

    report = compare_judge_to_human(
        judge_decisions=judge_decisions,
        ground_truth_source=synthetic_gt,
        benchmark_source=BENCHMARK_PATH,
    )

    assert report.total_benchmark_cases == 30
    assert report.total_evaluated_cases == 30
    assert report.failed_execution_cases == 0
    assert pytest.approx(report.overall_agreement, 0.01) == 1.0
    assert pytest.approx(report.cohens_kappa, 0.01) == 1.0
    assert report.candidate_win_agreement == 1.0
    assert report.baseline_win_agreement == 1.0
    assert report.tie_agreement == 1.0
    assert report.false_pass_count == 0
    assert report.false_pass_rate == 0.0
    assert report.false_regression_count == 0
    assert report.false_regression_rate == 0.0
    assert report.directional_agreement == 1.0

    # Verify category breakdown contains all 5 tracks
    for cat in ["DSA", "SYSTEM_DESIGN", "BACKEND", "FRONTEND", "BEHAVIORAL"]:
        assert cat in report.category_breakdown
        assert report.category_breakdown[cat].raw_agreement == 1.0

    # Verify probe breakdown
    assert "position" in report.probe_breakdown
    assert "verbosity" in report.probe_breakdown
    assert "hallucination" in report.probe_breakdown
    assert "near_tie" in report.probe_breakdown

    # Verify criterion breakdown
    assert "CORRECTNESS" in report.criterion_breakdown
    assert report.criterion_breakdown["CORRECTNESS"].raw_agreement == 1.0


def test_comparison_layer_false_passes_and_regressions():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]
    synthetic_gt = _generate_synthetic_ground_truth(cases)

    # Invert decisions to create deliberate false passes and false regressions
    judge_decisions = []
    for c in synthetic_gt["cases"]:
        gt_w = c["consensus"]["winner"]
        if gt_w == "BASELINE":
            j_w = "CANDIDATE"  # False pass
        elif gt_w == "CANDIDATE":
            j_w = "BASELINE"   # False regression
        else:
            j_w = "TIE"

        judge_decisions.append(
            CaseDecision(
                case_id=c["case_id"],
                execution_status="SUCCESS",
                normalized_decision=j_w,
            )
        )

    report = compare_judge_to_human(
        judge_decisions=judge_decisions,
        ground_truth_source=synthetic_gt,
        benchmark_source=BENCHMARK_PATH,
    )

    assert report.false_pass_count > 0
    assert report.false_pass_rate == 1.0  # 100% of BASELINE cases were falsely passed as CANDIDATE
    assert report.false_regression_count > 0
    assert report.false_regression_rate == 1.0  # 100% of CANDIDATE cases were falsely degraded to BASELINE
    assert report.cohens_kappa < 0.0  # Inverted predictions yield negative kappa


def test_comparison_layer_failed_cases_preserved_in_denominator():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]
    synthetic_gt = _generate_synthetic_ground_truth(cases)

    # 10 cases succeed with agreement, 20 cases fail due to provider timeout
    judge_decisions = []
    for idx, c in enumerate(synthetic_gt["cases"]):
        if idx < 10:
            judge_decisions.append(
                CaseDecision(
                    case_id=c["case_id"],
                    execution_status="SUCCESS",
                    normalized_decision=c["consensus"]["winner"],
                )
            )
        else:
            judge_decisions.append(
                CaseDecision(
                    case_id=c["case_id"],
                    execution_status="TIMEOUT",
                    failure_status="Gateway Timeout 504",
                    normalized_decision=None,
                )
            )

    report = compare_judge_to_human(
        judge_decisions=judge_decisions,
        ground_truth_source=synthetic_gt,
        benchmark_source=BENCHMARK_PATH,
    )

    assert report.total_benchmark_cases == 30
    assert report.total_evaluated_cases == 10
    assert report.failed_execution_cases == 20
    assert report.overall_agreement == 1.0  # 10 out of 10 evaluated agreed


def test_comparison_layer_benchmark_hash_mismatch_blocks():
    # Pass mismatched benchmark SHA
    with pytest.raises(ValueError, match="Benchmark SHA mismatch in comparison layer"):
        compare_judge_to_human(
            judge_decisions=[],
            ground_truth_source={"status": "GROUND_TRUTH_FROZEN", "cases": []},
            benchmark_source={"benchmark_id": "conquer-benchmark-v1", "cases": []},
            verify_benchmark_hash=True,
        )
