from __future__ import annotations

from pathlib import Path
import pytest

from benizakura.benchmark import load_benchmark_data
from benizakura.bias_analysis import analyze_bias_probes
from benizakura.result_schema import CaseDecision, InstabilityStatus


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")


def test_bias_analysis_position_sensitivity():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    # Construct decisions: 25 stable cases, 5 unstable cases
    decisions = []
    for idx, c in enumerate(cases):
        cid = c["case_id"]
        if idx < 5:
            # Order unstable: pass 1 picks A (Baseline), pass 2 picks A (Candidate) -> Disagreement
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    pass_1_decision="A",
                    pass_2_decision="A",
                    normalized_decision="TIE",
                    is_unstable=True,
                    instability_status=InstabilityStatus.ORDER_UNSTABLE,
                    execution_status="SUCCESS",
                )
            )
        else:
            # Stable: pass 1 picks B (Candidate), pass 2 picks A (Candidate) -> Agreement
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    pass_1_decision="B",
                    pass_2_decision="A",
                    normalized_decision="CANDIDATE",
                    is_unstable=False,
                    instability_status=InstabilityStatus.STABLE,
                    execution_status="SUCCESS",
                )
            )

    report = analyze_bias_probes(
        case_decisions=decisions,
        benchmark_source=BENCHMARK_PATH,
    )

    ps = report.position_sensitivity
    assert ps.total_bidirectional_cases == 30
    assert ps.order_instability_rate == pytest.approx(5 / 30, 0.01)
    assert ps.order_disagreement_count == 5
    assert ps.position_dependent_errors == 5
    assert len(ps.details) == 30


def test_bias_analysis_verbosity_sensitivity():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    # Construct decisions where concise answers always win on verbosity probe cases
    decisions = []
    for c in cases:
        cid = c["case_id"]
        probes = c.get("bias_probes", [])
        if "verbosity" in probes:
            # Probe expects concise answer to win
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision="CANDIDATE",  # Concise target
                )
            )
        else:
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision="TIE",
                )
            )

    report = analyze_bias_probes(
        case_decisions=decisions,
        benchmark_source=BENCHMARK_PATH,
    )

    vs = report.verbosity_sensitivity
    assert vs.total_probe_cases == 5
    assert vs.concise_wins == 5
    assert vs.verbose_wins == 0
    assert vs.concise_win_rate == 1.0
    assert vs.verbosity_favored is False


def test_bias_analysis_hallucination_detection():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    decisions = []
    for c in cases:
        cid = c["case_id"]
        probes = c.get("bias_probes", [])
        if "hallucination" in probes:
            # 3 detected (BASELINE), 1 false pass (CANDIDATE)
            if cid == "dsa-005":
                win = "CANDIDATE"  # False pass
            else:
                win = "BASELINE"   # Detected
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision=win,
                )
            )
        else:
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision="TIE",
                )
            )

    report = analyze_bias_probes(
        case_decisions=decisions,
        benchmark_source=BENCHMARK_PATH,
    )

    hs = report.hallucination_detection
    assert hs.total_probe_cases == 4
    assert hs.hallucination_detected_count == 3
    assert hs.false_pass_count == 1
    assert hs.hallucination_detection_rate == 0.75
    assert hs.false_pass_rate == 0.25


def test_bias_analysis_near_tie_and_alternative_architecture():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    decisions = []
    for c in cases:
        cid = c["case_id"]
        probes = c.get("bias_probes", [])
        if "near_tie" in probes:
            # Judge recognizes all 7 near-ties as TIE
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision="TIE",
                )
            )
        elif "alternative_architecture" in probes:
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision="CANDIDATE",  # Valid alternative accepted
                )
            )
        else:
            decisions.append(
                CaseDecision(
                    case_id=cid,
                    execution_status="SUCCESS",
                    normalized_decision="TIE",
                )
            )

    report = analyze_bias_probes(
        case_decisions=decisions,
        benchmark_source=BENCHMARK_PATH,
    )

    ts = report.near_tie_behavior
    assert ts.total_probe_cases == 7
    assert ts.ties_recognized == 7
    assert ts.tie_recognition_rate == 1.0
    assert ts.artificial_decisiveness_count == 0

    alt = report.alternative_architecture
    assert alt.total_probe_cases == 4
    assert alt.valid_alternatives_accepted == 4
    assert alt.acceptance_rate == 1.0
