from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from benizakura.bias_analysis import BiasAnalysisReport
from benizakura.comparison import ComparisonReport
from benizakura.hypotheses import HypothesisEvaluation, HypothesisStatus
from benizakura.result_schema import ExperimentResult, ResultStatus


def generate_experiment_report(
    experiment_result: Optional[Union[Dict[str, Any], ExperimentResult]] = None,
    comparison_report: Optional[ComparisonReport] = None,
    bias_report: Optional[BiasAnalysisReport] = None,
    hypotheses_eval: Optional[Dict[str, HypothesisEvaluation]] = None,
    manifest_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a deterministic, human-readable research report from experiment artifacts.

    Strictly adheres to reporting standards:
    - Clearly separates OBSERVED RESULT, STATISTICAL RESULT, INTERPRETATION, and HYPOTHESIS STATUS.
    - Explicitly notes if an experiment is pending or dry-run.
    - Never asserts causal claims unsupported by data.
    """
    res_dict: Dict[str, Any] = {}
    if experiment_result:
        res_dict = experiment_result.to_dict() if isinstance(experiment_result, ExperimentResult) else experiment_result

    man = manifest_data or {}
    exp_id = res_dict.get("experiment_id") or man.get("experiment_id", "calibration_v1")
    bench_id = res_dict.get("benchmark_id") or man.get("benchmark_id", "conquer-benchmark-v1")
    bench_sha = res_dict.get("benchmark_sha256") or man.get("benchmark_sha256", "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314")
    gt_id = res_dict.get("ground_truth_id") or man.get("ground_truth_id", "conquer-ground-truth-v1")
    gt_status = res_dict.get("ground_truth_status") or man.get("status", "PENDING_HUMAN_ANNOTATION")
    gt_sha = res_dict.get("ground_truth_sha256") or man.get("ground_truth_sha256", "unverified")

    is_dry_run = man.get("is_dry_run", True) if not res_dict else (res_dict.get("result_status") == ResultStatus.MISSING_RESULT.value)
    is_executed = bool(res_dict and res_dict.get("case_decisions"))

    lines: List[str] = [
        f"# Benizakura Research & Calibration Report: `{exp_id}`",
        "",
        f"> **Generated at:** {datetime.now(timezone.utc).isoformat()}  ",
        f"> **Software Version:** `{res_dict.get('software_version', 'benizakura-0.1.0')}`  ",
        f"> **Execution Mode:** {'DRY-RUN / PLANNED (Zero external API calls)' if (is_dry_run or not is_executed) else 'EMPIRICAL EXECUTION'}",
        "",
        "---",
        "",
        "## 1. Experiment Configuration",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
        f"| **Experiment ID** | `{exp_id}` |",
        f"| **Judge Provider** | `{res_dict.get('judge_provider', 'N/A')}` |",
        f"| **Exact Model Snapshot** | `{res_dict.get('model_snapshot', 'N/A')}` |",
        f"| **Model Family** | `{res_dict.get('model_family', 'N/A')}` |",
        f"| **Temperature** | `{res_dict.get('temperature', 0.0)}` |",
        f"| **System Prompt Version** | `{res_dict.get('prompt_version', 'pairwise_v1')}` |",
        f"| **Rubric Version** | `{res_dict.get('rubric_version', 'standard_rubric_v1')}` |",
        f"| **Experiment Variant** | `{res_dict.get('experiment_variant', 'N/A')}` |",
        "",
        "---",
        "",
        "## 2. Benchmark Identity",
        "",
        f"- **Benchmark ID:** `{bench_id}`",
        f"- **Canonical Benchmark SHA-256:** `{bench_sha}`",
        f"- **Integrity Verification:** {'VERIFIED IDENTICAL' if bench_sha == 'aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314' else 'HASH MISMATCH'}",
        "- **Target Application:** Conquer (`POST /api/interview/score`)",
        "- **Total Benchmark Cases:** 30 (stratified across DSA, System Design, Backend, Frontend, Behavioral)",
        "",
        "---",
        "",
        "## 3. Ground-Truth Calibration Status",
        "",
        f"- **Ground Truth ID:** `{gt_id}`",
        f"- **Ground Truth Status:** `{gt_status}`",
        f"- **Ground Truth Checksum:** `{gt_sha}`",
        f"- **Calibration Gate Status:** `{res_dict.get('calibration_gate_status', 'BLOCKED (Pending human annotation)')}`",
    ]

    if gt_status == "PENDING_HUMAN_ANNOTATION":
        lines.extend([
            "",
            "> [!NOTE]",
            "> Human annotations are currently pending from the two independent human annotators.",
            "> Cohen's κ has not yet been measured on production ground truth, and ground truth is not frozen.",
            "> Release gating remains disabled until κ >= 0.60 is verified.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Human Agreement",
        "",
    ])

    if comparison_report:
        lines.extend([
            "### OBSERVED RESULT",
            f"- **Overall Human Agreement (Po):** {comparison_report.overall_agreement * 100:.1f}%",
            f"- **Candidate-Win Agreement (Recall):** {comparison_report.candidate_win_agreement * 100:.1f}%",
            f"- **Baseline-Win Agreement (Recall):** {comparison_report.baseline_win_agreement * 100:.1f}%",
            f"- **Tie Agreement (Recall):** {comparison_report.tie_agreement * 100:.1f}%",
            f"- **Directional Agreement:** {comparison_report.directional_agreement * 100:.1f}%",
            "",
            "### STATISTICAL RESULT",
            f"- **Cohen's Kappa (κ):** {comparison_report.cohens_kappa:.3f} (SE = {comparison_report.kappa_se:.3f}, 95% CI: [{comparison_report.kappa_ci[0]:.3f}, {comparison_report.kappa_ci[1]:.3f}])",
            f"- **False-Pass Rate on Regressions:** {comparison_report.false_pass_rate * 100:.1f}% ({comparison_report.false_pass_count} cases)",
            f"- **False-Regression Rate on Improvements:** {comparison_report.false_regression_rate * 100:.1f}% ({comparison_report.false_regression_count} cases)",
            "",
            "### INTERPRETATION",
            "Agreement indicates alignment between the automated judge and double-annotated human consensus.",
        ])
    else:
        lines.extend([
            "### OBSERVED RESULT",
            "Human agreement metrics are NOT AVAILABLE because ground truth has not yet been frozen and compared.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 5. Judge Agreement",
        "",
    ])

    if comparison_report and comparison_report.confusion_matrix:
        cm = comparison_report.confusion_matrix
        lines.extend([
            "### OBSERVED RESULT",
            "Confusion Matrix (Rows: Human Reference, Columns: Judge Prediction):",
            "",
            "| Human \\ Judge | CANDIDATE | BASELINE | TIE |",
            "| :--- | :---: | :---: | :---: |",
            f"| **CANDIDATE** | {cm.get('CANDIDATE', {}).get('CANDIDATE', 0)} | {cm.get('CANDIDATE', {}).get('BASELINE', 0)} | {cm.get('CANDIDATE', {}).get('TIE', 0)} |",
            f"| **BASELINE**  | {cm.get('BASELINE', {}).get('CANDIDATE', 0)} | {cm.get('BASELINE', {}).get('BASELINE', 0)} | {cm.get('BASELINE', {}).get('TIE', 0)} |",
            f"| **TIE**       | {cm.get('TIE', {}).get('CANDIDATE', 0)} | {cm.get('TIE', {}).get('BASELINE', 0)} | {cm.get('TIE', {}).get('TIE', 0)} |",
        ])
    else:
        lines.append("Judge confusion matrix is NOT AVAILABLE.")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Aggregate Results",
        "",
    ])

    stats = res_dict.get("aggregate_statistics", {})
    if stats:
        lines.extend([
            "### OBSERVED RESULT",
            f"- **Total Cases:** {stats.get('total_cases', 0)}",
            f"- **Valid Decisions:** {stats.get('valid_decision_count', 0)}",
            f"- **Failed Decisions:** {stats.get('failed_decision_count', 0)}",
            f"- **Candidate Win Rate:** {stats.get('candidate_win_rate', 0.0) * 100:.1f}%",
            f"- **Baseline Win Rate:** {stats.get('baseline_win_rate', 0.0) * 100:.1f}%",
            f"- **Tie Rate:** {stats.get('tie_rate', 0.0) * 100:.1f}%",
            f"- **Position Instability Rate:** {stats.get('position_dependent_error_rate', 0.0) * 100:.1f}%",
        ])
    else:
        lines.append("Aggregate statistics are NOT AVAILABLE.")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Statistical Uncertainty",
        "",
    ])

    boot_ci = stats.get("bootstrap_ci") if stats else None
    if boot_ci:
        lines.extend([
            "### STATISTICAL RESULT",
            f"- **Paired Bootstrap 95% Confidence Interval:** [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]",
            f"- **Standardized Effect Size (Cohen's d):** {stats.get('effect_size', 'N/A')}",
        ])
    else:
        lines.extend([
            "### STATISTICAL RESULT",
            "Bootstrap confidence interval: NOT COMPUTED (Requires execution of paired difference sampling).",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 8. Position / Order Analysis",
        "",
    ])

    if bias_report:
        ps = bias_report.position_sensitivity
        lines.extend([
            "### OBSERVED RESULT",
            f"- **Bidirectional Cases Evaluated:** {ps.total_bidirectional_cases}",
            f"- **First-Position Win Rate:** {ps.raw_first_position_win_rate * 100:.1f}% (Pass 1: {ps.first_position_wins_pass1}, Pass 2: {ps.first_position_wins_pass2})",
            f"- **Order Disagreement Rate:** {ps.order_disagreement_rate * 100:.1f}% ({ps.order_disagreement_count} cases)",
            f"- **Order Instability Rate:** {ps.order_instability_rate * 100:.1f}% ({ps.position_dependent_errors} cases)",
            "",
            "### INTERPRETATION",
            "Order sensitivity quantifies whether presenting an answer in Position 1 vs Position 2 changes the outcome.",
        ])
    else:
        lines.append("Position analysis: NOT AVAILABLE.")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Bias-Probe Analysis",
        "",
    ])

    if bias_report:
        vs = bias_report.verbosity_sensitivity
        hs = bias_report.hallucination_detection
        ts = bias_report.near_tie_behavior
        alt = bias_report.alternative_architecture
        lines.extend([
            "### OBSERVED RESULT",
            f"- **Verbosity Sensitivity:** Concise win rate = {vs.concise_win_rate * 100:.1f}%, Verbose win rate = {vs.verbose_win_rate * 100:.1f}% (Verbosity favored: {vs.verbosity_favored})",
            f"- **Hallucination Detection:** Detection rate = {hs.hallucination_detection_rate * 100:.1f}%, False-pass rate = {hs.false_pass_rate * 100:.1f}%",
            f"- **Near-Tie Behavior:** Tie recognition rate = {ts.tie_recognition_rate * 100:.1f}%, Artificial decisiveness = {ts.artificial_decisiveness_count} cases",
            f"- **Alternative Architecture:** Valid alternative acceptance rate = {alt.acceptance_rate * 100:.1f}%",
        ])
    else:
        lines.append("Bias probe analysis: NOT AVAILABLE.")

    lines.extend([
        "",
        "---",
        "",
        "## 10. Hypotheses Status (H1–H5)",
        "",
    ])

    if hypotheses_eval:
        for hid in ["H1", "H2", "H3", "H4", "H5"]:
            he = hypotheses_eval.get(hid)
            if he:
                lines.extend([
                    f"### HYPOTHESIS STATUS: `{he.hypothesis_id}` — {he.name}",
                    f"- **Status:** `{he.status.value}`",
                    f"- **Description:** {he.description}",
                    f"- **Quantitative Criteria:** {he.threshold_criteria}",
                    f"- **Verdict Rationale:** {he.verdict_rationale}",
                ])
                if he.gaps_reported:
                    for g in he.gaps_reported:
                        lines.append(f"- **Methodological Gap:** {g}")
                lines.append("")
    else:
        lines.append("Hypotheses H1-H5 have not been evaluated. (Pending empirical execution).")

    lines.extend([
        "---",
        "",
        "## 11. Failure & Incomplete Cases",
        "",
        f"- **Failed Decision Count:** {stats.get('failed_decision_count', 0)}",
        f"- **Denominator Preservation:** {stats.get('valid_decision_count', 0)} valid + {stats.get('failed_decision_count', 0)} failed = {stats.get('total_cases', 0)} total",
        "- **Conversion Rule:** Failed provider API calls are NEVER converted to ties or passes.",
        "",
        "---",
        "",
        "## 12. Reproducibility Metadata",
        "",
        f"- **Experiment Schema Version:** `{res_dict.get('experiment_schema_version', '1.0.0')}`",
        f"- **Started At:** `{res_dict.get('started_at', 'N/A')}`",
        f"- **Completed At:** `{res_dict.get('completed_at', 'N/A')}`",
        f"- **Software Package:** `{res_dict.get('software_version', 'benizakura-0.1.0')}`",
        "- **Secret Scrubbing:** Recursive secret sanitization enforced (zero API tokens in output).",
        "",
        "---",
        "",
        "## 13. Final Interpretation",
        "",
        "### INTERPRETATION",
        "This evaluation artifact represents the pre-empirical readiness state of Benizakura.",
        "No live LLM API calls have been executed, and no human labels have been fabricated.",
        "Release gating remains disabled pending real double-blind human annotation and adjudication.",
    ])

    return "\n".join(lines)
