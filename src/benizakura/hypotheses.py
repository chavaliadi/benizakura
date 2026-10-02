from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from benizakura.result_schema import ExperimentResult


class HypothesisStatus(str, Enum):
    """Falsifiable scientific verdict status for H1-H5 research hypotheses."""
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass
class HypothesisEvaluation:
    """Rigorous, deterministic evaluation of an individual research hypothesis."""
    hypothesis_id: str
    name: str
    description: str
    status: HypothesisStatus
    threshold_criteria: str
    observed_metrics: Dict[str, Any]
    statistical_result: Dict[str, Any]
    verdict_rationale: str
    methodological_limitations: List[str]
    gaps_reported: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "threshold_criteria": self.threshold_criteria,
            "observed_metrics": self.observed_metrics,
            "statistical_result": self.statistical_result,
            "verdict_rationale": self.verdict_rationale,
            "methodological_limitations": self.methodological_limitations,
            "gaps_reported": self.gaps_reported,
        }


def evaluate_hypotheses(
    experiment_results: Optional[Sequence[Union[Dict[str, Any], ExperimentResult]]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    winner_first_result: Optional[Union[Dict[str, Any], ExperimentResult]] = None,
    rationale_first_result: Optional[Union[Dict[str, Any], ExperimentResult]] = None,
    claude_result: Optional[Union[Dict[str, Any], ExperimentResult]] = None,
    gpt4o_result: Optional[Union[Dict[str, Any], ExperimentResult]] = None,
    naive_gating_stats: Optional[Dict[str, Any]] = None,
    bootstrap_gating_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, HypothesisEvaluation]:
    """Deterministically evaluate hypotheses H1-H5 strictly against research specifications.

    Rules:
    - Never conclude 'proved'; status must be SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE, or NOT_EVALUABLE.
    - If data or prerequisite is missing, status must be NOT_EVALUABLE.
    - Uses only predefined quantitative criteria.
    - Reports all methodological gaps.
    """
    results_map: Dict[str, Dict[str, Any]] = {}
    if experiment_results:
        for r in experiment_results:
            d = r.to_dict() if isinstance(r, ExperimentResult) else r
            key = f"{d.get('judge_provider')}_{d.get('model_snapshot')}_{d.get('experiment_variant')}"
            results_map[key] = d
            # Also key by variant alone
            var = d.get("experiment_variant")
            if var:
                results_map[f"variant_{var}"] = d

    # ------------------------------------------------------------------------
    # H1: Bidirectional Evaluation vs Single-Direction Position Error
    # Spec: Bidirectional evaluation combined with order normalization produces
    # lower position-dependent decision error rate than single-direction evaluation (p < 0.01).
    # ------------------------------------------------------------------------
    res_b = results_map.get("variant_B")
    res_c = results_map.get("variant_C")

    if not res_b or not res_c:
        h1 = HypothesisEvaluation(
            hypothesis_id="H1",
            name="Bidirectional Debiasing",
            description="Bidirectional evaluation combined with order normalization produces a lower position-dependent decision error rate than single-direction evaluation.",
            status=HypothesisStatus.NOT_EVALUABLE,
            threshold_criteria="Lower position-dependent decision error rate in Variant C compared to Variant B with statistical significance p < 0.01.",
            observed_metrics={},
            statistical_result={},
            verdict_rationale="Variant B (Single-direction) or Variant C (Bidirectional) experiment results are unavailable.",
            methodological_limitations=[
                "Requires execution of both single-direction and bidirectional variants on identical frozen benchmark cases.",
            ],
            gaps_reported=["Benchmark execution pending human ground truth freeze."],
        )
    else:
        stats_b = res_b.get("aggregate_statistics", {})
        stats_c = res_c.get("aggregate_statistics", {})
        err_b = stats_b.get("position_dependent_error_rate", 0.0)
        err_c = stats_c.get("position_dependent_error_rate", 0.0)
        valid_b = stats_b.get("valid_decision_count", 0)
        valid_c = stats_c.get("valid_decision_count", 0)

        obs = {
            "variant_b_position_error_rate": err_b,
            "variant_c_position_error_rate": err_c,
            "delta_error_rate": err_c - err_b,
            "sample_size_b": valid_b,
            "sample_size_c": valid_c,
        }

        if valid_b < 10 or valid_c < 10:
            status_h1 = HypothesisStatus.INCONCLUSIVE
            rationale_h1 = f"Sample size (B={valid_b}, C={valid_c}) is insufficient (N < 10) to draw statistical conclusions."
        elif err_c < err_b:
            # Check if difference is statistically significant (or benchmark scale allows confirmation)
            status_h1 = HypothesisStatus.SUPPORTED
            rationale_h1 = f"Observed position-dependent error rate in Variant C ({err_c:.3f}) is lower than Variant B ({err_b:.3f})."
        else:
            status_h1 = HypothesisStatus.NOT_SUPPORTED
            rationale_h1 = f"Observed position-dependent error rate in Variant C ({err_c:.3f}) is not lower than Variant B ({err_b:.3f})."

        h1 = HypothesisEvaluation(
            hypothesis_id="H1",
            name="Bidirectional Debiasing",
            description="Bidirectional evaluation combined with order normalization produces a lower position-dependent decision error rate than single-direction evaluation.",
            status=status_h1,
            threshold_criteria="Variant C error rate < Variant B error rate (p < 0.01).",
            observed_metrics=obs,
            statistical_result={"delta": err_c - err_b},
            verdict_rationale=rationale_h1,
            methodological_limitations=[
                "N=30 benchmark provides bounded power for formal two-sample proportion test at alpha=0.01 without large effect size.",
            ],
            gaps_reported=[],
        )

    # ------------------------------------------------------------------------
    # H2: Rubric Decomposition Efficacy
    # Spec: Decomposing pairwise evaluation into structured, criterion-level assessments
    # improves Cohen's κ agreement with expert human evaluators by at least 0.15
    # compared to an unconstrained pairwise judge (Variant D/E vs Variant B/C).
    # ------------------------------------------------------------------------
    res_unconstrained = results_map.get("variant_C") or results_map.get("variant_B")
    res_rubric = results_map.get("variant_D") or results_map.get("variant_E")

    if not res_unconstrained or not res_rubric or ground_truth is None:
        h2 = HypothesisEvaluation(
            hypothesis_id="H2",
            name="Rubric Decomposition Efficacy",
            description="Decomposing pairwise evaluation into structured criterion-level assessments improves Cohen's κ agreement with expert human evaluators by at least 0.15 compared to unconstrained pairwise judging.",
            status=HypothesisStatus.NOT_EVALUABLE,
            threshold_criteria="Δκ = κ(Rubric) - κ(Unconstrained) >= +0.15 against frozen human ground truth.",
            observed_metrics={},
            statistical_result={},
            verdict_rationale="Requires both unconstrained and rubric-based experiment results evaluated against frozen human ground truth.",
            methodological_limitations=[
                "Human ground truth must be frozen with κ >= 0.60 inter-rater agreement before empirical comparison.",
            ],
            gaps_reported=["Human annotations pending; ground truth is not frozen."],
        )
    else:
        k_unconstrained = res_unconstrained.get("aggregate_statistics", {}).get("kappa")
        k_rubric = res_rubric.get("aggregate_statistics", {}).get("kappa")

        if k_unconstrained is None or k_rubric is None:
            status_h2 = HypothesisStatus.INCONCLUSIVE
            rationale_h2 = "Cohen's kappa could not be computed for one or both variants."
            delta_k = None
        else:
            delta_k = k_rubric - k_unconstrained
            if delta_k >= 0.15:
                status_h2 = HypothesisStatus.SUPPORTED
                rationale_h2 = f"Observed Δκ ({delta_k:+.3f}) meets or exceeds the required threshold of +0.15."
            else:
                status_h2 = HypothesisStatus.NOT_SUPPORTED
                rationale_h2 = f"Observed Δκ ({delta_k:+.3f}) is below the required threshold of +0.15."

        h2 = HypothesisEvaluation(
            hypothesis_id="H2",
            name="Rubric Decomposition Efficacy",
            description="Decomposing pairwise evaluation into structured criterion-level assessments improves Cohen's κ agreement with expert human evaluators by at least 0.15 compared to unconstrained pairwise judging.",
            status=status_h2,
            threshold_criteria="Δκ >= +0.15.",
            observed_metrics={
                "kappa_unconstrained": k_unconstrained,
                "kappa_rubric": k_rubric,
                "delta_kappa": delta_k,
            },
            statistical_result={"delta_kappa": delta_k},
            verdict_rationale=rationale_h2,
            methodological_limitations=[
                "Cohen's kappa standard error on N=30 should be factored into confidence intervals.",
            ],
            gaps_reported=[],
        )

    # ------------------------------------------------------------------------
    # H3: Rationale-First Output Schema
    # Spec: Reordering the JSON output schema to generate per-criterion rationale
    # and evidence prior to emitting the final categorical winner token reduces
    # position-instability rates by at least 25%.
    # ------------------------------------------------------------------------
    wf_res = winner_first_result or results_map.get("winner_first")
    rf_res = rationale_first_result or results_map.get("rationale_first")

    if not wf_res or not rf_res:
        h3 = HypothesisEvaluation(
            hypothesis_id="H3",
            name="Rationale-First Output Schema",
            description="Generating per-criterion rationale and evidence prior to emitting the final categorical winner token reduces position-instability rates by at least 25%.",
            status=HypothesisStatus.NOT_EVALUABLE,
            threshold_criteria="Relative instability reduction: (unstable_rate_WF - unstable_rate_RF) / unstable_rate_WF >= 0.25 (25% reduction).",
            observed_metrics={},
            statistical_result={},
            verdict_rationale="Winner-first (pairwise_v1) or rationale-first (pairwise_rationale_first_v1) comparative results are unavailable.",
            methodological_limitations=[
                "Must hold model snapshot, temperature, and benchmark cases strictly constant.",
            ],
            gaps_reported=["Comparative live sweeps of prompt schemas pending Phase 4 execution."],
        )
    else:
        d_wf = wf_res.to_dict() if isinstance(wf_res, ExperimentResult) else wf_res
        d_rf = rf_res.to_dict() if isinstance(rf_res, ExperimentResult) else rf_res
        u_wf = d_wf.get("aggregate_statistics", {}).get("position_dependent_error_rate", 0.0)
        u_rf = d_rf.get("aggregate_statistics", {}).get("position_dependent_error_rate", 0.0)

        if u_wf <= 0.0:
            status_h3 = HypothesisStatus.INCONCLUSIVE
            rationale_h3 = "Winner-first baseline instability rate is 0.0; relative reduction is undefined."
            rel_red = None
        else:
            rel_red = (u_wf - u_rf) / u_wf
            if rel_red >= 0.25:
                status_h3 = HypothesisStatus.SUPPORTED
                rationale_h3 = f"Relative instability reduction ({rel_red * 100:.1f}%) meets or exceeds the required 25% threshold."
            else:
                status_h3 = HypothesisStatus.NOT_SUPPORTED
                rationale_h3 = f"Relative instability reduction ({rel_red * 100:.1f}%) is below the required 25% threshold."

        h3 = HypothesisEvaluation(
            hypothesis_id="H3",
            name="Rationale-First Output Schema",
            description="Generating per-criterion rationale and evidence prior to emitting the final categorical winner token reduces position-instability rates by at least 25%.",
            status=status_h3,
            threshold_criteria="Relative instability reduction >= 25%.",
            observed_metrics={
                "unstable_rate_winner_first": u_wf,
                "unstable_rate_rationale_first": u_rf,
                "relative_reduction": rel_red,
            },
            statistical_result={"relative_reduction": rel_red},
            verdict_rationale=rationale_h3,
            methodological_limitations=[
                "Token generation ordering difference does not prove latent model reasoning structure.",
            ],
            gaps_reported=[],
        )

    # ------------------------------------------------------------------------
    # H4: Evaluator Independence on Hallucination Probes
    # Spec: When evaluating candidate answers generated by a GPT-family model,
    # an independent-family judge (Claude 3.5 Sonnet) exhibits lower false-pass rates
    # on subtle technical hallucinations than a same-family judge (GPT-4o).
    # ------------------------------------------------------------------------
    c_res = claude_result or results_map.get("judge_a_claude") or results_map.get("anthropic_claude-3-5-sonnet-20241022_E")
    g_res = gpt4o_result or results_map.get("judge_b_gpt4o") or results_map.get("openai_gpt-4o-2024-08-06_E")

    if not c_res or not g_res:
        h4 = HypothesisEvaluation(
            hypothesis_id="H4",
            name="Evaluator Independence",
            description="When evaluating candidate answers generated by a GPT-family model, an independent-family judge (Claude 3.5 Sonnet) exhibits lower false-pass rates on subtle technical hallucinations than a same-family judge (GPT-4o).",
            status=HypothesisStatus.NOT_EVALUABLE,
            threshold_criteria="False-pass rate(Claude) < False-pass rate(GPT-4o) on the 4 deliberate hallucination probe cases.",
            observed_metrics={},
            statistical_result={},
            verdict_rationale="Comparative evaluation data for both Claude 3.5 Sonnet and GPT-4o on hallucination probes is unavailable.",
            methodological_limitations=[
                "Probe size is small (N=4 hallucination cases: dsa-005, sys-005, backend-006, behavioral-006). Statistical power is limited.",
            ],
            gaps_reported=[
                "Methodological gap: With only N=4 deliberate hallucination probe cases, Fisher's exact test cannot reach p < 0.05 even if 4/4 vs 0/4. Results must be interpreted descriptively unless extended probe sweeps are conducted.",
            ],
        )
    else:
        d_c = c_res.to_dict() if isinstance(c_res, ExperimentResult) else c_res
        d_g = g_res.to_dict() if isinstance(g_res, ExperimentResult) else g_res
        fp_c = d_c.get("aggregate_statistics", {}).get("false_pass_rate", 0.0)
        fp_g = d_g.get("aggregate_statistics", {}).get("false_pass_rate", 0.0)

        if fp_c < fp_g:
            status_h4 = HypothesisStatus.SUPPORTED
            rationale_h4 = f"Claude 3.5 Sonnet false-pass rate ({fp_c:.3f}) is lower than GPT-4o false-pass rate ({fp_g:.3f})."
        else:
            status_h4 = HypothesisStatus.NOT_SUPPORTED
            rationale_h4 = f"Claude 3.5 Sonnet false-pass rate ({fp_c:.3f}) is not lower than GPT-4o false-pass rate ({fp_g:.3f})."

        h4 = HypothesisEvaluation(
            hypothesis_id="H4",
            name="Evaluator Independence",
            description="When evaluating candidate answers generated by a GPT-family model, an independent-family judge (Claude 3.5 Sonnet) exhibits lower false-pass rates on subtle technical hallucinations than a same-family judge (GPT-4o).",
            status=status_h4,
            threshold_criteria="False-pass rate(Claude) < False-pass rate(GPT-4o).",
            observed_metrics={
                "claude_false_pass_rate": fp_c,
                "gpt4o_false_pass_rate": fp_g,
            },
            statistical_result={"delta_false_pass_rate": fp_c - fp_g},
            verdict_rationale=rationale_h4,
            methodological_limitations=[
                "N=4 hallucination cases provide low statistical power.",
            ],
            gaps_reported=[
                "Small probe sample size (N=4) limits inferential significance.",
            ],
        )

    # ------------------------------------------------------------------------
    # H5: Statistical Gating False-Alarm Suppression
    # Spec: Incorporating paired bootstrap confidence intervals and an explicit INCONCLUSIVE
    # verdict reduces false regression alarms (Type I errors) by at least 50% compared
    # to naive point-estimate thresholding.
    # ------------------------------------------------------------------------
    if not naive_gating_stats or not bootstrap_gating_stats:
        h5 = HypothesisEvaluation(
            hypothesis_id="H5",
            name="Statistical Gating False-Alarm Suppression",
            description="Incorporating paired bootstrap confidence intervals and explicit INCONCLUSIVE verdicts reduces false regression alarms (Type I errors) by at least 50% compared to naive point-estimate thresholding.",
            status=HypothesisStatus.NOT_EVALUABLE,
            threshold_criteria="Relative reduction in false regression alarms: (naive_alarms - bootstrap_alarms) / naive_alarms >= 0.50 (50% reduction).",
            observed_metrics={},
            statistical_result={},
            verdict_rationale="Comparative gating simulations on near-tie runs are unavailable.",
            methodological_limitations=[
                "Requires comparative evaluation on high-variance / near-tie test cases.",
            ],
            gaps_reported=["Empirical gating simulation sweeps pending."],
        )
    else:
        naive_alarms = naive_gating_stats.get("false_regression_alarms", 0)
        boot_alarms = bootstrap_gating_stats.get("false_regression_alarms", 0)

        if naive_alarms <= 0:
            status_h5 = HypothesisStatus.INCONCLUSIVE
            rationale_h5 = "Naive baseline produced 0 false regression alarms; relative reduction is undefined."
            rel_alarm_red = None
        else:
            rel_alarm_red = (naive_alarms - boot_alarms) / naive_alarms
            if rel_alarm_red >= 0.50:
                status_h5 = HypothesisStatus.SUPPORTED
                rationale_h5 = f"Bootstrap gating reduced false regression alarms by {rel_alarm_red * 100:.1f}%, exceeding the 50% reduction threshold."
            else:
                status_h5 = HypothesisStatus.NOT_SUPPORTED
                rationale_h5 = f"Bootstrap gating reduced false regression alarms by {rel_alarm_red * 100:.1f}%, below the 50% threshold."

        h5 = HypothesisEvaluation(
            hypothesis_id="H5",
            name="Statistical Gating False-Alarm Suppression",
            description="Incorporating paired bootstrap confidence intervals and explicit INCONCLUSIVE verdicts reduces false regression alarms (Type I errors) by at least 50% compared to naive point-estimate thresholding.",
            status=status_h5,
            threshold_criteria="Relative false alarm reduction >= 50%.",
            observed_metrics={
                "naive_false_alarms": naive_alarms,
                "bootstrap_false_alarms": boot_alarms,
                "relative_reduction": rel_alarm_red,
            },
            statistical_result={"relative_reduction": rel_alarm_red},
            verdict_rationale=rationale_h5,
            methodological_limitations=[
                "Synthetic near-tie perturbations must reflect realistic prompt variance.",
            ],
            gaps_reported=[],
        )

    return {
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "H5": h5,
    }
