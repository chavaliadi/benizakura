from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from benizakura.benchmark import load_benchmark_data
from benizakura.result_schema import CaseDecision


@dataclass
class PositionSensitivityReport:
    """Detailed analysis of order sensitivity and position-dependent errors.

    Terminology strictly applied:
    - 'order sensitivity' / 'forward_reverse_disagreement': raw pass-1 vs pass-2 disagreement in normalized verdict.
    - 'order instability rate': fraction of evaluated cases where normalized(pass-1) != normalized(pass-2).
    - 'position_dependent_error': cases where presenting in Position 1 vs Position 2 alters the final release decision.
    - 'raw_first_position_preference': proportion of passes where Response A (Position 1) was awarded the win.
    """
    total_bidirectional_cases: int
    first_position_wins_pass1: int
    first_position_wins_pass2: int
    raw_first_position_win_rate: float
    order_disagreement_count: int
    order_disagreement_rate: float
    order_instability_rate: float
    position_dependent_errors: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bidirectional_cases": self.total_bidirectional_cases,
            "first_position_wins_pass1": self.first_position_wins_pass1,
            "first_position_wins_pass2": self.first_position_wins_pass2,
            "raw_first_position_win_rate": self.raw_first_position_win_rate,
            "order_disagreement_count": self.order_disagreement_count,
            "order_disagreement_rate": self.order_disagreement_rate,
            "order_instability_rate": self.order_instability_rate,
            "position_dependent_errors": self.position_dependent_errors,
            "details": self.details,
        }


@dataclass
class VerbositySensitivityReport:
    """Analysis of judge preference between concise correctness and verbose padding."""
    total_probe_cases: int
    concise_wins: int
    verbose_wins: int
    ties: int
    concise_win_rate: float
    verbose_win_rate: float
    verbosity_favored: bool
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_probe_cases": self.total_probe_cases,
            "concise_wins": self.concise_wins,
            "verbose_wins": self.verbose_wins,
            "ties": self.ties,
            "concise_win_rate": self.concise_win_rate,
            "verbose_win_rate": self.verbose_win_rate,
            "verbosity_favored": self.verbosity_favored,
            "details": self.details,
        }


@dataclass
class HallucinationDetectionReport:
    """Analysis of judge detection of authoritative technical hallucinations."""
    total_probe_cases: int
    hallucination_detected_count: int
    hallucination_detection_rate: float
    false_pass_count: int
    false_pass_rate: float
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_probe_cases": self.total_probe_cases,
            "hallucination_detected_count": self.hallucination_detected_count,
            "hallucination_detection_rate": self.hallucination_detection_rate,
            "false_pass_count": self.false_pass_count,
            "false_pass_rate": self.false_pass_rate,
            "details": self.details,
        }


@dataclass
class NearTieBehaviorReport:
    """Analysis of judge handling of genuine technical near-ties."""
    total_probe_cases: int
    ties_recognized: int
    tie_recognition_rate: float
    artificial_decisiveness_count: int
    artificial_candidate_wins: int
    artificial_baseline_wins: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_probe_cases": self.total_probe_cases,
            "ties_recognized": self.ties_recognized,
            "tie_recognition_rate": self.tie_recognition_rate,
            "artificial_decisiveness_count": self.artificial_decisiveness_count,
            "artificial_candidate_wins": self.artificial_candidate_wins,
            "artificial_baseline_wins": self.artificial_baseline_wins,
            "details": self.details,
        }


@dataclass
class AlternativeArchitectureReport:
    """Analysis of judge handling of valid technical alternatives."""
    total_probe_cases: int
    valid_alternatives_accepted: int
    acceptance_rate: float
    alternatives_penalized: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_probe_cases": self.total_probe_cases,
            "valid_alternatives_accepted": self.valid_alternatives_accepted,
            "acceptance_rate": self.acceptance_rate,
            "alternatives_penalized": self.alternatives_penalized,
            "details": self.details,
        }


@dataclass
class CriterionTradeoffReport:
    """Analysis of judge handling of multi-criterion trade-offs."""
    total_probe_cases: int
    tradeoffs_evaluated: int
    balanced_verdicts: int
    one_sided_verdicts: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_probe_cases": self.total_probe_cases,
            "tradeoffs_evaluated": self.tradeoffs_evaluated,
            "balanced_verdicts": self.balanced_verdicts,
            "one_sided_verdicts": self.one_sided_verdicts,
            "details": self.details,
        }


@dataclass
class BiasAnalysisReport:
    """Comprehensive analysis report across all deliberate benchmark probes."""
    position_sensitivity: PositionSensitivityReport
    verbosity_sensitivity: VerbositySensitivityReport
    hallucination_detection: HallucinationDetectionReport
    near_tie_behavior: NearTieBehaviorReport
    alternative_architecture: AlternativeArchitectureReport
    criterion_tradeoff: CriterionTradeoffReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_sensitivity": self.position_sensitivity.to_dict(),
            "verbosity_sensitivity": self.verbosity_sensitivity.to_dict(),
            "hallucination_detection": self.hallucination_detection.to_dict(),
            "near_tie_behavior": self.near_tie_behavior.to_dict(),
            "alternative_architecture": self.alternative_architecture.to_dict(),
            "criterion_tradeoff": self.criterion_tradeoff.to_dict(),
        }


def analyze_bias_probes(
    case_decisions: Sequence[Union[Dict[str, Any], CaseDecision]],
    benchmark_source: Union[Dict[str, Any], Path, str] = "evals/conquer/benchmark_v1.json",
    ground_truth_source: Optional[Union[Dict[str, Any], Path, str]] = None,
) -> BiasAnalysisReport:
    """Evaluate judge decisions across the 6 deliberate benchmark probe categories."""
    bench_data = load_benchmark_data(benchmark_source)
    cases = {c["case_id"]: c for c in bench_data.get("cases", [])}

    gt_cases: Dict[str, Any] = {}
    if ground_truth_source is not None:
        if isinstance(ground_truth_source, (str, Path)) and Path(ground_truth_source).is_file():
            with open(ground_truth_source, "r", encoding="utf-8") as f:
                gt_cases = {c["case_id"]: c for c in json.load(f).get("cases", [])}
        elif isinstance(ground_truth_source, dict):
            gt_cases = {c["case_id"]: c for c in ground_truth_source.get("cases", [])}

    decisions: Dict[str, CaseDecision] = {}
    for item in case_decisions:
        if isinstance(item, CaseDecision):
            decisions[item.case_id] = item
        elif isinstance(item, dict):
            cd = CaseDecision.from_dict(item)
            decisions[cd.case_id] = cd

    # 1. Position / Order Sensitivity
    pos_details = []
    p1_a_wins = 0
    p2_a_wins = 0
    total_passes = 0
    disagreements = 0
    instability_count = 0
    total_bidi = 0

    for cid, b_case in cases.items():
        cd = decisions.get(cid)
        if not cd or cd.execution_status != "SUCCESS":
            continue

        p1 = cd.pass_1_decision
        p2 = cd.pass_2_decision

        if p1 is not None:
            total_passes += 1
            if p1 == "A":
                p1_a_wins += 1

        if p2 is not None:
            total_passes += 1
            if p2 == "A":
                p2_a_wins += 1

        if p1 is not None and p2 is not None:
            total_bidi += 1
            # Note: in pass 1, A is Baseline, B is Candidate
            # in pass 2, A is Candidate, B is Baseline
            # Normalized winners:
            p1_norm = "BASELINE" if p1 == "A" else ("CANDIDATE" if p1 == "B" else "TIE")
            p2_norm = "CANDIDATE" if p2 == "A" else ("BASELINE" if p2 == "B" else "TIE")

            is_disagree = (p1_norm != p2_norm)
            if is_disagree:
                disagreements += 1

            if cd.is_unstable:
                instability_count += 1

            pos_details.append({
                "case_id": cid,
                "pass_1": p1,
                "pass_2": p2,
                "pass_1_norm": p1_norm,
                "pass_2_norm": p2_norm,
                "disagreement": is_disagree,
                "is_unstable": cd.is_unstable,
            })

    first_pos_win_rate = (p1_a_wins + p2_a_wins) / total_passes if total_passes > 0 else 0.0
    disagree_rate = disagreements / total_bidi if total_bidi > 0 else 0.0
    instability_rate = instability_count / total_bidi if total_bidi > 0 else 0.0

    pos_report = PositionSensitivityReport(
        total_bidirectional_cases=total_bidi,
        first_position_wins_pass1=p1_a_wins,
        first_position_wins_pass2=p2_a_wins,
        raw_first_position_win_rate=first_pos_win_rate,
        order_disagreement_count=disagreements,
        order_disagreement_rate=disagree_rate,
        order_instability_rate=instability_rate,
        position_dependent_errors=instability_count,
        details=pos_details,
    )

    # 2. Verbosity Sensitivity (5 cases)
    verb_cases = [c for c in cases.values() if "verbosity" in c.get("bias_probes", [])]
    concise_wins = 0
    verbose_wins = 0
    verb_ties = 0
    verb_details = []

    for c in verb_cases:
        cid = c["case_id"]
        cd = decisions.get(cid)
        if not cd or cd.normalized_decision is None:
            continue

        meta = c.get("metadata", {})
        concise_target = meta.get("concise_answer_target", "CANDIDATE")  # Default in probe design
        norm = cd.normalized_decision

        if norm == concise_target:
            concise_wins += 1
            verdict = "CONCISE_WINS"
        elif norm == "TIE":
            verb_ties += 1
            verdict = "TIE"
        else:
            verbose_wins += 1
            verdict = "VERBOSE_WINS"

        verb_details.append({
            "case_id": cid,
            "normalized_decision": norm,
            "concise_target": concise_target,
            "verdict": verdict,
        })

    tot_verb = len(verb_details)
    c_rate = concise_wins / tot_verb if tot_verb > 0 else 0.0
    v_rate = verbose_wins / tot_verb if tot_verb > 0 else 0.0
    verb_report = VerbositySensitivityReport(
        total_probe_cases=len(verb_cases),
        concise_wins=concise_wins,
        verbose_wins=verbose_wins,
        ties=verb_ties,
        concise_win_rate=c_rate,
        verbose_win_rate=v_rate,
        verbosity_favored=(v_rate > c_rate),
        details=verb_details,
    )

    # 3. Hallucination Detection (4 cases)
    hall_cases = [c for c in cases.values() if "hallucination" in c.get("bias_probes", [])]
    hall_detected = 0
    hall_fp = 0
    hall_details = []

    for c in hall_cases:
        cid = c["case_id"]
        cd = decisions.get(cid)
        if not cd or cd.normalized_decision is None:
            continue

        # In hallucination probes, candidate answer contains fatal flaw, so BASELINE is the correct winner
        gt_winner = "BASELINE"
        norm = cd.normalized_decision

        if norm == gt_winner:
            hall_detected += 1
            status = "DETECTED"
        elif norm == "CANDIDATE":
            hall_fp += 1
            status = "FALSE_PASS"
        else:
            status = "TIE"

        hall_details.append({
            "case_id": cid,
            "normalized_decision": norm,
            "status": status,
        })

    tot_hall = len(hall_details)
    hall_report = HallucinationDetectionReport(
        total_probe_cases=len(hall_cases),
        hallucination_detected_count=hall_detected,
        hallucination_detection_rate=hall_detected / tot_hall if tot_hall > 0 else 0.0,
        false_pass_count=hall_fp,
        false_pass_rate=hall_fp / tot_hall if tot_hall > 0 else 0.0,
        details=hall_details,
    )

    # 4. Near-Tie Behavior (7 cases)
    tie_cases = [c for c in cases.values() if "near_tie" in c.get("bias_probes", [])]
    ties_rec = 0
    art_cand = 0
    art_base = 0
    tie_details = []

    for c in tie_cases:
        cid = c["case_id"]
        cd = decisions.get(cid)
        if not cd or cd.normalized_decision is None:
            continue

        norm = cd.normalized_decision
        if norm == "TIE":
            ties_rec += 1
            verdict = "TIE_RECOGNIZED"
        elif norm == "CANDIDATE":
            art_cand += 1
            verdict = "ARTIFICIAL_CANDIDATE_WIN"
        else:
            art_base += 1
            verdict = "ARTIFICIAL_BASELINE_WIN"

        tie_details.append({
            "case_id": cid,
            "normalized_decision": norm,
            "verdict": verdict,
        })

    tot_tie = len(tie_details)
    art_dec = art_cand + art_base
    tie_report = NearTieBehaviorReport(
        total_probe_cases=len(tie_cases),
        ties_recognized=ties_rec,
        tie_recognition_rate=ties_rec / tot_tie if tot_tie > 0 else 0.0,
        artificial_decisiveness_count=art_dec,
        artificial_candidate_wins=art_cand,
        artificial_baseline_wins=art_base,
        details=tie_details,
    )

    # 5. Alternative Architecture Handling (4 cases)
    alt_cases = [c for c in cases.values() if "alternative_architecture" in c.get("bias_probes", [])]
    alt_accepted = 0
    alt_penalized = 0
    alt_details = []

    for c in alt_cases:
        cid = c["case_id"]
        cd = decisions.get(cid)
        if not cd or cd.normalized_decision is None:
            continue

        norm = cd.normalized_decision
        # Valid alternatives are near-ties or valid candidate designs; penalizing means declaring unwarranted regression
        if norm in ("TIE", "CANDIDATE"):
            alt_accepted += 1
            status = "ACCEPTED"
        else:
            alt_penalized += 1
            status = "PENALIZED"

        alt_details.append({
            "case_id": cid,
            "normalized_decision": norm,
            "status": status,
        })

    tot_alt = len(alt_details)
    alt_report = AlternativeArchitectureReport(
        total_probe_cases=len(alt_cases),
        valid_alternatives_accepted=alt_accepted,
        acceptance_rate=alt_accepted / tot_alt if tot_alt > 0 else 0.0,
        alternatives_penalized=alt_penalized,
        details=alt_details,
    )

    # 6. Criterion Trade-Off Behavior (4 cases)
    tradeoff_cases = [c for c in cases.values() if "criterion_tradeoff" in c.get("bias_probes", [])]
    balanced_count = 0
    one_sided_count = 0
    tradeoff_details = []

    for c in tradeoff_cases:
        cid = c["case_id"]
        cd = decisions.get(cid)
        if not cd or cd.normalized_decision is None:
            continue

        norm = cd.normalized_decision
        if norm == "TIE":
            balanced_count += 1
            verdict = "BALANCED_TRADEOFF"
        else:
            one_sided_count += 1
            verdict = f"DECISIVE_{norm}"

        tradeoff_details.append({
            "case_id": cid,
            "normalized_decision": norm,
            "verdict": verdict,
        })

    tot_tr = len(tradeoff_details)
    tradeoff_report = CriterionTradeoffReport(
        total_probe_cases=len(tradeoff_cases),
        tradeoffs_evaluated=tot_tr,
        balanced_verdicts=balanced_count,
        one_sided_verdicts=one_sided_count,
        details=tradeoff_details,
    )

    return BiasAnalysisReport(
        position_sensitivity=pos_report,
        verbosity_sensitivity=verb_report,
        hallucination_detection=hall_report,
        near_tie_behavior=tie_report,
        alternative_architecture=alt_report,
        criterion_tradeoff=tradeoff_report,
    )
