from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from benizakura.benchmark import (
    ALLOWED_CATEGORIES,
    compute_benchmark_hash,
    load_benchmark_data,
)
from benizakura.calibration import ConfusionMatrix, compute_cohens_kappa
from benizakura.result_schema import CaseDecision


CANONICAL_BENCHMARK_SHA256 = "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"
CLASSES = ["CANDIDATE", "BASELINE", "TIE"]


@dataclass
class CategoryComparisonStats:
    """Stratified comparison metrics for an individual domain category."""
    category: str
    total_cases: int
    evaluated_cases: int
    agreed_cases: int
    raw_agreement: float
    cohens_kappa: Optional[float]
    false_pass_count: int
    false_regression_count: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "total_cases": self.total_cases,
            "evaluated_cases": self.evaluated_cases,
            "agreed_cases": self.agreed_cases,
            "raw_agreement": self.raw_agreement,
            "cohens_kappa": self.cohens_kappa,
            "false_pass_count": self.false_pass_count,
            "false_regression_count": self.false_regression_count,
            "details": self.details,
        }


@dataclass
class ProbeComparisonStats:
    """Stratified comparison metrics for an individual deliberate bias probe."""
    probe_type: str
    total_probe_cases: int
    evaluated_probe_cases: int
    agreed_probe_cases: int
    raw_agreement: float
    cohens_kappa: Optional[float]
    false_pass_count: int
    false_regression_count: int
    case_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_type": self.probe_type,
            "total_probe_cases": self.total_probe_cases,
            "evaluated_probe_cases": self.evaluated_probe_cases,
            "agreed_probe_cases": self.agreed_probe_cases,
            "raw_agreement": self.raw_agreement,
            "cohens_kappa": self.cohens_kappa,
            "false_pass_count": self.false_pass_count,
            "false_regression_count": self.false_regression_count,
            "case_ids": self.case_ids,
        }


@dataclass
class CriterionComparisonStats:
    """Agreement statistics across individual rubric criteria."""
    criterion_name: str
    evaluated_cases: int
    agreed_cases: int
    raw_agreement: float
    cohens_kappa: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_name": self.criterion_name,
            "evaluated_cases": self.evaluated_cases,
            "agreed_cases": self.agreed_cases,
            "raw_agreement": self.raw_agreement,
            "cohens_kappa": self.cohens_kappa,
        }


@dataclass
class ComparisonReport:
    """Complete evaluation report comparing LLM judge decisions against frozen human ground truth.

    Exact Metric Definitions:
    - overall_agreement (Po): matching cases divided by total validly compared cases.
    - cohens_kappa: multi-class Cohen's kappa across classes {CANDIDATE, BASELINE, TIE}.
    - candidate_win_agreement: recall for CANDIDATE = P(Judge=CANDIDATE | Human=CANDIDATE).
    - baseline_win_agreement: recall for BASELINE = P(Judge=BASELINE | Human=BASELINE).
    - tie_agreement: recall for TIE = P(Judge=TIE | Human=TIE).
    - false_pass_count / rate: Judge says CANDIDATE when Human Ground Truth says BASELINE.
      Rate is computed relative to Human BASELINE cases (denominator = total Human BASELINE cases).
    - false_regression_count / rate: Judge says BASELINE when Human Ground Truth says CANDIDATE.
      Rate is computed relative to Human CANDIDATE cases (denominator = total Human CANDIDATE cases).
    - directional_agreement: proportion of cases where sign(Judge) == sign(Human)
      with sign(CANDIDATE) = +1, sign(BASELINE) = -1, sign(TIE) = 0.
    """
    benchmark_id: str
    benchmark_sha256: str
    ground_truth_id: str
    ground_truth_sha256: str
    ground_truth_status: str
    total_benchmark_cases: int
    total_evaluated_cases: int
    failed_execution_cases: int
    overall_agreement: float
    cohens_kappa: float
    kappa_se: float
    kappa_ci: Tuple[float, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    candidate_win_agreement: float
    baseline_win_agreement: float
    tie_agreement: float
    false_pass_count: int
    false_pass_rate: float
    false_regression_count: int
    false_regression_rate: float
    directional_agreement: float
    category_breakdown: Dict[str, CategoryComparisonStats]
    probe_breakdown: Dict[str, ProbeComparisonStats]
    criterion_breakdown: Dict[str, CriterionComparisonStats] = field(default_factory=dict)
    unmatched_cases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_sha256": self.benchmark_sha256,
            "ground_truth_id": self.ground_truth_id,
            "ground_truth_sha256": self.ground_truth_sha256,
            "ground_truth_status": self.ground_truth_status,
            "total_benchmark_cases": self.total_benchmark_cases,
            "total_evaluated_cases": self.total_evaluated_cases,
            "failed_execution_cases": self.failed_execution_cases,
            "overall_agreement": self.overall_agreement,
            "cohens_kappa": self.cohens_kappa,
            "kappa_se": self.kappa_se,
            "kappa_ci": list(self.kappa_ci),
            "confusion_matrix": self.confusion_matrix,
            "candidate_win_agreement": self.candidate_win_agreement,
            "baseline_win_agreement": self.baseline_win_agreement,
            "tie_agreement": self.tie_agreement,
            "false_pass_count": self.false_pass_count,
            "false_pass_rate": self.false_pass_rate,
            "false_regression_count": self.false_regression_count,
            "false_regression_rate": self.false_regression_rate,
            "directional_agreement": self.directional_agreement,
            "category_breakdown": {k: v.to_dict() for k, v in self.category_breakdown.items()},
            "probe_breakdown": {k: v.to_dict() for k, v in self.probe_breakdown.items()},
            "criterion_breakdown": {k: v.to_dict() for k, v in self.criterion_breakdown.items()},
            "unmatched_cases": self.unmatched_cases,
        }


def compare_judge_to_human(
    judge_decisions: Sequence[Union[Dict[str, Any], CaseDecision]],
    ground_truth_source: Union[Dict[str, Any], Path, str],
    benchmark_source: Union[Dict[str, Any], Path, str] = "evals/conquer/benchmark_v1.json",
    verify_benchmark_hash: bool = True,
) -> ComparisonReport:
    """Compare LLM judge decisions directly against human ground truth consensus.

    Strictly validates:
    - Benchmark hash matches canonical SHA when benchmark_id is conquer-benchmark-v1
    - Ground truth contains valid consensus records
    - Failed judge API calls are preserved in denominator and excluded from agreement numerator
    - All metrics match research specifications
    """
    # 1. Load benchmark
    bench_data = load_benchmark_data(benchmark_source)
    bench_id = bench_data.get("benchmark_id", "conquer-benchmark-v1")
    bench_sha = compute_benchmark_hash(benchmark_source)
    if verify_benchmark_hash and bench_id == "conquer-benchmark-v1":
        if bench_sha != CANONICAL_BENCHMARK_SHA256:
            raise ValueError(
                f"Benchmark SHA mismatch in comparison layer: expected {CANONICAL_BENCHMARK_SHA256}, got {bench_sha}."
            )

    benchmark_cases = {c["case_id"]: c for c in bench_data.get("cases", [])}

    # 2. Load ground truth
    if isinstance(ground_truth_source, (str, Path)):
        p = Path(ground_truth_source)
        if not p.is_file():
            raise FileNotFoundError(f"Ground truth source file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    elif isinstance(ground_truth_source, dict):
        gt_data = ground_truth_source
    else:
        raise TypeError(f"Unsupported ground_truth_source type: {type(ground_truth_source)}")

    gt_id = gt_data.get("ground_truth_id", "conquer-ground-truth-v1")
    gt_status = gt_data.get("status", "UNKNOWN")
    gt_sha = hashlib.sha256(json.dumps(gt_data, sort_keys=True).encode("utf-8")).hexdigest()
    gt_cases = {c["case_id"]: c for c in gt_data.get("cases", [])}

    # 3. Process judge decisions
    decisions_map: Dict[str, CaseDecision] = {}
    for item in judge_decisions:
        if isinstance(item, CaseDecision):
            decisions_map[item.case_id] = item
        elif isinstance(item, dict):
            cd = CaseDecision.from_dict(item)
            decisions_map[cd.case_id] = cd

    judge_ratings: List[str] = []
    human_ratings: List[str] = []
    failed_count = 0
    unmatched_cases: List[str] = []

    # Category and probe collectors
    cat_pairs: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {cat: [] for cat in ALLOWED_CATEGORIES}
    probe_pairs: Dict[str, List[Tuple[str, str, str]]] = {}
    criterion_pairs: Dict[str, List[Tuple[str, str]]] = {}

    for cid, b_case in benchmark_cases.items():
        cat = b_case.get("category", "DSA")
        probes = b_case.get("bias_probes", [])

        # Check ground truth
        gt_case = gt_cases.get(cid)
        if not gt_case:
            unmatched_cases.append(cid)
            continue

        consensus = gt_case.get("consensus")
        if not consensus or not isinstance(consensus, dict) or "winner" not in consensus:
            unmatched_cases.append(cid)
            continue

        h_winner = consensus["winner"]
        if h_winner not in CLASSES:
            unmatched_cases.append(cid)
            continue

        # Check judge decision
        j_dec = decisions_map.get(cid)
        if not j_dec:
            failed_count += 1
            continue

        if j_dec.execution_status != "SUCCESS" or j_dec.normalized_decision is None:
            failed_count += 1
            continue

        j_winner = j_dec.normalized_decision
        if j_winner not in CLASSES:
            failed_count += 1
            continue

        judge_ratings.append(j_winner)
        human_ratings.append(h_winner)

        detail_entry = {
            "case_id": cid,
            "human_winner": h_winner,
            "judge_winner": j_winner,
            "agreed": (j_winner == h_winner),
            "is_unstable": j_dec.is_unstable,
        }
        if cat in cat_pairs:
            cat_pairs[cat].append((j_winner, h_winner, detail_entry))

        for p in probes:
            probe_pairs.setdefault(p, []).append((j_winner, h_winner, cid))

        # Check per-criterion assessments if present
        h_crits = consensus.get("criterion_assessments", [])
        j_crits = j_dec.criterion_assessments
        if isinstance(h_crits, list) and isinstance(j_crits, dict):
            for hc in h_crits:
                cname = hc.get("criterion_name")
                hw = hc.get("winner")
                jw = j_crits.get(cname)
                if cname and hw in CLASSES and jw in CLASSES:
                    criterion_pairs.setdefault(cname, []).append((jw, hw))

    total_valid = len(judge_ratings)

    # Compute overall agreement & Cohen's kappa
    if total_valid > 0:
        agreed_count = sum(1 for j, h in zip(judge_ratings, human_ratings) if j == h)
        overall_agree = agreed_count / total_valid
        kappa, se, ci, conf_mat = compute_cohens_kappa(judge_ratings, human_ratings, classes=CLASSES)
    else:
        overall_agree = 0.0
        kappa, se, ci = 0.0, 0.0, (0.0, 0.0)
        conf_mat = ConfusionMatrix(classes=CLASSES, matrix={r: {c: 0 for c in CLASSES} for r in CLASSES}, total_observations=0)

    # Class-specific agreements (Recall)
    human_cand_total = sum(1 for h in human_ratings if h == "CANDIDATE")
    cand_win_agree = (
        sum(1 for j, h in zip(judge_ratings, human_ratings) if h == "CANDIDATE" and j == "CANDIDATE") / human_cand_total
        if human_cand_total > 0 else 1.0
    )

    human_base_total = sum(1 for h in human_ratings if h == "BASELINE")
    base_win_agree = (
        sum(1 for j, h in zip(judge_ratings, human_ratings) if h == "BASELINE" and j == "BASELINE") / human_base_total
        if human_base_total > 0 else 1.0
    )

    human_tie_total = sum(1 for h in human_ratings if h == "TIE")
    tie_agree = (
        sum(1 for j, h in zip(judge_ratings, human_ratings) if h == "TIE" and j == "TIE") / human_tie_total
        if human_tie_total > 0 else 1.0
    )

    # False pass & false regression
    # False pass: Human=BASELINE (candidate regressed), Judge=CANDIDATE
    false_pass_count = sum(1 for j, h in zip(judge_ratings, human_ratings) if h == "BASELINE" and j == "CANDIDATE")
    false_pass_rate = false_pass_count / human_base_total if human_base_total > 0 else 0.0

    # False regression: Human=CANDIDATE (candidate improved), Judge=BASELINE
    false_regression_count = sum(1 for j, h in zip(judge_ratings, human_ratings) if h == "CANDIDATE" and j == "BASELINE")
    false_regression_rate = false_regression_count / human_cand_total if human_cand_total > 0 else 0.0

    # Directional agreement: sign(judge) == sign(human)
    sign_map = {"CANDIDATE": 1, "BASELINE": -1, "TIE": 0}
    dir_agree_count = sum(1 for j, h in zip(judge_ratings, human_ratings) if sign_map[j] == sign_map[h])
    directional_agree = dir_agree_count / total_valid if total_valid > 0 else 0.0

    # Category breakdown
    cat_breakdown: Dict[str, CategoryComparisonStats] = {}
    for cat, pairs in cat_pairs.items():
        n_cat = len(pairs)
        if n_cat > 0:
            j_sub = [p[0] for p in pairs]
            h_sub = [p[1] for p in pairs]
            details = [p[2] for p in pairs]
            agr = sum(1 for j, h in zip(j_sub, h_sub) if j == h)
            raw_agr = agr / n_cat
            k_sub, _, _, _ = compute_cohens_kappa(j_sub, h_sub, classes=CLASSES) if len(set(j_sub) | set(h_sub)) > 1 else (1.0 if raw_agr == 1.0 else 0.0, 0.0, (0.0, 0.0), None)
            fp_sub = sum(1 for j, h in zip(j_sub, h_sub) if h == "BASELINE" and j == "CANDIDATE")
            fr_sub = sum(1 for j, h in zip(j_sub, h_sub) if h == "CANDIDATE" and j == "BASELINE")
        else:
            raw_agr, k_sub, fp_sub, fr_sub = 0.0, None, 0, 0
            agr = 0
            details = []

        cat_breakdown[cat] = CategoryComparisonStats(
            category=cat,
            total_cases=len([c for c in benchmark_cases.values() if c.get("category") == cat]),
            evaluated_cases=n_cat,
            agreed_cases=agr,
            raw_agreement=raw_agr,
            cohens_kappa=k_sub,
            false_pass_count=fp_sub,
            false_regression_count=fr_sub,
            details=details,
        )

    # Probe breakdown
    probe_breakdown: Dict[str, ProbeComparisonStats] = {}
    for p_name, pairs in probe_pairs.items():
        n_p = len(pairs)
        if n_p > 0:
            j_sub = [p[0] for p in pairs]
            h_sub = [p[1] for p in pairs]
            cids = [p[2] for p in pairs]
            agr = sum(1 for j, h in zip(j_sub, h_sub) if j == h)
            raw_agr = agr / n_p
            k_sub, _, _, _ = compute_cohens_kappa(j_sub, h_sub, classes=CLASSES) if len(set(j_sub) | set(h_sub)) > 1 else (1.0 if raw_agr == 1.0 else 0.0, 0.0, (0.0, 0.0), None)
            fp_sub = sum(1 for j, h in zip(j_sub, h_sub) if h == "BASELINE" and j == "CANDIDATE")
            fr_sub = sum(1 for j, h in zip(j_sub, h_sub) if h == "CANDIDATE" and j == "BASELINE")
        else:
            raw_agr, k_sub, fp_sub, fr_sub = 0.0, None, 0, 0
            agr = 0
            cids = []

        probe_breakdown[p_name] = ProbeComparisonStats(
            probe_type=p_name,
            total_probe_cases=len([c for c in benchmark_cases.values() if p_name in c.get("bias_probes", [])]),
            evaluated_probe_cases=n_p,
            agreed_probe_cases=agr,
            raw_agreement=raw_agr,
            cohens_kappa=k_sub,
            false_pass_count=fp_sub,
            false_regression_count=fr_sub,
            case_ids=cids,
        )

    # Criterion breakdown
    crit_breakdown: Dict[str, CriterionComparisonStats] = {}
    for cname, pairs in criterion_pairs.items():
        n_c = len(pairs)
        if n_c > 0:
            j_sub = [p[0] for p in pairs]
            h_sub = [p[1] for p in pairs]
            agr = sum(1 for j, h in zip(j_sub, h_sub) if j == h)
            raw_agr = agr / n_c
            k_sub, _, _, _ = compute_cohens_kappa(j_sub, h_sub, classes=CLASSES)
        else:
            agr, raw_agr, k_sub = 0, 0.0, None

        crit_breakdown[cname] = CriterionComparisonStats(
            criterion_name=cname,
            evaluated_cases=n_c,
            agreed_cases=agr,
            raw_agreement=raw_agr,
            cohens_kappa=k_sub,
        )

    return ComparisonReport(
        benchmark_id=bench_id,
        benchmark_sha256=bench_sha,
        ground_truth_id=gt_id,
        ground_truth_sha256=gt_sha,
        ground_truth_status=gt_status,
        total_benchmark_cases=len(benchmark_cases),
        total_evaluated_cases=total_valid,
        failed_execution_cases=failed_count,
        overall_agreement=overall_agree,
        cohens_kappa=kappa,
        kappa_se=se,
        kappa_ci=ci,
        confusion_matrix=conf_mat.matrix,
        candidate_win_agreement=cand_win_agree,
        baseline_win_agreement=base_win_agree,
        tie_agreement=tie_agree,
        false_pass_count=false_pass_count,
        false_pass_rate=false_pass_rate,
        false_regression_count=false_regression_count,
        false_regression_rate=false_regression_rate,
        directional_agreement=directional_agree,
        category_breakdown=cat_breakdown,
        probe_breakdown=probe_breakdown,
        criterion_breakdown=crit_breakdown,
        unmatched_cases=unmatched_cases,
    )
