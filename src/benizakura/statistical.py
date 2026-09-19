from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benizakura.models import (
    BidirectionalEvaluationResult,
    ConsistencyOutcome,
    NormalizedWinner,
    classify_consistency,
)


@dataclass
class EvaluationSample:
    """Represents a single evaluated benchmark case preserving case identity and bidirectional outcomes.

    Attributes:
        case_id: Unique identifier matching the benchmark test case.
        result: BidirectionalEvaluationResult containing forward and reverse passes.
        metadata: Optional dictionary for auxiliary case execution details.
    """
    case_id: str
    result: BidirectionalEvaluationResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("EvaluationSample case_id must be a non-empty string.")

    @property
    def consistency(self) -> ConsistencyOutcome:
        """Classify overall consistency across both presentation passes."""
        return classify_consistency(
            self.result.pass1_normalized_winner,
            self.result.pass2_normalized_winner,
        )

    @property
    def resolved_winner(self) -> Optional[NormalizedWinner]:
        """Return NormalizedWinner if both passes consistently agreed; None if position unstable."""
        if self.result.pass1_normalized_winner == self.result.pass2_normalized_winner:
            return self.result.pass1_normalized_winner
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "result": self.result.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationSample:
        return cls(
            case_id=data["case_id"],
            result=BidirectionalEvaluationResult.from_dict(data["result"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationDataset:
    """Represents an ordered collection of EvaluationSample instances for a benchmark run.

    Enforces case ID uniqueness to guarantee that statistical pairing remains valid.
    """
    samples: List[EvaluationSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen_ids = set()
        for sample in self.samples:
            cid = sample.case_id.strip()
            if cid in seen_ids:
                raise ValueError(f"Duplicate case_id detected in dataset: '{sample.case_id}'.")
            seen_ids.add(cid)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> EvaluationSample:
        return self.samples[index]

    @property
    def case_ids(self) -> List[str]:
        return [s.case_id for s in self.samples]

    def get_sample(self, case_id: str) -> Optional[EvaluationSample]:
        target = case_id.strip()
        for s in self.samples:
            if s.case_id.strip() == target:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationDataset:
        return cls(
            samples=[EvaluationSample.from_dict(s) for s in data.get("samples", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PairedCaseOutcome:
    """Represents the paired comparison delta and consistency classification for a single case.

    Paired difference convention:
    +1.0 = Candidate consistently won this case
    -1.0 = Baseline consistently won this case (Candidate regressed)
     0.0 = Consistent tie or position-unstable (no clear advantage demonstrated)

    Attributes:
        case_id: Identifier of the evaluated case.
        outcome: Overall consistency outcome.
        paired_difference: Numeric delta in {-1.0, 0.0, +1.0}.
        criterion_outcomes: Map of criterion name to its ConsistencyOutcome (only for assessed criteria).
    """
    case_id: str
    outcome: ConsistencyOutcome
    paired_difference: float
    criterion_outcomes: Dict[str, ConsistencyOutcome] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "outcome": self.outcome.value,
            "paired_difference": self.paired_difference,
            "criterion_outcomes": {k: v.value for k, v in self.criterion_outcomes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PairedCaseOutcome:
        return cls(
            case_id=data["case_id"],
            outcome=ConsistencyOutcome(data["outcome"]),
            paired_difference=float(data["paired_difference"]),
            criterion_outcomes={
                k: ConsistencyOutcome(v) for k, v in data.get("criterion_outcomes", {}).items()
            },
        )


@dataclass
class CriterionAggregate:
    """Aggregated evaluation metrics for a specific rubric criterion across assessed cases.

    Only cases where this criterion was explicitly assessed in both passes are included.

    Attributes:
        criterion_name: Name of the criterion.
        total_assessed_cases: Denominator: number of cases where this criterion was evaluated.
        candidate_wins: Consistent candidate wins on this criterion.
        baseline_wins: Consistent baseline wins on this criterion.
        ties: Consistent ties on this criterion.
        inconsistent_cases: Position unstable cases on this criterion.
        candidate_win_rate: candidate_wins / total_assessed_cases (0.0 if total is 0).
        baseline_win_rate: baseline_wins / total_assessed_cases (0.0 if total is 0).
        tie_rate: ties / total_assessed_cases (0.0 if total is 0).
        consistency_rate: (candidate_wins + baseline_wins + ties) / total_assessed_cases.
        observed_difference: candidate_win_rate - baseline_win_rate.
    """
    criterion_name: str
    total_assessed_cases: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    inconsistent_cases: int
    candidate_win_rate: float
    baseline_win_rate: float
    tie_rate: float
    consistency_rate: float
    observed_difference: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_name": self.criterion_name,
            "total_assessed_cases": self.total_assessed_cases,
            "candidate_wins": self.candidate_wins,
            "baseline_wins": self.baseline_wins,
            "ties": self.ties,
            "inconsistent_cases": self.inconsistent_cases,
            "candidate_win_rate": self.candidate_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "tie_rate": self.tie_rate,
            "consistency_rate": self.consistency_rate,
            "observed_difference": self.observed_difference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CriterionAggregate:
        return cls(
            criterion_name=data["criterion_name"],
            total_assessed_cases=int(data["total_assessed_cases"]),
            candidate_wins=int(data["candidate_wins"]),
            baseline_wins=int(data["baseline_wins"]),
            ties=int(data["ties"]),
            inconsistent_cases=int(data["inconsistent_cases"]),
            candidate_win_rate=float(data["candidate_win_rate"]),
            baseline_win_rate=float(data["baseline_win_rate"]),
            tie_rate=float(data["tie_rate"]),
            consistency_rate=float(data["consistency_rate"]),
            observed_difference=float(data["observed_difference"]),
        )


@dataclass
class EffectSize:
    """Quantifies practical magnitude of difference separate from statistical uncertainty.

    Attributes:
        metric_name: Name of the effect size metric (e.g., 'net_win_rate_difference', 'cohens_g').
        value: Numeric value of the metric.
        cohens_g: Optional Cohen's g for sign test on decisive non-tie cases: (P_cand - 0.5) in [-0.5, +0.5].
        interpretation: Descriptive summary of magnitude and direction.
    """
    metric_name: str
    value: float
    cohens_g: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "metric_name": self.metric_name,
            "value": self.value,
            "interpretation": self.interpretation,
        }
        if self.cohens_g is not None:
            res["cohens_g"] = self.cohens_g
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EffectSize:
        return cls(
            metric_name=data["metric_name"],
            value=float(data["value"]),
            cohens_g=float(data["cohens_g"]) if data.get("cohens_g") is not None else None,
            interpretation=data.get("interpretation", ""),
        )


@dataclass
class StatisticalAnalysis:
    """Structured statistical summary of a pairwise evaluation dataset.

    Reports empirical evidence, paired differences, bootstrap confidence intervals, and effect size.
    Does NOT declare a PASS/FAIL release gate.

    Attributes:
        sample_size: Total number of evaluated cases (denominator for overall rates).
        candidate_wins: Consistent candidate wins across both passes.
        baseline_wins: Consistent baseline wins across both passes.
        ties: Consistent ties across both passes.
        inconsistent_cases: Cases where presentation order altered the outcome (position unstable).
        candidate_win_rate: candidate_wins / sample_size.
        baseline_win_rate: baseline_wins / sample_size.
        tie_rate: ties / sample_size.
        consistency_rate: (candidate_wins + baseline_wins + ties) / sample_size.
        observed_difference: candidate_win_rate - baseline_win_rate.
        confidence_level: Configured bootstrap confidence level (e.g., 0.95).
        confidence_interval: (ci_lower, ci_upper) bootstrap bounds for observed_difference.
        num_bootstrap_samples: Number of bootstrap iterations performed.
        random_seed: Random seed used for reproducible bootstrap resampling.
        effect_size: EffectSize object detailing magnitude.
        criterion_results: Dictionary mapping criterion names to their CriterionAggregate.
        paired_cases: List of PairedCaseOutcome instances preserving per-case identities.
        metadata: Optional dictionary for execution metadata.
    """
    sample_size: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    inconsistent_cases: int

    candidate_win_rate: float
    baseline_win_rate: float
    tie_rate: float
    consistency_rate: float
    observed_difference: float

    confidence_level: float
    confidence_interval: Tuple[float, float]
    num_bootstrap_samples: int
    random_seed: Optional[int]

    effect_size: EffectSize
    criterion_results: Dict[str, CriterionAggregate] = field(default_factory=dict)
    paired_cases: List[PairedCaseOutcome] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_regressed_cases(self) -> List[str]:
        """Return case IDs where baseline won consistently (Candidate performance regressed)."""
        return [p.case_id for p in self.paired_cases if p.outcome == ConsistencyOutcome.CONSISTENT_BASELINE_WIN]

    def get_improved_cases(self) -> List[str]:
        """Return case IDs where candidate won consistently (Candidate performance improved)."""
        return [p.case_id for p in self.paired_cases if p.outcome == ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN]

    def get_unstable_cases(self) -> List[str]:
        """Return case IDs where the judge suffered position bias (presentation unstable)."""
        return [p.case_id for p in self.paired_cases if p.outcome == ConsistencyOutcome.POSITION_UNSTABLE]

    def get_tied_cases(self) -> List[str]:
        """Return case IDs where both variants tied consistently."""
        return [p.case_id for p in self.paired_cases if p.outcome == ConsistencyOutcome.CONSISTENT_TIE]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "candidate_wins": self.candidate_wins,
            "baseline_wins": self.baseline_wins,
            "ties": self.ties,
            "inconsistent_cases": self.inconsistent_cases,
            "candidate_win_rate": self.candidate_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "tie_rate": self.tie_rate,
            "consistency_rate": self.consistency_rate,
            "observed_difference": self.observed_difference,
            "confidence_level": self.confidence_level,
            "confidence_interval": list(self.confidence_interval),
            "num_bootstrap_samples": self.num_bootstrap_samples,
            "random_seed": self.random_seed,
            "effect_size": self.effect_size.to_dict(),
            "criterion_results": {k: v.to_dict() for k, v in self.criterion_results.items()},
            "paired_cases": [p.to_dict() for p in self.paired_cases],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StatisticalAnalysis:
        ci_raw = data["confidence_interval"]
        ci: Tuple[float, float] = (float(ci_raw[0]), float(ci_raw[1]))
        return cls(
            sample_size=int(data["sample_size"]),
            candidate_wins=int(data["candidate_wins"]),
            baseline_wins=int(data["baseline_wins"]),
            ties=int(data["ties"]),
            inconsistent_cases=int(data["inconsistent_cases"]),
            candidate_win_rate=float(data["candidate_win_rate"]),
            baseline_win_rate=float(data["baseline_win_rate"]),
            tie_rate=float(data["tie_rate"]),
            consistency_rate=float(data["consistency_rate"]),
            observed_difference=float(data["observed_difference"]),
            confidence_level=float(data["confidence_level"]),
            confidence_interval=ci,
            num_bootstrap_samples=int(data["num_bootstrap_samples"]),
            random_seed=data.get("random_seed"),
            effect_size=EffectSize.from_dict(data["effect_size"]),
            criterion_results={
                k: CriterionAggregate.from_dict(v) for k, v in data.get("criterion_results", {}).items()
            },
            paired_cases=[PairedCaseOutcome.from_dict(p) for p in data.get("paired_cases", [])],
            metadata=data.get("metadata", {}),
        )


class StatisticalAnalyzer:
    """Computes deterministic paired difference statistics and bootstrap confidence intervals.

    Operates strictly on pairwise evaluation datasets without external APIs or nondeterminism.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        num_bootstrap_samples: int = 1000,
        random_seed: Optional[int] = 42,
    ):
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be between 0.0 and 1.0 exclusive, got {confidence_level}.")
        if num_bootstrap_samples < 1:
            raise ValueError(f"num_bootstrap_samples must be at least 1, got {num_bootstrap_samples}.")

        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.random_seed = random_seed

    def analyze(self, dataset: EvaluationDataset) -> StatisticalAnalysis:
        """Analyze an EvaluationDataset and compute empirical rates, paired deltas, and bootstrap CI."""
        n = len(dataset)

        # Handle empty dataset edge case explicitly
        if n == 0:
            return StatisticalAnalysis(
                sample_size=0,
                candidate_wins=0,
                baseline_wins=0,
                ties=0,
                inconsistent_cases=0,
                candidate_win_rate=0.0,
                baseline_win_rate=0.0,
                tie_rate=0.0,
                consistency_rate=0.0,
                observed_difference=0.0,
                confidence_level=self.confidence_level,
                confidence_interval=(0.0, 0.0),
                num_bootstrap_samples=self.num_bootstrap_samples,
                random_seed=self.random_seed,
                effect_size=EffectSize(
                    metric_name="net_win_rate_difference",
                    value=0.0,
                    cohens_g=None,
                    interpretation="No evaluation cases present (sample size 0).",
                ),
                criterion_results={},
                paired_cases=[],
            )

        candidate_wins = 0
        baseline_wins = 0
        ties = 0
        inconsistent_cases = 0

        paired_cases: List[PairedCaseOutcome] = []
        criterion_case_map: Dict[str, Dict[str, ConsistencyOutcome]] = {}

        for sample in dataset:
            outcome = sample.consistency
            delta: float = 0.0

            if outcome == ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN:
                candidate_wins += 1
                delta = 1.0
            elif outcome == ConsistencyOutcome.CONSISTENT_BASELINE_WIN:
                baseline_wins += 1
                delta = -1.0
            elif outcome == ConsistencyOutcome.CONSISTENT_TIE:
                ties += 1
                delta = 0.0
            else:  # ConsistencyOutcome.POSITION_UNSTABLE
                inconsistent_cases += 1
                delta = 0.0

            # Inspect per-criterion outcomes across both passes
            case_crit_outcomes: Dict[str, ConsistencyOutcome] = {}
            # Collect all criterion names that appeared in either pass
            crit_names = set()
            for ca in sample.result.pass1_judgment.criterion_assessments:
                crit_names.add(ca.criterion_name.strip().lower())
            for ca in sample.result.pass2_judgment.criterion_assessments:
                crit_names.add(ca.criterion_name.strip().lower())

            for c_name in crit_names:
                norm1, norm2 = sample.result.get_criterion_normalized_winners(c_name)
                # Only include when assessments exist in both passes
                if norm1 is not None and norm2 is not None:
                    c_outcome = classify_consistency(norm1, norm2)
                    case_crit_outcomes[c_name] = c_outcome
                    if c_name not in criterion_case_map:
                        criterion_case_map[c_name] = {}
                    criterion_case_map[c_name][sample.case_id] = c_outcome

            paired_cases.append(
                PairedCaseOutcome(
                    case_id=sample.case_id,
                    outcome=outcome,
                    paired_difference=delta,
                    criterion_outcomes=case_crit_outcomes,
                )
            )

        # Compute empirical rates (explicit denominator: total sample size n)
        candidate_win_rate = round(candidate_wins / n, 4)
        baseline_win_rate = round(baseline_wins / n, 4)
        tie_rate = round(ties / n, 4)
        consistency_rate = round((candidate_wins + baseline_wins + ties) / n, 4)
        observed_difference = round(candidate_win_rate - baseline_win_rate, 4)

        # Compute criterion-level aggregates
        criterion_results: Dict[str, CriterionAggregate] = {}
        for c_name, case_outcomes in criterion_case_map.items():
            c_total = len(case_outcomes)
            if c_total == 0:
                continue
            c_cand = sum(1 for o in case_outcomes.values() if o == ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN)
            c_base = sum(1 for o in case_outcomes.values() if o == ConsistencyOutcome.CONSISTENT_BASELINE_WIN)
            c_ties = sum(1 for o in case_outcomes.values() if o == ConsistencyOutcome.CONSISTENT_TIE)
            c_unstable = sum(1 for o in case_outcomes.values() if o == ConsistencyOutcome.POSITION_UNSTABLE)

            c_c_rate = round(c_cand / c_total, 4)
            c_b_rate = round(c_base / c_total, 4)
            c_t_rate = round(c_ties / c_total, 4)
            c_cons_rate = round((c_cand + c_base + c_ties) / c_total, 4)
            c_diff = round(c_c_rate - c_b_rate, 4)

            criterion_results[c_name] = CriterionAggregate(
                criterion_name=c_name,
                total_assessed_cases=c_total,
                candidate_wins=c_cand,
                baseline_wins=c_base,
                ties=c_ties,
                inconsistent_cases=c_unstable,
                candidate_win_rate=c_c_rate,
                baseline_win_rate=c_b_rate,
                tie_rate=c_t_rate,
                consistency_rate=c_cons_rate,
                observed_difference=c_diff,
            )

        # Bootstrap confidence interval on paired difference
        ci = self._compute_bootstrap_ci([p.paired_difference for p in paired_cases])

        # Effect size computation
        effect_size = self._compute_effect_size(candidate_wins, baseline_wins, observed_difference, n)

        return StatisticalAnalysis(
            sample_size=n,
            candidate_wins=candidate_wins,
            baseline_wins=baseline_wins,
            ties=ties,
            inconsistent_cases=inconsistent_cases,
            candidate_win_rate=candidate_win_rate,
            baseline_win_rate=baseline_win_rate,
            tie_rate=tie_rate,
            consistency_rate=consistency_rate,
            observed_difference=observed_difference,
            confidence_level=self.confidence_level,
            confidence_interval=ci,
            num_bootstrap_samples=self.num_bootstrap_samples,
            random_seed=self.random_seed,
            effect_size=effect_size,
            criterion_results=criterion_results,
            paired_cases=paired_cases,
        )

    def _compute_bootstrap_ci(self, deltas: List[float]) -> Tuple[float, float]:
        """Compute percentile bootstrap confidence interval on mean paired difference.

        Resamples paired cases with replacement repeatedly.
        """
        n = len(deltas)
        if n == 0:
            return (0.0, 0.0)
        if n == 1:
            val = round(deltas[0], 4)
            return (val, val)

        # Deterministic RNG initialized with configured seed
        rng = random.Random(self.random_seed)

        bootstrap_means: List[float] = []
        for _ in range(self.num_bootstrap_samples):
            # Resample n paired observations with replacement
            resample = rng.choices(deltas, k=n)
            bootstrap_means.append(sum(resample) / n)

        bootstrap_means.sort()

        alpha = 1.0 - self.confidence_level
        lower_idx = max(0, int((alpha / 2.0) * self.num_bootstrap_samples))
        upper_idx = min(self.num_bootstrap_samples - 1, int((1.0 - alpha / 2.0) * self.num_bootstrap_samples))

        ci_lower = round(bootstrap_means[lower_idx], 4)
        ci_upper = round(bootstrap_means[upper_idx], 4)

        return (ci_lower, ci_upper)

    @staticmethod
    def _compute_effect_size(
        candidate_wins: int,
        baseline_wins: int,
        observed_difference: float,
        total_cases: int,
    ) -> EffectSize:
        """Compute descriptive effect size metrics: win-rate delta and Cohen's g for decisive pairs."""
        decisive_cases = candidate_wins + baseline_wins
        cohens_g: Optional[float] = None

        if decisive_cases > 0:
            # Proportion of decisive (non-tied, consistent) wins won by Candidate
            p_cand = candidate_wins / decisive_cases
            # Cohen's g = P - 0.5; range is [-0.5, +0.5]
            cohens_g = round(p_cand - 0.5, 4)

        if observed_difference > 0:
            direction = "Candidate advantage"
        elif observed_difference < 0:
            direction = "Baseline advantage"
        else:
            direction = "Neutral / balanced"

        interp = (
            f"{direction} (net difference: {observed_difference:+.2f} across N={total_cases} cases"
        )
        if cohens_g is not None:
            interp += f", Cohen's g={cohens_g:+.2f} over {decisive_cases} decisive cases"
        interp += ")."

        return EffectSize(
            metric_name="net_win_rate_difference",
            value=observed_difference,
            cohens_g=cohens_g,
            interpretation=interp,
        )
