from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from benizakura.models import Verdict
from benizakura.statistical import EffectSize, StatisticalAnalysis


@dataclass
class GateConfig:
    """Configurable decision policy parameters for the Benizakura release gate.

    NOTE: All default values represent Benizakura policy defaults, not universal statistical constants.
    Organizations may configure these parameters to match their risk tolerance.

    Attributes:
        min_cases: Minimum benchmark cases required to evaluate a release decision (default: 10).
        max_unstable_rate: Maximum permitted proportion of position-unstable cases before declaring
            the benchmark inconclusive due to judge bias (default: 0.20, i.e. 20%).
        regression_tolerance: Margin of acceptable negative movement or uncertainty on observed difference
            (default: 0.05, i.e. 5% net win rate drop tolerance).
        max_regression_rate: Maximum permitted proportion of cases where Candidate regressed against Baseline
            before triggering an automatic FAIL (default: 0.25, i.e. 25% of cases).
        min_effect_size: Minimum net observed advantage (candidate_win_rate - baseline_win_rate)
            required for a PASS determination (default: 0.05).
    """
    min_cases: int = 10
    max_unstable_rate: float = 0.20
    regression_tolerance: float = 0.05
    max_regression_rate: float = 0.25
    min_effect_size: float = 0.05

    def __post_init__(self) -> None:
        if self.min_cases < 1:
            raise ValueError(f"min_cases must be at least 1, got {self.min_cases}.")
        if not (0.0 <= self.max_unstable_rate <= 1.0):
            raise ValueError(f"max_unstable_rate must be between 0.0 and 1.0, got {self.max_unstable_rate}.")
        if self.regression_tolerance < 0.0:
            raise ValueError(f"regression_tolerance must be non-negative, got {self.regression_tolerance}.")
        if not (0.0 <= self.max_regression_rate <= 1.0):
            raise ValueError(f"max_regression_rate must be between 0.0 and 1.0, got {self.max_regression_rate}.")
        if self.min_effect_size < 0.0:
            raise ValueError(f"min_effect_size must be non-negative, got {self.min_effect_size}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_cases": self.min_cases,
            "max_unstable_rate": self.max_unstable_rate,
            "regression_tolerance": self.regression_tolerance,
            "max_regression_rate": self.max_regression_rate,
            "min_effect_size": self.min_effect_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GateConfig:
        return cls(
            min_cases=int(data.get("min_cases", 10)),
            max_unstable_rate=float(data.get("max_unstable_rate", 0.20)),
            regression_tolerance=float(data.get("regression_tolerance", 0.05)),
            max_regression_rate=float(data.get("max_regression_rate", 0.25)),
            min_effect_size=float(data.get("min_effect_size", 0.05)),
        )


@dataclass
class GateResult:
    """Structured, explainable evaluation release decision.

    Attributes:
        verdict: Release determination (PASS, FAIL, INCONCLUSIVE).
        reason: Plain-language explanation justifying the verdict based on empirical evidence and policy.
        observed_difference: Candidate win rate minus Baseline win rate.
        confidence_interval: Bootstrap confidence interval bounds (ci_lower, ci_upper).
        sample_size: Number of evaluated cases.
        regression_count: Number of cases where Baseline won consistently (Candidate regressed).
        regression_rate: Proportion of total cases that suffered regressions.
        regressed_cases: Explicit list of case IDs that regressed.
        unstable_count: Number of position-unstable cases.
        unstable_rate: Proportion of total cases with judge position bias.
        unstable_cases: Explicit list of case IDs with position instability.
        effect_size: EffectSize detailing practical magnitude.
        config: Snapshot of GateConfig parameters used to evaluate the gate.
        metadata: Optional dictionary for tracking execution context.
    """
    verdict: Verdict
    reason: str
    observed_difference: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    regression_count: int
    regression_rate: float
    regressed_cases: List[str]
    unstable_count: int
    unstable_rate: float
    unstable_cases: List[str]
    effect_size: EffectSize
    config: GateConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "observed_difference": self.observed_difference,
            "confidence_interval": list(self.confidence_interval),
            "sample_size": self.sample_size,
            "regression_count": self.regression_count,
            "regression_rate": self.regression_rate,
            "regressed_cases": self.regressed_cases,
            "unstable_count": self.unstable_count,
            "unstable_rate": self.unstable_rate,
            "unstable_cases": self.unstable_cases,
            "effect_size": self.effect_size.to_dict(),
            "config": self.config.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GateResult:
        ci_raw = data["confidence_interval"]
        ci: Tuple[float, float] = (float(ci_raw[0]), float(ci_raw[1]))
        return cls(
            verdict=Verdict(data["verdict"]),
            reason=data["reason"],
            observed_difference=float(data["observed_difference"]),
            confidence_interval=ci,
            sample_size=int(data["sample_size"]),
            regression_count=int(data["regression_count"]),
            regression_rate=float(data["regression_rate"]),
            regressed_cases=list(data.get("regressed_cases", [])),
            unstable_count=int(data["unstable_count"]),
            unstable_rate=float(data["unstable_rate"]),
            unstable_cases=list(data.get("unstable_cases", [])),
            effect_size=EffectSize.from_dict(data["effect_size"]),
            config=GateConfig.from_dict(data["config"]),
            metadata=data.get("metadata", {}),
        )


class ReleaseGate:
    """Evaluates statistical evidence against an explicit, configurable decision policy.

    The gate strictly consumes StatisticalAnalysis and does not perform LLM calls or raw execution.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(self, analysis: StatisticalAnalysis) -> GateResult:
        """Apply the decision policy hierarchy to a StatisticalAnalysis result."""
        sample_size = analysis.sample_size
        regressed_cases = analysis.get_regressed_cases()
        unstable_cases = analysis.get_unstable_cases()

        regression_count = len(regressed_cases)
        regression_rate = round(regression_count / sample_size, 4) if sample_size > 0 else 0.0

        unstable_count = len(unstable_cases)
        unstable_rate = round(unstable_count / sample_size, 4) if sample_size > 0 else 0.0

        ci_lower, ci_upper = analysis.confidence_interval
        observed_diff = analysis.observed_difference

        # ====================================================================
        # Decision Precedence Tree
        # ====================================================================

        # 1. Minimum sample size safeguard
        if sample_size < self.config.min_cases:
            verdict = Verdict.INCONCLUSIVE
            reason = (
                f"Sample size N={sample_size} is below the required policy minimum of "
                f"{self.config.min_cases} cases; statistical evidence is insufficient to gate release."
            )

        # 2. Excessive position instability safeguard
        elif unstable_rate > self.config.max_unstable_rate:
            verdict = Verdict.INCONCLUSIVE
            reason = (
                f"Judge position instability rate {unstable_rate:.1%} ({unstable_count} of {sample_size} cases) "
                f"exceeds maximum permitted threshold of {self.config.max_unstable_rate:.1%}; presentation bias "
                f"compromises evidence reliability."
            )

        # 3. Excessive regression rate safeguard
        elif regression_rate > self.config.max_regression_rate:
            verdict = Verdict.FAIL
            reason = (
                f"Candidate regression rate {regression_rate:.1%} ({regression_count} of {sample_size} cases) "
                f"exceeds the maximum allowed regression threshold of {self.config.max_regression_rate:.1%} "
                f"(regressed cases: {regressed_cases})."
            )

        # 4. Confident negative regression
        elif ci_upper < -self.config.regression_tolerance:
            verdict = Verdict.FAIL
            reason = (
                f"Evidence demonstrates statistically confident regression: upper bootstrap bound "
                f"{ci_upper:+.2f} is strictly below the regression tolerance limit "
                f"{-self.config.regression_tolerance:+.2f} (observed diff: {observed_diff:+.2f}, "
                f"{analysis.confidence_level:.0%} CI: [{ci_lower:+.2f}, {ci_upper:+.2f}])."
            )

        # 5. Confident positive improvement
        elif (
            observed_diff >= self.config.min_effect_size
            and ci_lower >= -self.config.regression_tolerance
            and (ci_lower >= 0.0 or observed_diff > self.config.min_effect_size * 2)
        ):
            verdict = Verdict.PASS
            reason = (
                f"Candidate demonstrates statistically supported improvement under configured policy: "
                f"observed difference {observed_diff:+.2f} meets minimum effect size {self.config.min_effect_size:+.2f}, "
                f"bootstrap CI [{ci_lower:+.2f}, {ci_upper:+.2f}] remains within regression tolerance "
                f"{-self.config.regression_tolerance:+.2f}, and regression rate {regression_rate:.1%} is acceptable."
            )

        # 6. Uncertain / overlapping boundary
        else:
            verdict = Verdict.INCONCLUSIVE
            reason = (
                f"Statistical evidence is inconclusive under configured policy: observed difference {observed_diff:+.2f} "
                f"and bootstrap {analysis.confidence_level:.0%} CI [{ci_lower:+.2f}, {ci_upper:+.2f}] overlap the decision "
                f"boundary without meeting criteria for confident PASS or definitive FAIL."
            )

        return GateResult(
            verdict=verdict,
            reason=reason,
            observed_difference=observed_diff,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=sample_size,
            regression_count=regression_count,
            regression_rate=regression_rate,
            regressed_cases=regressed_cases,
            unstable_count=unstable_count,
            unstable_rate=unstable_rate,
            unstable_cases=unstable_cases,
            effect_size=analysis.effect_size,
            config=self.config,
        )
