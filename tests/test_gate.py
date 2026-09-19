from __future__ import annotations

import pytest

from benizakura.gate import GateConfig, GateResult, ReleaseGate
from benizakura.models import ConsistencyOutcome, Verdict
from benizakura.statistical import EffectSize, PairedCaseOutcome, StatisticalAnalysis


def _make_analysis(
    sample_size: int = 20,
    candidate_wins: int = 14,
    baseline_wins: int = 2,
    ties: int = 4,
    inconsistent_cases: int = 0,
    observed_difference: float = 0.60,
    confidence_interval: tuple[float, float] = (0.35, 0.85),
    confidence_level: float = 0.95,
) -> StatisticalAnalysis:
    """Helper to build a deterministic StatisticalAnalysis for testing."""
    paired_cases: list[PairedCaseOutcome] = []
    idx = 1

    for _ in range(candidate_wins):
        paired_cases.append(
            PairedCaseOutcome(
                case_id=f"case-{idx:02d}",
                outcome=ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN,
                paired_difference=1.0,
            )
        )
        idx += 1

    for _ in range(baseline_wins):
        paired_cases.append(
            PairedCaseOutcome(
                case_id=f"case-{idx:02d}",
                outcome=ConsistencyOutcome.CONSISTENT_BASELINE_WIN,
                paired_difference=-1.0,
            )
        )
        idx += 1

    for _ in range(ties):
        paired_cases.append(
            PairedCaseOutcome(
                case_id=f"case-{idx:02d}",
                outcome=ConsistencyOutcome.CONSISTENT_TIE,
                paired_difference=0.0,
            )
        )
        idx += 1

    for _ in range(inconsistent_cases):
        paired_cases.append(
            PairedCaseOutcome(
                case_id=f"case-{idx:02d}",
                outcome=ConsistencyOutcome.POSITION_UNSTABLE,
                paired_difference=0.0,
            )
        )
        idx += 1

    return StatisticalAnalysis(
        sample_size=sample_size,
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        inconsistent_cases=inconsistent_cases,
        candidate_win_rate=candidate_wins / sample_size if sample_size > 0 else 0.0,
        baseline_win_rate=baseline_wins / sample_size if sample_size > 0 else 0.0,
        tie_rate=ties / sample_size if sample_size > 0 else 0.0,
        consistency_rate=(candidate_wins + baseline_wins + ties) / sample_size if sample_size > 0 else 0.0,
        observed_difference=observed_difference,
        confidence_level=confidence_level,
        confidence_interval=confidence_interval,
        num_bootstrap_samples=1000,
        random_seed=42,
        effect_size=EffectSize(
            metric_name="net_win_rate_difference",
            value=observed_difference,
            cohens_g=0.35,
            interpretation="Substantial improvement",
        ),
        paired_cases=paired_cases,
    )


class TestGateConfig:
    def test_default_config(self) -> None:
        cfg = GateConfig()
        assert cfg.min_cases == 10
        assert cfg.max_unstable_rate == 0.20
        assert cfg.regression_tolerance == 0.05
        assert cfg.max_regression_rate == 0.25
        assert cfg.min_effect_size == 0.05

    def test_custom_valid_config(self) -> None:
        cfg = GateConfig(
            min_cases=30,
            max_unstable_rate=0.10,
            regression_tolerance=0.02,
            max_regression_rate=0.15,
            min_effect_size=0.08,
        )
        assert cfg.min_cases == 30
        assert cfg.max_unstable_rate == 0.10
        assert cfg.regression_tolerance == 0.02
        assert cfg.max_regression_rate == 0.15
        assert cfg.min_effect_size == 0.08

    def test_invalid_min_cases(self) -> None:
        with pytest.raises(ValueError, match="min_cases must be at least 1"):
            GateConfig(min_cases=0)

    def test_invalid_max_unstable_rate(self) -> None:
        with pytest.raises(ValueError, match="max_unstable_rate must be between 0.0 and 1.0"):
            GateConfig(max_unstable_rate=1.5)
        with pytest.raises(ValueError, match="max_unstable_rate must be between 0.0 and 1.0"):
            GateConfig(max_unstable_rate=-0.1)

    def test_invalid_regression_tolerance(self) -> None:
        with pytest.raises(ValueError, match="regression_tolerance must be non-negative"):
            GateConfig(regression_tolerance=-0.05)

    def test_invalid_max_regression_rate(self) -> None:
        with pytest.raises(ValueError, match="max_regression_rate must be between 0.0 and 1.0"):
            GateConfig(max_regression_rate=1.2)

    def test_invalid_min_effect_size(self) -> None:
        with pytest.raises(ValueError, match="min_effect_size must be non-negative"):
            GateConfig(min_effect_size=-0.01)

    def test_serialization_round_trip(self) -> None:
        cfg = GateConfig(
            min_cases=25,
            max_unstable_rate=0.15,
            regression_tolerance=0.03,
            max_regression_rate=0.20,
            min_effect_size=0.10,
        )
        d = cfg.to_dict()
        restored = GateConfig.from_dict(d)
        assert restored == cfg


class TestReleaseGateDecisions:
    def test_pass_on_strong_positive_evidence(self) -> None:
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=15,
            baseline_wins=2,
            ties=3,
            inconsistent_cases=0,
            observed_difference=0.65,
            confidence_interval=(0.40, 0.85),
        )
        gate = ReleaseGate(GateConfig())
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.PASS
        assert "Candidate demonstrates statistically supported improvement" in result.reason
        assert result.observed_difference == 0.65
        assert result.confidence_interval == (0.40, 0.85)
        assert result.regression_count == 2
        assert result.regression_rate == 0.10
        assert result.unstable_count == 0
        assert result.unstable_rate == 0.0

    def test_fail_on_confident_negative_regression(self) -> None:
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=4,
            baseline_wins=10,
            ties=6,
            inconsistent_cases=0,
            observed_difference=-0.30,
            confidence_interval=(-0.55, -0.08),
        )
        # CI upper is -0.08, which is < -regression_tolerance (-0.05)
        gate = ReleaseGate(GateConfig(max_regression_rate=0.60))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.FAIL
        assert "upper bootstrap bound" in result.reason
        assert result.observed_difference == -0.30
        assert result.confidence_interval == (-0.55, -0.08)

    def test_fail_on_excessive_regression_rate(self) -> None:
        # 7 of 20 cases regressed = 35% regression rate > max_regression_rate (25%)
        # even if observed difference is mildly positive
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=10,
            baseline_wins=7,
            ties=3,
            inconsistent_cases=0,
            observed_difference=0.15,
            confidence_interval=(-0.10, 0.40),
        )
        gate = ReleaseGate(GateConfig(max_regression_rate=0.25))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.FAIL
        assert "regression rate 35.0%" in result.reason
        assert result.regression_count == 7
        assert len(result.regressed_cases) == 7

    def test_inconclusive_on_insufficient_sample_size(self) -> None:
        analysis = _make_analysis(
            sample_size=5,
            candidate_wins=5,
            baseline_wins=0,
            ties=0,
            inconsistent_cases=0,
            observed_difference=1.0,
            confidence_interval=(0.8, 1.0),
        )
        gate = ReleaseGate(GateConfig(min_cases=10))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.INCONCLUSIVE
        assert "below the required policy minimum" in result.reason
        assert result.sample_size == 5

    def test_inconclusive_on_excessive_position_instability(self) -> None:
        # 5 of 20 cases unstable = 25% > max_unstable_rate (20%)
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=12,
            baseline_wins=2,
            ties=1,
            inconsistent_cases=5,
            observed_difference=0.50,
            confidence_interval=(0.20, 0.70),
        )
        gate = ReleaseGate(GateConfig(max_unstable_rate=0.20))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.INCONCLUSIVE
        assert "position instability rate 25.0%" in result.reason
        assert result.unstable_count == 5
        assert len(result.unstable_cases) == 5

    def test_inconclusive_on_uncertain_boundary_overlap(self) -> None:
        # Sample size OK, instability OK, regression rate OK (1/20 = 5%),
        # but observed difference is tiny (0.02 < 0.05 min_effect_size)
        # and CI spans from -0.15 to +0.20
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=5,
            baseline_wins=4,
            ties=11,
            inconsistent_cases=0,
            observed_difference=0.05,
            confidence_interval=(-0.10, 0.20),
        )
        gate = ReleaseGate(GateConfig(min_effect_size=0.08))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.INCONCLUSIVE
        assert "inconclusive under configured policy" in result.reason


class TestRegressionSafeguards:
    def test_zero_regressions_passes(self) -> None:
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=16,
            baseline_wins=0,
            ties=4,
            inconsistent_cases=0,
            observed_difference=0.80,
            confidence_interval=(0.60, 0.95),
        )
        gate = ReleaseGate(GateConfig())
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.PASS
        assert result.regression_count == 0
        assert result.regression_rate == 0.0
        assert result.regressed_cases == []

    def test_single_regression_permitted_under_threshold(self) -> None:
        # 1 of 20 = 5% <= 25% max_regression_rate
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=15,
            baseline_wins=1,
            ties=4,
            inconsistent_cases=0,
            observed_difference=0.70,
            confidence_interval=(0.45, 0.90),
        )
        gate = ReleaseGate(GateConfig())
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.PASS
        assert result.regression_count == 1
        assert result.regression_rate == 0.05
        assert len(result.regressed_cases) == 1

    def test_several_regressions_permitted_under_threshold(self) -> None:
        # 4 of 20 = 20% <= 25% max_regression_rate
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=14,
            baseline_wins=4,
            ties=2,
            inconsistent_cases=0,
            observed_difference=0.50,
            confidence_interval=(0.25, 0.75),
        )
        gate = ReleaseGate(GateConfig(max_regression_rate=0.25))
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.PASS
        assert result.regression_count == 4
        assert result.regression_rate == 0.20
        assert len(result.regressed_cases) == 4

    def test_high_regression_rate_fails_even_with_positive_average(self) -> None:
        # 8 of 20 = 40% > 25% max_regression_rate
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=10,
            baseline_wins=8,
            ties=2,
            inconsistent_cases=0,
            observed_difference=0.10,
            confidence_interval=(-0.10, 0.35),
        )
        gate = ReleaseGate(GateConfig(max_regression_rate=0.25))
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.FAIL
        assert result.regression_count == 8
        assert result.regression_rate == 0.40
        assert "regression rate 40.0%" in result.reason


class TestDecisionPrecedence:
    def test_sample_size_precedes_instability_and_failure(self) -> None:
        # Even if regression rate is 100%, if sample_size < min_cases, it must be INCONCLUSIVE
        analysis = _make_analysis(
            sample_size=4,
            candidate_wins=0,
            baseline_wins=4,
            ties=0,
            inconsistent_cases=0,
            observed_difference=-1.0,
            confidence_interval=(-1.0, -1.0),
        )
        gate = ReleaseGate(GateConfig(min_cases=10))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.INCONCLUSIVE
        assert "below the required policy minimum" in result.reason

    def test_instability_precedes_regression_failure(self) -> None:
        # If unstable rate is excessive, judge bias invalidates evidence before failing on regression
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=2,
            baseline_wins=8,
            ties=2,
            inconsistent_cases=8,  # 8/20 = 40% > 20%
            observed_difference=-0.30,
            confidence_interval=(-0.60, -0.05),
        )
        gate = ReleaseGate(GateConfig(max_unstable_rate=0.20, max_regression_rate=0.25))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.INCONCLUSIVE
        assert "position instability rate 40.0%" in result.reason

    def test_regression_rate_failure_precedes_positive_ci(self) -> None:
        # Candidate has +0.30 observed difference and positive CI,
        # but 6 of 20 cases (30%) regressed against baseline, exceeding max_regression_rate 25%
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=12,
            baseline_wins=6,
            ties=2,
            inconsistent_cases=0,
            observed_difference=0.30,
            confidence_interval=(0.05, 0.55),
        )
        gate = ReleaseGate(GateConfig(max_regression_rate=0.25))
        result = gate.evaluate(analysis)

        assert result.verdict == Verdict.FAIL
        assert "regression rate 30.0%" in result.reason


class TestBoundaryConditions:
    def test_exact_minimum_sample_size(self) -> None:
        analysis = _make_analysis(
            sample_size=10,
            candidate_wins=8,
            baseline_wins=1,
            ties=1,
            inconsistent_cases=0,
            observed_difference=0.70,
            confidence_interval=(0.40, 0.90),
        )
        gate = ReleaseGate(GateConfig(min_cases=10))
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.PASS

    def test_one_below_minimum_sample_size(self) -> None:
        analysis = _make_analysis(
            sample_size=9,
            candidate_wins=8,
            baseline_wins=1,
            ties=0,
            inconsistent_cases=0,
            observed_difference=0.77,
            confidence_interval=(0.40, 0.95),
        )
        gate = ReleaseGate(GateConfig(min_cases=10))
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.INCONCLUSIVE

    def test_exact_maximum_unstable_rate(self) -> None:
        # 4 of 20 = exactly 20.0%, max is 0.20 (<= max_unstable_rate is permitted)
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=13,
            baseline_wins=2,
            ties=1,
            inconsistent_cases=4,
            observed_difference=0.55,
            confidence_interval=(0.25, 0.80),
        )
        gate = ReleaseGate(GateConfig(max_unstable_rate=0.20))
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.PASS

    def test_exact_maximum_regression_rate(self) -> None:
        # 5 of 20 = exactly 25.0%, max is 0.25 (<= max_regression_rate is permitted)
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=13,
            baseline_wins=5,
            ties=2,
            inconsistent_cases=0,
            observed_difference=0.40,
            confidence_interval=(0.10, 0.65),
        )
        gate = ReleaseGate(GateConfig(max_regression_rate=0.25))
        result = gate.evaluate(analysis)
        assert result.verdict == Verdict.PASS

    def test_exact_regression_tolerance_boundary(self) -> None:
        # ci_upper == -0.05, regression_tolerance == 0.05 (-tolerance is -0.05)
        # ci_upper < -tolerance is False (-0.05 < -0.05 is False), so does not trigger rule 4
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=5,
            baseline_wins=5,
            ties=10,
            inconsistent_cases=0,
            observed_difference=-0.15,
            confidence_interval=(-0.30, -0.05),
        )
        gate = ReleaseGate(GateConfig(regression_tolerance=0.05, max_regression_rate=0.50))
        result = gate.evaluate(analysis)
        # Boundary not strictly exceeded, falls into INCONCLUSIVE
        assert result.verdict == Verdict.INCONCLUSIVE


class TestGateResultSerializationAndExplainability:
    def test_result_round_trip_serialization(self) -> None:
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=14,
            baseline_wins=3,
            ties=3,
            inconsistent_cases=0,
            observed_difference=0.55,
            confidence_interval=(0.30, 0.75),
        )
        gate = ReleaseGate(GateConfig())
        result = gate.evaluate(analysis)

        d = result.to_dict()
        assert d["verdict"] == "PASS"
        assert d["sample_size"] == 20
        assert d["regression_count"] == 3
        assert d["regressed_cases"] == ["case-15", "case-16", "case-17"]

        restored = GateResult.from_dict(d)
        assert restored.verdict == result.verdict
        assert restored.reason == result.reason
        assert restored.observed_difference == result.observed_difference
        assert restored.confidence_interval == result.confidence_interval
        assert restored.sample_size == result.sample_size
        assert restored.regression_count == result.regression_count
        assert restored.regressed_cases == result.regressed_cases
        assert restored.config.min_cases == result.config.min_cases

    def test_explainability_content(self) -> None:
        analysis = _make_analysis(
            sample_size=20,
            candidate_wins=2,
            baseline_wins=12,
            ties=6,
            inconsistent_cases=0,
            observed_difference=-0.50,
            confidence_interval=(-0.75, -0.25),
        )
        gate = ReleaseGate(GateConfig())
        result = gate.evaluate(analysis)

        # Confirm non-empty informative reason containing concrete numbers
        assert "12 of 20" in result.reason
        assert "case-03" in result.reason or len(result.regressed_cases) == 12


class TestPureStatisticalIndependence:
    def test_no_provider_or_network_needed(self) -> None:
        """Verify that the ReleaseGate operates purely on StatisticalAnalysis without any judge or network."""
        analysis = _make_analysis()
        gate = ReleaseGate()
        result = gate.evaluate(analysis)
        assert isinstance(result, GateResult)
        assert result.verdict in {Verdict.PASS, Verdict.FAIL, Verdict.INCONCLUSIVE}
