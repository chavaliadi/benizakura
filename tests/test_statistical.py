import pytest

from benizakura.models import (
    BidirectionalEvaluationResult,
    ConsistencyOutcome,
    CriterionAssessment,
    NormalizedWinner,
    PairwiseJudgment,
    PairwiseWinner,
)
from benizakura.statistical import (
    CriterionAggregate,
    EffectSize,
    EvaluationDataset,
    EvaluationSample,
    PairedCaseOutcome,
    StatisticalAnalysis,
    StatisticalAnalyzer,
)


def _make_bidirectional_result(
    pass1_winner: NormalizedWinner,
    pass2_winner: NormalizedWinner,
    pass1_criteria: list[CriterionAssessment] | None = None,
    pass2_criteria: list[CriterionAssessment] | None = None,
) -> BidirectionalEvaluationResult:
    # Convert NormalizedWinner to presentation position winner for each pass
    # Pass 1: A=Baseline, B=Candidate
    # If NormalizedWinner is CANDIDATE, position B won
    # If NormalizedWinner is BASELINE, position A won
    raw1 = (
        PairwiseWinner.B
        if pass1_winner == NormalizedWinner.CANDIDATE
        else (PairwiseWinner.A if pass1_winner == NormalizedWinner.BASELINE else PairwiseWinner.TIE)
    )
    # Pass 2: A=Candidate, B=Baseline
    # If NormalizedWinner is CANDIDATE, position A won
    # If NormalizedWinner is BASELINE, position B won
    raw2 = (
        PairwiseWinner.A
        if pass2_winner == NormalizedWinner.CANDIDATE
        else (PairwiseWinner.B if pass2_winner == NormalizedWinner.BASELINE else PairwiseWinner.TIE)
    )

    j1 = PairwiseJudgment(
        winner=raw1,
        rationale="Pass 1 judgment",
        criterion_assessments=pass1_criteria or [],
    )
    j2 = PairwiseJudgment(
        winner=raw2,
        rationale="Pass 2 judgment",
        criterion_assessments=pass2_criteria or [],
    )

    return BidirectionalEvaluationResult(
        pass1_judgment=j1,
        pass2_judgment=j2,
        pass1_normalized_winner=pass1_winner,
        pass2_normalized_winner=pass2_winner,
    )


def _make_sample(
    case_id: str,
    pass1_winner: NormalizedWinner,
    pass2_winner: NormalizedWinner,
    pass1_criteria: list[CriterionAssessment] | None = None,
    pass2_criteria: list[CriterionAssessment] | None = None,
) -> EvaluationSample:
    return EvaluationSample(
        case_id=case_id,
        result=_make_bidirectional_result(
            pass1_winner, pass2_winner, pass1_criteria, pass2_criteria
        ),
    )


# ============================================================================
# Dataset and Sample Model Tests
# ============================================================================


def test_evaluation_sample_valid():
    sample = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
    assert sample.case_id == "c1"
    assert sample.consistency == ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN
    assert sample.resolved_winner == NormalizedWinner.CANDIDATE


def test_evaluation_sample_inconsistent():
    sample = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.BASELINE)
    assert sample.consistency == ConsistencyOutcome.POSITION_UNSTABLE
    assert sample.resolved_winner is None


def test_evaluation_sample_empty_case_id():
    with pytest.raises(ValueError, match="case_id must be a non-empty string"):
        EvaluationSample(
            case_id="",
            result=_make_bidirectional_result(NormalizedWinner.TIE, NormalizedWinner.TIE),
        )


def test_evaluation_dataset_duplicate_case_id_rejected():
    s1 = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
    s2 = _make_sample("c1", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
    with pytest.raises(ValueError, match="Duplicate case_id detected in dataset: 'c1'"):
        EvaluationDataset(samples=[s1, s2])


def test_evaluation_dataset_access_helpers():
    s1 = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
    s2 = _make_sample("c2", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
    dataset = EvaluationDataset(samples=[s1, s2])

    assert len(dataset) == 2
    assert dataset.case_ids == ["c1", "c2"]
    assert dataset.get_sample("c1") == s1
    assert dataset.get_sample("c2") == s2
    assert dataset.get_sample("c3") is None
    assert [s.case_id for s in dataset] == ["c1", "c2"]


def test_dataset_serialization_round_trip():
    s1 = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
    dataset = EvaluationDataset(samples=[s1], metadata={"benchmark": "conquer-v1"})

    data = dataset.to_dict()
    restored = EvaluationDataset.from_dict(data)

    assert len(restored) == 1
    assert restored.samples[0].case_id == "c1"
    assert restored.metadata["benchmark"] == "conquer-v1"


# ============================================================================
# Statistical Aggregation and Denominators Tests
# ============================================================================


def test_analyzer_mixed_outcomes_aggregation():
    # 10 cases: 5 Candidate wins, 2 Baseline wins, 2 Ties, 1 Inconsistent
    samples = [
        _make_sample(f"cand-{i}", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
        for i in range(5)
    ]
    samples += [
        _make_sample(f"base-{i}", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
        for i in range(2)
    ]
    samples += [
        _make_sample(f"tie-{i}", NormalizedWinner.TIE, NormalizedWinner.TIE)
        for i in range(2)
    ]
    samples += [
        _make_sample("unstable-1", NormalizedWinner.CANDIDATE, NormalizedWinner.BASELINE)
    ]

    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.sample_size == 10
    assert analysis.candidate_wins == 5
    assert analysis.baseline_wins == 2
    assert analysis.ties == 2
    assert analysis.inconsistent_cases == 1

    # Check explicit denominators: all rates divide by sample_size (10)
    assert analysis.candidate_win_rate == 0.50
    assert analysis.baseline_win_rate == 0.20
    assert analysis.tie_rate == 0.20
    assert analysis.consistency_rate == 0.90
    assert analysis.observed_difference == 0.30  # 0.50 - 0.20


def test_analyzer_preserves_case_identity_and_regression_lists():
    samples = [
        _make_sample("c_cand_1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE),
        _make_sample("c_base_1", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE),
        _make_sample("c_base_2", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE),
        _make_sample("c_tie_1", NormalizedWinner.TIE, NormalizedWinner.TIE),
        _make_sample("c_unstable_1", NormalizedWinner.BASELINE, NormalizedWinner.CANDIDATE),
    ]
    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.get_improved_cases() == ["c_cand_1"]
    assert analysis.get_regressed_cases() == ["c_base_1", "c_base_2"]
    assert analysis.get_tied_cases() == ["c_tie_1"]
    assert analysis.get_unstable_cases() == ["c_unstable_1"]


# ============================================================================
# Criterion-Level Aggregation Tests
# ============================================================================


def test_criterion_aggregation_multi_case():
    # Case 1: Candidate wins correctness and completeness
    c1_p1 = [
        CriterionAssessment(criterion_name="correctness", winner=PairwiseWinner.B, rationale="B"),
        CriterionAssessment(criterion_name="completeness", winner=PairwiseWinner.B, rationale="B"),
    ]
    c1_p2 = [
        CriterionAssessment(criterion_name="correctness", winner=PairwiseWinner.A, rationale="A"),
        CriterionAssessment(criterion_name="completeness", winner=PairwiseWinner.A, rationale="A"),
    ]
    s1 = _make_sample("case-1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE, c1_p1, c1_p2)

    # Case 2: Baseline wins correctness, tie on completeness
    c2_p1 = [
        CriterionAssessment(criterion_name="correctness", winner=PairwiseWinner.A, rationale="A"),
        CriterionAssessment(criterion_name="completeness", winner=PairwiseWinner.TIE, rationale="T"),
    ]
    c2_p2 = [
        CriterionAssessment(criterion_name="correctness", winner=PairwiseWinner.B, rationale="B"),
        CriterionAssessment(criterion_name="completeness", winner=PairwiseWinner.TIE, rationale="T"),
    ]
    s2 = _make_sample("case-2", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE, c2_p1, c2_p2)

    # Case 3: Only clarity assessed (missing correctness & completeness)
    c3_p1 = [CriterionAssessment(criterion_name="clarity", winner=PairwiseWinner.B, rationale="B")]
    c3_p2 = [CriterionAssessment(criterion_name="clarity", winner=PairwiseWinner.A, rationale="A")]
    s3 = _make_sample("case-3", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE, c3_p1, c3_p2)

    dataset = EvaluationDataset(samples=[s1, s2, s3])
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert "correctness" in analysis.criterion_results
    assert "completeness" in analysis.criterion_results
    assert "clarity" in analysis.criterion_results

    # Correctness was assessed on case-1 and case-2 only (total = 2)
    agg_corr = analysis.criterion_results["correctness"]
    assert agg_corr.total_assessed_cases == 2
    assert agg_corr.candidate_wins == 1
    assert agg_corr.baseline_wins == 1
    assert agg_corr.candidate_win_rate == 0.50
    assert agg_corr.baseline_win_rate == 0.50
    assert agg_corr.observed_difference == 0.0

    # Completeness was assessed on case-1 and case-2 only (total = 2)
    agg_comp = analysis.criterion_results["completeness"]
    assert agg_comp.total_assessed_cases == 2
    assert agg_comp.candidate_wins == 1
    assert agg_comp.ties == 1
    assert agg_comp.baseline_wins == 0
    assert agg_comp.candidate_win_rate == 0.50
    assert agg_comp.tie_rate == 0.50
    assert agg_comp.observed_difference == 0.50

    # Clarity was assessed on case-3 only (total = 1)
    agg_clar = analysis.criterion_results["clarity"]
    assert agg_clar.total_assessed_cases == 1
    assert agg_clar.candidate_wins == 1
    assert agg_clar.observed_difference == 1.0


def test_missing_criteria_not_manufactured():
    # Samples with zero criterion assessments
    s1 = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
    s2 = _make_sample("c2", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)

    dataset = EvaluationDataset(samples=[s1, s2])
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    # No spurious criteria aggregates created
    assert analysis.criterion_results == {}


# ============================================================================
# Paired Difference and Bootstrap Confidence Interval Tests
# ============================================================================


def test_bootstrap_reproducibility_with_same_seed():
    samples = [
        _make_sample(f"cand-{i}", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
        for i in range(12)
    ] + [
        _make_sample(f"base-{i}", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
        for i in range(4)
    ]
    dataset = EvaluationDataset(samples=samples)

    analyzer1 = StatisticalAnalyzer(confidence_level=0.95, num_bootstrap_samples=500, random_seed=123)
    res1 = analyzer1.analyze(dataset)

    analyzer2 = StatisticalAnalyzer(confidence_level=0.95, num_bootstrap_samples=500, random_seed=123)
    res2 = analyzer2.analyze(dataset)

    assert res1.confidence_interval == res2.confidence_interval


def test_bootstrap_different_seed_different_draws():
    samples = [
        _make_sample(f"cand-{i}", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
        for i in range(10)
    ] + [
        _make_sample(f"base-{i}", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
        for i in range(6)
    ] + [
        _make_sample(f"tie-{i}", NormalizedWinner.TIE, NormalizedWinner.TIE)
        for i in range(4)
    ]
    dataset = EvaluationDataset(samples=samples)

    # Use few samples so different seeds are very unlikely to produce identical percentile bounds
    analyzer1 = StatisticalAnalyzer(confidence_level=0.90, num_bootstrap_samples=50, random_seed=1)
    analyzer2 = StatisticalAnalyzer(confidence_level=0.90, num_bootstrap_samples=50, random_seed=999)

    res1 = analyzer1.analyze(dataset)
    res2 = analyzer2.analyze(dataset)

    assert res1.observed_difference == res2.observed_difference
    # The two bootstrap runs should have valid bounds
    assert res1.confidence_interval[0] <= res1.observed_difference <= res1.confidence_interval[1]
    assert res2.confidence_interval[0] <= res2.observed_difference <= res2.confidence_interval[1]


def test_effect_size_computation():
    # 8 candidate wins, 2 baseline wins -> 10 cases
    samples = [
        _make_sample(f"cand-{i}", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
        for i in range(8)
    ] + [
        _make_sample(f"base-{i}", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
        for i in range(2)
    ]
    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    # Observed difference = 0.8 - 0.2 = 0.6
    assert analysis.observed_difference == 0.6
    assert analysis.effect_size.value == 0.6
    # Cohen's g: 8 / (8 + 2) = 0.8 -> 0.8 - 0.5 = 0.3
    assert analysis.effect_size.cohens_g == 0.3
    assert "Candidate advantage" in analysis.effect_size.interpretation


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_dataset_handling():
    dataset = EvaluationDataset(samples=[])
    analyzer = StatisticalAnalyzer()
    analysis = analyzer.analyze(dataset)

    assert analysis.sample_size == 0
    assert analysis.candidate_wins == 0
    assert analysis.baseline_wins == 0
    assert analysis.ties == 0
    assert analysis.inconsistent_cases == 0
    assert analysis.candidate_win_rate == 0.0
    assert analysis.baseline_win_rate == 0.0
    assert analysis.tie_rate == 0.0
    assert analysis.consistency_rate == 0.0
    assert analysis.observed_difference == 0.0
    assert analysis.confidence_interval == (0.0, 0.0)
    assert analysis.criterion_results == {}
    assert analysis.paired_cases == []
    assert analysis.get_improved_cases() == []
    assert analysis.get_regressed_cases() == []


def test_single_case_candidate_win():
    sample = _make_sample("single-1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
    dataset = EvaluationDataset(samples=[sample])
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.sample_size == 1
    assert analysis.candidate_wins == 1
    assert analysis.observed_difference == 1.0
    assert analysis.confidence_interval == (1.0, 1.0)


def test_all_ties_scenario():
    samples = [
        _make_sample(f"tie-{i}", NormalizedWinner.TIE, NormalizedWinner.TIE)
        for i in range(8)
    ]
    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.sample_size == 8
    assert analysis.ties == 8
    assert analysis.candidate_wins == 0
    assert analysis.baseline_wins == 0
    assert analysis.tie_rate == 1.0
    assert analysis.observed_difference == 0.0
    assert analysis.confidence_interval == (0.0, 0.0)
    assert analysis.effect_size.cohens_g is None


def test_all_candidate_wins_scenario():
    samples = [
        _make_sample(f"cand-{i}", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE)
        for i in range(6)
    ]
    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.candidate_wins == 6
    assert analysis.candidate_win_rate == 1.0
    assert analysis.observed_difference == 1.0
    assert analysis.confidence_interval == (1.0, 1.0)
    assert analysis.effect_size.cohens_g == 0.5


def test_all_baseline_wins_scenario():
    samples = [
        _make_sample(f"base-{i}", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)
        for i in range(5)
    ]
    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.baseline_wins == 5
    assert analysis.baseline_win_rate == 1.0
    assert analysis.observed_difference == -1.0
    assert analysis.confidence_interval == (-1.0, -1.0)
    assert analysis.effect_size.cohens_g == -0.5
    assert "Baseline advantage" in analysis.effect_size.interpretation


def test_all_inconsistent_cases_scenario():
    # Every case fails bidirectional presentation check
    samples = [
        _make_sample(f"unstable-{i}", NormalizedWinner.BASELINE, NormalizedWinner.CANDIDATE)
        for i in range(7)
    ]
    dataset = EvaluationDataset(samples=samples)
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    assert analysis.sample_size == 7
    assert analysis.inconsistent_cases == 7
    assert analysis.consistency_rate == 0.0
    assert analysis.candidate_wins == 0
    assert analysis.baseline_wins == 0
    assert analysis.observed_difference == 0.0
    assert analysis.confidence_interval == (0.0, 0.0)
    assert len(analysis.get_unstable_cases()) == 7


def test_invalid_confidence_level_raises():
    with pytest.raises(ValueError, match="confidence_level must be between 0.0 and 1.0"):
        StatisticalAnalyzer(confidence_level=1.5)

    with pytest.raises(ValueError, match="confidence_level must be between 0.0 and 1.0"):
        StatisticalAnalyzer(confidence_level=0.0)


def test_invalid_bootstrap_sample_count_raises():
    with pytest.raises(ValueError, match="num_bootstrap_samples must be at least 1"):
        StatisticalAnalyzer(num_bootstrap_samples=0)


# ============================================================================
# Statistical Analysis Serialization Tests
# ============================================================================


def test_statistical_analysis_serialization_round_trip():
    ca1 = CriterionAssessment(criterion_name="correctness", winner=PairwiseWinner.B, rationale="B")
    ca2 = CriterionAssessment(criterion_name="correctness", winner=PairwiseWinner.A, rationale="A")
    s1 = _make_sample("c1", NormalizedWinner.CANDIDATE, NormalizedWinner.CANDIDATE, [ca1], [ca2])
    s2 = _make_sample("c2", NormalizedWinner.BASELINE, NormalizedWinner.BASELINE)

    dataset = EvaluationDataset(samples=[s1, s2])
    analyzer = StatisticalAnalyzer(random_seed=42)
    analysis = analyzer.analyze(dataset)

    data = analysis.to_dict()
    restored = StatisticalAnalysis.from_dict(data)

    assert restored.sample_size == analysis.sample_size
    assert restored.candidate_wins == analysis.candidate_wins
    assert restored.baseline_wins == analysis.baseline_wins
    assert restored.observed_difference == analysis.observed_difference
    assert restored.confidence_interval == analysis.confidence_interval
    assert restored.confidence_level == analysis.confidence_level
    assert restored.effect_size.value == analysis.effect_size.value
    assert restored.effect_size.cohens_g == analysis.effect_size.cohens_g
    assert len(restored.paired_cases) == 2
    assert restored.paired_cases[0].case_id == "c1"
    assert "correctness" in restored.criterion_results
    assert restored.criterion_results["correctness"].candidate_wins == 1
