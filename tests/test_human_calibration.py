from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import (
    REQUIRED_CATEGORY_COUNTS,
    REQUIRED_TOTAL_CASES,
    compute_benchmark_hash,
    generate_blinded_annotation_tasks,
    load_benchmark_data,
)
from benizakura.calibration import (
    AdjudicationRecord,
    AgreementStatistics,
    AnnotatorSubmission,
    ConfusionMatrix,
    CriterionAssessmentData,
    GroundTruthArtifact,
    GroundTruthStatus,
    STANDARD_CRITERIA_NAMES,
    apply_adjudications,
    calculate_human_agreement,
    compute_cohens_kappa,
    create_adjudication_item,
    create_pending_ground_truth_artifact,
    freeze_ground_truth_artifact,
    load_annotator_submissions,
    validate_annotator_submission,
    validate_submissions_batch,
)


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")
BLINDING_KEY_PATH = Path("evals/conquer/annotations/blinding_key_v1.json")
GROUND_TRUTH_PATH = Path("evals/conquer/ground_truth_v1.json")
EXPECTED_BENCHMARK_SHA = "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"


def _create_synthetic_criteria(winner: str = "A") -> list[dict]:
    """Helper to generate a complete set of 6 standard criterion assessments for testing."""
    return [
        {
            "criterion_name": crit,
            "winner": winner,
            "rationale": f"Synthetic rationale assessing {crit.lower()}.",
            "evidence": "Quoted excerpt from candidate response.",
        }
        for crit in STANDARD_CRITERIA_NAMES
    ]


def _create_synthetic_submission(
    case_id: str = "dsa-001",
    annotator_id: str = "human-001",
    winner: str = "A",
    confidence: float = 0.95,
) -> dict:
    """Helper to generate a valid synthetic annotator submission dictionary for unit tests."""
    return {
        "case_id": case_id,
        "annotator_id": annotator_id,
        "presentation_order": "randomized",
        "winner": winner,
        "confidence": confidence,
        "rationale": "Detailed technical justification explaining the algorithmic choice.",
        "criterion_assessments": _create_synthetic_criteria(winner=winner),
        "timestamp": "2026-09-27T10:00:00Z",
        "status": "SUBMITTED",
    }


# ============================================================================
# 1. Annotation Validation Tests
# ============================================================================

def test_valid_submission_accepted():
    valid_sub = _create_synthetic_submission()
    errors = validate_annotator_submission(valid_sub, valid_case_ids={"dsa-001"})
    assert len(errors) == 0


def test_validation_rejects_unknown_case_id():
    sub = _create_synthetic_submission(case_id="unknown-999")
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("Unknown case_id" in err for err in errors)


def test_validation_rejects_invalid_winner():
    sub = _create_synthetic_submission(winner="INVALID")
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("Invalid winner value" in err for err in errors)


def test_validation_rejects_out_of_bounds_confidence():
    sub = _create_synthetic_submission(confidence=1.5)
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("Confidence value 1.5 out of range" in err for err in errors)


def test_validation_rejects_too_brief_rationale():
    sub = _create_synthetic_submission()
    sub["rationale"] = "too short"
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("Rationale too brief" in err for err in errors)


def test_validation_rejects_missing_criteria():
    sub = _create_synthetic_submission()
    # Remove one required criterion
    sub["criterion_assessments"] = sub["criterion_assessments"][:-1]
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("Missing required criteria" in err for err in errors)


def test_validation_rejects_duplicate_criteria():
    sub = _create_synthetic_submission()
    sub["criterion_assessments"].append(sub["criterion_assessments"][0])
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("Duplicate assessment for criterion" in err for err in errors)


def test_validation_rejects_pii_in_annotator_id():
    sub = _create_synthetic_submission(annotator_id="engineer@company.com")
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("contains email address or PII" in err for err in errors)


def test_validation_rejects_forbidden_metadata_leakage():
    sub = _create_synthetic_submission()
    sub["intended_signal"] = "candidate_improvement"
    errors = validate_annotator_submission(sub, valid_case_ids={"dsa-001"})
    assert any("forbidden ground-truth metadata key" in err for err in errors)


def test_batch_validation_rejects_duplicate_submission():
    bench_cases = [{"case_id": "dsa-001"}, {"case_id": "dsa-002"}]
    sub1 = _create_synthetic_submission(case_id="dsa-001", annotator_id="human-001")
    sub2 = _create_synthetic_submission(case_id="dsa-001", annotator_id="human-001")  # duplicate!
    is_valid, errors = validate_submissions_batch([sub1, sub2], bench_cases)
    assert not is_valid
    assert any("Duplicate submission detected" in err for err in errors)


# ============================================================================
# 2. Blinding Tests
# ============================================================================

def test_blinding_hides_identities_and_metadata():
    tasks, key_map = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=42)
    assert len(tasks) == REQUIRED_TOTAL_CASES

    for task in tasks:
        assert "response_a" in task
        assert "response_b" in task
        assert "baseline_answer" not in task
        assert "candidate_answer" not in task
        assert "intended_signal" not in task
        assert "bias_probes" not in task
        assert "expected_winner" not in task
        assert "human_annotation" not in task


def test_blinding_deterministic_with_same_seed():
    tasks_1, key_1 = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=999)
    tasks_2, key_2 = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=999)
    assert key_1 == key_2
    assert tasks_1 == tasks_2


def test_blinding_different_seeds_produce_different_mappings():
    _, key_1 = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=1)
    _, key_2 = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=2)
    # At least some cases must have flipped order across seeds
    diff_count = sum(
        1 for cid in key_1["mappings"]
        if key_1["mappings"][cid]["swapped"] != key_2["mappings"][cid]["swapped"]
    )
    assert diff_count > 0, "Different seeds should produce different swap mappings"


# ============================================================================
# 3. Cohen's Kappa & Agreement Statistics Tests
# ============================================================================

def test_cohens_kappa_perfect_agreement():
    r1 = ["CANDIDATE", "BASELINE", "TIE", "CANDIDATE", "BASELINE"]
    r2 = ["CANDIDATE", "BASELINE", "TIE", "CANDIDATE", "BASELINE"]
    kappa, se, ci, conf_mat = compute_cohens_kappa(r1, r2)
    assert kappa == 1.0
    assert conf_mat.total_observations == 5


def test_cohens_kappa_partial_agreement():
    r1 = ["CANDIDATE", "CANDIDATE", "BASELINE", "TIE", "TIE", "BASELINE"]
    r2 = ["CANDIDATE", "BASELINE", "BASELINE", "TIE", "CANDIDATE", "BASELINE"]
    kappa, se, ci, conf_mat = compute_cohens_kappa(r1, r2)
    assert 0.0 < kappa < 1.0


def test_cohens_kappa_complete_disagreement():
    r1 = ["CANDIDATE", "CANDIDATE", "BASELINE", "BASELINE"]
    r2 = ["BASELINE", "BASELINE", "CANDIDATE", "CANDIDATE"]
    kappa, se, ci, conf_mat = compute_cohens_kappa(r1, r2)
    assert kappa < 0.0, "Complete disagreement should yield negative kappa"


def test_cohens_kappa_three_classes_handled():
    classes = ["CANDIDATE", "BASELINE", "TIE"]
    r1 = ["CANDIDATE"] * 10 + ["BASELINE"] * 10 + ["TIE"] * 10
    r2 = ["CANDIDATE"] * 8 + ["BASELINE"] * 2 + ["BASELINE"] * 8 + ["TIE"] * 2 + ["TIE"] * 8 + ["CANDIDATE"] * 2
    kappa, se, ci, conf_mat = compute_cohens_kappa(r1, r2, classes=classes)
    assert kappa > 0.60
    assert conf_mat.classes == classes


def test_human_agreement_pipeline_with_synthetic_submissions():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]
    with open(BLINDING_KEY_PATH, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # Annotator 1 and Annotator 2 both evaluate all 30 cases with 24 agreements and 6 disagreements
    subs_1: list[AnnotatorSubmission] = []
    subs_2: list[AnnotatorSubmission] = []

    for idx, c in enumerate(cases):
        cid = c["case_id"]
        # In case 0..23, both pick A
        # In case 24..29, Ann1 picks A and Ann2 picks B (6 disagreements)
        w1 = "A"
        w2 = "A" if idx < 24 else "B"

        s1_dict = _create_synthetic_submission(case_id=cid, annotator_id="human-001", winner=w1)
        s2_dict = _create_synthetic_submission(case_id=cid, annotator_id="human-002", winner=w2)

        subs_1.append(AnnotatorSubmission.from_dict(s1_dict))
        subs_2.append(AnnotatorSubmission.from_dict(s2_dict))

    stats = calculate_human_agreement(subs_1, subs_2, key_data, bench_data)

    assert stats.total_cases == 30
    assert stats.raw_agreement_count == 24
    assert stats.disagreement_count == 6
    assert pytest.approx(stats.raw_agreement, 0.01) == 0.80
    assert stats.cohens_kappa > 0.50
    assert len(stats.category_agreement) == 5
    assert len(stats.criterion_agreement) == 6

    # Verify summary string formats cleanly
    summary_str = stats.summary()
    assert "Human Annotation Agreement" in summary_str
    assert "Confusion Matrix" in summary_str
    assert "DSA" in summary_str


def test_calibration_kill_gate_fail_logic():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]
    with open(BLINDING_KEY_PATH, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # Induce massive disagreement (kappa < 0.60)
    subs_1: list[AnnotatorSubmission] = []
    subs_2: list[AnnotatorSubmission] = []

    for idx, c in enumerate(cases):
        cid = c["case_id"]
        w1 = "A" if idx % 2 == 0 else "B"
        w2 = "B" if idx % 2 == 0 else "A"

        subs_1.append(AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id=cid, annotator_id="human-001", winner=w1)))
        subs_2.append(AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id=cid, annotator_id="human-002", winner=w2)))

    stats = calculate_human_agreement(subs_1, subs_2, key_data, bench_data, threshold=0.60)
    assert stats.calibration_gate == "FAIL"
    assert "HUMAN CALIBRATION FAILED" in stats.gate_reason
    assert len(stats.recommendations) > 0


# ============================================================================
# 4. Adjudication & Consensus Tests
# ============================================================================

def test_consensus_creates_adjudication_item_for_disagreements():
    sub1 = AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id="dsa-001", annotator_id="human-001", winner="A"))
    sub2 = AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id="dsa-001", annotator_id="human-002", winner="B"))
    key_case = {"response_a": "BASELINE", "response_b": "CANDIDATE", "swapped": False}

    item = create_adjudication_item("dsa-001", sub1, sub2, key_case)
    assert item["case_id"] == "dsa-001"
    assert item["annotator_1"]["normalized_winner"] == "BASELINE"
    assert item["annotator_2"]["normalized_winner"] == "CANDIDATE"
    assert item["adjudicator_decision"] is None


def test_adjudication_resolution_creates_adjudicated_consensus():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    with open(BLINDING_KEY_PATH, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # 1 case agreement, 1 case disagreement
    s1_a = AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id="dsa-001", annotator_id="human-001", winner="A"))
    s2_a = AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id="dsa-001", annotator_id="human-002", winner="A"))

    s1_b = AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id="dsa-002", annotator_id="human-001", winner="A"))
    s2_b = AnnotatorSubmission.from_dict(_create_synthetic_submission(case_id="dsa-002", annotator_id="human-002", winner="B"))

    adjudication = AdjudicationRecord(
        case_id="dsa-002",
        adjudicator_id="lead-human-000",
        adjudicator_decision="CANDIDATE",
        adjudicator_rationale="Annotator 2 correctly identified that candidate implements O(1) eviction.",
    )

    mini_bench = {"cases": [bench_data["cases"][0], bench_data["cases"][1]]}
    consensus_cases, pending = apply_adjudications(
        [s1_a, s1_b],
        [s2_a, s2_b],
        key_data,
        mini_bench,
        adjudication_records=[adjudication],
    )

    assert len(pending) == 0
    assert consensus_cases[0]["status"] == "CONSENSED"
    assert consensus_cases[0]["consensus"]["agreement_type"] == "UNANIMOUS"
    assert consensus_cases[1]["status"] == "ADJUDICATED"
    assert consensus_cases[1]["consensus"]["agreement_type"] == "ADJUDICATED"
    assert consensus_cases[1]["consensus"]["adjudicator_id"] == "lead-human-000"


def test_missing_adjudication_blocks_freezing(tmp_path):
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = list(bench_data["cases"])

    # Build a ground-truth artifact with 1 pending case
    gt_cases = []
    for c in cases:
        gt_cases.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "status": "CONSENSED" if c["case_id"] != "dsa-006" else "SUBMITTED",
            "consensus": {"winner": "CANDIDATE"} if c["case_id"] != "dsa-006" else None,
        })

    gt_artifact = GroundTruthArtifact(
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256=EXPECTED_BENCHMARK_SHA,
        schema_version="1.0.0",
        status="HUMAN_AGREEMENT_MEASURED",
        case_count=30,
        human_labels="PENDING",
        agreement_statistics=None,
        cases=gt_cases,
        created_at="2026-09-27T10:00:00Z",
    )

    out_gt = tmp_path / "ground_truth_test.json"
    with pytest.raises(ValueError, match="Cannot freeze ground truth while cases are incomplete"):
        freeze_ground_truth_artifact(gt_artifact, BENCHMARK_PATH, out_gt)


# ============================================================================
# 5. Ground Truth Artifact Integrity & Freezing Tests
# ============================================================================

def test_pending_ground_truth_matches_benchmark_hash():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    assert gt_data["benchmark_id"] == "conquer-benchmark-v1"
    assert gt_data["benchmark_sha256"] == EXPECTED_BENCHMARK_SHA
    assert gt_data["status"] == GroundTruthStatus.PENDING_HUMAN_ANNOTATION
    assert gt_data["human_labels"] == "PENDING"
    assert gt_data["case_count"] == REQUIRED_TOTAL_CASES
    assert len(gt_data["cases"]) == REQUIRED_TOTAL_CASES


def test_ground_truth_freezing_succeeds_when_fully_consensed(tmp_path):
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    # All 30 cases have consensus
    gt_cases = []
    for c in cases:
        gt_cases.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "status": "CONSENSED",
            "annotator_1": {"winner": "A", "normalized_winner": "CANDIDATE"},
            "annotator_2": {"winner": "A", "normalized_winner": "CANDIDATE"},
            "consensus": {"winner": "CANDIDATE", "agreement_type": "UNANIMOUS"},
        })

    gt_artifact = GroundTruthArtifact(
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256=EXPECTED_BENCHMARK_SHA,
        schema_version="1.0.0",
        status="ADJUDICATION_COMPLETE",
        case_count=30,
        human_labels="PENDING",
        agreement_statistics={"cohens_kappa": 0.85},
        cases=gt_cases,
        created_at="2026-09-27T10:00:00Z",
    )

    out_gt = tmp_path / "ground_truth_v1.json"
    out_manifest = tmp_path / "ground_truth_v1.manifest.json"
    out_sha = tmp_path / "ground_truth_v1.sha256"

    frozen_artifact, sha = freeze_ground_truth_artifact(
        gt_artifact,
        BENCHMARK_PATH,
        out_json=out_gt,
        out_manifest=out_manifest,
        out_sha256=out_sha,
    )

    assert frozen_artifact.status == GroundTruthStatus.GROUND_TRUTH_FROZEN
    assert out_gt.is_file()
    assert out_manifest.is_file()
    assert out_sha.is_file()

    with open(out_sha, "r", encoding="utf-8") as f:
        sha_file_content = f.read()
    assert sha in sha_file_content


def test_freezing_rejects_altered_benchmark_hash(tmp_path):
    gt_artifact = GroundTruthArtifact(
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256="altered_hash_00000000000000000000000000000000000000000000000000000000",
        schema_version="1.0.0",
        status="ADJUDICATION_COMPLETE",
        case_count=30,
        human_labels="PENDING",
        agreement_statistics=None,
        cases=[{"case_id": f"c-{i}", "status": "CONSENSED", "consensus": {"winner": "TIE"}} for i in range(30)],
        created_at="2026-09-27T10:00:00Z",
    )

    out_gt = tmp_path / "gt.json"
    with pytest.raises(ValueError, match="Ground truth references benchmark hash"):
        freeze_ground_truth_artifact(gt_artifact, BENCHMARK_PATH, out_gt)


def test_benchmark_hash_immutability_remains_exact():
    """Verify that benchmark_v1.json has NOT been modified during this milestone."""
    actual_hash = compute_benchmark_hash(BENCHMARK_PATH)
    assert actual_hash == EXPECTED_BENCHMARK_SHA, (
        f"CRITICAL: Benchmark hash mismatch! Expected {EXPECTED_BENCHMARK_SHA}, got {actual_hash}"
    )
