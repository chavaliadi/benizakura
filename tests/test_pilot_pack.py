from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import (
    REQUIRED_TOTAL_CASES,
    compute_benchmark_hash,
    load_benchmark_data,
)
from benizakura.calibration import (
    GroundTruthArtifact,
    GroundTruthStatus,
    PILOT_V1_CASE_IDS,
    STANDARD_CRITERIA_NAMES,
    calculate_human_agreement,
    freeze_ground_truth_artifact,
    load_annotator_submissions,
    validate_annotator_submission,
    validate_pilot_submissions_batch,
    validate_submissions_batch,
)


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")
CANONICAL_BENCHMARK_SHA = "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"

PILOT_DIR = Path("evals/conquer/annotations/pilot_v1")
PILOT_TASKS_PATH = PILOT_DIR / "pilot_tasks.json"
PILOT_TEMPLATE_PATH = PILOT_DIR / "submission_template.json"
PILOT_README_PATH = PILOT_DIR / "README.md"
PILOT_GUIDE_PATH = PILOT_DIR / "ANNOTATOR_GUIDE.md"
PILOT_SCHEMA_PATH = PILOT_DIR / "ANNOTATION_SCHEMA.md"
PILOT_SELECTION_PATH = PILOT_DIR / "SELECTION_AND_SCOPE.md"
BLINDING_KEY_PATH = Path("evals/conquer/annotations/blinding_key_v1.json")


def _create_valid_pilot_submission(case_id: str, annotator_id: str = "human-001", winner: str = "A") -> dict:
    """Helper to generate a schema-valid pilot submission for unit tests."""
    return {
        "case_id": case_id,
        "annotator_id": annotator_id,
        "presentation_order": "randomized",
        "winner": winner,
        "confidence": 0.90,
        "rationale": "Clear technical rationale that comfortably exceeds the fifteen character requirement.",
        "criterion_assessments": [
            {
                "criterion_name": crit,
                "winner": winner,
                "rationale": f"Valid justification for criterion {crit} exceeding minimum chars.",
                "evidence": None,
            }
            for crit in STANDARD_CRITERIA_NAMES
        ],
        "timestamp": "2026-10-11T12:00:00Z",
        "status": "SUBMITTED",
    }


# ============================================================================
# 1. Pilot Pack Selection & Integrity Tests
# ============================================================================

def test_pilot_pack_has_exactly_15_unique_cases():
    assert PILOT_TASKS_PATH.is_file(), f"Missing {PILOT_TASKS_PATH}"
    with open(PILOT_TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    assert len(tasks) == 15
    case_ids = [t["case_id"] for t in tasks]
    assert len(set(case_ids)) == 15
    assert set(case_ids) == PILOT_V1_CASE_IDS


def test_all_pilot_cases_exist_in_official_benchmark():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    bench_cases = {c["case_id"]: c for c in bench_data["cases"]}

    with open(PILOT_TASKS_PATH, "r", encoding="utf-8") as f:
        pilot_tasks = json.load(f)

    for pt in pilot_tasks:
        cid = pt["case_id"]
        assert cid in bench_cases, f"Pilot case {cid} not found in official benchmark"
        bc = bench_cases[cid]
        assert pt["category"] == bc["category"]
        assert pt["question"] == bc["question"]
        assert pt["subcategory"] == bc["subcategory"]


def test_official_benchmark_remains_unchanged():
    actual_hash = compute_benchmark_hash(BENCHMARK_PATH)
    assert actual_hash == CANONICAL_BENCHMARK_SHA, "Canonical benchmark SHA-256 has been modified!"


def test_selected_cases_retain_original_case_ids():
    with open(PILOT_TASKS_PATH, "r", encoding="utf-8") as f:
        pilot_tasks = json.load(f)

    for pt in pilot_tasks:
        assert pt["case_id"] in PILOT_V1_CASE_IDS
        # Verify naming prefix matches category
        prefix = pt["case_id"].split("-")[0]
        expected_prefixes = {"dsa", "sys", "backend", "frontend", "behavioral"}
        assert prefix in expected_prefixes


def test_category_and_bias_probe_coverage():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    bench_cases = {c["case_id"]: c for c in bench_data["cases"]}

    categories = {}
    probes_present = set()

    for cid in PILOT_V1_CASE_IDS:
        c = bench_cases[cid]
        cat = c["category"]
        categories[cat] = categories.get(cat, 0) + 1
        for p in c.get("bias_probes", []):
            probes_present.add(p)

    # Stratified 50% representation across all 5 categories
    assert categories["DSA"] == 3
    assert categories["SYSTEM_DESIGN"] == 4
    assert categories["BACKEND"] == 3
    assert categories["FRONTEND"] == 2
    assert categories["BEHAVIORAL"] == 3
    assert sum(categories.values()) == 15

    # Required probes must be represented
    for required_probe in ["verbosity", "hallucination", "near_tie", "alternative_architecture", "criterion_tradeoff"]:
        assert required_probe in probes_present, f"Missing required probe: {required_probe}"


# ============================================================================
# 2. Template and Schema Validation Tests
# ============================================================================

def test_submission_template_contains_no_fabricated_judgments():
    assert PILOT_TEMPLATE_PATH.is_file(), f"Missing {PILOT_TEMPLATE_PATH}"
    with open(PILOT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    assert len(template) == 15
    for item in template:
        assert item["case_id"] in PILOT_V1_CASE_IDS
        # Check no judgment is pre-filled
        assert item["winner"] == "", "Winner must not be pre-filled in template"
        assert item["rationale"] == "", "Rationale must not be pre-filled in template"
        assert item["confidence"] == 0.0, "Confidence must not be pre-filled with a score"
        assert item["timestamp"] == "YYYY-MM-DDTHH:MM:SSZ", "Timestamp must be uncompleted placeholder"
        assert item["annotator_id"] != "human-001", "Annotator ID must be an obvious placeholder"

        assert len(item["criterion_assessments"]) == 6
        for crit in item["criterion_assessments"]:
            assert crit["criterion_name"] in STANDARD_CRITERIA_NAMES
            assert crit["winner"] == "", "Criterion winner must not be pre-filled"
            assert crit["rationale"] == "", "Criterion rationale must not be pre-filled"
            assert crit["evidence"] is None


def test_submission_template_passes_validation_when_completed():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    with open(PILOT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    completed_submissions = []
    for item in template:
        filled = dict(item)
        filled["annotator_id"] = "human-001"
        filled["winner"] = "A"
        filled["confidence"] = 0.85
        filled["timestamp"] = "2026-10-12T10:00:00Z"
        filled["rationale"] = "Valid detailed justification exceeding length threshold."
        filled_criteria = []
        for crit in filled["criterion_assessments"]:
            c = dict(crit)
            c["winner"] = "A"
            c["rationale"] = "Valid technical assessment rationale."
            filled_criteria.append(c)
        filled["criterion_assessments"] = filled_criteria
        completed_submissions.append(filled)

    is_valid, errors = validate_pilot_submissions_batch(completed_submissions, cases)
    assert is_valid, f"Validation failed with errors: {errors}"
    assert len(errors) == 0


def test_pilot_validation_rejects_missing_cases():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    # Provide only 14 cases instead of 15
    subs = [_create_valid_pilot_submission(cid) for cid in sorted(PILOT_V1_CASE_IDS)[:-1]]
    is_valid, errors = validate_pilot_submissions_batch(subs, cases)
    assert not is_valid
    assert any("missing 1 pilot cases" in err for err in errors)


def test_pilot_validation_rejects_non_pilot_cases():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    # Include dsa-001 (which is in the benchmark but NOT in pilot_v1)
    subs = [_create_valid_pilot_submission(cid) for cid in sorted(PILOT_V1_CASE_IDS)[:-1]]
    subs.append(_create_valid_pilot_submission("dsa-001"))

    is_valid, errors = validate_pilot_submissions_batch(subs, cases)
    assert not is_valid
    assert any("Case is not part of the pilot subset" in err for err in errors)


def test_both_annotators_can_use_same_task_file():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    cases = bench_data["cases"]

    subs_ann1 = [_create_valid_pilot_submission(cid, annotator_id="human-001", winner="A") for cid in PILOT_V1_CASE_IDS]
    subs_ann2 = [_create_valid_pilot_submission(cid, annotator_id="human-002", winner="B") for cid in PILOT_V1_CASE_IDS]

    # Validate Ann 1 batch
    ok1, err1 = validate_pilot_submissions_batch(subs_ann1, cases)
    assert ok1 and len(err1) == 0

    # Validate Ann 2 batch
    ok2, err2 = validate_pilot_submissions_batch(subs_ann2, cases)
    assert ok2 and len(err2) == 0

    # Validate combined batch
    ok_combined, err_combined = validate_pilot_submissions_batch(subs_ann1 + subs_ann2, cases)
    assert ok_combined and len(err_combined) == 0


# ============================================================================
# 3. Blinding, Isolation & Security Tests
# ============================================================================

def test_blinding_and_secret_checks_pass():
    shareable_files = [
        PILOT_TASKS_PATH,
        PILOT_TEMPLATE_PATH,
        PILOT_README_PATH,
        PILOT_GUIDE_PATH,
        PILOT_SCHEMA_PATH,
    ]

    forbidden_strings = [
        "blinding_key",
        "baseline_answer",
        "candidate_answer",
        "intended_signal",
        "bias_probes",
        "expected_winner",
        "claude-3-5",
        "gpt-4o",
        "sk-ant",
        "sk-proj",
    ]

    for p in shareable_files:
        assert p.is_file(), f"Missing shareable file {p}"
        content = p.read_text(encoding="utf-8")
        for forbidden in forbidden_strings:
            assert forbidden not in content, f"Forbidden string '{forbidden}' leaked in shareable file {p}"

    # Verify blinding key is NOT in pilot directory
    assert not (PILOT_DIR / "blinding_key_v1.json").exists()
    assert not (PILOT_DIR / "blinding_key.json").exists()


def test_readme_links_are_relative_and_resolve():
    import re

    assert PILOT_README_PATH.is_file()
    content = PILOT_README_PATH.read_text(encoding="utf-8")

    # Ensure no absolute file:/// or /Users paths
    assert "file:///" not in content, "README contains absolute file:/// links"
    assert "/Users/" not in content, "README contains machine-specific /Users/ paths"

    # Find all Markdown links
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
    assert len(links) > 0, "No markdown links found in README"

    for text, target in links:
        if target.startswith("http"):
            continue
        resolved_path = PILOT_DIR / target
        assert resolved_path.is_file(), f"Relative link '{target}' in README does not resolve to an existing file"


def test_owner_only_selection_document_is_distinct_and_documented():
    assert PILOT_SELECTION_PATH.is_file()
    content = PILOT_SELECTION_PATH.read_text(encoding="utf-8")
    assert "OWNER-FACING RESEARCH RECORD" in content
    assert "DO NOT DISTRIBUTE TO HUMAN ANNOTATORS" in content
    assert CANONICAL_BENCHMARK_SHA in content


# ============================================================================
# 4. Gating Safeguard Enforcement Tests
# ============================================================================

def test_pilot_submissions_cannot_become_official_ground_truth(tmp_path):
    # Construct a GroundTruthArtifact with 15 pilot cases
    gt_cases = [
        {
            "case_id": cid,
            "status": "CONSENSED",
            "consensus": {"winner": "CANDIDATE", "agreement_type": "UNANIMOUS"},
        }
        for cid in PILOT_V1_CASE_IDS
    ]

    gt_artifact = GroundTruthArtifact(
        benchmark_id="conquer-benchmark-v1",
        benchmark_version="1.0.0",
        benchmark_sha256=CANONICAL_BENCHMARK_SHA,
        schema_version="1.0.0",
        status="ADJUDICATION_COMPLETE",
        case_count=15,
        human_labels="PENDING",
        agreement_statistics={"cohens_kappa": 0.90},
        cases=gt_cases,
        created_at="2026-10-11T12:00:00Z",
    )

    out_gt = tmp_path / "ground_truth_v1.json"
    with pytest.raises(ValueError, match=f"Ground truth has 15 cases; expected exactly {REQUIRED_TOTAL_CASES}"):
        freeze_ground_truth_artifact(gt_artifact, BENCHMARK_PATH, out_gt)


def test_pilot_agreement_cannot_unlock_phase_4():
    bench_data = load_benchmark_data(BENCHMARK_PATH)
    with open(BLINDING_KEY_PATH, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # 15 cases evaluated with 100% agreement
    subs_1 = [
        load_annotator_submissions([_create_valid_pilot_submission(cid, annotator_id="human-001", winner="A")])[0]
        for cid in PILOT_V1_CASE_IDS
    ]
    subs_2 = [
        load_annotator_submissions([_create_valid_pilot_submission(cid, annotator_id="human-002", winner="A")])[0]
        for cid in PILOT_V1_CASE_IDS
    ]

    stats = calculate_human_agreement(subs_1, subs_2, key_data, bench_data)

    # Must be marked INCOMPLETE and cannot pass calibration gate
    assert stats.total_cases == 15
    assert stats.calibration_gate == "INCOMPLETE"
    assert "Incomplete evaluations: only 15/30" in stats.gate_reason
    assert stats.calibration_gate != "PASS"
