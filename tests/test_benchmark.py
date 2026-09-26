from __future__ import annotations

import json
from pathlib import Path
import pytest

from benizakura.benchmark import (
    REQUIRED_CATEGORY_COUNTS,
    REQUIRED_TOTAL_CASES,
    BenchmarkCase,
    audit_evaluation_case_leakage,
    compute_benchmark_hash,
    generate_blinded_annotation_tasks,
    load_benchmark_data,
    unblind_human_judgment,
    validate_benchmark,
)


BENCHMARK_PATH = Path("evals/conquer/benchmark_v1.json")
MANIFEST_PATH = Path("evals/conquer/benchmark_v1.manifest.json")
CHECKSUM_PATH = Path("evals/conquer/benchmark_v1.sha256")


def test_official_benchmark_exists_and_loads():
    assert BENCHMARK_PATH.is_file(), f"Benchmark file missing at {BENCHMARK_PATH}"
    data = load_benchmark_data(BENCHMARK_PATH)
    assert isinstance(data, dict)
    assert "cases" in data
    assert len(data["cases"]) == REQUIRED_TOTAL_CASES


def test_official_benchmark_validation_report():
    report = validate_benchmark(BENCHMARK_PATH)
    assert report.is_valid, f"Benchmark failed validation with errors:\n{report.summary()}"
    assert report.case_count == REQUIRED_TOTAL_CASES
    assert report.leakage_detected is False
    assert len(report.errors) == 0

    # Verify exact category counts
    for category, expected in REQUIRED_CATEGORY_COUNTS.items():
        assert report.category_counts[category] == expected

    # Verify that all 30 cases are probed for position bias
    assert report.probes_detected["position"] == REQUIRED_TOTAL_CASES


def test_official_benchmark_all_cases_pending():
    data = load_benchmark_data(BENCHMARK_PATH)
    for case in data["cases"]:
        annotation = case["human_annotation"]
        assert annotation["status"] == "PENDING"
        assert annotation["consensus"] is None
        assert annotation["annotator_1"] is None
        assert annotation["annotator_2"] is None


def test_manifest_consistency_and_checksum():
    assert MANIFEST_PATH.is_file(), "Manifest file missing"
    assert CHECKSUM_PATH.is_file(), "Checksum file missing"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["benchmark_id"] == "conquer-benchmark-v1"
    assert manifest["case_count"] == REQUIRED_TOTAL_CASES
    assert manifest["status"] == "FROZEN_PENDING_HUMAN_ANNOTATION"
    assert manifest["human_labels"] == "PENDING"

    computed_hash = compute_benchmark_hash(BENCHMARK_PATH)
    assert manifest["sha256"] == computed_hash

    with open(CHECKSUM_PATH, "r", encoding="utf-8") as f:
        checksum_content = f.read().strip()
    assert computed_hash in checksum_content


def test_benchmark_case_ids_unique_and_formatted():
    data = load_benchmark_data(BENCHMARK_PATH)
    seen_ids = set()
    category_prefixes = {
        "DSA": "dsa-",
        "SYSTEM_DESIGN": "sys-",
        "BACKEND": "backend-",
        "FRONTEND": "frontend-",
        "BEHAVIORAL": "behavioral-",
    }

    for case in data["cases"]:
        case_id = case["case_id"]
        assert case_id not in seen_ids, f"Duplicate ID: {case_id}"
        seen_ids.add(case_id)

        cat = case["category"]
        expected_prefix = category_prefixes[cat]
        assert case_id.startswith(expected_prefix), (
            f"Case '{case_id}' in category '{cat}' does not start with '{expected_prefix}'"
        )


def test_evaluation_case_leakage_safeguard():
    data = load_benchmark_data(BENCHMARK_PATH)
    for case_dict in data["cases"]:
        case = BenchmarkCase.from_dict(case_dict)
        eval_case = case.to_evaluation_case()

        # Audit that no forbidden metadata exists
        issues = audit_evaluation_case_leakage(eval_case)
        assert len(issues) == 0, f"Leakage detected in {case.case_id}: {issues}"

        # Confirm fields
        assert eval_case.id == case.case_id
        assert eval_case.topic == case.category
        assert eval_case.candidate_answer == case.candidate_answer
        assert "intended_signal" not in eval_case.metadata
        assert "human_annotation" not in eval_case.metadata
        assert "winner" not in eval_case.metadata


def test_validator_rejects_malformed_case_count():
    data = load_benchmark_data(BENCHMARK_PATH)
    # Remove one case
    truncated_data = {"cases": data["cases"][:-1]}
    report = validate_benchmark(truncated_data)
    assert not report.is_valid
    assert any("Expected exactly 30 cases" in err for err in report.errors)


def test_validator_rejects_duplicate_case_id():
    data = load_benchmark_data(BENCHMARK_PATH)
    cases = list(data["cases"])
    cases[1] = dict(cases[1])
    cases[1]["case_id"] = cases[0]["case_id"]
    bad_data = {"cases": cases}
    report = validate_benchmark(bad_data)
    assert not report.is_valid
    assert any("Duplicate case_id" in err for err in report.errors)


def test_validator_rejects_empty_answer():
    data = load_benchmark_data(BENCHMARK_PATH)
    cases = list(data["cases"])
    cases[0] = dict(cases[0])
    cases[0]["candidate_answer"] = "   "
    bad_data = {"cases": cases}
    report = validate_benchmark(bad_data)
    assert not report.is_valid
    assert any("empty or invalid 'candidate_answer'" in err for err in report.errors)


def test_validator_rejects_premature_consensus_when_pending():
    data = load_benchmark_data(BENCHMARK_PATH)
    cases = list(data["cases"])
    cases[0] = dict(cases[0])
    cases[0]["human_annotation"] = {
        "status": "PENDING",
        "consensus": {"winner": "CANDIDATE"},
    }
    bad_data = {"cases": cases}
    report = validate_benchmark(bad_data)
    assert not report.is_valid
    assert any("illegal premature label" in err for err in report.errors)


def test_validator_detects_metadata_leakage():
    data = load_benchmark_data(BENCHMARK_PATH)
    cases = list(data["cases"])
    cases[0] = dict(cases[0])
    # Deliberately inject ground truth into metadata
    cases[0]["metadata"] = {"intended_signal": "candidate_improvement", "winner": "CANDIDATE"}
    # BenchmarkCase.to_evaluation_case() strips metadata, but let's test if audit flags it when present in eval_case
    case_obj = BenchmarkCase.from_dict(cases[0])
    eval_case = case_obj.to_evaluation_case()
    eval_case.metadata["winner"] = "CANDIDATE"

    issues = audit_evaluation_case_leakage(eval_case)
    assert len(issues) > 0
    assert any("forbidden ground-truth key 'winner'" in issue for issue in issues)


def test_deterministic_blinding_and_unblinding_roundtrip():
    tasks, key_map = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=12345)
    assert len(tasks) == REQUIRED_TOTAL_CASES
    assert len(key_map["mappings"]) == REQUIRED_TOTAL_CASES

    # Verify deterministic repeatability with same seed
    tasks_2, key_map_2 = generate_blinded_annotation_tasks(BENCHMARK_PATH, seed=12345)
    assert key_map == key_map_2

    # Check unblinding logic for every task
    for task in tasks:
        cid = task["case_id"]
        key_case = key_map["mappings"][cid]

        # Blind choice A
        unblinded_a = unblind_human_judgment("A", key_case)
        assert unblinded_a in ("BASELINE", "CANDIDATE")
        assert unblinded_a == key_case["response_a"]

        # Blind choice B
        unblinded_b = unblind_human_judgment("B", key_case)
        assert unblinded_b in ("BASELINE", "CANDIDATE")
        assert unblinded_b == key_case["response_b"]

        # Blind choice TIE
        assert unblind_human_judgment("TIE", key_case) == "TIE"
