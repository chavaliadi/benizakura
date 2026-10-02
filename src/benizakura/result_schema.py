from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


EXPERIMENT_SCHEMA_VERSION = "1.0.0"
BENCHMARK_CANONICAL_SHA256 = "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"


class ResultStatus(str, Enum):
    """Execution and statistical status of an experiment result."""
    VALID_RESULT = "VALID_RESULT"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    INCONCLUSIVE_RESULT = "INCONCLUSIVE_RESULT"
    MISSING_RESULT = "MISSING_RESULT"


class InstabilityStatus(str, Enum):
    """Order instability classification."""
    STABLE = "STABLE"
    ORDER_UNSTABLE = "ORDER_UNSTABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


ALLOWED_NORMALIZED_WINNERS: Set[str] = {"CANDIDATE", "BASELINE", "TIE"}


@dataclass
class CaseDecision:
    """Case-level evaluation decision preserving both passes and normalized verdict."""
    case_id: str
    pass_1_decision: Optional[str] = None
    pass_2_decision: Optional[str] = None
    normalized_decision: Optional[str] = None
    is_unstable: bool = False
    instability_status: InstabilityStatus = InstabilityStatus.NOT_APPLICABLE
    execution_status: str = "SUCCESS"
    failure_status: Optional[str] = None
    criterion_assessments: Dict[str, Any] = field(default_factory=dict)
    rationale: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.instability_status, str):
            try:
                self.instability_status = InstabilityStatus(self.instability_status)
            except ValueError:
                self.instability_status = InstabilityStatus.NOT_APPLICABLE

        # Safety rule: A failed execution must NEVER be represented as a valid decision
        if self.execution_status != "SUCCESS":
            if self.normalized_decision is not None:
                raise ValueError(
                    f"Invalid CaseDecision for '{self.case_id}': execution_status is '{self.execution_status}' "
                    f"but normalized_decision is '{self.normalized_decision}'. Failed calls must never be assigned a winner or tie."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pass_1_decision": self.pass_1_decision,
            "pass_2_decision": self.pass_2_decision,
            "normalized_decision": self.normalized_decision,
            "is_unstable": self.is_unstable,
            "instability_status": self.instability_status.value if isinstance(self.instability_status, InstabilityStatus) else str(self.instability_status),
            "execution_status": self.execution_status,
            "failure_status": self.failure_status,
            "criterion_assessments": self.criterion_assessments,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaseDecision:
        return cls(
            case_id=str(data["case_id"]),
            pass_1_decision=data.get("pass_1_decision"),
            pass_2_decision=data.get("pass_2_decision"),
            normalized_decision=data.get("normalized_decision"),
            is_unstable=bool(data.get("is_unstable", False)),
            instability_status=InstabilityStatus(data.get("instability_status", "NOT_APPLICABLE"))
            if data.get("instability_status") in [e.value for e in InstabilityStatus]
            else InstabilityStatus.NOT_APPLICABLE,
            execution_status=str(data.get("execution_status", "SUCCESS")),
            failure_status=data.get("failure_status"),
            criterion_assessments=dict(data.get("criterion_assessments", {})),
            rationale=data.get("rationale"),
            evidence=data.get("evidence"),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            latency_ms=float(data["latency_ms"]) if data.get("latency_ms") is not None else None,
            token_usage=dict(data["token_usage"]) if data.get("token_usage") else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AggregateStatistics:
    """Aggregate statistics for an experiment run preserving denominators and uncertainties."""
    total_cases: int
    valid_decision_count: int
    failed_decision_count: int
    unstable_count: int
    position_dependent_error_rate: float
    candidate_win_rate: float
    baseline_win_rate: float
    tie_rate: float
    bootstrap_ci: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    regression_rate: Optional[float] = None
    false_pass_rate: Optional[float] = None
    false_regression_rate: Optional[float] = None
    kappa: Optional[float] = None
    kappa_se: Optional[float] = None
    directional_agreement: Optional[float] = None
    raw_agreement: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "valid_decision_count": self.valid_decision_count,
            "failed_decision_count": self.failed_decision_count,
            "unstable_count": self.unstable_count,
            "position_dependent_error_rate": self.position_dependent_error_rate,
            "candidate_win_rate": self.candidate_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "tie_rate": self.tie_rate,
            "bootstrap_ci": list(self.bootstrap_ci) if self.bootstrap_ci else None,
            "effect_size": self.effect_size,
            "regression_rate": self.regression_rate,
            "false_pass_rate": self.false_pass_rate,
            "false_regression_rate": self.false_regression_rate,
            "kappa": self.kappa,
            "kappa_se": self.kappa_se,
            "directional_agreement": self.directional_agreement,
            "raw_agreement": self.raw_agreement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AggregateStatistics:
        b_ci = data.get("bootstrap_ci")
        ci_tuple = tuple(b_ci) if isinstance(b_ci, (list, tuple)) and len(b_ci) == 2 else None
        return cls(
            total_cases=int(data["total_cases"]),
            valid_decision_count=int(data["valid_decision_count"]),
            failed_decision_count=int(data["failed_decision_count"]),
            unstable_count=int(data.get("unstable_count", 0)),
            position_dependent_error_rate=float(data.get("position_dependent_error_rate", 0.0)),
            candidate_win_rate=float(data.get("candidate_win_rate", 0.0)),
            baseline_win_rate=float(data.get("baseline_win_rate", 0.0)),
            tie_rate=float(data.get("tie_rate", 0.0)),
            bootstrap_ci=ci_tuple,  # type: ignore[arg-type]
            effect_size=float(data["effect_size"]) if data.get("effect_size") is not None else None,
            regression_rate=float(data["regression_rate"]) if data.get("regression_rate") is not None else None,
            false_pass_rate=float(data["false_pass_rate"]) if data.get("false_pass_rate") is not None else None,
            false_regression_rate=float(data["false_regression_rate"]) if data.get("false_regression_rate") is not None else None,
            kappa=float(data["kappa"]) if data.get("kappa") is not None else None,
            kappa_se=float(data["kappa_se"]) if data.get("kappa_se") is not None else None,
            directional_agreement=float(data["directional_agreement"]) if data.get("directional_agreement") is not None else None,
            raw_agreement=float(data["raw_agreement"]) if data.get("raw_agreement") is not None else None,
        )


@dataclass
class ExperimentResult:
    """Explicit, versioned schema for completed Benizakura experiment results."""
    experiment_id: str
    benchmark_id: str
    benchmark_version: str
    benchmark_sha256: str
    ground_truth_status: str
    ground_truth_sha256: str
    judge_provider: str
    model_snapshot: str
    model_family: str
    temperature: float
    prompt_version: str
    rubric_version: str
    experiment_variant: str
    result_status: ResultStatus
    case_decisions: List[CaseDecision]
    aggregate_statistics: AggregateStatistics
    calibration_gate_status: str = "PENDING"
    kappa: Optional[float] = None
    experiment_schema_version: str = EXPERIMENT_SCHEMA_VERSION
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    software_version: str = "benizakura-0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_schema_version": self.experiment_schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_sha256": self.benchmark_sha256,
            "ground_truth_status": self.ground_truth_status,
            "ground_truth_sha256": self.ground_truth_sha256,
            "kappa": self.kappa,
            "calibration_gate_status": self.calibration_gate_status,
            "judge_provider": self.judge_provider,
            "model_snapshot": self.model_snapshot,
            "model_family": self.model_family,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "rubric_version": self.rubric_version,
            "experiment_variant": self.experiment_variant,
            "result_status": self.result_status.value if isinstance(self.result_status, ResultStatus) else str(self.result_status),
            "case_decisions": [cd.to_dict() for cd in self.case_decisions],
            "aggregate_statistics": self.aggregate_statistics.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "software_version": self.software_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentResult:
        res_status = data.get("result_status", "VALID_RESULT")
        if res_status in [e.value for e in ResultStatus]:
            status_enum = ResultStatus(res_status)
        else:
            status_enum = ResultStatus.VALID_RESULT

        return cls(
            experiment_id=str(data["experiment_id"]),
            benchmark_id=str(data.get("benchmark_id", "conquer-benchmark-v1")),
            benchmark_version=str(data.get("benchmark_version", "1.0.0")),
            benchmark_sha256=str(data["benchmark_sha256"]),
            ground_truth_status=str(data.get("ground_truth_status", "PENDING_HUMAN_ANNOTATION")),
            ground_truth_sha256=str(data.get("ground_truth_sha256", "unverified")),
            judge_provider=str(data["judge_provider"]),
            model_snapshot=str(data["model_snapshot"]),
            model_family=str(data["model_family"]),
            temperature=float(data.get("temperature", 0.0)),
            prompt_version=str(data.get("prompt_version", "pairwise_v1")),
            rubric_version=str(data.get("rubric_version", "standard_rubric_v1")),
            experiment_variant=str(data["experiment_variant"]),
            result_status=status_enum,
            case_decisions=[CaseDecision.from_dict(cd) for cd in data.get("case_decisions", [])],
            aggregate_statistics=AggregateStatistics.from_dict(data["aggregate_statistics"]),
            calibration_gate_status=str(data.get("calibration_gate_status", "PENDING")),
            kappa=float(data["kappa"]) if data.get("kappa") is not None else None,
            experiment_schema_version=str(data.get("experiment_schema_version", EXPERIMENT_SCHEMA_VERSION)),
            started_at=str(data.get("started_at", "")),
            completed_at=data.get("completed_at"),
            software_version=str(data.get("software_version", "benizakura-0.1.0")),
            metadata=dict(data.get("metadata", {})),
        )


def validate_experiment_result(result: Union[Dict[str, Any], ExperimentResult]) -> List[str]:
    """Strictly validate an ExperimentResult structure and consistency rules.

    Checks:
    - Required schema fields present
    - Schema version matches
    - Valid ResultStatus enum
    - Denominators are mathematically consistent (valid + failed == total)
    - Failed executions NEVER have normalized_decision set (never converted to TIE)
    - Case IDs are unique
    - Benchmark SHA matches canonical hash when benchmark_id is conquer-benchmark-v1
    """
    errors: List[str] = []
    data = result.to_dict() if isinstance(result, ExperimentResult) else result

    required_fields = [
        "experiment_id",
        "experiment_schema_version",
        "benchmark_id",
        "benchmark_version",
        "benchmark_sha256",
        "ground_truth_status",
        "ground_truth_sha256",
        "judge_provider",
        "model_snapshot",
        "model_family",
        "temperature",
        "prompt_version",
        "rubric_version",
        "experiment_variant",
        "result_status",
        "case_decisions",
        "aggregate_statistics",
        "started_at",
        "software_version",
    ]

    for rf in required_fields:
        if rf not in data or data[rf] is None:
            errors.append(f"Missing required field in ExperimentResult: '{rf}'.")

    # Version check
    schema_ver = data.get("experiment_schema_version")
    if schema_ver != EXPERIMENT_SCHEMA_VERSION:
        errors.append(f"Unsupported experiment_schema_version: '{schema_ver}'. Expected: '{EXPERIMENT_SCHEMA_VERSION}'.")

    # Status check
    status_str = data.get("result_status")
    if status_str not in [e.value for e in ResultStatus]:
        errors.append(f"Invalid result_status '{status_str}'. Allowed: {[e.value for e in ResultStatus]}.")

    # Benchmark SHA check
    bench_id = data.get("benchmark_id")
    bench_sha = data.get("benchmark_sha256")
    if bench_id == "conquer-benchmark-v1" and bench_sha != BENCHMARK_CANONICAL_SHA256:
        errors.append(
            f"Canonical benchmark SHA mismatch: expected {BENCHMARK_CANONICAL_SHA256}, got {bench_sha}."
        )

    # Decisions validation
    cases = data.get("case_decisions", [])
    if not isinstance(cases, list):
        errors.append("Field 'case_decisions' must be a list.")
        cases = []

    seen_cids: Set[str] = set()
    failed_count = 0
    valid_count = 0
    unstable_count = 0

    for idx, c in enumerate(cases):
        if not isinstance(c, dict):
            errors.append(f"Case decision at index {idx} is not a dictionary.")
            continue

        cid = c.get("case_id")
        if not cid:
            errors.append(f"Case decision at index {idx} missing 'case_id'.")
        elif cid in seen_cids:
            errors.append(f"Duplicate case_id '{cid}' in case_decisions.")
        else:
            seen_cids.add(cid)

        exec_status = c.get("execution_status", "SUCCESS")
        norm_decision = c.get("normalized_decision")

        if exec_status != "SUCCESS":
            failed_count += 1
            if norm_decision is not None:
                errors.append(
                    f"Case '{cid}' has execution failure '{exec_status}' but normalized_decision '{norm_decision}'. "
                    "Failed decisions must NEVER be assigned a winner or tie."
                )
        else:
            valid_count += 1
            if norm_decision not in ALLOWED_NORMALIZED_WINNERS:
                errors.append(
                    f"Case '{cid}' has valid execution but unrecognized normalized_decision '{norm_decision}'. "
                    f"Allowed: {sorted(ALLOWED_NORMALIZED_WINNERS)}."
                )

        if c.get("is_unstable"):
            unstable_count += 1

    # Aggregate statistics consistency
    stats = data.get("aggregate_statistics", {})
    if isinstance(stats, dict):
        tot = stats.get("total_cases", 0)
        v_cnt = stats.get("valid_decision_count", 0)
        f_cnt = stats.get("failed_decision_count", 0)

        if tot != len(cases):
            errors.append(f"Aggregate total_cases ({tot}) does not match case_decisions length ({len(cases)}).")
        if v_cnt != valid_count:
            errors.append(f"Aggregate valid_decision_count ({v_cnt}) does not match observed valid cases ({valid_count}).")
        if f_cnt != failed_count:
            errors.append(f"Aggregate failed_decision_count ({f_cnt}) does not match observed failed cases ({failed_count}).")
        if v_cnt + f_cnt != tot:
            errors.append(f"Denominator inconsistency: valid ({v_cnt}) + failed ({f_cnt}) != total ({tot}).")

        # Result status consistency
        if status_str == ResultStatus.VALID_RESULT.value:
            if failed_count > 0 and valid_count == 0:
                errors.append("Result status is VALID_RESULT but all case decisions failed.")
            if tot == 0:
                errors.append("Result status is VALID_RESULT but case count is zero.")
        elif status_str == ResultStatus.EXECUTION_FAILURE.value:
            if failed_count == 0 and valid_count > 0:
                errors.append("Result status is EXECUTION_FAILURE but all cases succeeded.")

    return errors
