from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from benizakura.benchmark import (
    REQUIRED_CATEGORY_COUNTS,
    REQUIRED_TOTAL_CASES,
    compute_benchmark_hash,
    load_benchmark_data,
    unblind_human_judgment,
)


STANDARD_CRITERIA_NAMES: Set[str] = {
    "CORRECTNESS",
    "RELEVANCE",
    "COMPLETENESS",
    "TECHNICAL_DEPTH",
    "CLARITY",
    "GROUNDEDNESS",
}

ALLOWED_BLIND_WINNERS: Set[str] = {"A", "B", "TIE"}
ALLOWED_NORMALIZED_WINNERS: Set[str] = {"CANDIDATE", "BASELINE", "TIE"}

KAPPA_KILL_GATE_THRESHOLD: float = 0.60

FORBIDDEN_LEAKAGE_KEYS: Set[str] = {
    "intended_signal",
    "bias_probes",
    "baseline_answer",
    "candidate_answer",
    "expected_winner",
    "ground_truth",
}


class GroundTruthStatus:
    PENDING_HUMAN_ANNOTATION = "PENDING_HUMAN_ANNOTATION"
    ANNOTATIONS_IN_PROGRESS = "ANNOTATIONS_IN_PROGRESS"
    HUMAN_AGREEMENT_MEASURED = "HUMAN_AGREEMENT_MEASURED"
    CALIBRATION_PASSED = "CALIBRATION_PASSED"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"
    ADJUDICATION_COMPLETE = "ADJUDICATION_COMPLETE"
    GROUND_TRUTH_FROZEN = "GROUND_TRUTH_FROZEN"


@dataclass
class CriterionAssessmentData:
    """Evaluation data for an individual rubric criterion."""
    criterion_name: str
    winner: str
    rationale: str
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_name": self.criterion_name,
            "winner": self.winner,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CriterionAssessmentData:
        return cls(
            criterion_name=str(data.get("criterion_name", "")).upper(),
            winner=str(data.get("winner", "")).upper(),
            rationale=str(data.get("rationale", "")),
            evidence=data.get("evidence"),
        )


@dataclass
class AnnotatorSubmission:
    """Individual human annotator submission for a single benchmark case."""
    case_id: str
    annotator_id: str
    winner: str
    confidence: float
    rationale: str
    criterion_assessments: List[CriterionAssessmentData]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "SUBMITTED"
    presentation_order: str = "randomized"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "annotator_id": self.annotator_id,
            "presentation_order": self.presentation_order,
            "winner": self.winner,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "criterion_assessments": [c.to_dict() for c in self.criterion_assessments],
            "timestamp": self.timestamp,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotatorSubmission:
        criteria = [
            CriterionAssessmentData.from_dict(c)
            for c in data.get("criterion_assessments", [])
            if isinstance(c, dict)
        ]
        return cls(
            case_id=str(data["case_id"]),
            annotator_id=str(data["annotator_id"]),
            winner=str(data["winner"]).upper(),
            confidence=float(data.get("confidence", 1.0)),
            rationale=str(data.get("rationale", "")),
            criterion_assessments=criteria,
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            status=str(data.get("status", "SUBMITTED")),
            presentation_order=str(data.get("presentation_order", "randomized")),
        )


@dataclass
class ConfusionMatrix:
    """Confusion matrix between Annotator 1 and Annotator 2."""
    classes: List[str]
    matrix: Dict[str, Dict[str, int]]
    total_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "matrix": self.matrix,
            "total_observations": self.total_observations,
        }

    def format_ascii(self, name_ann1: str = "Annotator 1", name_ann2: str = "Annotator 2") -> str:
        col_width = 12
        header = f"{'':>14} | " + " | ".join(f"{c:^{col_width}}" for c in self.classes)
        sep = "-" * len(header)
        lines = [
            f"Confusion Matrix (Rows: {name_ann1}, Cols: {name_ann2}):",
            sep,
            header,
            sep,
        ]
        for row_cls in self.classes:
            row_str = f"{row_cls:>14} | " + " | ".join(
                f"{self.matrix.get(row_cls, {}).get(col_cls, 0):^{col_width}}"
                for col_cls in self.classes
            )
            lines.append(row_str)
        lines.append(sep)
        return "\n".join(lines)


@dataclass
class AgreementStatistics:
    """Comprehensive statistical report of inter-annotator reliability."""
    benchmark_id: str
    total_cases: int
    annotators: Dict[str, int]
    raw_agreement: float
    raw_agreement_count: int
    cohens_kappa: float
    kappa_se: float
    kappa_ci: Tuple[float, float]
    disagreement_count: int
    disagreements: List[Dict[str, Any]]
    confusion_matrix: ConfusionMatrix
    category_agreement: Dict[str, Dict[str, Any]]
    probe_agreement: Dict[str, Dict[str, Any]]
    criterion_agreement: Dict[str, Dict[str, Any]]
    calibration_gate: str
    gate_reason: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "total_cases": self.total_cases,
            "annotators": self.annotators,
            "raw_agreement": self.raw_agreement,
            "raw_agreement_count": self.raw_agreement_count,
            "cohens_kappa": self.cohens_kappa,
            "kappa_se": self.kappa_se,
            "kappa_ci": list(self.kappa_ci),
            "disagreement_count": self.disagreement_count,
            "disagreements": self.disagreements,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "category_agreement": self.category_agreement,
            "probe_agreement": self.probe_agreement,
            "criterion_agreement": self.criterion_agreement,
            "calibration_gate": self.calibration_gate,
            "gate_reason": self.gate_reason,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        lines = [
            "Human Annotation Agreement",
            "--------------------------",
            f"Benchmark:        {self.benchmark_id}",
            f"Cases Evaluated:  {self.total_cases}",
            "",
            "Annotator Submission Counts:",
        ]
        for ann_id, count in sorted(self.annotators.items()):
            lines.append(f"  {ann_id}: {count}/{self.total_cases}")

        lines.extend([
            "",
            f"Raw agreement:    {self.raw_agreement_count}/{self.total_cases} ({self.raw_agreement * 100:.1f}%)",
            f"Cohen's kappa:    {self.cohens_kappa:.4f} (SE: {self.kappa_se:.4f}, 95% CI: [{self.kappa_ci[0]:.4f}, {self.kappa_ci[1]:.4f}])",
            f"Disagreements:    {self.disagreement_count}",
            "",
            self.confusion_matrix.format_ascii(),
            "",
            "Agreement by Benchmark Category:",
        ])

        for cat, stats in sorted(self.category_agreement.items()):
            n = stats["cases"]
            agree = stats["agree_count"]
            pct = stats["raw_agreement"] * 100
            k_val = stats.get("cohens_kappa")
            k_str = f"{k_val:.2f}" if k_val is not None else "N/A"
            lines.append(f"  {cat:<18}: {agree:>2}/{n:>2} ({pct:>5.1f}%) | Cohen's kappa: {k_str}")

        lines.extend([
            "",
            "Agreement by Criterion Dimension:",
        ])
        for crit, stats in sorted(self.criterion_agreement.items()):
            n = stats["cases"]
            agree = stats["agree_count"]
            pct = stats["raw_agreement"] * 100
            k_val = stats.get("cohens_kappa")
            k_str = f"{k_val:.2f}" if k_val is not None else "N/A"
            lines.append(f"  {crit:<18}: {agree:>2}/{n:>2} ({pct:>5.1f}%) | Cohen's kappa: {k_str}")

        lines.extend([
            "",
            f"Calibration gate: {self.calibration_gate}",
            f"Gate decision:    {self.gate_reason}",
        ])

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


@dataclass
class AdjudicationRecord:
    """Authoritative review record resolving an annotator disagreement."""
    case_id: str
    adjudicator_id: str
    adjudicator_decision: str
    adjudicator_rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "adjudicator_id": self.adjudicator_id,
            "adjudicator_decision": self.adjudicator_decision,
            "adjudicator_rationale": self.adjudicator_rationale,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdjudicationRecord:
        decision = str(data["adjudicator_decision"]).strip().upper()
        if decision not in ALLOWED_NORMALIZED_WINNERS:
            raise ValueError(f"Invalid adjudicator_decision '{decision}'. Must be one of {sorted(ALLOWED_NORMALIZED_WINNERS)}.")
        return cls(
            case_id=str(data["case_id"]),
            adjudicator_id=str(data["adjudicator_id"]),
            adjudicator_decision=decision,
            adjudicator_rationale=str(data["adjudicator_rationale"]),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
        )


@dataclass
class GroundTruthArtifact:
    """Dedicated frozen ground-truth container, separate from the raw benchmark."""
    benchmark_id: str
    benchmark_version: str
    benchmark_sha256: str
    schema_version: str
    status: str
    case_count: int
    human_labels: str
    agreement_statistics: Optional[Dict[str, Any]]
    cases: List[Dict[str, Any]]
    created_at: str
    frozen_at: Optional[str] = None
    target_application: str = "Conquer (POST /api/interview/score)"
    production_ci_gating: str = "DISABLED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_sha256": self.benchmark_sha256,
            "schema_version": self.schema_version,
            "status": self.status,
            "case_count": self.case_count,
            "human_labels": self.human_labels,
            "target_application": self.target_application,
            "production_ci_gating": self.production_ci_gating,
            "created_at": self.created_at,
            "frozen_at": self.frozen_at,
            "agreement_statistics": self.agreement_statistics,
            "cases": self.cases,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthArtifact:
        return cls(
            benchmark_id=str(data["benchmark_id"]),
            benchmark_version=str(data.get("benchmark_version", "1.0.0")),
            benchmark_sha256=str(data["benchmark_sha256"]),
            schema_version=str(data.get("schema_version", "1.0.0")),
            status=str(data["status"]),
            case_count=int(data.get("case_count", len(data.get("cases", [])))),
            human_labels=str(data.get("human_labels", "PENDING")),
            agreement_statistics=data.get("agreement_statistics"),
            cases=list(data.get("cases", [])),
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            frozen_at=data.get("frozen_at"),
            target_application=str(data.get("target_application", "Conquer (POST /api/interview/score)")),
            production_ci_gating=str(data.get("production_ci_gating", "DISABLED")),
        )

    def compute_sha256(self) -> str:
        """Compute reproducible SHA-256 over normalized JSON string."""
        normalized_json = json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=True)
        return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


# ============================================================================
# Validation Functions
# ============================================================================

def validate_annotator_id(annotator_id: str) -> Optional[str]:
    """Validate that annotator_id is anonymized and contains no PII."""
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        return "Annotator ID must be a non-empty string."
    clean = annotator_id.strip()
    if "@" in clean:
        return f"Annotator ID '{clean}' contains email address or PII, violating anonymization policy."
    if not re.match(r"^[a-zA-Z0-9_\-]+$", clean):
        return f"Annotator ID '{clean}' contains invalid characters. Use alphanumeric, hyphens, or underscores."
    return None


def validate_annotator_submission(
    data: Dict[str, Any],
    valid_case_ids: Set[str],
    allowed_annotators: Optional[Set[str]] = None,
) -> List[str]:
    """Strictly validate an individual annotator submission against schema and benchmark constraints."""
    errors: List[str] = []

    if not isinstance(data, dict):
        return ["Submission payload must be a JSON object dictionary."]

    # Check forbidden leakage metadata keys
    for forbidden_key in FORBIDDEN_LEAKAGE_KEYS:
        if forbidden_key in data:
            errors.append(f"Submission contains forbidden ground-truth metadata key '{forbidden_key}'.")

    # Annotator ID validation
    annotator_id = data.get("annotator_id")
    if not annotator_id:
        errors.append("Missing required field 'annotator_id'.")
    else:
        id_err = validate_annotator_id(str(annotator_id))
        if id_err:
            errors.append(id_err)
        elif allowed_annotators and str(annotator_id) not in allowed_annotators:
            errors.append(f"Annotator ID '{annotator_id}' is not in authorized list: {sorted(allowed_annotators)}.")

    # Case ID validation
    case_id = data.get("case_id")
    if not case_id:
        errors.append("Missing required field 'case_id'.")
    elif not isinstance(case_id, str):
        errors.append("Field 'case_id' must be a string.")
    elif case_id not in valid_case_ids:
        errors.append(f"Unknown case_id '{case_id}'. Case is outside the frozen benchmark cases.")

    # Winner validation
    winner = data.get("winner")
    if not winner:
        errors.append("Missing required field 'winner'.")
    elif str(winner).upper() not in ALLOWED_BLIND_WINNERS:
        errors.append(f"Invalid winner value '{winner}'. Allowed blind winners: {sorted(ALLOWED_BLIND_WINNERS)}.")

    # Confidence validation
    confidence = data.get("confidence")
    if confidence is None:
        errors.append("Missing required field 'confidence'.")
    else:
        try:
            c_val = float(confidence)
            if not (0.0 <= c_val <= 1.0):
                errors.append(f"Confidence value {c_val} out of range [0.0, 1.0].")
        except (ValueError, TypeError):
            errors.append(f"Field 'confidence' must be numeric, got: {confidence}.")

    # Rationale validation
    rationale = data.get("rationale")
    if not rationale or not isinstance(rationale, str) or not rationale.strip():
        errors.append("Missing or empty required field 'rationale'.")
    elif len(rationale.strip()) < 15:
        errors.append(f"Rationale too brief ({len(rationale.strip())} chars). Submissions require meaningful qualitative justification (>= 15 chars).")

    # Status validation
    status = data.get("status")
    if status is not None and str(status).upper() != "SUBMITTED":
        errors.append(f"Invalid submission status '{status}'. Expected 'SUBMITTED'.")

    # Criterion assessments validation
    criteria_list = data.get("criterion_assessments")
    if not isinstance(criteria_list, list):
        errors.append("Missing or invalid 'criterion_assessments' list.")
    else:
        seen_criteria: Set[str] = set()
        for idx, crit in enumerate(criteria_list):
            if not isinstance(crit, dict):
                errors.append(f"Criterion assessment at index {idx} must be a dictionary.")
                continue

            c_name = crit.get("criterion_name")
            if not c_name or not isinstance(c_name, str):
                errors.append(f"Criterion assessment at index {idx} missing valid 'criterion_name'.")
                continue

            norm_name = c_name.strip().upper()
            if norm_name not in STANDARD_CRITERIA_NAMES:
                errors.append(f"Unrecognized criterion '{c_name}'. Allowed: {sorted(STANDARD_CRITERIA_NAMES)}.")
            if norm_name in seen_criteria:
                errors.append(f"Duplicate assessment for criterion '{norm_name}'.")
            seen_criteria.add(norm_name)

            c_winner = crit.get("winner")
            if not c_winner or str(c_winner).upper() not in ALLOWED_BLIND_WINNERS:
                errors.append(f"Criterion '{norm_name}' has invalid winner '{c_winner}'. Allowed: {sorted(ALLOWED_BLIND_WINNERS)}.")

            c_rationale = crit.get("rationale")
            if not c_rationale or not isinstance(c_rationale, str) or not c_rationale.strip():
                errors.append(f"Criterion '{norm_name}' missing required non-empty rationale.")

        # Ensure all 6 standard criteria are present
        missing_criteria = STANDARD_CRITERIA_NAMES - seen_criteria
        if missing_criteria:
            errors.append(f"Missing required criteria: {sorted(missing_criteria)}.")

    return errors


def validate_submissions_batch(
    submissions: List[Dict[str, Any]],
    benchmark_cases: List[Dict[str, Any]],
    allowed_annotators: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """Validate a batch of annotator submissions ensuring schema compliance and no duplicate entries."""
    errors: List[str] = []
    valid_ids: Set[str] = {c["case_id"] for c in benchmark_cases}

    seen_pairs: Set[Tuple[str, str]] = set()

    for idx, sub in enumerate(submissions):
        sub_errors = validate_annotator_submission(sub, valid_ids, allowed_annotators)
        if sub_errors:
            case_label = sub.get("case_id", f"index-{idx}")
            for err in sub_errors:
                errors.append(f"[{case_label}] {err}")
            continue

        pair_key = (str(sub["annotator_id"]), str(sub["case_id"]))
        if pair_key in seen_pairs:
            errors.append(f"Duplicate submission detected from annotator '{pair_key[0]}' for case '{pair_key[1]}'.")
        seen_pairs.add(pair_key)

    return len(errors) == 0, errors


# ============================================================================
# Loading and File Parsing
# ============================================================================

def load_annotator_submissions(
    source: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]
) -> List[AnnotatorSubmission]:
    """Load AnnotatorSubmission objects from directory, file, or memory list."""
    if isinstance(source, list):
        return [AnnotatorSubmission.from_dict(item) if isinstance(item, dict) else item for item in source]
    if isinstance(source, dict):
        if "submissions" in source and isinstance(source["submissions"], list):
            return [AnnotatorSubmission.from_dict(item) for item in source["submissions"]]
        return [AnnotatorSubmission.from_dict(source)]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Submissions path does not exist: {path}")

    raw_items: List[Dict[str, Any]] = []

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raw_items.extend(data)
        elif isinstance(data, dict):
            if "submissions" in data and isinstance(data["submissions"], list):
                raw_items.extend(data["submissions"])
            else:
                raw_items.append(data)
    elif path.is_dir():
        for json_file in sorted(path.rglob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, list):
                    raw_items.extend(content)
                elif isinstance(content, dict):
                    if "submissions" in content and isinstance(content["submissions"], list):
                        raw_items.extend(content["submissions"])
                    else:
                        raw_items.append(content)
            except Exception as exc:
                raise ValueError(f"Failed to read/parse submission file {json_file}: {exc}")

    return [AnnotatorSubmission.from_dict(item) for item in raw_items]


# ============================================================================
# Statistical Analysis: Cohen's Kappa & Agreement
# ============================================================================

def compute_cohens_kappa(
    ratings_1: List[str],
    ratings_2: List[str],
    classes: Optional[List[str]] = None,
) -> Tuple[float, float, Tuple[float, float], ConfusionMatrix]:
    """Calculate multi-class Cohen's kappa (κ), standard error, 95% CI, and confusion matrix.

    Classes default to ['CANDIDATE', 'BASELINE', 'TIE'].
    """
    if len(ratings_1) != len(ratings_2):
        raise ValueError(f"Ratings lists length mismatch: {len(ratings_1)} vs {len(ratings_2)}.")

    n = len(ratings_1)
    if classes is None:
        classes = sorted(list(set(ratings_1) | set(ratings_2)))
        if not classes:
            classes = ["CANDIDATE", "BASELINE", "TIE"]

    # Build confusion matrix
    matrix: Dict[str, Dict[str, int]] = {r: {c: 0 for c in classes} for r in classes}
    for r1, r2 in zip(ratings_1, ratings_2):
        if r1 in matrix and r2 in matrix[r1]:
            matrix[r1][r2] += 1

    conf_mat = ConfusionMatrix(classes=classes, matrix=matrix, total_observations=n)

    if n == 0:
        return 0.0, 0.0, (0.0, 0.0), conf_mat

    # Observed agreement Po
    agree_count = sum(matrix[c][c] for c in classes)
    po = agree_count / n

    # Expected agreement Pe
    pe = 0.0
    for c in classes:
        row_sum = sum(matrix[c][col] for col in classes)
        col_sum = sum(matrix[row][c] for row in classes)
        pe += (row_sum / n) * (col_sum / n)

    # Edge cases
    if math.isclose(pe, 1.0, abs_tol=1e-9):
        kappa = 1.0 if math.isclose(po, 1.0, abs_tol=1e-9) else 0.0
        se = 0.0
        return kappa, se, (kappa, kappa), conf_mat

    if math.isclose(po, pe, abs_tol=1e-9):
        kappa = 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    kappa = max(-1.0, min(1.0, kappa))

    # Standard error of kappa
    denom = (1.0 - pe)
    if denom <= 0:
        se = 0.0
    else:
        var_kappa = (po * (1.0 - po)) / (n * (denom ** 2))
        se = math.sqrt(max(0.0, var_kappa))

    ci_lower = max(-1.0, kappa - 1.96 * se)
    ci_upper = min(1.0, kappa + 1.96 * se)

    return kappa, se, (ci_lower, ci_upper), conf_mat


def calculate_human_agreement(
    ann1_submissions: List[AnnotatorSubmission],
    ann2_submissions: List[AnnotatorSubmission],
    key_mapping: Dict[str, Any],
    benchmark_data: Dict[str, Any],
    threshold: float = KAPPA_KILL_GATE_THRESHOLD,
) -> AgreementStatistics:
    """Perform comprehensive inter-rater reliability analysis between Annotator 1 and Annotator 2."""
    benchmark_id = benchmark_data.get("benchmark_id", "conquer-benchmark-v1")
    benchmark_cases = {c["case_id"]: c for c in benchmark_data.get("cases", [])}
    mappings = key_mapping.get("mappings", {})

    # Group submissions by case_id
    dict_ann1: Dict[str, AnnotatorSubmission] = {s.case_id: s for s in ann1_submissions}
    dict_ann2: Dict[str, AnnotatorSubmission] = {s.case_id: s for s in ann2_submissions}

    shared_case_ids = sorted(list(set(dict_ann1.keys()) & set(dict_ann2.keys())))
    total_shared = len(shared_case_ids)

    ann1_id = ann1_submissions[0].annotator_id if ann1_submissions else "annotator_1"
    ann2_id = ann2_submissions[0].annotator_id if ann2_submissions else "annotator_2"
    annotator_counts = {
        ann1_id: len(dict_ann1),
        ann2_id: len(dict_ann2),
    }

    norm_winners_1: List[str] = []
    norm_winners_2: List[str] = []
    disagreements: List[Dict[str, Any]] = []

    # Category trackers
    cat_winners_1: Dict[str, List[str]] = {}
    cat_winners_2: Dict[str, List[str]] = {}

    # Probe trackers
    probe_winners_1: Dict[str, List[str]] = {}
    probe_winners_2: Dict[str, List[str]] = {}

    # Criterion trackers
    crit_winners_1: Dict[str, List[str]] = {crit: [] for crit in STANDARD_CRITERIA_NAMES}
    crit_winners_2: Dict[str, List[str]] = {crit: [] for crit in STANDARD_CRITERIA_NAMES}

    for case_id in shared_case_ids:
        sub1 = dict_ann1[case_id]
        sub2 = dict_ann2[case_id]
        case_info = benchmark_cases.get(case_id, {})
        category = case_info.get("category", "UNKNOWN")
        probes = case_info.get("bias_probes", [])

        key_case = mappings.get(case_id)
        if not key_case:
            raise KeyError(f"Missing unblinding key entry for case_id '{case_id}'.")

        # Unblind primary winner
        norm1 = unblind_human_judgment(sub1.winner, key_case)
        norm2 = unblind_human_judgment(sub2.winner, key_case)

        norm_winners_1.append(norm1)
        norm_winners_2.append(norm2)

        # Category accumulation
        cat_winners_1.setdefault(category, []).append(norm1)
        cat_winners_2.setdefault(category, []).append(norm2)

        # Probe accumulation
        for p in probes:
            probe_winners_1.setdefault(p, []).append(norm1)
            probe_winners_2.setdefault(p, []).append(norm2)

        # Criterion assessments unblinding
        crit_map_1 = {c.criterion_name.upper(): c for c in sub1.criterion_assessments}
        crit_map_2 = {c.criterion_name.upper(): c for c in sub2.criterion_assessments}

        for crit in STANDARD_CRITERIA_NAMES:
            c1_raw = crit_map_1.get(crit)
            c2_raw = crit_map_2.get(crit)
            if c1_raw and c2_raw:
                c1_norm = unblind_human_judgment(c1_raw.winner, key_case)
                c2_norm = unblind_human_judgment(c2_raw.winner, key_case)
                crit_winners_1[crit].append(c1_norm)
                crit_winners_2[crit].append(c2_norm)

        # Track disagreement
        if norm1 != norm2:
            disagreements.append({
                "case_id": case_id,
                "category": category,
                "annotator_1": {
                    "annotator_id": sub1.annotator_id,
                    "blind_winner": sub1.winner,
                    "normalized_winner": norm1,
                    "confidence": sub1.confidence,
                    "rationale": sub1.rationale,
                },
                "annotator_2": {
                    "annotator_id": sub2.annotator_id,
                    "blind_winner": sub2.winner,
                    "normalized_winner": norm2,
                    "confidence": sub2.confidence,
                    "rationale": sub2.rationale,
                },
            })

    classes = ["CANDIDATE", "BASELINE", "TIE"]
    kappa, se, ci, conf_mat = compute_cohens_kappa(norm_winners_1, norm_winners_2, classes=classes)

    agree_count = sum(1 for w1, w2 in zip(norm_winners_1, norm_winners_2) if w1 == w2)
    raw_agreement = agree_count / total_shared if total_shared > 0 else 0.0

    # Category agreement breakdown
    category_agreement: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(cat_winners_1.keys()):
        w1_list = cat_winners_1[cat]
        w2_list = cat_winners_2[cat]
        c_count = len(w1_list)
        c_agree = sum(1 for a, b in zip(w1_list, w2_list) if a == b)
        c_raw = c_agree / c_count if c_count > 0 else 0.0
        c_k, _, _, _ = compute_cohens_kappa(w1_list, w2_list, classes=classes)
        category_agreement[cat] = {
            "cases": c_count,
            "agree_count": c_agree,
            "raw_agreement": c_raw,
            "cohens_kappa": c_k,
        }

    # Probe agreement breakdown
    probe_agreement: Dict[str, Dict[str, Any]] = {}
    for probe in sorted(probe_winners_1.keys()):
        pw1 = probe_winners_1[probe]
        pw2 = probe_winners_2[probe]
        p_count = len(pw1)
        p_agree = sum(1 for a, b in zip(pw1, pw2) if a == b)
        p_raw = p_agree / p_count if p_count > 0 else 0.0
        p_k, _, _, _ = compute_cohens_kappa(pw1, pw2, classes=classes)
        probe_agreement[probe] = {
            "cases": p_count,
            "agree_count": p_agree,
            "raw_agreement": p_raw,
            "cohens_kappa": p_k,
        }

    # Criterion-level agreement breakdown
    criterion_agreement: Dict[str, Dict[str, Any]] = {}
    for crit in sorted(STANDARD_CRITERIA_NAMES):
        cw1 = crit_winners_1[crit]
        cw2 = crit_winners_2[crit]
        cr_count = len(cw1)
        cr_agree = sum(1 for a, b in zip(cw1, cw2) if a == b)
        cr_raw = cr_agree / cr_count if cr_count > 0 else 0.0
        cr_k, _, _, _ = compute_cohens_kappa(cw1, cw2, classes=classes)
        criterion_agreement[crit] = {
            "cases": cr_count,
            "agree_count": cr_agree,
            "raw_agreement": cr_raw,
            "cohens_kappa": cr_k,
        }

    # Gate decision
    recommendations: List[str] = []
    if total_shared < REQUIRED_TOTAL_CASES:
        calibration_gate = "INCOMPLETE"
        gate_reason = f"Incomplete evaluations: only {total_shared}/{REQUIRED_TOTAL_CASES} benchmark cases evaluated by both annotators."
        recommendations.append("Ensure both annotators independently evaluate all 30 benchmark cases.")
    elif kappa >= threshold:
        calibration_gate = "PASS"
        gate_reason = (
            f"Predefined human calibration criterion met: Cohen's kappa {kappa:.4f} >= {threshold:.2f}. "
            "Proceed to consensus adjudication phase. Production CI gating remains disabled."
        )
        recommendations.append("Proceed to adjudication review for the detected disagreement cases.")
    else:
        calibration_gate = "FAIL"
        gate_reason = (
            f"HUMAN CALIBRATION FAILED: Cohen's kappa {kappa:.4f} < {threshold:.2f} threshold. "
            "Do NOT proceed to LLM judge calibration."
        )
        # Identify categories with highest disagreement
        worst_cats = sorted(
            category_agreement.items(),
            key=lambda item: item[1]["raw_agreement"]
        )
        if worst_cats:
            lowest_cat_names = [c[0] for c in worst_cats[:2]]
            recommendations.append(f"Review benchmark cases and rubrics in lowest agreement categories: {', '.join(lowest_cat_names)}.")

        # Identify criteria with highest disagreement
        worst_crits = sorted(
            criterion_agreement.items(),
            key=lambda item: item[1]["raw_agreement"]
        )
        if worst_crits:
            lowest_crit_names = [c[0] for c in worst_crits[:2]]
            recommendations.append(f"Review annotator guidelines for ambiguous criteria: {', '.join(lowest_crit_names)}.")

        recommendations.append("Conduct annotator alignment session on near-tie cases and update ANNOTATOR_GUIDE.md before recalibrating.")

    return AgreementStatistics(
        benchmark_id=benchmark_id,
        total_cases=total_shared,
        annotators=annotator_counts,
        raw_agreement=raw_agreement,
        raw_agreement_count=agree_count,
        cohens_kappa=kappa,
        kappa_se=se,
        kappa_ci=ci,
        disagreement_count=len(disagreements),
        disagreements=disagreements,
        confusion_matrix=conf_mat,
        category_agreement=category_agreement,
        probe_agreement=probe_agreement,
        criterion_agreement=criterion_agreement,
        calibration_gate=calibration_gate,
        gate_reason=gate_reason,
        recommendations=recommendations,
    )


# ============================================================================
# Adjudication & Consensus Engine
# ============================================================================

def create_adjudication_item(
    case_id: str,
    ann1_sub: AnnotatorSubmission,
    ann2_sub: AnnotatorSubmission,
    key_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an adjudication review item for a disagreed benchmark case."""
    norm1 = unblind_human_judgment(ann1_sub.winner, key_case)
    norm2 = unblind_human_judgment(ann2_sub.winner, key_case)

    return {
        "case_id": case_id,
        "annotator_1": {
            "annotator_id": ann1_sub.annotator_id,
            "raw_winner": ann1_sub.winner,
            "normalized_winner": norm1,
            "confidence": ann1_sub.confidence,
            "rationale": ann1_sub.rationale,
        },
        "annotator_2": {
            "annotator_id": ann2_sub.annotator_id,
            "raw_winner": ann2_sub.winner,
            "normalized_winner": norm2,
            "confidence": ann2_sub.confidence,
            "rationale": ann2_sub.rationale,
        },
        "adjudicator_id": None,
        "adjudicator_decision": None,
        "adjudicator_rationale": None,
    }


def apply_adjudications(
    ann1_submissions: List[AnnotatorSubmission],
    ann2_submissions: List[AnnotatorSubmission],
    key_mapping: Dict[str, Any],
    benchmark_data: Dict[str, Any],
    adjudication_records: Optional[Union[List[AdjudicationRecord], Dict[str, Any], Path, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Synthesize final consensus ground truth cases from independent submissions and adjudication records.

    Returns:
        (consensus_cases, pending_errors)
    """
    dict_ann1 = {s.case_id: s for s in ann1_submissions}
    dict_ann2 = {s.case_id: s for s in ann2_submissions}
    mappings = key_mapping.get("mappings", {})

    adj_map: Dict[str, AdjudicationRecord] = {}
    if adjudication_records:
        if isinstance(adjudication_records, (str, Path)):
            adj_path = Path(adjudication_records)
            if adj_path.exists():
                with open(adj_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                items = content if isinstance(content, list) else content.get("adjudications", [])
                for it in items:
                    rec = AdjudicationRecord.from_dict(it)
                    adj_map[rec.case_id] = rec
        elif isinstance(adjudication_records, list):
            for it in adjudication_records:
                rec = it if isinstance(it, AdjudicationRecord) else AdjudicationRecord.from_dict(it)
                adj_map[rec.case_id] = rec
        elif isinstance(adjudication_records, dict):
            items = adjudication_records.get("adjudications", adjudication_records)
            if isinstance(items, dict):
                for cid, it in items.items():
                    data = dict(it)
                    data.setdefault("case_id", cid)
                    adj_map[cid] = AdjudicationRecord.from_dict(data)
            elif isinstance(items, list):
                for it in items:
                    rec = AdjudicationRecord.from_dict(it)
                    adj_map[rec.case_id] = rec

    consensus_cases: List[Dict[str, Any]] = []
    pending_errors: List[str] = []

    for case in benchmark_data.get("cases", []):
        cid = case["case_id"]
        sub1 = dict_ann1.get(cid)
        sub2 = dict_ann2.get(cid)

        if not sub1 or not sub2:
            pending_errors.append(f"Case '{cid}' is missing submissions from one or both annotators.")
            consensus_cases.append({
                "case_id": cid,
                "category": case["category"],
                "status": "PENDING",
                "annotator_1": sub1.to_dict() if sub1 else None,
                "annotator_2": sub2.to_dict() if sub2 else None,
                "consensus": None,
            })
            continue

        key_case = mappings.get(cid)
        if not key_case:
            raise KeyError(f"Missing unblinding key for case '{cid}'.")

        norm1 = unblind_human_judgment(sub1.winner, key_case)
        norm2 = unblind_human_judgment(sub2.winner, key_case)

        if norm1 == norm2:
            # Unanimous agreement
            consensus_cases.append({
                "case_id": cid,
                "category": case["category"],
                "status": "CONSENSED",
                "annotator_1": {
                    "annotator_id": sub1.annotator_id,
                    "raw_winner": sub1.winner,
                    "normalized_winner": norm1,
                    "confidence": sub1.confidence,
                    "rationale": sub1.rationale,
                },
                "annotator_2": {
                    "annotator_id": sub2.annotator_id,
                    "raw_winner": sub2.winner,
                    "normalized_winner": norm2,
                    "confidence": sub2.confidence,
                    "rationale": sub2.rationale,
                },
                "consensus": {
                    "winner": norm1,
                    "agreement_type": "UNANIMOUS",
                    "adjudicator_id": None,
                    "adjudication_rationale": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })
        else:
            # Disagreement requiring adjudication
            adj = adj_map.get(cid)
            if not adj:
                pending_errors.append(f"Case '{cid}' has disagreement ({norm1} vs {norm2}) but lacks an authoritative adjudication record.")
                consensus_cases.append({
                    "case_id": cid,
                    "category": case["category"],
                    "status": "SUBMITTED",
                    "annotator_1": {
                        "annotator_id": sub1.annotator_id,
                        "raw_winner": sub1.winner,
                        "normalized_winner": norm1,
                        "confidence": sub1.confidence,
                        "rationale": sub1.rationale,
                    },
                    "annotator_2": {
                        "annotator_id": sub2.annotator_id,
                        "raw_winner": sub2.winner,
                        "normalized_winner": norm2,
                        "confidence": sub2.confidence,
                        "rationale": sub2.rationale,
                    },
                    "consensus": None,
                })
            else:
                consensus_cases.append({
                    "case_id": cid,
                    "category": case["category"],
                    "status": "ADJUDICATED",
                    "annotator_1": {
                        "annotator_id": sub1.annotator_id,
                        "raw_winner": sub1.winner,
                        "normalized_winner": norm1,
                        "confidence": sub1.confidence,
                        "rationale": sub1.rationale,
                    },
                    "annotator_2": {
                        "annotator_id": sub2.annotator_id,
                        "raw_winner": sub2.winner,
                        "normalized_winner": norm2,
                        "confidence": sub2.confidence,
                        "rationale": sub2.rationale,
                    },
                    "consensus": {
                        "winner": adj.adjudicator_decision,
                        "agreement_type": "ADJUDICATED",
                        "adjudicator_id": adj.adjudicator_id,
                        "adjudication_rationale": adj.adjudicator_rationale,
                        "timestamp": adj.timestamp,
                    },
                })

    return consensus_cases, pending_errors


# ============================================================================
# Ground-Truth Artifact Creation & Freezing
# ============================================================================

def create_pending_ground_truth_artifact(
    benchmark_source: Union[str, Path, Dict[str, Any]],
) -> GroundTruthArtifact:
    """Create initial PENDING GroundTruthArtifact linked to frozen benchmark."""
    data = load_benchmark_data(benchmark_source)
    benchmark_hash = compute_benchmark_hash(benchmark_source)

    cases = []
    for c in data.get("cases", []):
        cases.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "status": "PENDING",
            "annotator_1": None,
            "annotator_2": None,
            "consensus": None,
        })

    return GroundTruthArtifact(
        benchmark_id=data.get("benchmark_id", "conquer-benchmark-v1"),
        benchmark_version=data.get("version", "1.0.0"),
        benchmark_sha256=benchmark_hash,
        schema_version="1.0.0",
        status=GroundTruthStatus.PENDING_HUMAN_ANNOTATION,
        case_count=len(cases),
        human_labels="PENDING",
        agreement_statistics=None,
        cases=cases,
        created_at=datetime.now(timezone.utc).isoformat(),
        frozen_at=None,
    )


def freeze_ground_truth_artifact(
    artifact_source: Union[str, Path, Dict[str, Any], GroundTruthArtifact],
    benchmark_source: Union[str, Path, Dict[str, Any]],
    out_json: Path,
    out_manifest: Optional[Path] = None,
    out_sha256: Optional[Path] = None,
) -> Tuple[GroundTruthArtifact, str]:
    """Validate completeness and permanently freeze a ground-truth artifact.

    Strictly enforces:
    - Benchmark hash matches.
    - All 30 cases have completed consensus (no PENDING / SUBMITTED).
    - Writes deterministic JSON, manifest, and sha256.
    """
    if isinstance(artifact_source, GroundTruthArtifact):
        artifact = artifact_source
    else:
        if isinstance(artifact_source, (str, Path)):
            with open(artifact_source, "r", encoding="utf-8") as f:
                d = json.load(f)
        else:
            d = dict(artifact_source)
        artifact = GroundTruthArtifact.from_dict(d)

    # Verify benchmark hash
    expected_benchmark_hash = compute_benchmark_hash(benchmark_source)
    if artifact.benchmark_sha256 != expected_benchmark_hash:
        raise ValueError(
            f"Ground truth references benchmark hash '{artifact.benchmark_sha256}', "
            f"but current benchmark hash is '{expected_benchmark_hash}'."
        )

    # Verify all 30 cases are completed
    if len(artifact.cases) != REQUIRED_TOTAL_CASES:
        raise ValueError(f"Ground truth has {len(artifact.cases)} cases; expected exactly {REQUIRED_TOTAL_CASES}.")

    unresolved: List[str] = []
    for c in artifact.cases:
        cid = c.get("case_id")
        status = c.get("status")
        consensus = c.get("consensus")
        if status not in ("CONSENSED", "ADJUDICATED") or not consensus or not consensus.get("winner"):
            unresolved.append(f"{cid} (status={status})")

    if unresolved:
        raise ValueError(
            f"Cannot freeze ground truth while cases are incomplete or unadjudicated ({len(unresolved)} cases): {unresolved[:5]}..."
        )

    # Transition to frozen
    artifact.status = GroundTruthStatus.GROUND_TRUTH_FROZEN
    artifact.human_labels = "FROZEN_GROUND_TRUTH"
    artifact.frozen_at = datetime.now(timezone.utc).isoformat()

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(artifact.to_dict(), sort_keys=True, indent=2, ensure_ascii=True)
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(json_str)

    computed_sha = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    if out_sha256:
        out_sha256 = Path(out_sha256)
        out_sha256.parent.mkdir(parents=True, exist_ok=True)
        with open(out_sha256, "w", encoding="utf-8") as f:
            f.write(f"{computed_sha}  {out_json}\n")

    if out_manifest:
        out_manifest = Path(out_manifest)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "ground_truth_id": f"{artifact.benchmark_id}-ground-truth",
            "benchmark_id": artifact.benchmark_id,
            "benchmark_version": artifact.benchmark_version,
            "benchmark_sha256": artifact.benchmark_sha256,
            "schema_version": artifact.schema_version,
            "status": artifact.status,
            "human_labels": artifact.human_labels,
            "case_count": artifact.case_count,
            "sha256": computed_sha,
            "cohens_kappa": artifact.agreement_statistics.get("cohens_kappa") if artifact.agreement_statistics else None,
            "created_at": artifact.created_at,
            "frozen_at": artifact.frozen_at,
        }
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    return artifact, computed_sha
