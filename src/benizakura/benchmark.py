from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from benizakura.models import EvaluationCase


# Canonical benchmark category requirements
REQUIRED_CATEGORY_COUNTS: Dict[str, int] = {
    "DSA": 6,
    "SYSTEM_DESIGN": 8,
    "BACKEND": 6,
    "FRONTEND": 4,
    "BEHAVIORAL": 6,
}
REQUIRED_TOTAL_CASES: int = 30

ALLOWED_CATEGORIES: Set[str] = set(REQUIRED_CATEGORY_COUNTS.keys())
ALLOWED_DIFFICULTIES: Set[str] = {"easy", "medium", "hard"}
ALLOWED_INTENDED_SIGNALS: Set[str] = {
    "candidate_improvement",
    "candidate_regression",
    "near_tie",
    "tradeoff_neutral",
}
ALLOWED_BIAS_PROBES: Set[str] = {
    "position",
    "verbosity",
    "hallucination",
    "near_tie",
    "alternative_architecture",
    "criterion_tradeoff",
    "subtle_regression",
    "edge_case_failure",
    "complexity_regression",
    "clearly_better",
}
ALLOWED_ANNOTATION_STATUSES: Set[str] = {
    "PENDING",
    "IN_PROGRESS",
    "SUBMITTED",
    "ADJUDICATED",
    "CONSENSED",
}
ALLOWED_HUMAN_WINNERS: Set[str] = {"CANDIDATE", "BASELINE", "TIE"}


@dataclass
class AcceptedAlternative:
    """Documented valid technical approach for human annotation reference."""
    approach: str
    conditions: str
    tradeoffs: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "approach": self.approach,
            "conditions": self.conditions,
            "tradeoffs": self.tradeoffs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AcceptedAlternative:
        return cls(
            approach=str(data.get("approach", "")),
            conditions=str(data.get("conditions", "")),
            tradeoffs=str(data.get("tradeoffs", "")),
        )


@dataclass
class HumanAnnotation:
    """Human ground-truth annotation container."""
    status: str = "PENDING"
    annotator_1: Optional[Dict[str, Any]] = None
    annotator_2: Optional[Dict[str, Any]] = None
    consensus: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "annotator_1": self.annotator_1,
            "annotator_2": self.annotator_2,
            "consensus": self.consensus,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HumanAnnotation:
        return cls(
            status=str(data.get("status", "PENDING")),
            annotator_1=data.get("annotator_1"),
            annotator_2=data.get("annotator_2"),
            consensus=data.get("consensus"),
        )


@dataclass
class BenchmarkCase:
    """Machine-readable specification of an evaluation benchmark test case."""
    case_id: str
    category: str
    subcategory: str
    difficulty: str
    question: str
    context: str
    baseline_answer: str
    candidate_answer: str
    human_annotation: HumanAnnotation
    bias_probes: List[str]
    accepted_alternatives: List[AcceptedAlternative]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "difficulty": self.difficulty,
            "question": self.question,
            "context": self.context,
            "baseline_answer": self.baseline_answer,
            "candidate_answer": self.candidate_answer,
            "human_annotation": self.human_annotation.to_dict(),
            "bias_probes": list(self.bias_probes),
            "accepted_alternatives": [a.to_dict() for a in self.accepted_alternatives],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkCase:
        annotation_data = data.get("human_annotation", {})
        if isinstance(annotation_data, dict):
            annotation = HumanAnnotation.from_dict(annotation_data)
        else:
            annotation = HumanAnnotation(status="PENDING")

        alternatives = [
            AcceptedAlternative.from_dict(a)
            for a in data.get("accepted_alternatives", [])
            if isinstance(a, dict)
        ]

        return cls(
            case_id=str(data["case_id"]),
            category=str(data["category"]),
            subcategory=str(data.get("subcategory", "")),
            difficulty=str(data.get("difficulty", "medium")),
            question=str(data["question"]),
            context=str(data.get("context", "")),
            baseline_answer=str(data["baseline_answer"]),
            candidate_answer=str(data["candidate_answer"]),
            human_annotation=annotation,
            bias_probes=list(data.get("bias_probes", [])),
            accepted_alternatives=alternatives,
            metadata=dict(data.get("metadata", {})),
        )

    def to_evaluation_case(self) -> EvaluationCase:
        """Extract a sanitized EvaluationCase for LLM judge evaluation.

        CRITICAL: Completely strips human annotations, consensus labels, and intended signals
        to prevent evaluator leakage.
        """
        # Strictly sanitized metadata — no ground truth or intended signals
        safe_metadata = {
            "benchmark_case_id": self.case_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "difficulty": self.difficulty,
            "has_context": bool(self.context.strip()),
        }

        # Include question context if present
        full_question = self.question
        if self.context.strip():
            full_question = f"{self.context.strip()}\n\n{self.question}"

        return EvaluationCase(
            id=self.case_id,
            topic=self.category,
            question=full_question,
            candidate_answer=self.candidate_answer,
            mode="STANDARD",
            metadata=safe_metadata,
        )


@dataclass
class BenchmarkValidationReport:
    """Validation report detailing structural, content, and leakage audit results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    case_count: int
    category_counts: Dict[str, int]
    probes_detected: Dict[str, int]
    leakage_detected: bool

    def summary(self) -> str:
        lines = [
            "Benchmark Validation Report",
            "===========================",
            f"Status:             {'VALID' if self.is_valid else 'INVALID'}",
            f"Total Cases:        {self.case_count} (Expected: {REQUIRED_TOTAL_CASES})",
            f"Leakage Detected:   {self.leakage_detected}",
            "",
            "Category Distribution:",
        ]
        for cat, expected in REQUIRED_CATEGORY_COUNTS.items():
            actual = self.category_counts.get(cat, 0)
            status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
            lines.append(f"  - {cat:<15}: {actual:>2} [{status}]")

        lines.append("")
        lines.append("Detected Bias Probes:")
        for probe, count in sorted(self.probes_detected.items()):
            lines.append(f"  - {probe:<25}: {count:>2} cases")

        if self.errors:
            lines.append("")
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  [ERROR] {err}")

        if self.warnings:
            lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  [WARN]  {warn}")

        return "\n".join(lines)


def load_benchmark_data(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load benchmark data from a JSON file path or dictionary."""
    if isinstance(source, dict):
        return source
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_benchmark(source: Union[str, Path, Dict[str, Any]]) -> BenchmarkValidationReport:
    """Rigorously validate benchmark structure, stratification, content, and leakage safeguards."""
    errors: List[str] = []
    warnings: List[str] = []
    category_counts: Dict[str, int] = {cat: 0 for cat in REQUIRED_CATEGORY_COUNTS}
    probes_detected: Dict[str, int] = {}
    leakage_detected = False

    try:
        data = load_benchmark_data(source)
    except Exception as exc:
        return BenchmarkValidationReport(
            is_valid=False,
            errors=[f"Failed to read/parse benchmark JSON: {exc}"],
            warnings=[],
            case_count=0,
            category_counts=category_counts,
            probes_detected={},
            leakage_detected=False,
        )

    if not isinstance(data, dict):
        return BenchmarkValidationReport(
            is_valid=False,
            errors=["Root of benchmark JSON must be an object with a 'cases' array."],
            warnings=[],
            case_count=0,
            category_counts=category_counts,
            probes_detected={},
            leakage_detected=False,
        )

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        return BenchmarkValidationReport(
            is_valid=False,
            errors=["Benchmark file must contain a 'cases' list."],
            warnings=[],
            case_count=0,
            category_counts=category_counts,
            probes_detected={},
            leakage_detected=False,
        )

    total_cases = len(raw_cases)
    if total_cases != REQUIRED_TOTAL_CASES:
        errors.append(f"Expected exactly {REQUIRED_TOTAL_CASES} cases, but found {total_cases}.")

    seen_ids: Set[str] = set()
    seen_questions: Set[str] = set()

    for idx, case_dict in enumerate(raw_cases):
        if not isinstance(case_dict, dict):
            errors.append(f"Case index {idx} is not a valid JSON dictionary.")
            continue

        case_id = case_dict.get("case_id")
        if not case_id or not isinstance(case_id, str):
            errors.append(f"Case at index {idx} has invalid or missing 'case_id'.")
            continue

        if case_id in seen_ids:
            errors.append(f"Duplicate case_id detected: '{case_id}'.")
        seen_ids.add(case_id)

        # Category validation
        category = case_dict.get("category")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"Case '{case_id}' has unrecognized category '{category}'. Allowed: {sorted(ALLOWED_CATEGORIES)}.")
        else:
            category_counts[category] = category_counts.get(category, 0) + 1

        # Difficulty validation
        difficulty = case_dict.get("difficulty")
        if difficulty not in ALLOWED_DIFFICULTIES:
            errors.append(f"Case '{case_id}' has unrecognized difficulty '{difficulty}'. Allowed: {sorted(ALLOWED_DIFFICULTIES)}.")

        # Question & Answer content validation
        question = case_dict.get("question", "")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"Case '{case_id}' has empty or invalid 'question'.")
        else:
            q_norm = " ".join(question.strip().lower().split())
            if q_norm in seen_questions:
                warnings.append(f"Case '{case_id}' appears to have a duplicate or near-duplicate question.")
            seen_questions.add(q_norm)

        baseline_ans = case_dict.get("baseline_answer", "")
        if not isinstance(baseline_ans, str) or not baseline_ans.strip():
            errors.append(f"Case '{case_id}' has empty or invalid 'baseline_answer'.")

        candidate_ans = case_dict.get("candidate_answer", "")
        if not isinstance(candidate_ans, str) or not candidate_ans.strip():
            errors.append(f"Case '{case_id}' has empty or invalid 'candidate_answer'.")

        # Probes validation
        probes = case_dict.get("bias_probes", [])
        if not isinstance(probes, list) or not probes:
            errors.append(f"Case '{case_id}' must specify at least one bias probe in 'bias_probes'.")
        else:
            for p in probes:
                if p not in ALLOWED_BIAS_PROBES:
                    warnings.append(f"Case '{case_id}' specifies unrecognized bias probe '{p}'.")
                probes_detected[p] = probes_detected.get(p, 0) + 1

        # Intended signal validation
        metadata = case_dict.get("metadata", {})
        if isinstance(metadata, dict):
            signal = metadata.get("intended_signal")
            if signal and signal not in ALLOWED_INTENDED_SIGNALS:
                errors.append(f"Case '{case_id}' has unrecognized intended_signal '{signal}'. Allowed: {sorted(ALLOWED_INTENDED_SIGNALS)}.")
        else:
            errors.append(f"Case '{case_id}' metadata must be a dictionary.")

        # Human Annotation checks
        annotation = case_dict.get("human_annotation", {})
        if not isinstance(annotation, dict):
            errors.append(f"Case '{case_id}' 'human_annotation' must be a dictionary.")
        else:
            status = annotation.get("status")
            if status not in ALLOWED_ANNOTATION_STATUSES:
                errors.append(f"Case '{case_id}' has invalid annotation status '{status}'. Allowed: {sorted(ALLOWED_ANNOTATION_STATUSES)}.")

            # Ground truth integrity: cannot have consensus if status is PENDING
            if status == "PENDING":
                if annotation.get("consensus") is not None:
                    errors.append(f"Case '{case_id}' has status 'PENDING' but includes a consensus object (illegal premature label).")
                if annotation.get("annotator_1") is not None or annotation.get("annotator_2") is not None:
                    warnings.append(f"Case '{case_id}' has status 'PENDING' with partial annotator data.")
            elif status in ("CONSENSED", "ADJUDICATED"):
                consensus = annotation.get("consensus")
                if not isinstance(consensus, dict) or "winner" not in consensus:
                    errors.append(f"Case '{case_id}' has status '{status}' but is missing a valid consensus winner.")

        # Leakage audit on judge conversion
        try:
            case_obj = BenchmarkCase.from_dict(case_dict)
            eval_case = case_obj.to_evaluation_case()
            leakage_issues = audit_evaluation_case_leakage(eval_case)
            if leakage_issues:
                leakage_detected = True
                for issue in leakage_issues:
                    errors.append(f"Leakage in '{case_id}': {issue}")
        except Exception as exc:
            errors.append(f"Failed to instantiate BenchmarkCase for '{case_id}': {exc}")

    # Check exact category counts
    for cat, expected in REQUIRED_CATEGORY_COUNTS.items():
        actual = category_counts.get(cat, 0)
        if actual != expected:
            errors.append(f"Category '{cat}' count mismatch: expected {expected}, found {actual}.")

    # Position probe must be present on all cases
    if probes_detected.get("position", 0) != REQUIRED_TOTAL_CASES:
        warnings.append(
            f"'position' bias probe is marked on {probes_detected.get('position', 0)} of {REQUIRED_TOTAL_CASES} cases. "
            "All pairwise benchmark cases should be evaluated bidirectionally."
        )

    is_valid = len(errors) == 0
    return BenchmarkValidationReport(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        case_count=total_cases,
        category_counts=category_counts,
        probes_detected=probes_detected,
        leakage_detected=leakage_detected,
    )


def audit_evaluation_case_leakage(eval_case: EvaluationCase) -> List[str]:
    """Verify that an EvaluationCase contains no ground truth or intended signal leakage."""
    issues = []
    meta = eval_case.metadata

    forbidden_meta_keys = {
        "human_annotation",
        "intended_signal",
        "winner",
        "consensus",
        "primary_failure_mode",
        "expected_quality",
        "ground_truth",
        "annotator_1",
        "annotator_2",
    }
    for key in forbidden_meta_keys:
        if key in meta:
            issues.append(f"EvaluationCase metadata contains forbidden ground-truth key '{key}'.")

    # Verify that question text does not leak winner tags
    for leak_phrase in ("intended_signal", "winner: candidate", "winner: baseline", "ground truth"):
        if leak_phrase in eval_case.question.lower():
            issues.append(f"EvaluationCase question text contains suspicious phrase '{leak_phrase}'.")

    return issues


def compute_benchmark_hash(source: Union[str, Path, Dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 hash over normalized benchmark cases.

    Sorts keys and enforces consistent whitespace/encoding so the hash is strictly reproducible.
    """
    data = load_benchmark_data(source)
    # Extract only the benchmark payload for deterministic hashing
    normalized_json = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True)
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


def generate_blinded_annotation_tasks(
    source: Union[str, Path, Dict[str, Any]],
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate double-blind annotation tasks and a private unblinding key.

    Randomizes the presentation order of Baseline vs Candidate into Response A vs Response B.
    The returned task set is safe to distribute to human annotators without leaking system identities.
    """
    data = load_benchmark_data(source)
    cases = data.get("cases", [])

    rng = random.Random(seed)
    blinded_tasks: List[Dict[str, Any]] = []
    key_mapping: Dict[str, Any] = {
        "seed": seed,
        "benchmark_id": data.get("benchmark_id", "conquer-benchmark-v1"),
        "mappings": {},
    }

    for case in cases:
        case_id = case["case_id"]
        baseline_text = case["baseline_answer"]
        candidate_text = case["candidate_answer"]

        # Randomize: False = (A is Baseline, B is Candidate); True = (A is Candidate, B is Baseline)
        swap = rng.choice([False, True])

        if not swap:
            resp_a = baseline_text
            resp_b = candidate_text
            mapping = {"A": "BASELINE", "B": "CANDIDATE"}
        else:
            resp_a = candidate_text
            resp_b = baseline_text
            mapping = {"A": "CANDIDATE", "B": "BASELINE"}

        key_mapping["mappings"][case_id] = {
            "response_a": mapping["A"],
            "response_b": mapping["B"],
            "swapped": swap,
        }

        task = {
            "case_id": case_id,
            "category": case["category"],
            "subcategory": case.get("subcategory", ""),
            "difficulty": case.get("difficulty", "medium"),
            "question": case["question"],
            "context": case.get("context", ""),
            "response_a": resp_a,
            "response_b": resp_b,
            "accepted_alternatives": case.get("accepted_alternatives", []),
        }
        blinded_tasks.append(task)

    return blinded_tasks, key_mapping


def unblind_human_judgment(
    blind_winner: str,
    key_for_case: Dict[str, Any],
) -> str:
    """Translate a blinded choice ('A', 'B', 'TIE') back into system identity ('CANDIDATE', 'BASELINE', 'TIE')."""
    clean_winner = blind_winner.strip().upper()
    if clean_winner == "TIE":
        return "TIE"
    if clean_winner == "A":
        return str(key_for_case["response_a"])
    if clean_winner == "B":
        return str(key_for_case["response_b"])
    raise ValueError(f"Invalid blind winner: '{blind_winner}'. Expected 'A', 'B', or 'TIE'.")
