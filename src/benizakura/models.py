from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    """Evaluation outcome verdict for comparing baseline vs candidate AI behavior."""
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class EvaluationCase:
    """Represents a single evaluation test case input.

    Attributes:
        id: Unique identifier for the test case.
        topic: Domain/track topic (e.g., 'System Design', 'Algorithms').
        question: The interview question or prompt presented to the candidate.
        candidate_answer: The candidate's response text being graded.
        mode: Interview mode (e.g., 'STANDARD', 'QUICK_FIRE', 'DEEP_DIVE').
        rubric: Optional placeholder for case-specific grading rubric criteria.
        metadata: Optional dictionary for additional metadata (tags, expected level, etc.).
    """
    id: str
    topic: str
    question: str
    candidate_answer: str
    mode: str = "STANDARD"
    rubric: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Represents the evaluation output produced for one evaluation case.

    Attributes:
        case_id: Reference identifier matching the evaluated EvaluationCase.id.
        score: Numerical evaluation score (typically 0.0 to 10.0 in Conquer).
        feedback: Qualitative feedback or critique generated for the response.
        metadata: Optional dictionary for auxiliary output (e.g. latency, token count, profile delta).
    """
    case_id: str
    score: float
    feedback: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRun:
    """Represents the collected results from executing an evaluation set against a target version.

    Attributes:
        run_id: Unique identifier for this execution run.
        version: Version or configuration identifier of the AI feature/prompt tested (e.g., 'baseline-v1', 'candidate-v2').
        results: Collection of EvaluationResult objects for each executed case.
        metadata: Optional run-level metadata (timestamp, model name, provider).
    """
    run_id: str
    version: str
    results: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def get_result(self, case_id: str) -> Optional[EvaluationResult]:
        for res in self.results:
            if res.case_id == case_id:
                return res
        return None


@dataclass
class ComparisonResult:
    """Represents the structured comparison outcome between baseline and candidate runs.

    Attributes:
        verdict: Final gate determination (PASS, FAIL, INCONCLUSIVE).
        summary: Human-readable narrative explanation of the decision.
        baseline_version: Identifier of baseline configuration.
        candidate_version: Identifier of candidate configuration.
        baseline_avg_score: Average score across baseline evaluation run.
        candidate_avg_score: Average score across candidate evaluation run.
        score_diff: Net difference (candidate_avg_score - baseline_avg_score).
        regressions: List of identified case-level regressions or negative anomalies.
        improvements: List of identified case-level improvements.
        metadata: Additional diagnostic details or statistical evidence.
    """
    verdict: Verdict
    summary: str
    baseline_version: str
    candidate_version: str
    baseline_avg_score: float
    candidate_avg_score: float
    score_diff: float
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pairwise Evaluation Domain Models
# ============================================================================


@dataclass
class Criterion:
    """Represents an individual evaluation criterion within a rubric.

    Attributes:
        name: Name or title of the criterion.
        description: Detailed guidance on what this criterion evaluates.
        weight: Numeric weight of the criterion, defaults to 1.0. Must be non-negative.
    """
    name: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Criterion name must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Criterion description must be a non-empty string.")
        try:
            self.weight = float(self.weight)
        except (TypeError, ValueError):
            raise TypeError("Criterion weight must be a numeric value.")
        if self.weight < 0.0:
            raise ValueError("Criterion weight must be non negative.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Criterion:
        return cls(
            name=data["name"],
            description=data["description"],
            weight=data.get("weight", 1.0),
        )


class StandardCriteria:
    """Domain-independent, reusable evaluation criteria library."""

    CORRECTNESS = Criterion(
        name="correctness",
        description="Assesses factual accuracy, logical validity, and freedom from errors or misconceptions.",
        weight=1.0,
    )
    RELEVANCE = Criterion(
        name="relevance",
        description="Assesses whether the response directly addresses the question asked without extraneous tangent.",
        weight=1.0,
    )
    COMPLETENESS = Criterion(
        name="completeness",
        description="Assesses whether all core components, constraints, and edge cases of the problem are covered.",
        weight=1.0,
    )
    TECHNICAL_DEPTH = Criterion(
        name="technical_depth",
        description="Assesses depth of explanation, understanding of trade-offs, architecture, and underlying mechanisms.",
        weight=1.0,
    )
    CLARITY = Criterion(
        name="clarity",
        description="Assesses precision of expression, logical organization, and conciseness.",
        weight=1.0,
    )
    GROUNDEDNESS = Criterion(
        name="groundedness",
        description="Assesses absence of hallucinations, fabricated facts, or unsupported claims.",
        weight=1.0,
    )


@dataclass
class Rubric:
    """Represents a structured evaluation rubric composed of criteria and evaluation instructions.

    Attributes:
        name: Name of the rubric.
        criteria: List of Criterion objects.
        instructions: Domain-independent evaluation instructions guiding the judge.
    """
    name: str
    criteria: List[Criterion] = field(default_factory=list)
    instructions: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Rubric name must be a non-empty string.")
        
        seen_names = set()
        for c in self.criteria:
            c_key = c.name.strip().lower()
            if c_key in seen_names:
                raise ValueError(f"Duplicate criterion name detected in rubric: '{c.name}'.")
            seen_names.add(c_key)

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    def get_criterion(self, name: str) -> Optional[Criterion]:
        target = name.strip().lower()
        for c in self.criteria:
            if c.name.strip().lower() == target:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "criteria": [c.to_dict() for c in self.criteria],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Rubric:
        return cls(
            name=data["name"],
            instructions=data.get("instructions", ""),
            criteria=[Criterion.from_dict(c) for c in data.get("criteria", [])],
        )


class PairwiseWinner(str, Enum):
    """Raw outcome from a pairwise comparison judge referring strictly to presented positions."""
    A = "A"
    B = "B"
    TIE = "TIE"


RawPositionWinner = PairwiseWinner


@dataclass
class CriterionAssessment:
    """Represents a judge's structured evaluation for an individual criterion.

    Attributes:
        criterion_name: Name matching the evaluated Criterion.
        winner: Position winner for this criterion (A, B, or TIE).
        rationale: Evidence and qualitative reasoning for this criterion judgment.
        confidence: Optional confidence score between 0.0 and 1.0.
        evidence: Optional excerpt or specific quotation grounding the assessment.
        metadata: Optional dictionary for auxiliary judge details.
    """
    criterion_name: str
    winner: PairwiseWinner
    rationale: str = ""
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.criterion_name, str) or not self.criterion_name.strip():
            raise ValueError("CriterionAssessment criterion_name must be a non-empty string.")
        # Synchronize rationale and reason for backward compatibility
        if not self.rationale and self.reason:
            self.rationale = self.reason
        elif not self.reason and self.rationale:
            self.reason = self.rationale
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0.")

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "criterion_name": self.criterion_name,
            "winner": self.winner.value,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
        if self.confidence is not None:
            res["confidence"] = self.confidence
        if self.evidence is not None:
            res["evidence"] = self.evidence
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CriterionAssessment:
        return cls(
            criterion_name=data["criterion_name"],
            winner=PairwiseWinner(data["winner"]),
            rationale=data.get("rationale", data.get("reason", "")),
            confidence=data.get("confidence"),
            evidence=data.get("evidence"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PairwiseJudgment:
    """Represents a judgment returned by a judge comparing two presented positions.

    Attributes:
        winner: Raw position winner (A, B, or TIE).
        rationale: Qualitative explanation justifying the overall decision.
        confidence: Optional overall confidence score between 0.0 and 1.0.
        criterion_assessments: Structured per-criterion evaluations justifying the verdict.
        metadata: Optional auxiliary details from the judge.
        reason: Alias for rationale to ensure backward compatibility.
    """
    winner: PairwiseWinner
    rationale: str = ""
    confidence: Optional[float] = None
    criterion_assessments: List[CriterionAssessment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        # Keep rationale and reason synchronized for backward compatibility
        if not self.rationale and self.reason:
            self.rationale = self.reason
        elif not self.reason and self.rationale:
            self.reason = self.rationale
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0.")

    def get_assessment(self, criterion_name: str) -> Optional[CriterionAssessment]:
        target = criterion_name.strip().lower()
        for ca in self.criterion_assessments:
            if ca.criterion_name.strip().lower() == target:
                return ca
        return None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "winner": self.winner.value,
            "rationale": self.rationale,
            "criterion_assessments": [ca.to_dict() for ca in self.criterion_assessments],
            "metadata": self.metadata,
        }
        if self.confidence is not None:
            res["confidence"] = self.confidence
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PairwiseJudgment:
        assessments = [
            CriterionAssessment.from_dict(ca)
            for ca in data.get("criterion_assessments", [])
        ]
        return cls(
            winner=PairwiseWinner(data["winner"]),
            rationale=data.get("rationale", data.get("reason", "")),
            confidence=data.get("confidence"),
            criterion_assessments=assessments,
            metadata=data.get("metadata", {}),
        )


class NormalizedWinner(str, Enum):
    """Underlying candidate versus baseline comparison identity after position normalization."""
    BASELINE = "BASELINE"
    CANDIDATE = "CANDIDATE"
    TIE = "TIE"


ComparisonIdentity = NormalizedWinner


@dataclass
class BidirectionalEvaluationResult:
    """Represents the results of evaluating a Baseline versus Candidate pair across both presentation orders.

    Pass 1 (forward): A is Baseline, B is Candidate
    Pass 2 (reverse): A is Candidate, B is Baseline

    Attributes:
        pass1_judgment: Raw judgment from pass 1.
        pass2_judgment: Raw judgment from pass 2.
        pass1_normalized_winner: Normalized comparison identity for pass 1.
        pass2_normalized_winner: Normalized comparison identity for pass 2.
        metadata: Optional dictionary for execution metadata.
    """
    pass1_judgment: PairwiseJudgment
    pass2_judgment: PairwiseJudgment
    pass1_normalized_winner: NormalizedWinner
    pass2_normalized_winner: NormalizedWinner
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass1_raw_judgment(self) -> PairwiseJudgment:
        return self.pass1_judgment

    @property
    def pass2_raw_judgment(self) -> PairwiseJudgment:
        return self.pass2_judgment

    def get_criterion_normalized_winners(
        self, criterion_name: str
    ) -> tuple[Optional[NormalizedWinner], Optional[NormalizedWinner]]:
        """Extract and normalize criterion-level winners from both passes, if present."""
        ca1 = self.pass1_judgment.get_assessment(criterion_name)
        ca2 = self.pass2_judgment.get_assessment(criterion_name)

        norm1: Optional[NormalizedWinner] = None
        if ca1 is not None:
            if ca1.winner == PairwiseWinner.A:
                norm1 = NormalizedWinner.BASELINE
            elif ca1.winner == PairwiseWinner.B:
                norm1 = NormalizedWinner.CANDIDATE
            else:
                norm1 = NormalizedWinner.TIE

        norm2: Optional[NormalizedWinner] = None
        if ca2 is not None:
            if ca2.winner == PairwiseWinner.A:
                norm2 = NormalizedWinner.CANDIDATE
            elif ca2.winner == PairwiseWinner.B:
                norm2 = NormalizedWinner.BASELINE
            else:
                norm2 = NormalizedWinner.TIE

        return norm1, norm2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass1_judgment": self.pass1_judgment.to_dict(),
            "pass2_judgment": self.pass2_judgment.to_dict(),
            "pass1_normalized_winner": self.pass1_normalized_winner.value,
            "pass2_normalized_winner": self.pass2_normalized_winner.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BidirectionalEvaluationResult:
        return cls(
            pass1_judgment=PairwiseJudgment.from_dict(data["pass1_judgment"]),
            pass2_judgment=PairwiseJudgment.from_dict(data["pass2_judgment"]),
            pass1_normalized_winner=NormalizedWinner(data["pass1_normalized_winner"]),
            pass2_normalized_winner=NormalizedWinner(data["pass2_normalized_winner"]),
            metadata=data.get("metadata", {}),
        )


BidirectionalResult = BidirectionalEvaluationResult


class ConsistencyOutcome(str, Enum):
    """Possible outcomes when analyzing consistency across bidirectional evaluation passes."""
    CONSISTENT_CANDIDATE_WIN = "CONSISTENT_CANDIDATE_WIN"
    CONSISTENT_BASELINE_WIN = "CONSISTENT_BASELINE_WIN"
    CONSISTENT_TIE = "CONSISTENT_TIE"
    POSITION_UNSTABLE = "POSITION_UNSTABLE"


ConsistencyClassification = ConsistencyOutcome


def classify_consistency(
    pass1_winner: NormalizedWinner,
    pass2_winner: NormalizedWinner,
) -> ConsistencyOutcome:
    """Classify consistency between two normalized bidirectional evaluation passes.

    If both passes agree on the normalized winner, the outcome is consistent.
    If the passes conflict, the outcome is classified as POSITION_UNSTABLE.
    """
    if pass1_winner == pass2_winner:
        if pass1_winner == NormalizedWinner.CANDIDATE:
            return ConsistencyOutcome.CONSISTENT_CANDIDATE_WIN
        if pass1_winner == NormalizedWinner.BASELINE:
            return ConsistencyOutcome.CONSISTENT_BASELINE_WIN
        return ConsistencyOutcome.CONSISTENT_TIE
    return ConsistencyOutcome.POSITION_UNSTABLE

