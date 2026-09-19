from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from benizakura.errors import JudgeParseError, JudgeValidationError
from benizakura.models import (
    CriterionAssessment,
    PairwiseJudgment,
    PairwiseWinner,
    Rubric,
)


class JudgeResponseParser:
    """Parses and validates raw LLM responses into typed PairwiseJudgment instances."""

    _FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

    def parse(
        self,
        raw_output: str,
        rubric: Optional[Rubric] = None,
    ) -> PairwiseJudgment:
        """Parse raw model output text into a validated PairwiseJudgment.

        Raises:
            JudgeParseError: If text is not valid JSON.
            JudgeValidationError: If JSON violates the evaluation contract.
        """
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise JudgeParseError("Raw model output is empty or not a string.")

        json_text = self._extract_json_payload(raw_output.strip())

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as err:
            raise JudgeParseError(f"Failed to parse model output as JSON: {err}") from err

        if not isinstance(data, dict):
            raise JudgeValidationError(f"Expected JSON object at top level, got {type(data).__name__}.")

        return self._validate_and_build(data, rubric=rubric)

    def _extract_json_payload(self, text: str) -> str:
        """Extract JSON content from markdown code fences if present; otherwise return text."""
        match = self._FENCE_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return text

    def _validate_and_build(
        self,
        data: Dict[str, Any],
        rubric: Optional[Rubric] = None,
    ) -> PairwiseJudgment:
        """Validate parsed dictionary against domain contract and build PairwiseJudgment."""
        # 1. Validate overall winner
        if "winner" not in data:
            raise JudgeValidationError("Missing required field 'winner' in judge response.")

        raw_winner = data["winner"]
        if not isinstance(raw_winner, str):
            raise JudgeValidationError(
                f"Field 'winner' must be a string, got {type(raw_winner).__name__}."
            )

        winner_clean = raw_winner.strip().upper()
        if winner_clean not in ("A", "B", "TIE"):
            raise JudgeValidationError(
                f"Invalid winner value '{raw_winner}'. Must be one of 'A', 'B', or 'TIE'."
            )
        winner = PairwiseWinner(winner_clean)

        # 2. Validate rationale
        rationale = data.get("rationale") or data.get("reason")
        if not isinstance(rationale, str) or not rationale.strip():
            raise JudgeValidationError(
                "Missing or empty required field 'rationale' (or 'reason') in judge response."
            )
        rationale = rationale.strip()

        # 3. Validate confidence
        confidence: Optional[float] = None
        if "confidence" in data and data["confidence"] is not None:
            raw_conf = data["confidence"]
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError) as err:
                raise JudgeValidationError(
                    f"Field 'confidence' must be numeric, got {raw_conf!r}."
                ) from err

            if not (0.0 <= confidence <= 1.0):
                raise JudgeValidationError(
                    f"Field 'confidence' must be between 0.0 and 1.0, got {confidence}."
                )

        # 4. Validate criterion assessments
        assessments: List[CriterionAssessment] = []
        raw_assessments = data.get("criterion_assessments", [])
        if not isinstance(raw_assessments, list):
            raise JudgeValidationError(
                f"Field 'criterion_assessments' must be a list, got {type(raw_assessments).__name__}."
            )

        valid_criteria_map: Dict[str, str] = {}
        if rubric is not None:
            valid_criteria_map = {c.name.strip().lower(): c.name for c in rubric.criteria}

        for idx, item in enumerate(raw_assessments):
            if not isinstance(item, dict):
                raise JudgeValidationError(
                    f"Criterion assessment at index {idx} must be a dictionary, got {type(item).__name__}."
                )

            # Criterion name
            c_name = item.get("criterion_name")
            if not isinstance(c_name, str) or not c_name.strip():
                raise JudgeValidationError(
                    f"Criterion assessment at index {idx} missing non-empty 'criterion_name'."
                )
            c_name = c_name.strip()

            # Rubric containment check
            if rubric is not None:
                c_key = c_name.lower()
                if c_key not in valid_criteria_map:
                    valid_names = list(valid_criteria_map.values())
                    raise JudgeValidationError(
                        f"Unknown criterion '{c_name}' not defined in rubric '{rubric.name}'. "
                        f"Allowed criteria: {valid_names}."
                    )
                # Standardize to rubric's canonical casing
                c_name = valid_criteria_map[c_key]

            # Criterion winner
            if "winner" not in item:
                raise JudgeValidationError(
                    f"Criterion assessment '{c_name}' missing required field 'winner'."
                )
            c_raw_winner = item["winner"]
            if not isinstance(c_raw_winner, str):
                raise JudgeValidationError(
                    f"Criterion assessment '{c_name}' winner must be a string, got {type(c_raw_winner).__name__}."
                )
            c_winner_clean = c_raw_winner.strip().upper()
            if c_winner_clean not in ("A", "B", "TIE"):
                raise JudgeValidationError(
                    f"Invalid winner '{c_raw_winner}' in criterion assessment '{c_name}'. "
                    "Must be one of 'A', 'B', or 'TIE'."
                )
            c_winner = PairwiseWinner(c_winner_clean)

            # Criterion rationale
            c_rationale = item.get("rationale") or item.get("reason") or ""
            if not isinstance(c_rationale, str):
                raise JudgeValidationError(
                    f"Criterion assessment '{c_name}' rationale must be a string."
                )

            # Criterion confidence
            c_conf: Optional[float] = None
            if "confidence" in item and item["confidence"] is not None:
                try:
                    c_conf = float(item["confidence"])
                except (TypeError, ValueError) as err:
                    raise JudgeValidationError(
                        f"Criterion assessment '{c_name}' confidence must be numeric, got {item['confidence']!r}."
                    ) from err
                if not (0.0 <= c_conf <= 1.0):
                    raise JudgeValidationError(
                        f"Criterion assessment '{c_name}' confidence must be between 0.0 and 1.0, got {c_conf}."
                    )

            # Criterion evidence
            c_evidence = item.get("evidence")
            if c_evidence is not None and not isinstance(c_evidence, str):
                raise JudgeValidationError(
                    f"Criterion assessment '{c_name}' evidence must be a string or null."
                )

            assessments.append(
                CriterionAssessment(
                    criterion_name=c_name,
                    winner=c_winner,
                    rationale=c_rationale.strip(),
                    confidence=c_conf,
                    evidence=c_evidence.strip() if isinstance(c_evidence, str) else None,
                    metadata=item.get("metadata", {}),
                )
            )

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        return PairwiseJudgment(
            winner=winner,
            rationale=rationale,
            confidence=confidence,
            criterion_assessments=assessments,
            metadata=metadata,
        )
