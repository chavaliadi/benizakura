from __future__ import annotations

from typing import Optional

from benizakura.models import EvaluationCase, Rubric
from benizakura.provider import JudgeRequest


SYSTEM_PROMPT_TEMPLATE = """You are an expert, impartial evaluation judge comparing two AI-generated responses to an evaluation question.

Your task is to compare Response A and Response B strictly on their substantive merits based on the provided criteria.

Evaluation Rules:
1. Objectivity: Base your judgment strictly on technical accuracy, completeness, depth, and adherence to instructions.
2. Presentation Symmetry: Treat Response A and Response B neutrally without position bias.
3. Decisiveness: Declare "A" or "B" as winner whenever one response is meaningfully superior. Only declare "TIE" if they are substantively equivalent.
4. Evidence: Provide specific rationale and quotations/evidence justifying your determination.

You must respond with a single valid JSON object adhering to this schema:
{
  "winner": "A" | "B" | "TIE",
  "rationale": "<Overall qualitative explanation of why the winner was chosen>",
  "confidence": <float between 0.0 and 1.0>,
  "criterion_assessments": [
    {
      "criterion_name": "<exact name of the criterion>",
      "winner": "A" | "B" | "TIE",
      "rationale": "<specific rationale for this criterion>",
      "evidence": "<optional direct quote or excerpt from the response>"
    }
  ]
}

Respond ONLY with the JSON object. Do not include markdown preamble or postscript."""


class JudgePromptBuilder:
    """Builds structured JudgeRequest payloads for pairwise comparison."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT_TEMPLATE):
        self.system_prompt = system_prompt

    def build(
        self,
        case: EvaluationCase,
        output_a: str,
        output_b: str,
        rubric: Optional[Rubric] = None,
    ) -> JudgeRequest:
        """Construct a JudgeRequest from an evaluation case, presented outputs, and optional rubric."""
        user_prompt_lines = [
            f"# Evaluation Case: {case.id}",
            f"Topic: {case.topic}",
            f"Question:\n{case.question}\n",
        ]

        if rubric is not None:
            user_prompt_lines.append(f"# Evaluation Rubric: {rubric.name}")
            if rubric.instructions:
                user_prompt_lines.append(f"Instructions:\n{rubric.instructions}\n")

            if rubric.criteria:
                user_prompt_lines.append("Criteria:")
                for c in rubric.criteria:
                    user_prompt_lines.append(f"- **{c.name}** (weight: {c.weight}): {c.description}")
                user_prompt_lines.append("")

        user_prompt_lines.extend([
            "---",
            "# Response A:",
            output_a.strip(),
            "",
            "---",
            "# Response B:",
            output_b.strip(),
            "",
            "---",
            "Evaluate Response A vs Response B and emit the structured JSON evaluation.",
        ])

        user_prompt = "\n".join(user_prompt_lines)

        metadata = {
            "case_id": case.id,
            "has_rubric": rubric is not None,
            "rubric_name": rubric.name if rubric else None,
        }

        return JudgeRequest(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )
