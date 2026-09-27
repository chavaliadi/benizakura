from __future__ import annotations

from typing import Optional

from benizakura.models import EvaluationCase, Rubric
from benizakura.provider import JudgeRequest


SYSTEM_PROMPT_PAIRWISE_V1 = """You are an expert, impartial evaluation judge comparing two AI-generated responses to an evaluation question.

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

# Canonical default
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_PAIRWISE_V1


SYSTEM_PROMPT_PAIRWISE_RUBRIC_V1 = """You are a rigorous technical evaluation judge assessing two candidate engineering interview answers against an explicit domain rubric.

Evaluation Instructions:
1. Deconstruct both responses across each rubric criterion independently.
2. Verify algorithmic asymptotic complexity, architectural trade-offs, and boundary conditions.
3. Do NOT favor verbose or stylistic answers over concise, mathematically and technically accurate answers.
4. If an answer makes an authoritative but factually flawed claim, severely penalize it under Correctness and Groundedness.
5. Weigh criteria according to their stated weights.

Respond with a single valid JSON object adhering to this schema:
{
  "winner": "A" | "B" | "TIE",
  "rationale": "<Synthesized qualitative rationale referencing the criteria trade-offs>",
  "confidence": <float between 0.0 and 1.0>,
  "criterion_assessments": [
    {
      "criterion_name": "<exact name of the criterion>",
      "winner": "A" | "B" | "TIE",
      "rationale": "<comparative rationale for this criterion>",
      "evidence": "<direct quotation or excerpt from Response A or B>"
    }
  ]
}

Respond ONLY with the JSON object. Do not include markdown preamble or postscript."""


SYSTEM_PROMPT_PAIRWISE_RATIONALE_FIRST_V1 = """You are an expert, impartial evaluation judge comparing two AI-generated responses to an evaluation question.

Critical Evaluation Process:
You must perform criterion-level analysis and write your qualitative reasoning BEFORE determining the overall winner. Emitting rationale tokens first ensures objective analysis before committing to a final verdict.

Evaluation Rules:
1. Analyze each rubric criterion separately, evaluating Response A vs Response B with evidence quotes.
2. Synthesize an overall comparative rationale evaluating the net engineering trade-offs.
3. Finally, decide the overall winner: "A", "B", or "TIE".

You must respond with a single valid JSON object adhering to this exact schema (criteria and rationale FIRST):
{
  "criterion_assessments": [
    {
      "criterion_name": "<exact name of the criterion>",
      "winner": "A" | "B" | "TIE",
      "rationale": "<specific comparative rationale for this criterion>",
      "evidence": "<direct quotation from response>"
    }
  ],
  "overall_rationale": "<Comprehensive qualitative synthesis justifying the final decision>",
  "confidence": <float between 0.0 and 1.0>,
  "winner": "A" | "B" | "TIE"
}

Respond ONLY with the JSON object. Do not include markdown preamble or postscript."""


SYSTEM_PROMPT_POINTWISE_V1 = """You are an expert evaluation judge scoring a single AI-generated response to an interview question.

Scoring Rules:
1. Assign an absolute quality score between 0.0 and 10.0 (where 0.0 is completely incorrect, 5.0 is acceptable baseline, and 10.0 is flawless mastery).
2. Assess algorithmic correctness, systems feasibility, completeness, and clarity.
3. Provide granular per-criterion scores.

Respond with a single valid JSON object adhering to this schema:
{
  "score": <float between 0.0 and 10.0>,
  "rationale": "<Detailed technical assessment of the response>",
  "criterion_scores": {
    "correctness": <float 0.0 to 10.0>,
    "relevance": <float 0.0 to 10.0>,
    "completeness": <float 0.0 to 10.0>,
    "technical_depth": <float 0.0 to 10.0>,
    "clarity": <float 0.0 to 10.0>,
    "groundedness": <float 0.0 to 10.0>
  }
}

Respond ONLY with the JSON object. Do not include markdown preamble or postscript."""


PROMPT_VERSION_MAP = {
    "pairwise_v1": SYSTEM_PROMPT_PAIRWISE_V1,
    "pairwise_rubric_v1": SYSTEM_PROMPT_PAIRWISE_RUBRIC_V1,
    "pairwise_rationale_first_v1": SYSTEM_PROMPT_PAIRWISE_RATIONALE_FIRST_V1,
    "pointwise_v1": SYSTEM_PROMPT_POINTWISE_V1,
}


class JudgePromptBuilder:
    """Builds structured JudgeRequest payloads for pairwise comparison and pointwise evaluation."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or SYSTEM_PROMPT_TEMPLATE

    def build(
        self,
        case: EvaluationCase,
        output_a: str,
        output_b: str,
        rubric: Optional[Rubric] = None,
        prompt_version: str = "pairwise_v1",
    ) -> JudgeRequest:
        """Construct a JudgeRequest from an evaluation case, presented outputs, and optional rubric."""
        system_prompt = PROMPT_VERSION_MAP.get(prompt_version, self.system_prompt)

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
            "prompt_version": prompt_version,
        }

        return JudgeRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )

    def build_pointwise(
        self,
        case: EvaluationCase,
        output: str,
        rubric: Optional[Rubric] = None,
        prompt_version: str = "pointwise_v1",
    ) -> JudgeRequest:
        """Construct a JudgeRequest for absolute single-answer evaluation (Variant A)."""
        system_prompt = PROMPT_VERSION_MAP.get(prompt_version, SYSTEM_PROMPT_POINTWISE_V1)

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
                user_prompt_lines.append("Criteria to evaluate:")
                for c in rubric.criteria:
                    user_prompt_lines.append(f"- **{c.name}** (weight: {c.weight}): {c.description}")
                user_prompt_lines.append("")

        user_prompt_lines.extend([
            "---",
            "# Candidate Response:",
            output.strip(),
            "",
            "---",
            "Score this response from 0.0 to 10.0 and emit the structured JSON score object.",
        ])

        user_prompt = "\n".join(user_prompt_lines)

        metadata = {
            "case_id": case.id,
            "evaluation_mode": "pointwise",
            "prompt_version": prompt_version,
        }

        return JudgeRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )

