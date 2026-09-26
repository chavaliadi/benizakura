# Benizakura — Human Ground-Truth Annotation Schema Specification

> **Document Type:** Specification & Schema Definition  
> **Status:** Active / Frozen for Benchmark v1.0.0  
> **Target Dataset:** [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json)  
> **Target Application:** Conquer (`POST /api/interview/score`)

---

## 1. Overview & Purpose

This document formally specifies the machine-readable schema and data lifecycle for human ground-truth annotations in Benizakura. To scientifically validate automated LLM judges (such as Anthropic Claude 3.5 Sonnet and OpenAI GPT-4o), human labels must be captured independently, double-blinded, and stored in a structured format prior to running automated evaluation pipelines.

---

## 2. Annotation Lifecycle

Every benchmark evaluation case undergoes an explicit five-state annotation lifecycle:

```text
                  ┌──────────────┐
                  │   PENDING    │
                  └──────┬───────┘
                         │ Assign to 2 independent engineers
                         ▼
                  ┌──────────────┐
                  │ IN_PROGRESS  │
                  └──────┬───────┘
                         │ Both annotators submit independently
                         ▼
                  ┌──────────────┐
                  │  SUBMITTED   │
                  └──────┬───────┘
                         │
        ┌────────────────┴────────────────┐
        │ Agreement Check                 │
        ▼                                 ▼
   [Ann 1 == Ann 2]                [Ann 1 != Ann 2]
        │                                 │
        ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│  CONSENSED   │                  │ ADJUDICATED  │
│ (Automatic)  │                  │ (Lead Review)│
└──────────────┘                  └──────────────┘
```

### Lifecycle States
1. **`PENDING`**: Benchmark cases are frozen, but human evaluation has not yet commenced. No ground-truth label exists.
2. **`IN_PROGRESS`**: Case has been partitioned and distributed to two double-blind human annotators.
3. **`SUBMITTED`**: Both Annotator 1 and Annotator 2 have completed and submitted independent assessments.
4. **`CONSENSED`**: Annotator 1 and Annotator 2 independently selected the identical winner (`A == B`). The consensus label is finalized automatically.
5. **`ADJUDICATED`**: Annotator 1 and Annotator 2 selected different winners (e.g. `A` vs `B`, or `A` vs `TIE`). An Engineering Lead reviewed both arguments, conducted an adjudication review, and assigned the authoritative final consensus label.

---

## 3. Double-Blind Presentation Protocol

Human annotators must never see which response originates from the `baseline` system or the `candidate` system.

### Presentation Format
```json
{
  "task_id": "blind-dsa-001",
  "case_id": "dsa-001",
  "category": "DSA",
  "subcategory": "Arrays & Two Pointers",
  "difficulty": "medium",
  "question": "Given a 1-indexed sorted array of integers numbers and an integer target...",
  "context": "Technical interview screening for software engineering candidates.",
  "response_a": "<Response A text>",
  "response_b": "<Response B text>",
  "accepted_alternatives": [
    {
      "approach": "Two Pointers (left & right)",
      "conditions": "Array is sorted; optimal O(N) time and O(1) space.",
      "tradeoffs": "Directly satisfies prompt constraints."
    }
  ]
}
```

The mapping between `{A, B}` and `{BASELINE, CANDIDATE}` is generated via `benizakura benchmark blind` and stored in a separate, encrypted key file not accessible to annotators during labeling.

---

## 4. Annotator Submission Schema

Each independent annotator produces a submission JSON adhering to this schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AnnotatorSubmission",
  "type": "object",
  "required": [
    "case_id",
    "annotator_id",
    "winner",
    "confidence",
    "rationale",
    "criterion_assessments",
    "timestamp",
    "status"
  ],
  "properties": {
    "case_id": {
      "type": "string",
      "description": "Unique benchmark case ID (e.g., 'dsa-001')."
    },
    "annotator_id": {
      "type": "string",
      "pattern": "^human-[0-9]{3}$",
      "description": "Anonymized identifier for the human engineer (e.g. 'human-001')."
    },
    "presentation_order": {
      "type": "string",
      "enum": ["randomized", "standard"],
      "default": "randomized"
    },
    "winner": {
      "type": "string",
      "enum": ["A", "B", "TIE"],
      "description": "Selected winner in blinded presentation view."
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Annotator confidence in determination (0.50 = uncertain, 1.0 = absolute certainty)."
    },
    "rationale": {
      "type": "string",
      "minLength": 20,
      "description": "Qualitative justification explaining why the winning response was chosen over the alternative."
    },
    "criterion_assessments": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["criterion_name", "winner", "rationale"],
        "properties": {
          "criterion_name": {
            "type": "string",
            "enum": [
              "CORRECTNESS",
              "RELEVANCE",
              "COMPLETENESS",
              "TECHNICAL_DEPTH",
              "CLARITY",
              "GROUNDEDNESS"
            ]
          },
          "winner": {
            "type": "string",
            "enum": ["A", "B", "TIE"]
          },
          "rationale": {
            "type": "string",
            "minLength": 10
          },
          "evidence": {
            "type": ["string", "null"],
            "description": "Direct excerpt or quotation from the response justifying this criterion score."
          }
        }
      }
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "status": {
      "type": "string",
      "enum": ["SUBMITTED"]
    }
  }
}
```

---

## 5. Evaluation Criteria Definitions

Annotators evaluate candidate answers across six core dimensions:

1. **`CORRECTNESS`**: Algorithmic correctness, system feasibility, absence of hallucinations, and accurate asymptotic bounds ($O(N)$, etc.).
2. **`RELEVANCE`**: Directness in addressing the interview prompt without evasive preamble or irrelevant tangents.
3. **`COMPLETENESS`**: Addressing all explicit requirements of the question, including edge cases, memory constraints, and boundary conditions.
4. **`TECHNICAL_DEPTH`**: Demonstration of mechanical sympathy, low-level execution understanding, and concrete trade-off analysis (e.g. locks, indexes, page splits).
5. **`CLARITY`**: Communication structure, logical progression, and technical precision.
6. **`GROUNDEDNESS`**: Whether the candidate's claims and metrics are physically and mathematically defensible in real-world systems.

---

## 6. Consensus & Adjudication Container Schema

Once unblinded, the final consensus container inside `benchmark_v1.json` is populated as:

```json
{
  "status": "CONSENSED",
  "annotator_1": {
    "annotator_id": "human-001",
    "raw_winner": "A",
    "normalized_winner": "CANDIDATE",
    "confidence": 0.95,
    "rationale": "Candidate provides optimal O(N) two-pointer pass; Baseline uses brute force O(N^2)."
  },
  "annotator_2": {
    "annotator_id": "human-002",
    "raw_winner": "A",
    "normalized_winner": "CANDIDATE",
    "confidence": 0.90,
    "rationale": "Response A solves prompt in O(1) space and O(N) time; Response B fails complexity constraint."
  },
  "consensus": {
    "winner": "CANDIDATE",
    "agreement_type": "UNANIMOUS",
    "cohen_kappa_contribution": 1.0,
    "adjudicator_id": null,
    "adjudication_rationale": null,
    "timestamp": "2026-09-27T10:00:00Z"
  }
}
```

If the annotators disagreed (e.g., Annotator 1 selected `CANDIDATE`, Annotator 2 selected `TIE`):

```json
{
  "status": "ADJUDICATED",
  "annotator_1": {
    "annotator_id": "human-001",
    "normalized_winner": "CANDIDATE"
  },
  "annotator_2": {
    "annotator_id": "human-002",
    "normalized_winner": "TIE"
  },
  "consensus": {
    "winner": "CANDIDATE",
    "agreement_type": "ADJUDICATED",
    "adjudicator_id": "lead-human-000",
    "adjudication_rationale": "Annotator 2 felt both answers were valid, but Response A explicitly handles cache stampede using probabilistic XFetch while Response B only suggests general locks. Candidate is meaningfully superior.",
    "timestamp": "2026-09-27T11:30:00Z"
  }
}
```

---

## 7. Storage & Directory Structure

All human annotation artifacts are organized within the repository under:

```text
evals/conquer/
├── benchmark_v1.json               # Canonical 30 cases with human_annotation containers
├── benchmark_v1.manifest.json      # Metadata manifest and status
├── benchmark_v1.sha256             # Frozen cryptographic checksum
└── annotations/
    ├── ANNOTATION_SCHEMA.md        # This specification document
    ├── ANNOTATOR_GUIDE.md          # Handbook for human engineers
    ├── blinding_key_v1.json        # Private unblinding key (Restricted access)
    ├── submissions/                # Raw submitted JSONs from annotators
    │   ├── human_001/
    │   └── human_002/
    └── adjudicated/                # Disagreement review records
```
