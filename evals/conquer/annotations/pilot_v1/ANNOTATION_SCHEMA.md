# Benizakura Pilot Evaluation — Annotation Schema Specification

> **Document Type:** Machine-Readable Schema Specification  
> **Target Dataset:** 15-Case Pilot Annotation Subset (`pilot_tasks.json`)  
> **Output Artifact:** `pilot_submission_<annotator_id>.json`  
> **Status:** Active / Frozen for Pilot v1

---

## 1. Overview

Each evaluator returns a single JSON file containing a list of **15 evaluation submissions**, one for each case in `pilot_tasks.json`.

Submissions are validated against Benizakura's automated schema validator:
```bash
benizakura benchmark annotation-validate pilot_submission_<annotator_id>.json --pilot
```

---

## 2. Top-Level Structure

The submission file must be a JSON array containing exactly 15 objects:

```json
[
  { /* Submission for case 1 */ },
  { /* Submission for case 2 */ },
  ...
  { /* Submission for case 15 */ }
]
```

Alternatively, a JSON dictionary with a `"submissions"` key containing the 15 objects is also accepted:
```json
{
  "submissions": [
    { /* Submission for case 1 */ },
    ...
  ]
}
```

---

## 3. Case Submission Field Specifications

Each case object in the array requires the following fields:

| Field Name | Type | Allowed Values / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `case_id` | String | Must match one of the 15 pilot case IDs (e.g. `"dsa-002"`) | The unique identifier of the case. |
| `annotator_id` | String | Alphanumeric, hyphens, or underscores (e.g. `"human-001"`, `"human-002"`). **No email addresses or PII.** | Anonymized identifier for the evaluator. |
| `presentation_order` | String | `"randomized"` or `"standard"` (default: `"randomized"`) | Blinded presentation order indicator. |
| `winner` | String | `"A"`, `"B"`, or `"TIE"` | Overall winning response from your blinded view. |
| `confidence` | Number | Float between `0.0` and `1.0`, inclusive | Annotator confidence in determination. Allowed range is 0.0 to 1.0 inclusive. Recommended reference anchors: `0.50` (borderline / near-tie), `0.80` (clear preference), `1.0` (high certainty). Any float within [0.0, 1.0] is valid. |
| `rationale` | String | Minimum length: **15 characters**. Recommended: 2–4 sentences. | Technical explanation justifying your verdict. |
| `criterion_assessments` | Array | Exactly **6 items** corresponding to the standard criteria. | Detailed evaluation breakdown across dimensions. |
| `timestamp` | String | ISO-8601 formatted date-time string (e.g. `"2026-10-12T14:30:00Z"`) | Timestamp of evaluation in ISO-8601 format. Evaluators must replace the template placeholder `"YYYY-MM-DDTHH:MM:SSZ"` with their actual evaluation/submission timestamp before returning. |
| `status` | String | `"SUBMITTED"` | Submission status flag. |

---

## 4. Criterion Assessment Field Specifications

The `criterion_assessments` array must contain exactly 6 objects, one for each of the following criterion names:
- `"CORRECTNESS"`
- `"RELEVANCE"`
- `"COMPLETENESS"`
- `"TECHNICAL_DEPTH"`
- `"CLARITY"`
- `"GROUNDEDNESS"`

Each item in `criterion_assessments` must have:

| Field Name | Type | Allowed Values / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `criterion_name` | String | One of the 6 exact criterion names above. | Dimension being assessed. |
| `winner` | String | `"A"`, `"B"`, or `"TIE"` | Verdict on this specific criterion. |
| `rationale` | String | Minimum length: **10 characters**. | Qualitative justification for this criterion rating. |
| `evidence` | String or Null | A string excerpt from the answer, or `null`. | Optional direct quote supporting the rating. |

---

## 5. Concrete Valid Example

Below is an example of a completed, schema-valid entry for a single case:

```json
{
  "case_id": "dsa-002",
  "annotator_id": "human-001",
  "presentation_order": "randomized",
  "winner": "B",
  "confidence": 0.95,
  "rationale": "Response B provides a concrete, mechanically sound implementation combining a doubly linked list with a hash map and sentinel nodes. Response A is overly verbose and fails to give explicit pointer manipulation details.",
  "criterion_assessments": [
    {
      "criterion_name": "CORRECTNESS",
      "winner": "B",
      "rationale": "Response B correctly specifies O(1) eviction using sentinel nodes; Response A is hand-wavy.",
      "evidence": "node.prev.next = node.next; node.next.prev = node.prev"
    },
    {
      "criterion_name": "RELEVANCE",
      "winner": "TIE",
      "rationale": "Both responses address the LRU cache design prompt.",
      "evidence": null
    },
    {
      "criterion_name": "COMPLETENESS",
      "winner": "B",
      "rationale": "Response B details both get and put operations with eviction logic.",
      "evidence": "If size > capacity, evict tail.prev"
    },
    {
      "criterion_name": "TECHNICAL_DEPTH",
      "winner": "B",
      "rationale": "Response B explains sentinel dummy nodes and explicit pointer detachment.",
      "evidence": null
    },
    {
      "criterion_name": "CLARITY",
      "winner": "B",
      "rationale": "Response B uses clear structured bullet points instead of narrative padding.",
      "evidence": null
    },
    {
      "criterion_name": "GROUNDEDNESS",
      "winner": "TIE",
      "rationale": "Both answers reference standard software caching patterns.",
      "evidence": null
    }
  ],
  "timestamp": "2026-10-11T10:00:00Z",
  "status": "SUBMITTED"
}
```

---

## 6. Validation Checklist Before Returning

Before submitting:
1. Verify that all 15 cases from `submission_template.json` are present in your file.
2. Verify that your `"annotator_id"` is set consistently across all 15 cases (e.g. `"human-001"`).
3. Ensure no `"winner"` or criterion `"winner"` is left empty.
4. Ensure all rationales meet the minimum character lengths (>= 15 characters for case rationale, >= 10 characters for criteria).
5. Replace the template's `"YYYY-MM-DDTHH:MM:SSZ"` placeholder in `"timestamp"` with your actual evaluation timestamp in ISO-8601 format (e.g., `"2026-10-12T14:30:00Z"`).
6. Verify JSON syntax has no syntax errors (valid commas, closed braces, matching quotes).
