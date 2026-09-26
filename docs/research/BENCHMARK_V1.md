# Benizakura — Official 30-Case Benchmark & Human Ground-Truth Specification (`benchmark_v1`)

> **Document Type:** Benchmark Specification & Dataset Governance  
> **Status:** Frozen Pending Human Annotation (`FROZEN_PENDING_HUMAN_ANNOTATION`)  
> **Target System:** Conquer (`POST /api/interview/score`)  
> **Dataset File:** [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json)  
> **Manifest File:** [`evals/conquer/benchmark_v1.manifest.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.manifest.json)  
> **Checksum File:** [`evals/conquer/benchmark_v1.sha256`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.sha256)  
> **Accompanying Research:** [`docs/research/LLM_JUDGE_SELECTION.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/research/LLM_JUDGE_SELECTION.md)

---

## 1. Benchmark Purpose

The primary objective of `benchmark_v1` is to provide a scientifically controlled, double-blind evaluation dataset that enables Benizakura to validate automated LLM-as-a-Judge pipelines before trusting them in CI/CD release gating.

Specifically, this benchmark is constructed to empirically answer:
1. **Human Agreement:** Does the automated judge (specifically Claude 3.5 Sonnet or GPT-4o) achieve substantial agreement (**Cohen's $\kappa \ge 0.60$**) with expert human software engineers on technical interview evaluations?
2. **Component Attribution:** Does each architectural component of Benizakura (bidirectional position swapping, rubric decomposition, criterion evidence, paired bootstrap confidence intervals) measurably improve evaluation reliability over naive pointwise or unconstrained pairwise baselines?
3. **Bias Resilience:** How resilient is the evaluation pipeline to known LLM biases: position bias, verbosity bias, authoritative hallucinations, and architectural favoritism?

---

## 2. Case Distribution & Stratification

The benchmark consists of exactly **30 evaluation cases**, stratified across five core technical engineering interview tracks:

```text
Track / Category           Count   Case ID Range             Primary Focus Areas
---------------------------------------------------------------------------------------------------------
DSA                        6       dsa-001  .. dsa-006       Asymptotic complexity, pointer manipulation,
                                                             boundary edge cases, graph theory.
System Design              8       sys-001  .. sys-008       Distributed ID generation, rate limiters,
                                                             fan-out, caching stampedes, CDN, idempotency.
Backend Engineering        6       backend-001 .. backend-006 Concurrency control, MVCC isolation,
                                                             composite indexing, connection pooling, zero-downtime DDL.
Frontend Engineering       4       frontend-001 .. frontend-004 React re-renders/closures, DOM virtualization,
                                                             Core Web Vitals (LCP/INP/CLS), server vs client state.
Behavioral / STAR          6       behavioral-001 .. behavioral-006 P0 outage leadership, technical disagreements,
                                                             tech debt vs velocity, coaching, accountability.
---------------------------------------------------------------------------------------------------------
TOTAL                      30
```

Every case is assigned an immutable, structured ID prefixed by its track (`dsa-`, `sys-`, `backend-`, `frontend-`, `behavioral-`).

---

## 3. Deliberate Bias Probes

Rather than containing generic textbook problems, every benchmark case is engineered with explicit bias probes targeting vulnerabilities documented in literature (Zheng et al., 2023; Wang et al., 2024; Dubois et al., 2024):

| Probe Type | Cases Probed | Experimental Mechanism & Testing Goal |
| :--- | :--- | :--- |
| **Position Bias** (`position`) | All 30 cases (100%) | Every pair is evaluated in both $(A, B)$ and $(B, A)$ orders to measure order-instability rate (`unstable_rate`). |
| **Verbosity Bias** (`verbosity`) | `dsa-002`, `sys-006`, `backend-003`, `frontend-002`, `behavioral-004` (5 cases) | Compares a concise, mathematically exact answer (100–140 words) against a bloated, superficial answer (380–450 words) to detect if the judge favors length over substance. |
| **Authoritative Hallucination** (`hallucination`) | `dsa-005`, `sys-005`, `backend-006`, `behavioral-006` (4 cases) | Compares a polished, confident answer containing a subtle fatal technical error against a plain, accurate answer. |
| **Near-Tie Calibration** (`near_tie`) | `dsa-003`, `sys-001`, `sys-004`, `backend-004`, `frontend-003`, `frontend-004`, `behavioral-005` (7 cases) | Both responses present valid, balanced trade-offs. The correct ground-truth label is `TIE`, testing if the judge forces an arbitrary win. |
| **Alternative Architecture** (`alternative_architecture`) | `dsa-003`, `sys-001`, `backend-004`, `frontend-004` (4 cases) | Both solutions solve the problem using different valid paradigms (e.g. KGS vs Snowflake; TanStack Query vs Redux; SQS vs Redis Streams). |
| **Criterion Trade-off** (`criterion_tradeoff`) | `dsa-004`, `sys-003`, `backend-005`, `behavioral-003` (4 cases) | One response improves one criterion (e.g. theoretical time complexity) while regressing another (e.g. operational simplicity or memory). |
| **Subtle Regression** (`subtle_regression`) | `sys-002`, `backend-001`, `frontend-001` (3 cases) | Candidate appears clean but introduces a race condition (Redis GET/SET), a transaction locking misconception, or a React hook stale closure. |
| **Edge-Case Failure** (`edge_case_failure`) | `dsa-006`, `sys-008` (2 cases) | Candidate works on happy paths but fails on touching boundary intervals (`[1, 4]` and `[4, 5]`) or external gateway timeouts. |
| **Complexity Regression** (`complexity_regression`) | `dsa-001` (1 case) | Candidate regresses time complexity from $O(N)$ two-pointer to $O(N^2)$ brute-force loop. |
| **Clearly Better Candidate** (`clearly_better`) | `sys-007`, `backend-002`, `behavioral-001`, `behavioral-002` (4 cases) | Meaningful, unambiguous technical superiority in candidate response. |

---

## 4. Human Annotation Methodology

To avoid benchmark contamination and biased labels, human ground truth is generated via a strict double-annotated procedure:

1. **Two Independent Senior Engineers:** Each case is independently reviewed by two senior engineers with $>5$ years of domain experience.
2. **Double-Blind Presentation:** Annotators receive responses labeled strictly as `Response A` and `Response B`.
3. **Independent Rating:** Annotators submit their evaluations into isolated files without visibility into peer ratings.
4. **Adjudication:** Disagreements between Annotator 1 and Annotator 2 trigger an adjudication review led by an Engineering Lead.

Complete instructions are documented in [`evals/conquer/annotations/ANNOTATOR_GUIDE.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/annotations/ANNOTATOR_GUIDE.md).

---

## 5. Blinding & Presentation Randomization

To eliminate human evaluator bias, Benizakura provides an automated blinding engine:

```bash
# Generate double-blind tasks and secure unblinding key
benizakura benchmark blind evals/conquer/benchmark_v1.json \
  --out-tasks evals/conquer/annotations/blinded_tasks.json \
  --out-key evals/conquer/annotations/blinding_key_v1.json \
  --seed 42
```

* **Blinded Tasks File:** Contains `case_id`, `question`, `context`, `response_a`, `response_b`, and `accepted_alternatives`.
* **Private Key File:** Contains the cryptographic map (`A -> BASELINE`, `B -> CANDIDATE` or vice-versa) generated via pseudorandom seed. This key is stored separately and unblinded only during consensus aggregation.

---

## 6. Consensus & Adjudication Procedure

```text
Annotator 1 (human-001)           Annotator 2 (human-002)
          \                                 /
           \                               /
            ▼                             ▼
       [Winner: A]                   [Winner: A]
            \                             /
             └─── Agreement? YES ────────┘
                        │
                        ▼
            Consensus Winner: CANDIDATE
            Agreement Type:   UNANIMOUS
            Status:           CONSENSED
```

If Annotator 1 and Annotator 2 disagree:
```text
       [Winner: A]                   [Winner: TIE]
            \                             /
             └─── Agreement? NO ─────────┘
                        │
                        ▼
            Engineering Lead Adjudication
            (Reviews arguments & writes rationale)
                        │
                        ▼
            Consensus Winner: CANDIDATE
            Agreement Type:   ADJUDICATED
            Status:           ADJUDICATED
```

---

## 7. Ground-Truth Governance & Scientific Integrity

### Strict Separation of Evaluation Target from Ground Truth
The benchmark strictly decouples what the LLM judge sees from the ground-truth metadata:

```text
Benchmark Case
  ├── question, context, baseline_answer, candidate_answer ───► Extracted to EvaluationCase (Judge Input)
  └── human_annotation, intended_signal, bias_probes       ───► Retained in Ground Truth Store (Judge Blind)
```

The conversion method [`BenchmarkCase.to_evaluation_case()`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/benchmark.py) programmatically strips:
* `human_annotation` (status, annotators, consensus, winner)
* `metadata.intended_signal`
* `metadata.primary_failure_mode`
* `metadata.expected_quality`

### Prevention of Retrospective Labeling
Human labels must be established and frozen **before** running any model evaluations. We explicitly reject the invalid methodology of running Claude or GPT-4o first and assuming model outputs represent ground truth.

---

## 8. Offline Validation & Leakage Safeguards

Benizakura includes a dedicated CLI command to validate benchmark structure, stratification, and leakage protection:

```bash
benizakura benchmark validate evals/conquer/benchmark_v1.json
```

The validator verifies:
1. **Structural Integrity:** Exactly 30 cases, unique IDs, correct category breakdown (DSA: 6, System Design: 8, Backend: 6, Frontend: 4, Behavioral: 6).
2. **Content Validity:** No empty fields, no duplicate questions, valid difficulty and probe values.
3. **Annotation Integrity:** Verifies no illegal premature consensus exists when status is `PENDING`.
4. **Leakage Detection:** Simulates extraction to `EvaluationCase` and scans metadata and question strings for leaked ground-truth tokens.

---

## 9. Deterministic Cryptographic Hashing

To ensure the benchmark cannot be silently mutated after calibration, Benizakura enforces SHA-256 integrity verification:

```bash
benizakura benchmark hash evals/conquer/benchmark_v1.json --write
```

* **Canonical SHA-256:** `aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`
* Any modification to whitespace, casing, or case text will alter this hash, immediately failing the test suite (`tests/test_benchmark.py`).

---

## 10. Versioning & Dataset Immutability

* **Benchmark ID:** `conquer-benchmark-v1`
* **Version:** `1.0.0`
* **Immutability Contract:** Once human annotations are recorded, this dataset version is frozen permanently. Any future revisions or prompt-specific additions must be released under a new version identifier (e.g. `benchmark_v2.json`).

---

## 11. Calibration Kill-Gate

In strict accordance with [DECISIONS.md (D8)](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/DECISIONS.md#L79-L90):

> **Judge calibration is an absolute kill-gate before enabling automated CI release gating.**

1. Automated release gating (`PASS` / `FAIL` / `INCONCLUSIVE`) remains **DISABLED**.
2. Before the automated gate can be activated, the primary judge must be executed against the double-annotated consensus labels of this benchmark.
3. If the judge achieves **Cohen's $\kappa \ge 0.60$** (substantial agreement) and **position instability rate $\le 20\%$**, the calibration gate passes.
4. If $\kappa < 0.60$, the judge prompt and rubric criteria must be refined and re-evaluated against the calibration split; automated gating will not be enabled.

---

## 12. Current Status & Next Milestones

```text
Benchmark Cases:              READY (30/30 frozen)
Case Stratification:          VERIFIED (DSA: 6, Sys: 8, Backend: 6, Frontend: 4, Behavioral: 6)
Integrity Hash:               aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314
Human Annotations:            PENDING
Human Consensus:              PENDING
Cohen's Kappa (κ):            NOT YET MEASURED
Judge Calibration:            NOT YET PERFORMED
Production CI Gating:         DISABLED
```

### Next Immediate Milestones:
1. **Double-Blind Human Annotation Phase:** Distribute blinded tasks to two independent senior engineers; collect submissions.
2. **Consensus & Adjudication:** Resolve any inter-rater disagreements and freeze ground-truth consensus labels in `benchmark_v1.json`.
3. **Judge Execution & Baseline Comparison:** Connect offline adapters for Claude 3.5 Sonnet and GPT-4o, measure inter-rater reliability ($\kappa$), and test research hypotheses H1–H5.
