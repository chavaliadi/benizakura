# Benizakura

> **CI for AI Behaviour — Detecting silent regressions in LLM-powered applications.**

**Benizakura** is a research-first evaluation and regression-testing change-gate framework designed specifically for Large Language Model (LLM) powered applications.

The central research thesis is:

> **Can explicit bidirectional evaluation, rubric-based evidence, human calibration, statistical uncertainty modeling, and deterministic release gating make LLM evaluation reliable enough for automated continuous integration (CI)?**

---

## What is Benizakura?

In simple terms: **Benizakura is CI for AI behaviour.**

The goal of Benizakura is not simply to score LLM responses on arbitrary 1–10 scales, but to detect when an AI change (a modified prompt, new model version, temperature tweak, or updated retrieval pipeline) appears to improve one behaviour while silently degrading another.

---

## Why Does It Exist?

In traditional software engineering, code modifications are validated by deterministic test suites:

```text
Code Change ──▶ Run Test Suite ──▶ Deterministic Assertion ──▶ PASS or FAIL
```

If a function computing tax rates has a bug, an assertion fails, the build breaks, and deployment stops.

In LLM-powered applications, traditional software tests are almost completely blind to quality regressions:

```text
Prompt Edit ──▶ Run Test Suite ──▶ API Returns 200 OK ──▶ Unit Tests Pass ✅
                                                          AI Behaviour Silently Regressed ❌
```

When an engineer modifies a prompt or configuration:
* The HTTP response remains 200 OK.
* The JSON matches the schema.
* No exceptions are thrown.

Yet in reality, the change may:
* Improve answers on common DSA questions while hallucinating on distributed systems.
* Introduce severe **position bias** (favoring whichever answer is presented first).
* Reward **verbosity** (favoring long, low-information padding over concise correctness).
* Produce **unstable judgments** where small context shifts cause decision flips.
* Trade off precision on critical criteria (e.g., correctness) for superficial gains in fluency.

Benizakura exists to make these probabilistic, multidimensional behavioral regressions measurable, statistically rigorous, and actionable before deployment.

---

## Current Target Application: Conquer

The initial real-world evaluation target for Benizakura is **Conquer**, an AI-powered technical interview preparation simulator.

Specifically, the system targets Conquer's core evaluation endpoint:
`POST /api/interview/score` (Technical Answer Scorer & Candidate Profiler).

In Conquer, this feature:
1. Evaluates candidate responses across technical interview domains (DSA, System Design, Backend, Frontend, Behavioral).
2. Generates qualitative diagnostic feedback and criterion ratings on a 0.0–10.0 scale.
3. Updates candidate skill profiles that drive adaptive question selection.

> **Architecture Boundary:** While Conquer serves as the initial benchmark domain, Benizakura's evaluation engine, statistical models, and release gates are domain-agnostic.

---

## Current Architecture

### 1. Operational Release-Gating Pipeline

When evaluating an application update against a baseline:

```text
Evaluation Cases
       ↓
Pairwise Judge
       ↓
Bidirectional Evaluation (Forward + Reverse presentation)
       ↓
Order Normalization (Mapping raw A/B back to system identity)
       ↓
Instability Detection (Surfacing order-induced flip rate)
       ↓
Statistical Analysis (Paired deltas, bootstrap CI, effect size)
       ↓
Release Gate (Predefined precedence hierarchy)
       ↓
PASS / FAIL / INCONCLUSIVE
```

### 2. Research Calibration Pipeline

Before any LLM judge is trusted to drive automated release gating, it must pass empirical calibration against human consensus:

```text
Frozen Benchmark (Stratified 30-case dataset with deliberate bias probes)
       ↓
Blind Human Annotation (Two independent annotators, blind A/B presentation)
       ↓
Human-Human Agreement (Inter-rater reliability, Cohen's κ >= 0.60)
       ↓
Adjudication (Senior engineer consensus on disagreements)
       ↓
Frozen Human Ground Truth (Immutable ground-truth artifact + SHA-256)
       ↓
LLM Judge Calibration (Evaluating LLM configurations against ground truth)
       ↓
Controlled Experiments (Variants A–E, H1–H5 empirical validation)
```

---

## Current Phase 3 Status

Benizakura is currently at the conclusion of the implementation-heavy portion of **Phase 3**. All infrastructure, benchmark assets, human calibration pipelines, provider abstractions, and dry-run safety gates are complete and verified.

```text
Phase 3 — Research & Calibration Infrastructure

Status: INFRASTRUCTURE COMPLETE | BENCHMARK FROZEN PENDING HUMAN ANNOTATION

Completed Infrastructure:
✅ LLM judge research specification (Anthropic Claude 3.5 Sonnet vs. OpenAI GPT-4o)
✅ Frozen 30-case Conquer benchmark with verified SHA-256
✅ Benchmark leakage auditing and strict case stratification
✅ Blind human annotation task generation (two-sided blinded A/B presentation) & key management
✅ Blind annotation submission validation against JSON schema
✅ Human-human inter-rater agreement computation & Cohen's κ
✅ Disagreement adjudication and consensus compilation
✅ Ground-truth artifact generation and integrity verification
✅ Multi-provider LLM abstraction (Anthropic, OpenAI, Deterministic Mock)
✅ Five controlled experiment variants (A, B, C, D, E)
✅ Prompt schema versioning (Winner-First, Rubric, Rationale-First)
✅ Dry-run workload planner & token/cost estimator
✅ Human-ground-truth prerequisite gate (Cohen's κ >= 0.60 required)
✅ Recursive secret sanitization across manifests and logs
✅ Paired bootstrap confidence intervals and statistical analysis

Pending Empirical Work:
⏳ Real human annotation (two independent human annotators)
⏳ Human-human Cohen's κ measurement
⏳ Consensus & disagreement adjudication
⏳ Frozen human ground truth artifact
⏳ Live LLM calibration execution (Variants A–E)
⏳ Empirical validation of hypotheses H1–H5
```

> [!IMPORTANT]
> **Scientific Integrity Guarantee:**
> * Actual human annotations are **PENDING**.
> * Human-human Cohen's $\kappa$ is **NOT YET MEASURED**.
> * Ground truth is **NOT FROZEN** until real human annotation and adjudication occur.
> * Live LLM experiments are **NOT EXECUTED**.
> * Production CI gating remains **DISABLED** until empirical calibration evidence is established.
> * No human labels are fabricated or generated by LLMs.

---

## Benchmark Details: Conquer Benchmark v1

The official benchmark dataset is frozen at [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json).

```text
Benchmark ID:      conquer-benchmark-v1
Version:           1.0.0
Total Cases:       30
Canonical SHA-256: aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314
Status:            FROZEN
```

### Domain Stratification

| Category | Cases | Focus Areas |
| :--- | :---: | :--- |
| **DSA** | 6 | Asymptotic complexity, edge cases, recursion depth, cycle detection, algorithmic regressions |
| **System Design** | 8 | Partitioning, consensus protocols, cache invalidation, availability vs. consistency, write paths |
| **Backend** | 6 | Concurrency, idempotency, database migrations, connection pool starvation, transaction isolation |
| **Frontend** | 4 | State synchronization, bundle bloat, render waterfalls, accessibility, reactivity cycles |
| **Behavioral** | 6 | STAR structure, conflict resolution, technical ownership, failure postmortems, engineering ethics |

### Position-Order Evaluation & Deliberate Bias Probes

* **Position-order evaluation (all 30 cases):** Every case can be evaluated in both A/B orientations to measure order sensitivity.

Separately, the benchmark incorporates targeted, deliberate failure-mode probes across specific cases:
* **Verbosity Bias Probes (5 cases):** Verbose, low-information candidate answers paired against concise, technically accurate baselines.
* **Authoritative Hallucination Probes (4 cases):** Fluent, confident answers asserting plausible but non-existent APIs, incorrect algorithmic complexities, or false architectural guarantees (`dsa-005`, `sys-005`, `backend-006`, `behavioral-006`).
* **Near Ties (7 cases):** Semantically balanced pairs testing whether judges force arbitrary, uncalibrated wins.
* **Alternative Valid Architectures (4 cases):** Divergent but technically sound designs (e.g., event-driven vs. polling) to check if the judge penalizes non-textbook solutions.
* **Criterion Trade-offs (4 cases):** Responses excelling on clarity but lacking technical depth, testing rubric decomposition.

---

## Human Ground-Truth Calibration Protocol

An automated judge cannot be evaluated without an independent, calibrated baseline. Benizakura enforces a formal human calibration protocol:

1. **Two Independent Annotators:** Qualified software engineers annotate all 30 benchmark cases independently.
2. **Blind A/B Presentation:** Annotators receive a two-sided blinded presentation of randomized `Option A` and `Option B`. Annotators do not know which response is baseline or candidate, do not receive the intended benchmark probe classification or expected outcome, and have no visibility into the other annotator's ratings.
3. **Structured Rubric Evaluation:** Annotators record per-criterion ratings (`CORRECTNESS`, `RELEVANCE`, `COMPLETENESS`, `TECHNICAL_DEPTH`, `CLARITY`, `GROUNDEDNESS`) and overall preference (`A`, `B`, or `TIE`).
4. **Inter-Annotator Agreement (Cohen's $\kappa$):** Multi-class Cohen's $\kappa$ is computed on overall preference across $\{\text{CANDIDATE}, \text{BASELINE}, \text{TIE}\}$.
5. **Predefined Calibration Threshold:** If $\kappa < 0.60$, the predefined calibration gate fails and automated judge calibration is **strictly blocked**. The disagreement patterns must then be reviewed to determine whether ambiguity in the benchmark, annotation guide, or evaluator interpretation contributed.
6. **Disagreement Adjudication:** Disagreements between Annotator 1 and Annotator 2 are reviewed by a third senior engineer (Adjudicator) to produce a finalized consensus ground truth.
7. **Ground-Truth Freezing:** Once adjudicated, the ground-truth artifact (`evals/conquer/ground_truth_v1.json`) is checksummed and frozen.

> **Key Rule:** No human labels are generated automatically. The final ground truth must come from real human annotation and adjudication.

---

## LLM Judge Calibration & Research Hypotheses

Live LLM calibration experiments are planned and ready for execution once human ground truth is frozen.

### Evaluated Configurations (Intended Research Specification)

The initial research specification defines the following intended experiment configurations:

* **Primary Candidate (`judge_a_claude`):** Anthropic Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`), temperature 0.0. Represents an **independent model family** relative to Conquer's OpenAI-based scorer.
* **Cross-Evaluator (`judge_b_gpt4o`):** OpenAI GPT-4o (`gpt-4o-2024-08-06`), temperature 0.0. Represents a **same-family evaluator** to quantify self-enhancement bias.

> [!NOTE]
> **Model Snapshot Availability & Tracking:**
> These snapshot identifiers represent the initial research specification and intended experiment configuration, not a guarantee that these exact dated provider snapshots remain indefinitely accessible via external APIs. Before executing live experiments:
> * Provider endpoint and model snapshot availability must be verified.
> * The exact model snapshot actually used must be recorded in the experiment manifest.
> * Any required substitution must be explicitly documented and version-tracked rather than silently swapped.

### Five Controlled Experiment Variants

```text
Variant A (Pointwise)
   ↓
Variant B (Single-Direction Pairwise)
   ↓
Variant C (Bidirectional Pairwise)
   ↓
Variant D (Rubric-Based Bidirectional Pairwise)
   ↓
Variant E (Full Benizakura: Rubric + Evidence + Bidirectional + Paired Bootstrap CI)
```

1. **Variant A — Simple Pointwise (`POINTWISE`):** Independent scalar scoring ($0.0 \dots 10.0$) mapped to win/loss via score delta.
2. **Variant B — Single-Direction Pairwise (`SINGLE_DIRECTION`):** Single presentation order ($A = \text{Baseline}, B = \text{Candidate}$) without order controls.
3. **Variant C — Bidirectional Pairwise (`BIDIRECTIONAL`):** Forward and reverse passes with order normalization; flags order flips (`is_unstable = True`).
4. **Variant D — Rubric-Based Bidirectional (`RUBRIC_BIDIRECTIONAL`):** Bidirectional evaluation decomposed across explicit criteria.
5. **Variant E — Full Benizakura (`FULL_BENIZAKURA`):** Complete architecture combining rubric decomposition, direct quotation evidence, bidirectional order normalization, position instability gating, and paired bootstrap confidence intervals.

### Research Hypotheses Under Empirical Test

All methodological enhancements are treated as formal, falsifiable hypotheses:

* **Hypothesis H1 (Position Debiasing):** Bidirectional evaluation combined with order normalization produces a lower position-dependent decision error rate than single-direction evaluation.  
  *Status:* **HYPOTHESIS — NOT YET TESTED.**
* **Hypothesis H2 (Rubric Decomposition Efficacy):** Decomposing pairwise evaluation into structured, criterion-level assessments improves Cohen's $\kappa$ agreement with expert human evaluators compared to an unconstrained pairwise judge.  
  *Status:* **HYPOTHESIS — NOT YET TESTED.**
* **Hypothesis H3 (Rationale-First Schema):** Emitting per-criterion evaluations and qualitative rationale *prior* to emitting the final categorical winner token reduces position-instability rates compared to winner-first schemas.  
  *Status:* **HYPOTHESIS — NOT YET TESTED.**
* **Hypothesis H4 (Evaluator Independence):** When evaluating candidate answers generated by a GPT-family model, an independent-family judge (Claude 3.5 Sonnet) exhibits lower false-pass rates on subtle technical hallucinations than a same-family judge (GPT-4o).  
  *Status:* **HYPOTHESIS — NOT YET TESTED.**
* **Hypothesis H5 (Statistical Gating False-Alarm Suppression):** Incorporating paired bootstrap confidence intervals and an explicit `INCONCLUSIVE` verdict reduces false regression alarms compared to naive point-estimate thresholding.  
  *Status:* **HYPOTHESIS — NOT YET TESTED.**

---

## Reproducibility & CLI Reference

### 1. Test Suite Verification

Run the complete deterministic test suite:

```bash
pytest -v
```

*198 tests pass in ~0.2s with zero external API calls or network dependencies.*

---

### 2. Benchmark Utilities

```bash
# Validate benchmark structure, category counts, and leakage
benizakura benchmark validate evals/conquer/benchmark_v1.json

# Compute canonical deterministic SHA-256
benizakura benchmark hash evals/conquer/benchmark_v1.json

# Generate blind annotation tasks and private unblinding key
benizakura benchmark blind \
  --benchmark evals/conquer/benchmark_v1.json \
  --output-tasks evals/conquer/annotations/tasks_blind.json \
  --output-key evals/conquer/annotations/blinding_key.json

# Validate annotator submissions against schema
benizakura benchmark annotation-validate evals/conquer/annotations/submissions

# Check annotation progress and coverage
benizakura benchmark annotation-status evals/conquer/annotations/submissions

# Compute human-human agreement (Cohen's κ) and check the calibration gate
benizakura benchmark agreement \
  --submissions-dir evals/conquer/annotations/submissions \
  --key evals/conquer/annotations/blinding_key.json

# Adjudicate disagreements and compile consensus ground truth
benizakura benchmark consensus \
  --submissions-dir evals/conquer/annotations/submissions \
  --key evals/conquer/annotations/blinding_key.json \
  --adjudications evals/conquer/annotations/adjudications.json \
  --output evals/conquer/ground_truth_v1.json

# Inspect and verify ground-truth artifact status
benizakura benchmark ground-truth status --path evals/conquer/ground_truth_v1.json
benizakura benchmark ground-truth verify --path evals/conquer/ground_truth_v1.json
```

---

### 3. Experiment Planning & Calibration

```bash
# Check calibration prerequisites (reports blocked status while annotation is pending)
benizakura experiment validate \
  --benchmark evals/conquer/benchmark_v1.json \
  --ground-truth evals/conquer/ground_truth_v1.json

# Generate declarative experiment workload plan (dry-run request & token estimator)
benizakura experiment plan \
  --benchmark evals/conquer/benchmark_v1.json \
  --ground-truth evals/conquer/ground_truth_v1.json \
  --experiment-id calibration_v1

# Execute dry-run experiment (default: creates runs/ and manifest without API calls)
benizakura experiment run calibration_v1

# Execute live calibration run (requires frozen ground truth; fails safely if unfrozen)
benizakura experiment run calibration_v1 --execute

# Inspect status of an experiment run
benizakura experiment status calibration_v1

# Summarize experiment results against human ground truth and evaluate H1–H5
benizakura experiment summarize calibration_v1
```

> [!NOTE]
> `benizakura experiment run` runs in **dry-run mode by default**. The `--execute` flag is strictly required to make live provider API calls. If ground truth is not frozen, execution is safely blocked before any network call occurs.

---

## Safety & Research Integrity

* **Zero Fabricated Human Labels:** No synthetic, automated, or LLM-generated human labels are permitted. Ground truth must come from real human annotation.
* **Deterministic Fail-Safe Gates:** `check_calibration_prerequisites` enforces benchmark immutability (`SHA-256`), frozen ground-truth status, and human agreement ($\kappa \ge 0.60$).
* **Distinguishable Failure Classes:** API errors, timeouts, rate limits, schema validation failures, and JSON parse errors are recorded distinctly (`TIMEOUT`, `RATE_LIMIT`, `PROVIDER_ERROR`, `PARSE_ERROR`, `VALIDATION_ERROR`). Failures are **never** silently converted into ties or dropped from denominators.
* **Immutable Benchmark Boundary:** The canonical benchmark SHA-256 (`aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`) is checked cryptographically on every run.
* **Complete Run Manifests:** Every experiment run persists a detailed `manifest.json` containing the benchmark hash, ground truth hash, prompt versions, rubric versions, temperature, seed, software version, and run timestamps.
* **Secret Governance:** API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are read strictly from environment variables and recursively scrubbed by `sanitize_secrets` before serialization to manifests or logs.
* **Production Gating Disabled:** Automated production CI merge-blocking remains disabled until empirical calibration proves judge reliability.

---

## Code Example: Core Pairwise & Release Gate Pipeline

```python
from benizakura.gate import GateConfig, ReleaseGate
from benizakura.judge import MockPairwiseJudge
from benizakura.models import (
    CriterionAssessment,
    EvaluationCase,
    PairwiseJudgment,
    PairwiseWinner,
    Rubric,
    StandardCriteria,
    Verdict,
)
from benizakura.pairwise_runner import BidirectionalPairwiseRunner
from benizakura.statistical import EvaluationDataset, PairedCaseOutcome, StatisticalAnalysis

# 1. Define evaluation case and explicit rubric
case = EvaluationCase(
    id="sys-001",
    topic="SYSTEM_DESIGN",
    question="Design a distributed rate limiter.",
    candidate_answer="Token bucket with Redis clusters and local sliding window caching.",
)
rubric = Rubric(
    name="SystemDesignRubric",
    criteria=[StandardCriteria.CORRECTNESS, StandardCriteria.TECHNICAL_DEPTH],
    instructions="Evaluate concurrency safety, clock drift tolerance, and failure modes.",
)

# 2. Dual-pass bidirectional evaluation with order normalization
judge = MockPairwiseJudge(default_judgment=PairwiseJudgment(
    winner=PairwiseWinner.B,
    rationale="Candidate in position B provided failure domain analysis.",
))
runner = BidirectionalPairwiseRunner(judge=judge)
result = runner.evaluate(
    case=case,
    baseline_output="Centralized counter with simple TTL.",
    candidate_output="Token bucket with local memory buffer and Redis cluster synchronization.",
    rubric=rubric,
)

# 3. Release gate evaluation
gate = ReleaseGate(config=GateConfig(min_cases=10, max_unstable_rate=0.20))
print(f"Pass 1 Winner:      {result.pass1_normalized_winner.value}")
print(f"Pass 2 Winner:      {result.pass2_normalized_winner.value}")
print(f"Order Inconsistent: {result.is_order_inconsistent}")
```

---

## Project Roadmap

```text
Phase 1 — Core Evaluation Architecture        ✅ (Models, Bidirectional Runner, Normalization)
Phase 2 — Statistics & Release Gating         ✅ (Bootstrap CI, Paired Deltas, ReleaseGate)
Phase 3 — Research & Calibration Infrastructure ✅
  ├── 3A: LLM Judge Research Specification    ✅ (Selection rationale, independence, hypotheses H1–H5)
  ├── 3B: Frozen Benchmark Construction       ✅ (30-case Conquer benchmark v1, leakage audited)
  ├── 3C: Human Calibration Infrastructure   ✅ (Blind task generation, validation, Cohen's κ, adjudication tooling)
  └── 3D: LLM Calibration Infrastructure     ✅ (Provider abstraction, variants A–E, dry-run safety)

NEXT STAGES:
Phase 3 Empirical Work — Human Ground Truth Calibration ⏳
  ├── Benchmark frozen pending human annotation
  ├── Human annotation pending (two independent engineer annotators)
  ├── Inter-rater agreement calculation (Cohen's κ)
  ├── Disagreement adjudication & consensus pending
  └── Ground-truth freeze pending (evals/conquer/ground_truth_v1.json)
Phase 4 — Empirical LLM Judge Evaluation      ⏳
  └── Execute and compare Variants A–E after human ground truth is frozen
Phase 5 — CI / Release-Gate Integration       ⏳ (Automated GitHub Actions PR regression gate)
```

---

## Developer

**Adithya Chavali**
* Repository: [chavaliadi/change-gate](https://github.com/chavaliadi/change-gate)

*Benizakura is an open research and engineering project exploring reliable, reproducible CI/CD gates for AI application behaviour.*
