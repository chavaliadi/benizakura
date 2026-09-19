# Benizakura

> **Detecting silent regressions in AI behaviour.**

**Benizakura** is an experimental regression-testing and release-gating framework designed specifically for AI-powered applications.

In traditional software, when you change code, a suite of unit and integration tests verifies that nothing broke. But in applications powered by Large Language Models (LLMs), a developer might change:

* A system prompt or formatting instruction
* The underlying model or model version
* Temperature, top-p, or sampling parameters
* An evaluation or scoring rubric

When these changes occur, **conventional software tests almost always still pass**. The API returns HTTP 200, the JSON payload matches the expected schema, and no exceptions are raised. Yet the actual quality of the AI's responses may have silently degraded.

Benizakura explores how to catch these behavioural regressions before new prompts and configurations are released into production.

```text
Case
 │
 ▼
Baseline Variant vs. Candidate Variant
 │
 ▼
Pairwise Judge
 │
 ▼
Bidirectional Evaluation (swapped orders to expose bias)
 │
 ▼
Order Normalization & Instability Detection
 │
 ▼
Statistical Analysis (paired differences, bootstrap CI, effect size)
 │
 ▼
Release Gate (deterministic decision precedence hierarchy)
 │
 ▼
PASS / FAIL / INCONCLUSIVE
```

---

## The Problem: Silent AI Regressions

To understand why Benizakura is needed, consider how regression testing works in standard software versus AI applications:

### Traditional Software Regression

```text
Code Change ──▶ Run Test Suite ──▶ Deterministic Assertion ──▶ PASS or FAIL
```

If a function that calculates payment totals returns an incorrect value, a unit test catches it immediately with a clear assertion failure.

### AI Behaviour Regression

```text
Prompt Edit ──▶ Run Test Suite ──▶ API Returns 200 OK ──▶ Tests Pass ✅
                                                          AI Quality Quietly Regressed ❌
```

Suppose you are building an automated technical interview grader.

* **Before prompt change (Baseline):** A candidate explains horizontal scaling well. The AI grader awards **8.5 / 10** with actionable technical feedback.
* **After prompt change (Candidate):** The prompt is edited to make responses "more concise." The candidate submits the exact same answer. The AI grader now awards **6.0 / 10**, missing key technical points because the new instructions inadvertently penalised detailed explanations.

No runtime error was thrown. The database update succeeded. All existing unit tests passed. Yet the product became significantly worse for candidates.

Benizakura provides the automated gating layer to detect, measure, and flag these regressions before deployment.

---

## Current Target Application: Conquer

The initial real-world evaluation target for Benizakura is **Conquer**, an AI-powered technical interview preparation simulator.

Specifically, the system targets Conquer's core evaluation endpoint:
`POST /api/interview/score` (Per-Question Answer Scorer & Profile Generator).

In Conquer, this feature:
1. Evaluates candidate answers across technical tracks (System Design, Algorithms, Distributed Systems).
2. Generates qualitative feedback and numeric scores on a 0.0–10.0 scale.
3. Emits skill profile deltas that adjust adaptive question difficulty.

Gating this feature ensures that whenever Conquer's engineering team adjusts scoring prompts or upgrades models, grading standards remain consistent, fair, and calibrated.

> **Extensibility Note:** While Conquer serves as our first real-world domain, the Benizakura architecture is domain-agnostic and designed to gate any prompt- or model-driven AI feature.

---

## Core Concept: Why Pairwise Evaluation?

Most early AI evaluation systems rely on **pointwise evaluation**: an LLM judge is shown a single response and asked:

> *"Rate this response on a scale of 1 to 10."*

Research and practical experience show that pointwise scoring suffers from significant weaknesses:
* **Score Drift:** An LLM might give an 8/10 today and a 6/10 tomorrow for the same response due to subtle temperature or context shifts.
* **Calibration Complexity:** Aligning what an "8" means across different questions, topics, and models requires extensive manual calibration.
* **Scale Compression:** Judges tend to cluster scores in narrow ranges (such as 7–8), making small regressions hard to detect.

Instead of asking for an absolute number, **pairwise evaluation** presents two competing outputs simultaneously and asks:

> *"Comparing Output A and Output B for this question, which one is better according to the rubric?"*

Comparing two outputs directly forces a relative decision:
* Output A is better
* Output B is better
* Both are equivalent (Tie)

---

## The Major Innovation: Bidirectional Evaluation & Position Bias

While pairwise comparison is powerful, LLM judges suffer from a well-documented flaw known as **position bias**: **evaluators frequently favor whichever output is presented first (Position A), regardless of substance.**

If you only compare Baseline vs. Candidate once, a judge with position bias will always declare Position A the winner. If Baseline was placed in Position A, you might falsely conclude that Candidate regressed.

Benizakura solves this by evaluating every pair **twice** with presentation positions swapped:

```text
                ┌──────────────────────────────────────────────┐
                │             Evaluation Case Input            │
                │        Baseline Output & Candidate Output     │
                └──────────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             [ Pass 1: Forward ]                  [ Pass 2: Reverse ]
             Position A = Baseline                Position A = Candidate
             Position B = Candidate               Position B = Baseline
                    │                                     │
                    ▼                                     ▼
           Raw Judge Decision                    Raw Judge Decision
              (A / B / TIE)                         (A / B / TIE)
                    │                                     │
                    ▼                                     ▼
           Order Normalization                   Order Normalization
         (BASELINE/CANDIDATE/TIE)              (BASELINE/CANDIDATE/TIE)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                         Consistency Classification
                      (Check if both passes agree)
```

### Why Normalization Is Required

The raw judge only sees positions (`A` or `B`). It has no knowledge of which system is the Baseline or Candidate.

The system normalizes raw positions back to the true system identity:

| Pass | Position A | Position B | If Judge picks A | If Judge picks B | If Judge picks TIE |
|---|---|---|---|---|---|
| **Pass 1 (Forward)** | Baseline | Candidate | **BASELINE** wins | **CANDIDATE** wins | **TIE** |
| **Pass 2 (Reverse)** | Candidate | Baseline | **CANDIDATE** wins | **BASELINE** wins | **TIE** |

### Detecting Position Bias in Action

Consider a biased judge that blindly votes for Position A:

```text
Pass 1:
  Position A = Baseline
  Position B = Candidate
  Judge Decision: "A wins"  ──▶  Normalized Winner: BASELINE

Pass 2 (Swapped):
  Position A = Candidate
  Position B = Baseline
  Judge Decision: "A wins"  ──▶  Normalized Winner: CANDIDATE
```

Although the raw judge selected "A" in both passes, the underlying winners **directly contradict each other** (Baseline in Pass 1 vs. Candidate in Pass 2).

Benizakura detects this contradiction and flags the result as:
```text
POSITION_UNSTABLE
```

Blindly trusting a single pass would have yielded false confidence. Bidirectional evaluation surfaces the bias immediately.

---

## Consistency Outcome Scenarios

When comparing two passes, the system classifies the outcome into four clear categories:

| Outcome | Pass 1 (A=Base, B=Cand) | Pass 2 (A=Cand, B=Base) | Meaning |
|---|---|---|---|
| **`CONSISTENT_CANDIDATE_WIN`** | Judge chooses B (`CANDIDATE`) | Judge chooses A (`CANDIDATE`) | Candidate legitimately outperformed Baseline in both orders. |
| **`CONSISTENT_BASELINE_WIN`** | Judge chooses A (`BASELINE`) | Judge chooses B (`BASELINE`) | Baseline outperformed Candidate in both orders (regression detected). |
| **`CONSISTENT_TIE`** | Judge chooses `TIE` | Judge chooses `TIE` | Both outputs are equivalent in quality. |
| **`POSITION_UNSTABLE`** | Normalized winners disagree (e.g. Judge votes for `A` both times) | The judge was influenced by presentation order. Output is unreliable. |

---

## Current Architecture

The codebase is strictly layered so that domain concepts, judge protocols, runners, and comparators have clear boundaries and zero tight coupling to external providers:

```text
src/benizakura/
├── models.py           # Core domain entities & enums
├── runner.py           # Pointwise evaluator protocol & single-version runner
├── judge.py            # PairwiseJudge protocol & deterministic MockPairwiseJudge
├── pairwise_runner.py  # BidirectionalPairwiseRunner (swaps order & normalizes)
├── comparator.py       # Provisional SimpleComparator & release gate logic
└── cli.py              # Demonstration command-line interface
```

### 1. Typed Domain Models (`models.py`)

* **`EvaluationCase`**: A single test scenario containing the question, candidate answer, topic, mode, optional rubric, and metadata.
* **`Criterion` & `Rubric`**: Structured criteria defining evaluation guidelines and relative weights.
* **`PairwiseWinner`**: Raw position choices returned by a judge: `A`, `B`, or `TIE`.
* **`PairwiseJudgment`**: The raw output of a comparison, holding `winner`, qualitative `reason`, and optional `confidence`.
* **`NormalizedWinner`**: The normalized identity: `BASELINE`, `CANDIDATE`, or `TIE`.
* **`BidirectionalEvaluationResult`**: Container holding both raw judgments and normalized winners for forward and reverse passes.
* **`ConsistencyOutcome` & `classify_consistency()`**: Categorises outcomes as `CONSISTENT_CANDIDATE_WIN`, `CONSISTENT_BASELINE_WIN`, `CONSISTENT_TIE`, or `POSITION_UNSTABLE`.
* **`Verdict` & `ComparisonResult`**: Gate outcomes (`PASS`, `FAIL`, `INCONCLUSIVE`) used by release comparators.

### 2. Pointwise Evaluation Layer (`runner.py`)

* **`Evaluator`**: A protocol defining `evaluate(case: EvaluationCase) -> EvaluationResult`.
* **`MockEvaluator`**: A deterministic test double allowing custom scores and per-case overrides.
* **`EvaluationRunner`**: Executes a collection of cases against an evaluator for a single version.

### 3. Pairwise Judge Abstraction (`judge.py`)

* **`PairwiseJudge`**: A runtime-checkable `Protocol` with the signature:
  ```python
  def judge(
      case: EvaluationCase,
      output_a: str,
      output_b: str,
      rubric: Optional[Rubric] = None,
  ) -> PairwiseJudgment: ...
  ```
* **`MockPairwiseJudge`**: A fully deterministic mock judge designed for comprehensive testing without external network or LLM dependencies. It supports:
  * Default fallback judgments
  * FIFO queues of predetermined judgments (to simulate multi-pass sequences)
  * Per-case overrides
  * Call recording via `PairwiseJudgeCall` to inspect what the judge received
* **Isolation Guarantee:** The judge sees only `output_a` and `output_b`. It remains completely unaware of which output belongs to Baseline or Candidate.

### 4. Bidirectional Pairwise Runner (`pairwise_runner.py`)

* **`BidirectionalPairwiseRunner`**: Orchestrates the dual-pass evaluation:
  1. Presents `output_a = baseline_output`, `output_b = candidate_output` (Pass 1).
  2. Presents `output_a = candidate_output`, `output_b = baseline_output` (Pass 2).
  3. Maps raw A/B choices to underlying system identities (`_normalize_pass1` and `_normalize_pass2`).
  4. Packages both passes into a `BidirectionalEvaluationResult`.
* **Scope Boundary:** This runner owns execution and normalization. It intentionally does not decide final release gate verdicts.

### 5. Provisional Comparator (`comparator.py`)

* **`SimpleComparator`**: The initial threshold-based comparator implementing basic heuristic rules:
  * **`PASS`**: Candidate demonstrates net improvement with zero critical regressions.
  * **`FAIL`**: Triggered by critical score drops or overall degradation.
  * **`INCONCLUSIVE`**: Differences fall within noise thresholds.

> **Honest Architecture Note:** `SimpleComparator` currently operates on pointwise score runs from the initial prototype spike. It will be superseded in upcoming phases by a statistical pairwise comparator that aggregates bidirectional consistency outcomes.

---

## Current Repository Structure

```text
benizakura/
├── README.md                          # Project documentation and architecture guide
├── pyproject.toml                     # Build configuration, metadata, and test dependencies
├── evals/
│   └── conquer/
│       └── cases.json                 # 5 development test cases for Conquer interview Q&A
├── src/
│   └── benizakura/
│       ├── __init__.py                # Clean public API exports
│       ├── models.py                  # Domain models, enums, rubrics, and consistency logic
│       ├── runner.py                  # Pointwise Evaluator protocol, MockEvaluator, EvaluationRunner
│       ├── judge.py                   # PairwiseJudge protocol, MockPairwiseJudge, call records
│       ├── pairwise_runner.py         # BidirectionalPairwiseRunner & order normalization
│       ├── comparator.py              # Provisional SimpleComparator & gating heuristics
│       └── cli.py                     # Interactive CLI demonstration entry point
├── tests/
│   ├── test_models.py                 # Domain models, rubrics, enums & consistency tests (13 tests)
│   ├── test_comparator.py             # Provisional SimpleComparator tests (5 tests)
│   ├── test_runner.py                 # Pointwise runner & mock evaluator tests (3 tests)
│   ├── test_judge.py                  # PairwiseJudge protocol & mock judge tests (11 tests)
│   └── test_pairwise_runner.py        # Bidirectional runner, normalization & bias tests (10 tests)
└── docs/
    ├── NOTES.md                       # Engineering notes, research roadmap & Conquer context
    ├── DECISIONS.md                   # Architectural decision records (ADRs)
    ├── PROJECT_PLAN-2.md              # Master engineering & milestone plan
    └── DEVELOPMENT_RULES-2.md         # Code conventions & phase discipline
```

---

## Current Implementation Status

| Component | Status | Notes |
|---|:---:|---|
| **Typed Domain Models** | ✅ Implemented | Criteria, rubrics, pairwise judgments, criterion assessments, and consistency outcomes. |
| **Rubric & Criteria Contract** | ✅ Implemented | Weighted criteria, validation, instructions, and domain-independent standard criteria catalog (`StandardCriteria`). |
| **Structured Judge Output** | ✅ Implemented | `PairwiseJudgment` with structured `CriterionAssessment`, evidence quotes, confidence bounds, and serialization. |
| **Bidirectional Runner** | ✅ Implemented | Swaps presentation order, normalizes overall & per-criterion identities, preserves raw and normalized data. |
| **Position Bias Detection** | ✅ Implemented | Identifies order instability via `ConsistencyOutcome.POSITION_UNSTABLE`. |
| **Evaluation Dataset Abstraction** | ✅ Implemented | `EvaluationSample` and `EvaluationDataset` preserving case identities with duplicate detection. |
| **Statistical Aggregations & Rates** | ✅ Implemented | Empirical candidate/baseline/tie win rates and consistency rates with explicit sample-size denominators. |
| **Criterion-Level Aggregation** | ✅ Implemented | `CriterionAggregate` breakdown by criterion without fabricating missing assessments. |
| **Paired Differences & Attribution** | ✅ Implemented | `PairedCaseOutcome` ($D_i \in \{-1, 0, +1\}$) tracking regression lists (`get_regressed_cases()`, `get_improved_cases()`). |
| **Bootstrap Confidence Intervals** | ✅ Implemented | Deterministic paired case resampling with replacement for percentile confidence intervals on win-rate differences. |
| **Effect Size Representation** | ✅ Implemented | `EffectSize` capturing net win rate difference ($\Delta_{win}$) and Cohen's $g$ for decisive pairs. |
| **Provider-Independent Judge Abstraction** | ✅ Implemented | `JudgeProvider` protocol, `JudgeRequest`, `JudgeResponse`, and `MockJudgeProvider` test double. |
| **Prompt Construction & Response Parsing** | ✅ Implemented | `JudgePromptBuilder`, `JudgeResponseParser` (JSON & markdown fenced), and typed error hierarchy (`JudgeParseError`, `JudgeValidationError`). |
| **LLM Pairwise Judge** | ✅ Implemented | `LLMPairwiseJudge` coordinating prompt builder, provider, and parser while conforming to `PairwiseJudge`. |
| **Unit Test Suite** | ✅ Implemented | **113 passing tests** across 10 test suites. Zero external API calls required. |
| **Real LLM Provider Adapters** | 📌 Planned | Hand-written concrete adapters (OpenAI, Anthropic, Groq) implementing `JudgeProvider`. |
| **Pairwise Release Gate Policy** | 📌 Planned | Formal PASS / FAIL / INCONCLUSIVE release gating thresholds built on top of the statistical layer. |
| **Conquer Golden Benchmark** | 📌 Planned | Expanding beyond 5 placeholder cases to 25–30 stratified interview scenarios. |
| **GitHub Actions CI/CD Gate** | 📌 Planned | Automated PR commentary and blocking merge gates on detected regressions. |

---

## Quick Start

### Requirements

* **Python 3.11+**

### 1. Installation

Clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/chavaliadi/change-gate.git
cd change-gate

# Install in editable mode
pip install -e .

# Install development/testing dependencies (pytest)
pip install -e ".[dev]"
```

### 2. Run the Test Suite

Execute the entire unit test suite using `pytest`:

```bash
pytest -v
```

All 42 tests run deterministically in under 0.1 seconds without requiring API keys or internet access.

### 3. Run the CLI Demonstrations

Run the built-in CLI to see how Benizakura reports verdicts on sample interview cases:

```bash
# Default demonstration: Candidate improves over Baseline (PASS)
benizakura

# Or run directly via Python module:
python3 -m benizakura.cli

# Simulate a critical regression in candidate behavior (FAIL):
benizakura --demo fail

# Simulate negligible difference where evidence is balanced (INCONCLUSIVE):
benizakura --demo inconclusive
```

Example output from `benizakura --demo fail`:

```text
Benizakura
==========

Evaluation cases: 5

Baseline:  v1 (avg: 8.00)
Candidate: v2 (avg: 7.10)

Verdict: FAIL

Summary:
Critical regression detected in 1 case(s). Candidate average score 7.10 vs baseline 8.00 (diff: -0.90).

Detected Regressions:
  - [conquer-001-process-vs-thread] baseline=8.0 -> candidate=3.5 (Score dropped past critical regression threshold)
```

---

## Code Example: Using Rubrics & Bidirectional Runner

Here is how the rubric contract, structured criterion assessments, and bidirectional runner interact in Python code:

```python
from benizakura.models import (
    CriterionAssessment,
    EvaluationCase,
    NormalizedWinner,
    PairwiseJudgment,
    PairwiseWinner,
    Rubric,
    StandardCriteria,
    classify_consistency,
    ConsistencyOutcome,
)
from benizakura.judge import MockPairwiseJudge
from benizakura.pairwise_runner import BidirectionalPairwiseRunner

# 1. Define an evaluation case and explicit rubric contract
case = EvaluationCase(
    id="case-101",
    topic="System Design",
    question="Explain horizontal vs. vertical scaling.",
    candidate_answer="Vertical adds RAM/CPU; horizontal adds machines.",
)

rubric = Rubric(
    name="System Architecture Rubric",
    criteria=[StandardCriteria.CORRECTNESS, StandardCriteria.TECHNICAL_DEPTH],
    instructions="Focus on architectural trade-offs, SPOFs, and failure domains.",
)

# 2. Configure a mock judge returning structured criterion-level judgments
pass1_judgment = PairwiseJudgment(
    winner=PairwiseWinner.B,
    rationale="Candidate in position B provided explicit failure mode analysis.",
    criterion_assessments=[
        CriterionAssessment(
            criterion_name="correctness",
            winner=PairwiseWinner.B,
            rationale="B accurately explained horizontal scaling state management.",
        ),
        CriterionAssessment(
            criterion_name="technical_depth",
            winner=PairwiseWinner.B,
            rationale="B addressed load balancer bottlenecks and failover.",
        ),
    ],
)

pass2_judgment = PairwiseJudgment(
    winner=PairwiseWinner.A,
    rationale="Candidate in position A provided explicit failure mode analysis.",
    criterion_assessments=[
        CriterionAssessment(
            criterion_name="correctness",
            winner=PairwiseWinner.A,
            rationale="A accurately explained horizontal scaling state management.",
        ),
        CriterionAssessment(
            criterion_name="technical_depth",
            winner=PairwiseWinner.A,
            rationale="A addressed load balancer bottlenecks and failover.",
        ),
    ],
)

mock_judge = MockPairwiseJudge(judgments=[pass1_judgment, pass2_judgment])

# 3. Execute bidirectional evaluation
runner = BidirectionalPairwiseRunner(mock_judge)
result = runner.evaluate(
    case=case,
    baseline_output="Scale vertically by upgrading your server instance.",
    candidate_output="Scale horizontally with stateless workers behind a load balancer.",
    rubric=rubric,
)

# 4. Verify normalized overall outcomes
assert result.pass1_normalized_winner == NormalizedWinner.CANDIDATE
assert result.pass2_normalized_winner == NormalizedWinner.CANDIDATE

# 5. Verify normalized criterion-level outcomes
norm_corr1, norm_corr2 = result.get_criterion_normalized_winners("correctness")
assert norm_corr1 == NormalizedWinner.CANDIDATE
assert norm_corr2 == NormalizedWinner.CANDIDATE

# 6. Classify consistency across presentation passes
outcome = classify_consistency(
    result.pass1_normalized_winner,
    result.pass2_normalized_winner,
)
print(f"Outcome: {outcome.value}")
# Prints: Outcome: CONSISTENT_CANDIDATE_WIN
```

---

## Code Example: Statistical Evaluation & Bootstrap Confidence

Here is how a collection of evaluation cases is aggregated into paired differences, empirical win rates, and bootstrap confidence intervals:

```python
from benizakura.statistical import (
    EvaluationDataset,
    EvaluationSample,
    StatisticalAnalyzer,
)

# 1. Collect evaluated benchmark samples (preserving case identities)
dataset = EvaluationDataset(samples=[
    EvaluationSample(case_id="case-101", result=result),
    # ... additional evaluated cases ...
])

# 2. Configure deterministic statistical analyzer
analyzer = StatisticalAnalyzer(
    confidence_level=0.95,
    num_bootstrap_samples=1000,
    random_seed=42,
)

# 3. Compute empirical rates and paired bootstrap confidence interval
analysis = analyzer.analyze(dataset)

print(f"Cases Evaluated:      {analysis.sample_size}")
print(f"Candidate Win Rate:   {analysis.candidate_win_rate:.2%}")
print(f"Baseline Win Rate:    {analysis.baseline_win_rate:.2%}")
print(f"Observed Difference:  {analysis.observed_difference:+.2f}")
print(f"95% Bootstrap CI:     [{analysis.confidence_interval[0]:+.2f}, {analysis.confidence_interval[1]:+.2f}]")
print(f"Effect Size:          {analysis.effect_size.interpretation}")

# 4. Explicit case attribution: which exact cases regressed?
regressed = analysis.get_regressed_cases()
improved = analysis.get_improved_cases()
print(f"Regressed Cases:      {regressed}")
print(f"Improved Cases:       {improved}")
```

---

---

## Code Example: LLM Pairwise Judge with Provider Abstraction

Here is how the provider-independent `LLMPairwiseJudge` orchestrates prompt generation, provider execution, and structured response parsing:

```python
from benizakura.judge import (
    LLMPairwiseJudge,
    MockJudgeProvider,
)
from benizakura.models import EvaluationCase, Rubric, StandardCriteria
from benizakura.pairwise_runner import BidirectionalPairwiseRunner

# 1. Configure a provider (here using the deterministic mock provider)
mock_response = """
{
    "winner": "B",
    "rationale": "Position B provided concrete failure domain analysis.",
    "confidence": 0.88,
    "criterion_assessments": [
        {
            "criterion_name": "correctness",
            "winner": "B",
            "rationale": "Position B accurately explained write amplification."
        }
    ]
}
"""
provider = MockJudgeProvider(default_response=mock_response)

# 2. Instantiate LLMPairwiseJudge
judge = LLMPairwiseJudge(provider=provider)

# 3. Use interchangeably with BidirectionalPairwiseRunner
runner = BidirectionalPairwiseRunner(judge=judge)
result = runner.evaluate(
    case=EvaluationCase(
        id="case-001",
        topic="Databases",
        question="Explain write amplification in LSM trees.",
        candidate_answer="Compaction merges SSTables to reclaim disk space.",
    ),
    baseline_output="LSM trees write to memtable and flush to disk.",
    candidate_output="LSM trees buffer writes in memtable, flush to SSTables, and compact in background.",
    rubric=Rubric(name="DB", criteria=[StandardCriteria.CORRECTNESS]),
)

print(f"Pass 1 Winner: {result.pass1_normalized_winner.value}")
```

---

## Code Example: Release Gate Decision Policy

Benizakura strictly separates statistical measurement from release gating policy. `StatisticalAnalysis` answers *"What happened in the evaluation?"*, while `ReleaseGate` answers *"Given our predefined policy, should this candidate be released?"*:

```python
from benizakura.gate import GateConfig, ReleaseGate
from benizakura.models import Verdict

# 1. Configure the decision policy (all thresholds are explicit policy choices)
config = GateConfig(
    min_cases=10,              # Minimum benchmark cases required
    max_unstable_rate=0.20,    # Max 20% position-unstable cases permitted
    regression_tolerance=0.05, # 5% negative buffer on observed difference
    max_regression_rate=0.25,  # Max 25% individual case regressions permitted
    min_effect_size=0.05,      # Minimum net advantage (+5%) required to pass
)

# 2. Evaluate statistical evidence
gate = ReleaseGate(config=config)
result = gate.evaluate(analysis)

# 3. Inspect structured, explainable release decision
print(f"Verdict:             {result.verdict.value}")
print(f"Observed Difference: {result.observed_difference:+.2f}")
print(f"95% Bootstrap CI:    [{result.confidence_interval[0]:+.2f}, {result.confidence_interval[1]:+.2f}]")
print(f"Regression Rate:     {result.regression_rate:.1%} ({result.regression_count} cases: {result.regressed_cases})")
print(f"Instability Rate:    {result.unstable_rate:.1%} ({result.unstable_count} cases)")
print(f"Decision Reason:\n  {result.reason}")
```

### Release Gate Verdict Semantics

The gate applies a deterministic decision precedence hierarchy yielding one of three explicit outcomes:

* **`PASS`**: Evidence supports accepting the candidate under the configured policy. The observed advantage meets the minimum effect size (`min_effect_size`), the bootstrap confidence interval remains within the configured regression tolerance (`regression_tolerance`), and individual case regressions are below `max_regression_rate`.
* **`FAIL`**: Evidence supports rejecting the candidate because unacceptable regression occurred. Triggered when the bootstrap upper confidence bound falls strictly below `-regression_tolerance` (confident regression), or when the proportion of regressed cases exceeds `max_regression_rate`.
* **`INCONCLUSIVE`**: Evidence is insufficient or unreliable enough that Benizakura refuses to make a confident release decision. Triggered when the sample size is below `min_cases`, judge position instability exceeds `max_unstable_rate`, or the confidence interval overlaps the decision boundary without clear separation.

> **Important Methodological Note:**
> Gate thresholds are **policy parameters, not universal statistical truths**. Organizations can and should calibrate `GateConfig` to match their risk tolerance and benchmark characteristics without altering evaluation code.

---

## Testing & Quality Assurance

Benizakura enforces strict test-driven development. The test suite verifies every layer of the architecture:

```text
tests/
├── test_models.py           # 13 tests: enums, rubric validation, normalization, and consistency
├── test_rubric.py           # 25 tests: rubric contracts, criteria validation, structured judgments, serialization
├── test_statistical.py      # 22 tests: dataset abstraction, paired differences, bootstrap CI, effect size
├── test_gate.py             # 29 tests: release gate policy, precedence, safeguards, boundaries, explainability
├── test_prompt.py           #  4 tests: prompt construction, symmetry, instructions, deterministic building
├── test_parser.py           # 14 tests: JSON parsing, markdown fences, validation rules, error types
├── test_llm_judge.py        #  7 tests: LLMPairwiseJudge pipeline, mock provider, bidirectional runner
├── test_comparator.py       #  5 tests: PASS, FAIL, INCONCLUSIVE rules and score regression detection
├── test_runner.py           #  3 tests: Pointwise MockEvaluator, overrides, and runner flow
├── test_judge.py            # 11 tests: PairwiseJudge protocol, MockPairwiseJudge queue & exhaustion
└── test_pairwise_runner.py  # 10 tests: Order swapping, dual-call verification & position bias detection
────────────────────────────────────────────────────────────────────────────────
Total: 142 passed in ~0.10s
```

Run test coverage inspection anytime:

```bash
pytest --tb=short
```

---

## Design Principles

1. **Deterministic Core First:** We build and thoroughly test the execution, normalization, and gating algorithms using deterministic test doubles before connecting expensive, nondeterministic external LLM APIs.
2. **Provider Independence:** The evaluation contract belongs strictly to Benizakura. The model provider only executes raw text generation (`JudgeRequest` → `JudgeResponse`). Vendor SDKs never leak into domain models.
3. **Explicit Evaluation Contracts:** We evaluate pairs against clear, weighted rubrics and evaluation instructions rather than unconstrained comparison prompts.
4. **Structured Evidence over Opaque Scores:** Judges emit structured per-criterion assessments (`CriterionAssessment`) with rationales and evidence excerpts alongside overall verdicts.
5. **Preserve Paired Case Identity:** Statistical analysis operates on paired observations $(B_i, C_i)$ with exact case tracking; regressions and improvements are never lost in aggregate means.
6. **Surface Uncertainty, Don't Guess:** Bootstrap confidence intervals quantify empirical uncertainty. Release gating decisions are kept strictly separate from measurement.
7. **Strict Separation of Presentation vs. Identity:** The judge only evaluates presented positions (`A` and `B`). The runner manages presentation order and system identities (`Baseline` and `Candidate`). Neither layer leaks into the other.
8. **Preserve Raw Evidence:** The system never discards raw judge outputs. Both raw position judgments and normalized identities are recorded side-by-side for full auditability.
9. **Zero Heavy Framework Bloat:** We avoid heavyweight orchestration frameworks (like LangChain) in core execution paths. Transparent, standard-library Python dataclasses and protocols make attribution and debugging obvious.

---

## Research Motivation & Roadmap

Benizakura is an active engineering and research exploration into reliable LLM evaluation methodology. Our roadmap draws directly from foundational literature:

* **LLM-as-a-Judge & Chatbot Arena** (*Zheng et al., 2023*): Validated pairwise comparison as more robust than absolute scoring.
* **G-Eval & Rubric-based Scoring** (*Liu et al., 2023*): Formulating explicit criteria and probability-weighted evaluations.
* **Position & Cognitive Bias Mitigation**: Designing algorithmic defenses against presentation order bias, verbosity bias, and model self-enhancement.
* **Statistical Calibration & Paired Bootstrap**: Measuring paired case deltas and estimating confidence intervals via paired case resampling with replacement.

### Project Roadmap

* [x] **Phase 1: Deterministic Foundation & Evaluation Core (Completed)**
  * Typed domain models, rubric contracts, and domain-independent standard criteria
  * Structured judge outputs (`CriterionAssessment`) and serialization
  * `PairwiseJudge` protocol and deterministic `MockPairwiseJudge`
  * `BidirectionalPairwiseRunner` with presentation swapping and order normalization
  * Position instability classification
  * `EvaluationDataset` preserving case identities and validating uniqueness
  * Empirical rate calculations with explicit sample-size denominators
  * Per-criterion aggregations and paired delta modeling ($D_i \in \{-1, 0, +1\}$)
  * Paired bootstrap confidence intervals and effect size representation
  * `JudgeProvider` protocol, `JudgePromptBuilder`, and `JudgeResponseParser`
  * Typed error hierarchy (`JudgeParseError`, `JudgeValidationError`, `JudgeProviderError`)
  * `LLMPairwiseJudge` coordinating prompt building, provider generation, and validation
  * 113 unit tests with zero third-party dependencies
* [x] **Phase 2: Release Gate Decision Policy (Completed)**
  * Structured `GateConfig` with explicit, validated policy parameters and defaults
  * Structured, explainable `GateResult` tracking verdict, reason, regressions, and instability
  * Deterministic `ReleaseGate` applying an explicit decision precedence hierarchy
  * Pure statistical independence: operates entirely on `StatisticalAnalysis`
  * CLI gate demonstration (`benizakura --demo gate`)
  * 142 total unit tests (29 dedicated gate tests)
* [ ] **Phase 3: Real LLM Provider Selection & Integration**
  * Research-driven provider selection and lightweight adapters (Groq, Anthropic, OpenAI) implementing `JudgeProvider`
  * Exact-match caching to guarantee zero redundant API spend
* [ ] **Phase 4: Conquer Golden Benchmark**
  * 25–30 stratified test cases spanning System Design, Algorithms, and Architecture
  * Edge cases: subtle technical errors, verbose superficial answers, code formatting
* [ ] **Phase 5: CI/CD Release Gate**
  * GitHub Actions automated evaluation on pull requests modifying prompts
  * Automated PR summary comments detailing regressions, wins, and consistency rates

---

## Current Limitations

In the interest of engineering transparency, here is what Benizakura does **not** yet do in its current stage:

* **No live LLM calls:** The repository currently uses deterministic mock judges and mock evaluators. Live API integrations are scheduled for Phase 2.
* **Provisional comparator:** The current CLI demo uses `SimpleComparator` (pointwise delta thresholds) while the pairwise statistical comparator is under development.
* **Development test cases:** The 5 evaluation cases in `evals/conquer/cases.json` are placeholder development inputs, not the final 30-case calibrated benchmark.
* **Single variable:** The gate currently assumes prompt variations only; model swaps and retrieval parameter variations are out of scope for early phases.

These decisions are deliberate: establishing correct domain abstractions, position normalization, and deterministic test coverage first prevents building on an unstable foundation.

---

## Why Benizakura?

The name reflects the central problem explored by this project: a system change can appear to make something stronger or better while quietly introducing hidden degradation.

In AI systems, a prompt, model, or configuration change may appear to improve behaviour while silently causing regressions elsewhere. Benizakura is designed to make those regressions visible before release.

---

## Developer

**Adithya Chavali**

* GitHub: [chavaliadi/change-gate](https://github.com/chavaliadi/change-gate)

*Benizakura is an open engineering project exploring how to build reliable, reproducible CI/CD gates for AI application behaviour.*
