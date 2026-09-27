# Benizakura — LLM Judge Calibration & Empirical Validation Protocol

> **Document Type:** Research & Experimental Protocol Specification  
> **Status:** Active / Ready for Controlled Research Execution  
> **Benchmark Dataset:** [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json) (SHA-256: `aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`)  
> **Ground-Truth Target:** [`evals/conquer/ground_truth_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/ground_truth_v1.json)  
> **Target Application:** Conquer (`POST /api/interview/score`)  
> **Companion Specifications:** [`docs/research/BENCHMARK_V1.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/research/BENCHMARK_V1.md), [`docs/research/LLM_JUDGE_SELECTION.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/research/LLM_JUDGE_SELECTION.md), [`docs/research/HUMAN_CALIBRATION.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/research/HUMAN_CALIBRATION.md)

---

## 1. Central Research Question

The core scientific objective of **Phase 3D: LLM Judge Calibration** in **Benizakura** is to empirically determine:

> **How accurately and consistently can different LLM judge configurations reproduce calibrated human pairwise judgments on the frozen Conquer benchmark, and do Benizakura's proposed architectural safeguards measurably reduce known evaluator failure modes?**

Rather than assuming any LLM judge is trustworthy out-of-the-box, Benizakura measures:
1. **Human Alignment:** Agreement rate, Cohen's $\kappa$, and confusion dynamics against double-annotated, adjudicated human consensus.
2. **Order & Position Stability:** Rate of evaluation flips when swapping response presentation positions ($A \leftrightarrow B$).
3. **Decomposition Efficacy:** Whether structured criterion-level evaluation improves alignment over unconstrained holism.
4. **Reasoning-Order Dynamics:** Whether forcing criterion evidence and qualitative rationale prior to verdict token generation stabilizes decisions.
5. **Architectural Independence:** Empirical divergence between independent-family evaluators (Anthropic Claude) and same-family evaluators (OpenAI GPT-4o) on subtle hallucination probes.

---

## 2. Critical Prerequisite Gate

Live judge calibration experiments are strictly gated. The evaluation runner refuses execution unless:

```text
1. Ground Truth Status:      ground_truth_v1.status == "GROUND_TRUTH_FROZEN"
2. Human Inter-Rater Gate:   human-human Cohen's κ >= 0.60
3. Benchmark Immutability:   benchmark SHA-256 == "aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314"
```

If any condition is unsatisfied:

```text
CALIBRATION BLOCKED
```

A dedicated `--dry-run` and `--fixture` facility is provided exclusively for unit testing and verifying infrastructure pipelines with synthetic fixtures. Synthetic fixtures must never be committed as benchmark evidence.

---

## 3. Judge Configurations & Model Selection

To prevent hardcoding vendor assumptions, judges are declared as declarative configuration records:

```json
{
  "judge_id": "judge_a_claude",
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "temperature": 0.0,
  "system_prompt_version": "pairwise_v1",
  "rubric_version": "standard_rubric_v1",
  "prompt_schema_version": "1.0.0",
  "max_tokens": 1024,
  "model_family": "anthropic",
  "target_model_family": "openai",
  "same_family": false
}
```

### Evaluated Configurations

| Configuration ID | Provider | Model ID | Target Application Model | Architectural Relationship | Intended Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`judge_a_claude`** | Anthropic | `claude-3-5-sonnet-20241022` | `openai/gpt-oss-120b` | **Independent Family** (`same_family: false`) | Primary Judge Candidate |
| **`judge_b_gpt4o`** | OpenAI | `gpt-4o-2024-08-06` | `openai/gpt-oss-120b` | **Same Family** (`same_family: true`) | Cross-Evaluator / Comparison |

*API Availability Rule:* Model identifiers must reflect active frontier endpoints. If any provider deprecates a model identifier, the successor model is recorded in the manifest without altering the experimental protocol.

---

## 4. Controlled Experiment Variants

Benizakura evaluates five progressive experimental variants to isolate component attribution:

```text
Variant A (Pointwise)
   ↓
Variant B (Single-Direction Pairwise)
   ↓
Variant C (Bidirectional Pairwise)
   ↓
Variant D (Rubric-Based Bidirectional Pairwise)
   ↓
Variant E (Full Benizakura: Rubric + Evidence + Bidirectional + Paired Bootstrap)
```

1. **Variant A — Simple Pointwise (`POINTWISE`):**
   * Judge evaluates Candidate and Baseline answers independently on an absolute scalar scale ($0.0 \dots 10.0$).
   * Outcome mapped via difference $\Delta = \text{Score}_{\text{cand}} - \text{Score}_{\text{base}}$ ($\Delta > 0.5 \implies \text{CANDIDATE}$, $\Delta < -0.5 \implies \text{BASELINE}$, else $\text{TIE}$).
2. **Variant B — Single-Direction Pairwise (`SINGLE_DIRECTION`):**
   * Presents $A = \text{Baseline}, B = \text{Candidate}$ in a single orientation without swap controls.
3. **Variant C — Bidirectional Pairwise (`BIDIRECTIONAL`):**
   * Forward pass: $A = \text{Baseline}, B = \text{Candidate}$.
   * Reverse pass: $A = \text{Candidate}, B = \text{Baseline}$.
   * Reverse judgment is normalized. Detects order flips (`is_unstable = True`).
4. **Variant D — Rubric-Based Bidirectional (`RUBRIC_BIDIRECTIONAL`):**
   * Bidirectional passes governed by explicit multi-criterion rubric decomposition (`CORRECTNESS`, `RELEVANCE`, `COMPLETENESS`, `TECHNICAL_DEPTH`, `CLARITY`, `GROUNDEDNESS`).
5. **Variant E — Full Benizakura (`FULL_BENIZAKURA`):**
   * Full pipeline combining rubric decomposition, direct evidence quotations, bidirectional order normalization, position instability gating, and paired bootstrap confidence intervals.

---

## 5. Prompt Specifications & Versioning

Prompts are immutable, version-controlled artifacts:

1. **`pairwise_v1` (Winner-First):**
   * Traditional autoregressive schema where `"winner"` token is emitted first, followed by qualitative rationale and criterion assessments.
2. **`pairwise_rubric_v1` (Rubric-Decomposed):**
   * Incorporates criterion weights, definitions, and boundary condition guidelines.
3. **`pairwise_rationale_first_v1` (Rationale-First / R2):**
   * Inverted schema where `"criterion_assessments"` and `"overall_rationale"` are emitted *before* the final `"winner"` token to test whether intermediate reasoning tokens reduce stochastic flips.
4. **`pointwise_v1`:**
   * Single-response scoring prompt with absolute numeric and per-criterion grading.

---

## 6. Secret Governance & API Key Safety

To maintain institutional compliance and repository security:
* API keys are loaded strictly from environment variables: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.
* Provider adapters enforce recursive secret sanitization (`sanitize_secrets`) across request metadata, response payloads, error traces, and experiment manifests.
* Headers such as `Authorization: Bearer ...` and `x-api-key: ...` are never serialized, logged, or cached.

---

## 7. Position Bias & Instability Methodology

Every bidirectional evaluation executes both forward and reverse presentations:

$$\text{Forward: } A = \text{Baseline}, \; B = \text{Candidate} \implies \text{Winner}_{\text{fwd}}$$
$$\text{Reverse: } A = \text{Candidate}, \; B = \text{Baseline} \implies \text{Winner}_{\text{rev}}$$

After normalising both outcomes relative to system identity (`CANDIDATE`, `BASELINE`, `TIE`), outcomes are classified into four mutually exclusive categories:
* **`CONSISTENT_CANDIDATE_WIN`**: Both passes independently select Candidate.
* **`CONSISTENT_BASELINE_WIN`**: Both passes independently select Baseline.
* **`CONSISTENT_TIE`**: Both passes independently declare a Tie.
* **`POSITION_UNSTABLE`**: Normalized winners differ (e.g. forward pass picks Candidate, reverse pass picks Baseline or Tie).

### Position Instability Rate
$$\text{position\_instability\_rate} = \frac{\sum_{i=1}^N \mathbf{1}[\text{NormalizedWinner}_{\text{fwd}}^{(i)} \ne \text{NormalizedWinner}_{\text{rev}}^{(i)}]}{N}$$

---

## 8. Repeated Evaluation for Stochastic Stability

Where budget permits, cases are evaluated across $R$ independent stochastic repetitions (e.g. $R = 3$ or $R = 5$ at $\text{temperature} = 0.0$ or minimum supported sampling) to measure inference cluster non-determinism. Each execution records:
* `run_id`
* `case_id`
* `orientation` (`forward` vs `reverse`)
* `repeat_index`
* `raw_winner` and `normalized_winner`
* `latency_ms`
* `token_usage`
* `raw_content_sha256`

---

## 9. Ground-Truth Comparative Metrics

When evaluated against frozen human ground truth, Benizakura calculates:

1. **Observed Agreement ($P_o$):**
   $$P_o = \frac{\text{Matching Cases}}{N}$$
2. **Multi-Class Cohen's $\kappa$:**
   $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
   Measured across the three-class outcome space: $\{\text{CANDIDATE}, \text{BASELINE}, \text{TIE}\}$.
3. **$3 \times 3$ Confusion Matrix:** Mapping Judge predictions against Human consensus.
4. **Stratified Agreement:** Cohen's $\kappa$ computed across benchmark categories (DSA, System Design, Backend, Frontend, Behavioral) and bias probe types.

---

## 10. False-Pass and False-Regression Analysis

Using deliberate failure probes in `benchmark_v1.json`:
* **False Pass:** Judge declares Candidate superior (`CANDIDATE`) when human ground truth confirms Candidate regressed (`BASELINE`).
* **False Regression:** Judge declares Candidate inferior (`BASELINE`) when human ground truth confirms Candidate improved (`CANDIDATE`).
* **Artificial Decisiveness:** Judge forces an arbitrary win when human ground truth is a legitimate `TIE`.

---

## 11. Cost Safety, Rate Limits, and Dry-Run Planner

To protect against unexpected cloud expenditures or runaway loops:
1. **Default Dry-Run Mode:** Running `benizakura experiment run` defaults to dry-run mode, calculating total request count, token estimates, and verifying configuration without executing network calls.
2. **Explicit Execution Flag:** Live provider calls occur **only** when the explicit `--execute` flag is passed.
3. **Bounded Retries:** Provider calls utilize exponential backoff capped at 3 retries.
4. **Explicit Failure Logging:** Timeouts, 429 rate limits, and 500 server errors are logged explicitly as `TIMEOUT`, `PROVIDER_ERROR`, or `PARSE_ERROR`. They are **never** silently converted into ties.

---

## 12. Research Hypotheses Under Empirical Test

Benizakura treats all methodological enhancements as formal research hypotheses rather than foregone conclusions:

### Hypothesis H1 — Bidirectional Debiasing
> **Hypothesis:** Bidirectional evaluation detects and neutralizes presentation order bias, yielding lower position-dependent error rates than single-direction evaluation.  
> **Test:** Measure `position_instability_rate` across Variant B vs Variant C.

### Hypothesis H2 — Rubric Decomposition
> **Hypothesis:** Structured multi-criterion rubric decomposition improves alignment with human engineers compared with unconstrained pairwise prompts.  
> **Test:** Compare Cohen's $\kappa$ against human ground truth between Variant B/C (unconstrained) and Variant D/E (rubric).

### Hypothesis H3 — Rationale-First Schema
> **Hypothesis:** Emitting criterion evaluations and qualitative rationale before the overall winner token reduces stochastic order instability compared with winner-first schemas.  
> **Test:** Compare instability rates and human agreement between `pairwise_rationale_first_v1` (R2) and `pairwise_v1` (R1).

### Hypothesis H4 — Model Family Independence
> **Hypothesis:** An independent-family judge (Anthropic Claude 3.5 Sonnet) demonstrates higher sensitivity to subtle hallucinations in GPT-generated answers than a same-family judge (OpenAI GPT-4o).  
> **Test:** Compare detection rates and agreement across the 4 deliberate hallucination probe cases (`dsa-005`, `sys-005`, `backend-006`, `behavioral-006`).

### Hypothesis H5 — Statistical Release Gating
> **Hypothesis:** Paired bootstrap confidence intervals and explicit `INCONCLUSIVE` determinations prevent false release rejections caused by small-sample variance.  
> **Test:** Compare release gate decisions against naive scalar win-rate differences.

---

## 13. Methodological Distinctions & Scientific Modesty

To prevent overclaiming, Benizakura enforces strict reporting standards:

* **Observed Result:** An empirical measurement obtained from executing a specific benchmark run (e.g., *"Claude 3.5 Sonnet achieved raw agreement of 83.3% on 30 cases"*).
* **Hypothesis:** A testable proposition regarding model behavior (e.g., *"Rationale-first schemas may reduce instability"*).
* **Interpretation:** A methodological discussion of why a pattern might have emerged.
* **Limitation:** Known boundaries of the experiment (e.g., sample size $N=30$, specific domain of technical interviews).

Passing a calibration experiment **does not prove** that an LLM judge is universally objective or omniscient; it establishes that the judge meets predefined empirical criteria on this frozen benchmark.
