# Benizakura — Human Ground-Truth Calibration & Inter-Rater Reliability Protocol

> **Document Type:** Research & Calibration Methodology Specification  
> **Status:** Active / Ready for Human Annotation Execution  
> **Benchmark Dataset:** [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json) (SHA-256: `aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`)  
> **Ground-Truth Target:** [`evals/conquer/ground_truth_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/ground_truth_v1.json)  
> **Target Application:** Conquer (`POST /api/interview/score`)  
> **Accompanying Specifications:** [`docs/research/BENCHMARK_V1.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/research/BENCHMARK_V1.md), [`docs/research/LLM_JUDGE_SELECTION.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/research/LLM_JUDGE_SELECTION.md)

---

## 1. Executive Summary & Core Purpose

This document specifies the scientific methodology, operational lifecycle, statistical formulation, and governance rules for **Phase 3C: Human Ground-Truth Calibration** in **Benizakura**.

Before any automated Large Language Model (LLM) judge—such as Anthropic Claude 3.5 Sonnet or OpenAI GPT-4o—can be evaluated, calibrated, or trusted to gate releases in CI/CD pipelines, we must establish whether qualified human software engineers can agree consistently on the 30 evaluation cases in the frozen Conquer benchmark.

### Fundamental Scientific Axioms

1. **The benchmark remains immutable; human annotations and consensus are separate artifacts.** The canonical benchmark dataset ([`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json)) is frozen permanently under SHA-256 `aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`. All annotator evaluations, adjudication records, and consensus decisions populate a dedicated, decoupled ground-truth artifact ([`evals/conquer/ground_truth_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/ground_truth_v1.json)).
2. **No automated or synthetic ground-truth labels:** LLM APIs, automated inference, synthetic proxies, or heuristic guesses must never be substituted for real human judgment in this phase.
3. **Passing human-human agreement does not prove that the benchmark is objectively correct. It establishes a reproducible human reference point suitable for subsequent judge calibration.**
4. **Production CI release gating remains strictly DISABLED.**

---

## 2. Why Human Calibration is Required

Automating evaluation via an LLM judge creates a recursive validation problem: *Who evaluates the evaluator?*

If we configure an automated judge without measuring human agreement first:
* We cannot distinguish genuine candidate regressions from idiosyncratic judge prompt artifacts.
* We cannot determine whether a disagreement between two models reflects a genuine ambiguity in software engineering or an artifact of model architecture.
* A judge with 100% internal consistency could simply be executing a 100% consistent position or verbosity bias.

Human calibration provides the empirical baseline against which the automated judge is compared. If two expert human engineers cannot reach agreement on whether Response A or Response B is superior for a given prompt, it is scientifically invalid to expect an automated LLM judge to render an authoritative verdict on that same case.

---

## 3. Why Two Independent Annotators are Used

Single-annotator labeling is notoriously prone to individual idiosyncratic preferences, cognitive fatigue, and localized knowledge gaps. An annotator with strong personal affinity for functional programming might penalize an object-oriented solution, while another annotator might overweight conciseness.

Benizakura mandates **two independent annotators** for every benchmark case because:
1. **Inter-Rater Reliability Quantification:** Cohen's kappa ($\kappa$) mathematically requires two independent raters evaluating the identical set of items.
2. **Identification of Inherent Problem Ambiguity:** When two qualified senior engineers reach opposite conclusions after reviewing the identical prompt and responses, the divergence reveals either a flawed benchmark question, an incomplete rubric, or a genuine multi-criterion engineering trade-off.
3. **Decoupling Agreement from Authority:** Measuring human-vs-human agreement *before* adjudication prevents lead reviewers from artificially smoothing over natural variance.

---

## 4. Why Blind A/B Presentation Matters

To eliminate subconscious bias, annotators must evaluate candidate answers in a **double-blind presentation view**.

### Information Concealment Matrix

Human annotators are presented with an isolated evaluation task containing only:
* Case ID (`case_id`)
* Technical Question (`question`)
* Interview Scenario Context (`context`)
* Blinded Response A (`response_a`)
* Blinded Response B (`response_b`)
* Accepted Engineering Alternatives (`accepted_alternatives`)
* Standard Rubric Criteria & Annotator Guide instructions

Annotators are **strictly blinded** to:
* Underlying system identity (Baseline vs Candidate)
* Model/provider identity (e.g. GPT-4o vs Claude 3.5 Sonnet vs human)
* Intended signal (e.g. `candidate_regression`, `candidate_improvement`, `near_tie`)
* Bias-probe classification (e.g. `verbosity`, `hallucination`, `edge_case_failure`)
* Expected or hypothetical winner
* Peer annotator ratings, rationales, or identity
* Adjudication records or consensus verdicts

### Randomized A/B Presentation Mapping

The mapping between `{Response A, Response B}` and `{Baseline, Candidate}` is generated pseudorandomly via `benizakura benchmark blind` with a deterministic seed (`--seed 42`). The resulting unblinding key (`evals/conquer/annotations/blinding_key_v1.json`) is stored in a private directory inaccessible to annotators during labeling.

---

## 5. End-to-End Human Annotation Lifecycle

```text
                  ┌─────────────────────────────────────┐
                  │ 1. FROZEN BENCHMARK v1.0.0          │
                  │ (SHA-256: aaf9f663...)              │
                  └──────────────────┬──────────────────┘
                                     │ benizakura benchmark blind
                                     ▼
                  ┌─────────────────────────────────────┐
                  │ 2. BLINDED ANNOTATION PACKETS       │
                  │ (Response A vs Response B)          │
                  └──────────┬──────────────────┬───────┘
                             │                  │
               Independent   │                  │  Independent
               Evaluation    ▼                  ▼  Evaluation
                  ┌────────────────┐      ┌────────────────┐
                  │ Annotator A    │      │ Annotator B    │
                  │ (30/30 Cases)  │      │ (30/30 Cases)  │
                  └────────┬───────┘      └────────┬───────┘
                           │                       │
                           └───────────┬───────────┘
                                       │ benizakura benchmark agreement
                                       ▼
                  ┌─────────────────────────────────────┐
                  │ 3. HUMAN AGREEMENT & COHEN'S KAPPA  │
                  │ (Raw agreement, κ, confusion matrix)│
                  └──────────────────┬──────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │ Calibration Kill-Gate           │
                    ▼                                 ▼
           [κ < 0.60: FAILED]               [κ >= 0.60: PASSED]
                    │                                 │
                    ▼                                 ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │ Halt Pipeline;        │         │ 4. ADJUDICATION       │
        │ Review Guide/Cases    │         │ (Lead reviews diffs)  │
        └───────────────────────┘         └───────────┬───────────┘
                                                      │
                                                      ▼
                                          ┌───────────────────────┐
                                          │ 5. CONSENSUS GROUND   │
                                          │ TRUTH ARTIFACT        │
                                          └───────────┬───────────┘
                                                      │ benizakura benchmark ground-truth freeze
                                                      ▼
                                          ┌───────────────────────┐
                                          │ 6. FROZEN GROUND      │
                                          │ TRUTH v1.0.0          │
                                          └───────────────────────┘
```

### Lifecycle States of Ground Truth
1. **`PENDING_HUMAN_ANNOTATION`**: Initial repository state. Benchmark cases exist and are frozen, but independent human labeling has not yet begun.
2. **`ANNOTATIONS_IN_PROGRESS`**: Blinded tasks distributed; annotators actively evaluating.
3. **`HUMAN_AGREEMENT_MEASURED`**: Both annotators have submitted complete 30-case evaluations; statistical agreement metrics calculated.
4. **`CALIBRATION_PASSED`** or **`CALIBRATION_FAILED`**: Formal kill-gate determination based on whether $\kappa \ge 0.60$.
5. **`ADJUDICATION_COMPLETE`**: All inter-rater disagreements have been authoritatively adjudicated with documented rationales.
6. **`GROUND_TRUTH_FROZEN`**: Cryptographic SHA-256 checksum and manifest generated; ground-truth artifact locked permanently.

---

## 6. Cohen's Kappa ($\kappa$) Statistical Methodology

To measure inter-rater reliability on the 3-class categorical decision outcome ($\mathcal{C} = \{\text{CANDIDATE}, \text{BASELINE}, \text{TIE}\}$), Benizakura implements multi-class **Cohen's kappa**:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

### Mathematical Definitions

Let $N$ be the total number of benchmark cases evaluated ($N = 30$).  
Let $n_{ij}$ denote the count of cases where Annotator 1 assigned class $i$ and Annotator 2 assigned class $j$.

#### 1. Observed Proportion of Agreement ($P_o$)
$$P_o = \frac{1}{N} \sum_{k \in \mathcal{C}} n_{kk}$$

#### 2. Expected Chance Agreement ($P_e$)
Under the hypothesis of rater independence, the expected marginal probability that both annotators choose class $k$ by chance is:
$$p_{1k} = \frac{1}{N} \sum_{j \in \mathcal{C}} n_{kj} \quad (\text{Annotator 1 marginal})$$
$$p_{2k} = \frac{1}{N} \sum_{i \in \mathcal{C}} n_{ik} \quad (\text{Annotator 2 marginal})$$
$$P_e = \sum_{k \in \mathcal{C}} p_{1k} \cdot p_{2k}$$

#### 3. Standard Error & 95% Confidence Interval
$$\text{SE}(\kappa) = \sqrt{\frac{P_o (1 - P_o)}{N (1 - P_e)^2}}$$
$$\text{CI}_{95\%} = [\kappa - 1.96 \cdot \text{SE}(\kappa), \; \kappa + 1.96 \cdot \text{SE}(\kappa)]$$

#### 4. Edge Case Handling
* If $N = 0$: Returns $\kappa = 0.0$.
* If $P_e = 1.0$: Both annotators exclusively assigned 100% of cases to the identical category. Returns $\kappa = 1.0$ if $P_o = 1.0$, else $0.0$.
* If $P_o \le P_e$: $\kappa \le 0.0$ (agreement no better than random guessing).

---

## 7. Predefined Methodological Threshold: $\kappa \ge 0.60$

In accordance with [DECISIONS.md (D8)](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/DECISIONS.md#L79-L90), Benizakura establishes an explicit, predefined calibration threshold:

$$\text{Calibration Gate: } \kappa \ge 0.60$$

### Standard Landis & Koch (1977) Categorization
* $\kappa < 0.20$: Slight agreement
* $0.21 \le \kappa \le 0.40$: Fair agreement
* $0.41 \le \kappa \le 0.60$: Moderate agreement
* $0.61 \le \kappa \le 0.80$: **Substantial agreement** (Benizakura Target)
* $0.81 \le \kappa \le 1.00$: Almost perfect agreement

Treat this threshold as a **predefined methodological threshold**, not as a universal scientific constant.

---

## 8. Calibration Kill-Gate Execution Logic

### Case A — If $\kappa < 0.60$: HUMAN CALIBRATION FAILED
If the independent annotators fail to achieve $\kappa \ge 0.60$:
1. The human calibration milestone is marked **FAILED**.
2. **LLM judge calibration must NOT proceed.**
3. The system generates an actionable diagnostic report:
   * Overall disagreement count and rate.
   * Categories with the highest disagreement rates (e.g. System Design vs Behavioral).
   * Rubric criteria with the lowest agreement (e.g. `TECHNICAL_DEPTH` vs `CLARITY`).
   * 3x3 confusion matrix highlighting directional asymmetry.
4. **Mandatory Remediation:**
   * Review [`ANNOTATOR_GUIDE.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/annotations/ANNOTATOR_GUIDE.md) for ambiguous scoring definitions.
   * Clarify guidelines on how to treat borderline near-ties.
   * If a benchmark case question contains contradictory constraints, refine the benchmark under a new version (`benchmark_v2.json`).

### Case B — If $\kappa \ge 0.60$: HUMAN CALIBRATION PASSED
If $\kappa \ge 0.60$:
1. Human calibration is marked **PASSED**.
2. Inter-annotator reliability is sufficient to proceed to consensus adjudication.
3. Production CI gating remains **DISABLED**.

---

## 9. Criterion-Level Agreement Analysis

In addition to the primary pairwise winner, Benizakura computes descriptive agreement and Cohen's $\kappa$ across the six standard rubric criteria:
1. `CORRECTNESS`
2. `RELEVANCE`
3. `COMPLETENESS`
4. `TECHNICAL_DEPTH`
5. `CLARITY`
6. `GROUNDEDNESS`

This granular decomposition reveals where human engineers align easily (typically `CORRECTNESS` and `RELEVANCE`) versus where subjective trade-offs cause friction (`TECHNICAL_DEPTH` vs `CLARITY`).

---

## 10. The Adjudication Process

When independent annotators disagree on a case ($\text{NormalizedWinner}_1 \ne \text{NormalizedWinner}_2$):

1. **Original Annotations are Immutable:** Neither annotator's ratings or rationales are altered or deleted.
2. **Independent Adjudicator Role:** An Engineering Lead acting as Adjudicator reviews:
   * Prompt and evaluation context
   * Candidate Answer and Baseline Answer
   * Annotator 1's choice, confidence, and rationale
   * Annotator 2's choice, confidence, and rationale
   * Criterion-level assessment discrepancies
3. **Adjudication Record Generation:** The adjudicator creates an explicit record containing:
   * Case ID (`case_id`)
   * Adjudicator identifier (`adjudicator_id`, e.g. `lead-human-000`)
   * Authoritative decision (`adjudicator_decision`: `CANDIDATE`, `BASELINE`, or `TIE`)
   * Detailed technical justification (`adjudicator_rationale`)
   * Submission timestamp
4. **Consensus Container:** The consensus label is stored as a new artifact with `agreement_type = "ADJUDICATED"`. Unanimous cases are stored with `agreement_type = "UNANIMOUS"`.

---

## 11. Ground-Truth Artifact Freezing & Immutability

Consensus ground truth is stored in a dedicated artifact:
* Dataset: `evals/conquer/ground_truth_v1.json`
* Manifest: `evals/conquer/ground_truth_v1.manifest.json`
* Checksum: `evals/conquer/ground_truth_v1.sha256`

### Strict Freezing Invariants
The ground-truth artifact cannot transition to `GROUND_TRUTH_FROZEN` unless:
1. **Benchmark Hash Integrity:** The benchmark SHA-256 strictly matches `aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`.
2. **Complete Coverage:** All 30 benchmark cases have a final consensus label (`status` is `CONSENSED` or `ADJUDICATED`).
3. **Zero Pending Disagreements:** Every disagreement has an accompanying valid adjudication record.
4. **Deterministic Serialization:** JSON serialization uses sorted keys, 2-space indentation, and ASCII encoding.

---

## 12. Benchmark and Ground-Truth Versioning

The benchmark and ground truth maintain strictly coupled semantic versions:

```text
Benchmark:    conquer-benchmark-v1 (v1.0.0) [FROZEN]
Ground Truth: conquer-ground-truth-v1 (v1.0.0)
```

If the benchmark prompt cases or questions are modified in the future:
* The benchmark SHA-256 changes.
* `ground_truth_v1.json` becomes invalid for that new benchmark.
* A new dataset version (`benchmark_v2.json`) must be created, requiring a fresh round of double-blind human calibration.

---

## 13. Why Automated & LLM-Generated Labels are Strictly Forbidden

It is tempting in fast-paced software development to prompt Claude 3.5 Sonnet or GPT-4o to "generate the human ground truth labels" to speed up development.

Benizakura **strictly forbids** this practice:
1. **Circularity:** Using an LLM to generate ground truth and then using that ground truth to measure whether an LLM can be trusted is circular logic.
2. **Bias Propagation:** LLM-generated labels inherit the exact model-family, verbosity, and position biases Benizakura is designed to detect and prevent.
3. **Scientific Invalidation:** Results derived from synthetic ground truth cannot be published or defended to auditors or customers.

---

## 14. What This Phase Does NOT Prove

It is essential to maintain scientific modesty regarding what human calibration proves and does not prove:

> **Important Methodological Disclaimers:**
>
> 1. **Passing human-human agreement ($\kappa \ge 0.60$) does NOT prove that the benchmark is objectively correct.** It proves only that two qualified human software engineers achieved substantial consensus under the specified rubric guidelines.
> 2. **Human agreement does NOT prove that human judgment is objective or free from bias.** Human engineers also share cultural, pedagogical, and stylistic habits.
> 3. **Human calibration does NOT prove that an LLM judge is trustworthy.** LLM judge validation is an independent empirical milestone that begins *after* human ground truth is frozen.
> 4. **Production CI gating remains DISABLED.** Automated gating will be considered only after LLM judges demonstrate strong agreement ($\kappa \ge 0.60$) and low position instability ($\le 20\%$) against this frozen human ground truth.

---

## 15. Summary of Artifacts

| Artifact | Path | Governance |
| :--- | :--- | :--- |
| **Official Benchmark Dataset** | [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json) | **FROZEN** (`aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`) |
| **Double-Blind Tasks** | [`evals/conquer/annotations/blinded_tasks.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/annotations/blinded_tasks.json) | Distributed to human annotators |
| **Private Blinding Key** | [`evals/conquer/annotations/blinding_key_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/annotations/blinding_key_v1.json) | Restricted access |
| **Annotator Handbook** | [`evals/conquer/annotations/ANNOTATOR_GUIDE.md`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/annotations/ANNOTATOR_GUIDE.md) | Standardized grading guide |
| **Ground-Truth Store** | [`evals/conquer/ground_truth_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/ground_truth_v1.json) | `PENDING_HUMAN_ANNOTATION` |
| **Ground-Truth Manifest** | [`evals/conquer/ground_truth_v1.manifest.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/ground_truth_v1.manifest.json) | Metadata manifest |
