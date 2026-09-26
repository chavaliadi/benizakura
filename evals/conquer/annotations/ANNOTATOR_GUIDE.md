# Benizakura — Human Annotator & Evaluation Guide

> **Document Type:** Evaluator Instruction Handbook  
> **Target Dataset:** [`evals/conquer/benchmark_v1.json`](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/evals/conquer/benchmark_v1.json)  
> **Intended Audience:** Senior Software Engineers & Tech Leads participating in double-blind benchmark calibration  
> **Status:** Active / Frozen for Calibration Milestone

---

## Welcome to the Benizakura Calibration Process

As a human annotator, you are establishing the ground-truth standard against which Benizakura's automated LLM evaluation pipeline will be measured. Your decisions directly determine whether automated judges (such as Claude 3.5 Sonnet and GPT-4o) can be trusted to gate code and prompt releases in CI/CD.

We are evaluating technical interview answers generated for **Conquer** (`POST /api/interview/score`), spanning Data Structures & Algorithms (DSA), System Design, Backend Engineering, Frontend Engineering, and Behavioral STAR evaluations.

---

## The Ten Core Evaluation Principles

Every annotator must internalize and adhere to these ten principles:

### Principle 1 — Evaluate Technical Substance, Not Writing Style
Base your decision on whether the engineer understands the underlying computer science, systems architecture, and engineering trade-offs. Do not reward flowery prose, conversational warmth, introductory fluff, or decorative markdown formatting if technical depth is absent.

### Principle 2 — Longer Answers Are NOT Automatically Better (Beware Verbosity Bias)
A concise, 100-word response that correctly identifies asymptotic complexity, memory layout, and failure modes is strictly superior to a 500-word essay that speaks vaguely about architectural philosophy without solving the problem. Actively penalize unnecessary padding and repetitive explanations.

### Principle 3 — Polished Language is NOT Evidence of Correctness
Watch out for **authoritative hallucinations**: answers written with high confidence, impeccable grammar, and professional jargon that contain a subtle, fatal technical flaw (e.g. claiming standard BFS detects directed graph cycles, or claiming Read Committed transaction isolation holds shared locks until commit). Verify every technical assertion against primary engineering principles.

### Principle 4 — Valid Alternative Architectures Must NOT Be Penalized
In software engineering, there is rarely a single "correct" design. If the question asks for an LRU cache or a URL shortener, an answer proposing a Key Generation Service (KGS) and an answer proposing Twitter Snowflake are both valid engineering solutions with distinct trade-offs. Consult the `accepted_alternatives` section of the task sheet. Grade based on whether the candidate articulates valid assumptions and trade-offs, not whether they matched a single textbook implementation.

### Principle 5 — TIE is a First-Class, Valid Verdict
Do **not** force an arbitrary winner. When both answers are technically accurate, present comparable depth, and make reasonable engineering trade-offs, declare **`TIE`**. Benizakura's statistical engine relies on genuine ties to calibrate judge sensitivity.

### Principle 6 — Penalize Missing Information Only When Material
Do not downgrade an answer for omitting tangential trivia. Penalize missing details only if the omission causes the solution to fail in production (e.g., omitting idempotency keys in a payment processing pipeline, or ignoring disk page splits in database index writes).

### Principle 7 — Distinguish Technical Errors from Stylistic Imperfections
A technical error (e.g. $O(N^2)$ loop instead of $O(N)$ two-pointer, off-by-one boundary failure, race condition in Redis GET/SET) is a severe flaw. A stylistic imperfection (e.g. naming a variable `i` instead of `index`, or using a slightly informal phrasing) is minor. Always prioritize correctness and computational invariants over aesthetics.

### Principle 8 — Strict Double-Blind Evaluation
You must never attempt to guess which response is "Baseline" or "Candidate". You will receive tasks labeled only as **`Response A`** and **`Response B`**. The mapping to underlying system models is randomized and cryptographically stored. Evaluate strictly on the relative merits of A versus B.

### Principle 9 — Independent, Isolated Evaluation
Annotators must work in complete isolation. You must not discuss evaluation cases, share notes, or compare answers with other annotators until after your submission JSON has been formally submitted and committed to version control.

### Principle 10 — Disagreements Require Explicit Adjudication
Never compromise by silently altering your scores to match a peer. If Annotator 1 selects `A` and Annotator 2 selects `B`, this represents an informative boundary condition. Disagreements will be reviewed by an Engineering Lead who will conduct an adjudication session and document the technical rationale for the final consensus label.

---

## Step-by-Step Annotation Walkthrough

### Step 1: Read the Prompt & Context
Carefully inspect the `question`, `context`, and `accepted_alternatives`. Identify the core invariant:
* *What are the time and space constraints?*
* *What edge cases must be handled (null inputs, boundary equality, race conditions)?*
* *What are the failure modes under high load?*

### Step 2: Analyze Response A and Response B
Read both responses side-by-side. On scratch paper or your editor, note:
* **Factual Inaccuracies:** Does either response make false claims about operating systems, networking protocols, or algorithms?
* **Edge Cases:** Does the solution break on touching intervals, empty lists, or concurrent callers?
* **Depth & Evidence:** Does the author explain *why* something works (e.g. leftmost prefix index seek) or merely state that it works?

### Step 3: Grade Specific Criteria
Complete the criterion-level assessments:
1. `CORRECTNESS`: Which response is free from algorithmic and architectural bugs?
2. `RELEVANCE`: Does the response directly address the question without dodging?
3. `COMPLETENESS`: Are all parts of the multi-part question answered?
4. `TECHNICAL_DEPTH`: Does the response demonstrate mechanical sympathy and low-level understanding?
5. `CLARITY`: Is the explanation structured, unambiguous, and coherent?
6. `GROUNDEDNESS`: Are the performance claims and operational metrics realistic?

### Step 4: Choose the Overall Winner
* Select **`A`** if Response A is substantively superior.
* Select **`B`** if Response B is substantively superior.
* Select **`TIE`** if both responses are of approximately equivalent quality or make equally valid alternative trade-offs.

### Step 5: Write the Justification & Assign Confidence
Write a 2–4 sentence technical justification highlighting the decisive reason for your verdict. Cite direct quotes where applicable. Assign your confidence score:
* `1.0`: Absolute certainty (e.g., Response A is $O(N)$ and Response B has a fatal bug).
* `0.80`: Clear preference (e.g., Response A handles edge cases and operational trade-offs more thoroughly).
* `0.50`: Borderline decision / near-tie where one answer barely edges out the other.

---

## Submission Protocol

1. Annotators receive their assigned task batches in `evals/conquer/annotations/tasks/task_batch_<id>.json`.
2. Save your completed evaluations into:
   ```text
   evals/conquer/annotations/submissions/<annotator_id>/<case_id>.json
   ```
3. Commit and push your submissions branch for verification.

Thank you for contributing to the scientific rigor of Benizakura!
