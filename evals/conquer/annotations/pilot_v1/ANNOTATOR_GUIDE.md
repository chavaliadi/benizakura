# Benizakura Pilot Evaluation — Annotator & Evaluation Guide

> **Document Type:** Evaluator Instruction Handbook  
> **Dataset Scope:** 15-Case Pilot Annotation Subset (`pilot_tasks.json`)  
> **Audience:** Human Evaluators participating in independent technical comparison  
> **Status:** Active / Frozen for Pilot v1

---

## 1. Introduction

As an evaluator for this pilot study, your task is to assess and compare technical interview responses. Your independent judgments will help us measure how closely automated evaluators mirror real software engineering standards.

You will evaluate 15 blinded comparison cases across five domains:
1. **Data Structures & Algorithms (DSA)**
2. **System Design**
3. **Backend Engineering**
4. **Frontend Engineering**
5. **Behavioral & Engineering Leadership**

---

## 2. Core Principles for Technical Evaluation

Please keep these ten principles in mind throughout your evaluation:

### Principle 1 — Evaluate Technical Substance, Not Writing Style
Focus on whether the response demonstrates true understanding of computer science fundamentals, system architecture, and trade-offs. Do not reward conversational fluff, verbose introductions, or polished markdown formatting if core technical substance is missing.

### Principle 2 — Longer Answers Are NOT Automatically Better (Beware Verbosity Bias)
A concise, 100-word response that correctly identifies asymptotic complexity, memory layout, and concurrency edge cases is strictly superior to a 500-word essay that speaks vaguely about architectural philosophy without solving the problem. Actively penalize unnecessary padding and repetitive explanations.

### Principle 3 — Polished Language is NOT Evidence of Correctness (Beware Authoritative Hallucinations)
Watch out for answers written with high confidence and professional phrasing that contain subtle but fatal technical errors (e.g., claiming standard BFS detects directed graph cycles, or claiming Read Committed transaction isolation prevents non-repeatable reads). Verify technical claims against primary engineering principles.

### Principle 4 — Valid Alternative Architectures Must NOT Be Penalized
In software engineering, there is rarely only one "right" solution. If an answer proposes an alternative design (e.g., Key Generation Service vs. Snowflake for distributed IDs, or recursive vs. iterative tree traversal), consult the `accepted_alternatives` section of the task. As long as the candidate articulates valid assumptions and trade-offs, do not penalize them simply because their approach differs from a textbook answer.

### Principle 5 — TIE is a First-Class, Valid Verdict
Do **not** force an arbitrary winner. When both answers are technically sound, exhibit comparable depth, and make reasonable engineering trade-offs, select **`TIE`**. Reliable calibration relies on honest ties.

### Principle 6 — Penalize Missing Details Only When Material
Do not downgrade an answer for omitting tangential trivia. Penalize missing details only if the omission would cause the solution to fail in production (e.g., omitting idempotency keys in payment processing, or ignoring boundary edge cases in pointer logic).

### Principle 7 — Distinguish Technical Errors from Stylistic Imperfections
A technical flaw (e.g., $O(N^2)$ loop instead of $O(N)$ two-pointer pass, race conditions, or off-by-one errors) is serious. A stylistic imperfection (e.g., naming a variable `i` instead of `index`, or slightly informal language) is minor. Always prioritize correctness over superficial style.

### Principle 8 — Strict Double-Blind Evaluation
Never attempt to guess the author or model of either answer. Tasks are presented strictly as **Response A** and **Response B** with randomized ordering. Judge purely on the merits of A versus B.

### Principle 9 — Independent, Isolated Evaluation
You must complete your evaluations in complete isolation. Do **not** discuss cases, compare notes, or share answers with the other evaluator before your submission has been completed and returned.

### Principle 10 — Disagreements are Valuable Data
Do not second-guess yourself to try to predict what someone else might choose. Disagreements between experienced engineers represent informative boundary cases and help us improve evaluation rubrics.

---

## 3. The Six Evaluation Criteria

For each case, in addition to choosing an overall winner, you will evaluate six specific dimensions:

1. **`CORRECTNESS`**  
   Is the answer free of algorithmic bugs, architectural errors, false claims, and incorrect complexity bounds ($O(N)$, etc.)?
2. **`RELEVANCE`**  
   Does the response directly address the specific prompt asked, without dodging or going off on tangents?
3. **`COMPLETENESS`**  
   Does the response cover all explicit requirements of the question, including constraints, boundary conditions, and edge cases?
4. **`TECHNICAL_DEPTH`**  
   Does the answer show low-level systems understanding, mechanical sympathy, and concrete trade-off analysis (e.g., memory layouts, locking, indexing, failure modes)?
5. **`CLARITY`**  
   Is the explanation structured, unambiguous, easy to follow, and precise?
6. **`GROUNDEDNESS`**  
   Are the performance claims, latency estimates, and capacity numbers physically and mathematically defensible in real-world systems?

For each criterion, choose **`A`**, **`B`**, or **`TIE`**, provide a brief rationale (at least 10 characters), and optionally include direct quote evidence.

---

## 4. Confidence Rating Guidance

When assigning your overall confidence score for each case:
- **Allowed Range:** Any numeric value from `0.0` through `1.0`, inclusive.
- **Recommended Reference Anchors:**
  - **`0.50` (Borderline / Near-Tie):** Marginal difference where one answer barely edges out the other, or a tough call between competing trade-offs.
  - **`0.80` (Clear Preference):** One answer is noticeably better (e.g., handles edge cases and operational trade-offs more thoroughly).
  - **`1.0` (High Certainty):** Clear, decisive difference (e.g., one answer is optimal and the other has a fatal bug or incorrect complexity).

> **Note on Reference Values:** The scores `0.50`, `0.80`, and `1.0` are recommended reference anchors to guide your calibration, not the only permitted values. Any numeric value between `0.0` and `1.0` (for example, `0.70`, `0.85`, or `0.95`) is valid.

---

## 5. Walkthrough: How to Evaluate a Case

1. **Read Question & Context:** Read the interview question and `accepted_alternatives`. Identify the core requirements (asymptotic bounds, failure modes, scale).
2. **Analyze Both Answers:** Read Response A and Response B side-by-side. Check for:
   - Factual inaccuracies or flawed assumptions.
   - Unhandled edge cases.
   - Depth of justification (explaining *why* vs. asserting *that*).
3. **Assess Criteria:** Fill in the 6 criterion assessments (`A`, `B`, or `TIE`).
4. **Choose Overall Winner:** Select **`A`**, **`B`**, or **`TIE`**.
5. **Write Rationale & Confidence:** Write a 2–4 sentence summary of why you favored one answer (or why they are tied), citing specific reasons. Set your confidence score (`0.0` to `1.0`, referencing anchors `0.50`, `0.80`, `1.0`).

Thank you for contributing your technical expertise to this evaluation!
