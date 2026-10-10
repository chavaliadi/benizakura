# Benizakura Pilot Annotation Pack v1 — Selection Methodology, Probe Coverage & Scope Limitations

> **Document Classification:** OWNER-FACING RESEARCH RECORD  
> **RESTRICTION:** DO NOT DISTRIBUTE TO HUMAN ANNOTATORS. Contains hidden bias probe classifications and benchmark design metadata.  
> **Pilot Version:** `pilot_v1`  
> **Creation Date:** 2026-10-11  
> **Canonical Full Benchmark Target:** `evals/conquer/benchmark_v1.json`  
> **Canonical Full Benchmark SHA-256:** `aaf9f66360620603d0144796dcb954345a6d28d2b26e3a1d032c7b6cd4df9314`

---

## 1. Executive Summary & Purpose

The `pilot_v1` pack provides a reduced-scope, self-contained 15-case evaluation packet designed to collect initial human annotations from two independent evaluators without requiring codebase access or command-line execution.

**Scientific Integrity Notice:**  
This pilot is strictly an initial calibration exercise. It **does not** replace, modify, or supersede the official frozen 30-case benchmark (`benchmark_v1.json`), and its inter-annotator agreement metrics **cannot** be used to freeze official ground truth or unlock Phase 4 automated gating.

---

## 2. Selection Methodology

The 15 cases were selected deterministically from the 30-case canonical benchmark using a stratified sampling approach:

1. **Exact Proportional Category Representation:**  
   Every category in the benchmark is represented at exactly 50% of its full benchmark count:
   - **DSA:** 3 / 6 cases (50.0%)
   - **SYSTEM_DESIGN:** 4 / 8 cases (50.0%)
   - **BACKEND:** 3 / 6 cases (50.0%)
   - **FRONTEND:** 2 / 4 cases (50.0%)
   - **BEHAVIORAL:** 3 / 6 cases (50.0%)
   - **Total:** 15 / 30 cases (50.0%)

2. **Diverse Bias-Probe Coverage:**  
   Cases were chosen to test all key bias dimensions without inspecting expected winners or biasing the outcome:
   - `verbosity`: Testing resilience against lengthy but shallow answers.
   - `hallucination`: Testing detection of authoritative but factually flawed claims.
   - `near_tie`: Testing calibration on subtle quality boundaries and legitimate ties.
   - `alternative_architecture`: Testing tolerance for valid alternative designs.
   - `criterion_tradeoff`: Testing multi-dimensional trade-off decisions.

3. **No Outcome-Based Selection:**  
   Cases were selected strictly based on category and probe diversity. No candidate or baseline win expectations were used to favor any system or model.

4. **Preservation of Case IDs and Blinding:**  
   Original case IDs are preserved verbatim. The A/B presentation order matches the official blinding key (`blinding_key_v1.json`, seed 42) so that subsequent unblinding is exact.

---

## 3. Selected Case Inventory

The 15 selected cases are listed below with their category, subcategory, and associated bias probes:

| # | Case ID | Category | Subcategory | Bias Probes (Owner Record) |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `dsa-002` | DSA | Data Structures & Hash Tables | `position`, `verbosity` |
| 2 | `dsa-003` | DSA | Binary Trees | `position`, `alternative_architecture`, `near_tie` |
| 3 | `dsa-005` | DSA | Graphs & Topological Sort | `position`, `hallucination` |
| 4 | `sys-001` | SYSTEM_DESIGN | Distributed ID Generation | `position`, `alternative_architecture`, `near_tie` |
| 5 | `sys-003` | SYSTEM_DESIGN | Real-time Notifications | `position`, `criterion_tradeoff` |
| 6 | `sys-005` | SYSTEM_DESIGN | Social Media Feed Architecture | `position`, `hallucination` |
| 7 | `sys-006` | SYSTEM_DESIGN | Caching & High Concurrency | `position`, `verbosity` |
| 8 | `backend-003` | BACKEND | Connection Pools & Sizing | `position`, `verbosity` |
| 9 | `backend-004` | BACKEND | Message Queues & Reliability | `position`, `alternative_architecture`, `near_tie` |
| 10 | `backend-005` | BACKEND | E-Commerce Concurrency | `position`, `criterion_tradeoff` |
| 11 | `frontend-002` | FRONTEND | DOM Performance & Virtual Lists | `position`, `verbosity` |
| 12 | `frontend-004` | FRONTEND | State Management Architecture | `position`, `alternative_architecture`, `near_tie` |
| 13 | `behavioral-003` | BEHAVIORAL | Tech Debt vs Product Velocity | `position`, `criterion_tradeoff` |
| 14 | `behavioral-005` | BEHAVIORAL | Ambiguity & Scope Triage | `position`, `near_tie` |
| 15 | `behavioral-006` | BEHAVIORAL | Incident Postmortem | `position`, `hallucination` |

---

## 4. Bias Probe Representation Breakdown

Across the 15 selected cases, probe representation is as follows:

- **`verbosity` (4 cases):** `dsa-002`, `sys-006`, `backend-003`, `frontend-002`
- **`hallucination` (3 cases):** `dsa-005`, `sys-005`, `behavioral-006`
- **`near_tie` (5 cases):** `dsa-003`, `sys-001`, `backend-004`, `frontend-004`, `behavioral-005`
- **`alternative_architecture` (4 cases):** `dsa-003`, `sys-001`, `backend-004`, `frontend-004`
- **`criterion_tradeoff` (3 cases):** `sys-003`, `backend-005`, `behavioral-003`
- **`position` (15 cases):** All 15 cases (evaluated bidirectionally during benchmark blinding)

---

## 5. Limitations of the 15-Case Pilot Subset

1. **Statistical Power:** A 15-case sample has a larger confidence interval for Cohen's kappa than the full 30-case benchmark. Standard errors will be approximately $\sqrt{2}$ times larger.
2. **Generalizability:** While category proportions match the full benchmark (50% each), niche edge cases in the remaining 15 benchmark cases (e.g. `dsa-006` interval edge case, `sys-008` two-phase commit edge case, `backend-006` migration locks) are not represented in this pilot.
3. **No Gating Authority:** The pilot results must **never** be used to freeze `ground_truth_v1.json` or unlock Phase 4 CI/CD gating. The full 30-case benchmark remains the sole authoritative instrument for release gating.

---

## 6. Artifact Distribution Control Matrix

| File Path | Recipient | Contains Secrets / Unblinding Key? | Notes |
| :--- | :--- | :-: | :--- |
| `evals/conquer/annotations/pilot_v1/README.md` | **Shareable** (Annotators) | NO | Beginner-friendly instructions |
| `evals/conquer/annotations/pilot_v1/pilot_tasks.json` | **Shareable** (Annotators) | NO | 15 blinded tasks (Response A / B) |
| `evals/conquer/annotations/pilot_v1/ANNOTATOR_GUIDE.md` | **Shareable** (Annotators) | NO | Rubric instructions and principles |
| `evals/conquer/annotations/pilot_v1/ANNOTATION_SCHEMA.md` | **Shareable** (Annotators) | NO | Submission schema and types |
| `evals/conquer/annotations/pilot_v1/submission_template.json` | **Shareable** (Annotators) | NO | Blank submission template |
| `evals/conquer/annotations/pilot_v1/SELECTION_AND_SCOPE.md` | **OWNER-ONLY** (Maintainer) | Contains Probe Labels | **DO NOT SHARE WITH ANNOTATORS** |
| `evals/conquer/annotations/blinding_key_v1.json` | **RESTRICTED** (Maintainer) | Private Key | Keep in private location |
| `evals/conquer/benchmark_v1.json` | **INTERNAL** (Repository) | Ground Truth Signals | Not shared with annotators |
