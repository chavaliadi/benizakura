# Benizakura Pilot Evaluation — Annotator Instructions

Welcome, and thank you for helping evaluate technical interview responses!

---

## 1. What is Benizakura?

**Benizakura** is an evaluation and regression-testing framework for AI-generated technical interview evaluations. It rigorously measures the quality and depth of engineering answers across diverse computer science domains. To ensure our automated evaluation systems remain reliable and fair, we calibrate them against independent human judgments from software engineers.

---

## 2. What Are You Being Asked to Do?

You are being asked to compare two anonymous technical interview responses (**Response A** and **Response B**) for **15 evaluation cases**.

For each case, you will:
1. Read the interview question and context.
2. Read both responses side-by-side.
3. Decide which response is technically superior, or whether they are essentially tied (**`A`**, **`B`**, or **`TIE`**).
4. Evaluate six specific criteria (Correctness, Relevance, Completeness, Technical Depth, Clarity, Groundedness).
5. Provide a brief technical rationale (2–4 sentences) and your confidence level.

---

## 3. Important Details

- **Number of Cases:** Exactly **15 cases**.
- **Estimated Time Commitment:** Approximately **45 to 60 minutes** in total (~3 to 4 minutes per case). This is an estimate; please take the time you need to make sound technical judgments.
- **Zero Technical Setup Required:** You do **not** need:
  - Access to Conquer or its source code.
  - Access to the Benizakura repository.
  - Python, Git, a terminal, or any development environment.
  - To run Benizakura or execute validation commands yourself.
  You only need the five shared files in this folder and a standard text editor or JSON editor.
- **Independent Evaluation:** You **must work completely independently**. Do **not** consult, compare notes with, or discuss any of the cases with the other evaluator before both evaluations have been submitted.
- **No "Right Answers" or Performance Test:** This is **not** a test of your personal knowledge or interview skills. We want your honest engineering judgment on the merits of the two answers.
- **Ties are Fully Valid:** If neither answer is clearly better, or if both present equally sound engineering trade-offs, choose **`TIE`**. Do not force a winner when there is no clear distinction.
- **Double-Blind Evaluation:** The identities of the underlying models generating the responses are blinded and randomized. Focus purely on technical substance.

---

## 4. How to Complete the Task

The evaluation process consists of five simple steps:

### Step 1: Read the Instructions
Read this `README.md` and [ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md). The guide explains:
- The ten core evaluation principles (e.g. why longer answers are not automatically better, how to spot confident technical errors, and how to treat valid alternative designs).
- Definitions and expectations for the six evaluation criteria.

### Step 2: Open the Tasks File
Open [pilot_tasks.json](pilot_tasks.json) in your preferred text editor (e.g. VS Code, Sublime Text, Notepad++, or any text editor) to inspect all 15 questions and response pairs.
This file contains the 15 blinded comparison cases, including the question, context, Response A, Response B, and any accepted technical alternatives.

### Step 3: Make a Copy of the Submission Template
Make a separate copy of [submission_template.json](submission_template.json).
Save your copy as:
```text
pilot_submission_<your_id>.json
```
*(For example: `pilot_submission_human_001.json` or `pilot_submission_human_002.json`)*

### Step 4: Complete All 15 Judgments Independently
For each of the 15 cases in your template:
1. Set `"annotator_id"` to your assigned ID (e.g., `"human-001"` or `"human-002"`).
2. Set `"winner"` to `"A"`, `"B"`, or `"TIE"`.
3. Set `"confidence"` to a numeric score between `0.0` and `1.0` inclusive. Recommended reference anchors:
   - `0.50`: Borderline or near-tie (marginal difference).
   - `0.80`: Clear preference (one answer is noticeably better).
   - `1.0`: High certainty (decisive difference).
   *(Note: These are recommended reference values, not the only allowed scores; any float from 0.0 to 1.0 is valid).*
4. Write a 2–4 sentence `"rationale"` explaining your decision (minimum 15 characters).
5. For each of the 6 criteria in `"criterion_assessments"`:
   - Select `"winner"` (`"A"`, `"B"`, or `"TIE"`).
   - Write a short `"rationale"` (minimum 10 characters).
   - Optionally provide `"evidence"` (a brief quote from the answer, or `null`).
6. Replace the placeholder `"YYYY-MM-DDTHH:MM:SSZ"` in `"timestamp"` with your actual evaluation timestamp in ISO-8601 format (e.g., `"2026-10-12T14:30:00Z"`).

Refer to [ANNOTATION_SCHEMA.md](ANNOTATION_SCHEMA.md) for full schema details and examples.

---

## 5. Editing JSON Without Headaches

If you are not used to editing JSON files:
- Use a code editor like **VS Code**, **Sublime Text**, or **Notepad++**, which highlights JSON syntax and red flags missing commas or unclosed quotes.
- Alternatively, you can use a free in-browser JSON editor such as [JSON Editor Online](https://jsoneditoronline.org/) to edit and validate your JSON structure without installing any software.
- Ensure all quotes are standard double quotes (`"`) and that numbers (like `0.85`) do not have quotes around them.

---

## 6. Returning Your Completed Work

Once you have filled in all 15 cases:
1. Save your completed file as `pilot_submission_<your_id>.json`.
2. Return your completed JSON submission file to me via email or direct message.
3. Please **do not share your answers or discuss any case with the other evaluator** until both submissions have been completed and collected.

---

## 7. Questions or Clarifications?

If any question is ambiguous, or if you run into any formatting or file issues:
- **Contact:** The project maintainer directly.
- Do **not** consult the other annotator.

Thank you very much for your time and contribution!
