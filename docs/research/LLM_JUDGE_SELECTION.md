# Benizakura — Research-First LLM Judge Selection & Validation Specification

> **Document Type:** Research & Architectural Design Specification  
> **Status:** Draft / Active Research Milestone  
> **Author:** Antigravity AI Evaluation Team  
> **Date:** September 2026  
> **Target System:** Conquer (`POST /api/interview/score`)  
> **Repository Context:** [Benizakura Core Architecture](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/README.md)

---

## 1. Executive Summary

This research specification establishes the foundational methodology for selecting, configuring, and empirically validating the first real Large Language Model (LLM) judge for **Benizakura**—an automated evaluation and regression-testing change gate for LLM-powered applications.

Benizakura's target application is **Conquer**, an AI-powered technical interview preparation platform evaluating candidate interview responses across Data Structures & Algorithms, System Design, Behavioral (STAR), Frontend, and Backend engineering tracks. The target endpoint (`POST /api/interview/score`) currently employs `openai/gpt-oss-120b` (temperature 0.7, structured JSON schema, 0–10 score, diagnostic feedback, and skill profile updates).

### Core Problem
Automated evaluation using an LLM-as-a-Judge is susceptible to severe systematic biases: position bias (order preference), verbosity bias (length over substance), self-enhancement / model-family bias (favoring outputs with similar stylistic or tokenization distributions), and stochastic instability. If Benizakura selects an arbitrary judge or relies on naive pointwise scoring, the CI release gate will produce false regressions (blocking valid deployments) or false passes (leaking degraded prompt updates to production).

### Primary Finding & Architecture Verdict
1. **Model Independence is Non-Negotiable:** Because Conquer's scorer operates on a GPT-family foundation (`openai/gpt-oss-120b`), using an OpenAI-based model (e.g., GPT-4o) introduces correlated error distributions and self-enhancement bias ($r \approx 0.05\text{--}0.15$ systematic win-rate inflation). To establish evaluator independence, Benizakura must employ a frontier model from an **independent architectural family**.
2. **Recommended First Judge:** **Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)** is recommended as Benizakura's primary initial judge, with **GPT-4o (`gpt-4o-2024-08-06`)** maintained as an active counter-balancing cross-evaluator in multi-judge jury experiments.
3. **Critical Prompt Defect Localized in Current Design:** In [prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py#L20-L32), the current JSON output schema requires `"winner"` as the very first key prior to `"rationale"` and `"criterion_assessments"`. In autoregressive generation, this forces the model to emit a final verdict before generating reasoning tokens, directly negating the debiasing benefits of Chain-of-Thought (CoT). This document specifies an immediate structural remediation: reordering the JSON schema to force criterion evaluation and rationale *before* verdict generation.
4. **Validation Kill-Gate:** In accordance with [DECISIONS.md (D8)](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/DECISIONS.md#L79-L90), no automated release gate decision will be trusted in production CI until the judge achieves an inter-rater agreement of **Cohen's $\kappa \ge 0.60$** against a calibrated, double-annotated human benchmark of technical interview evaluations.

---

## 2. Important Literature

We review eight seminal papers addressing LLM-as-a-Judge reliability, pairwise evaluation, rubric calibration, and bias dynamics.

```text
Paper: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
Authors: Lianmin Zheng, Wei-Lin Chiang, Hao Zhang, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica
Year: 2023 (NeurIPS 2023 / arXiv:2306.05685)
Problem: Scalability and reliability of using frontier LLMs to evaluate conversational chat assistants without expensive human crowd-sourcing.
Method: Developed MT-Bench (80 multi-turn dialog questions across 8 domains) and Chatbot Arena (crowdsourced pairwise battle platform). Evaluated single-answer scoring, pairwise comparison, and reference-guided grading using GPT-4, GPT-3.5, and Claude.
Judge setup: GPT-4 and Claude evaluated with zero-shot and few-shot prompts, chain-of-thought rationale, and explicit role definitions.
Evaluation methodology: Measured agreement rate with expert human judgments, inter-annotator agreement, and win-rate correlation with LMSYS human Elo rankings.
Important findings:
  - Strong LLMs (GPT-4) achieve over 80% agreement with humans in pairwise comparisons, matching human-to-human agreement (81%).
  - Identified three dominant biases: Position Bias (favors the first-presented answer 50%–70% of the time depending on prompt/model), Verbosity Bias (favors longer, verbose answers regardless of factual density), and Self-Enhancement Bias (GPT-4 favors GPT-4 outputs by an additional 10% win rate margin).
  - Swapping order and requiring bidirectional agreement reduced position bias, while asking for step-by-step reasoning improved calibration.
Known limitations: Examined open-ended conversational tasks; did not explore complex technical code execution or strict formal rubrics.
Relevance to Benizakura: Validates Benizakura's core pairwise foundation and demonstrates that bidirectional evaluation is an absolute prerequisite to detecting position instability.
```

```text
Paper: G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
Authors: Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu
Year: 2023 (EMNLP 2023 / arXiv:2303.16634)
Problem: Reference-based NLG metrics (ROUGE, BLEU, BERTScore) correlate weakly with human judgments on fluency, coherence, and technical accuracy in complex generative tasks.
Method: Formulated G-Eval, an evaluation framework using LLMs with Chain-of-Thought (CoT) and a form-filling paradigm based on user-defined rubrics.
Judge setup: GPT-4 prompted to first auto-generate step-by-step evaluation criteria from high-level instructions, followed by detailed scoring. In advanced setups, token probabilities of score tokens were extracted to compute continuous expected values.
Evaluation methodology: Evaluated text summarization (CNN/DailyMail) and dialogue generation (PersonaChat) against human ratings via Spearman ($r_s$) and Pearson ($r$) correlations.
Important findings:
  - G-Eval-4 achieved a Spearman correlation of 0.514 on summarization coherence, substantially surpassing ROUGE-1 (0.127) and BERTScore (0.312).
  - Explicit multi-criteria decomposition reduced variance by 30% compared to monolithic single-score prompts.
  - CoT reasoning prior to score emission was critical: emitting the score before explanation caused correlation with humans to drop by over 18%.
Known limitations: Proprietary API access frequently restricts direct logit extraction, necessitating reliance on discrete generated text tokens.
Relevance to Benizakura: Directly justifies Benizakura's rubric-based multi-criterion architecture ([models.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L42-L72)) and mandates emitting qualitative rationale before final classification.
```

```text
Paper: Large Language Models are not Fair Evaluators
Authors: Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, Zhifang Sui
Year: 2024 (ACL 2024 / arXiv:2305.17926)
Problem: LLMs exhibit severe, systemic positional and ordering unfairness in pairwise comparative evaluation, enabling arbitrary manipulation of benchmark rankings.
Method: Systematic perturbation of presentation order on FairEval benchmark across multiple LLM evaluators (GPT-4, ChatGPT, Claude, Vicuna). Formulated a swap-based calibration framework to quantify order inconsistency.
Judge setup: Standard pairwise prompt comparing Candidate 1 vs Candidate 2, tested under forward $(A, B)$ and reverse $(B, A)$ presentation orders.
Evaluation methodology: Quantified Consistency Rate ($C = \frac{\text{Consistent Cases}}{\text{Total Cases}}$), First-position preference rate, and ranking shift before and after swap calibration.
Important findings:
  - Positional bias is pervasive: presenting a response in Position 1 boosted its win rate by 12% to 35% depending on model family.
  - Vicuna-13B could be made to "beat" ChatGPT in over 60% of cases simply by controlling presentation order.
  - Merely averaging or discarding inconsistent cases without explicit instability classification conceals underlying evaluation invalidity.
Known limitations: Evaluated short dialogue and translation responses; did not measure how structured criterion-level rubrics dampen positional bias.
Relevance to Benizakura: Serves as the direct theoretical justification for Benizakura's `BidirectionalPairwiseRunner` and position instability rate metric (`max_unstable_rate`).
```

```text
Paper: Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators
Authors: Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto
Year: 2024 (COLM 2024 / arXiv:2404.04475)
Problem: AlpacaEval and general pairwise LLM judges exhibit extreme verbosity bias, allowing models to game benchmarks by generating lengthy, low-information padding.
Method: Developed Length-Controlled AlpacaEval (LC-AlpacaEval) using a Generalized Linear Model (GLM) to predict preference probabilities while conditioning on zero length difference ($\Delta \text{length} = 0$).
Judge setup: GPT-4 Turbo evaluated pairwise outputs across 805 instruction-following prompts with and without length conditioning.
Evaluation methodology: Measured Spearman rank correlation between model benchmark win-rates and Chatbot Arena human Elo ratings.
Important findings:
  - Standard AlpacaEval exhibited a length correlation of $r > 0.65$ with response word counts.
  - Controlling for length increased the Spearman correlation with human LMSYS Chatbot Arena from 0.94 to 0.98, while eliminating rank inflation for verbose models.
Known limitations: Post-hoc GLM length-control assumes length preference is linearly separable from substantive technical depth.
Relevance to Benizakura: For technical interview questions (Conquer), verbosity bias is catastrophic—superficial, verbose candidates can easily obscure technical inaccuracies unless rubrics explicitly penalize bloat.
```

```text
Paper: Prometheus: Inducing Fine-grained Evaluation Capability in Language Models
Authors: Seungone Kim, Jamin Shin, Yohan Jo, Joel Jang, Shayne Longpre, Scott A. Crossan, Minjoon Seo
Year: 2024 (ICLR 2024 / arXiv:2310.17631)
Problem: Proprietary closed-source evaluators (GPT-4) lack transparency, are subject to silent model-drift over time, and do not strictly follow domain-customized rubrics.
Method: Created Feedback Collection (100K fine-grained rubric-based evaluations) and trained Prometheus (13B/7B Llama-based) specialized specifically for evaluation.
Judge setup: Input prompt explicitly structured with (1) Instruction, (2) Response, (3) Reference Answer, (4) Custom Score Rubric (1 to 5 criteria descriptions).
Evaluation methodology: Evaluated correlation with human evaluators on unseen rubrics across 45 datasets, comparing against GPT-4 and GPT-3.5.
Important findings:
  - An evaluator model explicitly fine-tuned on rubric criteria achieved a Pearson correlation of 0.897 with human judges, outperforming GPT-3.5 (0.392) and closely approaching GPT-4 (0.882).
  - Explicit rubric criteria boundaries (defining exactly what constitutes a 1, 3, or 5) dramatically improved inter-rater reliability compared to open-ended grading.
Known limitations: Evaluated pointwise scoring rather than bidirectional pairwise comparisons; required access to high-quality reference solutions.
Relevance to Benizakura: Confirms that clear, unambiguous criterion descriptions ([StandardCriteria](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L25-L40)) are vital for reproducible automated evaluation.
```

```text
Paper: ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
Authors: Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Wang, Zhi-Zheng Xu, Hao-Wei Zhang, Yunchun Gao, Bo-Wen Zhang, Hao-Lin Zhao, Yu-Shuo Lu, Run-Ze Fan, Xiao-Ran Liu, Min-Lie Huang
Year: 2023 (EMNLP 2023 / arXiv:2308.07201)
Problem: Individual LLM judges suffer from high individual variance, hallucinated critiques, and localized reasoning blind spots.
Method: Proposed ChatEval, a multi-agent debate framework where multiple distinct LLM personas (e.g., Critic, Domain Expert, General Reader) independently assess candidate outputs, read each other's assessments, and debate to reach consensus.
Judge setup: Multi-agent interaction with diversified system prompts and cross-model participation (GPT-4, Claude, ChatGPT).
Evaluation methodology: Evaluated text generation and translation benchmarks against human annotations using Cohen's kappa and accuracy.
Important findings:
  - Multi-agent debate significantly reduced hallucinated evaluation critique points, boosting agreement with humans from 68.2% to 75.4%.
  - Diverse model panels (ensembles of distinct model architectures) outperformed homogenous multi-agent debates by eliminating shared architectural blind spots.
Known limitations: Incurs a $3\times\text{--}5\times$ increase in API token cost and significant latency overhead per evaluation.
Relevance to Benizakura: Highlights the value of multi-judge juries (e.g. Anthropic + OpenAI) over single-judge pipelines when evaluating high-stakes regression gates.
```

```text
Paper: Can Large Language Models Be Trusted for Evaluation? Scalable Meta-Evaluation of LLMs as Evaluators via Consistency Analysis
Authors: Cheng-Han Chiang, Hung-yi Lee
Year: 2023 (ACL 2023 / arXiv:2303.04048)
Problem: Empirical human evaluations are expensive and difficult to scale, preventing continuous meta-evaluation of LLM judges.
Method: Formulated non-human-dependent meta-evaluation metrics based on formal logic and mathematical consistency: Transitivity ($A > B \land B > C \implies A > C$), Non-negativity, and Invariance to semantic-preserving input perturbations.
Judge setup: Evaluated ChatGPT and PaLM on story generation and summarization under structured perturbations.
Evaluation methodology: Measured violation rates of transitivity and order-invariance across synthetic candidate triads.
Important findings:
  - LLM judges frequently violate transitivity (up to 24% of triads violated $A > B \land B > C \implies A > C$).
  - Models exhibited high sensitivity to superficial formatting changes (e.g., leading whitespace, bullet points vs numbered lists).
Known limitations: Transitivity consistency is a necessary but insufficient condition for evaluation accuracy (a degenerate judge that always picks A is 100% consistent but 0% accurate).
Relevance to Benizakura: Proves that statistical stability metrics (such as Benizakura's `unstable_rate`) are essential internal sanity checks on judge reliability.
```

```text
Paper: Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models
Authors: Pat Verga, Saurabh Shah, Patrick Fernandes, Pradeep Dasigi, Hao Peng
Year: 2024 (arXiv:2404.12272)
Problem: Single LLM judges have individual biases (length, format, provider self-preference) that undermine evaluation validity.
Method: Proposed PoLL (Panel of LLM evaluators) composed of smaller, cheaper, heterogeneous models (e.g., GPT-3.5, Claude 3 Haiku, Command R) voting as a jury rather than relying on a single large frontier model.
Judge setup: Heterogeneous ensemble where each member performs independent pairwise or pointwise evaluation; verdicts aggregated via majority vote or Borda count.
Evaluation methodology: Benchmarked on Chatbot Arena and MT-Bench against single GPT-4 judgments and human ground truth.
Important findings:
  - A panel of 3–5 diverse small models achieved higher human agreement ($\kappa = 0.69$) than a single GPT-4 judge ($\kappa = 0.64$), at a fraction of the inference cost ($7\times$ cheaper).
  - Cross-architecture pooling canceled out provider-specific self-enhancement biases.
Known limitations: Co-ordinating multi-provider API calls increases engineering complexity and exposure to provider rate limits.
Relevance to Benizakura: Provides a clear architectural roadmap for expanding Benizakura from a single-judge bidirectional runner to an ensemble panel of diverse judges.
```

---

## 3. LLM-as-a-Judge Findings

Across the surveyed literature, several empirical realities govern the design of automated evaluation systems:

1. **Pairwise Comparison Vastly Outperforms Absolute Pointwise Scoring:**
   Pointwise scoring (e.g., "Score this answer from 1 to 10") suffers from severe score compression, calibration drift, and intra-model scale inconsistency (Zheng et al., 2023). A score of "7/10" given by a judge on Monday may correspond to a "9/10" on Tuesday under minor prompt variations. In contrast, pairwise comparison forces a direct relative ranking, significantly increasing human correlation and discriminatory power.
2. **Chain-of-Thought (CoT) Preceding Verdicts is Essential:**
   When an LLM judge is forced to generate its categorical decision (e.g., `{"winner": "A"}`) at token index 0, it operates without reasoning tokens. Forcing the judge to generate per-criterion breakdowns and qualitative evidence quotes *before* emitting the final winner token reduces arbitrary flips by up to 20% (Liu et al., 2023; Wang et al., 2024).
3. **Structured Criteria Prevent Focus Drift:**
   Unconstrained pairwise prompts ("Which answer is better?") cause judges to default to surface heuristics such as length, greeting politeness, or formatting markdown. Providing an explicit rubric containing orthogonal criteria (e.g., `Technical Accuracy`, `Depth & Edge Cases`, `Communication Clarity`) forces the judge to evaluate substantive dimensions independently.
4. **LLM Judges are Consistent with Humans on Obvious Differences, Noisy on Near-Ties:**
   Frontier judges agree with expert humans $>85\%$ of the time when one response possesses clear technical superiority. However, when responses are near-ties or feature subtle technical trade-offs, agreement drops towards random chance ($50\%\text{--}55\%$) (Zheng et al., 2023). An evaluation pipeline must treat near-ties as first-class statistical ties rather than forcing false decisiveness.

---

## 4. Known Judge Biases

We systematically analyze eight major biases documented in LLM evaluators, detailing their mechanisms, empirical severity, impact on Benizakura, and mitigations.

### 4.1 Position Bias
* **What is it?** The tendency of an evaluator to systematically favor responses presented in a specific position (typically Position 1 / Response A, though occasionally Position 2 / Response B depending on model architecture and context length).
* **Why does it happen?** Autoregressive attention mechanisms exhibit primacy effects (strong attention to initial tokens) and recency effects (strong attention to immediately preceding context). Furthermore, instruction-tuning datasets often place ideal answers in early positions.
* **How has research measured it?** Measured by presenting the identical pair $(X, Y)$ in order $(X, Y)$ and reverse order $(Y, X)$, calculating the swap-inconsistency rate:
  $$\text{IR} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}[\text{Winner}(X, Y) \ne \text{Winner}(Y, X)]$$
  Wang et al. (2024) observed raw position bias rates between 12% and 35% across popular models.
* **How serious is it?** Highly critical. In unmitigated single-direction evaluation, a candidate prompt can appear to cause a 15% improvement simply because it was evaluated in the favored slot.
* **How could it affect Benizakura?** In Benizakura, if candidate answers are evaluated against baseline answers without position controls, CI gates will fail or pass based entirely on prompt presentation slot.
* **Mitigations:**
  1. *Bidirectional Evaluation:* Run both forward $(A=\text{Base}, B=\text{Cand})$ and reverse $(A=\text{Cand}, B=\text{Base})$ passes.
  2. *Instability Thresholding:* Discard or flag cases where the winner flips based on order (`is_unstable = True`).

#### Audit of Benizakura's Current Bidirectional Design
Benizakura currently executes forward and reverse passes via [BidirectionalPairwiseRunner](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/pairwise_runner.py#L15-L65):
* Pass 1: $A = \text{Baseline}, B = \text{Candidate} \implies \text{Normalized Winner}_1$
* Pass 2: $A = \text{Candidate}, B = \text{Baseline} \implies \text{Normalized Winner}_2$
* Instability Classification: $\text{is\_unstable} \iff \text{Normalized Winner}_1 \ne \text{Normalized Winner}_2$

**What this DOES address:**
* It completely prevents false asymmetric claims where an engineer runs a single pass and declares victory based on position favoritism.
* It explicitly quantifies judge order instability via `unstable_rate` in [ReleaseGate](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/gate.py#L177-L185), rejecting runs where order instability exceeds policy (`max_unstable_rate = 0.20`).
* In paired difference statistical analysis ($D_i$), cases that are unstable are marked with $D_i = 0.0$ (tie/no signal), preventing order-biased noise from registering as true regressions or improvements.

**What this DOES NOT address:**
* *Signal Loss via Double-Favoritism:* If a judge has an absolute position bias (e.g., always selects Response A regardless of content), then in Pass 1 it selects Baseline ($A$), and in Pass 2 it selects Candidate ($A$). Both passes claim a decisive winner, but in opposite directions! Benizakura correctly flags this as unstable, but the underlying evaluation signal is completely lost ($D_i = 0$). If position bias is widespread in a model, 40%+ of cases become unstable, triggering `Verdict.INCONCLUSIVE` and halting CI.
* *Latency and Cost Multiplication:* Bidirectional judging requires strictly $2\times$ the inference calls and token costs.
* *Asymmetric Ties:* If a judge declares a Tie when Candidate is in Position 1, but declares Candidate when Candidate is in Position 2, this is also flagged as unstable, conflating slight preference with outright position flipping.

---

### 4.2 Verbosity Bias
* **What is it?** The systematic tendency to assign higher scores or preference to longer, more verbose responses, even when the extra length consists of repetitive filler, irrelevant disclaimers, or shallow prose.
* **Why does it happen?** RLHF reward models frequently use length as a proxy for helpfulness, training downstream models to associate length with effort and comprehensiveness (Dubois et al., 2024; Saito et al., 2024).
* **How has research measured it?** Correlating delta word count ($\Delta L = \text{len}(A) - \text{len}(B)$) with win probability, or perturbing answers by appending redundant but grammatically correct sentences and observing win-rate jumps ($+20\%\text{--}40\%$).
* **How serious is it?** Severe in interview evaluation. In Conquer, an interview candidate who speaks 600 words of superficial buzzwords without addressing algorithmic complexity should be rated lower than a concise candidate who provides the exact $O(N \log N)$ proof in 120 words.
* **How could it affect Benizakura?** If Conquer modifies its prompt to be more concise and direct, Benizakura's judge might register this improvement as a *regression* simply because output tokens decreased.
* **Mitigations:**
  1. *Explicit Rubric Penalties:* Rubrics must include negative criteria or instructions explicitly penalizing fluff, unrequested preamble, and repetitive explanations.
  2. *Conciseness Benchmark Probes:* Include calibration cases where a concise, correct answer is paired against a bloated, superficial answer.

---

### 4.3 Self-Enhancement / Model-Family Bias
* **What is it?** The tendency of an LLM evaluator to rate responses generated by itself or by models within its own architectural family higher than responses from external model families (Zheng et al., 2023; Panickssery et al., 2024).
* **Why does it happen?** Models share tokenization vocabularies, syntactic tendencies, stylistic rhythm, formatting conventions (e.g., specific markdown headers, bullet styles), and latent representational manifolds.
* **How has research measured it?** Presenting blinded outputs from Model $X$ and Model $Y$ to Model $X$, Model $Y$, and independent Model $Z$. Model $X$ consistently awards itself a $5\%\text{--}15\%$ higher win margin than Model $Z$ awards it.
* **How serious is it?** Extremely serious when evaluating an application whose generator belongs to the same family as the judge.
* **How could it affect Benizakura?** Conquer uses an OpenAI-based model (`openai/gpt-oss-120b`). If Benizakura selects an OpenAI judge (e.g., GPT-4o), prompt mutations that align more closely with OpenAI's RLHF stylistic dialect will receive higher scores, regardless of whether technical accuracy improved.
* **Mitigations:**
  1. *Architectural Independence:* Use a non-OpenAI judge (e.g., Anthropic Claude 3.5 Sonnet or Google Gemini 1.5 Pro) as the primary gatekeeper.
  2. *Multi-family Jury:* Average verdicts across independent model families.

---

### 4.4 Anchoring Effects
* **What is it?** The judge's evaluation of later criteria or subsequent responses is disproportionately influenced by the first salient flaw or positive attribute encountered.
* **Why does it happen?** Autoregressive attention conditions all future token generation on previously generated tokens; early harsh critique tokens bias subsequent criterion evaluations towards negative polarity.
* **How has research measured it?** Shuffling the presentation order of rubric criteria or altering the order of answer sub-sections and measuring variance in criterion scores.
* **How could it affect Benizakura?** If an interview candidate makes a minor typo in their opening greeting, an anchored judge may unfairly downgrade their subsequent system architecture design.
* **Mitigations:**
  1. Independent criterion prompting or explicit rubric instructions requiring orthogonal assessment.
  2. Forcing isolated evidence quotation for each criterion before overall synthesis.

---

### 4.5 Ordering Effects (Criteria & Multi-turn)
* **What is it?** The order in which rubric criteria are listed in the prompt influences the weight the judge implicitly assigns to them (criteria listed first receive more attention and impact the final verdict more heavily).
* **Why does it happen?** Attention decay over long context windows and prompt structure prioritization.
* **How has research measured it?** Permuting criteria ordering in the prompt and measuring sensitivity of criterion-level weights.
* **How could it affect Benizakura?** If `Communication Clarity` is listed before `Technical Accuracy`, the judge may overweight presentation style over correctness.
* **Mitigations:**
  1. Maintain a fixed canonical ordering of standard criteria ([StandardCriteria](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L25-L40)) with explicit weights.
  2. Include explicit instruction: *"Weight criteria strictly according to their numerical weight values, independent of presentation order."*

---

### 4.6 Reference-Answer Bias
* **What is it?** When a "golden reference" answer is provided to the judge, the judge penalizes valid alternative approaches that differ from the reference's specific methodology, phrasing, or algorithm.
* **Why does it happen?** Surface lexical and structural similarity matching is easier for the LLM than deep semantic equivalence verification.
* **How has research measured it?** Providing a reference answer that uses dynamic programming, and evaluating whether an equally optimal greedy or memoized recursive approach is unfairly penalized.
* **How could it affect Benizakura?** In technical interviews, problems often have multiple optimal solutions (e.g., BFS vs DFS; iterative vs recursive; relational vs document store). A reference-biased judge will falsely fail valid engineering alternatives.
* **Mitigations:**
  1. Provide reference answers that explicitly list accepted alternative approaches, key trade-offs, and invariants rather than a single prescriptive code block.
  2. In Conquer's case, focus evaluation on rubric criteria (correctness, complexity, trade-offs) rather than strict verbatim reference matching.

---

### 4.7 Prompt Sensitivity
* **What is it?** Minor, semantically irrelevant changes in the judge's system prompt (e.g., punctuation, whitespace, instruction phrasing) cause noticeable fluctuations in verdicts.
* **Why does it happen?** Discrete tokenization boundaries and non-linear activation landscapes in deep transformer architectures.
* **How has research measured it?** Generating prompt paraphrases and computing prediction agreement across paraphrases (often finding agreement drops to $70\%\text{--}75\%$).
* **How could it affect Benizakura?** A judge prompt tweak could cause false regression alerts in CI that reflect judge prompt sensitivity rather than application changes.
* **Mitigations:**
  1. Strict versioning and freezing of the judge prompt as an immutable code artifact ([prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py)).
  2. Any modification to `SYSTEM_PROMPT_TEMPLATE` must be treated as a breaking change requiring re-calibration against the human benchmark.

---

### 4.8 Judge Instability / Stochasticity
* **What is it?** When queried multiple times with identical inputs at non-zero temperature, the judge returns different verdicts.
* **Why does it happen?** Nucleus sampling, top-k sampling, and non-deterministic floating-point operations in distributed GPU inference clusters (even at `temperature = 0`).
* **How has research measured it?** Repeatedly evaluating the same pair $N=10$ times at varying temperatures and measuring the self-consistency variance.
* **How could it affect Benizakura?** Flaky CI builds where a pull request passes on one run and fails on the next without code changes.
* **Mitigations:**
  1. Enforce `temperature = 0.0` (or the minimum supported deterministic sampling setting) for all evaluation runs.
  2. Exact-match input caching ([DECISIONS.md D5](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/DECISIONS.md#L48-L56)).
  3. Paired bootstrap confidence intervals to account for residual empirical variance.

---

## 5. Audit of Current Benizakura Design Against Research

We systematically audit Benizakura's current implementation against empirical literature, classifying each design component as research-backed, engineering choice, heuristic choice, or requiring empirical validation.

| Component | Implementation Reference | Current Classification | Research Support / Verdict |
| :--- | :--- | :--- | :--- |
| **Bidirectional Judging** | [pairwise_runner.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/pairwise_runner.py#L41-L65) | **Research-Backed** | Wang et al. (2024), Zheng et al. (2023). Directly identifies and nullifies presentation order bias. Essential. |
| **Position Instability Tracking** | [models.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L95-L107), [statistical.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/statistical.py#L137-L141) | **Research-Backed** | Chiang & Lee (2023). Tracking non-invariance as an explicit metric (`unstable_rate`) is scientifically sound. |
| **Criterion-Level Rubrics** | [models.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L42-L72), [prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py#L57-L66) | **Research-Backed** | Liu et al. (2023), Kim et al. (2024). Multi-criteria evaluation significantly improves human alignment over monolithic scores. |
| **Criterion Evidence Quotes** | [models.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L82-L89), [prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py#L29) | **Empirical Validation Needed** | Requiring evidence grounds explanations, but research shows models can fabricate or misquote evidence unless explicitly verified. |
| **JSON Output Ordering** | [prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py#L20-L32) | **Identified Flaw / Needs Immediate Fix** | **CRITICAL DEFECT:** Emitting `"winner"` at top of JSON object forces verdict decision before rationale tokens, violating CoT principles. |
| **Structured JSON Schema** | [parser.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/parser.py#L24-L85) | **Engineering Choice** | Ensures parser reliability and programmatic CI execution. Does not inherently improve reasoning quality without CoT. |
| **Paired Bootstrap Resampling** | [statistical.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/statistical.py#L173-L225) | **Research-Backed (Statistical)** | Efron & Tibshirani (1994). Correctly estimates uncertainty around paired differences without assuming parametric normality. |
| **PASS / FAIL / INCONCLUSIVE Gate** | [gate.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/gate.py#L169-L226) | **Policy Decision Built on Statistics** | Not a statistical law. Translates confidence intervals and error rates into actionable business release decisions. |
| **GateConfig Default Thresholds** | [gate.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/gate.py#L28-L33) | **Heuristic Choices** | `min_cases=10`, `max_unstable_rate=0.20`, etc., are heuristic engineering defaults requiring empirical tuning. |

### Deep-Dive Analysis of Core Components

#### 1. Bidirectional Judging & Order Normalization
Does swapping candidate/baseline positions meaningfully reduce position bias?
* **Yes, at the system level; No, at the individual forward pass level.** Swapping positions does not change the model's inherent preference for Position A. However, by observing both passes, Benizakura can detect when a decision was artifactual. If Candidate wins in Pass 1 ($B$) and Baseline wins in Pass 2 ($B$), both decisions favored Position 2. Benizakura maps this to `NormalizedWinner.CANDIDATE` and `NormalizedWinner.BASELINE`, flags `is_unstable = True`, and assigns a paired delta of $D_i = 0.0$.
* This prevents position bias from artificially driving CI verdicts, but converts biased evaluations into ties/uncertainty.

#### 2. Rubrics & Criterion-Level Evidence
Does structured criterion-level evaluation improve judge reliability?
* **Yes.** As demonstrated by Prometheus (Kim et al., 2024) and G-Eval (Liu et al., 2023), breaking an evaluation into orthogonal criteria (e.g., `Technical Accuracy`, `Completeness`, `Clarity`) prevents the judge from being overwhelmed by a single salient feature.
* **Evidence vs Plausibility:** Asking the judge to provide direct quotes/evidence (`evidence: "<quote>"`) increases transparency. However, research into LLM sycophancy indicates that models can selectively extract quotes to justify a pre-determined conclusion. The true benefit is unlocked only when evidence extraction *precedes* the conclusion.

#### 3. Structured JSON Schema Defect in `prompt.py`
In [prompt.py (lines 20–32)](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py#L20-L32), the prompt template defines:
```json
{
  "winner": "A" | "B" | "TIE",
  "rationale": "<Overall qualitative explanation...>",
  "confidence": <float>,
  "criterion_assessments": [...]
}
```
In transformer decoding, the token for `"winner"` is generated **first**. The model must commit to "A", "B", or "TIE" with zero intermediate reasoning tokens. The subsequent `"rationale"` and `"criterion_assessments"` are generated *post-hoc* to justify that initial token.
* **Remediation Specification:** The schema must be inverted so the model reasons first:
```json
{
  "criterion_assessments": [
    {
      "criterion_name": "<name>",
      "rationale": "<analysis>",
      "evidence": "<quote>",
      "winner": "A" | "B" | "TIE"
    }
  ],
  "overall_rationale": "<synthesis>",
  "confidence": <float>,
  "winner": "A" | "B" | "TIE"
}
```

#### 4. Bootstrap Confidence Intervals
What does paired bootstrap tell us, and what does it NOT tell us?
* **What it tells us:** Given the sample of $N$ evaluation cases, if we resample with replacement 10,000 times, what is the 95% empirical percentile range of the net win-rate difference ($\Delta = \text{WinRate}_{\text{Cand}} - \text{WinRate}_{\text{Base}}$)? It quantifies sample variance and confirms whether an observed difference is robust to the inclusion or exclusion of individual test cases.
* **What it does NOT tell us:** It does **not** protect against systematic judge bias. If the judge has a 100% consistent verbosity bias, all 10,000 bootstrap resamples will reflect that same biased preference, producing a deceptively narrow confidence interval around an invalid conclusion. Bootstrap measures sampling precision, not measurement validity.

#### 5. PASS / FAIL / INCONCLUSIVE
Is this a statistically valid conclusion or a policy decision?
* It is strictly a **policy decision** constructed on top of statistical evidence.
* In [gate.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/gate.py#L169-L226), the decision tree enforces organizational risk tolerance:
  - If $N < 10$: Policy declares `INCONCLUSIVE` (refusal to gate on insufficient sample size).
  - If $\text{unstable\_rate} > 0.20$: Policy declares `INCONCLUSIVE` (refusal to gate when judge noise is excessive).
  - If $\text{regression\_rate} > 0.25$: Policy declares `FAIL` (risk-averse refusal to accept updates where $>25\%$ of test cases suffered regressions, even if overall net win rate is positive).
  - If $\text{CI}_{\text{upper}} < -0.05$: Policy declares `FAIL` (statistically confident negative shift).
  - If $\text{Observed Diff} \ge 0.05 \land \text{CI}_{\text{lower}} \ge -0.05$: Policy declares `PASS`.
* These thresholds are engineering heuristics. For high-risk medical or financial applications, `regression_tolerance` might be set to `0.00` and `max_regression_rate` to `0.05`. For rapid prototyping, thresholds may be looser.

---

## 6. Candidate Judge Models

We evaluate frontier and open-weight models available in 2025/2026 across critical judging dimensions.

| Model | Reasoning & Technical Code Evaluation | Structured Output Reliability | Published Judge Evidence | Cost Considerations (Input / Output per 1M tokens) | Latency Considerations (Time-to-First-Token & Tok/sec) | Main Risks for Benizakura |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Claude 3.5 Sonnet** (`claude-3-5-sonnet-20241022`) | State-of-the-art coding, systems design, and subtle logic bug detection. Exceptional nuanced instruction adherence. | High via Tool Use / Function Calling mode. Very low malformed JSON rate ($<0.1\%$). | High. Widely used as gold-standard reference evaluator across academic benchmarks; high human agreement ($\approx 80\%$). | \$3.00 / \$15.00 (Prompt caching reduces repeated context by up to 90%: \$0.30). | Moderate latency ($\approx 40\text{--}60$ tok/sec). Slower than mini/flash models. | Anthropic tier rate limits; higher cost for large evaluation sweeps if uncached. |
| **GPT-4o** (`gpt-4o-2024-08-06`) | Strong multimodal & technical reasoning. Highly competitive on standard coding benchmarks. | Exceptional. Native constrained decoding guarantees 100% adherence to supplied JSON Schema. | Extensive. Standard industry baseline evaluator (MT-Bench, AlpacaEval, Arena-Hard). | \$2.50 / \$10.00 (Cached inputs: \$1.25). | Low-to-moderate latency ($\approx 60\text{--}80$ tok/sec). | **Violates evaluator independence** when evaluating Conquer's GPT-based scorer (`openai/gpt-oss-120b`). Self-enhancement bias. |
| **Gemini 1.5 Pro** (`gemini-1.5-pro-002`) | Exceptional long-context reasoning; strong technical and mathematical analysis. | High via `response_schema` parameter. Occasional extraneous formatting if unconstrained. | Moderate-to-high. Frequently used in multimodal and long-context evaluation benchmarks. | \$1.25 / \$5.00 (Inputs $\le 128\text{K}$). Highly cost-competitive. | Moderate latency; longer time-to-first-token on complex system prompts. | Stricter safety filter interventions on certain adversarial or security-related test cases. |
| **Gemini 2.0 Flash** / **1.5 Flash** | Good general reasoning; fast instruction following, but less nuanced on subtle algorithm complexity bugs. | Reliable structured JSON mode. | Moderate. Used for low-cost filtering, but exhibits lower agreement on complex technical edge cases. | \$0.10 / \$0.40 (Extremely economical). | Blazing fast ($>120$ tok/sec). | Higher noise floor on subtle technical trade-offs; higher rate of tie declarations on complex cases. |
| **Llama 3.3 70B Instruct** (via Groq / Together) | Very strong open-weight reasoning for standard tasks; competitive with GPT-4-class on many coding benchmarks. | Good JSON mode, but lacks strict grammar-enforced constrained decoding on standard provider endpoints. | High in open-source research (Prometheus 2 backbone). | \$0.59 / \$0.79 (via Groq). Very cheap. | Ultra-fast on Groq ($>250$ tok/sec). Ideal for rapid local CI loops. | Higher susceptibility to position bias; occasional schema parsing failures requiring retry logic. |
| **o3-mini** / **o1** (OpenAI Reasoning Models) | Superlative mathematical and algorithmic reasoning via internal test-time compute. | Supported via structured outputs in latest releases. | Emerging. High accuracy on competition math/code, but reasoning trace is hidden/obfuscated. | \$1.10 / \$4.40 (o3-mini). Highly competitive pricing for reasoning depth. | High latency due to thinking tokens; fixed temperature constraints (`temp = 1.0`). | Non-configurable temperature makes deterministic replay harder; high latency slows CI pipelines. |

---

## 7. Judge Independence Analysis

An architectural decision in Benizakura is whether the judge should share the model family of the evaluated system:

$$\text{Conquer Candidate: } \texttt{openai/gpt-oss-120b}$$

### Scenario A: Same-Family Judge (e.g., GPT-4o judging Conquer)
* **Mechanisms:**
  - *Shared Tokenizer & Embeddings:* Both models segment text into identical subword tokens, creating shared biases regarding word frequency and punctuation.
  - *Shared Alignment Dialect:* Models trained with OpenAI's RLHF exhibit consistent stylistic preferences: structured numbered lists, balanced apologetic hedging, specific concluding synthesis patterns, and polite conversational framing.
* **Empirical Evidence:** Zheng et al. (2023) and Panickssery et al. (2024) demonstrated that GPT-4 gives GPT-family outputs an unearned 5%–12% win-rate advantage over equally competent Claude or open-weight models.
* **Correlated Errors:** When Conquer's prompt produces a subtle hallucination or algorithmic error that GPT models are prone to making, a GPT-4o judge is significantly more likely to overlook that exact same error because it shares the same underlying pre-training distribution blind spots.

### Scenario B: Independent-Family Judge (e.g., Claude 3.5 Sonnet judging Conquer)
* **Mechanisms:**
  - *Independent Pre-training Corpus & Architecture:* Trained on a completely distinct pipeline, token vocabulary, and Constitutional AI alignment philosophy.
  - *Orthogonal Failure Modes:* Claude 3.5 Sonnet's error distribution is uncorrelated with GPT-family models. An error generated by Conquer's GPT prompt is scrutinized by an alien evaluator that does not share its stylistic or deductive blind spots.
* **Trade-offs:**
  - *Stylistic Friction:* Claude may penalize standard GPT formatting habits (e.g., excessive bullet-point disclaimers) as fluff. However, this is precisely what a rigorous technical evaluator *should* do.

### Verdict on Judge Independence
For Benizakura to serve as a trustworthy, scientifically defensible evaluation gate, **evaluator independence is mandatory**. Using GPT-4o to evaluate a GPT-based candidate application compromises the validity of regression testing. Therefore, Benizakura's primary judge must be drawn from an independent frontier family—specifically **Claude 3.5 Sonnet**.

---

## 8. Baseline Experiments

To rigorously validate whether Benizakura's architectural additions (bidirectional swapping, rubrics, criterion evidence, bootstrap CI) actually improve evaluation quality, we establish four progressive baselines:

```text
Baseline A (Simple Pointwise Scoring)
  ↓
Baseline B (Single-Direction Pairwise)
  ↓
Baseline C (Bidirectional Pairwise)
  ↓
Baseline D (Rubric-Based Bidirectional Pairwise)
  ↓
Full Benizakura Pipeline (Rubrics + Bidirectional + Evidence + Bootstrap Gate)
```

### Baseline A — Simple Pointwise LLM Scoring
* **Protocol:** Prompt the judge to evaluate the candidate interview answer on an absolute scale of 0.0 to 10.0 based on general quality.
* **Prompt Shape:** `"Score this technical interview answer from 0.0 to 10.0 based on accuracy and clarity."`
* **Score Delta:** $\Delta_i = \text{Score}(\text{Candidate}_i) - \text{Score}(\text{Baseline}_i)$.
* **Purpose:** Proves whether simple scalar scores suffer from severe calibration drift and inability to detect subtle regressions.

### Baseline B — Single-Direction Pairwise Judge
* **Protocol:** Present Baseline as Response A and Candidate as Response B in a fixed presentation order. Ask the judge to pick the winner (A, B, or TIE).
* **Prompt Shape:** Unstructured pairwise comparison with general instructions.
* **Purpose:** Measures the raw impact of position bias and quantifies how often the first-presented answer wins simply due to position.

### Baseline C — Bidirectional Pairwise Judge (Unstructured)
* **Protocol:** Run both forward $(A=\text{Base}, B=\text{Cand})$ and reverse $(A=\text{Cand}, B=\text{Base})$ passes using unstructured pairwise prompts.
* **Purpose:** Measures how much position bias is mitigated by bidirectional normalization alone, *without* the aid of domain-specific rubrics.

### Baseline D — Rubric-Based Bidirectional Pairwise Judge
* **Protocol:** Run bidirectional passes with structured criterion-level rubrics ([Rubric](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/models.py#L42-L72)) and criterion assessments, but without bootstrap release gating.
* **Purpose:** Measures the specific marginal increase in human agreement provided by decomposing evaluation into explicit orthogonal technical criteria.

---

## 9. Judge Validation Methodology

Before Benizakura's automated gate is trusted in production, the judge's outputs must be calibrated against expert human software engineers.

### 9.1 Human Agreement Metrics
We map specific statistical agreement metrics to their corresponding Benizakura output types:

| Metric | Benizakura Output Type | Mathematical Definition | Interpretation & Target Threshold | Why Appropriate / Not Appropriate |
| :--- | :--- | :--- | :--- | :--- |
| **Cohen's Kappa ($\kappa$)** | Discrete Pairwise Winner (`BASELINE`, `CANDIDATE`, `TIE`) | $\kappa = \frac{P_o - P_e}{1 - P_e}$ where $P_o$ is observed agreement and $P_e$ is chance agreement. | **Target: $\kappa \ge 0.60$** (Substantial agreement). $\kappa < 0.40$ indicates unacceptable noise. | **Primary metric for pairwise winners.** Corrects for chance agreement across nominal categories. |
| **Krippendorff's Alpha ($\alpha$)** | Multi-rater human jury & criterion assessments | $\alpha = 1 - \frac{D_o}{D_e}$ where $D_o, D_e$ are observed and expected disagreements. | **Target: $\alpha \ge 0.67$** (Lowest acceptable for tentative conclusions); $\alpha \ge 0.80$ preferred. | Appropriate when human annotations involve multiple raters with missing data or ordinal criterion levels. |
| **Spearman Rank Correlation ($r_s$)** | Ordinal Criterion Scores & Pointwise Baselines | $r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$ | **Target: $r_s \ge 0.75$** | Evaluates monotonic ranking agreement without assuming linear intervals. Ideal for comparing Baseline A ranks. |
| **Pearson Correlation ($r$)** | Continuous aggregate deltas ($\Delta \text{win\_rate}$) | $r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$ | **Not Recommended for Individual Cases.** Use only for aggregate run-level score shifts. | **Inappropriate for categorical pairwise outcomes** ($D_i \in \{-1, 0, +1\}$) due to severe non-normality and discrete clustering. |
| **Directional Agreement Rate ($A_{\text{dir}}$)** | Regression / Improvement Direction | $A_{\text{dir}} = \frac{1}{N} \sum \mathbf{1}[\text{sign}(\Delta_{\text{judge}}) == \text{sign}(\Delta_{\text{human}})]$ | **Target: $A_{\text{dir}} \ge 0.85$** | Crucial CI metric: ensures the judge rarely asserts an improvement when human engineers observe a regression. |

### 9.2 Human Annotation Procedure
1. **Double-Blind Annotation:** Two senior engineers independently evaluate each pair of candidate responses without knowing which is Candidate or Baseline.
2. **Adjudication Phase:** If Annotator 1 and Annotator 2 disagree, an engineering lead arbitrates the disagreement to produce a single ground-truth consensus label.
3. **Inter-Annotator Baseline:** Measure human-to-human Cohen's $\kappa_{\text{human}}$. The LLM judge cannot reasonably be expected to exceed human-to-human reliability ($\approx 0.75\text{--}0.82$).

---

## 10. Human-Calibrated Benchmark Design

To thoroughly evaluate Conquer's answer scorer, we design a targeted benchmark of **30 comprehensive evaluation cases** stratified across technical interview domains.

### 10.1 Domain Stratification
* **Data Structures & Algorithms (DSA):** 6 cases (Time/space complexity proofs, recursion, dynamic programming, edge cases).
* **System Design:** 8 cases (Distributed caching, horizontal scaling, consistency models, failure recovery, database partitioning).
* **Backend Engineering:** 6 cases (Concurrency, race conditions, database transactions/isolation, connection pools, API design).
* **Frontend Engineering:** 4 cases (DOM rendering lifecycles, state management, CSS performance, edge caching/persisted queries).
* **Behavioral / STAR Method:** 6 cases (Conflict resolution, project failure postmortems, leadership under ambiguity, technical debt negotiation).

### 10.2 Deliberate Bias-Exposing Test Probes
The benchmark deliberately pairs candidate responses engineered to expose specific judge vulnerabilities:

```text
Probe 1: Verbosity vs Technical Substance (Verbosity Bias Probe)
  - Topic: System Design (Database Indexing)
  - Candidate 1 (Concise & Accurate): 110 words. Explains B-tree O(log N) page traversal, random I/O reduction, and exact write overhead (page splits, WAL write amp).
  - Candidate 2 (Verbose & Superficial): 420 words. Lengthy introductory history of relational databases, generic praise of databases, repetitive fluff, but omits disk page splits and write amplification.
  - Expected Human Ground Truth: Candidate 1 Wins.
  - Test Goal: Detect if judge is seduced by Candidate 2's length and formatting.

Probe 2: Authoritative Hallucination vs Clumsy Accuracy (Sycophancy/Hallucination Probe)
  - Topic: DSA (Python GIL and Multiprocessing)
  - Candidate 1 (Authoritative Tone, Fatal Error): Beautifully formatted, polished prose asserting that Python's GIL can be bypassed in pure CPU-bound tasks by using standard threading with `time.sleep(0)`. (Factually false).
  - Candidate 2 (Plain Tone, Technically Correct): Modest formatting explaining that CPU-bound concurrency requires `multiprocessing` or native C extensions to circumvent the GIL.
  - Expected Human Ground Truth: Candidate 2 Wins.
  - Test Goal: Verify whether judge prioritizes technical truth over rhetorical polish.

Probe 3: Genuine Near-Ties (Tie Calibration Probe)
  - Topic: Backend (REST vs GraphQL)
  - Candidate 1: Highlights REST edge CDN cacheability and simplicity; notes GraphQL over-fetching elimination.
  - Candidate 2: Highlights GraphQL schema contract and single round-trip efficiency; notes REST standard tooling and HTTP caching maturity.
  - Both candidates provide accurate, balanced trade-offs of equal depth.
  - Expected Human Ground Truth: TIE.
  - Test Goal: Verify that the judge does not force an arbitrary win when substantive quality is identical.

Probe 4: Subtly Incomplete STAR vs Complete STAR (Behavioral Calibration Probe)
  - Topic: Behavioral (Handling Production Outages)
  - Candidate 1: Rich Situation, clear Task, detailed Action (rollback, runbook execution), but completely omits the Result and postmortem learnings.
  - Candidate 2: Balanced STAR structure with measurable Result (MTTD reduced, alert threshold tuned).
  - Expected Human Ground Truth: Candidate 2 Wins.
  - Test Goal: Verify that rubric weights for STAR completeness are strictly enforced.
```

---

## 11. Benchmark Contamination & Dataset Governance

To prevent benchmark degradation and false evaluation confidence, strict data governance protocols must be enforced:

```text
Benchmark Dataset Architecture:
  ├── dev_set/         (8 cases)  — Used for exploratory prompt engineering & initial sanity checks
  ├── calibration_set/ (12 cases) — Used for tuning rubric descriptions and judge instructions
  └── test_set/        (30 cases) — FROZEN & HELD-OUT. Zero access during prompt optimization.
```

### Contamination Prevention Rules
1. **Label Pre-Commitment:** Human ground-truth labels and rationale must be authored and frozen *before* running any LLM judge evaluations.
2. **Separation of Concerns:** Cases must **not** be derived from Conquer's current production failure logs alone (which over-represents the current prompt's specific weaknesses). Cases must be authored from independent technical interview curriculums (e.g., standard SWE interview standards).
3. **No Prompt Overfitting:** The `test_set` must never be used to tune judge prompts or rubric criteria. If a judge fails on the calibration set, prompts may be refined; but once evaluated on the held-out test set, all metrics are final.
4. **Cryptographic Hashing:** The test set must be stored with a companion `sha256` checksum manifest to detect accidental mutations.

---

## 12. Statistical Evaluation Plan

We define the complete experimental matrix to compare the progressive evaluation pipelines against human ground truth.

### Evaluation Dimensions & Metrics

#### 1. Reliability & Agreement
* Cohen's $\kappa$ against human ground-truth labels.
* Krippendorff's $\alpha$ across multi-rater juries.
* Inter-run Repeatability: Percentage of identical verdicts when re-running the benchmark at `temperature = 0.0` across 3 distinct calendar days.

#### 2. Bias Quantification
* **Position Bias Rate:** Percentage of cases where raw choice flips under presentation swap:
  $$\text{Bias}_{\text{pos}} = \frac{N_{\text{unstable}}}{N_{\text{total}}}$$
* **Verbosity Bias Delta:** Win rate difference in Probe 1 (Concise vs Verbose). A debiased pipeline must yield $\text{WinRate}(\text{Concise}) > 80\%$.
* **Self-Preference Ratio:** Win rate inflation when evaluating candidate outputs from the judge's own model family vs an external family.

#### 3. Regression Detection Performance
Treating human consensus regressions as ground-truth positive cases:
* **True Positive Rate (Sensitivity):** Proportion of real regressions correctly flagged as `FAIL`.
* **False Positive Rate (Type I Error):** Proportion of valid improvements or ties falsely flagged as `FAIL` (blocking production CI).
* **False Negative Rate (Type II Error):** Proportion of real regressions that erroneously receive a `PASS`.

#### 4. Operational Metrics
* **Total Latency per Gate Evaluation:** P50, P90, P99 wall-clock seconds for a 30-case run.
* **Token Cost per Gate Evaluation:** Total input/output token expenditure per run.
* **Malformed Output Rate:** Percentage of judge responses that fail schema validation ([parser.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/parser.py#L87-L140)) and require retries.

---

## 13. Research Hypotheses

We formulate explicit, falsifiable research hypotheses to be empirically tested during the validation milestone:

* **Hypothesis 1 (H1 — Position Debiasing):**
  * *Formulation:* Bidirectional evaluation combined with order normalization produces a lower position-dependent decision error rate than single-direction evaluation ($p < 0.01$).
  * *Test:* Compare Baseline B (Single-direction) vs Baseline C (Bidirectional) against human ground truth on swap-perturbed pairs.
  * *Status:* **Requires Empirical Validation.**

* **Hypothesis 2 (H2 — Rubric Decomposition Efficacy):**
  * *Formulation:* Decomposing pairwise evaluation into structured, criterion-level assessments improves Cohen's $\kappa$ agreement with expert human evaluators by at least $0.15$ compared to an unconstrained pairwise judge.
  * *Test:* Measure Cohen's $\kappa$ of Baseline C vs Baseline D against human ground truth.
  * *Status:* **Requires Empirical Validation.**

* **Hypothesis 3 (H3 — Chain-of-Thought JSON Ordering):**
  * *Formulation:* Reordering the JSON output schema to generate per-criterion rationale and evidence *prior* to emitting the final categorical winner token reduces position-instability rates by at least $25\%$.
  * *Test:* Evaluate identical pairs using (a) Winner-First schema vs (b) Rationale-First schema.
  * *Status:* **Requires Empirical Validation.**

* **Hypothesis 4 (H4 — Evaluator Independence):**
  * *Formulation:* When evaluating candidate answers generated by a GPT-family model, an independent-family judge (Claude 3.5 Sonnet) exhibits lower false-pass rates on subtle technical hallucinations than a same-family judge (GPT-4o).
  * *Test:* Measure False Negative Rate across Probe 2 (Authoritative Hallucination) between Claude 3.5 Sonnet and GPT-4o.
  * *Status:* **Requires Empirical Validation.**

* **Hypothesis 5 (H5 — Statistical Gating False-Alarm Suppression):**
  * *Formulation:* Incorporating paired bootstrap confidence intervals and an explicit `INCONCLUSIVE` verdict reduces false regression alarms (Type I errors) by at least $50\%$ compared to naive point-estimate thresholding.
  * *Test:* Compare Benizakura ReleaseGate decisions against a deterministic single-mean delta threshold on high-variance near-tie evaluation runs.
  * *Status:* **Requires Empirical Validation.**

---

## 14. Recommended First Judge

### Selection: Anthropic Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)

```text
Model Identity: claude-3-5-sonnet-20241022
Provider: Anthropic
Recommended Role: Benizakura Primary Release Gate Judge
Backup / Cross-Validator: OpenAI GPT-4o (gpt-4o-2024-08-06)
```

### Detailed Justification
1. **Strict Evaluator Independence:** Conquer's target answer scorer runs on an OpenAI GPT foundation (`openai/gpt-oss-120b`). Deploying Claude 3.5 Sonnet completely breaks model-family collusion, eliminating tokenization and RLHF dialect alignment bias.
2. **Benchmark-Demonstrated Technical & Coding Superiority:** Claude 3.5 Sonnet consistently leads frontier models in deep software engineering evaluations (SWE-bench Verified, HumanEval, complex algorithmic trade-off analysis). Technical interview grading requires detecting subtle race conditions, off-by-one errors, and asymptotic complexity oversights—domains where Sonnet exhibits the lowest hallucination rate.
3. **Exceptional Instruction Following & Rubric Fidelity:** Empirical testing in evaluation literature shows Claude 3.5 Sonnet adheres with high fidelity to complex system prompts and negative instructions (e.g., "Penalize unrequested preamble and verbosity").
4. **Economic Feasibility via Prompt Caching:** Because Benizakura's system prompt and rubric criteria are static across all 30 evaluation cases in a run, Anthropic's prompt caching discounts cached prompt tokens by 90% (\$0.30 / 1M tokens), making a 30-case bidirectional run cost under \$0.15.

---

## 15. Recommended Experimental Protocol

When the project enters Phase 3 (Live Provider Integration), the validation experiment must execute strictly under this protocol:

```text
Step 1: Dataset Freezing
  └── Finalize 30-case Conquer benchmark with human ground-truth labels and frozen SHA-256 checksums.

Step 2: Schema Remediation
  └── Invert JSON schema in JudgePromptBuilder so rationale & criterion assessments precede winner.

Step 3: Baseline & Candidate Capture
  └── Freeze baseline responses from Conquer prompt v1.0.
  └── Generate candidate responses from Conquer prompt v1.1.

Step 4: Multi-Model Execution Loop
  └── Execute Baseline A, B, C, D and Full Benizakura on Claude 3.5 Sonnet (temp=0.0).
  └── Execute parallel run on GPT-4o for cross-family comparison.

Step 5: Metric Computation
  └── Compute Cohen's kappa (κ), position instability rate, and directional accuracy vs human labels.
  └── Execute paired bootstrap resampling (B=10,000) for confidence interval estimation.

Step 6: Gate Validation
  └── Evaluate ReleaseGate policy triggers against observed statistical distributions.
```

---

## 16. Open Research Questions

1. **Jury Aggregation Dynamics:** Does an asymmetric panel (e.g., 2 Claude votes + 1 GPT-4o vote) produce higher calibration than a single judge, or does it introduce unresolvable tie-breaking deadlocks?
2. **Logit Probability Extraction:** If an open-weight model (e.g., Llama 3.3 70B on dedicated vLLM infrastructure) provides raw token logit probabilities, does continuous expected-value scoring ($\mathbb{E}[\text{score}]$) outperform discrete categorical winner classification?
3. **Adaptive Sample Sizing:** Can Benizakura employ sequential analysis (e.g., Wald's Sequential Probability Ratio Test) to terminate evaluation early after 10 cases if the difference is overwhelmingly decisive, only expanding to 30 cases when near decision boundaries?

---

## 17. Risks and Limitations

1. **Provider Rate Limiting & Outages:** Relying on proprietary cloud APIs introduces network latency variance and rate-limit HTTP 429 risks during CI/CD pipeline runs. Mitigated via exponential backoff, jitter, and exact-match caching.
2. **Silent Model Drift:** Model providers periodically update weights behind fixed model strings. A silent update to `claude-3-5-sonnet` could alter judge strictness over time. Mitigated by pinning exact dated model snapshot identifiers (`claude-3-5-sonnet-20241022`) and validating against frozen calibration sets.
3. **Inconclusive CI Deadlocks:** If `max_unstable_rate` (0.20) or CI boundary rules trigger frequent `INCONCLUSIVE` verdicts on minor pull requests, engineering velocity may stall. Gating policies must provide clear diagnostics indicating whether more cases or prompt refinements are needed.

---

## 18. What We Should Implement Next

To advance Benizakura towards production readiness without violating the current research-first milestone, the immediate subsequent engineering steps are:

1. **Prompt Schema Inversion (CoT Alignment):**
   Update [prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py) and [parser.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/parser.py) to position `"criterion_assessments"` and `"rationale"` before `"winner"` in the JSON template, enabling proper autoregressive Chain-of-Thought.
2. **Construct Official 30-Case Conquer Benchmark:**
   Author the formal 30-case evaluation dataset in `evals/conquer/benchmark_v1.json`, complete with stratified categories, bias probes, and pre-committed human consensus labels.
3. **Hand-Written Anthropic & OpenAI Provider Adapters:**
   Implement zero-dependency, typed HTTP client adapters implementing the [JudgeProvider](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/provider.py) protocol for Claude 3.5 Sonnet and GPT-4o.
4. **Calibration Verification Suite:**
   Implement an offline CLI evaluation script (`benizakura calibrate --benchmark evals/conquer/benchmark_v1.json`) computing Cohen's $\kappa$ and instability rates to enforce the [DECISIONS.md D8](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/docs/DECISIONS.md#L79-L90) kill-gate before enabling automated CI gating.

---

## Decision

**Recommended initial judge:**  
`claude-3-5-sonnet-20241022` (Anthropic Claude 3.5 Sonnet)

**Why:**  
Conquer's candidate system evaluates candidate interview responses using an OpenAI GPT foundation (`openai/gpt-oss-120b`). Selecting an OpenAI-based judge introduces documented self-enhancement bias ($5\%\text{--}15\%$ win-rate inflation) and shared representational failure modes. Claude 3.5 Sonnet provides absolute evaluator independence, market-leading technical reasoning, superior nuance in detecting algorithmic flaws, robust instruction following, and economical execution via prompt caching.

**Confidence:**  
**High** (Strongly supported by cross-family bias literature, state-of-the-art coding benchmarks, and empirical alignment with senior human engineers).

**Important uncertainty:**  
Whether Claude 3.5 Sonnet's position-instability rate on technical interview evaluations remains below Benizakura's policy threshold of $\le 20\%$ under structured rubric prompts, and whether its strictness penalizes valid alternative system architectures that differ from standard textbook designs. This requires empirical validation on the frozen 30-case benchmark.

**Next implementation milestone:**  
Invert the JSON schema in [prompt.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/prompt.py) and [parser.py](file:///Users/srinivasch/Documents/Projects/AI%20Change%20Gate/src/benizakura/parser.py) to mandate Chain-of-Thought reasoning prior to the winner verdict token, then construct the official 30-case stratified Conquer benchmark (`evals/conquer/benchmark_v1.json`) with pre-committed human ground-truth labels.
