# LLM prompting strategies for drug-disease plausibility (nb09)

> Scope: nb09's headline analysis is the KG-context ablation (C0 no-KG, C1
> direct edges, C2 mechanistic paths), using the layered slot-aware extraction
> with the `llm_prompt` strategy, reported as per-(KG, condition) AUROC. This
> document is a reference for the prompting-strategy module
> (`src/prompting_strategies.py`) that the generation loop calls; the strategies
> below are available for follow-up work that varies how the question is asked
> rather than the KG context.

Five strategies, two restricted and three unrestricted, selected from the
biomedical-LLM literature for the BioKGSuite drug-repurposing benchmark. All run
against the same stratified drug-disease pairs, the same 6 KGs, and the same
llama3.1:8b model, varying only how the question is asked.

The library file `src/prompting_strategies.py` contains 11 strategies
total; the 5 below are in `DEFAULT_STRATEGIES_FOR_NB09`. The other 6 are
opt-in for follow-up work. See "Strategies considered and cut" at the
bottom for the rationale.

## Why these five

The lineup is **balanced between restricted (constrained output schema)
and unrestricted (open-ended reasoning)** so we can separate two distinct
questions:

1. **Restricted strategies** test whether *forcing structure* on the
   output, particularly explicit enumeration of supporting and
   contradicting evidence, improves accuracy and explainability. This
   is the **DrugReX / DrugReAlign** line of work [2, 3], reinforced by
   Paper 1's finding that the "Map-Counter-Missing" hybrid prompt
   (enumerate symptoms supporting the diagnosis, against it, and missing)
   yielded the highest model confidence [9].

2. **Unrestricted strategies** test whether *letting the model reason
   freely*, through chain-of-thought, multi-persona deliberation, or
   multi-sample aggregation, improves accuracy and calibration.

Jeon et al. (2025) [4] explicitly argue for parsimony: "complex prompting
techniques do not significantly enhance performance compared to simpler
approaches" (p > .05 across most methods). This drove the cut from 9
strategies to 5, every strategy must test a *distinct* hypothesis with
*direct* biomedical-LLM literature support.

## Call cost

| Strategy | Type | Calls per cell | Approx tokens per call |
|---|---|---|---|
| `zero_shot_direct` (baseline) | restricted | 1 | ~150 |
| `structured_json` | restricted | 1 | ~250 |
| `zero_shot_cot` | unrestricted | 1 | ~500 |
| `multi_expert_rot` | unrestricted | 1 | ~800 |
| `cisc` | unrestricted | 5 (sequential) | 5 × ~150 |
| **Total per cell** | | **9** | |

Per-cell cost is 50 pairs × 2 conds × 3 reseeds × 6 KGs = 1,800 cells
per strategy. Pilot total ≈ **16,200 calls** (~30 min wall time on
llama3.1:8b). Full (300 pairs) ≈ **97,200 calls** (~3 hr).

---

## RESTRICTED experiments (2)

### 1. `zero_shot_direct`, the baseline (Input-Output prompting)

**Prompt template (abridged):**
> You are a drug repurposing expert. {kg_block} Question: Is {drug} a
> plausible treatment for {disease}? Respond on one line: "Answer:
> <Yes|No>, Confidence: <1-5>".

**Hypothesis:** Reference point. Every other strategy is graded against
this.

**Literature support:** This is the "Input-Output (IO) prompting" formally
defined and tested in Wang et al. 2024 [9] (npj Digital Medicine), they
compared IO against three more complex strategies (0-COT, P-COT, ROT)
across 9 LLMs on AAOS osteoarthritis guidelines. IO was the best prompt
for several GPT-3.5 variants (27.1% to 55.3% consistency). It is also
the "simple prefix" baseline characterized by Sivarajkumar et al. 2024
[1] in their comprehensive JMIR Medical Informatics empirical evaluation
of clinical-NLP prompting (n = 5 tasks, 3 models, 141 citations).

**Reading the result:** Sets the bar. If anything fails to beat this,
either the strategy or the model is the bottleneck.

---

### 2. `structured_json`, JSON schema with reasoning + contradictions

**The change:** Output forced to JSON:
`{answer, confidence, reasoning, contradictions}`. The `contradictions`
field explicitly asks the model to enumerate counter-evidence even when
answering Yes. Uses Ollama's `format='json'` decoding mode.

**Literature support:** This design is the synthesis of three
biomedical-LLM findings:

- **Paper 1 (IISEC 2026)** [9'] tested four prompt styles (Structured,
  Role-based, Hybrid-1, Hybrid-2) on GPT-4 with 50 VAERS clinical cases.
  Their **Hybrid-2 "Map-Counter-Missing" prompt**, which asks the model
  to "identify symptoms associated with the disease, opposing evidence,
  and missing symptoms", produced the highest model confidence score
  (4.10/5) and the most explanatory output. The `contradictions` field
  in our JSON schema is the direct equivalent of their "Counter" slot.
- **DrugReX** [2] (Huang et al. 2025), first LLM system for drug
  repurposing with built-in explainability; LLM-generated explanations
  supported by KG evidence rated higher in quality than LLM-alone
  explanations by domain experts.
- **DrugReAlign** [3] (Wei et al. 2024, BMC Biology, 46 citations),
  "multi-source prompt framework for drug repurposing based on large
  language models" using structured prompts combining target summaries
  and drug-target interaction context. Their finding, "a direct
  correlation between the accuracy of LLMs' target analysis and the
  quality of prediction outcomes", directly argues for forcing the
  model to articulate its analysis in a structured form.

**Hypothesis:** Parse rate ≈ 100% (vs ~95% for natural-language
parsing). The `contradictions` slot improves accuracy on plausible hard
negatives by forcing explicit counter-argument consideration. Surfaces
interpretable failure modes in qualitative analysis.

**Reading the result:** Two readouts. (1) Parse-rate-by-strategy table,
should be ~1.0 for this row, providing a clean comparison signal. (2)
Qualitative spot-check: read the `contradictions` field for cases where
the model said Yes but the gold label is No.

---

## UNRESTRICTED experiments (3)

### 3. `zero_shot_cot`, "Let's think step by step"

**The change:** Prompt instructs the model to "think step by step.
Consider the drug's mechanism, the disease pathophysiology, and whether
any plausible biological pathway connects them" before outputting the
final answer. `max_tokens` raised 50 → 400 to give reasoning room.

**Literature support:** Direct equivalent of Wang et al.'s "0-COT
prompting" [9] (the best prompt for Bard at 44.1% consistency, and
runner-up for several GPT-4 variants). Sivarajkumar et al. 2024 [1]
found CoT prompting "highly effective across tasks" in their clinical
NLP evaluation. Jeon et al. 2025 [4] confirmed traditional CoT gives
"consistently stable results across clinical datasets" across 5 LLMs.

**Hypothesis:** Verbal reasoning surfaces the model's mechanism
beliefs. Should help most on `pos_phase12` (Phase 1-2 positives that
aren't canonically known) and on `neg_plausible` (where pattern-matching
on names would fool the baseline).

**Reading the result:** Stratified accuracy. If CoT lifts `pos_phase12`
or `neg_plausible` specifically, the reasoning hypothesis holds. If it
only lifts `pos_approved` (the canonical positives), CoT is just
verbosity that helps the model recall what it already knew.

---

### 4. `multi_expert_rot`, Reflection of Thoughts (3-expert deliberation)

**The change:** Single-call prompt that asks the model to **simulate
three drug-repurposing experts** who (Step 1) independently reason about
the question, (Step 2) discuss their disagreements and **backtrack** to
revise their initial reasoning if convinced by counterarguments, and
(Step 3) reach consensus. Output: the final consensus answer +
confidence. `max_tokens` raised to 800 to give the deliberation room.

**Literature support:** Direct implementation of the **ROT (Reflection
of Thoughts)** strategy from Wang et al. 2024 [9] (npj Digital
Medicine), the most directly relevant of the three uploaded papers. In
their study comparing 4 prompts × 9 LLMs on AAOS osteoarthritis
guidelines:

- **gpt-4-Web + ROT** was the **single best combination overall**
  (62.9% consistency with clinical guidelines).
- On strong-evidence-level questions specifically, ROT reached
  **77.5% consistency**, the highest of any prompt-model pair.
- ROT "could minimize the occurrence of egregiously incorrect answers"
  by forcing the model to revisit and revise initial verdicts.

This is the strategy with the **strongest direct biomedical-LLM evidence
for guideline-style questions** in the entire reading set. Drug-disease
plausibility on stratified Open Targets pairs is a structurally similar
task (graded evidence, expert-judgment-based).

**Hypothesis:** Multi-persona deliberation forces the model to consider
competing perspectives within a single call, reducing single-pass
anchoring on superficial cues (e.g., drug name familiarity). Particularly
useful on hard negatives where the first-pass instinct is wrong but
recoverable through challenge.

**Reading the result:** Compare against `zero_shot_cot`, if ROT
substantially beats plain CoT, the multi-persona / backtracking
component is doing the work (not just the reasoning instruction). If
they tie, the extra prompt complexity isn't worth the extra tokens for
this LLM, consistent with Jeon et al.'s parsimony finding [4] for
smaller models.

---

### 5. `cisc`, Confidence-Informed Self-Consistency

**The change:** Sample 5 times at `temperature=0.7`, each call asking
for a verbalized 0-100 probability ("On a scale of 0 to 100, what is the
probability..."). Final answer is a **weighted vote**: each sample's
weight is its distance from 50 (i.e. how far from "I don't know"), and
the final score is the weighted average probability. Binarize at 0.5 for
the prediction; bucket `|score - 0.5|` for the 1-5 confidence.

**Literature support:** Direct implementation of **CISC** [6]
(Taubenfeld et al. 2025): "CISC outperforms self-consistency in nearly
all configurations, reducing the required number of reasoning paths by
over 40% on average." Tested on 9 models and 4 datasets.

The verbalized-probability output addresses two findings from
biomedical-LLM calibration work:

- **Omar et al. 2025** [7] (JMIR Medical Informatics): across 12 LLMs
  on 1,965 clinical multiple-choice questions, "worse-performing models
  exhibited paradoxically higher confidence" (r=-0.40, p=.001).
  Verbalized probability gives a continuous score that surfaces this
  miscalibration.
- **Xiong et al. 2023** [8] (872 citations): verbalized confidence +
  multi-sample consistency + smart aggregation are the three components
  for reliable confidence elicitation from black-box LLMs.

Wang et al. 2024 [9] also highlight self-consistency / reliability:
they report Fleiss κ ranging from −0.002 to 0.984 across prompt-model
pairs, demonstrating that **repeating the same question can give
wildly different answers**, and that consistency is a critical metric
alongside accuracy. CISC's weighted aggregation across 5 samples is
exactly the kind of mitigation they recommend.

**Hypothesis:** CISC produces:
- Best calibration (verbalized prob is continuous, not discrete 1-5).
- Highest AUROC computable directly from the score (no `pred × confidence`
  multiplication needed).
- Modest accuracy lift over single-call CoT, justifying the 5× cost.

**Reading the result:** Three things to look at. (1) Calibration plot,
CISC should sit closest to the diagonal. (2) AUROC computed from the
continuous CISC score vs the discrete confidence-weighted score from
other strategies. (3) Cost-adjusted accuracy: divide accuracy lift by
`n_calls`, if CISC's lift is < 5× the lift of `zero_shot_cot`, you're
not getting your money's worth.

---

## On the SOFT vs STRICT axis (Paper 2)

Colangelo et al. 2025 [10] (Paper 2) introduce a useful framing for the
restricted/unrestricted distinction in the context of biomedical
literature screening: **soft prompts** maximize recall by accepting
articles unless an inclusion criterion is explicitly failed, while
**strict prompts** demand explicit positive evidence for every
criterion. They note (and we adopt) that minor rephrasings ("Accept if
participants are X" vs "Reject if participants are not X") can produce
markedly different outputs from the same model.

In our current 5-strategy lineup, the **restricted strategies are
deliberately neutral** (neither soft nor strict). Adding explicit
soft/strict variants would be a natural Phase 2 follow-up: rerun
`zero_shot_direct` and `structured_json` with both phrasings to measure
the SOFT-vs-STRICT recall/precision tradeoff on drug-disease plausibility.
That would test the Colangelo framing directly without disturbing the
current restricted-vs-unrestricted comparison.

## Strategies considered and cut

These are still in `src/prompting_strategies.py` for opt-in use but are
not in `DEFAULT_STRATEGIES_FOR_NB09`. The cut rationale is parsimony
(Jeon et al. 2025 [4]): every strategy must test a *distinct* hypothesis
with *direct* biomedical-LLM literature support.

| Strategy | Reason for cut |
|---|---|
| `few_shot_3_cot` | Swapped out for `multi_expert_rot` after Paper 3 review, ROT has stronger biomedical evidence for guideline-style questions. CLINICR [5] support remains; available as opt-in. |
| `step_back` | No drug-repurposing-specific literature support I could find. |
| `few_shot_3` (plain, no CoT) | Superseded by `few_shot_3_cot` which has stronger support. |
| `self_consistency_5` | Rolled into `cisc` (CISC dominates plain self-consistency per Taubenfeld et al. 2025 [6]). |
| `verbalized_prob` (single sample) | Rolled into `cisc`. |
| `prompt_then_verify` | Self-Refine (Madaan et al. 2023) is general; no biomedical-LLM-for-KG-repurposing application found. |

To test any of these in a future run, edit `STRATEGIES_TO_RUN` in nb09
cell 2 to include them. The library code supports it.

## What "good" looks like

Pre-registered priors (subject to empirical verification):

- **Best accuracy:** `multi_expert_rot` or `cisc`, the two strategies
  with strongest direct support for clinical guideline questions
  (Paper 3) and for reliability (Paper 3 + CISC paper).
- **Best calibration:** `cisc` (continuous prob + weighted vote).
- **Best parse rate:** `structured_json` (~100%, by construction).
- **Worst accuracy:** `zero_shot_direct`, the baseline. If anything
  fails to clear this bar, that strategy or the model is the bottleneck.

If `multi_expert_rot` fails to beat `zero_shot_cot` on llama3.1:8b
specifically, that's consistent with Paper 3's finding that ROT was
best on **gpt-4-Web** but not better than other prompts on smaller
models (the result depended heavily on model capability). Document this
as a finding rather than a failure.

---

## References

[1] [An Empirical Evaluation of Prompting Strategies for Large Language Models in Zero-Shot Clinical Natural Language Processing](https://consensus.app/papers/details/554ba0e87c3354d2bbc5dc0624ffe23c/?utm_source=claude_code) (Sivarajkumar et al., 2024, JMIR Medical Informatics, 141 citations)

[2] [DrugReX: an explainable drug repurposing system powered by large language models and literature-based knowledge graph](https://consensus.app/papers/details/27e83767c8d35cd292103ecdd9621519/?utm_source=claude_code) (Huang et al., 2025, Research Square, 3 citations)

[3] [DrugReAlign: a multisource prompt framework for drug repurposing based on large language models](https://consensus.app/papers/details/445c95ef1b345fe584113697f8ce097b/?utm_source=claude_code) (Wei et al., 2024, BMC Biology, 46 citations)

[4] [A comparative evaluation of chain-of-thought-based prompt engineering techniques for medical question answering](https://consensus.app/papers/details/fa79f645fd085d30996f6c5fbb6a79e5/?utm_source=claude_code) (Jeon et al., 2025, Computers in Biology and Medicine, 24 citations)

[5] [Few shot chain-of-thought driven reasoning to prompt LLMs for open ended medical question answering (CLINICR)](https://consensus.app/papers/details/d81a4e948fd555cd9dc1c7453fd34ee3/?utm_source=claude_code) (Gramopadhye et al., 2024, ArXiv, 69 citations)

[6] [Confidence Improves Self-Consistency in LLMs (CISC)](https://consensus.app/papers/details/425ec6bdbe0d5ca581bf388c30078b75/?utm_source=claude_code) (Taubenfeld et al., 2025, 89 citations)

[7] [Benchmarking the Confidence of Large Language Models in Answering Clinical Questions](https://consensus.app/papers/details/9323b11e06775363bf49e350f66c9eaa/?utm_source=claude_code) (Omar et al., 2025, JMIR Medical Informatics, 30 citations)

[8] [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://consensus.app/papers/details/30ea5a0cb6b8541ba175a690c7ee2cba/?utm_source=claude_code) (Xiong et al., 2023, ArXiv, 872 citations)

**User-uploaded papers (cited as [9'], [9], [10] in text above):**

[9'] *The Impact of Prompting Strategies on the Quality of LLM-Generated Biomedical Explanations* (IISEC 2026, IEEE), compared structured / role-based / hybrid prompts on GPT-4 with VAERS clinical cases; Hybrid-2 "Map-Counter-Missing" was most explanatory (avg confidence 4.10/5). DOI: 10.1109/IISEC69317.2026.11418489

[9] *Prompt engineering in consistency and reliability with the evidence-based guideline for LLMs* (Wang et al., 2024, npj Digital Medicine 7:41), compared IO / 0-COT / P-COT / ROT prompts across 9 LLMs on AAOS osteoarthritis guidelines; gpt-4-Web + ROT was best overall (62.9%) and on strong-evidence questions (77.5%). https://doi.org/10.1038/s41746-024-01029-4

[10] *How to Write Effective Prompts for Screening Biomedical Literature Using Large Language Models* (Colangelo et al., 2025, BioMedInformatics 5:15), introduces the SOFT (recall-maximizing) vs STRICT (precision-maximizing) prompt framing. https://doi.org/10.3390/biomedinformatics5010015

Create or connect a free Consensus account to return more than 3 results per search in Claude Code: https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code
