# BioKGSuite → Nature Communications: manuscript outline

*Working document. Built from the repo as it stands (6 KGs × 7 quality dimensions; 116-pair leakage-controlled prospective LLM benchmark; Llama 1B→405B scaling sweep). Numbers are pulled from `results/tables/00_benchmark_summary.csv` and `results/tables/09_llm_runs/`.*

---

## 0. The honest framing call (read this first)

You have two papers' worth of material and they are **not** equally strong for *Nature Communications*:

- **(A) The resource** — a 7-dimension, 18-metric quality benchmark across 6 biomedical KGs. Valuable, reproducible, but on its own this is a *Scientific Data / Communications Biology / Bioinformatics / NeurIPS D&B* paper. Nature Comms rarely takes "we built a benchmark."
- **(B) The finding** — in a **leakage-controlled, prospective** drug-repurposing task, KG grounding gives a large MRR lift over a strong no-KG prior; the lift is a **model-scale phenomenon that saturates ~70B**; it is robust across three independent model families; and difficulty is intrinsic to the pair, not the model.

**My call:** lead with **(B) as the finding**, use **(A) as the instrument** that makes the finding measurable, and make the intellectual spine the link between them: *does intrinsic graph quality predict downstream usefulness to an LLM?* That question is the part with broad significance — it reframes a decade of "build a better KG" work around "better *for what, measured how*."

**The catch you must close to make that spine land:** your intrinsic-quality winner is **MATRIX (0.854 overall)**, but the LLM task only ran **PrimeKG, DRKG, BioKG**. You cannot currently test "intrinsic quality predicts downstream LLM utility" because the intrinsic leader was never in the downstream task. This is the single highest-leverage gap (see §7). Close it and you have a genuine, surprising, Nature-Comms-shaped claim. Leave it open and the paper is "KG helps LLMs + a nice benchmark," which is a weaker, more expected story.

**Realistic venue read:** with the gap closed and the finding framing, Nature Comms is a credible *shot* but not a layup — the base "KG helps LLMs" intuition is not surprising; your novelty is **prospective + leakage-controlled + scaling law + intrinsic-vs-extrinsic**. Prepare *Communications Biology*, *Patterns*, or *Cell Reports Methods* as high-probability fallbacks, and *NeurIPS/ICLR Datasets & Benchmarks* if you want the ML-community stamp on the benchmark.

---

## 1. Title (options)

1. *Knowledge-graph grounding improves language-model drug repurposing in proportion to model scale* — finding-forward.
2. *Intrinsic knowledge-graph quality does not predict its value to a language model: a prospective drug-repurposing benchmark* — the surprising-claim version (only if §7 gap closed).
3. *A leakage-controlled prospective benchmark for knowledge-graph–grounded drug repurposing with language models* — resource-forward, safest.

Recommend **#1** for Nature Comms; keep **#2** if the all-6-KG run supports the disconnect.

(≤15 words; all three comply.)

---

## 2. Abstract (≈200 words, draft skeleton)

- **Problem (2 sentences):** LLMs are increasingly used to prioritise drug–disease hypotheses, and KGs are the obvious source of grounding — but evaluations are retrospective and contaminated (the "novel" pair is already an edge in the KG), so reported gains may be memorisation, and there is no principled way to choose *which* KG to ground on.
- **What we did:** built (i) a 7-dimension, 18-metric quality benchmark over 6 public biomedical KGs, and (ii) a **prospective** ranking benchmark of 116 drug–disease indications approved in 2023–2026, filtered to remove answer leakage and unresolved IDs.
- **Findings:** KG grounding raises mean reciprocal rank from a ~0.30 no-KG prior to ~0.64–0.73 across three model families; the lift over no-KG **grows with model scale and saturates near 70B** (Llama 1B→405B); difficulty is intrinsic to the pair (tracks model confidence, r≈0.98, and agrees across families); and intrinsic graph-quality scores [do / do not — pending §7] rank KGs the way downstream utility does.
- **Significance:** grounding, not model size or KG choice alone, is the dominant lever; we release the benchmark and code.

---

## 3. Introduction (≈600–800 words, 3–4 paragraphs)

1. **Stakes.** Drug repurposing is high-value; LLMs now triage hypotheses at scale; biomedical KGs (PrimeKG, Hetionet, DRKG, OpenBioLink, BioKG, MATRIX) are the standard grounding substrate.
2. **The two unsolved problems.** (a) *Evaluation is contaminated and retrospective* — most LLM+KG benchmarks score pairs whose answer edge is already in the KG, so "reasoning" is recall. (b) *KG choice is unprincipled* — KGs are compared by ad-hoc, single-axis metrics (size, KGC AUROC), and there is no multi-dimensional quality standard, nor evidence that any intrinsic score predicts downstream task value.
3. **Gap in prior work.** Position against: PrimeKG/Hetionet as resources; KGC-topology studies (Bioinformatics 2025); LLM+KG method papers (K-Paths, KEDRec-LM, LLM-assisted concept representation); retrieval/hallucination benchmarks (STaRK, MultiHal). None combine a *prospective, leakage-controlled* task with a *multi-dimensional quality* benchmark across *multiple KGs and model families and scales*.
4. **Contributions (bulleted in text):** (i) BioKGSuite quality benchmark; (ii) the 116-pair prospective leakage-controlled task + bias audit; (iii) the cross-family grounding result and the scaling law; (iv) the intrinsic-vs-extrinsic analysis; (v) open release.

---

## 4. Results (the spine — each subsection = one display item)

**R1 — A multi-dimensional quality benchmark separates six biomedical KGs.**
→ *Fig 1.* 7 dimensions / 18 metrics × 6 KGs. Headline ranking (overall mean): MATRIX 0.85 > PrimeKG 0.73 > DRKG 0.70 > OpenBioLink 0.69 > BioKG 0.64 > Hetionet 0.63. Point: no KG dominates every axis; coverage, trustworthiness and generalisation trade off. (Source: notebook 00 + 01–08.)

**R2 — A prospective, leakage-controlled benchmark for realistic ranking.**
→ *Fig 2.* Construction funnel 253→116 (removed: 81 answer-leakage, 37 no-KG-coverage, 19 unresolved IDs) + selection-bias audit showing the kept set matches the dropped set on modality, therapeutic area and recency (only older-drug ≤2015 under-represented, by design, OR≈0.45). Point: the benchmark measures *prospective* ranking, not recall, and is an unbiased subset. (09 / 09a.)

**R3 — KG grounding gives a large, robust lift across model families.**
→ *Fig 3 (the money figure; current `fig1_llm_main`).* no-KG prior MRR ~0.29–0.33 → KG arm ~0.62–0.73 for GPT-4.1-mini, Gemini-3.1-flash-lite, Llama-3.3-70B. Per-disease lift distributions + rank-position shift. Point: grounding moves the true drug toward rank 1; effect is consistent across three independent families. (09.)

**R4 — The grounding benefit is a scaling phenomenon that saturates.**
→ *Fig 4 (09c).* Llama 1B→405B: covered-arm lift over no-KG −0.03 (1B) → +0.24 (8B) → +0.38 (405B), flattening after 70B, while the **no-KG prior stays flat (~0.30)**. Point: the gain is grounding the model can *use*, not bigger-model memorisation; small models can't exploit the KG. (09c.)

**R5 — Where, and for whom, grounding fails (a four-question diagnostic).**
→ *Fig 5 (09b), 3 panels.* Structured as four questions:
- **Which pairs are hardest?** The 20 lowest-MRR pairs (panel a = the ranked list).
- **What do they share?** Not modality and **not recency** (identical splits) — the separators are **low model confidence (1.7 vs 3.9 / 5)** and **sparse/inconsistent KG evidence (frac_consistent 0.17 vs 0.72)**; secondary signals are therapeutic area (Metabolic/Infectious hard, Oncology easy) and lower KG coverage (2.0 vs 2.6 KGs).
- **Are KGs uniformly unhelpful on them?** No — and crucially, *not uniform across KGs*. Holding the model fixed (GPT-4.1-mini) and comparing the three KGs pair-by-pair, KG agreement is only moderate (mean pairwise r ≈ 0.47; **6/20** hardest pairs shared by all three) — far below model agreement. Pooling all three models (3× data per cell) barely moves it (0.47→0.53), so it is **not** a sampling artifact. *Which* pairs a KG helps is KG-specific; the KGs are **complementary**, arguing for a KG union/ensemble (panel b).
- **Do models fail on the same pairs?** Yes — per-pair MRR correlates r=0.81–0.88 across GPT/Gemini/Llama; **14/20** hardest shared (panel c). Difficulty is model-invariant.

Point: difficulty is a property of the *pair with respect to the model* (model-invariant) but a property of the *pair × KG* (KG-specific) — a principled, reusable stress set, and direct evidence that combining KGs covers more hard pairs than any single one. (09b.)

*Methods note:* the no-KG arm sits at/below chance (mean MRR 0.28 vs chance 0.34), so defining "hard" by the no-KG prior selects largely on noise and inflates apparent KG lift via regression to the mean. Difficulty and KG-helpfulness are therefore assessed by **cross-KG and cross-model agreement**, not by lift over the no-KG baseline.

**R6 — [Conditional] Intrinsic quality vs downstream utility.**
→ *Fig 6 (TO BE RUN).* Correlate each KG's intrinsic dimension scores against its downstream LLM MRR, across **all 6 KGs**. This is the surprising claim — currently unsupported because MATRIX/Hetionet/OpenBioLink were never in the LLM task. If the correlation is weak (likely, given BioKG ranks 5th intrinsically but ties best downstream for GPT), that *is* the Nature-Comms headline.

*(Optional supplement: notebook 08 — trained KG embeddings (TransE/RotatE) beat an EmbeddingGemma name-prior, especially on id-only KGs, controlling for "the model just knows the name.")*

---

## 5. Discussion (≈500–700 words, no subheadings)

- **What's new:** prospective + leakage-controlled design; scaling law of grounding; intrinsic≠extrinsic (if R6 lands).
- **Why it matters:** practitioners should spend effort on *grounding and model scale ≥8B*, not on chasing the "best" KG by intrinsic metrics; benchmark contamination has been inflating LLM+KG results.
- **Limitations (state plainly):** one task family (drug repurposing ranking); 116 prospective pairs is modest n; closed-model versions drift; KG snapshots age; intrinsic metrics are our operationalisation, not ground truth.
- **Outlook:** extend to mechanism/path explanations, multi-task, periodic prospective refresh as new approvals land.

---

## 6. Methods (≤3,000 words; subheadings)

KGs and snapshots · the 7 quality dimensions and 18 metrics (one paragraph each, with the exact estimator) · gold-standard construction and ID resolution · the 116-pair prospective set + leakage filter + bias audit (Fisher OR, Mann–Whitney) · LLM ranking protocol (pool construction, prompt, 3 seeds × 2 shuffles, MRR/Hits@k, no-KG vs KG arms, "covered" vs "pooled") · the Llama scaling ladder · embedding validation (TransE/RotatE/EmbeddingGemma, resampled CIs) · statistics and reproducibility (seeds, bootstrap vs rerun CIs, code/data availability).

---

## 7. Gaps to close before submission (prioritised — the referee bait)

1. **[Highest leverage] Run the LLM task on all 6 KGs** (esp. MATRIX, the intrinsic winner, + Hetionet, OpenBioLink). Without this you cannot make the intrinsic-vs-extrinsic claim (R6), which is the strongest Nature-Comms angle.
2. **More than one model per family / one closed + one open at matched scale** — strengthens "robust across families" beyond GPT/Gemini/Llama singletons.
3. **Statistical rigor pass:** paired tests for KG-vs-no-KG lift per model; multiple-comparison control; report effect sizes with CIs everywhere (you already have bootstrap-vs-rerun machinery — surface it).
4. **n and power:** 116 prospective pairs is the headline benchmark's whole weight — pre-register the next approval-year refresh to grow n and show stability over time.
5. **Ablate the grounding mechanism:** is the lift from *relevant* subgraph or just *more tokens*? A shuffled/irrelevant-KG control would kill the "it's just context length" referee objection. (You partly have this via no-KG; add a *random-KG* arm.)
6. **Leakage audit as a figure, not a footnote** — referees will probe contamination first; make it bulletproof.
7. **Decontaminate model knowledge cutoffs vs approval dates** — confirm 2023–26 indications post-date training cutoffs for each model, or stratify by it.

---

## 8. Display-item budget (Nature Comms: ≤10; you're at 5 main + supp)

| # | Figure | Source | Status |
|---|--------|--------|--------|
| 1 | 7-dimension quality benchmark × 6 KGs | nb 00 | ✅ have |
| 2 | Prospective benchmark construction + bias audit | 09/09a | ✅ have |
| 3 | KG-grounding lift across 3 families | 09 (`fig1_llm_main`) | ✅ have |
| 4 | Grounding lift vs model scale (saturation) | 09c | ✅ have |
| 5 | Difficulty is intrinsic (confidence r≈0.98) | 09b | ✅ have |
| 6 | Intrinsic quality vs downstream utility | **new** | ⚠ needs all-6-KG run |
| S1–Sn | Embedding validation; per-dimension detail; per-KG tables | 08, 01–07 | ✅ have |

---

## 9. Positioning / key references to cite (starter set)

- **KG resources:** PrimeKG (Chandak et al.), Hetionet (Himmelstein et al.), DRKG, OpenBioLink, BioKG, MATRIX/Every Cure.
- **KG quality / topology & KGC:** "role of graph topology in biomedical KGC" (Bioinformatics 2025); KGC benchmark literature.
- **LLM + KG for biomedicine:** K-Paths (KDD 2025); KEDRec-LM; "LLM-assisted biomedical concept representation for drug repurposing" (EMNLP Findings 2025); LLMs-for-KG-embedding survey (2025).
- **LLM retrieval / contamination benchmarks:** STaRK (2024); MultiHal (2025); data-contamination literature.
- **Drug repurposing background:** computational repositioning reviews.

*(Full DOIs to be filled at drafting; keep ≤70 refs.)*

---

## 10. Suggested next actions (sequenced)

1. **Decide the spine** — finding (#1 title) vs surprising-claim (#2). This gates whether §7.1 is mandatory.
2. If going for the strong claim: **submit the all-6-KG LLM jobs** (you already have the HPC scaffolding; it's the same `run_llm_*` path with the extra KGs).
3. **Lock Figs 1–5** from current outputs (already regenerable headless via `scripts/regenerate_figs.py`).
4. Draft **Abstract + Intro + R1–R5** first (they're fully supported now); leave R6 as a stub until §7.1 lands.
5. Internal red-team against §7 before external submission.
