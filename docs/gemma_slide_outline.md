# Slide outline — EmbeddingGemma word-priors baseline for BioKGSuite

Audience: supervisor / weekly roundup.
Tone & length: matches `weekly_roundup_may18.pptx`.
Recommended deck length: **4–5 slides**.

The headline result is missing until you've run `scripts/run_gemma_benchmark.sh`
on your machine. Numbers in [brackets] below are the values the bar chart
and concordance table will provide once the run completes. The slide
structure does not depend on the actual numbers.

---

## Slide 1 — Title + question

**Title:** How much of the drug–disease indication signal is already in a
pretrained language model's word priors?

**Setup line (one sentence):** Added EmbeddingGemma-300m as a non-training
baseline to the BioKGSuite embedding analysis (notebook 08), to measure
the fraction of indication signal recoverable from entity *names alone*,
with zero knowledge of graph structure.

**Why this matters:** Calibrates the marginal value of training a KG
embedding model. If a pretrained text encoder captures most of the
signal from names, the marginal benefit of training TransE/RotatE on a
specific KG is smaller than it looks.

---

## Slide 2 — Method (deliberately spartan)

**Three bullets:**

- **Model.** `google/embeddinggemma-300m`, 768-d output, Matryoshka-truncatable.
- **Score.** Per entity, embed the bare `nodes_df['name']` string. No type
  prefix, no task prefix, no neighborhood. Score (drug, disease) pair by
  cosine similarity. No training.
- **Eval.** Same held-out drug–disease pairs and same type-constrained
  negatives as TransE/RotatE in nb08 — so AUROC/MRR/Hits@K are directly
  comparable.

**Footnote:** Bare names chosen deliberately to isolate the "what does
the model already know about these words?" signal. Type prefixes or
EmbeddingGemma's task-specific prompts would conflate word priors with
trained retrieval behavior.

---

## Slide 3 — Headline result

**Figure:** `results/figures/08_gemma_vs_kge_auroc.{pdf,png}`
(generated when you re-execute nb08 — TransE / RotatE / Gemma bars per KG,
type-constrained negatives, 95% bootstrap CIs).

**One-line readout (fill in after run):**
> EmbeddingGemma reaches AUROC ≈ [X] on Hetionet and ≈ [Y] on PrimeKG using
> only word priors, vs. ≈ [Z] for TransE trained on the full graph.
> The gap (TransE – Gemma) on readable-name KGs estimates the share of
> indication signal that requires actual graph structure beyond what a
> pretrained LM already encodes about drug and disease names.

**Caveat box (bottom-right of slide):**
> DRKG, OpenBioLink, BioKG marked with `†` — these KGs' loaders return
> opaque ID strings (`"2157"`, `"DB00001"`, `"0001234"`) instead of real
> names, so EmbeddingGemma sees noise on those. Result there is at random
> baseline by construction, not a model failure. Fixing this is a separate
> name-resolution task (see follow-up doc).

---

## Slide 4 — Concordance check

**Question:** Does Gemma rank the readable-name KGs in the same order
that TransE / RotatE do?

**Table** (from the new concordance cell in nb08; fill in after run):

| Strategy | Reference | Spearman ρ vs Gemma | p | n |
|---|---|---|---|---|
| type-constrained | TransE | [ρ] | [p] | [n] |
| type-constrained | RotatE | [ρ] | [p] | [n] |
| random           | TransE | [ρ] | [p] | [n] |
| ...              | ...    | ... | ... | ... |

**Interpretation lines (choose based on the result):**

- **High ρ (≥ 0.7):** "KG-quality ranking is largely recoverable from name
  priors alone; trained KG embeddings reinforce a ranking that text already
  predicts."
- **Low ρ (≤ 0.3):** "KG-quality ranking is driven by graph structure that
  word priors don't capture — trained embeddings are doing real work
  independent of textual priors."
- **Mixed (~0.4–0.6):** "Word priors partially predict KG ranking; this is
  consistent with name conventions being correlated with KG curation
  practices."

---

## Slide 5 — Limitations + next steps

**Limitations:**

- Only 2 of 6 KGs (Hetionet, PrimeKG) have human-readable names in their
  default loader output. DRKG, OpenBioLink, BioKG need name resolution
  before this experiment is meaningful for them.
- "Bare names, no context" is one design point. A neighborhood-augmented
  variant (`"Aspirin targets COX1, COX2; treats pain..."`) would test a
  different question — how much structure can a text encoder absorb when
  given graph context as text. Not done in this iteration.
- Cosine on bare embeddings is symmetric; the actual indication relation
  isn't. EmbeddingGemma supports asymmetric query/document prompts;
  intentionally skipped here to isolate word priors.
- MATRIX run uses subsampling by default — full run is feasible on GPU but
  hasn't been done yet.

**Next steps (priority order):**

1. **Name resolution for the 3 ID-only KGs** — would extend the experiment
   from 2 KGs to 5 and make the supervisor's "benchmark across our KGs"
   ask actually fully covered. See `docs/gemma_name_resolution_followup.md`.
2. **Full MATRIX run** on a GPU (~30 min vs. hours subsampled on CPU).
3. *(Optional)* Add a name + 1-hop-neighborhood variant for comparison —
   the cleanest follow-up if Gemma's word-prior signal turns out to be
   weak. Would answer "does adding graph context as text close the gap?"

---

## Speaking notes (off-slide)

- Lead with the question, not the method. The interesting part is "what
  fraction of the signal is in the words?" — not "we ran EmbeddingGemma."
- If asked why no `query:` / `title:` task prefixes — explain: those would
  prime the model toward a retrieval task and conflate "word prior" with
  "trained retrieval behavior." Bare embeddings answer the cleaner question.
- If asked about model choice — EmbeddingGemma was specifically requested
  and is a defensible pick: open weights, ≤500M params, top open model on
  MTEB at release, 768-d Matryoshka-truncatable for compute control.
- If pressed on the ID-only KGs being "broken" — reframe: that result *is*
  the experiment. If Gemma scored well on `"DB00001"` strings, that would
  imply something is wrong (memorized identifier-level priors). The
  random-baseline result on ID-only KGs is the expected, healthy outcome.
