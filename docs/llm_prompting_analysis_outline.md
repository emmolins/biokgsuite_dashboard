# Analysis outline — LLM prompting strategies (nb09 results)

How to read the pilot output once `bash scripts/run_prompting_pilot.sh`
finishes. Designed for either a written writeup or a 3-4 slide subsection
in the weekly roundup.

## What lives in `results/tables/09_llm_runs/responses.parquet`

One row per (`llm`, `kg`, `strategy`, `condition`, `pair_idx`, `reseed`).
Key columns:

- `label` — gold (0/1)
- `label_pred` — model's binary prediction (or null if it failed to parse)
- `confidence` — 1-5
- `correct` — bool
- `n_calls_made` — actual Ollama calls for that cell (5 for self_consistency,
  2 for step_back/prompt_then_verify, 1 otherwise)
- `response` — raw model text (concatenated for multi-call strategies)
- `error` — null on success

The new analysis cells in nb09 produce four headline tables / figures:

1. Accuracy by strategy × condition (averaged across KGs)
2. Parse rate by strategy
3. Call-cost by strategy
4. Calibration curve (confidence vs empirical accuracy, per strategy)

## Recommended reading order

### Step 1 — Sanity-check the run (5 min)

Open `responses.parquet` and check three things:

- **Parse rate by strategy is reasonable.** `structured_json` should be
  ≈ 1.0. The others should be ≥ 0.9. If anything is below 0.7, the
  strategy's parser is broken and that strategy's accuracy numbers are
  unreliable.
- **No strategy has > 5% errors.** Inspect a few `response` strings for
  strategies with high error rates — usually it's a token-limit issue
  (response truncated before reaching the "Answer: ..." line).
- **n_calls_made matches expectation.** Group by strategy, take mean
  `n_calls_made`. Should be 1.0 / 2.0 / 5.0 as documented in the strategies
  module. Drift suggests partial failures in multi-call strategies.

### Step 2 — Headline question: did any strategy beat the baseline? (10 min)

The `09_prompting_strategies.png` figure shows accuracy bars per strategy,
faceted by `with_kg` vs `no_kg`. Three reading questions:

- **Which strategies beat `zero_shot_direct` by more than the CI?** Those
  are the actionable wins.
- **Do the wins transfer across `with_kg` and `no_kg`?** If a strategy
  helps only with KG context, the win is conditional on retrieval quality.
  If it helps in both, it's a robust prompting win.
- **Is the lift larger or smaller than the KG-condition lift (with_kg vs
  no_kg)?** This puts prompting effects on the same scale as KG effects.
  If CoT lifts accuracy by 5 points but `with_kg` lifts by 10 points, the
  KG is doing more work than the prompt.

### Step 3 — Calibration question: is the model honest about uncertainty? (10 min)

The `09_prompting_calibration.png` figure shows accuracy at each
confidence level, per strategy. A well-calibrated model has accuracy
~0.20 at conf=1 and ~1.0 at conf=5 (the dashed diagonal).

- **Which strategies sit near the diagonal?** `verbalized_prob` and
  `few_shot_3` should — the others probably hug the top of the plot
  (conf=5 always, regardless of correctness).
- **If `structured_json` is well-calibrated**, the forced reasoning +
  contradictions field is doing useful work beyond just constraining the
  format.

### Step 4 — Per-stratum breakdown (15 min)

The existing cell after the new ones (originally cell 16) has
`per-stratum` accuracy. Filter to the new strategies and check:

- **`pos_phase12` accuracy.** This is the stratum that should benefit
  most from reasoning (Phase 1-2 drugs are non-canonical, so name-priors
  alone won't carry the day). If CoT / step_back / prompt_then_verify
  lift this stratum specifically, the reasoning hypothesis is confirmed.
- **`neg_plausible` accuracy.** This is the stratum that punishes
  superficial pattern-matching. Wins here would specifically validate
  `prompt_then_verify` (which is designed for this).

### Step 5 — Cost-adjusted ranking (5 min)

Build a single table: accuracy / call_count. The "best return per call"
strategy is usually a single-call CoT variant. The justification for
expensive strategies (`self_consistency_5`) has to be a substantial
absolute accuracy lift, not just any lift.

## How to present to the supervisor

**3-slide subsection within the weekly roundup:**

### Slide A — What we tested

One bullet per strategy bundle (reasoning / few-shot / robustness /
calibration), one line each. Don't enumerate all 8 — group them.

### Slide B — Headline result (the bar chart)

`09_prompting_strategies.png` full-bleed. One sentence: "Across 6 KGs,
strategy X gave the largest accuracy lift (Δ = ...), strategy Y the
best calibration." Bullet of the *interpretation*, not the numbers.

### Slide C — Where this goes next

Three bullets:

- The strategies that worked / didn't, in plain language.
- The most surprising result (if any) — write this *after* seeing the data;
  expect to fill it in last.
- One concrete follow-up: typically "rerun the winning strategy at full
  scale" or "try the same prompts on a stronger model to see if effects
  transfer."

## Pre-registering expected results (write these BEFORE running)

It's worth jotting down what you'd predict before looking at the parquet
— it sharpens the writeup and prevents post-hoc rationalization. The
five questions below are aligned with the five strategies actually being
tested:

- I expect `structured_json` to beat `zero_shot_direct` on accuracy
  by **____** points (Paper 1's Hybrid-2 prediction).
- I expect `multi_expert_rot` to beat `zero_shot_cot` by **____**
  points (Paper 3's gpt-4-Web finding, scaled down for llama3.1:8b).
- I expect `cisc` to be the best-calibrated strategy of the five.
- I expect `cisc`'s accuracy lift over `zero_shot_cot` to be **____**
  points (Taubenfeld et al.'s CISC paper claims 40% reduction in
  required reasoning paths — what does that translate to for accuracy?).
- I expect `multi_expert_rot` may NOT beat `zero_shot_cot` on
  llama3.1:8b specifically — Wang et al. 2024 noted ROT's benefit was
  much smaller on weaker models. Confirming this would itself be a
  publishable finding.

After the run, cross-reference. The strategies that didn't go as
expected are the most interesting paragraphs in the writeup.
