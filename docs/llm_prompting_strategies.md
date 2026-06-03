# LLM prompting for drug-disease plausibility (nb09)

nb09 poses each stratified drug-disease pair to a local LLM (`llama3.1:8b` via
Ollama) under three KG-context conditions and reports per-(KG, condition) AUROC
with analytic 95% confidence intervals (Hanley-McNeil; the data is a single run,
so cross-run resampling is not available and bootstrap is avoided).
The headline question is whether knowledge-graph context (direct edges, then
mechanistic paths) improves the model's plausibility judgments, and which KG
helps most. See the nb09 notebook header for the full design.

## The prompt (`llm_prompt`)

A single prompting strategy is used, defined as `LLMPrompt` in
`src/prompting_strategies.py` and selected with `STRATEGY = 'llm_prompt'`. It is
one JSON-formatted call per cell:

```
You are a drug repurposing expert. Below is knowledge-graph context about a
drug, followed by a yes/no question.

<kg_block>            # empty for C0; direct edges for C1; mechanistic paths for C2

Question: Is <drug> a plausible treatment for <disease>?

Respond ONLY with a JSON object of the form:
{ "answer": "Yes" | "No", "confidence": <integer 1-5>, "reasoning": "<one sentence>" }
```

- Decoding: `temperature=0.0`, `max_tokens=200`, Ollama `format='json'` to
  enforce JSON output (near-100% parse rate).
- Parsed into `(pred, confidence)`; the AUROC score per pair is
  `pred * (confidence / 5)`.
- The only thing that changes across conditions is the `kg_block` content, so
  the prompt isolates the effect of KG context rather than prompt wording.

The design combines a plain yes/no question (the "input-output" baseline,
Sivarajkumar et al. 2024) with JSON-schema enforcement plus a one-sentence
rationale (the structured-output line of DrugReX 2025 and DrugReAlign 2024). The
`contradictions` field from the original structured schema was dropped to keep
the prompt lean.

## Prompt-strategy comparison (retired)

An earlier version of nb09 compared several prompting strategies head-to-head
(zero-shot direct, chain-of-thought, structured JSON, multi-expert reflection,
confidence-informed self-consistency, and others). That comparison was retired
in favour of the single `llm_prompt` strategy above, so the analysis isolates
KG context rather than prompt phrasing. The earlier strategy library and its
literature rationale remain in the git history if that line of work is revived.
