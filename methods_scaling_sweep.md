# Same-family scaling sweep — Llama 3.x (1B → 405B)

**Question.** Does the KG lift (MRR_KG − MRR_no-KG) grow, shrink, or stay flat as the model scales?
A within-family ladder isolates the *capability* axis (architecture / training recipe held roughly
fixed), which the cross-family GPT/Gemini/Llama comparison cannot.

## The ladder

| Rung | Ollama tag | Generation | Status |
|---|---|---|---|
| 1B | `llama3.2:1b` | 3.2 | to run |
| 3B | `llama3.2:3b` | 3.2 | to run |
| 8B | `llama3.1:8b` | 3.1 | to run |
| 70B | `llama3.3:70b` | 3.3 | ✅ done (`09_big_llama*.csv`) |
| 70B (spine) | `llama3.1:70b` | 3.1 | optional — clean within-gen 70B for the 3.1 spine |
| 405B | `llama3.1:405b` | 3.1 | to run (multi-GPU) |

**Generation caveat.** No clean single-generation 5-point ladder exists in Llama: 1B/3B are 3.2,
8B is 3.1, the done 70B is 3.3, 405B is 3.1. So a raw size curve confounds size with generation.
Mitigation: generation is carried in the data and drawn as the **marker shape**; the **Llama-3.1
spine (8B → [70B] → 405B)** is the clean within-generation read. Running `llama3.1:70b` as well
gives a fully within-3.1 three-point spine (8B/70B/405B) alongside the full mixed-gen ladder.

## How to run

Small/mid rungs — one job per (model, seed), single H100, run in parallel:

```bash
cd ~/biokgsuite
for M in llama3.2:1b llama3.2:3b llama3.1:8b; do
  for S in 1 2 3; do
    sbatch --export=ALL,MODEL=$M,SEED=$S scripts/hpc/run_llm_ladder.sbatch
  done
done
# optional clean-spine 70B:
for S in 1 2 3; do sbatch --export=ALL,MODEL=llama3.1:70b,SEED=$S scripts/hpc/run_llm_ladder.sbatch; done
```

Top rung — 405B, 4×H100, one seed per job (long):

```bash
# pre-flight: df -h "$OLLAMA_MODELS"  (need >=250 GB free) ; squeue  (4-GPU node free?)
for S in 1 2 3; do sbatch --export=ALL,SEED=$S scripts/hpc/run_llm_405b.sbatch; done
```

Each job writes `results/tables/09_llm_runs/09_big_<safe-tag>_s<seed>.csv` (e.g.
`09_big_llama3_1_8b_s1.csv`). The analysis globs `09_big_*.csv` and filters by the `model` column,
so filenames only need to be unique — nothing else to wire.

## Expected wall-clock (anchored to the done 70B run ≈ 5h/seed on 1×H100)

| Rung | ~per-seed | Notes |
|---|---|---|
| 1B | ~20 min | trivial |
| 3B | ~35 min | trivial |
| 8B | ~55 min | trivial |
| 70B | ~5 h | already done |
| 405B | hours–day+ | 4-GPU queue + ~230 GB pull + slow layer-split inference |

Submitted in parallel, the small rungs all land within ~1 h + queue. 405B is the long pole; for
real throughput swap Ollama for vLLM tensor-parallel (TP=4).

## Analysis & figures

`src/scaling_sweep.py` is the single source of truth (no hardcoded numbers):

- `load_llama_runs(runs_dir)` — union of `09_big_*.csv`, parsed rows, restricted to the Llama ladder.
- `compute_scaling(df, cov_csv)` — per-model MRR on each arm + KG lift (pooled and **covered-only**),
  with 95% **cluster bootstrap** CIs over the 116 diseases.
- `fig_lift_vs_size(s)` — headline: lift vs size, pooled vs covered, marker = generation.
- `fig_mrr_vs_size(s)` — diagnostic: no-KG vs KG-arm MRR by size (rising no-KG line = memorization).

Rebuild the notebook (auto-embeds current figures, fills in as rungs finish):

```bash
python scripts/_build_09c_scaling.py   # -> eval_notebooks/09c_scaling_sweep.ipynb
```

## The confound to address in the writeup

Bigger/newer models memorise more drug–disease pairs, inflating the no-KG baseline and mechanically
shrinking apparent lift. A downward lift-vs-size trend in the **pooled** line is therefore ambiguous.
Resolve it with two checks already built into the figures: (1) the **covered-arm** lift (memorization
can't explain ranking on pairs the graph actually contains), and (2) the **no-KG MRR vs size** trend
in Figure 2 — if it rises with scale, that is the memorization channel, not reduced KG value.

## Reading the headline

- Lift **grows** with size → bigger models exploit the KG better.
- Lift **shrinks** → capability substitutes for the graph (rule out memorization first).
- Lift **flat** → KG value is capability-independent — strongest robustness claim for the method.
