# results/

Outputs from the BioKGSuite benchmark notebooks. `tables/` and `figures/` are
tracked in git; `checkpoints/` and `cache/` are gitignored and regenerable.

## Layout

```
results/
├── README.md
├── tables/                       ← all tabular outputs (.csv, .tsv, .md, .json, .parquet)
│   ├── 00_benchmark_summary.csv  ← headline benchmark scores (from nb00)
│   ├── NN_*.{csv,md,tsv,json}    ← per-notebook outputs, prefixed by notebook number
│   └── 09_llm_runs/              ← nb09 pair + response tables (pairs, responses_layered, ranking)
├── figures/                   ← all charts (.pdf, .png), prefixed by NN_*
├── checkpoints/               ← per-notebook .pkl files consumed by nb00 for the headline summary
└── cache/                     ← gitignored intermediates (regenerable from the source KGs)
```

## What's tracked vs gitignored

| Directory          | Tracked in git? | Why |
|--------------------|-----------------|-----|
| `tables/`          | yes             | Small, valuable, expensive to regenerate (e.g. LLM API calls) |
| `figures/`         | yes             | Final visual outputs the README and dashboard link to |
| `checkpoints/`     | no              | Regenerable from raw data + notebooks |
| `cache/`           | no              | Intermediate KG/embedding caches (multi-GB) |

## Regenerating outputs

Run the notebooks in order:
```bash
jupyter nbconvert --to notebook --execute eval_notebooks/01_coverage.ipynb \
  --output 01_coverage.ipynb
# ... repeat for 02 through 08, then 00 for the summary
```

Each notebook regenerates its own `figures/NN_*` and `tables/NN_*` outputs idempotently.

`nb09_llm_integration.ipynb` requires a running Ollama server (`llama3.1:8b`) and writes
to `tables/09_llm_runs/` and `figures/09_*.png`.

## Naming convention

- Tables: `tables/{nb#}_{description}.{csv,tsv,md,json,parquet}`
- Figures: `figures/{nb#}{optional letter}_{description}.{pdf,png}`
- Checkpoints: `checkpoints/{nb#}_{dimension}.pkl`

Sub-notebook outputs (e.g. four panels within nb04) use lowercase suffix letters:
`04_topology.pdf`, `04c_known_pair_recovery.pdf`, `04d_differential_resilience.pdf`.
