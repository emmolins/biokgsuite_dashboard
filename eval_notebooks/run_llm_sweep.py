#!/usr/bin/env python
"""Standalone BioKGSuite LLM ranking sweep — one model -> one CSV.
Runs ONLY the sweep (no exploratory trial/cap cells), reusing notebook 09's
tested functions. Usage (from the conda env):
    python eval_notebooks/run_llm_sweep.py --model groq:llama-3.3-70b-versatile
    python eval_notebooks/run_llm_sweep.py --model gpt-4.1-mini
    python eval_notebooks/run_llm_sweep.py --model gemini:gemini-3.5-flash
Config (KGs, gold file, N, seeds, shuffles) comes from cell 4 of the notebook;
override seeds/shuffles with --seeds/--shuffles if needed. Restartable: rerun
to recompute (FORCE_RERUN=True)."""
import os, sys, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, help="e.g. gpt-4.1-mini | gemini:gemini-3.5-flash | groq:llama-3.3-70b-versatile")
ap.add_argument('--out', help="output CSV name (default 09_ranking_<model>.csv)")
ap.add_argument('--seeds', default=None, help="comma seeds, e.g. 0 or 0,1")
ap.add_argument('--shuffles', default=None)
ap.add_argument('--n', default=None, help='limit N_DISEASES (for a quick smoke test)')
a = ap.parse_args()

os.environ['BKG_MODELS'] = a.model
if a.seeds:    os.environ['BKG_SEEDS'] = a.seeds
if a.shuffles: os.environ['BKG_SHUFFLES'] = a.shuffles
if a.n:        os.environ['BKG_NDISEASES'] = a.n
safe = a.model.replace(':', '_').replace('/', '_')
os.environ['BKG_OUT_CSV'] = a.out or f'09_ranking_{safe}.csv'

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)                       # so cell 2 resolves repo root
nb = json.load(open('09_llm_integration.ipynb'))

SETUP = [2, 4, 6, 8, 10, 12, 14, 16]   # imports/paths, config, prompt, parse, dispatch, helpers, kg-block, queries
RUN   = 22                              # the sweep loop
assert 'run_arm' in ''.join(nb['cells'][RUN]['source']), "cell 22 is not the run cell — check notebook structure"

g = {}
for i in SETUP:
    exec(''.join(nb['cells'][i]['source']), g)
print(f"[run] model={g['MODELS']}  KGS={g['KGS']}  gold={g['GOLD_FILE'].name}  "
      f"N={g['N_DISEASES']}  seeds={g['SEEDS']}  shuffles={g['SHUFFLES']}  -> {os.environ['BKG_OUT_CSV']}")
exec(''.join(nb['cells'][RUN]['source']), g)
print(f"[done] wrote {os.environ['BKG_OUT_CSV']}")
