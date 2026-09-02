#!/usr/bin/env python3
"""Build eval_notebooks/09c_scaling_sweep.ipynb (live calcs, figures embedded).

Same-family Llama 3.x capability-scaling sweep: does the KG lift grow/shrink/stay
flat from 1B -> 405B?  Plotting + stats live in src/scaling_sweep.py; this notebook
loads whatever ladder rungs have been run so far and renders the two figures.
Fills in automatically as more HPC jobs land — no hardcoded numbers.
"""
import os, sys, json, base64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGDIR = os.path.join(ROOT, 'results', 'figures', '09c_scaling')
OUT = os.path.join(ROOT, 'eval_notebooks', '09c_scaling_sweep.ipynb')

from src.scaling_sweep import (load_llama_runs, compute_scaling, fig_lift_vs_size,
                               fig_mrr_vs_size, fig_kg_by_model, kg_by_model_table)
os.makedirs(FIGDIR, exist_ok=True)
_df = load_llama_runs(os.path.join(ROOT, 'results', 'tables', '09_llm_runs'))
_cov = os.path.join(ROOT, 'data', 'gold_standards', 'coverage_annotation_v4.csv')
_s = compute_scaling(_df, _cov)
_FIGS = {'lift_vs_size': fig_lift_vs_size(_s), 'mrr_vs_size': fig_mrr_vs_size(_s),
         'kg_by_model': fig_kg_by_model(_df)}
for n, f in _FIGS.items():
    f.savefig(os.path.join(FIGDIR, n + '.png'), dpi=200, bbox_inches='tight', facecolor='white')
    f.savefig(os.path.join(FIGDIR, n + '.pdf'), bbox_inches='tight', facecolor='white')

def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in l]}
def code(s, img=None):
    outs = []
    if img and os.path.exists(os.path.join(FIGDIR, img)):
        outs.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(open(os.path.join(FIGDIR, img), 'rb').read()).decode()}})
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": outs,
            "source": [x + "\n" for x in s.split("\n")]}

C = []
C.append(md(
    "# 09c · Same-family scaling sweep (Llama 3.x)",
    "",
    "Does the KG lift (MRR$_{KG}$ − MRR$_{no\\,KG}$) **grow, shrink, or stay flat** as the model",
    "scales 1B → 405B?  A within-family ladder isolates the *capability* axis — architecture and",
    "training recipe held roughly fixed — which the cross-family GPT/Gemini/Llama comparison can't.",
    "",
    "**Ladder & a caveat:** no clean single-generation 5-point ladder exists in Llama, so the rungs",
    "span generations — 1B/3B = Llama 3.2, 8B = 3.1, 70B = 3.3, 405B = 3.1. Generation is carried",
    "through and drawn as the marker shape so the confound is visible; the Llama-3.1 spine (8B→405B)",
    "is the clean within-generation read.",
    "",
    "**Memorization confound (the whole game):** bigger/newer models memorise more drug–disease",
    "pairs, inflating the no-KG baseline and *mechanically* shrinking apparent lift. So every figure",
    "splits the KG arm into **covered** (the graph actually contains the pair) vs **all arms pooled**.",
    "All numbers are computed live; the notebook fills in as more rungs finish on HPC.",
))
C.append(md("## Setup"))
C.append(code(
    "import os, sys, importlib\n"
    "from IPython.display import display\n"
    "%matplotlib inline\n"
    "BASE = os.getcwd()\n"
    "while not os.path.exists(os.path.join(BASE, 'src', 'scaling_sweep.py')) and os.path.dirname(BASE) != BASE:\n"
    "    BASE = os.path.dirname(BASE)\n"
    "sys.path.insert(0, BASE)\n"
    "import src.scaling_sweep as _ss; importlib.reload(_ss)   # pick up edits without a kernel restart\n"
    "from src.scaling_sweep import (load_llama_runs, compute_scaling, fig_lift_vs_size,\n"
    "    fig_mrr_vs_size, fig_kg_by_model, kg_by_model_table)\n"
    "\n"
    "RUNS = os.path.join(BASE, 'results', 'tables', '09_llm_runs')\n"
    "COV  = os.path.join(BASE, 'data', 'gold_standards', 'coverage_annotation_v4.csv')\n"
    "df = load_llama_runs(RUNS)\n"
    "print('rungs run so far:', sorted(df.model.unique()))"))

C.append(md(
    "## Scaling table",
    "",
    "Per model: MRR on each arm, and the KG lift (pooled and covered-only) with a 95% **cluster",
    "bootstrap** CI over the 116 diseases — the same unit used elsewhere in the study."))
C.append(code(
    "s = compute_scaling(df, COV)\n"
    "s[['label','params','gen','mrr_nokg','mrr_kg','mrr_kg_cov',\n"
    "   'lift','lift_lo','lift_hi','lift_cov','lift_cov_lo','lift_cov_hi','n_dis']].round(3)"))

C.append(md(
    "### Covered vs pooled — side by side",
    "",
    "The KG arm two ways for every model: **pooled** (all 3 KG arms, including pairs the graph",
    "doesn't contain) vs **covered only** (restricted to cells where the KG actually holds the pair).",
    "`gap = covered − pooled` is the **empty-arm penalty** — how much the pooled number is dragged",
    "down by coverage holes rather than by the model failing to use evidence that's present."))
C.append(code(
    "cmp = s[['label','params','mrr_nokg','mrr_kg','mrr_kg_cov','lift','lift_cov']].copy()\n"
    "cmp['mrr_gap']  = cmp['mrr_kg_cov'] - cmp['mrr_kg']      # covered - pooled (MRR)\n"
    "cmp['lift_gap'] = cmp['lift_cov']   - cmp['lift']        # covered - pooled (lift)\n"
    "cmp = cmp.round(3)\n"
    "cmp.columns = ['model','params(B)','MRR no-KG','MRR KG pooled','MRR KG covered',\n"
    "               'lift pooled','lift covered','MRR gap (cov-pool)','lift gap (cov-pool)']\n"
    "cmp.reset_index(drop=True)"))

C.append(md(
    "## Figures",
    "",
    "**1 · KG lift vs model size** — the headline. Pooled (grey) vs covered-arm (green) lift across",
    "the ladder, error bars = 95% cluster bootstrap CI, marker shape = Llama generation. Read the",
    "*covered* line for the capability effect with memorization netted out; if pooled and covered",
    "diverge at the small end, that gap is the empty-arm/memorization artifact, not capability."))
C.append(code("fig_lift_vs_size(s);", "lift_vs_size.png"))
C.append(md(
    "**2 · KG context by graph and model** — the pooled KG arm broken out into the three individual",
    "graphs (PrimeKG, DRKG, BioKG) next to the no-KG baseline, across the ladder. Shows whether one",
    "graph drives the lift and how each KG's benefit scales with model size."))
C.append(code("display(kg_by_model_table(df).round(3))\nfig_kg_by_model(df);", "kg_by_model.png"))
C.append(md(
    "**3 · Where the lift comes from** — no-KG (prior-only) vs KG-arm MRR by size. A **rising no-KG",
    "line** with scale is the memorization signal (bigger models recall more pairs unaided); the gap",
    "to the KG line is the lift."))
C.append(code("fig_mrr_vs_size(s);", "mrr_vs_size.png"))

C.append(md(
    "## How to read the result",
    "",
    "- **Lift grows with size** → bigger models exploit the KG *better* (the graph compounds capability).",
    "- **Lift shrinks with size** → capability substitutes for the graph (the 'crutch' story) — but first",
    "  rule out memorization by checking the covered line and the no-KG trend in Figure 2.",
    "- **Lift flat** → KG value is capability-independent — the strongest robustness claim for the method.",
    "",
    "Compare the **pooled vs covered** lines: a pooled-only trend that vanishes on the covered line is an",
    "evidence-availability/memorization artifact, not a real capability effect.",
    "",
    "*Engine: `src/scaling_sweep.py`. Runs produced by `scripts/hpc/run_llm_ladder.sbatch` (1B/3B/8B/70B)",
    "and `scripts/hpc/run_llm_405b.sbatch` (405B).*"))

nb = {"cells": C, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w'), indent=1)
print('wrote', OUT, '| cells:', len(C), '| embedded figs:', sum(len(c.get('outputs', [])) for c in C), '| KB:', round(os.path.getsize(OUT) / 1024))
