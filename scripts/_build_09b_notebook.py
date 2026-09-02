#!/usr/bin/env python3
"""Assemble eval_notebooks/09b_headline_figures.ipynb (imports src.headline_figures, embeds renders)."""
import os, json, base64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'results', 'figures', '09_headline')
OUT = os.path.join(ROOT, 'eval_notebooks', '09b_headline_figures.ipynb')

def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in l]}
def code(src, img=None):
    outs = []
    if img:
        p = os.path.join(FIGS, img)
        if os.path.exists(p):
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(open(p, 'rb').read()).decode()}})
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": outs,
            "source": [x + "\n" for x in src.split("\n")]}

C = []
C.append(md(
    "# 09b · Headline LLM × KG figures",
    "",
    "The six main figures of the LLM × KG drug-ranking study, regenerated from the current",
    "`results/tables/09_llm_runs/09_big_*.csv` on every run. Plotting lives in",
    "`src/headline_figures.py`; this notebook just loads the runs and calls each figure.",
))
C.append(md("## Setup"))
C.append(code(
    "import os, sys; sys.path.insert(0, os.path.abspath('..'))\n"
    "from IPython.display import display\n"
    "from src.headline_figures import (load_runs, fig_kg_lift_bars, fig_lift_box,\n"
    "    fig_rank_distribution, fig_dim_tracks_lift, fig_lift_vs_coverage, fig_evidence_quality)\n"
    "%matplotlib inline\n"
    "\n"
    "BASE = os.path.abspath('..')\n"
    "RUNS = os.path.join(BASE, 'results', 'tables', '09_llm_runs')\n"
    "SUMMARY = os.path.join(BASE, 'results', 'tables', '00_benchmark_summary.csv')\n"
    "OUT = os.path.join(BASE, 'results', 'figures', '09_headline'); os.makedirs(OUT, exist_ok=True)\n"
    "\n"
    "df = load_runs(RUNS)\n"
    "print(df.Model.value_counts().to_dict(), '| diseases:', df.disease.nunique())"))

sections = [
    ("## 1 · KG lift by graph and model", "fig = fig_kg_lift_bars(df)", "fig1_kg_lift_bars"),
    ("## 2 · Per-disease lift (faceted by LLM)", "fig = fig_lift_box(df)", "fig2_lift_box"),
    ("## 3 · Rank-position distribution", "fig = fig_rank_distribution(df)", "fig3_rank_distribution"),
    ("## 4 · Which quality dimension tracks lift?", "fig = fig_dim_tracks_lift(df, SUMMARY)", "fig4_dim_tracks_lift"),
    ("## 5 · Lift vs coverage, faceted by LLM", "fig = fig_lift_vs_coverage(df, SUMMARY)", "fig5_lift_vs_coverage"),
    ("## 6 · The KG lift is an evidence-quality effect", "fig = fig_evidence_quality(df)", "fig6_evidence_quality"),
]
for title, call, name in sections:
    C.append(md(title))
    C.append(code(f"{call}\nfig.savefig(os.path.join(OUT, '{name}.png'), bbox_inches='tight', facecolor='white')\n"
                  f"fig.savefig(os.path.join(OUT, '{name}.pdf'), bbox_inches='tight', facecolor='white')\ndisplay(fig)", name + ".png"))

C.append(md(
    "---",
    "*To regenerate every figure to disk at once:* "
    "`from src.headline_figures import regenerate; regenerate(RUNS, SUMMARY, OUT)`."))

nb = {"cells": C, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w'), indent=1)
print('wrote', OUT, '| cells:', len(C), '| KB:', round(os.path.getsize(OUT) / 1024))
