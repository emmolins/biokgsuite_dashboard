#!/usr/bin/env python3
"""Build eval_notebooks/09b_hard_pairs.ipynb (live calcs, figures at the end, embedded)."""
import os, sys, json, base64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGDIR = os.path.join(ROOT, 'results', 'figures', '09b_hard_pairs')
OUT = os.path.join(ROOT, 'eval_notebooks', '09b_hard_pairs.ipynb')

# 1) (re)generate the five figures from current data so embedded previews are fresh
from src.headline_figures import load_runs
from src.hard_pairs import (compute_difficulty, fig_intrinsic_difficulty, fig_hardest_pairs,
                            fig_difficulty_drivers, fig_difficulty_by_area, fig_coverage_fair, fig_age_effect)
os.makedirs(FIGDIR, exist_ok=True)
_df = load_runs(os.path.join(ROOT, 'results', 'tables', '09_llm_runs'))
_per = compute_difficulty(_df, os.path.join(ROOT, 'data', 'gold_standards', 'gold_standard_v4_bigset.tsv'),
                          os.path.join(ROOT, 'data', 'gold_standards', 'usable_pairs_v2_audited.csv'))
_COV = os.path.join(ROOT, 'data', 'gold_standards', 'coverage_annotation_v4.csv')
_FIGS = {'intrinsic_difficulty': fig_intrinsic_difficulty(_per), 'hardest_pairs': fig_hardest_pairs(_per),
         'difficulty_drivers': fig_difficulty_drivers(_per), 'difficulty_by_area': fig_difficulty_by_area(_per),
         'coverage_fair': fig_coverage_fair(_df, _COV), 'age_effect': fig_age_effect(_per)}
for n, f in _FIGS.items():
    f.savefig(os.path.join(FIGDIR, n + '.png'), dpi=200, bbox_inches='tight', facecolor='white')
    f.savefig(os.path.join(FIGDIR, n + '.pdf'), bbox_inches='tight', facecolor='white')

# 2) assemble notebook
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
    "# 09b · Which drug–disease pairs rank poorly?",
    "",
    "Per-pair ranking difficulty on the KG arm, and what drives it. All numbers are computed live",
    "from the compiled runs; the figures are generated in the final section.",
))
C.append(md("## Setup"))
C.append(code(
    "import os, sys\n"
    "from IPython.display import display\n"
    "%matplotlib inline\n"
    "BASE = os.getcwd()\n"
    "while not os.path.exists(os.path.join(BASE, 'src', 'hard_pairs.py')) and os.path.dirname(BASE) != BASE:\n"
    "    BASE = os.path.dirname(BASE)\n"
    "sys.path.insert(0, BASE)\n"
    "from src.headline_figures import load_runs\n"
    "from src.hard_pairs import (compute_difficulty, cross_model_corr, fig_intrinsic_difficulty,\n"
    "    fig_hardest_pairs, fig_difficulty_drivers, fig_difficulty_by_area, fig_coverage_fair, fig_age_effect)\n"
    "\n"
    "GS = os.path.join(BASE, 'data', 'gold_standards')\n"
    "df  = load_runs(os.path.join(BASE, 'results', 'tables', '09_llm_runs'))\n"
    "per = compute_difficulty(df, os.path.join(GS, 'gold_standard_v4_bigset.tsv'),\n"
    "                         os.path.join(GS, 'usable_pairs_v2_audited.csv'))\n"
    "print(f'{len(per)} pairs · mean MRR on KG arm = {per.mrr.mean():.3f}')"))

C.append(md(
    "## Calculations",
    "",
    "**Is difficulty intrinsic?** Correlate per-pair MRR across the three models — a high",
    "correlation means hard pairs are hard for *every* model, not one model's quirk."))
C.append(code(
    "r_models = cross_model_corr(per)\n"
    "print(f'cross-model MRR correlation (avg pairwise) = {r_models:.2f}  ->  '\n"
    "      f'{\"intrinsic (hard for all models)\" if r_models > 0.6 else \"model-specific\"}')"))

C.append(md("**Hard vs easy pairs** — split into MRR tertiles and compare the model's confidence, "
            "the share of consistent KG evidence, and KG coverage."))
C.append(code(
    "q1, q2 = per.mrr.quantile([1/3, 2/3])\n"
    "hard, easy = per[per.mrr <= q1], per[per.mrr >= q2]\n"
    "import pandas as pd\n"
    "cmp = pd.DataFrame({\n"
    "    'hard third': [hard.conf.mean(), hard.frac_consistent.mean(), hard.n_kg.mean()],\n"
    "    'easy third': [easy.conf.mean(), easy.frac_consistent.mean(), easy.n_kg.mean()]},\n"
    "    index=['self-reported confidence (1-5)', 'share consistent KG evidence', '# KGs covering']).round(2)\n"
    "cmp"))

C.append(md("**What predicts difficulty?** Pearson correlation of per-pair MRR with each covariate, "
            "and mean MRR by drug modality and approval year-set. (Confidence and consistent-evidence "
            "are downstream of the KG; coverage / recency / modality are intrinsic pair attributes.)"))
C.append(code(
    "from scipy import stats as st\n"
    "rows = []\n"
    "for c, lab in [('n_kg', 'KG coverage (# KGs)'), ('orig', 'drug approval year (recency)'),\n"
    "               ('conf', 'self-reported confidence'), ('frac_consistent', 'consistent KG evidence')]:\n"
    "    s = per[[c, 'mrr']].dropna(); r, p = st.pearsonr(s[c], s.mrr)\n"
    "    rows.append({'covariate': lab, 'pearson_r': round(r, 2), 'p': round(p, 4), 'n': len(s)})\n"
    "display(pd.DataFrame(rows))\n"
    "print('\\nmean MRR by drug modality:'); print(per.groupby('modality').mrr.mean().round(3).to_string())\n"
    "print('\\nmean MRR by approval year-set:'); print(per.groupby('year').mrr.mean().round(3).to_string())"))

C.append(md("**The hardest pairs** (lowest mean MRR)."))
C.append(code("per[['disease', 'drug', 'area', 'mrr', 'hit1', 'n_kg']].head(15).reset_index(drop=True)"))

C.append(md(
    "## Figures",
    "",
    "Generated from `per` (re-run to refresh). Each shows once.",
    "",
    "**1 · Difficulty is intrinsic** — pairwise per-disease MRR between the three models with the "
    "y=x line. Points hug the diagonal (high r), so the same pairs are hard for every model — not a "
    "per-model quirk."))
C.append(code("fig_intrinsic_difficulty(per);", "intrinsic_difficulty.png"))
C.append(md("**2 · The hardest pairs** — pooled MRR (bar) with each model's MRR (dots); dots cluster, "
            "reinforcing that difficulty is shared across models."))
C.append(code("fig_hardest_pairs(per);", "hardest_pairs.png"))
C.append(md("**3 · Difficulty drivers** — per-pair MRR vs the model's confidence, coloured by the share "
            "of consistent KG evidence. The model is well-calibrated to its own difficulty."))
C.append(code("fig_difficulty_drivers(per);", "difficulty_drivers.png"))
C.append(md("**4 · By therapeutic area** — dot per disease, diamond = area mean, sorted easiest→hardest."))
C.append(code("fig_difficulty_by_area(per);", "difficulty_by_area.png"))
C.append(md(
    "**5 · Coverage effect, fairly attributed** — number of KGs covering the pair is the axis that *does* "
    "predict pooled difficulty, but most of that is an **empty-arm averaging artifact**. The pooled line "
    "(grey, all three KG arms) dives at 1-KG pairs only because two of the three arms are empty and score "
    "near zero. Restricting to the **covered arm** (green) — the graph that actually contains the pair — the "
    "curve is nearly flat (~0.70-0.75): when a KG covers a pair, the model ranks it well regardless of how "
    "many *other* graphs also cover it. The small residual at 1-KG reflects genuinely thinner evidence, not a "
    "coverage penalty."))
C.append(code("fig_coverage_fair(df, os.path.join(GS, 'coverage_annotation_v4.csv'));", "coverage_fair.png"))
C.append(md("**6 · Drug age (null)** — original approval year vs MRR; recency does **not** predict difficulty."))
C.append(code("fig_age_effect(per);", "age_effect.png"))

C.append(md(
    "## Takeaways",
    "",
    "- Difficulty is **intrinsic** — hard pairs are hard for all three models (see the cross-model",
    "  correlation above).",
    "- The model is **calibrated to difficulty**: hard pairs are exactly the low-confidence,",
    "  low-consistent-evidence ones.",
    "- The intrinsic attribute that predicts *pooled* difficulty is **KG coverage** — but most of that is",
    "  an empty-arm averaging artifact: on the covered arm the curve is nearly flat (Figure 5). Drug age,",
    "  modality and approval year-set are all near-null (table above).",
    "",
    "*Engine: `src/hard_pairs.py`. Per-pair table written to `results/tables/hard_pairs.csv` if you "
    "call `per.to_csv(...)`.*"))

nb = {"cells": C, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w'), indent=1)
print('wrote', OUT, '| cells:', len(C), '| embedded figs:', sum(len(c.get('outputs', [])) for c in C), '| KB:', round(os.path.getsize(OUT) / 1024))
