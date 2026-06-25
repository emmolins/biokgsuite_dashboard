#!/usr/bin/env python3
"""Assemble eval_notebooks/09a_selection_bias_audit.ipynb (generic, figures embedded)."""
import os, json, base64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'results', 'figures')
OUT = os.path.join(ROOT, 'eval_notebooks', '09a_selection_bias_audit.ipynb')

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}
def code(src, *imgs):
    outs = []
    for f in imgs:
        p = os.path.join(FIGS, f)
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, 'rb').read()).decode()
            outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": b64}})
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": outs,
            "source": [l + "\n" for l in src.split("\n")]}

C = []
C.append(md(
    "# 09a · Selection-bias audit",
    "",
    "A **reusable** check on whether a filtered/kept subset differs systematically from the records",
    "that were dropped — i.e. whether the selection introduces bias that limits what the final set",
    "generalises to.",
    "",
    "It is dataset-agnostic: supply a table with a boolean `kept` column, list the **categorical** and",
    "**continuous** covariates to test, and the notebook runs the statistics (Fisher odds ratios for",
    "categorical traits, Mann–Whitney for continuous ones) and draws three figures. Only the *Setup*",
    "cell is specific to a given dataset; everything below it is generic.",
))

C.append(md(
    "## Setup & configuration",
    "",
    "Edit this cell for your data. You need: a dataframe `df`, a boolean column `kept`, and two lists —",
    "`CAT_VARS` (categorical covariates) and `CONT_VARS` (continuous covariates). The example below",
    "loads the drug-repurposing benchmark and derives a few covariates, but any table works."))
C.append(code(
    "import os, re, numpy as np, pandas as pd\n"
    "from IPython.display import display\n"
    "import sys; sys.path.insert(0, os.path.abspath('..'))\n"
    "from src.bias_audit import run_bias_tests, plot_forest, plot_distribution, plot_composition\n"
    "%matplotlib inline\n"
    "\n"
    "BASE = os.path.abspath('..'); GS = os.path.join(BASE, 'data', 'gold_standards')\n"
    "\n"
    "# ----- load data + define `kept` (DATASET-SPECIFIC) ---------------------------------\n"
    "df    = pd.read_csv(os.path.join(GS, 'pairs_with_ids_v2.csv'))             # all candidates\n"
    "final = pd.read_csv(os.path.join(GS, 'gold_standard_v4_bigset.tsv'), sep='\\t')  # kept subset\n"
    "def _n(x): return str(x).strip().lower()\n"
    "def _ing(s): return set(t.strip() for t in re.split(r'[+/]', str(s).lower()) if t.strip())\n"
    "_gi = [(_n(r.disease_name), _ing(r.drug_name)) for _, r in final.iterrows()]\n"
    "df['_ing'] = df.Active.map(_ing)\n"
    "df['kept'] = df.apply(lambda c: any(dn == _n(c.NewIndication) and (gi & c['_ing']) for dn, gi in _gi), axis=1)\n"
    "\n"
    "# ----- feature engineering (DATASET-SPECIFIC) -------------------------------------\n"
    "def modality(row):\n"
    "    s = (str(row.Active) + ' ' + str(row.Drug)).lower()\n"
    "    if re.search(r'cabtagene|leucel|car-t', s): return 'Cell / gene therapy'\n"
    "    if 'vaccine' in s: return 'Vaccine'\n"
    "    if re.search(r'mab\\b|umab|zumab|ximab|cept\\b', s): return 'Antibody / biologic'\n"
    "    if re.search(r'tide\\b|glutide|parin|tatercept', s): return 'Peptide / other'\n"
    "    return 'Small molecule'\n"
    "df['Modality']            = df.apply(modality, axis=1)\n"
    "df['Therapeutic area']    = df.TA_simplified\n"
    "df['Approval year']       = df.YearSet.astype(str)\n"
    "df['Drug approval year']  = pd.to_numeric(df.OrigYear, errors='coerce')\n"
    "\n"
    "# ----- what to test (edit) --------------------------------------------------------\n"
    "CAT_VARS  = ['Modality', 'Therapeutic area', 'Approval year']\n"
    "CONT_VARS = ['Drug approval year']\n"
    "# optional: named binary traits for the forest plot (any boolean masks)\n"
    "TRAITS = {\n"
    "    'Older drug (\\u22642015)':    df['Drug approval year'] <= 2015,\n"
    "    'Novel modality':            df.Modality.isin(['Cell / gene therapy', 'Vaccine']),\n"
    "    'Antibody / biologic':       df.Modality.str.contains('Antibody'),\n"
    "    'Oncology':                  df['Therapeutic area'] == 'Oncology',\n"
    "    'Earliest year-set':         df['Approval year'] == sorted(df['Approval year'].unique())[0],\n"
    "}"))

C.append(md("## 1 · Kept vs dropped"))
C.append(code(
    "n_kept = int(df.kept.sum()); n_drop = int((~df.kept).sum())\n"
    "print(f'{len(df)} records  ->  {n_kept} kept  |  {n_drop} dropped')"))

C.append(md(
    "## 2 · Bias tests",
    "",
    "One-vs-rest **odds ratio** (of being kept) for each categorical level with n \\u2265 `min_n`, and a",
    "**Mann–Whitney** test for each continuous covariate. A trait whose 95% CI excludes 1 (or whose",
    "p < 0.05) is over- or under-represented in the kept set; everything else is balanced."))
C.append(code(
    "results = run_bias_tests(df, 'kept', CAT_VARS, CONT_VARS, min_n=8)\n"
    "results"))

C.append(md(
    "## 3 · Figures",
    "",
    "All three are regenerated from `df` on every run (the cells below show an example render)."))
C.append(md("### 3a · Bias scorecard — odds of being kept per trait"))
C.append(code("fig = plot_forest(df, 'kept', TRAITS); display(fig)", "bias_audit_forest.png"))
C.append(md("### 3b · Continuous covariate — kept vs dropped distribution"))
C.append(code("for v in CONT_VARS:\n    display(plot_distribution(df, 'kept', v))", "bias_audit_dist.png"))
C.append(md("### 3c · Composition of kept vs dropped across categorical covariates"))
C.append(code("fig = plot_composition(df, 'kept', CAT_VARS); display(fig)", "bias_audit_composition.png"))

C.append(md(
    "## 4 · Reading the results",
    "",
    "- **Forest plot:** dots on the *no-bias* line (OR = 1) mean the trait is equally represented in",
    "  kept and dropped. A dot whose CI sits entirely left of 1 is **under-represented** in the kept set;",
    "  entirely right means **over-represented**. Coloured = significant (Fisher p < 0.05).",
    "- **Distribution:** if the medians separate and the Mann–Whitney p is small, the kept set is shifted",
    "  on that continuous variable.",
    "- **Composition:** panels whose left (dropped) and right (kept) bars mirror each other are balanced;",
    "  a panel whose bars differ marks the covariate that the selection skews.",
    "",
    "Report any significant covariate as a bound on what the kept set generalises to, and note whether",
    "the skew is *incidental* or a *by-design* consequence of the selection rule.",
))

nb = {"cells": C, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w'), indent=1)
print('wrote', OUT, '| cells:', len(C), '| KB:', round(os.path.getsize(OUT)/1024))
