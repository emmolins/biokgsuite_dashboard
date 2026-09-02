#!/usr/bin/env python3
"""
Dimension-ranking, split by level so it is apples-to-apples.

Left panel  : 7 aggregate [dim] roll-up scores
Right panel : 18 leaf metrics
Both ranked by Pearson r with mean downstream lift (n = 3 testable KGs).
Bars coloured by whether the column's KG ordering matches the lift ordering.
Columns with no variance across the 3 KGs (r undefined) are shown greyed at the
bottom labelled 'n/a' rather than silently dropped.  EXPLORATORY: n = 3.

Usage:  python scripts/make_dim_ranking_split.py
Output: results/figures/dimension_ranking_split.(png|pdf)
"""
import os, numpy as np, pandas as pd
from scipy import stats as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'tables', '09_llm_runs')
BSUM = os.path.join(ROOT, 'results', 'tables', '00_benchmark_summary.csv')
FIGS = os.path.join(ROOT, 'results', 'figures'); os.makedirs(FIGS, exist_ok=True)

KGS = ['primekg', 'drkg', 'biokg']
MIN_RANGE = 0.02
MODEL_FILES = {
    'GPT-4.1-mini':          ['09_big_gpt.csv', '09_big_gpt_s12.csv'],
    'Gemini-3.1-flash-lite': ['09_big_glite.csv', '09_big_glite_s12.csv'],
    'Llama-3.3-70B':         ['09_big_llama.csv', '09_big_llama_s1.csv', '09_big_llama_s2.csv'],
}
GREEN, GREY, NAGREY = '#2E8B57', '#c4c4c4', '#e8e8e8'

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---- mean lift per KG (avg over diseases then over LLMs) ----
def load(parts):
    return pd.concat([pd.read_csv(os.path.join(RUNS, p)) for p in parts], ignore_index=True)
perkg = {kg: [] for kg in KGS}
for m, parts in MODEL_FILES.items():
    d = load(parts); d = d[d['parsed'] == 1].copy()
    d['arm'] = np.where(d['condition'] == 'no_kg', 'no_kg', d['kg'])
    nk = d[d.arm == 'no_kg'].groupby('disease')['reciprocal_rank'].mean()
    for kg in KGS:
        kk = d[d.arm == kg].groupby('disease')['reciprocal_rank'].mean()
        j = pd.concat([kk.rename('k'), nk.rename('n')], axis=1).dropna()
        perkg[kg].append((j.k - j.n).mean())
meanlift = {kg: float(np.mean(v)) for kg, v in perkg.items()}
lift_order = sorted(KGS, key=lambda k: -meanlift[k])
y = np.array([meanlift[k] for k in KGS])

b = pd.read_csv(BSUM).set_index('KG')
metric_cols = [c for c in b.columns if '[dim]' not in c and c != 'Overall Mean']
agg_cols = [c for c in b.columns if '[dim]' in c]

def rows_for(cols):
    var, na = [], []
    for c in cols:
        x = b.loc[KGS, c].astype(float).values
        if np.isnan(x).any() or (x.max() - x.min()) < MIN_RANGE:
            na.append(c)
        else:
            r = st.pearsonr(x, y)[0]
            match = sorted(KGS, key=lambda k: -b.loc[k, c]) == lift_order
            var.append((c, r, match))
    var.sort(key=lambda t: t[1])          # ascending -> highest r at top
    return var, na

def draw(ax, cols, title):
    var, na = rows_for(cols)
    labels, yi = [], 0
    # n/a rows at the bottom
    for c in na:
        ax.barh(yi, 0, color='none'); ax.scatter(0, yi, marker='|', s=0)
        ax.text(0.02, yi, 'n/a — no variance across KGs', va='center', ha='left',
                fontsize=6.8, color='#aaa', style='italic')
        labels.append(c); yi += 1
    # variance rows
    for c, r, match in var:
        ax.barh(yi, r, color=GREEN if match else GREY, edgecolor='white', lw=0.5)
        labels.append(c); yi += 1
    ax.axvline(0, color='#666', lw=0.8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, len(labels) - 0.4); ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel('Pearson r with mean lift', fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    # grey the n/a labels (only non-bar styling that remains)
    for t in ax.get_yticklabels():
        if t.get_text() in na:
            t.set_color('#aaa')

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.4),
                               gridspec_kw=dict(width_ratios=[1, 1], wspace=0.55))
fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.10)
draw(axL, agg_cols, f'Aggregate dimensions ({len(agg_cols)})')
draw(axR, metric_cols, f'Leaf metrics ({len(metric_cols)})')

# shared legend + caveat
h1 = plt.Rectangle((0, 0), 1, 1, color=GREEN)
h2 = plt.Rectangle((0, 0), 1, 1, color=GREY)
fig.legend([h1, h2], ['KG order matches lift order', 'order differs'],
           loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.965), fontsize=8, handlelength=1.1)
fig.suptitle('Is KG quality correlated with downstream lift?',
             fontsize=11.5, fontweight='bold', y=0.995)

out = os.path.join(FIGS, 'dimension_ranking_split')
fig.savefig(out + '.png', bbox_inches='tight', facecolor='white')
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
print('wrote', out, '| lift order', lift_order)
