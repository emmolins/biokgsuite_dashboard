#!/usr/bin/env python3
"""
Render the LLM x KG main figure (Figure 1) from the 116-set runs.

Panel a = rank-position distribution (true drug ranked 1/2/3/4-8/missed), per model.
Panel b = per-disease KG lift: box + IQR + dots (KG colours) + '+mean - %helped'.

Auto-detects whichever of the three 09_big_* files exist, so you can run it
after gpt finishes and re-run as gemini / llama land. Columns = models, so it
goes 2 -> 3 wide automatically.

Usage:  python scripts/make_fig1_llm.py
Output: results/figures/fig1_llm_main.(png|pdf)
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'tables', '09_llm_runs')
FIGS = os.path.join(ROOT, 'results', 'figures', '09_llm_integration'); os.makedirs(FIGS, exist_ok=True)

# model file -> nice label (edit filenames here if yours differ)
MODEL_FILES = [
    ('09_big_gpt.csv',   'GPT-4.1-mini'),
    ('09_big_glite.csv', 'Gemini-3.1-flash-lite'),
    ('09_big_llama.csv', 'Llama-3.3-70B'),
]
KGS = ['primekg', 'drkg', 'biokg']
LAB = {'primekg': 'PrimeKG', 'drkg': 'DRKG', 'biokg': 'BioKG'}
C   = {'primekg': '#3B6FB6', 'drkg': '#E1812C', 'biokg': '#3A923A'}   # blue / orange / green
THR = 0.03   # |lift| below this = "no change" for the %helped count
B   = 3000

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---- load whichever model files are present + parsed ----
loaded = []
for fn, lab in MODEL_FILES:
    p = os.path.join(RUNS, fn)
    if not os.path.exists(p):
        print(f'[skip] {fn} not found'); continue
    x = pd.read_csv(p)
    x = x[x['parsed'] == 1].copy()
    if len(x) == 0:
        print(f'[warn] {fn} has 0 parsed rows; skipping'); continue
    x['label'] = lab
    pr = round(x['parsed'].mean(), 3) if 'parsed' in x else 'n/a'
    print(f'[ok]   {fn:18s} -> {lab:24s} rows={len(x)} parsed={pr}')
    loaded.append(x)
if not loaded:
    sys.exit('No 09_big_* files with parsed rows yet. Run the sweeps first.')
df = pd.concat(loaded, ignore_index=True)
df['arm'] = np.where(df['condition'] == 'no_kg', 'no_kg', df['kg'])
MODELS = [lab for _, lab in MODEL_FILES if lab in set(df['label'])]
nM = len(MODELS)

def pdz(m, a):
    return df[(df.label == m) & (df.arm == a)].groupby('disease')['reciprocal_rank'].mean()
def lifts(m, kg):
    nk = pdz(m, 'no_kg')
    j = pd.concat([pdz(m, kg).rename('k'), nk.rename('n')], axis=1).dropna()
    return (j['k'] - j['n']).values
rng = np.random.default_rng(5)

fig = plt.figure(figsize=(3.6 * nM, 5.0))
gs = fig.add_gridspec(2, nM, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.30,
                      left=0.09, right=0.97, top=0.90, bottom=0.10)

# ---------- Panel a: rank-position distribution ----------
BINS = ['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4-8', 'Missed']
BC = ['#176D3A', '#4FA96A', '#A9DBA8', '#F4C338', '#CB3B2E']   # greens -> yellow -> red
WHITE_ON = {'#176D3A', '#CB3B2E'}
arms = ['no_kg'] + KGS
ALAB = {'no_kg': 'No KG', **LAB}
def rbins(s):
    r = s['rank']
    return [(r == 1).mean(), (r == 2).mean(), (r == 3).mean(),
            ((r >= 4) & (r <= 8)).mean(), (r >= 9).mean()]
for ci, m in enumerate(MODELS):
    ax = fig.add_subplot(gs[0, ci])
    for i, a in enumerate(arms):
        vals = rbins(df[(df.label == m) & (df.arm == a)]); left = 0
        for v, c in zip(vals, BC):
            ax.barh(i, v, left=left, height=0.74, color=c, edgecolor='white', lw=0.6)
            if v > 0.07:
                ax.text(left + v / 2, i, f'{v*100:.0f}', ha='center', va='center',
                        fontsize=6.5, color='white' if c in WHITE_ON else '#333')
            left += v
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([ALAB[a] for a in arms], fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('Fraction of queries', fontsize=7.5)
    ax.set_title(m, fontsize=8.5, fontweight='bold', pad=4); ax.tick_params(labelsize=7)
    if ci == 0:
        ax.text(-0.32, 1.12, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BC]
fig.legend(handles, BINS, ncol=5, loc='upper center', bbox_to_anchor=(0.53, 0.965),
           fontsize=7, handlelength=1.1, columnspacing=1.2)

# ---------- Panel b: per-disease lift, box + dots + %helped ----------
for ci, m in enumerate(MODELS):
    ax = fig.add_subplot(gs[1, ci])
    ax.axvline(0, color='#c4c4c4', lw=0.9, ls='--', zorder=1)
    for j, kg in enumerate(KGS):
        y = 2 - j; v = lifts(m, kg)
        if len(v) == 0:
            continue
        ax.boxplot(v, positions=[y], vert=False, widths=0.46, patch_artist=True,
                   showfliers=False, zorder=2,
                   medianprops=dict(color='#1a1a1a', lw=1.3),
                   boxprops=dict(facecolor=C[kg], alpha=0.20, edgecolor=C[kg], lw=1.1),
                   whiskerprops=dict(color=C[kg], lw=1.1), capprops=dict(color=C[kg], lw=1.1))
        jit = y + np.random.default_rng(j).uniform(-0.17, 0.17, len(v))
        ax.scatter(v, jit, s=10, color=C[kg], alpha=0.7, edgecolor='white', linewidth=0.25, zorder=3)
        mu = v.mean(); ph = (v > THR).mean() * 100
        ax.text(0.655, y + 0.34, f'+{mu:.2f} - {ph:.0f}% helped', fontsize=7.0,
                color=C[kg], fontweight='bold', ha='right')
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([LAB[k] for k in KGS], fontsize=8)
    ax.set_ylim(-0.55, 2.6); ax.set_xlim(-0.55, 0.68)
    ax.set_xlabel('Δ MRR vs no-KG  (each dot = one disease)', fontsize=7.5)
    ax.set_title(m, fontsize=8.5, fontweight='bold', pad=4); ax.tick_params(labelsize=7.5)
    if ci == 0:
        ax.text(-0.32, 1.12, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

out = os.path.join(FIGS, 'fig1_llm_main')
fig.savefig(out + '.png', bbox_inches='tight', facecolor='white')
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
print(f'\nWrote {out}.png and .pdf  ({nM} model column(s))')
