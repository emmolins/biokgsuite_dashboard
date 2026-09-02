#!/usr/bin/env python3
"""
Downstream LLM lift vs. KG quality — one series per LLM on a single axis.

x = KG benchmark quality (Overall Mean, 00_benchmark_summary.csv)
y = downstream lift = mean over diseases of (KG MRR - no-KG MRR), with SEM bars
Three LLMs drawn as distinct colour/marker series; faint vertical guides mark
each KG's quality so the (non-monotone) lift ordering is readable at a glance.

Usage:  python scripts/make_lift_vs_quality.py
Output: results/figures/lift_vs_quality.(png|pdf)
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
LAB = {'primekg': 'PrimeKG', 'drkg': 'DRKG', 'biokg': 'BioKG'}
MODEL_FILES = {
    'GPT-4.1-mini':          ['09_big_gpt.csv', '09_big_gpt_s12.csv'],
    'Gemini-3.1-flash-lite': ['09_big_glite.csv', '09_big_glite_s12.csv'],
    'Llama-3.3-70B':         ['09_big_llama.csv', '09_big_llama_s1.csv', '09_big_llama_s2.csv'],
}
# distinct, colour-blind-safe series per LLM
MC = {'GPT-4.1-mini': '#0F9D8C', 'Gemini-3.1-flash-lite': '#5B4FB8', 'Llama-3.3-70B': '#D1791F'}
MK = {'GPT-4.1-mini': 'o',       'Gemini-3.1-flash-lite': 's',       'Llama-3.3-70B': '^'}

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---- KG quality (x) ----
b = pd.read_csv(BSUM).set_index('KG')
qual = {kg: float(b.loc[kg, 'Overall Mean']) for kg in KGS}

# ---- downstream lift (y) per model x KG ----
def load(parts):
    return pd.concat([pd.read_csv(os.path.join(RUNS, p)) for p in parts], ignore_index=True)
stats = {}   # model -> kg -> (mu, ci95_halfwidth)
for m, parts in MODEL_FILES.items():
    d = load(parts); d = d[d['parsed'] == 1].copy()
    d['arm'] = np.where(d['condition'] == 'no_kg', 'no_kg', d['kg'])
    nk = d[d.arm == 'no_kg'].groupby('disease')['reciprocal_rank'].mean()
    stats[m] = {}
    for kg in KGS:
        kk = d[d.arm == kg].groupby('disease')['reciprocal_rank'].mean()
        j = pd.concat([kk.rename('k'), nk.rename('n')], axis=1).dropna()
        diff = (j.k - j.n); n = len(diff)
        sem = diff.std(ddof=1) / np.sqrt(n)
        ci95 = st.t.ppf(0.975, n - 1) * sem          # paired 95% CI half-width
        stats[m][kg] = (diff.mean(), ci95)

# ---- plot ----
fig, ax = plt.subplots(figsize=(5.4, 4.3))
fig.subplots_adjust(left=0.135, right=0.97, top=0.86, bottom=0.135)

# no-lift baseline (y=0): the natural anchor for a difference axis
ax.axhline(0, color='#9a9a9a', lw=1.0, ls='--', zorder=1)
ax.text(0.612, 0.004, 'no-KG baseline (no lift)', fontsize=7, color='#888',
        ha='left', va='bottom')

# faint vertical guides + KG labels at top
for kg in KGS:
    ax.axvline(qual[kg], color='#e2e2e2', lw=0.9, ls='-', zorder=0)
    ax.text(qual[kg], 1.012, LAB[kg], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='#555')

order = sorted(KGS, key=lambda k: qual[k])          # ascending quality
for m in MODEL_FILES:
    xs = np.array([qual[k] for k in order])
    ys = np.array([stats[m][k][0] for k in order])
    es = [stats[m][k][1] for k in order]
    # one dashed light-grey least-squares fit (y = mx + b) per LLM
    b1, b0 = np.polyfit(xs, ys, 1)
    xl = np.array([xs.min(), xs.max()])
    ax.plot(xl, b1 * xl + b0, color='#bdbdbd', lw=1.1, ls='--', zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt=MK[m], ms=7.5, color=MC[m], mec='white', mew=0.7,
                ecolor=MC[m], elinewidth=1.1, capsize=3, zorder=3, label=m)

ax.set_xlabel('KG benchmark quality  (Overall Mean)', fontsize=9)
ax.set_ylabel('Downstream lift  (mean ΔMRR vs no-KG)', fontsize=9)
ax.set_xlim(0.61, 0.76)
ax.set_ylim(0, 0.66)
ax.set_yticks(np.arange(0, 0.61, 0.1))
ax.legend(loc='lower right', fontsize=8, handletextpad=0.4, borderaxespad=0.6,
          title='error bars = 95% CI', title_fontsize=7.5)
ax.set_title('Downstream lift vs. KG quality, by LLM', fontsize=10, fontweight='bold', pad=20)

out = os.path.join(FIGS, 'lift_vs_quality')
fig.savefig(out + '.png', bbox_inches='tight', facecolor='white')
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
print(f'Wrote {out}.png and .pdf')
