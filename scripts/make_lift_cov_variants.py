#!/usr/bin/env python3
"""
Five alternative visualisations of the downstream-lift vs KG-coverage data.

v1 connected_scatter  - lift vs coverage, 3 LLM series + OLS fits + untestable strip
v2 grouped_bars       - KGs (sorted by coverage) x LLM grouped bars, 95% CI, untestable hatched
v3 heatmap            - LLM (rows) x KG (cols) lift cells + coverage strip on top
v4 model_centered     - lift minus each model's own mean -> isolates KG effect, single fit
v5 faceted            - one small panel per LLM, lift vs coverage with own fit

Usage:  python scripts/make_lift_cov_variants.py
Output: results/figures/cov_v{1..5}_*.png/.pdf
"""
import os, numpy as np, pandas as pd
from scipy import stats as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'tables', '09_llm_runs')
BSUM = os.path.join(ROOT, 'results', 'tables', '00_benchmark_summary.csv')
FIGS = os.path.join(ROOT, 'results', 'figures'); os.makedirs(FIGS, exist_ok=True)

KGS = ['primekg', 'drkg', 'biokg']
UNTESTABLE = ['openbilink', 'hetionet']
LAB = {'primekg': 'PrimeKG', 'drkg': 'DRKG', 'biokg': 'BioKG',
       'hetionet': 'HetioNet', 'openbilink': 'OpenBioLink'}
XDIM = 'Coverage [dim]'
MODEL_FILES = {
    'GPT-4.1-mini':          ['09_big_gpt.csv', '09_big_gpt_s12.csv'],
    'Gemini-3.1-flash-lite': ['09_big_glite.csv', '09_big_glite_s12.csv'],
    'Llama-3.3-70B':         ['09_big_llama.csv', '09_big_llama_s1.csv', '09_big_llama_s2.csv'],
}
MC = {'GPT-4.1-mini': '#0F9D8C', 'Gemini-3.1-flash-lite': '#5B4FB8', 'Llama-3.3-70B': '#D1791F'}
MK = {'GPT-4.1-mini': 'o', 'Gemini-3.1-flash-lite': 's', 'Llama-3.3-70B': '^'}
MODELS = list(MODEL_FILES)

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---------- data ----------
b = pd.read_csv(BSUM).set_index('KG')
cov = {kg: float(b.loc[kg, XDIM]) for kg in KGS + UNTESTABLE}
def load(parts):
    return pd.concat([pd.read_csv(os.path.join(RUNS, p)) for p in parts], ignore_index=True)
lift, ci = {}, {}      # [model][kg] -> mean, 95%CI halfwidth
for m, parts in MODEL_FILES.items():
    d = load(parts); d = d[d['parsed'] == 1].copy()
    d['arm'] = np.where(d['condition'] == 'no_kg', 'no_kg', d['kg'])
    nk = d[d.arm == 'no_kg'].groupby('disease')['reciprocal_rank'].mean()
    lift[m], ci[m] = {}, {}
    for kg in KGS:
        kk = d[d.arm == kg].groupby('disease')['reciprocal_rank'].mean()
        j = pd.concat([kk.rename('k'), nk.rename('n')], axis=1).dropna()
        diff = (j.k - j.n); n = len(diff)
        lift[m][kg] = diff.mean()
        ci[m][kg] = st.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
order_cov = sorted(KGS, key=lambda k: cov[k])          # ascending coverage

def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p + '.png', bbox_inches='tight', facecolor='white')
    fig.savefig(p + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig); print('wrote', name)

# ============ v1: connected scatter (reference) ============
fig, ax = plt.subplots(figsize=(6.0, 4.7)); fig.subplots_adjust(left=0.14, right=0.96, top=0.86, bottom=0.12)
allx = [cov[k] for k in KGS + UNTESTABLE]; lo, hi = min(allx), max(allx); pad = (hi - lo) * 0.12
YBOT = -0.085; ystrip = (YBOT - 0.012) / 2
ax.axhspan(YBOT, -0.012, color='#f0f0f0', zorder=0)
for kg in UNTESTABLE:
    ax.scatter(cov[kg], ystrip, marker='x', s=55, color='#8a8a8a', lw=1.6, zorder=4)
    ax.annotate(LAB[kg], (cov[kg], ystrip), xytext=(0, 7), textcoords='offset points',
                ha='center', va='bottom', fontsize=7.5, color='#777')
ax.text(hi + pad, YBOT + 0.006, 'untestable: no drug/disease pairs', fontsize=6.8,
        color='#aaa', ha='right', va='bottom', style='italic')
ax.axhline(0, color='#9a9a9a', lw=1.0, ls='--', zorder=1)
ax.text(hi + pad, 0.006, 'no-KG baseline', fontsize=7, color='#888', ha='right', va='bottom')
NUDGE = {'biokg': -0.028, 'drkg': +0.028}
for kg in KGS:
    ax.axvline(cov[kg], color='#e2e2e2', lw=0.9, zorder=0)
    dx = NUDGE.get(kg, 0)
    ax.text(cov[kg] + dx, 1.012, LAB[kg], transform=ax.get_xaxis_transform(), ha='center',
            va='bottom', fontsize=8, color='#555')
    if dx:
        ax.plot([cov[kg] + dx, cov[kg]], [1.010, 0.998], transform=ax.get_xaxis_transform(),
                color='#cfcfcf', lw=0.6, clip_on=False, zorder=0)
for m in MODELS:
    xs = np.array([cov[k] for k in order_cov]); ys = np.array([lift[m][k] for k in order_cov])
    es = [ci[m][k] for k in order_cov]
    s, i = np.polyfit(xs, ys, 1); xl = np.array([xs.min(), xs.max()])
    ax.plot(xl, s * xl + i, color='#bdbdbd', lw=1.1, ls='--', zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt=MK[m], ms=7.5, color=MC[m], mec='white', mew=0.7,
                ecolor=MC[m], elinewidth=1.1, capsize=3, zorder=3, label=m)
ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(YBOT, 0.66); ax.set_yticks(np.arange(0, 0.61, 0.1))
ax.set_xlabel('KG coverage score  (Coverage [dim])', fontsize=9)
ax.set_ylabel('Downstream lift  (mean ΔMRR vs no-KG)', fontsize=9)
ax.legend(loc='upper left', fontsize=8, handletextpad=0.4, title='error bars = 95% CI', title_fontsize=7.5)
ax.set_title('v1  Connected scatter', fontsize=10.5, fontweight='bold', pad=16)
save(fig, 'cov_v1_connected_scatter')

# ============ v2: grouped bars ============
fig, ax = plt.subplots(figsize=(6.4, 4.7)); fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.16)
groups = sorted(KGS + UNTESTABLE, key=lambda k: cov[k]); nG = len(groups); nM = len(MODELS); w = 0.24
xc = np.arange(nG)
for mi, m in enumerate(MODELS):
    offs = (mi - (nM - 1) / 2) * w
    vals = [lift[m].get(k, np.nan) for k in groups]; errs = [ci[m].get(k, np.nan) for k in groups]
    ax.bar(xc + offs, vals, w, color=MC[m], label=m, edgecolor='white', lw=0.5,
           yerr=errs, error_kw=dict(elinewidth=1.0, capsize=2.5, ecolor='#444'))
# hatched 'not runnable' bars for untestable
for gi, k in enumerate(groups):
    if k in UNTESTABLE:
        ax.bar(xc[gi], 0.66, 3 * w, color='none', edgecolor='#cccccc', hatch='//', lw=0.0, zorder=0)
        ax.text(xc[gi], 0.33, 'not\nrunnable', ha='center', va='center', fontsize=7.5, color='#999')
ax.axhline(0, color='#999', lw=0.8)
ax.set_xticks(xc)
ax.set_xticklabels([f'{LAB[k]}\ncov={cov[k]:.2f}' for k in groups], fontsize=8)
ax.set_ylim(0, 0.66); ax.set_ylabel('Downstream lift  (mean ΔMRR vs no-KG)', fontsize=9)
ax.set_xlabel('KG  (ordered by coverage →)', fontsize=9)
ax.legend(loc='upper right', fontsize=8, ncol=1, title='error bars = 95% CI', title_fontsize=7.5)
ax.set_title('v2  Grouped bars by KG and LLM', fontsize=10.5, fontweight='bold', pad=10)
save(fig, 'cov_v2_grouped_bars')

# ============ v3: heatmap ============
fig, ax = plt.subplots(figsize=(6.2, 3.6)); fig.subplots_adjust(left=0.17, right=0.9, top=0.74, bottom=0.13)
cols = order_cov; M = np.array([[lift[m][k] for k in cols] for m in MODELS])
im = ax.imshow(M, cmap='YlGnBu', aspect='auto', vmin=0.25, vmax=0.58)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([LAB[k] for k in cols], fontsize=9)
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=8.5)
for r, m in enumerate(MODELS):
    for c, k in enumerate(cols):
        v = lift[m][k]
        ax.text(c, r, f'{v:.2f}\n±{ci[m][k]:.02f}', ha='center', va='center',
                fontsize=7.6, color='white' if v > 0.45 else '#222')
# coverage strip above
for c, k in enumerate(cols):
    ax.text(c, -0.75, f'coverage\n{cov[k]:.2f}', ha='center', va='center', fontsize=7.3, color='#555')
ax.set_xlim(-0.5, len(cols) - 0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('mean ΔMRR', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.set_xlabel('KG  (ordered by coverage →)', fontsize=9, labelpad=6)
ax.set_title('v3  Lift heatmap (LLM × KG)', fontsize=10.5, fontweight='bold', pad=34)
save(fig, 'cov_v3_heatmap')

# ============ v4: model-centered effect ============
fig, ax = plt.subplots(figsize=(6.0, 4.7)); fig.subplots_adjust(left=0.14, right=0.96, top=0.86, bottom=0.12)
ax.axhline(0, color='#9a9a9a', lw=1.0, ls='--', zorder=1)
ax.text(0.985, 0.012, "each model's own mean", transform=ax.transAxes, fontsize=7,
        color='#888', ha='right', va='bottom')
allc = []
JIT = {MODELS[0]: -0.0028, MODELS[1]: 0.0, MODELS[2]: +0.0028}   # separate identical-x points
for m in MODELS:
    mm = np.mean([lift[m][k] for k in KGS])
    xs = np.array([cov[k] for k in order_cov]); ys = np.array([lift[m][k] - mm for k in order_cov])
    es = [ci[m][k] for k in order_cov]
    ax.errorbar(xs + JIT[m], ys, yerr=es, fmt=MK[m], ms=7.0, color=MC[m], mec='white', mew=0.7,
                ecolor=MC[m], elinewidth=1.0, capsize=2.5, alpha=0.9, zorder=3, label=m)
    allc += list(zip(xs, ys))
ac = np.array(allc); s, i = np.polyfit(ac[:, 0], ac[:, 1], 1)
xl = np.array([ac[:, 0].min(), ac[:, 0].max()])
r = st.pearsonr(ac[:, 0], ac[:, 1])[0]
ax.plot(xl, s * xl + i, color='#555', lw=1.4, ls='--', zorder=2, label=f'pooled fit (r={r:.2f})')
NUDGE = {'biokg': -0.028, 'drkg': +0.028}
for kg in KGS:
    dx = NUDGE.get(kg, 0)
    ax.text(cov[kg] + dx, 1.012, LAB[kg], transform=ax.get_xaxis_transform(), ha='center',
            va='bottom', fontsize=8, color='#555')
    if dx:
        ax.plot([cov[kg] + dx, cov[kg]], [1.010, 0.998], transform=ax.get_xaxis_transform(),
                color='#cfcfcf', lw=0.6, clip_on=False, zorder=0)
ax.set_xlabel('KG coverage score  (Coverage [dim])', fontsize=9)
ax.set_ylabel('KG effect on lift  (ΔMRR − model mean)', fontsize=9)
ax.legend(loc='upper left', fontsize=7.8, handletextpad=0.4)
ax.set_title('v4  Model-centered KG effect', fontsize=10.5, fontweight='bold', pad=16)
save(fig, 'cov_v4_model_centered')

# ============ v5: faceted small multiples ============
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=True)
fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.16, wspace=0.12)
for ax, m in zip(axes, MODELS):
    xs = np.array([cov[k] for k in order_cov]); ys = np.array([lift[m][k] for k in order_cov])
    es = [ci[m][k] for k in order_cov]
    s, i = np.polyfit(xs, ys, 1); xl = np.array([xs.min(), xs.max()])
    ax.plot(xl, s * xl + i, color='#bdbdbd', lw=1.1, ls='--', zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt=MK[m], ms=7.5, color=MC[m], mec='white', mew=0.7,
                ecolor=MC[m], elinewidth=1.1, capsize=3, zorder=3)
    for kg in KGS:
        ax.annotate(LAB[kg], (cov[kg], lift[m][kg]), xytext=(4, -9), textcoords='offset points',
                    fontsize=6.6, color='#777')
    ax.set_title(m, fontsize=9, fontweight='bold', color=MC[m], pad=4)
    ax.set_xlim(0.44, 0.61); ax.set_ylim(0.22, 0.62)
    ax.set_xlabel('coverage', fontsize=8.5)
axes[0].set_ylabel('Downstream lift (ΔMRR)', fontsize=9)
fig.suptitle('v5  Faceted by LLM', fontsize=10.5, fontweight='bold', x=0.085, ha='left')
save(fig, 'cov_v5_faceted')

print('done')
