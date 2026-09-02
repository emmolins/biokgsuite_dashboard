#!/usr/bin/env python3
"""
Which KG quality dimension best explains downstream LLM lift?

Panel a: downstream lift vs Coverage [dim], one series per LLM (95% CI bars,
         dashed light-grey OLS fit per LLM).
Panel b: every quality dimension ranked by Pearson r with mean lift (n=3 KGs),
         coloured by whether the dimension's KG ordering matches the lift
         ordering (PrimeKG > BioKG > DRKG). EXPLORATORY ONLY (n=3).

Usage:  python scripts/make_lift_dim_analysis.py
Output: results/figures/lift_dim_analysis.(png|pdf)
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

KGS = ['primekg', 'drkg', 'biokg']                 # testable (have drug/disease pairs)
UNTESTABLE = ['openbilink', 'hetionet']            # benchmarked but task not runnable
LAB = {'primekg': 'PrimeKG', 'drkg': 'DRKG', 'biokg': 'BioKG',
       'hetionet': 'HetioNet', 'openbilink': 'OpenBioLink'}
XDIM = 'Coverage [dim]'                 # panel-a x-axis
MIN_RANGE = 0.02                        # drop near-constant dims from panel b
MODEL_FILES = {
    'GPT-4.1-mini':          ['09_big_gpt.csv', '09_big_gpt_s12.csv'],
    'Gemini-3.1-flash-lite': ['09_big_glite.csv', '09_big_glite_s12.csv'],
    'Llama-3.3-70B':         ['09_big_llama.csv', '09_big_llama_s1.csv', '09_big_llama_s2.csv'],
}
MC = {'GPT-4.1-mini': '#0F9D8C', 'Gemini-3.1-flash-lite': '#5B4FB8', 'Llama-3.3-70B': '#D1791F'}
MK = {'GPT-4.1-mini': 'o',       'Gemini-3.1-flash-lite': 's',       'Llama-3.3-70B': '^'}

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---- lift per model x KG (mean ΔMRR + paired 95% CI) ----
def load(parts):
    return pd.concat([pd.read_csv(os.path.join(RUNS, p)) for p in parts], ignore_index=True)
stats, perkg = {}, {kg: [] for kg in KGS}
for m, parts in MODEL_FILES.items():
    d = load(parts); d = d[d['parsed'] == 1].copy()
    d['arm'] = np.where(d['condition'] == 'no_kg', 'no_kg', d['kg'])
    nk = d[d.arm == 'no_kg'].groupby('disease')['reciprocal_rank'].mean()
    stats[m] = {}
    for kg in KGS:
        kk = d[d.arm == kg].groupby('disease')['reciprocal_rank'].mean()
        j = pd.concat([kk.rename('k'), nk.rename('n')], axis=1).dropna()
        diff = (j.k - j.n); n = len(diff)
        ci = st.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        stats[m][kg] = (diff.mean(), ci); perkg[kg].append(diff.mean())
meanlift = {kg: float(np.mean(v)) for kg, v in perkg.items()}
lift_order = sorted(KGS, key=lambda k: -meanlift[k])

# ---- quality dimensions ----
b = pd.read_csv(BSUM).set_index('KG')
y = np.array([meanlift[k] for k in KGS])
rows = []
for c in b.columns:
    x = b.loc[KGS, c].astype(float).values
    if np.isnan(x).any() or (x.max() - x.min()) < MIN_RANGE:
        continue
    r = st.pearsonr(x, y)[0]
    order = sorted(KGS, key=lambda k: -b.loc[k, c])
    rows.append((c, r, order == lift_order))
T = pd.DataFrame(rows, columns=['dim', 'r', 'match']).sort_values('r')

# ================= FIGURE 1: lift vs coverage =================
fig1, ax = plt.subplots(figsize=(6.0, 4.8))
fig1.subplots_adjust(left=0.135, right=0.965, top=0.86, bottom=0.12)
qx = {kg: float(b.loc[kg, XDIM]) for kg in KGS}
ux = {kg: float(b.loc[kg, XDIM]) for kg in UNTESTABLE}
allx = list(qx.values()) + list(ux.values())
lo, hi = min(allx), max(allx); pad = (hi - lo) * 0.12
YBOT = -0.085                                    # untestable strip lives below 0
ystrip = (YBOT - 0.012) / 2

# untestable strip + crosses
ax.axhspan(YBOT, -0.012, color='#f0f0f0', zorder=0)
for kg in UNTESTABLE:
    ax.scatter(ux[kg], ystrip, marker='x', s=55, color='#8a8a8a', linewidth=1.6, zorder=4)
    ax.annotate(LAB[kg], (ux[kg], ystrip), xytext=(0, 7), textcoords='offset points',
                ha='center', va='bottom', fontsize=7.5, color='#777')
ax.text(hi + pad, YBOT + 0.006, 'untestable: no drug/disease pairs', fontsize=6.8,
        color='#aaa', ha='right', va='bottom', style='italic')

# no-KG baseline
ax.axhline(0, color='#9a9a9a', lw=1.0, ls='--', zorder=1)
ax.text(hi + pad, 0.006, 'no-KG baseline', fontsize=7, color='#888', ha='right', va='bottom')

# KG guides + top labels (BioKG/DRKG nearly identical x -> offset labels + leaders)
NUDGE = {'biokg': -0.028, 'drkg': +0.028}
for kg in KGS:
    ax.axvline(qx[kg], color='#e2e2e2', lw=0.9, zorder=0)
    dx = NUDGE.get(kg, 0)
    ax.text(qx[kg] + dx, 1.012, LAB[kg], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='#555')
    if dx:   # thin leader from label down to the guide line
        ax.plot([qx[kg] + dx, qx[kg]], [1.010, 0.998], transform=ax.get_xaxis_transform(),
                color='#cfcfcf', lw=0.6, clip_on=False, zorder=0)

order = sorted(KGS, key=lambda k: qx[k])
for m in MODEL_FILES:
    xs = np.array([qx[k] for k in order]); ys = np.array([stats[m][k][0] for k in order])
    es = [stats[m][k][1] for k in order]
    s, i = np.polyfit(xs, ys, 1); xl = np.array([xs.min(), xs.max()])
    ax.plot(xl, s * xl + i, color='#bdbdbd', lw=1.1, ls='--', zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt=MK[m], ms=7.5, color=MC[m], mec='white', mew=0.7,
                ecolor=MC[m], elinewidth=1.1, capsize=3, zorder=3, label=m)
ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(YBOT, 0.66); ax.set_yticks(np.arange(0, 0.61, 0.1))
ax.set_xlabel('KG coverage score  (Coverage [dim])', fontsize=9)
ax.set_ylabel('Downstream lift  (mean ΔMRR vs no-KG)', fontsize=9)
ax.legend(loc='upper left', fontsize=8, handletextpad=0.4,
          title='error bars = 95% CI', title_fontsize=7.5)
ax.set_title('Downstream lift vs. KG coverage', fontsize=10.5, fontweight='bold', pad=16)
out1 = os.path.join(FIGS, 'lift_vs_coverage')
fig1.savefig(out1 + '.png', bbox_inches='tight', facecolor='white')
fig1.savefig(out1 + '.pdf', bbox_inches='tight', facecolor='white')

# ================= FIGURE 2: dimension ranking =================
fig2, ax2 = plt.subplots(figsize=(6.6, 5.4))
fig2.subplots_adjust(left=0.30, right=0.965, top=0.90, bottom=0.11)
yy = np.arange(len(T))
cols = ['#2E8B57' if mt else '#c4c4c4' for mt in T['match']]
ax2.barh(yy, T['r'], color=cols, edgecolor='white', lw=0.5)
ax2.axvline(0, color='#666', lw=0.8)
ax2.set_yticks(yy); ax2.set_yticklabels(T['dim'], fontsize=7.6)
ax2.set_ylim(-0.6, len(T) - 0.4)
ax2.set_xlim(-1.05, 1.05); ax2.set_xlabel('Pearson r with mean lift  (n = 3 KGs)', fontsize=9)
ax2.set_title('Which quality dimension tracks lift?', fontsize=10.5, fontweight='bold', pad=10)
for t in ax2.get_yticklabels():
    if t.get_text() == XDIM:
        t.set_fontweight('bold'); t.set_color('#1b5e34')
h1 = plt.Rectangle((0, 0), 1, 1, color='#2E8B57')
h2 = plt.Rectangle((0, 0), 1, 1, color='#c4c4c4')
ax2.legend([h1, h2], ['KG order matches lift order', 'order differs'],
           loc='lower right', fontsize=7.8, handlelength=1.1, borderaxespad=0.6)
ax2.text(-1.03, len(T) - 0.7, 'exploratory: n = 3 KGs, no inference', ha='left', va='top',
         fontsize=6.8, color='#999', style='italic')
out2 = os.path.join(FIGS, 'dimension_ranking')
fig2.savefig(out2 + '.png', bbox_inches='tight', facecolor='white')
fig2.savefig(out2 + '.pdf', bbox_inches='tight', facecolor='white')

print(f'Wrote {out1}.png and {out2}.png | lift order {lift_order} | best matching dim:',
      T[T.match].iloc[-1]['dim'] if T.match.any() else 'none')
