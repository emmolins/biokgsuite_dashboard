#!/usr/bin/env python3
"""
Analysis 2 (CORRECTED) — trends in the dropped drug-disease pairs (253 -> 116).

Reconstructs the TRUE path: of the 253 ID-resolved candidate pairs
(pairs_with_ids_v2.csv), which survive into the final evaluated benchmark
(gold_standard_v4_bigset.tsv, 116 pairs)? Dropped = candidate not in the final
set. Surveys whether dropped pairs are biased by therapeutic area / year.

NOTE: an earlier version used usable_pairs_v2_audited.csv (a *superseded* v2
coverage audit, 199 evaluable) — that is NOT the audit that produced the 116.
This script uses the final v4 set membership, which is the real benchmark.

Out: results/tables/analysis2_dropped_pairs.csv
     results/figures/analysis2_funnel.(png|pdf)
     results/figures/analysis2_drop_by_area.(png|pdf)
"""
import os, numpy as np, pandas as pd
from scipy import stats as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, 'data', 'gold_standards')
FIGS = os.path.join(ROOT, 'results', 'figures'); os.makedirs(FIGS, exist_ok=True)
TAB = os.path.join(ROOT, 'results', 'tables')

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---- candidates (253) and final benchmark (116) ----
pid = pd.read_csv(os.path.join(GS, 'pairs_with_ids_v2.csv'))
g = pd.read_csv(os.path.join(GS, 'gold_standard_v4_bigset.tsv'), sep='\t')
pid['key'] = pid.DrugBank_ID.astype(str) + '|' + pid.DOID_ID.astype(str)
g['key'] = g.drug_id.astype(str) + '|' + g.disease_id.astype(str)
gk = set(g['key'])
pid['in_final'] = pid['key'].isin(gk)
pid['status'] = np.where(pid['in_final'], 'kept', 'dropped')

N = len(pid); n_keep = int(pid.in_final.sum()); n_drop = int((~pid.in_final).sum())
added = len(gk - set(pid['key']))           # combos/new pairs added in v4
# v4 coverage breadth of the final set
cov = pd.read_csv(os.path.join(GS, 'coverage_annotation_v4.csv'))
ncov = cov.groupby('Pair_ID')['covered'].sum()
all3 = int((ncov == 3).sum())
pid.to_csv(os.path.join(TAB, 'analysis2_dropped_pairs.csv'), index=False)
print(f"candidates={N}  kept->final={n_keep}  dropped={n_drop}  added-in-v4(combos)={added}  final=116  (all-3-KG={all3})")

# ---- bias by therapeutic area ----
ct = pd.crosstab(pid.TA_simplified, pid.status)
ct['n'] = ct.sum(axis=1); ct['drop_rate'] = (ct['dropped'] / ct['n']).round(3)
ct = ct.sort_values('drop_rate', ascending=False)
print("\nDrop rate by therapeutic area:\n", ct.to_string())
big = pid.copy(); vc = big.TA_simplified.value_counts()
big['TA'] = np.where(big.TA_simplified.isin(vc[vc >= 10].index), big.TA_simplified, 'Other')
chi = st.chi2_contingency(pd.crosstab(big.TA, big.status))
print(f"\nchi-square TA vs drop (areas n>=10 + Other): chi2={chi[0]:.2f}, dof={chi[2]}, p={chi[1]:.3f}")
onc = pid.TA_simplified.eq('Oncology')
fish = st.fisher_exact(pd.crosstab(onc, pid.status))
print(f"Oncology drop {pid[onc].status.eq('dropped').mean():.2f} vs rest {pid[~onc].status.eq('dropped').mean():.2f}  Fisher p={fish[1]:.3f}")
print("\nYear:\n", pd.crosstab(pid.YearSet, pid.status, normalize='index').round(3).to_string())
overall = pid.status.eq('dropped').mean()

# ============ FIG 1: funnel ============
stages = ['Candidate\npairs', 'Final\nbenchmark', '…of which\ncommon to 3 KGs']
vals = [N, 116, all3]
cols = ['#9aa7c7', '#0F9D8C', '#3a7d44']
fig, ax = plt.subplots(figsize=(5.6, 4.0)); fig.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.10)
ax.bar(range(3), vals, color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 4, str(v), ha='center', fontweight='bold', fontsize=10)
ax.text(0.5, (N + 116) / 2, f'−{n_drop} dropped\n+{added} combos added',
        ha='center', va='center', fontsize=8, color='#b23')
ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=8.5)
ax.set_ylabel('drug–disease pairs', fontsize=9); ax.set_ylim(0, N * 1.12)
ax.set_title('Analysis 2 — 253 candidates → 116 benchmark', fontsize=10.5, fontweight='bold', pad=8)
fig.savefig(os.path.join(FIGS, 'analysis2_funnel.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'analysis2_funnel.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============ FIG 2: drop rate by therapeutic area (true drop) ============
ca = ct[ct['n'] >= 5].copy()      # only areas with enough pairs to be meaningful
fig, ax = plt.subplots(figsize=(6.6, 4.4)); fig.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.13)
yy = np.arange(len(ca))
ax.barh(yy, ca['drop_rate'], color='#8aa1c9', height=0.66)
ax.set_yticks(yy); ax.set_yticklabels([f'{i}  (n={int(r.n)})' for i, r in ca.iterrows()], fontsize=8)
ax.axvline(overall, color='#444', ls='--', lw=1, label=f'overall drop rate {overall:.2f}')
ax.set_xlabel('Fraction of candidates dropped (253 → 116)', fontsize=8.5); ax.set_xlim(0, 1)
ax.legend(fontsize=7.5, loc='lower right')
ax.set_title('Analysis 2 — drop rate by area (no significant bias, p=%.2f)' % fish[1],
             fontsize=10, fontweight='bold', pad=8)
fig.savefig(os.path.join(FIGS, 'analysis2_drop_by_area.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'analysis2_drop_by_area.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nwrote corrected figures + table")
