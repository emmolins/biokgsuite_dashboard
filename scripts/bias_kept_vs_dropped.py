#!/usr/bin/env python3
"""
Selection-bias analysis: do the 137 dropped pairs differ systematically from the
116 kept pairs, in ways that would make the benchmark unrepresentative?

Compares kept vs dropped (anchored on real final-116 membership: disease + shared
ingredient match) across therapeutic area, approval year, agencies, repurposing
strictness, drug modality, and drug age, with appropriate tests + effect sizes.
Also breaks the effect down by drop stage (unresolved / leakage / coverage).

Out: results/tables/bias_kept_vs_dropped.csv     (per-characteristic test table)
     results/figures/bias_kept_vs_dropped.(png|pdf)
"""
import os, re, numpy as np, pandas as pd
from scipy import stats as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, 'data', 'gold_standards'); TAB = os.path.join(ROOT, 'results', 'tables')
FIGS = os.path.join(ROOT, 'results', 'figures'); os.makedirs(FIGS, exist_ok=True)
pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'figure.dpi': 200,
    'legend.frameon': False})

p = pd.read_csv(os.path.join(GS, 'pairs_with_ids_v2.csv'))
g = pd.read_csv(os.path.join(GS, 'gold_standard_v4_bigset.tsv'), sep='\t')
drop = pd.read_csv(os.path.join(TAB, 'dropped_253_to_116.csv'))[['Pair_ID', 'drop_stage']]

# kept / dropped label (combo-aware) + drop stage
def nm(x): return str(x).strip().lower()
def ings(s): return set(t.strip() for t in re.split(r'[+/]', str(s).lower()) if t.strip())
gidx = [(nm(r.disease_name), ings(r.drug_name)) for _, r in g.iterrows()]
p['ings'] = p.Active.map(ings)
p['kept'] = p.apply(lambda c: any(dn == nm(c.NewIndication) and (gi & c['ings']) for dn, gi in gidx), axis=1)
p = p.merge(drop, on='Pair_ID', how='left')
p['status'] = np.where(p.kept, 'kept', 'dropped')

# ---- derived covariates ----
def modality(row):
    s = (str(row.Active) + ' ' + str(row.Drug)).lower()
    if re.search(r'\bcar-t|cabtagene|cel\b|leucel', s): return 'cell/gene therapy'
    if 'vaccine' in s or 'ebola' in s: return 'vaccine'
    if re.search(r'mab\b|cept\b|umab|zumab|ximab', s): return 'antibody/biologic'
    if re.search(r'tide\b|glutide|tatercept|parin', s): return 'peptide/other biologic'
    return 'small molecule'
p['modality'] = p.apply(modality, axis=1)
p['biologic'] = p.modality != 'small molecule'
p['n_agencies'] = p.Agencies.astype(str).str.count(r'[+/&]') + 1
p['multi_agency'] = p.n_agencies >= 2
p['oncology'] = p.TA_simplified == 'Oncology'
p['orig_year'] = pd.to_numeric(p.OrigYear, errors='coerce')
p['drug_age'] = 2024 - p['orig_year']           # years since original approval

N = len(p); nk = int(p.kept.sum()); nd = N - nk
rows = []
def fisher(mask_name, mask):
    tab = pd.crosstab(mask, p.status)
    if tab.shape == (2, 2):
        odds, pv = st.fisher_exact(tab)
        kr = p[mask].kept.mean(); dr = p[~mask].kept.mean() if (~mask).any() else np.nan
        rows.append((mask_name, f'{mask.sum()} yes / {(~mask).sum()} no',
                     f'kept-rate {kr:.0%} vs {dr:.0%}', f'OR={odds:.2f}', f'p={pv:.4f}'))
fisher('Oncology', p.oncology)
fisher('Biologic (any)', p.biologic)
fisher('Multi-agency (>=2)', p.multi_agency)
fisher('Strict repurposing', p.StrictRepurposing == True)
fisher('2023 (vs 2024-26)', p.YearSet == 2023)
# drug age (continuous) Mann-Whitney
ka, da = p[p.kept].drug_age.dropna(), p[~p.kept].drug_age.dropna()
u, pu = st.mannwhitneyu(ka, da, alternative='two-sided')
rows.append(('Drug age (yrs since orig approval)', f'median kept {ka.median():.0f} vs dropped {da.median():.0f}',
             f'mean {ka.mean():.1f} vs {da.mean():.1f}', 'Mann-Whitney', f'p={pu:.4f}'))
# therapeutic area overall chi-square (collapse small cells)
vc = p.TA_simplified.value_counts(); p['TA_c'] = np.where(p.TA_simplified.isin(vc[vc >= 10].index), p.TA_simplified, 'Other')
chi = st.chi2_contingency(pd.crosstab(p.TA_c, p.status))
rows.append(('Therapeutic area (overall)', f'{p.TA_c.nunique()} groups', 'chi-square',
             f'chi2={chi[0]:.1f}', f'p={chi[1]:.4f}'))
res = pd.DataFrame(rows, columns=['characteristic', 'split', 'effect', 'stat', 'p_value'])
res.to_csv(os.path.join(TAB, 'bias_kept_vs_dropped.csv'), index=False)
print(f"253 pairs: {nk} kept, {nd} dropped\n")
print(res.to_string(index=False))

# modality composition kept vs dropped
print("\nModality composition (kept vs dropped):")
print(pd.crosstab(p.modality, p.status, normalize='columns').round(3).to_string())

# ---- figure: kept vs dropped composition on the dims that matter ----
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6)); fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.22, wspace=0.45)
def comp_bar(ax, col, title, order=None):
    ct = pd.crosstab(p[col], p.status, normalize='columns')
    if order: ct = ct.reindex(order)
    ct = ct[['dropped', 'kept']]
    ct.plot(kind='barh', ax=ax, color=['#CB3B2E', '#2E8B57'], width=0.74, legend=(ax is axes[0]))
    ax.set_title(title, fontsize=9.5, fontweight='bold'); ax.set_xlabel('fraction within group', fontsize=8); ax.set_ylabel('')
    ax.tick_params(labelsize=7.6)
comp_bar(axes[0], 'modality', 'a  Drug modality')
ta_order = p.TA_c.value_counts().index.tolist()
comp_bar(axes[1], 'TA_c', 'b  Therapeutic area', order=ta_order)
comp_bar(axes[2], 'YearSet', 'c  Approval year')
axes[0].legend(['dropped (137)', 'kept (116)'], fontsize=7.5, loc='lower right')
fig.suptitle('Selection bias: composition of dropped vs kept pairs', fontsize=11, fontweight='bold', x=0.5, y=0.97)
fig.savefig(os.path.join(FIGS, 'bias_kept_vs_dropped.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_kept_vs_dropped.pdf'), bbox_inches='tight', facecolor='white')
print("\nwrote bias_kept_vs_dropped.csv + .png")
