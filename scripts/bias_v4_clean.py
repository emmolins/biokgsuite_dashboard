#!/usr/bin/env python3
"""
Cleaner version of v4 (diverging paired bars: kept vs dropped composition).
Sorted bars, direct %-at-tip labels, left category labels, minimal axes.

Out: results/figures/bias_v4_paired_clean.(png|pdf)
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, 'data', 'gold_standards'); FIGS = os.path.join(ROOT, 'results', 'figures')
pref = ['Helvetica Neue', 'Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'savefig.dpi': 300, 'figure.dpi': 200, 'text.color': '#2b2b2b'})
KEPT, DROP = '#2BA37A', '#E07B39'        # slightly softer green / orange

p = pd.read_csv(os.path.join(GS, 'pairs_with_ids_v2.csv'))
g = pd.read_csv(os.path.join(GS, 'gold_standard_v4_bigset.tsv'), sep='\t')
def nm(x): return str(x).strip().lower()
def ings(s): return set(t.strip() for t in re.split(r'[+/]', str(s).lower()) if t.strip())
gidx = [(nm(r.disease_name), ings(r.drug_name)) for _, r in g.iterrows()]
p['ings'] = p.Active.map(ings)
p['kept'] = p.apply(lambda c: any(dn == nm(c.NewIndication) and (gi & c['ings']) for dn, gi in gidx), axis=1)
def modality(row):
    s = (str(row.Active) + ' ' + str(row.Drug)).lower()
    if re.search(r'cabtagene|leucel|car-t', s): return 'Cell / gene therapy'
    if 'vaccine' in s: return 'Vaccine'
    if re.search(r'mab\b|umab|zumab|ximab|cept\b', s): return 'Antibody / biologic'
    if re.search(r'tide\b|glutide|parin|tatercept', s): return 'Peptide / other'
    return 'Small molecule'
p['modality'] = p.apply(modality, axis=1)
vc = p.TA_simplified.value_counts(); p['TA'] = np.where(p.TA_simplified.isin(vc[vc >= 10].index), p.TA_simplified, 'Other')
p['Year'] = p.YearSet.astype(str)
p['grp'] = np.where(p.kept, 'kept', 'dropped')

panels = [('Drug modality', 'modality'), ('Therapeutic area', 'TA'), ('Approval year', 'Year')]
fig, axes = plt.subplots(1, 3, figsize=(9.8, 6.4))
fig.subplots_adjust(left=0.015, right=0.99, top=0.78, bottom=0.04, wspace=0.55)

NMAX = max(p[c].nunique() for _, c in panels)                 # = 8 (therapeutic area)
for ax, (title, col) in zip(axes, panels):
    ct = pd.crosstab(p[col], p.grp)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]           # smallest at bottom, biggest at top
    k = ct['kept'] / ct['kept'].sum(); d = ct['dropped'] / ct['dropped'].sum()
    n = len(ct); off = (NMAX - n) / 2.0                        # vertical-centre each panel
    y = np.arange(n) + off; cats = ct.index.tolist()
    ax.barh(y, -d.values, color=DROP, height=0.58, zorder=2)
    ax.barh(y, k.values, color=KEPT, height=0.58, zorder=2)
    ax.axvline(0, color='#bbb', lw=1.0, zorder=1)
    mx = max(k.max(), d.max())
    for yi, cat in zip(y, cats):
        ax.text(0, yi + 0.46, cat, ha='center', va='bottom', fontsize=9, color='#333')
        if d[cat] > 0.012:
            ax.text(-d[cat] - mx*0.03, yi, f'{d[cat]*100:.0f}%', ha='right', va='center', fontsize=7.8, color=DROP)
        if k[cat] > 0.012:
            ax.text(k[cat] + mx*0.03, yi, f'{k[cat]*100:.0f}%', ha='left', va='center', fontsize=7.8, color=KEPT)
    ax.set_xlim(-mx*1.32, mx*1.32); ax.set_ylim(-0.6, NMAX - 0.3)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=18, color='#222')

# title, subtitle, and one clean legend
from matplotlib.patches import Patch
fig.text(0.5, 0.965, 'Composition of kept vs dropped pairs', ha='center', fontsize=13.5, fontweight='bold')
fig.text(0.5, 0.925, 'Each group’s internal make-up. Area and year mirror each other (balanced); '
         'modality is the trait that shifts.', ha='center', fontsize=8.8, color='#666')
fig.legend(handles=[Patch(color=DROP, label='dropped (137)'), Patch(color=KEPT, label='kept (116)')],
           loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False,
           fontsize=9.5, handlelength=1.1, columnspacing=1.6)

out = os.path.join(FIGS, 'bias_v4_paired_clean')
fig.savefig(out + '.png', bbox_inches='tight', facecolor='white')
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
print('wrote', out)
