#!/usr/bin/env python3
"""
Three alternative visualisations of the kept-vs-dropped composition (v4 data).

v4a dumbbell   - kept% vs dropped% per category, connected (gap = imbalance)
v4b ratio      - representation ratio log2(kept share / dropped share), diverging
v4c slope      - dropped -> kept proportion slope per category (flat = balanced)

Out: results/figures/bias_v4a_dumbbell / v4b_ratio / v4c_slope .png/.pdf
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, 'data', 'gold_standards'); FIGS = os.path.join(ROOT, 'results', 'figures')
TAB = os.path.join(ROOT, 'results', 'tables')
pref = ['Helvetica Neue', 'Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'axes.linewidth': 0.9, 'savefig.dpi': 300, 'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'text.color': '#222', 'axes.edgecolor': '#666', 'xtick.color': '#555', 'ytick.color': '#333'})
KEPT, DROP = '#1B9E77', '#D95F02'

# ---- labels ----
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

# assemble (dimension, category, kept_share, dropped_share, kept_n, dropped_n)
rows = []
DIMS = [('Drug modality', 'modality', ['Small molecule', 'Antibody / biologic', 'Peptide / other', 'Cell / gene therapy', 'Vaccine']),
        ('Therapeutic area', 'TA', None),
        ('Approval year', 'Year', ['2023', '2024-2026'])]
for dim, col, order in DIMS:
    ct = pd.crosstab(p[col], p.grp)
    if order: ct = ct.reindex([o for o in order if o in ct.index])
    else: ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    ks, ds = ct['kept'] / ct['kept'].sum(), ct['dropped'] / ct['dropped'].sum()
    for cat in ct.index:
        rows.append((dim, cat, ks[cat], ds[cat], int(ct.loc[cat, 'kept']), int(ct.loc[cat, 'dropped'])))
D = pd.DataFrame(rows, columns=['dim', 'cat', 'k', 'd', 'kn', 'dn'])

# y layout with dimension gaps
ypos, ylab, seps, dim_centers = [], [], [], {}
y = 0
for dim in [d[0] for d in DIMS]:
    sub = D[D.dim == dim]; start = y
    for _, r in sub.iterrows():
        ypos.append(y); ylab.append(r['cat']); y += 1
    dim_centers[dim] = (start + y - 1) / 2; y += 1.2; seps.append(y - 0.6)
D['y'] = ypos
TOP = y

# ============ v4a DUMBBELL ============
fig, ax = plt.subplots(figsize=(7.2, 6.8)); fig.subplots_adjust(left=0.34, right=0.83, top=0.86, bottom=0.07)
for _, r in D.iterrows():
    yy = TOP - r['y']
    ax.plot([r['d']*100, r['k']*100], [yy, yy], color='#cfcfcf', lw=2.2, zorder=1, solid_capstyle='round')
    ax.scatter(r['d']*100, yy, s=70, color=DROP, zorder=3, edgecolor='white', lw=1)
    ax.scatter(r['k']*100, yy, s=70, color=KEPT, zorder=3, edgecolor='white', lw=1)
    ax.text(-2.0, yy, r['cat'], ha='right', va='center', fontsize=8.6)
for dim, yc in dim_centers.items():
    ax.text(-26, TOP - yc, dim, ha='left', va='center', fontsize=9, fontweight='bold', color='#444', rotation=90)
ax.set_xlim(-3, 62); ax.set_yticks([]); ax.set_ylim(-0.6, TOP + 0.4)
ax.set_xlabel('share within group (%)', fontsize=9.5)
ax.scatter([], [], color=DROP, label='dropped (137)'); ax.scatter([], [], color=KEPT, label='kept (116)')
ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.0, 1.02))
ax.set_title('Composition of kept vs dropped — connected gaps', fontsize=11.5, fontweight='bold', loc='left', pad=26)
ax.text(0, 1.045, 'Short connectors = balanced (area, year); long = shifted (modality).',
        transform=ax.transAxes, fontsize=8.3, color='#666')
fig.savefig(os.path.join(FIGS, 'bias_v4a_dumbbell.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v4a_dumbbell.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)

# ============ v4b REPRESENTATION RATIO ============
D['lr'] = np.log2((D['kn'] + 0.5) / (D['kn'] + D['dn'] + 1) / ((D['dn'] + 0.5) / (D['kn'] + D['dn'] + 1)))
fig, ax = plt.subplots(figsize=(7.6, 6.6)); fig.subplots_adjust(left=0.30, right=0.9, top=0.85, bottom=0.11)
ax.axvspan(-0.5, 0.5, color='#f0f0f0', zorder=0)               # "balanced" band
ax.axvline(0, color='#888', lw=1)
for _, r in D.iterrows():
    yy = TOP - r['y']; v = np.clip(r['lr'], -3, 3); col = KEPT if v > 0 else DROP
    ax.barh(yy, v, color=col, height=0.62, zorder=2, alpha=0.92)
    ax.text(-3.55, yy, r['cat'], ha='right', va='center', fontsize=8.6)
    if abs(r['lr']) > 0.5:
        ax.text(v + (0.12 if v > 0 else -0.12), yy, f"{r['kn']}/{r['dn']}", ha='left' if v > 0 else 'right',
                va='center', fontsize=7, color='#777')
# dimension headers above each group (not rotated)
for dim in [d[0] for d in DIMS]:
    top_y = TOP - D[D.dim == dim].y.min()
    ax.text(-3.55, top_y + 0.7, dim.upper(), ha='right', va='center', fontsize=7.8, fontweight='bold', color='#999')
ax.set_xlim(-3.4, 3.4); ax.set_yticks([]); ax.set_ylim(-0.6, TOP + 0.4)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xlabel('representation ratio   log2(kept share / dropped share)', fontsize=9.5)
ax.text(0.55, TOP + 0.1, 'over-represented in kept →', fontsize=8, color=KEPT, ha='left', fontweight='bold')
ax.text(-0.55, TOP + 0.1, '← over-represented in dropped', fontsize=8, color=DROP, ha='right', fontweight='bold')
ax.set_title('Which traits are over- or under-represented in the 116?', fontsize=11.5, fontweight='bold', loc='left', pad=22)
ax.text(0, 1.045, 'Grey band = balanced (±0.5 log2).  Labels = kept/dropped counts.',
        transform=ax.transAxes, fontsize=8.3, color='#666')
fig.savefig(os.path.join(FIGS, 'bias_v4b_ratio.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v4b_ratio.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)

# ============ v4c SLOPE ============
fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.6)); fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.10, wspace=0.5)
for ax, (dim, col, order) in zip(axes, DIMS):
    sub = D[D.dim == dim]
    for _, r in sub.iterrows():
        shift = abs(r['k'] - r['d'])
        c = '#c4c4c4' if shift < 0.05 else (KEPT if r['k'] > r['d'] else DROP)
        lw = 1.2 + shift * 14
        ax.plot([0, 1], [r['d']*100, r['k']*100], color=c, lw=lw, alpha=0.85, solid_capstyle='round', zorder=2)
        ax.scatter([0, 1], [r['d']*100, r['k']*100], s=22, color=c, zorder=3, edgecolor='white', lw=0.5)
        # only label categories with enough share to be legible (declutter the bottom)
        if max(r['k'], r['d']) >= 0.06:
            ax.text(-0.05, r['d']*100, r['cat'], ha='right', va='center', fontsize=7.8, color='#444')
            ax.text(1.05, r['k']*100, r['cat'], ha='left', va='center', fontsize=7.8, color='#444')
    ax.set_xlim(-0.55, 1.55); ax.set_xticks([0, 1]); ax.set_xticklabels(['dropped', 'kept'], fontsize=9, fontweight='bold')
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_title(dim, fontsize=10, fontweight='bold', pad=8)
axes[0].set_ylabel('share within group (%)', fontsize=9)
fig.suptitle('Composition shift: dropped → kept  (steep = shifted, flat = balanced)',
             fontsize=11.5, fontweight='bold', x=0.5, y=0.95)
fig.savefig(os.path.join(FIGS, 'bias_v4c_slope.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v4c_slope.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)
print('wrote bias_v4a_dumbbell, bias_v4b_ratio, bias_v4c_slope')
