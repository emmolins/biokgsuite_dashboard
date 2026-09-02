#!/usr/bin/env python3
"""
Four polished design variants of the selection-bias story (137 dropped vs 116 kept).

v1 forest      - bias scorecard: odds-of-being-kept per characteristic, 95% CI, null line
v2 recency     - drug-approval-year distributions per group (the headline effect)
v3 funnel      - 253->116 funnel with what each filter removes (composition-aware)
v4 paired      - kept vs dropped composition, diverging paired bars

Out: results/figures/bias_v{1..4}_*.png/.pdf
"""
import os, re, numpy as np, pandas as pd
from scipy import stats as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, 'data', 'gold_standards'); FIGS = os.path.join(ROOT, 'results', 'figures')
TAB = os.path.join(ROOT, 'results', 'tables'); os.makedirs(FIGS, exist_ok=True)

pref = ['Helvetica Neue', 'Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'axes.linewidth': 0.9, 'savefig.dpi': 300, 'figure.dpi': 200,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'text.color': '#222', 'axes.edgecolor': '#555', 'axes.labelcolor': '#222',
    'xtick.color': '#555', 'ytick.color': '#555'})
KEPT, DROP = '#1B9E77', '#D95F02'
ST_COL = {'unresolved': '#7570B3', 'leakage': '#D95F02', 'coverage': '#1F78B4', 'kept': '#1B9E77'}

# ---------------- data + labels ----------------
p = pd.read_csv(os.path.join(GS, 'pairs_with_ids_v2.csv'))
g = pd.read_csv(os.path.join(GS, 'gold_standard_v4_bigset.tsv'), sep='\t')
drop = pd.read_csv(os.path.join(TAB, 'dropped_253_to_116.csv'))[['Pair_ID', 'drop_stage']]
def nm(x): return str(x).strip().lower()
def ings(s): return set(t.strip() for t in re.split(r'[+/]', str(s).lower()) if t.strip())
gidx = [(nm(r.disease_name), ings(r.drug_name)) for _, r in g.iterrows()]
p['ings'] = p.Active.map(ings)
p['kept'] = p.apply(lambda c: any(dn == nm(c.NewIndication) and (gi & c['ings']) for dn, gi in gidx), axis=1)
p = p.merge(drop, on='Pair_ID', how='left')
p['orig'] = pd.to_numeric(p.OrigYear, errors='coerce')
def modality(row):
    s = (str(row.Active) + ' ' + str(row.Drug)).lower()
    if re.search(r'cabtagene|leucel|car-t', s): return 'Cell / gene therapy'
    if 'vaccine' in s: return 'Vaccine'
    if re.search(r'mab\b|umab|zumab|ximab|cept\b', s): return 'Antibody / biologic'
    if re.search(r'tide\b|glutide|parin|tatercept', s): return 'Peptide / other'
    return 'Small molecule'
p['modality'] = p.apply(modality, axis=1)
p['novel_mod'] = p.modality.isin(['Cell / gene therapy', 'Vaccine'])
p['n_ag'] = p.Agencies.astype(str).str.count(r'[+/&]') + 1
vc = p.TA_simplified.value_counts(); p['TA'] = np.where(p.TA_simplified.isin(vc[vc >= 10].index), p.TA_simplified, 'Other')
stg_lab = {'1_unresolved_ids': 'unresolved', '2_answer_leakage': 'leakage', '3_no_kg_coverage': 'coverage'}
p['stage'] = p.drop_stage.map(stg_lab).fillna('kept')

def OR(mask):
    a = ((mask) & p.kept).sum(); b = ((mask) & ~p.kept).sum()
    c = ((~mask) & p.kept).sum(); d = ((~mask) & ~p.kept).sum()
    a, b, c, d = a + .5, b + .5, c + .5, d + .5
    lor = np.log(a * d / (b * c)); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    pv = st.fisher_exact(pd.crosstab(mask, p.kept))[1]
    return np.exp(lor), np.exp(lor - 1.96*se), np.exp(lor + 1.96*se), pv

# ======================================================== v1 — FOREST
chars = [('Older drug (approved ≤2015)', p.orig <= 2015),
         ('Novel modality (cell/gene, vaccine)', p.novel_mod),
         ('Antibody / biologic', p.modality.str.contains('Antibody')),
         ('Oncology', p.TA_simplified == 'Oncology'),
         ('Multi-agency approval', p.n_ag >= 2),
         ('Strict repurposing', p.StrictRepurposing == True),
         ('Approved 2023 (vs 2024–26)', p.YearSet == 2023)]
data = [(lbl,) + OR(m) for lbl, m in chars]
data.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(7.4, 4.6)); fig.subplots_adjust(left=0.42, right=0.93, top=0.86, bottom=0.13)
ax.axvspan(0.1, 1, alpha=0); ax.axvline(1, color='#999', lw=1.1, ls='-', zorder=1)
for i, (lbl, orr, lo, hi, pv) in enumerate(data):
    sig = pv < 0.05
    col = '#C0392B' if (sig and orr < 1) else ('#1B9E77' if sig else '#9AA0A6')
    ax.plot([lo, hi], [i, i], color=col, lw=2.2, zorder=2, solid_capstyle='round')
    ax.scatter([orr], [i], s=80, color=col, zorder=3, edgecolor='white', linewidth=1.1)
    ax.text(0.022, i, lbl, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=9)
    star = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'n.s.'
    ax.text(hi*1.05, i, star, va='center', fontsize=8.5, color=col, fontweight='bold' if sig else 'normal')
ax.set_xscale('log'); ax.set_xlim(0.12, 9); ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
ax.set_yticks([]); ax.set_ylim(-0.7, len(data)-0.3)
ax.set_xlabel('odds of being KEPT in the benchmark  (95% CI, log scale)', fontsize=9.5)
ax.text(1, len(data)-0.1, 'no bias', fontsize=8, color='#999', ha='center')
ax.annotate('under-represented', xy=(0.16, -0.55), fontsize=8, color='#C0392B', ha='left', style='italic')
ax.annotate('over-represented', xy=(8.6, -0.55), fontsize=8, color='#1B9E77', ha='right', style='italic')
ax.set_title('Is the 116-pair benchmark biased?  Odds of selection per trait',
             fontsize=11, fontweight='bold', loc='left', pad=12)
fig.savefig(os.path.join(FIGS, 'bias_v1_forest.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v1_forest.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)

# ======================================================== v2 — RECENCY
groups = [('kept', 'Kept (116)', KEPT), ('coverage', 'Dropped – no KG coverage (37)', ST_COL['coverage']),
          ('unresolved', 'Dropped – unresolved IDs (19)', ST_COL['unresolved']),
          ('leakage', 'Dropped – answer leakage (81)', ST_COL['leakage'])]
fig, ax = plt.subplots(figsize=(7.8, 4.4)); fig.subplots_adjust(left=0.30, right=0.96, top=0.84, bottom=0.13)
rng = np.random.default_rng(3)
for i, (key, lbl, col) in enumerate(groups):
    y = len(groups) - 1 - i
    v = p[p.stage == key].orig.dropna()
    jit = y + rng.uniform(-0.16, 0.16, len(v))
    ax.scatter(v, jit, s=24, color=col, alpha=0.55, edgecolor='white', linewidth=0.3, zorder=2)
    med = v.median()
    ax.plot([med, med], [y-0.32, y+0.32], color=col, lw=3, zorder=3, solid_capstyle='round')
    ax.text(2027.3, y, lbl, ha='left', va='center', fontsize=8.8)
    ax.text(med, y+0.40, f'{med:.0f}', ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')
ax.set_yticks([]); ax.set_ylim(-0.7, len(groups)-0.3); ax.set_xlim(1983, 2027)
ax.set_xlabel("drug's original approval year  (older ←        → newer)", fontsize=9.5)
ax.axvline(p[p.kept].orig.median(), color='#bbb', ls=':', lw=1, zorder=1)
ax.set_title('Answer-leakage removes the oldest drugs (p < 0.0001)', fontsize=11, fontweight='bold', loc='left', pad=22)
ax.text(0, 1.04, 'Each dot is a drug. Leakage drops well-established drugs already encoded in a KG; '
        'no-coverage drops the newest; the kept set sits between.', transform=ax.transAxes, fontsize=8, color='#666')
fig.savefig(os.path.join(FIGS, 'bias_v2_recency.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v2_recency.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)

# ======================================================== v3 — FUNNEL + composition
stages = [('253\ncandidates', 253, None), ('234\nharness-ready', 234, ('−19', 'unresolved IDs\n(CAR-Ts, placeholders)')),
          ('143\nleakage-free', 143, ('−91', 'answer leakage\n(older, encoded drugs)')),
          ('116\nevaluable', 116, ('−27', 'no KG coverage\n(newest drugs)'))]
fig, ax = plt.subplots(figsize=(6.6, 5.2)); fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.04)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ys = [8.4, 6.0, 3.6, 1.2]; maxw = 5.4
cols = ['#6E7B8B', '#4C72B0', '#1B9E77', '#1B9E77']
for i, ((lbl, n, drp), yy, col) in enumerate(zip(stages, ys, cols)):
    w = maxw * n / 253; x0 = 2.4 - w/2 + 1.0
    box = FancyBboxPatch((x0, yy-0.62), w, 1.24, boxstyle='round,pad=0.02,rounding_size=0.12',
                         linewidth=0, facecolor=col, alpha=0.92 if i == 3 else 0.8)
    ax.add_patch(box)
    ax.text(x0 + w/2, yy, lbl, ha='center', va='center', color='white', fontsize=12 if i==3 else 10.5, fontweight='bold')
    if i < 3:
        ax.annotate('', xy=(x0 + w/2, ys[i+1]+0.72), xytext=(x0 + w/2, yy-0.66),
                    arrowprops=dict(arrowstyle='-|>', color='#888', lw=1.6))
    if drp:
        my = (ys[i-1] + yy) / 2            # drop belongs to the transition INTO this stage
        ax.plot([6.2, 6.7], [my, my], color='#D95F02', lw=1.1, ls=(0, (2, 2)), zorder=1)
        ax.text(6.9, my, f"{drp[0]}", ha='left', va='center', color='#D95F02', fontsize=11.5, fontweight='bold')
        ax.text(7.55, my, drp[1], ha='left', va='center', color='#666', fontsize=8.2)
ax.text(0.04, 0.985, 'How 253 candidates become the 116-pair benchmark', transform=ax.transAxes,
        fontsize=11.5, fontweight='bold', va='top')
ax.text(0.04, 0.935, 'Each filter removes a different slice — the kept set is recency- and modality-shaped, not area-biased.',
        transform=ax.transAxes, fontsize=8.4, color='#666', va='top')
fig.savefig(os.path.join(FIGS, 'bias_v3_funnel.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v3_funnel.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)

# ======================================================== v4 — PAIRED diverging composition
def comp(col, order=None):
    ct = pd.crosstab(p[col], np.where(p.kept, 'kept', 'dropped'), normalize='columns')
    if order: ct = ct.reindex([o for o in order if o in ct.index])
    return ct
panels = [('modality', 'Drug modality', None),
          ('TA', 'Therapeutic area', p.TA.value_counts().index.tolist()),
          ('YearSet', 'Approval year', None)]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9)); fig.subplots_adjust(left=0.13, right=0.99, top=0.80, bottom=0.10, wspace=1.05)
for ax, (col, title, order) in zip(axes, panels):
    ct = comp(col, order); cats = ct.index.tolist(); y = np.arange(len(cats))
    ax.barh(y, -ct['dropped'].values, color=DROP, alpha=0.9, height=0.62)
    ax.barh(y, ct['kept'].values, color=KEPT, alpha=0.9, height=0.62)
    ax.axvline(0, color='#444', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=7.8, color='#333')
    ax.set_ylim(-0.6, len(cats)-0.4)
    mx = max(ct.max().max(), 0.4); ax.set_xlim(-mx*1.15, mx*1.15)
    ax.set_xticks([-mx, 0, mx]); ax.set_xticklabels([f'{mx:.0%}', '0', f'{mx:.0%}'], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
axes[0].text(-0.0, 1.0, '← dropped', transform=axes[0].transAxes, color=DROP, fontsize=8.5, ha='left', fontweight='bold')
axes[0].text(1.0, 1.0, 'kept →', transform=axes[0].transAxes, color=KEPT, fontsize=8.5, ha='right', fontweight='bold')
fig.suptitle('Composition of kept vs dropped pairs  (area & year balanced; modality differs)',
             fontsize=11.5, fontweight='bold', x=0.5, y=0.96)
fig.savefig(os.path.join(FIGS, 'bias_v4_paired.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'bias_v4_paired.pdf'), bbox_inches='tight', facecolor='white'); plt.close(fig)
print('wrote bias_v1_forest, v2_recency, v3_funnel, v4_paired')
