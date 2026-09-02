#!/usr/bin/env python3
"""
Analysis 1 — which drug-disease pairs are hard for the LLMs to rank?

Aggregates KG-arm runs per disease (pooled over 3 models, seeds, shuffles),
joins drug identity + therapeutic area (gold v4_bigset) and KG coverage
(usable_pairs_v2_audited), ranks the hardest pairs, separates intrinsic
(hard for all models) from model-specific difficulty, and characterizes
hard vs easy by area / confidence / evidence-agreement / coverage.

Out: results/tables/analysis1_hard_pairs.csv
     results/figures/analysis1_hardest_pairs.(png|pdf)
     results/figures/analysis1_by_area.(png|pdf)
"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'tables', '09_llm_runs')
GS = os.path.join(ROOT, 'data', 'gold_standards')
FIGS = os.path.join(ROOT, 'results', 'figures'); os.makedirs(FIGS, exist_ok=True)
TAB = os.path.join(ROOT, 'results', 'tables')
MODELS = {'gpt': '#0F9D8C', 'gemini': '#5B4FB8', 'llama': '#D1791F'}

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'figure.dpi': 200, 'legend.frameon': False})

# ---- load runs ----
def fam(f):
    n = os.path.basename(f)
    return 'gpt' if 'gpt' in n else 'gemini' if 'glite' in n else 'llama'
parts = []
for f in glob.glob(os.path.join(RUNS, '09_big_*.csv')):
    x = pd.read_csv(f); x['fam'] = fam(f); parts.append(x)
d = pd.concat(parts, ignore_index=True)
d = d[d['parsed'] == 1].copy()
kg = d[d['condition'] == 'kg'].copy()
nokg = d[d['condition'] == 'no_kg'].copy()

# ---- per-disease aggregates ----
per = kg.groupby('disease').agg(
    mrr_kg=('reciprocal_rank', 'mean'), hit1_kg=('hit@1', 'mean'),
    conf=('pos_confidence', 'mean'), n=('reciprocal_rank', 'size')).reset_index()
per = per.merge(nokg.groupby('disease')['reciprocal_rank'].mean().rename('mrr_nokg'),
                on='disease', how='left')
per['lift'] = per['mrr_kg'] - per['mrr_nokg']
# per-model MRR (KG arm)
for m in MODELS:
    mm = kg[kg.fam == m].groupby('disease')['reciprocal_rank'].mean().rename(f'mrr_{m}')
    per = per.merge(mm, on='disease', how='left')
# agreement mix (% consistent)
agg_mix = kg.assign(cons=(kg['pos_agreement'] == 'consistent').astype(int)) \
            .groupby('disease')['cons'].mean().rename('frac_consistent')
per = per.merge(agg_mix, on='disease', how='left')

# ---- join drug identity + therapeutic area + coverage ----
g = pd.read_csv(os.path.join(GS, 'gold_standard_v4_bigset.tsv'), sep='\t')
meta = g.groupby('disease_name').agg(drug=('drug_name', lambda s: ' / '.join(sorted(set(s)))),
                                     area=('therapeutic_area', 'first'),
                                     disease_id=('disease_id', 'first')).reset_index()
per = per.merge(meta, left_on='disease', right_on='disease_name', how='left')
aud = pd.read_csv(os.path.join(GS, 'usable_pairs_v2_audited.csv'))
aud['n3'] = aud.eval_primekg + aud.eval_drkg + aud.eval_biokg
covmap = aud.groupby('DOID_ID')['n3'].max()
per['n_eval_kgs'] = per['disease_id'].map(covmap)

per = per.sort_values('mrr_kg').reset_index(drop=True)
per.to_csv(os.path.join(TAB, 'analysis1_hard_pairs.csv'), index=False, float_format='%.4f')

# intrinsic vs model-specific
mcols = [f'mrr_{m}' for m in MODELS]
per['model_min'] = per[mcols].min(axis=1); per['model_max'] = per[mcols].max(axis=1)
per['model_spread'] = per['model_max'] - per['model_min']
corr = per[mcols].corr().mean().mean()

# ============ FIG 1: hardest 20 pairs ============
h = per.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.6, 6.2)); fig.subplots_adjust(left=0.46, right=0.97, top=0.93, bottom=0.08)
y = np.arange(len(h))
ax.barh(y, h['mrr_kg'], color='#d9d9d9', height=0.62, zorder=1)
for m, c in MODELS.items():
    ax.scatter(h[f'mrr_{m}'], y, s=22, color=c, zorder=3, label=m.upper(), edgecolor='white', linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels([f'{r.disease[:42]}\n({r.drug[:30]} · {r.area})' for r in h.itertuples()], fontsize=6.6)
ax.set_xlabel('Mean MRR on KG arm  (grey = pooled; dots = per model)', fontsize=8.5)
ax.set_xlim(0, 1.0)
ax.legend(loc='lower right', fontsize=8, title='per-model MRR', title_fontsize=7.5)
ax.set_title('Analysis 1 — 20 hardest-to-rank pairs', fontsize=10.5, fontweight='bold', pad=8)
fig.savefig(os.path.join(FIGS, 'analysis1_hardest_pairs.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'analysis1_hardest_pairs.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============ FIG 2: difficulty by therapeutic area ============
ta = per.groupby('area').agg(mrr=('mrr_kg', 'mean'), n=('disease', 'size')).query('n>=2').sort_values('mrr')
fig, ax = plt.subplots(figsize=(6.4, 4.4)); fig.subplots_adjust(left=0.24, right=0.96, top=0.9, bottom=0.12)
yy = np.arange(len(ta))
ax.barh(yy, ta['mrr'], color='#5B4FB8', height=0.66)
ax.set_yticks(yy); ax.set_yticklabels([f'{i}  (n={int(r.n)})' for i, r in ta.iterrows()], fontsize=8)
ax.set_xlabel('Mean MRR on KG arm', fontsize=8.5); ax.set_xlim(0, 1.0)
ax.axvline(per['mrr_kg'].mean(), color='#888', ls='--', lw=1, label=f'overall mean {per.mrr_kg.mean():.2f}')
ax.legend(fontsize=7.5, loc='lower right')
ax.set_title('Analysis 1 — difficulty by therapeutic area', fontsize=10.5, fontweight='bold', pad=8)
fig.savefig(os.path.join(FIGS, 'analysis1_by_area.png'), bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGS, 'analysis1_by_area.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# ---- console summary ----
hard = per[per.mrr_kg < per.mrr_kg.quantile(1/3)]; easy = per[per.mrr_kg > per.mrr_kg.quantile(2/3)]
print(f"diseases={len(per)}  mean MRR(kg)={per.mrr_kg.mean():.3f}")
print(f"cross-model MRR corr (avg pairwise)={corr:.3f}  -> difficulty is {'mostly intrinsic' if corr>0.6 else 'partly model-specific'}")
print(f"HARD tertile (n={len(hard)}): mean conf={hard.conf.mean():.2f}, frac_consistent={hard.frac_consistent.mean():.2f}, n_eval_kgs={hard.n_eval_kgs.mean():.2f}")
print(f"EASY tertile (n={len(easy)}): mean conf={easy.conf.mean():.2f}, frac_consistent={easy.frac_consistent.mean():.2f}, n_eval_kgs={easy.n_eval_kgs.mean():.2f}")
print("\nTop area composition (hard tertile):"); print(hard.area.value_counts().head(6).to_string())
print("\n5 hardest:"); print(per[['disease','drug','area','mrr_kg','hit1_kg','n_eval_kgs']].head(5).to_string(index=False))
