#!/usr/bin/env python3
"""
Confidence calibration: Hits@1 vs the model's self-reported confidence (1-5),
pooled across the three LLMs on KG-arm queries. Clean version — no side labels.

Usage:  python scripts/make_confidence_calibration.py
Output: results/figures/confidence_calibration.(png|pdf)
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'tables', '09_llm_runs')
FIGS = os.path.join(ROOT, 'results', 'figures'); os.makedirs(FIGS, exist_ok=True)
FILES = ['09_big_gpt.csv', '09_big_gpt_s12.csv', '09_big_glite.csv', '09_big_glite_s12.csv',
         '09_big_llama.csv', '09_big_llama_s1.csv', '09_big_llama_s2.csv']

pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in pref if f in avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 9, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'figure.dpi': 200})

d = pd.concat([pd.read_csv(os.path.join(RUNS, f)) for f in FILES], ignore_index=True)
d = d[(d['parsed'] == 1) & (d['condition'] == 'kg')]
g = d.groupby('pos_confidence').agg(h=('hit@1', 'mean'), n=('hit@1', 'size')).sort_index()
x = g.index.to_numpy(); y = g['h'].to_numpy()

fig, ax = plt.subplots(figsize=(5.6, 4.4)); fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.13)
ax.grid(axis='y', color='#ececec', lw=0.8, zorder=0)
ax.plot(x, y, color='#9a9a9a', lw=1.6, zorder=2)
ax.scatter(x, y, s=320, c=[cm.RdYlGn(v) for v in y], edgecolor='white', linewidth=1.4, zorder=3)
for xi, yi in zip(x, y):
    ax.annotate(f'{yi*100:.0f}%', (xi, yi), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color='#333')
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlim(0.6, 5.4); ax.set_ylim(-0.04, 1.08)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel('Model’s self-reported confidence (1–5)', fontsize=9.5)
ax.set_ylabel('Hits@1', fontsize=9.5)
ax.set_title('…and it knows when it has the evidence', fontsize=11, fontweight='bold', pad=10)

out = os.path.join(FIGS, 'confidence_calibration')
fig.savefig(out + '.png', bbox_inches='tight', facecolor='white')
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
print('wrote', out, '| n per conf:', dict(zip(x.astype(int), g['n'].astype(int))))
