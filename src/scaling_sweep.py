"""
Same-family (Llama 3.x) capability-scaling sweep for the LLM x KG study.

Question: does the KG lift (MRR_kg - MRR_nokg) grow, shrink, or stay flat as the
model scales from 1B -> 405B parameters?  A within-family ladder isolates the
capability axis (architecture / training recipe held roughly fixed) in a way the
cross-family GPT/Gemini/Llama comparison cannot.

Memorization confound: bigger/newer models memorise more drug-disease pairs,
inflating the no-KG baseline and mechanically shrinking apparent lift.  So every
figure splits the KG arm into *covered* (the graph actually contains the pair --
memorization can't be the whole story) vs *all arms pooled*.

Single source of truth: notebooks import `compute_scaling` + the `fig_*`
functions; nothing is hard-coded.  Robust to having only a subset of the ladder
run so far (fills in as HPC jobs finish).
"""
import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

_pref = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _pref if f in _avail), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'axes.linewidth': 0.8, 'savefig.dpi': 300, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

# ---- the Llama 3.x ladder: raw model tag -> (params in B, generation, nice label) ----
# NB the ladder spans three generations (no clean single-gen 5-point ladder exists in Llama):
#   1B/3B = Llama 3.2 · 8B = Llama 3.1 · 70B = Llama 3.3 · 405B = Llama 3.1
# Generation is carried through so figures can mark it and the 3.1 spine (8B->405B) reads cleanly.
LADDER = {
    'llama3.2:1b':   (1.0,   '3.2', 'Llama-3.2-1B'),
    'llama3.2:3b':   (3.0,   '3.2', 'Llama-3.2-3B'),
    'llama3.1:8b':   (8.0,   '3.1', 'Llama-3.1-8B'),
    'llama3.1:70b':  (70.0,  '3.1', 'Llama-3.1-70B'),   # optional clean-spine 70B
    'llama3.3:70b':  (70.0,  '3.3', 'Llama-3.3-70B'),   # the 70B already run
    'llama3.1:405b': (405.0, '3.1', 'Llama-3.1-405B'),
}
GEN_MARK = {'3.1': 'o', '3.2': 's', '3.3': 'D'}
C_ALL = '#9AA0A6'    # pooled (all 3 KG arms)
C_COV = '#1B9E77'    # covered arm only
C_NOKG = '#C44E52'   # no-KG baseline


def load_llama_runs(runs_dir):
    """Combine all 09_big_*.csv, keep parsed rows, restrict to Llama-ladder models."""
    fs = glob.glob(os.path.join(runs_dir, '09_big_*.csv'))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[(df['parsed'] == 1) & (df['model'].isin(LADDER))].copy()
    df['params'] = df['model'].map(lambda m: LADDER[m][0])
    df['gen'] = df['model'].map(lambda m: LADDER[m][1])
    df['label'] = df['model'].map(lambda m: LADDER[m][2])
    return df


def _covmap(cov_csv):
    cov = pd.read_csv(cov_csv)
    cvd = cov.groupby(['disease_name', 'kg'])['covered'].max().reset_index()
    return {(r.disease_name, r.kg): int(r.covered) for r in cvd.itertuples()}


def _boot_ci(vals, n_boot=2000, seed=0):
    """Percentile CI from a cluster (per-disease) bootstrap of a difference vector."""
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def compute_scaling(df, cov_csv):
    """Per-model scaling table: MRR on each arm + KG lift (pooled and covered-only), with bootstrap CIs.

    Lift is computed per disease (mean KG-arm RR - no-KG RR) then averaged, so the CI is a
    cluster bootstrap over the 116 diseases — the same unit used elsewhere in the study.
    """
    cm = _covmap(cov_csv)
    df = df.copy()
    df['covered'] = df.apply(
        lambda r: cm.get((r.disease, r.kg), np.nan) if r.condition == 'kg' else np.nan, axis=1)
    rows = []
    for m, sub in df.groupby('model'):
        params, gen, label = LADDER[m]
        nokg = sub[sub.condition == 'no_kg'].groupby('disease')['reciprocal_rank'].mean()
        kg_all = sub[sub.condition == 'kg'].groupby('disease')['reciprocal_rank'].mean()
        kg_cov = sub[(sub.condition == 'kg') & (sub.covered == 1)].groupby('disease')['reciprocal_rank'].mean()
        lift_all = (kg_all - nokg).dropna().values
        lift_cov = (kg_cov - nokg).dropna().values
        ci_all, ci_cov = _boot_ci(lift_all), _boot_ci(lift_cov)
        rows.append(dict(
            model=m, label=label, params=params, gen=gen,
            mrr_nokg=nokg.mean(), mrr_kg=kg_all.mean(), mrr_kg_cov=kg_cov.mean(),
            lift=lift_all.mean(), lift_lo=ci_all[0], lift_hi=ci_all[1],
            lift_cov=lift_cov.mean(), lift_cov_lo=ci_cov[0], lift_cov_hi=ci_cov[1],
            n_dis=len(lift_all)))
    out = pd.DataFrame(rows).sort_values('params').reset_index(drop=True)
    return out


# ----------------------------- figures -----------------------------
def _logx(ax, params):
    ax.set_xscale('log')
    ax.set_xticks(sorted(set(params)))
    ax.set_xticklabels([f'{int(p)}B' for p in sorted(set(params))])
    ax.set_xlabel('model size (parameters, log scale)')


def fig_lift_vs_size(scaling):
    """Headline: KG lift vs model size — pooled vs covered-arm, error bars = 95% cluster bootstrap CI."""
    s = scaling.sort_values('params')
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.axhline(0, color='#bbb', ls=':', lw=1.2)
    # pooled
    ax.errorbar(s.params, s.lift, yerr=[s.lift - s.lift_lo, s.lift_hi - s.lift],
                fmt='-', color=C_ALL, lw=2, capsize=3, zorder=2, label='pooled (all 3 KG arms)')
    # covered-only
    ax.errorbar(s.params, s.lift_cov, yerr=[s.lift_cov - s.lift_cov_lo, s.lift_cov_hi - s.lift_cov],
                fmt='-', color=C_COV, lw=2, capsize=3, zorder=3, label='covered arm only')
    # generation-coded markers on the pooled line
    for _, r in s.iterrows():
        ax.scatter(r.params, r.lift, marker=GEN_MARK.get(r.gen, 'o'), s=85,
                   color=C_ALL, edgecolor='white', linewidth=1, zorder=4)
        ax.scatter(r.params, r.lift_cov, marker=GEN_MARK.get(r.gen, 'o'), s=85,
                   color=C_COV, edgecolor='white', linewidth=1, zorder=5)
    _logx(ax, s.params)
    ax.set_ylabel('KG lift  (MRR$_{KG}$ − MRR$_{no\\,KG}$)')
    ax.set_title('Does the KG lift grow or shrink as the model scales?',
                 fontsize=12.5, fontweight='bold', loc='left', pad=24)
    ax.text(0, 1.05, 'Covered-arm lift isolates KG use from memorization (which inflates the no-KG baseline).',
            transform=ax.transAxes, fontsize=8.4, color='#666')
    gens = sorted(s.gen.unique())
    gleg = [Line2D([0], [0], marker=GEN_MARK.get(g, 'o'), color='#666', ls='', ms=8,
                   label=f'Llama {g}') for g in gens]
    leg1 = ax.legend(loc='upper right', fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=gleg, loc='lower right', fontsize=8.5, title='generation', title_fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_mrr_vs_size(scaling):
    """Diagnostic: no-KG vs KG-arm MRR across size — a rising no-KG line is the memorization signal."""
    s = scaling.sort_values('params')
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(s.params, s.mrr_nokg, '-o', color=C_NOKG, lw=2, ms=9, label='no-KG (prior only)')
    ax.plot(s.params, s.mrr_kg, '-o', color=C_ALL, lw=2, ms=9, label='KG arm (pooled)')
    ax.plot(s.params, s.mrr_kg_cov, '-D', color=C_COV, lw=2, ms=9, label='KG arm (covered only)')
    for _, r in s.iterrows():
        for y in (r.mrr_nokg, r.mrr_kg, r.mrr_kg_cov):
            ax.annotate(f'{y:.2f}', (r.params, y), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=7.5, color='#666')
    _logx(ax, s.params)
    ax.set_ylabel('MRR on the 116-set')
    ax.set_ylim(0, 1)
    ax.set_title('Where the lift comes from: prior vs KG-grounded ranking by size',
                 fontsize=12.5, fontweight='bold', loc='left', pad=14)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    return fig
