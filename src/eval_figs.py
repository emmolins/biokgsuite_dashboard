"""Clean versions of the 02/03/04 evaluation figures, in notebook 00's house style.

Uses the shared `setup_style()` (Helvetica Neue, #333 text, light grid) and the
per-KG colours from `KG_PALETTE`, so 02/03/04 match 00 and the rest of the 00–08
set. Structural fixes only: rotated non-overlapping KG labels, readable layouts,
sentence-case descriptive titles. Notebooks call these as one-liners.
Verified headless against the cached pkls by scripts/refine_figs.py.
"""
import os, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
try:
    from .plotting import setup_style, KG_PALETTE, TEXT_COLOR, GRID_COLOR, HEATMAP_CMAP, ALERT_RED
except ImportError:
    from plotting import setup_style, KG_PALETTE, TEXT_COLOR, GRID_COLOR, HEATMAP_CMAP, ALERT_RED
setup_style()

KGS = ['primekg', 'hetionet', 'drkg', 'openbilink', 'biokg', 'matrix']
NICE = {'primekg': 'PrimeKG', 'hetionet': 'Hetionet', 'drkg': 'DRKG',
        'openbilink': 'OpenBioLink', 'biokg': 'BioKG', 'matrix': 'Matrix'}


def kgc(kg):
    return KG_PALETTE.get(kg, '#888888')          # Matrix (+any extra) -> grey, as in nb 00


def _present(cols):
    return [k for k in KGS if k in list(cols)]


def _xlabels(ax, names):
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NICE.get(n, n) for n in names], rotation=22, ha='right')


def _title(ax, title, subtitle=None, pad=10):
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left', color=TEXT_COLOR, pad=pad)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color='#777', va='bottom', ha='left')


def _df(data, index='KG'):
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if index in df.columns:
        df = df.set_index(index)
    return df


def _save(fig, folder, names):
    os.makedirs(folder, exist_ok=True)
    names = [names] if isinstance(names, str) else names
    for nm in names:
        for e in ('png', 'pdf'):
            fig.savefig(os.path.join(folder, f'{nm}.{e}'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ───────────────────────── 02 · entity validity heatmap ─────────────────────────
def fig02_validity(pivot_or_records, figs_dir, name='02_id_validation'):
    ROW = ['Drug/Compound', 'Gene/Protein', 'Disease', 'Biological Process', 'Molecular Function',
           'Cellular Component', 'Anatomy', 'Phenotype (HP)', 'Pathway', 'Exposure']
    if isinstance(pivot_or_records, pd.DataFrame) and 'KG' not in pivot_or_records.columns:
        piv = pivot_or_records
    else:
        v = _df(pivot_or_records, index=None)
        piv = v.pivot_table(index='Canonical Type', columns='KG', values='Rate (%)', aggfunc='mean')
    piv = piv.reindex(columns=_present(piv.columns))
    piv = piv.reindex([r for r in ROW if r in piv.index])
    piv = piv.reindex([r for r in piv.index if piv.loc[r].notna().all()])
    fig, ax = plt.subplots(figsize=(7.4, len(piv) * 0.62 + 1.8))
    im = ax.pcolormesh(piv.values.astype(float), cmap=HEATMAP_CMAP, vmin=0, vmax=100,
                       edgecolors='white', linewidth=2.5)
    ax.set_xticks([x + 0.5 for x in range(len(piv.columns))])
    ax.set_xticklabels([NICE.get(k, k) for k in piv.columns], rotation=22, ha='right')
    ax.set_yticks([y + 0.5 for y in range(len(piv.index))]); ax.set_yticklabels(piv.index)
    ax.tick_params(length=0); ax.invert_yaxis(); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            val = piv.iloc[i, j]
            if pd.notna(val):
                ax.text(j + 0.5, i + 0.5, f'{val:.0f}%', ha='center', va='center', fontsize=9,
                        fontweight='bold', color='white' if val >= 55 else TEXT_COLOR)
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label('validation rate (%)', fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    _title(ax, 'Entity validity rate by knowledge graph',
           'share of entities with a resolvable, type-consistent identifier', pad=34)
    fig.tight_layout(); _save(fig, figs_dir, name)


# ───────────────────────── 04 · graph cohesion ──────────────────────────────────
def fig04_cohesion(conn, figs_dir, name='04_graph_cohesion'):
    conn = _df(conn); conn = conn.reindex(_present(conn.index)); order = list(conn.index)
    cols = [kgc(k) for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    ax = axes[0]
    ax.bar(range(len(order)), conn['LCC (%)'], color=cols, edgecolor='white', linewidth=0.5, zorder=2)
    ax.axhline(95, color=ALERT_RED, ls='--', lw=1.2, label='95% threshold'); ax.set_ylim(0, 112)
    for i, val in enumerate(conn['LCC (%)']):
        ax.text(i, val + 1.5, f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold', color=TEXT_COLOR)
    _xlabels(ax, order); ax.set_ylabel('% of nodes in LCC'); ax.legend(loc='lower left')
    _title(ax, 'Largest connected component')
    ax = axes[1]
    comp = conn['Components'].values
    ax.bar(range(len(order)), comp, color=cols, edgecolor='white', linewidth=0.5, zorder=2); ax.set_yscale('log')
    for i, val in enumerate(comp):
        ax.text(i, val * 1.25, f'{int(val):,}', ha='center', fontsize=8, fontweight='bold', color=TEXT_COLOR)
    _xlabels(ax, order); ax.set_ylabel('component count (log scale)')
    _title(ax, 'Connected components')
    fig.tight_layout(); _save(fig, figs_dir, name)


# ───────────────────────── 04 · clustering coefficient ──────────────────────────
def fig04_clustering(clust, figs_dir, name='04_clustering_coefficient'):
    cl = _df(clust); cl = cl.reindex(_present(cl.index)); order = list(cl.index)
    cols = [kgc(k) for k in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.bar(range(len(order)), cl['C / C_ER'], color=cols, edgecolor='white', linewidth=0.5, zorder=2); ax.set_yscale('log')
    for i, val in enumerate(cl['C / C_ER']):
        ax.text(i, val * 1.18, f'{val:.0f}×', ha='center', fontsize=8, fontweight='bold', color=TEXT_COLOR)
    ax.axhline(1, color='#bbb', ls=':', lw=1)
    _xlabels(ax, order); ax.set_ylabel('clustering vs random  (C / C$_{ER}$, log)')
    _title(ax, 'Clustering coefficient relative to a random graph', 'values > 1 indicate small-world structure', pad=12)
    fig.tight_layout(); _save(fig, figs_dir, name)


# ───────────────────────── 04c · known-pair recovery ────────────────────────────
def fig04_recovery(recovery_results, figs_dir, name='04_known_pair_recovery'):
    ks = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for kg in _present(recovery_results.keys()):
        r = recovery_results.get(kg, {}).get('recovery', {})
        ys = [r.get(k, np.nan) * 100 for k in ks]
        ax.plot(ks, ys, marker='o', ms=5, lw=1.6, color=kgc(kg), mfc='white', mec=kgc(kg),
                mew=1.4, label=NICE.get(kg, kg))
    ax.set_xticks(ks); ax.set_xlabel('hop distance (k)')
    ax.set_ylabel('% of known pairs recovered within k hops'); ax.set_ylim(0, 104)
    ax.legend(loc='lower right', ncol=2)
    _title(ax, 'Known drug–disease pair recovery by hop distance')
    fig.tight_layout(); _save(fig, figs_dir, [name, name + '_standalone'])


# ───────────────────────── 04 · differential resilience (dropout curves) ────────
def fig04_resilience(diversity_results, figs_dir, name='04_differential_resilience'):
    from matplotlib.lines import Line2D
    kgs = _present(diversity_results.keys())
    ncol = len(kgs)
    fig, axes = plt.subplots(1, ncol, figsize=(2.7 * ncol + 0.6, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kg in zip(axes, kgs):
        r = diversity_results[kg]
        x = [v * 100 for v in r['drop_rates']]
        gc, rc = r['gold_curve'], r['rand_curve']; c = kgc(kg)
        ax.plot(x, gc, '-o', color=c, lw=2, ms=5, label='gold standard')
        ax.plot(x, rc, '--s', color='#9A9A9A', lw=1.8, ms=5, label='random pairs')
        ax.fill_between(x, rc, gc, where=[g >= rr for g, rr in zip(gc, rc)], alpha=0.15, color=c)
        ax.text(0.96, 0.05, f"DR {r['dr_score']:.2f}", transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, fontweight='bold', color=TEXT_COLOR,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ddd'))
        ax.set_title(NICE.get(kg, kg), fontsize=10.5, fontweight='bold', color=TEXT_COLOR)
        ax.set_xlabel('edge dropout (%)', fontsize=8.5); ax.set_ylim(0, 1.05); ax.set_xlim(-2, max(x) + 2)
    axes[0].set_ylabel('recovery fraction', fontsize=9)
    axes[0].legend(handles=[Line2D([0], [0], color='#555', lw=2, marker='o', ms=5, label='gold standard'),
                            Line2D([0], [0], color='#9A9A9A', lw=1.8, ls='--', marker='s', ms=5, label='random pairs')],
                   loc='lower left', fontsize=8)
    fig.suptitle('Differential resilience to edge dropout: gold vs random pairs', fontsize=12,
                 fontweight='bold', x=0.012, ha='left', color=TEXT_COLOR, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); _save(fig, figs_dir, name)


# ───────────────────────── 05 · ranking stability under dropout ─────────────────
def fig05_stability(raw_r, dropout_rates, figs_dir, name='05_stability'):
    from matplotlib.lines import Line2D
    kgs = _present(raw_r.keys())
    pct = [r * 100 for r in dropout_rates]
    ncol = 3; nrow = math.ceil(len(kgs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), sharey=True)
    axes = np.array(axes).reshape(-1)
    for idx, (ax, kg) in enumerate(zip(axes, kgs)):
        rnd = [raw_r[kg]['random'][r] for r in dropout_rates]
        per = [raw_r[kg]['periphery'][r] for r in dropout_rates]
        ax.plot(pct, rnd, '-o', color=kgc(kg), lw=2, ms=5, label='random dropout')
        ax.plot(pct, per, '--s', color='#555555', lw=1.8, ms=5, label='periphery dropout')
        ax.axhline(0.95, color='#655F55', ls='--', lw=0.9, alpha=0.8)
        ax.axhline(0.85, color='#E69F00', ls='--', lw=0.9, alpha=0.8)
        ax.set_xticks([10, 20, 30]); ax.set_xticklabels(['10%', '20%', '30%'])
        ax.set_ylim(0.65, 1.02)
        if idx % ncol == 0:
            ax.set_ylabel('Spearman r')
        if idx >= len(kgs) - ncol:
            ax.set_xlabel('edge dropout (%)')
        ax.set_title(NICE.get(kg, kg), fontsize=10.5, fontweight='bold', loc='left', color=TEXT_COLOR, pad=6)
    for ax in axes[len(kgs):]:
        ax.set_visible(False)
    axes[0].legend(handles=[Line2D([0], [0], color='#555', lw=2, marker='o', ms=5, label='random dropout'),
                            Line2D([0], [0], color='#555', lw=1.8, ls='--', marker='s', ms=5, label='periphery dropout'),
                            Line2D([0], [0], color='#655F55', lw=0.9, ls='--', label='r = 0.95'),
                            Line2D([0], [0], color='#E69F00', lw=0.9, ls='--', label='r = 0.85')],
                   loc='lower left', fontsize=7.5)
    fig.suptitle('Ranking stability under edge dropout', fontsize=13, fontweight='bold',
                 x=0.012, ha='left', color=TEXT_COLOR, y=0.99)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.10, wspace=0.18, hspace=0.42)
    _save(fig, figs_dir, name)


# ───────────────────────── 06 · link-prediction heatmap (decluttered) ───────────
def fig06_link_heatmap(link_records, primary_heuristic, figs_dir, name='06_link_prediction_heatmap'):
    df = pd.DataFrame(link_records)
    df = df[df['heuristic'] == primary_heuristic]
    SNICE = {'random': 'Random', 'type-constrained': 'Type-constrained', 'shared-target': 'Shared-target'}
    strat = [s for s in ['random', 'type-constrained', 'shared-target'] if s in set(df['strategy'])] or list(df['strategy'].unique())
    kgs = _present(df['kg'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, metric in zip(axes, ['auroc', 'auprc']):
        piv = df.pivot_table(index='kg', columns='strategy', values=metric, aggfunc='mean').reindex(index=kgs, columns=strat)
        im = ax.pcolormesh(piv.values.astype(float), cmap=HEATMAP_CMAP, vmin=0.45, vmax=1.0, edgecolors='white', linewidth=2)
        ax.set_xticks([x + 0.5 for x in range(len(strat))]); ax.set_xticklabels([SNICE.get(s, s) for s in strat], rotation=20, ha='right')
        ax.set_yticks([y + 0.5 for y in range(len(kgs))]); ax.set_yticklabels([NICE.get(k, k) for k in kgs])
        ax.tick_params(length=0); ax.invert_yaxis(); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        for i in range(len(kgs)):
            for j in range(len(strat)):
                v = piv.iloc[i, j]
                if pd.notna(v):
                    ax.text(j + 0.5, i + 0.5, f'{v:.3f}', ha='center', va='center', fontsize=9,
                            fontweight='bold', color='white' if v > 0.7 else TEXT_COLOR)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'{metric.upper()}  ({primary_heuristic})', fontsize=10.5, fontweight='bold', loc='left', color=TEXT_COLOR, pad=8)
    fig.suptitle('Link-prediction performance by negative-sampling strategy', fontsize=12,
                 fontweight='bold', x=0.012, ha='left', color=TEXT_COLOR, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); _save(fig, figs_dir, name)


# ───────────────────────── 07 · network proximity (redesigned scatter) ──────────
def fig07_proximity(all_proximity, figs_dir, name='07_network_proximity'):
    kgs = [k for k in _present(all_proximity.keys()) if all_proximity.get(k)]
    ncol = min(3, len(kgs)); nrow = math.ceil(len(kgs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.5 * nrow), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, kg in zip(axes, kgs):
        ok = [r for r in all_proximity[kg] if 'obs_dist' in r]
        rx = [r['random_mean'] for r in ok]; oy = [r['obs_dist'] for r in ok]
        sig = [bool(r.get('significant')) for r in ok]
        lim = max(max(rx + oy, default=1), 1) * 1.05
        ax.plot([0, lim], [0, lim], ls='--', color='#bbb', lw=1, zorder=1)
        ax.scatter([rx[i] for i in range(len(ok)) if not sig[i]], [oy[i] for i in range(len(ok)) if not sig[i]],
                   s=16, color='#CCCCCC', edgecolor='white', lw=0.3, zorder=2, label='n.s.')
        ax.scatter([rx[i] for i in range(len(ok)) if sig[i]], [oy[i] for i in range(len(ok)) if sig[i]],
                   s=20, color=kgc(kg), edgecolor='white', lw=0.3, zorder=3, label='closer than random')
        frac = 100 * sum(sig) / len(ok) if ok else 0
        ax.text(0.04, 0.96, f'{frac:.0f}% significant\n(n={len(ok)})', transform=ax.transAxes,
                va='top', fontsize=8, color=TEXT_COLOR)
        ax.set_title(NICE.get(kg, kg), fontsize=10.5, fontweight='bold', loc='left', color=TEXT_COLOR, pad=6)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
    for ax in axes[len(kgs):]: ax.set_visible(False)
    fig.supxlabel('expected distance (random mean)', fontsize=9.5)
    fig.supylabel('observed distance', fontsize=9.5)
    axes[0].legend(loc='lower right', fontsize=7.5)
    fig.suptitle('Network proximity of true drug–disease pairs vs random expectation',
                 fontsize=12, fontweight='bold', x=0.012, ha='left', color=TEXT_COLOR, y=1.0)
    fig.text(0.012, 0.945, 'points below the diagonal are closer than random expectation', fontsize=9, color='#777', ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93]); _save(fig, figs_dir, name)


# ───────────────────────── 07a · prospective AUROC (grouped bars) ────────────────
def fig07a_prospective(baseline_all, baseline_same, new_pairs, figs_dir, name='07a_prospective_auroc'):
    kgs = [k for k in _present(new_pairs.keys()) if not np.isnan(new_pairs.get(k, {}).get('auroc', np.nan))]
    series = [('Baseline (all)', baseline_all, KG_PALETTE['drkg']),
              ('Baseline (same)', baseline_same, KG_PALETTE['primekg']),
              ('New pairs', new_pairs, KG_PALETTE['hetionet'])]
    x = np.arange(len(kgs)); w = 0.26
    fig, ax = plt.subplots(figsize=(1.5 * len(kgs) + 2.2, 4.3))
    for i, (label, src, color) in enumerate(series):
        vals = [src.get(k, {}).get('auroc', np.nan) for k in kgs]
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, edgecolor='white', linewidth=0.5, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([NICE.get(k, k) for k in kgs])
    ax.set_ylim(0.45, 1.0); ax.set_ylabel('AUROC')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8.5)
    ax.set_title('Prospective AUROC: new pairs vs baselines', fontsize=11.5, fontweight='bold',
                 loc='left', color=TEXT_COLOR, pad=30)
    fig.tight_layout(); _save(fig, figs_dir, name)


# ───────────────────────── 03 · source-database diversity (redesigned) ──────────
def fig03_source_diversity(kg_counts, panel_config, figs_dir, name='03_source_diversity', top_n=8):
    kgs = [k for k in panel_config if k in kg_counts]
    ncol = 3; nrow = math.ceil(len(kgs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow + 0.5))
    axes = np.array(axes).reshape(-1)
    for ax, kg in zip(axes, kgs):
        cfg = panel_config[kg]; col = cfg['label_col']
        d = kg_counts[kg].sort_values('n_edges', ascending=False).head(top_n).iloc[::-1]
        lab = [str(s)[:24] + ('…' if len(str(s)) > 24 else '') for s in d[col]]
        ax.barh(range(len(d)), d['n_edges'], color=kgc(kg), edgecolor='white', lw=0.5, zorder=2)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(lab, fontsize=7.5)
        ax.set_xlabel('edge count', fontsize=8.5); ax.tick_params(axis='x', labelsize=7.5)
        ax.set_title(cfg['title'], fontsize=10.5, fontweight='bold', loc='left', color=TEXT_COLOR, pad=6)
        ax.set_xlim(0, d['n_edges'].max() * 1.18)
    for ax in axes[len(kgs):]:
        ax.set_visible(False)
    fig.suptitle('Source-database diversity across knowledge graphs', fontsize=13,
                 fontweight='bold', x=0.015, ha='left', color=TEXT_COLOR, y=0.99)
    fig.text(0.015, 0.945, f'top {top_n} sources by edge count per knowledge graph',
             fontsize=9.5, color='#777', ha='left')
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.09, wspace=0.55, hspace=0.55)
    _save(fig, figs_dir, name)
