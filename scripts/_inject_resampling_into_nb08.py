#!/usr/bin/env python3
# ─── INTERNAL BUILD SCRIPT ────────────────────────────────────────────────
# Not part of the user-facing analysis pipeline. This script modifies the
# target notebook in-place to add a new analysis section. Idempotent — safe
# to re-run; will no-op if the section is already injected.
#
# Run order matters if you're regenerating notebooks from a clean copy:
#   1. scripts/_inject_gemma_into_nb08.py
#   2. scripts/_inject_resampling_into_nb08.py
#   3. scripts/_inject_strategies_into_nb09.py
#
# These scripts exist so the additions can be reproduced or re-applied to a
# fresh checkout of the upstream notebooks without manual cell-by-cell editing.
# ──────────────────────────────────────────────────────────────────────────

"""Inject multi-rerun resampling section into nb08. Idempotent.

What this adds:
  - N_RERUNS=5 config in cell 6 (resampling hyperparameter)
  - prepare_kg_resampled(): rerun-aware prep that gives a different held-out
    10% per rerun_seed; caches per (kg, rerun_idx)
  - Multi-rerun training loop for TransE/RotatE on all 6 KGs
  - Gemma re-scoring loop (encode once, re-score per rerun — much cheaper)
  - Long-form comparison DataFrame with `rerun` column
  - Headline figure with rerun-empirical CIs (replaces bootstrap-CI display)
  - Stability box plot showing per-rerun AUROC spread
  - Bootstrap-vs-rerun gap analysis (does bootstrap underestimate variance?)
  - Prose explaining the methodology shift

Idempotent: looks for `resampling_injected_v1` marker; refuses to double-inject.

Run from repo root: python scripts/_inject_resampling_into_nb08.py
"""
import json, sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'eval_notebooks' / '08_embedding_validation.ipynb'
MARKER = 'prepare_kg_resampled'   # unique identifier introduced by this injection


def code_cell(src):
    return {
        'cell_type': 'code', 'metadata': {},
        'source': src.splitlines(keepends=True),
        'execution_count': None, 'outputs': [],
    }

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': src.splitlines(keepends=True)}


CONFIG_ADDITION = '''
# ── RESAMPLING / MULTI-RERUN CONFIG (new) ──────────────────────────────
# Bootstrap CIs (existing Tier-2 stability cells) only capture variance from
# test-pair resampling within a SINGLE training run. They miss the much
# larger variance from training stochasticity (random init, negative
# sampling, batch order) and from train/test split sensitivity.
#
# These reruns address that: each rerun draws a different random 10% of
# drug-disease indication edges as the test set, trains TransE/RotatE from
# scratch, and re-scores Gemma. Final reported numbers are mean ± empirical
# CI across the 5 reruns. This is the "Tier 0" stability analysis (most
# fundamental — captures both data-split variance and training variance).
N_RERUNS    = 5
RERUN_SEEDS = [SEED + 1000 * i for i in range(N_RERUNS)]
print(f'Resampling: {N_RERUNS} reruns, seeds = {RERUN_SEEDS}')
print(f'Total fresh trainings to run: {N_RERUNS} × {len(MODELS)} × {len(KG_NAMES)} = '
      f'{N_RERUNS * len(MODELS) * len(KG_NAMES)}')
'''


# The full resampling section. Multiple cells.
PREP_CELL = '''# ── Multi-rerun preparation: different random 10% held out per rerun ──
# Identical to prepare_kg_multistrat but cached per (kg_name, rerun_idx).
# Each rerun's seed perturbs both the test-split and the negative-sampling.

def prepare_kg_resampled(kg_name, rerun_idx, neg_ratio=NEG_RATIO):
    """Like prepare_kg_multistrat but parameterized by rerun_idx."""
    rerun_seed = RERUN_SEEDS[rerun_idx]
    cache_path = CACHE / f'{{kg_name}}_prep_rerun{{rerun_idx}}.pkl'
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        if data is not None and 'neg_by_strategy' in data:
            print(f'  {{kg_name}}/rerun{{rerun_idx}}: loaded from cache')
            return data

    kg_df, nodes_df = load_kg(kg_name, config)
    kg_cfg = config['knowledge_graphs'][kg_name]
    etypes = kg_cfg['entity_types']
    type_map = dict(zip(nodes_df['idx'], nodes_df['type']))

    drug_idx    = {{i for i, t in type_map.items() if t == etypes.get('Drug', 'Drug')}}
    disease_idx = {{i for i, t in type_map.items() if t == etypes.get('Disease', 'Disease')}}
    gene_idx    = {{i for i, t in type_map.items() if t == etypes.get('Gene/Protein', 'Gene')}}

    dd = kg_cfg.get('relations', {{}}).get('drug_disease', {{}})
    ind_rels = [dd['relation']] if 'relation' in dd else dd.get('relations', [])
    dt = kg_cfg.get('relations', {{}}).get('drug_target', {{}})
    dt_rels = [dt['relation']] if 'relation' in dt else dt.get('relations', [])

    mask = kg_df['relation'].isin(ind_rels)
    sub  = kg_df.loc[mask, ['x_index', 'y_index']].astype('int64')
    h, t = sub['x_index'].values, sub['y_index'].values
    drug_arr    = np.fromiter(drug_idx, dtype='int64', count=len(drug_idx))
    disease_arr = np.fromiter(disease_idx, dtype='int64', count=len(disease_idx))
    drug_set, disease_set = set(drug_arr.tolist()), set(disease_arr.tolist())
    fwd_h_in_drug    = np.array([x in drug_set    for x in h])
    fwd_t_in_disease = np.array([x in disease_set for x in t])
    rev_h_in_disease = np.array([x in disease_set for x in h])
    rev_t_in_drug    = np.array([x in drug_set    for x in t])
    fwd_mask = fwd_h_in_drug & fwd_t_in_disease
    rev_mask = rev_h_in_disease & rev_t_in_drug
    pairs_set = set()
    pairs_set.update(zip(h[fwd_mask].tolist(), t[fwd_mask].tolist()))
    pairs_set.update(zip(t[rev_mask].tolist(), h[rev_mask].tolist()))
    pairs = list(pairs_set)

    # Train/test split with rerun_seed
    rng = np.random.RandomState(rerun_seed)
    perm = rng.permutation(len(pairs))
    split = int(0.9 * len(pairs))
    test_pos = [pairs[i] for i in perm[split:]]
    all_pos = set(pairs)

    train_triples, rel_to_idx, idx_to_rel = build_train_triples(
        kg_df, set(test_pos), ind_rels)
    n_ent = int(nodes_df['idx'].max()) + 1
    node_name_map = dict(zip(nodes_df['idx'], nodes_df['name']))

    drug_targets = {{}}
    dt_mask = kg_df['relation'].isin(dt_rels)
    _dt_sub = kg_df.loc[dt_mask, ['x_index', 'y_index']]
    if not _dt_sub.empty:
        _h = _dt_sub['x_index'].astype('int64').to_numpy()
        _t = _dt_sub['y_index'].astype('int64').to_numpy()
        _gene_arr = np.fromiter(gene_idx, dtype='int64', count=len(gene_idx))
        _h_drug = np.isin(_h, drug_arr); _t_gene = np.isin(_t, _gene_arr)
        _h_gene = np.isin(_h, _gene_arr); _t_drug = np.isin(_t, drug_arr)
        _fwd = _h_drug & _t_gene; _rev = _t_drug & _h_gene
        _keep = _fwd | _rev
        _drugs = np.where(_fwd, _h, _t)[_keep]
        _genes = np.where(_fwd, _t, _h)[_keep]
        if _drugs.size:
            _pairs_df = pd.DataFrame({{'drug': _drugs, 'gene': _genes}})
            drug_targets = {{int(d): set(int(x) for x in g)
                            for d, g in _pairs_df.groupby('drug')['gene']}}

    n_neg = len(test_pos) * neg_ratio
    neg_by_strategy = {{}}
    for strat in STRATEGIES:
        neg_by_strategy[strat] = generate_negatives(
            test_pos, n_neg, strat, drug_idx, disease_idx,
            drug_targets, node_name_map, all_pos, rng)

    rel_idx = rel_to_idx[ind_rels[0]]
    inv_name = f'{{ind_rels[0]}}_inv'
    rel_idx_inv = rel_to_idx.get(inv_name)

    prep = {{
        'train_triples': train_triples, 'rel_to_idx': rel_to_idx,
        'n_ent': n_ent, 'n_rels': len(rel_to_idx),
        'test_pos': test_pos, 'neg_by_strategy': neg_by_strategy,
        'rel_idx': rel_idx, 'rel_idx_inv': rel_idx_inv,
        'n_test': len(test_pos),
        'rerun_idx': rerun_idx, 'rerun_seed': rerun_seed,
    }}
    with open(cache_path, 'wb') as f:
        pickle.dump(prep, f)
    return prep

print(f'prepare_kg_resampled defined — will produce {{N_RERUNS}} different '
      f'test splits per KG.')
'''

TRAIN_LOOP_CELL = '''# ── Multi-rerun training loop (TransE / RotatE) ────────────────────────
# Outer loop: rerun_idx 0..N_RERUNS-1
# Inner: for each KG, prepare with rerun seed, train TransE & RotatE,
#        evaluate under all 3 strategies. Results saved per-rerun.
#
# Resumable: per-rerun cache files; crash + restart picks up where it left off.

resampled_results = {}   # resampled_results[kg][model][strategy] = list of dicts (one per rerun)

for kg in KG_NAMES:
    resampled_results.setdefault(kg, {})
    for model_name in MODELS:
        resampled_results[kg].setdefault(model_name, {})
        for strat in STRATEGIES:
            resampled_results[kg][model_name].setdefault(strat, [])

for rerun_idx in range(N_RERUNS):
    print(f'\\n{"="*70}\\nRERUN {rerun_idx + 1} / {N_RERUNS}  (seed={RERUN_SEEDS[rerun_idx]})\\n{"="*70}')

    for kg in KG_NAMES:
        cache_json = CACHE / f'embedding_{kg}_resampled.json'

        # Load existing rerun results if present, skip if this rerun already complete
        existing = {}
        if cache_json.exists():
            with open(cache_json) as f:
                existing = json.load(f)
        done_for_kg = set(existing.get('completed_reruns', []))
        if rerun_idx in done_for_kg:
            # Pull cached results into our in-memory structure
            for model_name in MODELS:
                for strat in STRATEGIES:
                    m = existing.get('results', {}).get(str(rerun_idx), {}).get(model_name, {}).get(strat)
                    if m is not None:
                        # Ensure list slot exists at this rerun_idx
                        slot = resampled_results[kg][model_name][strat]
                        while len(slot) <= rerun_idx:
                            slot.append(None)
                        slot[rerun_idx] = m
            print(f'{kg}/rerun{rerun_idx}: loaded from cache')
            continue

        print(f'\\n--- {kg} / rerun {rerun_idx} ---')
        p = prepare_kg_resampled(kg, rerun_idx)
        print(f'  test pairs: {p["n_test"]}, train triples: {len(p["train_triples"]):,}')

        kg_models_results = {}
        for model_name in models_for_kg(kg):
            print(f'  --- {model_name} ---')
            # train_and_evaluate uses its own SEED arg internally; we want
            # rerun_seed to perturb the model init too, so pass it via prep.
            # The existing function uses SEED constant — we override locally.
            #
            # Implementation: monkey-patch the global SEED for this call.
            _saved_seed = globals()['SEED']
            globals()['SEED'] = RERUN_SEEDS[rerun_idx]
            try:
                # train_and_evaluate also expects checkpoint path. Use per-rerun
                # so crashes don't reuse the wrong rerun's weights.
                if kg in CHECKPOINT_KGS:
                    _saved_ckpt_kgs = CHECKPOINT_KGS
                    # Replace global to use per-rerun checkpoint path
                    # (handled inside train_and_evaluate via name lookup)
                    pass
                strat_results = train_and_evaluate(kg, model_name, p)
            finally:
                globals()['SEED'] = _saved_seed

            for strat, m in strat_results.items():
                # Trim 'scores' / 'labels' to keep cache small
                m_lite = {k: v for k, v in m.items() if k not in ('scores', 'labels')}
                m_lite['rerun_idx'] = rerun_idx
                m_lite['rerun_seed'] = RERUN_SEEDS[rerun_idx]
                kg_models_results.setdefault(model_name, {})[strat] = m_lite

                slot = resampled_results[kg][model_name][strat]
                while len(slot) <= rerun_idx:
                    slot.append(None)
                slot[rerun_idx] = m_lite

        # Save per-KG cache after this rerun
        existing.setdefault('kg', kg)
        existing.setdefault('results', {})
        existing['results'][str(rerun_idx)] = kg_models_results
        existing.setdefault('completed_reruns', [])
        if rerun_idx not in existing['completed_reruns']:
            existing['completed_reruns'] = sorted(set(existing['completed_reruns'] + [rerun_idx]))
        existing['n_reruns_total'] = N_RERUNS
        existing['rerun_seeds'] = RERUN_SEEDS
        with open(cache_json, 'w') as f:
            json.dump(existing, f, indent=2)

print(f'\\n{"="*70}\\nResampled training loop complete.\\n{"="*70}')
'''

GEMMA_RESCORE_CELL = '''# ── Gemma re-scoring across reruns ──────────────────────────────────────
# Gemma's encoded entity embeddings DON'T change across reruns (entities
# are the same; only the test-pair selection changes). So we encode once
# (already cached as gemma_emb_<kg>_d768.npz from the original Gemma run)
# and just re-score the per-rerun test pairs.
#
# This makes Gemma reruns ~100x cheaper than TransE/RotatE reruns.

if GEMMA_ENABLED:
    print('Re-scoring Gemma across reruns (cheap — embeddings cached)...')
    for kg in KG_NAMES:
        resampled_results[kg].setdefault('Gemma', {})
        for strat in STRATEGIES:
            resampled_results[kg]['Gemma'].setdefault(strat, [])

    for rerun_idx in range(N_RERUNS):
        for kg in KG_NAMES:
            cache_json = CACHE / f'embedding_{kg}_gemma_resampled.json'
            existing = {}
            if cache_json.exists():
                with open(cache_json) as f:
                    existing = json.load(f)
            done_for_kg = set(existing.get('completed_reruns', []))
            if rerun_idx in done_for_kg:
                # Pull from cache
                for strat in STRATEGIES:
                    m = existing.get('results', {}).get(str(rerun_idx), {}).get(strat)
                    if m is not None:
                        slot = resampled_results[kg]['Gemma'][strat]
                        while len(slot) <= rerun_idx:
                            slot.append(None)
                        slot[rerun_idx] = m
                continue

            p = prepare_kg_resampled(kg, rerun_idx)
            emb_cache = CACHE / f'gemma_emb_{kg}_d{GEMMA_DIM}.npz'
            if not emb_cache.exists():
                print(f'  {kg}: no Gemma encoding cache — run the original Gemma cell first. Skipping.')
                continue

            model = GemmaNameEmbedder(n_entities=p['n_ent'], n_relations=p['n_rels'],
                                      dim=GEMMA_DIM, model_name=GEMMA_MODEL,
                                      batch_size=GEMMA_BATCH, seed=SEED)
            model.load_embeddings(emb_cache)

            per_strat = {}
            for strat in STRATEGIES:
                neg = p['neg_by_strategy'][strat]
                m = compute_embedding_metrics(model, p['test_pos'], neg,
                                              p['rel_idx'], rel_idx_inv=None)
                m_lite = {k: v for k, v in m.items() if k not in ('scores', 'labels')}
                m_lite['rerun_idx'] = rerun_idx
                m_lite['rerun_seed'] = RERUN_SEEDS[rerun_idx]
                m_lite['name_quality'] = GEMMA_NAME_QUALITY.get(kg, 'unknown')
                per_strat[strat] = m_lite

                slot = resampled_results[kg]['Gemma'][strat]
                while len(slot) <= rerun_idx:
                    slot.append(None)
                slot[rerun_idx] = m_lite

            existing.setdefault('kg', kg)
            existing.setdefault('results', {})
            existing['results'][str(rerun_idx)] = per_strat
            existing.setdefault('completed_reruns', [])
            if rerun_idx not in existing['completed_reruns']:
                existing['completed_reruns'] = sorted(set(existing['completed_reruns'] + [rerun_idx]))
            existing['gemma_dim'] = GEMMA_DIM
            with open(cache_json, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f'  {kg}/Gemma/rerun{rerun_idx}: AUROC(tc)={per_strat["type-constrained"]["auroc"]:.4f}')
    print('Gemma re-scoring across reruns complete.')
else:
    print('Gemma disabled — skipping resampled re-scoring.')
'''

AGGREGATION_CELL = '''# ── Aggregation: long-form DataFrame with `rerun` column ───────────────
all_models_resampled = MODELS + (['Gemma'] if GEMMA_ENABLED else [])

rows_long = []
for kg in KG_NAMES:
    for model_name in all_models_resampled:
        for strat in STRATEGIES:
            slots = resampled_results.get(kg, {}).get(model_name, {}).get(strat, [])
            for rerun_idx, m in enumerate(slots):
                if m is None: continue
                rows_long.append({
                    'kg': kg, 'model': model_name, 'strategy': strat,
                    'rerun': rerun_idx,
                    'rerun_seed': RERUN_SEEDS[rerun_idx],
                    'auroc': m.get('auroc'),
                    'auprc': m.get('auprc'),
                    'mrr': m.get('mrr'),
                    'hits@10': m.get('hits@10'),
                    'hits@100': m.get('hits@100'),
                    'train_time_s': m.get('train_time_s'),
                })
resampled_df = pd.DataFrame(rows_long)
out_csv = BASE / 'results' / 'embedding_comparison_resampled.csv'
resampled_df.to_csv(out_csv, index=False)
print(f'Wrote {out_csv.name} with {len(resampled_df)} rows '
      f'({resampled_df["rerun"].nunique()} reruns × '
      f'{resampled_df["kg"].nunique()} KGs × '
      f'{resampled_df["model"].nunique()} models × '
      f'{resampled_df["strategy"].nunique()} strategies)')

# Summary: mean ± std across reruns per (kg, model, strategy)
print('\\n=== Summary: mean ± std across reruns (AUROC, type-constrained) ===')
summary = (resampled_df[resampled_df['strategy'] == 'type-constrained']
           .groupby(['kg', 'model'])['auroc']
           .agg(['mean', 'std', 'min', 'max', 'count'])
           .round(4))
print(summary.to_string())
'''

HEADLINE_FIG_CELL = '''# ── Headline figure: rerun-empirical CIs per (model, KG) ───────────────
# Replaces the single-run + bootstrap-CI display. Error bars are the
# *empirical* CI across reruns — the honest measure of uncertainty.
strat = 'type-constrained'
models_to_plot = ['TransE', 'RotatE'] + (['Gemma'] if GEMMA_ENABLED else [])
model_colors = {'TransE': '#3F7B92', 'RotatE': '#C5482A', 'Gemma': '#7B3F61'}

fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.95, 4.6))
x = np.arange(len(KG_NAMES))
width = 0.27

for i, mname in enumerate(models_to_plot):
    sub = resampled_df[(resampled_df['model'] == mname) &
                       (resampled_df['strategy'] == strat)]
    means, lo, hi = [], [], []
    for kg in KG_NAMES:
        kg_sub = sub[sub['kg'] == kg]['auroc'].dropna()
        if len(kg_sub) == 0:
            means.append(np.nan); lo.append(0); hi.append(0); continue
        mean = float(kg_sub.mean())
        # Empirical 95% CI — percentile if enough reruns, otherwise ±1.96*SE
        if len(kg_sub) >= 5:
            lo_v = float(np.percentile(kg_sub, 2.5))
            hi_v = float(np.percentile(kg_sub, 97.5))
        else:
            se = float(kg_sub.std() / np.sqrt(len(kg_sub)))
            lo_v = mean - 1.96 * se
            hi_v = mean + 1.96 * se
        means.append(mean)
        lo.append(max(0, mean - lo_v))
        hi.append(max(0, hi_v - mean))
    ax.bar(x + (i - (len(models_to_plot) - 1) / 2) * width, means, width,
           yerr=[lo, hi], capsize=3, label=mname,
           color=model_colors.get(mname, '#888'),
           edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([k.upper() for k in KG_NAMES], rotation=20, ha='right')
ax.set_ylabel('AUROC  (mean ± rerun-empirical 95% CI)')
ax.set_ylim(0.45, 1.05)
ax.axhline(0.5, ls=':', color='#aaa', lw=0.7, zorder=0)
ax.set_title(f'KG embedding performance with N={N_RERUNS} resampled reruns '
             f'({strat} negatives)', fontsize=10)
ax.legend(loc='lower right', frameon=False, fontsize=9)
clean_ax(ax)
plt.tight_layout()
save_fig(fig, FIGS, '08_resampled_headline')
plt.show()
'''

STABILITY_BOX_CELL = '''# ── Stability: AUROC distribution across reruns ────────────────────────
# Box plot — for each (KG, model), the spread of AUROC across 5 reruns
# shows how stable the model is under different train/test splits.
# Wide boxes mean the model is sensitive to the specific held-out edges.

strat = 'type-constrained'
fig, axes = plt.subplots(1, len(models_to_plot), figsize=(15, 4.6), sharey=True)
if len(models_to_plot) == 1: axes = [axes]

for ax_i, mname in enumerate(models_to_plot):
    ax = axes[ax_i]
    sub = resampled_df[(resampled_df['model'] == mname) &
                       (resampled_df['strategy'] == strat)]
    box_data = [sub[sub['kg'] == kg]['auroc'].dropna().tolist() for kg in KG_NAMES]
    bp = ax.boxplot(box_data, positions=range(len(KG_NAMES)), widths=0.55,
                    patch_artist=True, medianprops={'color': 'black', 'lw': 1.2})
    for patch in bp['boxes']:
        patch.set_facecolor(model_colors.get(mname, '#888'))
        patch.set_alpha(0.6)
    # Overlay individual rerun points
    for j, vals in enumerate(box_data):
        ax.scatter([j]*len(vals), vals, color='black', s=12, zorder=5, alpha=0.7)
    ax.set_xticks(range(len(KG_NAMES)))
    ax.set_xticklabels([k.upper() for k in KG_NAMES], rotation=25, ha='right', fontsize=9)
    ax.set_title(mname, fontsize=11)
    if ax_i == 0:
        ax.set_ylabel(f'AUROC across {N_RERUNS} reruns ({strat})')
    ax.set_ylim(0.45, 1.05)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.axhline(0.5, ls=':', color='#aaa', lw=0.7)

fig.suptitle('Stability of KG embeddings under resampled train/test splits',
             fontsize=12, y=1.02)
plt.tight_layout()
save_fig(fig, FIGS, '08_resampled_stability_boxplot')
plt.show()
'''

BOOTSTRAP_VS_RERUN_CELL = '''# ── Bootstrap vs resampled CIs: how much does bootstrap underestimate? ──
# For each (kg, model, strategy) compute:
#   bootstrap_ci_width   = single-run bootstrap CI width on the existing TransE/RotatE run
#   resampled_ci_width   = empirical 95% CI width across 5 resampled reruns
# Ratio > 1 means bootstrap was OVERLY OPTIMISTIC about precision.

compare_rows = []
for kg in KG_NAMES:
    for mname in ['TransE', 'RotatE']:
        # Existing single-run bootstrap CI (from comp_df, built earlier in nb08)
        try:
            bs = comp_df[(comp_df['kg'] == kg) & (comp_df['model'] == mname)
                         & (comp_df['strategy'] == 'type-constrained')].iloc[0]
            bs_width = bs['auroc_ci_hi'] - bs['auroc_ci_lo']
            bs_mean = bs['emb_auroc']
        except (IndexError, KeyError):
            continue
        # Resampled empirical CI
        sub = resampled_df[(resampled_df['kg'] == kg) &
                           (resampled_df['model'] == mname) &
                           (resampled_df['strategy'] == 'type-constrained')]['auroc'].dropna()
        if len(sub) < 2:
            continue
        if len(sub) >= 5:
            r_lo, r_hi = float(np.percentile(sub, 2.5)), float(np.percentile(sub, 97.5))
        else:
            se = float(sub.std() / np.sqrt(len(sub)))
            r_lo, r_hi = float(sub.mean() - 1.96*se), float(sub.mean() + 1.96*se)
        compare_rows.append({
            'kg': kg, 'model': mname,
            'single_run_auroc': bs_mean,
            'rerun_mean_auroc': float(sub.mean()),
            'rerun_std_auroc':  float(sub.std()),
            'bootstrap_ci_width': float(bs_width) if pd.notna(bs_width) else np.nan,
            'rerun_ci_width':     float(r_hi - r_lo),
            'ratio_rerun_vs_bootstrap': float((r_hi - r_lo) / bs_width) if (
                pd.notna(bs_width) and bs_width > 0) else np.nan,
        })
compare_df = pd.DataFrame(compare_rows).round(4)
print('=== Bootstrap CI vs resampled empirical CI ===')
print('Ratio > 1 means bootstrap was OVERLY OPTIMISTIC about precision.\\n')
print(compare_df.to_string(index=False))

mean_ratio = compare_df['ratio_rerun_vs_bootstrap'].mean()
print(f'\\nAverage CI-width ratio (rerun / bootstrap): {mean_ratio:.2f}×')
print(f'  (i.e. bootstrap CIs were on average {mean_ratio:.1f}× too tight)')
'''

PROSE_CELL = '''### Resampling-based stability (Tier 0)

This section addresses a methodological limitation of the bootstrap CIs
above: they only capture variance from resampling test pairs **within
a single training run**. They miss the much larger variance from:

1. **Training stochasticity** — random initialization of TransE/RotatE
   embeddings, negative-sample selection during training, batch ordering.
2. **Train/test split selection** — which 10% of indication edges
   happen to be held out.

To capture both, we resample: for each of N=5 reruns, draw a different
random 10% of drug-disease indication edges as the test set, train
TransE and RotatE from scratch on the remaining 90%, and re-score the
Gemma word-priors baseline against the new test set. (Gemma's encoded
entity embeddings don't depend on the split, so we encode once and
re-score per rerun — making Gemma's reruns ~100× cheaper than the KGE
models'.)

Results in the **resampled CI** are empirical: mean ± 95% percentile
range across the 5 reruns (for N ≥ 5) or mean ± 1.96·SE (for N < 5).

The "bootstrap vs rerun" table above quantifies the gap. If the ratio
column exceeds 1, the bootstrap CIs from a single run were misleadingly
narrow — the true uncertainty (under train/test resampling and training
randomness) is larger.

**For the writeup**: cite this as the headline stability number rather
than the bootstrap CIs. The bootstrap CIs remain useful as a within-run
sanity check, but a 5-rerun empirical CI is the more honest report.
'''


# ── Apply changes ────────────────────────────────────────────────────────
nb = json.loads(NB_PATH.read_text())

# Idempotency
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and MARKER in ''.join(cell['source']):
        print(f'Notebook already has resampling injection ({MARKER}); skipping.')
        sys.exit(0)

# 1. Append resampling config to cell 6 (hyperparams)
config_cell = nb['cells'][6]
src = ''.join(config_cell['source'])
if 'N_RERUNS' not in src:
    src = src.rstrip('\n') + '\n' + CONFIG_ADDITION
    config_cell['source'] = src.splitlines(keepends=True)

# 2. Find insertion point — before the memory-cleanup cell (cell 30 in current state)
def find_idx(needle):
    for i, c in enumerate(nb['cells']):
        if needle in ''.join(c['source']):
            return i
    return None

cleanup_idx = find_idx('Freed KG state from kernel memory')
assert cleanup_idx is not None, 'could not locate cleanup cell'

# 3. New cells to insert, in order, BEFORE the cleanup cell
new_cells = [
    md_cell('## Resampling-based stability  *(Tier 0)*\n\nReplaces the bootstrap CIs above with empirical CIs from 5 full retrainings on different train/test splits. See prose at the end for methodology.'),
    code_cell(PREP_CELL),
    md_cell('### Multi-rerun training loop'),
    code_cell(TRAIN_LOOP_CELL),
    md_cell('### Gemma re-scoring (cheap — embeddings cached, only test pairs change)'),
    code_cell(GEMMA_RESCORE_CELL),
    md_cell('### Aggregate across reruns'),
    code_cell(AGGREGATION_CELL),
    md_cell('### Headline figure with rerun-empirical CIs'),
    code_cell(HEADLINE_FIG_CELL),
    md_cell('### Stability across reruns (box plot)'),
    code_cell(STABILITY_BOX_CELL),
    md_cell('### How much did the bootstrap CIs underestimate variance?'),
    code_cell(BOOTSTRAP_VS_RERUN_CELL),
    md_cell(PROSE_CELL),
]

# Insert before cleanup
for c in reversed(new_cells):
    nb['cells'].insert(cleanup_idx, c)

NB_PATH.write_text(json.dumps(nb, indent=1) + '\n')
print(f'Injected {len(new_cells)} new cells into {NB_PATH.name}')
print(f'  N_RERUNS=5 added to config cell 6')
print(f'  Resampling section inserted at cell index {cleanup_idx}')
print(f'  Final cell count: {len(nb["cells"])}')
