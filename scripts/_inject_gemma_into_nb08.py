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

"""One-shot script: inject EmbeddingGemma word-priors section into nb08.

Idempotent: detects an existing 'gemma_injected' marker and refuses to
double-inject. Run from repo root: python scripts/_inject_gemma_into_nb08.py
"""
import json, sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'eval_notebooks' / '08_embedding_validation.ipynb'

MARKER = 'GemmaNameEmbedder'   # unique identifier introduced by this injection

def code_cell(src):
    return {
        'cell_type': 'code', 'metadata': {}, 'source': src.splitlines(keepends=True),
        'execution_count': None, 'outputs': [],
    }

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(keepends=True)}


GEMMA_IMPORT_LINE = "from src.embedding import GemmaNameEmbedder\n"

GEMMA_CONFIG_CELL = '''# ── EmbeddingGemma (word-priors) baseline configuration ────────────
# This is intentionally NOT a KG embedding method. It exists to answer:
# "How much of the drug-disease indication signal is latent in a
#  pretrained language model's word priors alone, with ZERO knowledge
#  of the graph?"
#
# The model embeds each entity's bare name (no type prefix, no task prefix,
# no neighborhood context) and scores pairs by cosine similarity.
GEMMA_ENABLED   = True
GEMMA_MODEL     = 'google/embeddinggemma-300m'
GEMMA_DIM       = 768       # 128/256/512/768 (Matryoshka)
GEMMA_BATCH     = 64        # batch size for encoding

# KGs where nodes_df['name'] is human-readable (Gemma's word priors apply).
# DRKG/OpenBioLink/BioKG return opaque ID strings — Gemma will produce
# near-noise on those without external name resolution. We still run them
# (the contrast confirms the hypothesis), but flag them in figures.
GEMMA_NAME_QUALITY = {{
    'hetionet':   'readable',
    'primekg':    'readable',
    'drkg':       'id-only',
    'openbilink': 'id-only',
    'biokg':      'id-only',
    'matrix':     'mixed',   # MATRIX has names for most drugs/diseases; gene/pathway IDs vary
}}
print(f'Gemma enabled: {{GEMMA_ENABLED}} | model={{GEMMA_MODEL}} | dim={{GEMMA_DIM}}')
'''

GEMMA_RUNNER_CELL = '''# ── Run EmbeddingGemma on each KG ──────────────────────────────────
# Reuses the same test_pos / neg_by_strategy splits as TransE/RotatE so
# metrics are directly comparable. Encoding is cached per (kg, dim) to a
# .npz on disk — re-running this cell is cheap after the first encode.
gemma_results = {}   # gemma_results[kg][strategy] = metrics dict

if GEMMA_ENABLED:
    for kg in KG_NAMES:
        cache_json = CACHE / f'embedding_gemma_{kg}.json'

        # Reuse cached JSON if present and dim matches
        if cache_json.exists():
            with open(cache_json) as f:
                cached = json.load(f)
            sample_strat = next(iter(cached.get('strategies', {}).values()), {})
            if sample_strat.get('dim') == GEMMA_DIM:
                gemma_results[kg] = cached['strategies']
                print(f'{kg}/Gemma: loaded from cache (dim={GEMMA_DIM})')
                continue

        print(f'\\n--- {kg} / Gemma ---')
        p = preps[kg]
        # Get name list
        kg_df, nodes_df = load_kg(kg, config)
        del kg_df  # not needed, free memory
        n_ent = p['n_ent']
        name_by_idx = dict(zip(nodes_df['idx'].astype(int), nodes_df['name']))
        names = [name_by_idx.get(i, '') for i in range(n_ent)]
        # Flag name quality
        nq = GEMMA_NAME_QUALITY.get(kg, 'unknown')
        print(f'  {n_ent:,} entities; name quality: {nq}')

        emb_cache = CACHE / f'gemma_emb_{kg}_d{GEMMA_DIM}.npz'
        model = GemmaNameEmbedder(n_entities=n_ent, n_relations=p['n_rels'],
                                  dim=GEMMA_DIM, model_name=GEMMA_MODEL,
                                  batch_size=GEMMA_BATCH, seed=SEED)
        t0 = time.time()
        if emb_cache.exists():
            try:
                model.load_embeddings(emb_cache)
                print(f'  Loaded encoded embeddings from {emb_cache.name}')
            except Exception as e:
                print(f'  Cache load failed ({e}) — re-encoding')
                model.encode_entities(names)
                model.save_embeddings(emb_cache)
        else:
            model.encode_entities(names)
            model.save_embeddings(emb_cache)
        enc_s = time.time() - t0

        # Score under each strategy
        per_strat = {}
        for strat in STRATEGIES:
            neg = p['neg_by_strategy'][strat]
            m = compute_embedding_metrics(model, p['test_pos'], neg,
                                          p['rel_idx'], rel_idx_inv=None)
            m['train_time_s'] = enc_s   # "training" = encoding
            m['n_epochs'] = 0
            m['dim'] = GEMMA_DIM
            m['name_quality'] = nq
            # Bootstrap CIs
            if 'scores' in m and 'labels' in m:
                _, auroc_lo, auroc_hi = bootstrap_metric_ci(
                    m['scores'], m['labels'],
                    lambda s, l: roc_auc_score(l, s))
                _, auprc_lo, auprc_hi = bootstrap_metric_ci(
                    m['scores'], m['labels'],
                    lambda s, l: average_precision_score(l, s))
                m['auroc_ci_lo'] = auroc_lo
                m['auroc_ci_hi'] = auroc_hi
                m['auprc_ci_lo'] = auprc_lo
                m['auprc_ci_hi'] = auprc_hi
            per_strat[strat] = m
            print(f'    {strat:>18s}: AUROC={m["auroc"]:.4f}  '
                  f'AUPRC={m["auprc"]:.4f}  H@10={m["hits@10"]:.4f}')

        gemma_results[kg] = per_strat

        # Save per-KG cache (separate file so it doesn't clobber TransE/RotatE)
        with open(cache_json, 'w') as f:
            json.dump({'kg': kg, 'strategies': per_strat,
                       'n_test': p['n_test'], 'n_entities': p['n_ent'],
                       'gemma_dim': GEMMA_DIM, 'gemma_model': GEMMA_MODEL,
                       'name_quality': nq}, f, indent=2)
        # Free encoder weights to keep RAM down between KGs
        del model
        gc.collect()
else:
    print('Gemma disabled (set GEMMA_ENABLED = True to run).')
'''

GEMMA_TABLE_CELL = '''# ── Build a TransE/RotatE/Gemma comparison table ───────────────────
# Append Gemma rows to the same shape as comp_df, so downstream code that
# expects (kg, model, strategy, emb_auroc, ...) just works.
gemma_rows = []
for kg in KG_NAMES:
    strat_map = gemma_results.get(kg, {})
    for strat in STRATEGIES:
        m = strat_map.get(strat, {})
        gemma_rows.append({
            'kg': kg, 'model': 'Gemma', 'strategy': strat,
            'heuristic_auroc': heur_best.get(kg, {}).get(strat),
            'emb_auroc': m.get('auroc'),
            'emb_auprc': m.get('auprc'),
            'auroc_ci_lo': m.get('auroc_ci_lo'),
            'auroc_ci_hi': m.get('auroc_ci_hi'),
            'auprc_ci_lo': m.get('auprc_ci_lo'),
            'auprc_ci_hi': m.get('auprc_ci_hi'),
            'mrr': m.get('mrr'),
            'hits@10': m.get('hits@10'),
            'hits@100': m.get('hits@100'),
            'train_time_s': m.get('train_time_s'),
            'name_quality': m.get('name_quality'),
        })
gemma_df = pd.DataFrame(gemma_rows)

# Concatenate into the full comparison table and overwrite the CSV
full_comp_df = pd.concat([comp_df, gemma_df.drop(columns=['name_quality'],
                                                  errors='ignore')],
                         ignore_index=True)
full_comp_df.to_csv(BASE / 'results' / 'embedding_comparison.csv', index=False)
print(f'Wrote embedding_comparison.csv with {len(full_comp_df)} rows '
      f'({len(comp_df)} KGE + {len(gemma_df)} Gemma)')
gemma_df.head(10)
'''

GEMMA_FIGURE_CELL = '''# ── Headline figure: TransE / RotatE / Gemma AUROC per KG ──────────
# Focus on the type-constrained strategy (the realistic eval) so the
# three-model comparison is clean.
strat = 'type-constrained'

fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.95, 4.4))
kgs = list(KG_NAMES)
x = np.arange(len(kgs))
width = 0.26
model_order = ['TransE', 'RotatE', 'Gemma']
model_colors = {'TransE': '#3F7B92', 'RotatE': '#C5482A', 'Gemma': '#7B3F61'}

for i, mname in enumerate(model_order):
    sub = full_comp_df[(full_comp_df['model'] == mname) &
                       (full_comp_df['strategy'] == strat)].set_index('kg')
    aurocs = [sub.loc[k, 'emb_auroc'] if k in sub.index else np.nan for k in kgs]
    los = [sub.loc[k, 'auroc_ci_lo']
           if k in sub.index and not np.isnan(sub.loc[k, 'auroc_ci_lo']) else np.nan
           for k in kgs]
    his = [sub.loc[k, 'auroc_ci_hi']
           if k in sub.index and not np.isnan(sub.loc[k, 'auroc_ci_hi']) else np.nan
           for k in kgs]
    yerr_lo = [(a - l) if (a is not None and l is not None
                            and not np.isnan(a) and not np.isnan(l)) else 0
               for a, l in zip(aurocs, los)]
    yerr_hi = [(h - a) if (a is not None and h is not None
                            and not np.isnan(a) and not np.isnan(h)) else 0
               for a, h in zip(aurocs, his)]
    bars = ax.bar(x + (i - 1) * width, aurocs, width,
                  yerr=[yerr_lo, yerr_hi], capsize=2,
                  label=mname, color=model_colors[mname],
                  edgecolor='white', linewidth=0.5)

# Mark KGs where Gemma's input is ID-only (its result is near-noise there)
for j, k in enumerate(kgs):
    if GEMMA_NAME_QUALITY.get(k) == 'id-only':
        ax.annotate('†', xy=(x[j] + width, 0.5), ha='center',
                    va='bottom', fontsize=12, color='#888', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([k.upper() for k in kgs], rotation=20, ha='right')
ax.set_ylabel('AUROC (drug-disease link prediction)')
ax.set_ylim(0.45, 1.02)
ax.axhline(0.5, ls=':', color='#aaa', lw=0.7, zorder=0)
ax.set_title(f'KG embedding vs. EmbeddingGemma word-priors  '
             f'({strat} negatives)', fontsize=10)
ax.legend(loc='lower right', frameon=False, fontsize=9)
ax.text(0.005, 0.02, '† Gemma sees opaque IDs (not real names) — '
        'expected to perform near random.',
        transform=ax.transAxes, fontsize=7.5, color='#666', style='italic')
clean_ax(ax)
plt.tight_layout()
save_fig(fig, FIGS, '08_gemma_vs_kge_auroc')
plt.show()
'''

GEMMA_CONCORDANCE_CELL = '''# ── Concordance: does Gemma rank KGs the same way as TransE / RotatE? ─
# Spearman rho between Gemma's KG AUROC ranking and each KGE model's.
# Restricted to readable-name KGs (the others are noise by construction).
readable_kgs = [k for k in KG_NAMES if GEMMA_NAME_QUALITY.get(k) == 'readable']
print(f'Readable-name KGs for concordance: {readable_kgs}')

if len(readable_kgs) >= 3:
    concordance_rows = []
    for strat in STRATEGIES:
        gemma_sub = full_comp_df[(full_comp_df['model'] == 'Gemma') &
                                  (full_comp_df['strategy'] == strat) &
                                  (full_comp_df['kg'].isin(readable_kgs))].set_index('kg')
        for ref_model in ['TransE', 'RotatE']:
            ref_sub = full_comp_df[(full_comp_df['model'] == ref_model) &
                                    (full_comp_df['strategy'] == strat) &
                                    (full_comp_df['kg'].isin(readable_kgs))].set_index('kg')
            common = sorted(set(gemma_sub.index) & set(ref_sub.index))
            if len(common) < 3:
                continue
            xs = [ref_sub.loc[k, 'emb_auroc'] for k in common]
            ys = [gemma_sub.loc[k, 'emb_auroc'] for k in common]
            rho, p = spearmanr(xs, ys)
            concordance_rows.append({
                'strategy': strat, 'reference_model': ref_model,
                'spearman_rho': rho, 'p_value': p, 'n_kgs': len(common),
            })
    concordance_df = pd.DataFrame(concordance_rows)
    print('\\nGemma rank-concordance with KGE models (readable-name KGs only):')
    print(concordance_df.to_string(index=False))
else:
    print(f'Only {len(readable_kgs)} readable-name KGs — concordance not computed.')
    concordance_df = pd.DataFrame()
'''

GEMMA_PROSE_CELL = '''### Word-priors baseline (EmbeddingGemma-300m)

**Question.** How much of the drug-disease indication signal is already
latent in a pretrained language model's word priors, with zero knowledge
of the graph?

**Setup.** For each entity, we embed only its bare name (no type prefix,
no task prefix, no neighborhood context). We score a (drug, disease) pair
by cosine similarity of the two embeddings. No training. The held-out
test pairs and negative samples are identical to those used by TransE
and RotatE, so all three numbers are directly comparable.

**Caveat.** This experiment is well-defined only when `nodes_df['name']`
contains human-readable strings. Of the six KGs in this benchmark:

| KG | Name quality | Interpretable Gemma result? |
|---|---|---|
| Hetionet | Real names | ✓ |
| PrimeKG | Real names + gene symbols | ✓ |
| MATRIX | Mostly real names | ✓ (partial) |
| DRKG | Numeric IDs after `::` | ✗ (near-noise) |
| OpenBioLink | Numeric IDs after `:` | ✗ (near-noise) |
| BioKG | DrugBank / MeSH IDs only | ✗ (near-noise) |

The KGs marked ✗ would need external name resolution (NCBI Gene Info,
MeSH master list, DrugBank synonyms, DOID/MONDO labels) before
EmbeddingGemma can see anything semantically meaningful. The contrast
between the two groups is itself a useful finding: it confirms that the
signal lives in the words specifically, not in arbitrary identifier
strings the model has never seen.

**What to read off the chart.** Compare the Gemma bar to the TransE /
RotatE bars on the readable-name KGs. The gap (KGE – Gemma) is the part
of the indication signal that requires actual graph structure to recover,
beyond what a pretrained language model already encodes about drug and
disease names.
'''

# ── Apply changes ────────────────────────────────────────────────────────
nb = json.loads(NB_PATH.read_text())

# Idempotency check
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and MARKER in ''.join(cell['source']):
        print(f"Notebook already has Gemma injection ({MARKER}); skipping.")
        sys.exit(0)

# 1. Add GemmaNameEmbedder to the imports cell (cell 2)
imports_cell = nb['cells'][2]
src = ''.join(imports_cell['source'])
if 'GemmaNameEmbedder' not in src:
    src = src.replace(
        'from src.embedding import (TransE, RotatE, build_train_triples,\n                           compute_embedding_metrics)',
        'from src.embedding import (TransE, RotatE, GemmaNameEmbedder,\n                           build_train_triples,\n                           compute_embedding_metrics)'
    )
    if 'GemmaNameEmbedder' not in src:
        # Fallback: simpler form
        src = src.replace(
            'from src.embedding import',
            'from src.embedding import GemmaNameEmbedder, '
        ).replace('GemmaNameEmbedder, GemmaNameEmbedder, ', 'GemmaNameEmbedder, ', 1)
    imports_cell['source'] = src.splitlines(keepends=True)

# 2. Insert new cells AFTER cell 11 (the existing TransE/RotatE runner loop)
#    so Gemma runs after the KGE models. Then insert the analysis cells
#    BEFORE the memory-cleanup cell at the end.
#
#    Find indexes by content (more robust than positional).
def find_cell_idx(needle):
    for i, c in enumerate(nb['cells']):
        if needle in ''.join(c['source']):
            return i
    return None

after_runner = find_cell_idx('# Run all models on all KGs')   # cell ~11
before_cleanup = find_cell_idx('Freed KG state from kernel memory')  # cell ~22

assert after_runner is not None, 'could not locate runner cell'
assert before_cleanup is not None, 'could not locate cleanup cell'

new_cells = [
    md_cell('## EmbeddingGemma word-priors baseline\n'
            '\nThe cells below add EmbeddingGemma-300m as a non-training baseline.'
            ' See the prose section at the bottom for the experimental question.'),
    code_cell(GEMMA_CONFIG_CELL),
    code_cell(GEMMA_RUNNER_CELL),
]
analysis_cells = [
    md_cell('## Word-priors vs. trained graph embeddings\n'),
    code_cell(GEMMA_TABLE_CELL),
    code_cell(GEMMA_FIGURE_CELL),
    code_cell(GEMMA_CONCORDANCE_CELL),
    md_cell(GEMMA_PROSE_CELL),
]

# Insert (back-to-front so indexes don't shift)
for c in reversed(analysis_cells):
    nb['cells'].insert(before_cleanup, c)
for c in reversed(new_cells):
    nb['cells'].insert(after_runner + 1, c)

NB_PATH.write_text(json.dumps(nb, indent=1) + '\n')
print(f'Injected {len(new_cells) + len(analysis_cells)} new cells into {NB_PATH.name}')
print(f'  Gemma runner inserted after cell {after_runner}')
print(f'  Gemma analysis inserted before cell {before_cleanup} (now shifted)')
