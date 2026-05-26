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

"""Inject prompting-strategy axis into nb09. Idempotent.

Changes:
  1. Cell 1 (imports): add `from prompting_strategies import STRATEGIES, get_strategy`
  2. Cell 2 (CONFIG):  add STRATEGIES_TO_RUN, SKIP_SHUFFLED_STRATEGIES_FOR_NEW
  3. Cell 12 (ollama): generalize ollama_query to accept temperature/max_tokens/format
  4. Cell 14 (loop):   add strategy dimension to loop; update completed_keys;
                       add 'strategy', 'n_calls_made' columns to responses
  5. Add new analysis cells (per-strategy accuracy, calibration plot for
     verbalized_prob, parse-rate by strategy, call-cost table)
  6. Add markdown header before the new section

Run from repo root: python scripts/_inject_strategies_into_nb09.py
"""
import json, sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'eval_notebooks' / '09_llm_integration.ipynb'
MARKER = 'STRATEGIES_TO_RUN'   # unique identifier introduced by this injection


def code_cell(src):
    return {
        'cell_type': 'code', 'metadata': {},
        'source': src.splitlines(keepends=True),
        'execution_count': None, 'outputs': [],
    }

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': src.splitlines(keepends=True)}


# ── New cell content ─────────────────────────────────────────────────────────

CONFIG_ADDITION = '''
# ── PROMPTING STRATEGIES (new) ─────────────────────────────────────────
# 8 new strategies + the original baseline. See src/prompting_strategies.py.
STRATEGIES_TO_RUN = [
    'zero_shot_direct',        # baseline (matches the old single-prompt flow)
    'zero_shot_cot',
    'step_back',
    'few_shot_3',
    'few_shot_3_cot',
    'self_consistency_5',
    'structured_json',
    'verbalized_prob',
    'prompt_then_verify',
]
# To save calls, new strategies skip the shuffled_kg control condition.
# zero_shot_direct under shuffled_kg already exists in the cache.
SKIP_SHUFFLED_FOR = {s for s in STRATEGIES_TO_RUN if s != 'zero_shot_direct'}

# Print updated call-budget estimate
from prompting_strategies import STRATEGIES as _ST, total_calls_per_cell
_total_strats = total_calls_per_cell(STRATEGIES_TO_RUN)
print(f'Strategies enabled: {STRATEGIES_TO_RUN}')
print(f'Sum of n_calls per cell across all strategies: {_total_strats}')
# Compute realistic n_cells given the reduced cross
_cells = 0
for s in STRATEGIES_TO_RUN:
    conds = [c for c in CONDITIONS if not (s in SKIP_SHUFFLED_FOR and c == 'shuffled_kg')]
    _cells += n_pairs * len(conds) * N_RESEEDS * len(KGS) * _ST[s].n_calls
print(f'Total LLM calls across strategies (n_pairs={n_pairs}): {_cells:,}')
'''


OLLAMA_REPLACEMENT_OLD = '''def ollama_query(prompt, seed=0):
    payload = {'model': LLM_TAG, 'prompt': prompt, 'stream': False,
               'options': {'temperature': TEMPERATURE, 'num_predict': MAX_TOKENS, 'seed': int(seed)}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60); r.raise_for_status()
        return {'response': r.json().get('response', ''), 'error': None}
    except Exception as e:
        return {'response': '', 'error': str(e)}'''

OLLAMA_REPLACEMENT_NEW = '''def ollama_query(prompt, seed=0, temperature=None, max_tokens=None, format=None):
    """Send one prompt to Ollama. New kwargs let prompting strategies override
    decoding (temperature, num_predict) and request JSON-mode output."""
    options = {'temperature': TEMPERATURE if temperature is None else temperature,
               'num_predict': MAX_TOKENS if max_tokens is None else max_tokens,
               'seed': int(seed)}
    payload = {'model': LLM_TAG, 'prompt': prompt, 'stream': False, 'options': options}
    if format is not None:
        payload['format'] = format   # e.g. 'json' for structured-output strategy
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120); r.raise_for_status()
        return {'response': r.json().get('response', ''), 'error': None}
    except Exception as e:
        return {'response': '', 'error': str(e)}'''


# Strategy-aware main loop. Sits in its own cell (additive — does not delete
# the original loop, which stays as historical reference). When you re-execute
# the notebook, run THIS cell instead of cell 14. The original loop is kept
# so that any existing analysis that depends on its in-memory state still works.
STRATEGY_LOOP_CELL = '''# ── Strategy-aware main loop ───────────────────────────────────────────
# Iterates over (strategy × kg × condition × pair × reseed). Reuses the
# same edge_index / retrieve_li / verbalize as the original loop. Each cell
# becomes ONE row in responses.parquet (multi-call strategies aggregate
# internally, see src/prompting_strategies.py).

from prompting_strategies import STRATEGIES, get_strategy

def completed_keys_with_strategy(fp):
    if not fp.exists(): return set()
    df = pd.read_parquet(fp, columns=['llm','kg','strategy','condition','pair_idx','reseed','error'])
    df = df[df['error'].isna()]
    if 'strategy' not in df.columns:
        return set()
    return set(zip(df.llm, df.kg, df.strategy, df.condition, df.pair_idx, df.reseed))

done = completed_keys_with_strategy(RESPONSES_FP)
# Also count old rows (no 'strategy' column) so we know how many are pre-strategy
if RESPONSES_FP.exists():
    _all = pd.read_parquet(RESPONSES_FP)
    n_legacy = int(_all.get('strategy', pd.Series(dtype=object)).isna().sum()) if 'strategy' in _all.columns else len(_all)
    print(f'Resuming — {{len(done):,}} strategy-aware cells complete, '
          f'{{n_legacy:,}} legacy rows (treated as zero_shot_direct/with_kg if matching)')
else:
    print('Fresh run — no responses.parquet yet')

pairs_by_idx = {{int(r['pair_idx']): r for _, r in pairs.iterrows()}}
pids         = sorted(pairs_by_idx)
perm         = {{pid: pids[(i + 1) % len(pids)] for i, pid in enumerate(pids)}}

buf = []
for kg_name in KGS:
    print(f'\\n=== {{kg_name}} ===')
    try:
        kg_df, nodes_df = load_kg(kg_name, config)
    except Exception as e:
        print(f'  [skip] {{e}}'); continue

    mp = build_lookup_maps(nodes_df)
    type_map, name_map = mp['node_type_map'], mp['node_name_map']
    idx_to_type = dict(zip(nodes_df['idx'].astype(int), nodes_df['type'].astype(str)))

    id_to_idx = defaultdict(list)
    for raw, idx in zip(nodes_df['id'].astype(str), nodes_df['idx'].astype(int)):
        for k in {{raw, raw.split('::', 1)[-1] if '::' in raw else raw}}:
            id_to_idx[k].append((idx, idx_to_type.get(idx)))

    et         = config['knowledge_graphs'][kg_name]['entity_types']
    drug_t     = et.get('Drug')
    disease_t  = et.get('Disease')
    def _tc(cands, want):
        for c in cands:
            for idx, t in id_to_idx.get(c, []):
                if t == want: return idx
        return None
    resolve_drug    = lambda db_id: _tc(to_kg_drug_ids(db_id, kg_name), drug_t)
    resolve_disease = lambda d_id:  _tc(to_kg_disease_ids(d_id, kg_name), disease_t)

    edge_index = build_edge_index(kg_df)

    subgraph_text, n_d, n_x = {{}}, 0, 0
    for pi in pids:
        row = pairs_by_idx[pi]
        d_idx = resolve_drug(row['drug_id']);    n_d += d_idx is not None
        n_x  += resolve_disease(row['disease_id']) is not None
        subgraph_text[pi] = verbalize(retrieve_li(d_idx, edge_index, type_map, kg_name), name_map)
    n_content = sum(1 for t in subgraph_text.values() if t)
    print(f'  anchors: drug={{n_d}}/{{len(pids)}}, disease={{n_x}}/{{len(pids)}}  |  with content: {{n_content}}/{{len(pids)}}')

    for strategy_name in STRATEGIES_TO_RUN:
        strat = get_strategy(strategy_name)
        # Reduced cross: skip shuffled_kg for new strategies
        conds_for_strat = [c for c in CONDITIONS
                           if not (strategy_name in SKIP_SHUFFLED_FOR and c == 'shuffled_kg')]
        print(f'  -- strategy={{strategy_name}} ({{strat.n_calls}}-call) | conditions={{conds_for_strat}}')

        for pi in tqdm(pids, desc=f'    {{strategy_name}}/{{kg_name}}', leave=False):
            row = pairs_by_idx[pi]
            kg_text = {{'with_kg': subgraph_text[pi],
                       'shuffled_kg': subgraph_text[perm[pi]],
                       'no_kg': ''}}
            for cond in conds_for_strat:
                for reseed in range(N_RESEEDS):
                    key = (LLM_TAG, kg_name, strategy_name, cond, pi, reseed)
                    if key in done: continue
                    result = strat.execute(
                        ollama_query, row['drug_name'], row['disease_name'],
                        kg_text[cond],
                        seed=RANDOM_SEED + reseed * 1000 + pi)
                    buf.append({{
                        'llm': LLM_TAG, 'kg': kg_name, 'strategy': strategy_name,
                        'condition': cond, 'pair_idx': pi, 'reseed': reseed,
                        'drug_name': row['drug_name'], 'disease_name': row['disease_name'],
                        'label': int(row['label']),
                        'label_pred': result['pred'], 'confidence': result['confidence'],
                        'correct': (result['pred'] == int(row['label']))
                                    if result['pred'] is not None else False,
                        'n_kg_edges': kg_text[cond].count('.') if kg_text[cond] else 0,
                        'n_calls_made': result['n_calls_made'],
                        'response': result['response'], 'error': result['error'],
                        'ts': time.time(),
                    }})
                    if len(buf) >= 20:
                        append_rows(buf, RESPONSES_FP); buf = []
            if buf:
                append_rows(buf, RESPONSES_FP); buf = []
    del kg_df, nodes_df, edge_index; gc.collect()

print('\\nStrategy-aware run done.')
'''


STRATEGY_METRICS_CELL = '''# ── Strategy-aware metrics ─────────────────────────────────────────────
r = pd.read_parquet(RESPONSES_FP)
# Backfill legacy rows (pre-strategy) as zero_shot_direct
if 'strategy' not in r.columns:
    r['strategy'] = 'zero_shot_direct'
else:
    r['strategy'] = r['strategy'].fillna('zero_shot_direct')

r = r[r['error'].isna() & r['label_pred'].notna()].copy()
r['label_pred'] = r['label_pred'].astype(int)
r = r.merge(pairs[['pair_idx', 'stratum']], on='pair_idx', how='left')

print(f'Analyzing {len(r):,} valid response rows '
      f'across {r["strategy"].nunique()} strategies, '
      f'{r["kg"].nunique()} KGs, {r["condition"].nunique()} conditions.')

# Overall accuracy per (strategy, condition) — averaged across KGs
print('\\n=== Accuracy by strategy × condition (averaged across KGs) ===')
strat_acc = r.groupby(['strategy', 'condition'])['correct'].agg(['mean', 'count'])
strat_acc.columns = ['accuracy', 'n']
strat_acc['accuracy'] = strat_acc['accuracy'].round(3)
print(strat_acc.to_string())

# Parse rate per strategy (fraction of responses that yielded a valid pred)
print('\\n=== Parse rate by strategy ===')
_all = pd.read_parquet(RESPONSES_FP)
if 'strategy' not in _all.columns:
    _all['strategy'] = 'zero_shot_direct'
_all = _all[_all['error'].isna()]
parse_rate = (_all.groupby('strategy')
                  .apply(lambda d: pd.Series({
                      'n_responses': len(d),
                      'parsed': d['label_pred'].notna().sum(),
                      'parse_rate': float(d['label_pred'].notna().mean()),
                  })))
print(parse_rate.round(3).to_string())

# Cost: total calls made per strategy
if 'n_calls_made' in _all.columns:
    print('\\n=== Call-cost by strategy ===')
    cost = (_all.dropna(subset=['n_calls_made'])
                .groupby('strategy')['n_calls_made'].agg(['sum', 'mean', 'count']))
    cost.columns = ['total_calls', 'mean_calls_per_cell', 'n_cells']
    print(cost.round(2).to_string())
'''

STRATEGY_FIGURE_CELL = '''# ── Headline figure: accuracy by strategy × condition ─────────────────
strats = STRATEGIES_TO_RUN
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, cond in zip(axes, ['with_kg', 'no_kg']):
    sub = r[r['condition'] == cond]
    acc = sub.groupby('strategy')['correct'].mean().reindex(strats)
    n   = sub.groupby('strategy')['correct'].size().reindex(strats)
    # Wilson 95% CI for binomial proportion
    from scipy.stats import norm
    z = norm.ppf(0.975)
    se = np.sqrt(acc * (1 - acc) / n.clip(lower=1))
    los = (acc - z * se).clip(lower=0)
    his = (acc + z * se).clip(upper=1)
    xs = np.arange(len(strats))
    colors = ['#666' if s == 'zero_shot_direct' else '#028090' for s in strats]
    ax.bar(xs, acc, yerr=[acc - los, his - acc], capsize=3,
           color=colors, edgecolor='white')
    ax.set_xticks(xs)
    ax.set_xticklabels(strats, rotation=35, ha='right', fontsize=8)
    ax.axhline(0.5, ls=':', color='#aaa', lw=0.7)
    ax.set_title(f'condition = {cond}', fontsize=11)
    ax.set_ylabel('Accuracy' if cond == 'with_kg' else '')
    ax.set_ylim(0, 1.0)
fig.suptitle(f'Prompting strategy comparison  ({LLM_TAG}, averaged over KGs)',
             y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(FIGS / '09_prompting_strategies.png', dpi=140, bbox_inches='tight')
plt.show()
'''


STRATEGY_CALIBRATION_CELL = '''# ── Calibration: confidence vs accuracy, per strategy ────────────────────
# Tests whether higher confidence corresponds to higher accuracy.
# Strategies that output uncalibrated confidence (everything always "5") will
# show a flat line; well-calibrated strategies show a monotonic relationship.

fig, ax = plt.subplots(figsize=(8, 5))
markers = {'zero_shot_direct': 'o', 'zero_shot_cot': 's',
           'few_shot_3': '^', 'few_shot_3_cot': 'v',
           'self_consistency_5': 'D', 'structured_json': 'P',
           'verbalized_prob': '*', 'prompt_then_verify': 'X',
           'step_back': 'h'}
for s in STRATEGIES_TO_RUN:
    sub = r[(r['strategy'] == s) & (r['condition'] == 'with_kg')
            & r['confidence'].notna()]
    if len(sub) < 10: continue
    # Bin by confidence and compute accuracy per bin
    cal = sub.groupby('confidence')['correct'].agg(['mean', 'count'])
    if len(cal) < 2: continue
    ax.plot(cal.index, cal['mean'], marker=markers.get(s, '.'),
            label=f'{s} (n={len(sub)})', linewidth=1, markersize=8)
ax.plot([1, 5], [0.2, 1.0], '--', color='#aaa', lw=0.7, label='perfect calibration')
ax.set_xlabel('Self-reported confidence (1-5)')
ax.set_ylabel('Empirical accuracy')
ax.set_title(f'Calibration by strategy ({LLM_TAG}, with_kg)')
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig(FIGS / '09_prompting_calibration.png', dpi=140, bbox_inches='tight')
plt.show()
'''


# ── Apply changes ────────────────────────────────────────────────────────────
nb = json.loads(NB_PATH.read_text())

# Idempotency
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and MARKER in ''.join(cell['source']):
        print(f'Notebook already has strategy injection ({MARKER}); skipping.')
        sys.exit(0)

# 1. Add import to cell 1
imports = nb['cells'][1]
src = ''.join(imports['source'])
if 'prompting_strategies' not in src:
    # Insert before the loading import
    src = src.replace(
        'from loading import load_kg, load_config, find_config',
        'from loading import load_kg, load_config, find_config\n'
        'from prompting_strategies import STRATEGIES, get_strategy, total_calls_per_cell'
    )
    imports['source'] = src.splitlines(keepends=True)

# 2. Append to config cell (cell 2)
config_cell = nb['cells'][2]
src = ''.join(config_cell['source'])
if 'STRATEGIES_TO_RUN' not in src:
    src = src.rstrip('\n') + '\n' + CONFIG_ADDITION
    config_cell['source'] = src.splitlines(keepends=True)

# 3. Replace ollama_query in cell 12
ollama_cell = nb['cells'][12]
src = ''.join(ollama_cell['source'])
if OLLAMA_REPLACEMENT_OLD in src:
    src = src.replace(OLLAMA_REPLACEMENT_OLD, OLLAMA_REPLACEMENT_NEW)
    ollama_cell['source'] = src.splitlines(keepends=True)
elif 'temperature=None' not in src:
    print('WARNING: could not patch ollama_query — exact match not found. '
          'Manual update needed in cell 12.')

# 4. Insert strategy-aware loop AFTER cell 14 (original loop kept as legacy)
#    Inserted cells: header + STRATEGY_LOOP_CELL + STRATEGY_METRICS_CELL +
#    STRATEGY_FIGURE_CELL + STRATEGY_CALIBRATION_CELL + prose
def find_idx(needle):
    for i, c in enumerate(nb['cells']):
        if needle in ''.join(c['source']):
            return i
    return None

after_orig_loop = find_idx('def completed_keys(fp):')
assert after_orig_loop is not None, 'could not find original main-loop cell'

new_cells = [
    md_cell('## 9 · Prompting strategy experiments  *(new)*\n\n'
            'The cells below add 8 new prompting strategies alongside the original '
            'zero-shot baseline, and rerun the loop over (strategy × kg × condition '
            '× pair × reseed). New strategies skip the `shuffled_kg` control to save '
            'calls (the `zero_shot_direct/shuffled_kg` numbers already exist).\n\n'
            'See `src/prompting_strategies.py` for templates and '
            '`docs/llm_prompting_strategies.md` for the experimental hypothesis '
            'behind each strategy.'),
    code_cell(STRATEGY_LOOP_CELL),
    md_cell('### Strategy-level metrics, parse rates, call cost'),
    code_cell(STRATEGY_METRICS_CELL),
    md_cell('### Headline strategy comparison'),
    code_cell(STRATEGY_FIGURE_CELL),
    md_cell('### Calibration: do confident answers match accuracy?'),
    code_cell(STRATEGY_CALIBRATION_CELL),
]

# Insert AFTER the original loop cell (after_orig_loop is cell 14)
for c in reversed(new_cells):
    nb['cells'].insert(after_orig_loop + 1, c)

NB_PATH.write_text(json.dumps(nb, indent=1) + '\n')
print(f'Injected {len(new_cells)} new cells into {NB_PATH.name}')
print(f'  Imports patched in cell 1')
print(f'  Config extended in cell 2')
print(f'  ollama_query patched in cell 12 (added temperature/max_tokens/format kwargs)')
print(f'  Strategy loop + analysis added after cell {after_orig_loop}')
