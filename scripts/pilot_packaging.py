"""Three-way KG-packaging pilot on primekg.

Tests three different ways of presenting KG content to the LLM, on the same
20 stratified pairs, with the same LLMPrompt strategy. Output goes to one
CSV per row × variant. The script prints per-variant accuracy, prompt-size
stats, per-stratum breakdown, and a flip analysis with the LLM's own
reasoning strings.

Variants
--------
  drug_only         — drug-side edges only, no Disease section at all.
                      Tests: is the disease section the mechanism-shortcut source?
  typed_current     — current nb09 design: drug = Gene/Protein; disease = any-
                      non-Drug; degree-ascending sort; cap 100 per side.
                      Baseline for comparison.
  hub_filtered      — typed_current + symmetric hub cutoff on both sides
                      (drops any neighbour with degree > HUB_THRESHOLD).
                      Tests: does aggressive hub removal fix the shortcut?
  disease_hub_only  — typed_current + hub cutoff on DISEASE side only
                      (drug side keeps all its G/P targets). Tests whether
                      the leakage is specifically from hub disease-genes
                      while drug targets need to stay hub-y to be useful.

Defaults
--------
  KG          : primekg
  N pairs     : 20 (stratified: 4 per stratum × 5 strata)
  Reseeds     : 1
  Hub thresh. : 50

Usage
-----
  python scripts/pilot_packaging.py
  python scripts/pilot_packaging.py --n-pairs 10
  python scripts/pilot_packaging.py --hub-threshold 25
  python scripts/pilot_packaging.py --variants drug_only typed_current

Outputs
-------
  results/tables/09_llm_runs/09_pilot_packaging.csv
  Summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'src'))

from loading import load_kg, load_config, find_config            # noqa: E402
from prompting_strategies import get_strategy                    # noqa: E402
from graph_utils import build_lookup_maps                        # noqa: E402

GOLD    = BASE / 'data' / 'gold_standards'
TABLES  = BASE / 'results' / 'tables' / '09_llm_runs'
OUT_CSV = TABLES / '09_pilot_packaging.csv'

# ── LLM client ──────────────────────────────────────────────────────────────
LLM_TAG     = 'llama3.1:8b'
OLLAMA_URL  = 'http://localhost:11434/api/generate'
TEMPERATURE = 0.0
NUM_CTX     = 16384
RANDOM_SEED = 42


def ollama_query(prompt, seed=0, temperature=None, max_tokens=None, format=None):
    payload = {
        'model':  LLM_TAG, 'prompt': prompt, 'stream': False,
        'options': {
            'temperature': TEMPERATURE if temperature is None else temperature,
            'num_predict': 200 if max_tokens is None else max_tokens,
            'num_ctx':     NUM_CTX,
            'seed':        int(seed),
        },
    }
    if format is not None:
        payload['format'] = format
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return {'response': r.json().get('response', ''), 'error': None}
    except Exception as e:
        return {'response': '', 'error': str(e)}


def assert_ollama_up():
    try:
        tags = {m['name'] for m in
                requests.get('http://localhost:11434/api/tags', timeout=3).json().get('models', [])}
    except Exception as e:
        raise SystemExit(f'Ollama not reachable at {OLLAMA_URL}: {e}')
    if LLM_TAG not in tags:
        raise SystemExit(f'Model {LLM_TAG!r} not pulled. Try: ollama pull {LLM_TAG}')
    print(f'Ollama up — {LLM_TAG} ✓')


# ── Crosswalks ──────────────────────────────────────────────────────────────
_BARE_MESH = re.compile(r'^[DC]\d+$')


def _id_variants(cands):
    out = set(cands)
    for c in list(cands):
        if not isinstance(c, str):
            continue
        if ':' in c:
            ns, _, rest = c.partition(':')
            out.update({c.replace(':', '_', 1),
                        f'{ns.upper()}:{rest}', f'{ns.lower()}:{rest}', rest})
            try:
                stripped = str(int(rest))
                if stripped != rest:
                    out.add(stripped)
            except (ValueError, TypeError):
                pass
        if '_' in c and ':' not in c:
            out.add(c.replace('_', ':', 1))
    return out


def load_crosswalks():
    do        = pd.read_csv(GOLD / 'do_diseases.csv')
    mesh_doid = pd.read_csv(GOLD / 'mesh_to_doid.csv', on_bad_lines='skip')
    xref      = pd.read_csv(GOLD / 'drugbank_xref.csv')
    sssom     = pd.read_csv(GOLD / 'mondo.sssom.tsv', sep='\t', comment='#')
    sssom     = sssom[sssom['predicate_id'] == 'skos:exactMatch']

    _do = do.dropna(subset=['mondo_id', 'doid']).copy()
    _do['m'] = _do['mondo_id'].str.replace('MONDO:', '', regex=False).str.lstrip('0').replace({'': '0'})
    _do['d'] = _do['doid'].str.replace('DOID:', '', regex=False).str.lstrip('0').replace({'': '0'})
    mondo_to_doid_all = defaultdict(set)
    doid_to_mondo_all = defaultdict(set)
    for _, r in _do.iterrows():
        mondo_to_doid_all[r['m']].add(r['d'])
        doid_to_mondo_all[r['d']].add(r['m'])

    mesh_doid['mesh_clean'] = mesh_doid['mesh_id'].str.replace(r'\s*\{.*\}', '', regex=True).str.strip()
    _mh = (mesh_doid.assign(src_rank=mesh_doid['source'].map({'DO': 0, 'MONDO': 1}).fillna(2))
                    .sort_values('src_rank').drop_duplicates('mesh_clean', keep='first'))
    mesh_to_doid = {r['mesh_clean']: (r['doid'].replace('DOID:', '').lstrip('0') or '0')
                    for _, r in _mh.iterrows() if pd.notna(r['doid'])}
    doid_to_mesh = defaultdict(set)
    for m, d in mesh_to_doid.items():
        if m.startswith('MESH:'):
            doid_to_mesh[d].add(m)

    drug_equiv = defaultdict(set)
    for db, ns, eid in zip(xref.drugbank_id.astype(str),
                           xref.namespace.astype(str),
                           xref.external_id.astype(str)):
        drug_equiv[db].update({f'{ns}:{eid}', eid})
        if ns == 'PUBCHEM.COMPOUND':
            drug_equiv[db].update({f'CID:{eid}', f'pubchem.compound:{eid}'})

    _eq = defaultdict(set)
    for s, o in zip(sssom['subject_id'], sssom['object_id']):
        _eq[s].add(o); _eq[o].add(s)
    disease_equiv = defaultdict(set)
    for k, vs in _eq.items():
        s = set(vs)
        for v in vs:
            s |= _eq.get(v, set())
        s.discard(k)
        disease_equiv[k] = s

    return {
        'mondo_to_doid_all': mondo_to_doid_all,
        'doid_to_mondo_all': doid_to_mondo_all,
        'mesh_to_doid':      mesh_to_doid,
        'doid_to_mesh':      doid_to_mesh,
        'drug_equiv':        drug_equiv,
        'disease_equiv':     disease_equiv,
    }


def make_resolvers(config, kg_name, xw):
    drug_equiv        = xw['drug_equiv']
    mondo_to_doid_all = xw['mondo_to_doid_all']
    doid_to_mondo_all = xw['doid_to_mondo_all']
    mesh_to_doid      = xw['mesh_to_doid']
    doid_to_mesh      = xw['doid_to_mesh']
    disease_equiv     = xw['disease_equiv']

    def to_kg_drug_ids(db_id):
        s = str(db_id).strip()
        cands = {s} | drug_equiv.get(s, set())
        return _id_variants(cands)

    def to_kg_disease_ids(disease_id):
        s = str(disease_id).strip()
        scheme = config['knowledge_graphs'][kg_name].get('disease_id_scheme', 'doid')
        if   s.startswith('MONDO:'):
            mondo = s.replace('MONDO:', '').lstrip('0') or '0'
        elif s.startswith('DOID:'):
            mondo = sorted(doid_to_mondo_all.get(
                s.replace('DOID:', '').lstrip('0') or '0', ['?']))[0]
        elif _BARE_MESH.match(s):
            d = mesh_to_doid.get(f'MESH:{s}')
            mondo = sorted(doid_to_mondo_all.get(d, ['?']))[0] if d else '?'
        else:
            mondo = s.lstrip('0') or '0'

        cands = set()
        if scheme == 'mondo':
            cands.update({mondo, mondo.zfill(7), f'MONDO:{mondo}'})
        elif scheme == 'doid':
            for d in mondo_to_doid_all.get(mondo, set()):
                cands.update({d, f'DOID:{d}'})
        elif scheme == 'doid_mesh':
            for d in mondo_to_doid_all.get(mondo, set()):
                cands.update({d, f'DOID:{d}'})
                cands |= doid_to_mesh.get(d, set())
        elif scheme == 'mesh':
            for d in mondo_to_doid_all.get(mondo, set()):
                for m in doid_to_mesh.get(d, set()):
                    cands.update({m, m.split(':', 1)[-1]})

        for c in list(cands) + [f'MONDO:{mondo}']:
            cands |= disease_equiv.get(c, set())
        return _id_variants(cands)

    return to_kg_drug_ids, to_kg_disease_ids


# ── Predicate normalization ─────────────────────────────────────────────────
PREDICATE_NORMALIZE = {
    'drug_protein': 'targets',  'DPI': 'targets',  'CbG': 'binds',
    'DRUGBANK::target::Compound:Gene':      'targets',
    'DRUGBANK::enzyme::Compound:Gene':      'is metabolised by',
    'DRUGBANK::carrier::Compound:Gene':     'is carried by',
    'DRUGBANK::transporter::Compound:Gene': 'is transported by',
    'DRUG_BINDING_GENE':   'binds',     'DRUG_BINDACT_GENE':   'binds and activates',
    'DRUG_BINDINH_GENE':   'binds and inhibits',
    'DRUG_ACTIVATION_GENE':'activates', 'DRUG_INHIBITION_GENE':'inhibits',
    'DRUG_CATALYSIS_GENE': 'is catalysed by', 'DRUG_REACTION_GENE': 'reacts with',
    'directly_physically_interacts_with': 'directly binds',
    'physically_interacts_with':          'physically interacts with',
    'interacts_with':                     'interacts with',
    'affects':                            'affects',
}


def normalise_predicate(raw):
    s = str(raw)
    if s in PREDICATE_NORMALIZE:
        return PREDICATE_NORMALIZE[s]
    stripped = re.sub(r'^(biolink:|RO:|SIO:|oio:)', '', s)
    return PREDICATE_NORMALIZE.get(stripped, stripped.replace('_', ' '))


def build_edge_index(kg_df):
    ix = defaultdict(list)
    for x, y, r in zip(kg_df['x_index'].values,
                       kg_df['y_index'].values,
                       kg_df['relation'].values):
        u, v, rs = int(x), int(y), str(r)
        ix[u].append((v, rs))
        ix[v].append((u, rs))
    return ix


# ── Retrieval with optional per-side hub cap ──────────────────────────────
def retrieve_subgraph(drug_idx, disease_idx, edge_index, type_map,
                      *, gene_type, drug_type, cap=100,
                      drug_max_degree=None, disease_max_degree=None):
    """Per-side hub thresholds — None means no filtering on that side."""
    def collect(anchor_idx, other_idx, keep_pred, max_neighbor_degree):
        if anchor_idx is None:
            return [], 0
        scored = []
        for v, e_rel in edge_index.get(anchor_idx, []):
            if v == other_idx:
                continue
            v_type = type_map.get(v)
            if not keep_pred(v_type):
                continue
            v_deg = len(edge_index.get(v, []))
            if max_neighbor_degree is not None and v_deg > max_neighbor_degree:
                continue
            scored.append((v_deg, anchor_idx, v, e_rel, v_type))
        scored.sort(key=lambda t: t[0])
        kept = [(h, t, r, vt) for _d, h, t, r, vt in scored[:cap]]
        return kept, len(scored)

    drug_keep    = lambda t: t == gene_type
    disease_keep = lambda t: t is not None and t != drug_type
    de, n_d = collect(drug_idx,    disease_idx, drug_keep,    drug_max_degree)
    xe, n_x = collect(disease_idx, drug_idx,    disease_keep, disease_max_degree)
    return de, xe, n_d, n_x


def verbalize_edges(edges, name_map):
    return '\n'.join(
        f'{name_map.get(u, u)} {normalise_predicate(rel)} {name_map.get(v, v)}.'
        for u, v, rel, _ttype in edges
    )


def _section(label, name, in_kg, edges, total, name_map):
    if not in_kg:
        body = f'(This KG does not contain {name}.)'
    elif total == 0:
        body = '(No edges in this KG.)'
    else:
        note = (f'(showing top {len(edges)} of {total} by neighbour rarity)\n'
                if total > len(edges) else '')
        body = note + verbalize_edges(edges, name_map)
    return f'{label}:\n{body}'


# ── The three packaging variants ───────────────────────────────────────────
def variant_drug_only(ctx, hub_threshold=None):
    de, _xe, n_d, _n_x = retrieve_subgraph(
        ctx['d_idx'], ctx['x_idx'], ctx['edge_index'], ctx['type_map'],
        gene_type=ctx['gene_t'], drug_type=ctx['drug_t'])
    return _section('Drug edges', ctx['drug_name'], ctx['drug_in_kg'],
                    de, n_d, ctx['name_map']), len(de), 0


def variant_typed_current(ctx, hub_threshold=None):
    de, xe, n_d, n_x = retrieve_subgraph(
        ctx['d_idx'], ctx['x_idx'], ctx['edge_index'], ctx['type_map'],
        gene_type=ctx['gene_t'], drug_type=ctx['drug_t'])
    drug_s    = _section('Drug edges',    ctx['drug_name'],    ctx['drug_in_kg'],    de, n_d, ctx['name_map'])
    disease_s = _section('Disease edges', ctx['disease_name'], ctx['disease_in_kg'], xe, n_x, ctx['name_map'])
    return f'{drug_s}\n\n{disease_s}', len(de), len(xe)


def variant_hub_filtered(ctx, hub_threshold=50):
    """Symmetric hub filter — applied to both drug and disease sides."""
    de, xe, n_d, n_x = retrieve_subgraph(
        ctx['d_idx'], ctx['x_idx'], ctx['edge_index'], ctx['type_map'],
        gene_type=ctx['gene_t'], drug_type=ctx['drug_t'],
        drug_max_degree=hub_threshold,
        disease_max_degree=hub_threshold)
    drug_s    = _section('Drug edges',    ctx['drug_name'],    ctx['drug_in_kg'],    de, n_d, ctx['name_map'])
    disease_s = _section('Disease edges', ctx['disease_name'], ctx['disease_in_kg'], xe, n_x, ctx['name_map'])
    return f'{drug_s}\n\n{disease_s}', len(de), len(xe)


def variant_disease_hub_only(ctx, hub_threshold=50):
    """Asymmetric — drug side keeps all G/P targets; disease side hub-filtered."""
    de, xe, n_d, n_x = retrieve_subgraph(
        ctx['d_idx'], ctx['x_idx'], ctx['edge_index'], ctx['type_map'],
        gene_type=ctx['gene_t'], drug_type=ctx['drug_t'],
        drug_max_degree=None,
        disease_max_degree=hub_threshold)
    drug_s    = _section('Drug edges',    ctx['drug_name'],    ctx['drug_in_kg'],    de, n_d, ctx['name_map'])
    disease_s = _section('Disease edges', ctx['disease_name'], ctx['disease_in_kg'], xe, n_x, ctx['name_map'])
    return f'{drug_s}\n\n{disease_s}', len(de), len(xe)


VARIANTS = {
    'drug_only':         variant_drug_only,
    'typed_current':     variant_typed_current,
    'hub_filtered':      variant_hub_filtered,
    'disease_hub_only':  variant_disease_hub_only,
}


# ── Sample selection ────────────────────────────────────────────────────────
def sample_pairs(pairs_df, n, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    strata = sorted(pairs_df['stratum'].unique())
    k_per = max(1, n // len(strata))
    picked = []
    for s in strata:
        pool = pairs_df[pairs_df['stratum'] == s]
        idx = rng.choice(len(pool), size=min(k_per, len(pool)), replace=False)
        picked.append(pool.iloc[idx])
    return pd.concat(picked, ignore_index=True).head(n)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kg', default='primekg')
    ap.add_argument('--n-pairs', type=int, default=20)
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument('--hub-threshold', type=int, default=50,
                    help='Drop neighbours with total degree > this in hub_filtered.')
    args = ap.parse_args()

    assert_ollama_up()
    config = load_config(find_config(BASE))
    pairs  = pd.read_parquet(TABLES / 'pairs.parquet')
    sample = sample_pairs(pairs, n=args.n_pairs).reset_index(drop=True)

    print(f'\nSample ({len(sample)} pairs) from {args.kg}:')
    print(sample[['pair_idx', 'drug_name', 'disease_name', 'stratum', 'label']]
          .to_string(index=False))
    print(f'\nWill make {len(sample) * len(args.variants)} LLM calls '
          f'({len(args.variants)} variants × {len(sample)} pairs × 1 reseed)')
    print(f'Variants: {args.variants}    Hub threshold: {args.hub_threshold}')

    strat = get_strategy('llm_prompt')
    xw    = load_crosswalks()
    rows  = []

    print(f'\n=== loading {args.kg} ===')
    kg_df, nodes_df = load_kg(args.kg, config)
    mp = build_lookup_maps(nodes_df)
    type_map, name_map = mp['node_type_map'], mp['node_name_map']
    idx_to_type = dict(zip(nodes_df['idx'].astype(int),
                           nodes_df['type'].astype(str)))

    et = config['knowledge_graphs'][args.kg]['entity_types']
    drug_t, disease_t = et.get('Drug'), et.get('Disease')
    gene_t = et.get('Gene/Protein')

    # Compound-ID-aware id_to_idx (mirrors nb09)
    disease_sep = config['knowledge_graphs'][args.kg].get('disease_id_separator')
    id_to_idx = defaultdict(list)
    for raw, idx in zip(nodes_df['id'].astype(str), nodes_df['idx'].astype(int)):
        node_t = idx_to_type.get(idx)
        keys = {raw, raw.split('::', 1)[-1] if '::' in raw else raw}
        if disease_sep and node_t == disease_t and disease_sep in raw:
            for part in raw.split(disease_sep):
                if part:
                    keys.add(part)
        for k in keys:
            id_to_idx[k].append((idx, node_t))

    def _tc(cands, want):
        for c in cands:
            for idx, t in id_to_idx.get(c, []):
                if t == want:
                    return idx
        return None

    to_kg_drug_ids, to_kg_disease_ids = make_resolvers(config, args.kg, xw)
    edge_index = build_edge_index(kg_df)

    t0 = time.time()
    for _, row in sample.iterrows():
        pi = int(row['pair_idx'])
        d_idx = _tc(to_kg_drug_ids(row['drug_id']),       drug_t)
        x_idx = _tc(to_kg_disease_ids(row['disease_id']), disease_t)
        drug_in = d_idx is not None
        dis_in  = x_idx is not None

        ctx = {
            'd_idx': d_idx, 'x_idx': x_idx,
            'edge_index': edge_index, 'type_map': type_map, 'name_map': name_map,
            'gene_t': gene_t, 'drug_t': drug_t,
            'drug_name':    row['drug_name'],
            'disease_name': row['disease_name'],
            'drug_in_kg':   drug_in,
            'disease_in_kg': dis_in,
        }

        for variant_name in args.variants:
            kg_text, n_d_kept, n_x_kept = VARIANTS[variant_name](
                ctx, hub_threshold=args.hub_threshold)
            t_call = time.time()
            result = strat.execute(
                ollama_query, row['drug_name'], row['disease_name'],
                kg_text, seed=RANDOM_SEED + pi)
            elapsed = time.time() - t_call
            rows.append({
                'pair_idx':         pi,
                'kg':               args.kg,
                'variant':          variant_name,
                'stratum':          row['stratum'],
                'label':            int(row['label']),
                'drug_name':        row['drug_name'],
                'disease_name':     row['disease_name'],
                'drug_in_kg':       drug_in,
                'disease_in_kg':    dis_in,
                'pred':             result['pred'],
                'confidence':       result['confidence'],
                'correct':          (result['pred'] == int(row['label']))
                                     if result['pred'] is not None else False,
                'prompt_chars':     len(kg_text),
                'n_drug_edges':     n_d_kept,
                'n_disease_edges':  n_x_kept,
                'elapsed_s':        round(elapsed, 1),
                'response':         result['response'],
                'error':            result['error'],
            })
        print(f'  pi={pi:3d}  {row["drug_name"][:30]:30s} × '
              f'{row["disease_name"][:30]:30s}  '
              f'(drug_in_kg={drug_in}, disease_in_kg={dis_in})')

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}   ({len(df)} rows, {(time.time() - t0)/60:.1f} min)')

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'Per-variant accuracy on {args.kg}  (full pool, n={args.n_pairs})')
    print('=' * 70)
    print(df.groupby('variant')['correct']
            .agg(accuracy='mean', n='count').round(3).to_string())

    anch = df[df['drug_in_kg'] & df['disease_in_kg']]
    print(f'\n=== Anchored-only (n={len(anch) // len(args.variants)}) ===')
    if len(anch):
        print(anch.groupby('variant')['correct']
                  .agg(accuracy='mean', n='count').round(3).to_string())

    print('\n=== Per-stratum × variant accuracy ===')
    print((df.groupby(['stratum', 'variant'])['correct']
             .mean().unstack().round(3).to_string()))

    print('\n=== Prompt size (chars) by variant ===')
    print(df.groupby('variant')['prompt_chars']
            .agg(median='median',
                 p95=lambda s: int(s.quantile(0.95)),
                 max='max')
            .to_string())

    print('\n=== Kept-edge counts by variant ===')
    print(df.groupby('variant')[['n_drug_edges', 'n_disease_edges']]
            .agg(['median', 'max']).to_string())

    # Flip analysis: pivot to compare predictions across variants on the same pair
    print('\n=== Flips: (pair) where prediction differs across variants ===')
    pivot = df.pivot_table(index=['pair_idx', 'stratum', 'label'],
                            columns='variant', values='pred',
                            aggfunc='first')
    flips = pivot[pivot.nunique(axis=1) > 1]
    if len(flips):
        print(f'  {len(flips)} pairs flip across variants. First 15:')
        print(flips.head(15).to_string())
    else:
        print('  no flips — all variants agreed')

    # ── Reasoning comparison blocks ────────────────────────────────────────
    def _reason(s):
        if not isinstance(s, str): return ''
        m = re.search(r'\{.*\}', s, re.DOTALL)
        if not m: return s[:160]
        try:
            return str(json.loads(m.group(0)).get('reasoning', ''))[:240]
        except Exception:
            return s[:160]

    def _compare_block(label, va, vb, stratum_filter=None):
        if va not in args.variants or vb not in args.variants:
            return
        sub_df = df[df['variant'].isin([va, vb])]
        if stratum_filter:
            sub_df = sub_df[sub_df['stratum'].isin(stratum_filter)]
        print(f'\n=== Reasoning: {label}  ({va} vs {vb}) ===')
        any_flip = False
        for pi in sorted(sub_df['pair_idx'].unique()):
            sub = sub_df[sub_df['pair_idx'] == pi]
            try:
                a = sub[sub['variant'] == va].iloc[0]
                b = sub[sub['variant'] == vb].iloc[0]
            except IndexError:
                continue
            if a['pred'] != b['pred']:
                any_flip = True
                print(f'\n  [pi={pi} | {a["stratum"]}] {a["drug_name"]} × {a["disease_name"]}  '
                      f'(truth={"YES" if a["label"] == 1 else "NO"})')
                print(f'    {va:18s} → pred={a["pred"]}  | "{_reason(a["response"])}"')
                print(f'    {vb:18s} → pred={b["pred"]}  | "{_reason(b["response"])}"')
        if not any_flip:
            print(f'  (no flips between {va} and {vb})')

    # The two pairings that actually answer the asymmetric-filter question:
    _compare_block('did asymmetric filter fix the negatives?',
                   'typed_current', 'disease_hub_only',
                   stratum_filter=['neg_plausible', 'neg_random'])
    _compare_block('did asymmetric filter preserve the positives?',
                   'hub_filtered', 'disease_hub_only',
                   stratum_filter=['approved', 'phase3', 'phase12'])


if __name__ == '__main__':
    main()
