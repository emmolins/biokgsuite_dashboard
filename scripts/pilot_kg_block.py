"""Mini comparison of kg_block design variants for nb09's drug–disease
plausibility experiment.

Picks N stratified pairs from pairs.parquet, runs them through 4 kg_block
builders × every KG against the local llama3.1:8b, and reports per-variant
accuracy + prompt-size distribution + per-pair flip table. Use this to
choose a kg_block design before running the full notebook.

Variants
--------
  no_kg          No KG context at all (baseline — LLM parametric memory).
  all_uncapped   All edges incident to drug + disease, no filter, no cap
                 (the current nb09 behaviour).
  typed          Gene/Protein neighbours only. Filters out phenotypes,
                 anatomy, side-effects, pathways, etc.
  typed_bridge   Gene/Protein only + an explicit "Shared mechanism" section
                 listing drug-target ∩ disease-gene pairs as 2-sentence
                 bridges.

Usage
-----
  python scripts/pilot_kg_block.py                          # all 6 KGs, 10 pairs
  python scripts/pilot_kg_block.py --kgs primekg matrix     # subset
  python scripts/pilot_kg_block.py --n-pairs 5              # smaller

Outputs
-------
  results/tables/09_llm_runs/09_pilot_kg_block.csv  (one row per pair × kg × variant)
  Summary tables to stdout.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path('/Users/shil6661/biokgsuite')
if not BASE.exists():
    BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'src'))

from loading import load_kg, load_config, find_config            # noqa: E402
from prompting_strategies import get_strategy                    # noqa: E402
from graph_utils import build_lookup_maps                        # noqa: E402

GOLD    = BASE / 'data' / 'gold_standards'
TABLES  = BASE / 'results' / 'tables' / '09_llm_runs'
OUT_CSV = TABLES / '09_pilot_kg_block.csv'

# ── LLM client ──────────────────────────────────────────────────────────────
LLM_TAG     = 'llama3.1:8b'
OLLAMA_URL  = 'http://localhost:11434/api/generate'
TEMPERATURE = 0.0
NUM_CTX     = 16384            # raise from Ollama default 2048 so long prompts don't truncate
RANDOM_SEED = 42


def ollama_query(prompt, seed=0, temperature=None, max_tokens=None, format=None):
    """Match nb09's ollama_query signature so the LLMPrompt strategy can call us."""
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
        raise SystemExit(f'Model {LLM_TAG!r} not pulled in Ollama. '
                         f'Try:  ollama pull {LLM_TAG}')
    print(f'Ollama up — {LLM_TAG} ✓')


# ── Crosswalks (subset of nb09 §2/§3) ───────────────────────────────────────
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
    """Assemble the gold-standard crosswalk lookups (mirrors nb09 §2)."""
    do        = pd.read_csv(GOLD / 'do_diseases.csv')
    mesh_doid = pd.read_csv(GOLD / 'mesh_to_doid.csv', on_bad_lines='skip')
    xref      = pd.read_csv(GOLD / 'drugbank_xref.csv')
    sssom     = pd.read_csv(GOLD / 'mondo.sssom.tsv', sep='\t', comment='#')
    sssom     = sssom[sssom['predicate_id'] == 'skos:exactMatch']

    # MONDO ↔ DOID
    _do = do.dropna(subset=['mondo_id', 'doid']).copy()
    _do['m'] = _do['mondo_id'].str.replace('MONDO:', '', regex=False).str.lstrip('0').replace({'': '0'})
    _do['d'] = _do['doid'].str.replace('DOID:', '', regex=False).str.lstrip('0').replace({'': '0'})
    mondo_to_doid_all = defaultdict(set)
    doid_to_mondo_all = defaultdict(set)
    for _, r in _do.iterrows():
        mondo_to_doid_all[r['m']].add(r['d'])
        doid_to_mondo_all[r['d']].add(r['m'])

    # MESH ↔ DOID
    mesh_doid['mesh_clean'] = mesh_doid['mesh_id'].str.replace(r'\s*\{.*\}', '', regex=True).str.strip()
    _mh = (mesh_doid.assign(src_rank=mesh_doid['source'].map({'DO': 0, 'MONDO': 1}).fillna(2))
                    .sort_values('src_rank').drop_duplicates('mesh_clean', keep='first'))
    mesh_to_doid = {r['mesh_clean']: (r['doid'].replace('DOID:', '').lstrip('0') or '0')
                    for _, r in _mh.iterrows() if pd.notna(r['doid'])}
    doid_to_mesh = defaultdict(set)
    for m, d in mesh_to_doid.items():
        if m.startswith('MESH:'):
            doid_to_mesh[d].add(m)

    # OpenBioLink PubChem → DrugBank
    pc_path = BASE / 'data' / 'openbilink' / 'pubchem_to_drugbank.csv'
    obl_db_to_pc = defaultdict(set)
    if pc_path.exists():
        _pc = pd.read_csv(pc_path)
        for pc, db in zip(_pc.iloc[:, 0].astype(str), _pc.iloc[:, 1].astype(str)):
            if db.startswith('DB'):
                obl_db_to_pc[db].add(pc)

    # DrugBank cross-namespace
    drug_equiv = defaultdict(set)
    for db, ns, eid in zip(xref.drugbank_id.astype(str),
                           xref.namespace.astype(str),
                           xref.external_id.astype(str)):
        drug_equiv[db].update({f'{ns}:{eid}', eid})
        if ns == 'PUBCHEM.COMPOUND':
            drug_equiv[db].update({f'CID:{eid}', f'pubchem.compound:{eid}'})

    # SSSOM disease equivalents (one-hop closure)
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
        'obl_db_to_pc':      obl_db_to_pc,
        'drug_equiv':        drug_equiv,
        'disease_equiv':     disease_equiv,
    }


def make_resolvers(config, kg_name, xw):
    """Build per-KG (to_kg_drug_ids, to_kg_disease_ids) closures."""
    drug_equiv        = xw['drug_equiv']
    obl_db_to_pc      = xw['obl_db_to_pc']
    mondo_to_doid_all = xw['mondo_to_doid_all']
    doid_to_mondo_all = xw['doid_to_mondo_all']
    mesh_to_doid      = xw['mesh_to_doid']
    doid_to_mesh      = xw['doid_to_mesh']
    disease_equiv     = xw['disease_equiv']

    def to_kg_drug_ids(db_id):
        s = str(db_id).strip()
        cands = {s} | drug_equiv.get(s, set())
        if kg_name == 'openbilink':
            for pc in obl_db_to_pc.get(s, set()):
                cands.update({f'PUBCHEM.COMPOUND:{pc}', f'PUBCHEM.COMPOUND_{pc}', pc})
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


# ── Predicate normalization (mirrors nb09 §4) ───────────────────────────────
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


# ── Variant builders ────────────────────────────────────────────────────────
# Each takes a uniform `ctx` dict and returns the kg_text. The strategy
# wraps it with "Knowledge graph context:\n" (or, for empty text, emits
# "No knowledge graph context provided.").

def _verbalize(edges, name_map):
    """Render typed edges (head, tail, rel, tail_type) as one-sentence lines."""
    return '\n'.join(
        f'{name_map.get(u, u)} {normalise_predicate(rel)} {name_map.get(v, v)}.'
        for u, v, rel, _ttype in edges
    )


def _two_section_parts(ctx, drug_edges, disease_edges):
    def section(label, name, in_kg, edges):
        if not in_kg:
            body = f'(This KG does not contain {name}.)'
        elif not edges:
            body = '(No edges in this KG.)'
        else:
            body = _verbalize(edges, ctx['name_map'])
        return f'{label}:\n{body}'
    return (section('Drug edges',    ctx['drug_name'],    ctx['drug_in_kg'],    drug_edges),
            section('Disease edges', ctx['disease_name'], ctx['disease_in_kg'], disease_edges))


def _two_section(ctx, drug_edges, disease_edges):
    a, b = _two_section_parts(ctx, drug_edges, disease_edges)
    return f'{a}\n\n{b}'


def variant_no_kg(ctx):
    return ''


def variant_all_uncapped(ctx):
    return _two_section(ctx, ctx['drug_edges'], ctx['disease_edges'])


def variant_typed(ctx):
    gt = ctx['gene_type']
    de = [e for e in ctx['drug_edges']    if e[3] == gt]
    xe = [e for e in ctx['disease_edges'] if e[3] == gt]
    return _two_section(ctx, de, xe)


def variant_typed_bridge(ctx):
    gt = ctx['gene_type']
    de = [e for e in ctx['drug_edges']    if e[3] == gt]
    xe = [e for e in ctx['disease_edges'] if e[3] == gt]

    drug_targets  = {e[1]: e[2] for e in de}   # tail_idx → predicate
    disease_genes = {e[1]: e[2] for e in xe}
    shared = sorted(drug_targets.keys() & disease_genes.keys())
    nm = ctx['name_map']
    if shared:
        bridge = '\n'.join(
            f'{ctx["drug_name"]} {normalise_predicate(drug_targets[g])} '
            f'{nm.get(g, g)}; {ctx["disease_name"]} '
            f'{normalise_predicate(disease_genes[g])} {nm.get(g, g)}.'
            for g in shared
        )
    else:
        bridge = '(No shared drug-target / disease-gene found in this KG.)'

    drug_section, disease_section = _two_section_parts(ctx, de, xe)
    return (f'Shared mechanism:\n{bridge}\n\n'
            f'{drug_section}\n\n{disease_section}')


VARIANTS = {
    'no_kg':         variant_no_kg,
    'all_uncapped':  variant_all_uncapped,
    'typed':         variant_typed,
    'typed_bridge':  variant_typed_bridge,
}


# ── Sample selection ────────────────────────────────────────────────────────
def sample_pairs(pairs_df, n, seed=RANDOM_SEED):
    """Stratified across pair strata; truncates to exactly n."""
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
    ap.add_argument('--n-pairs', type=int, default=10,
                    help='Total pairs (stratified across strata). Default 10.')
    ap.add_argument('--kgs', nargs='+',
                    default=['biokg', 'drkg', 'hetionet', 'matrix',
                             'openbilink', 'primekg'],
                    help='KGs to include.')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS),
                    choices=list(VARIANTS),
                    help='kg_block variants to compare.')
    args = ap.parse_args()

    assert_ollama_up()
    config = load_config(find_config(BASE))
    pairs  = pd.read_parquet(TABLES / 'pairs.parquet')
    sample = sample_pairs(pairs, n=args.n_pairs).reset_index(drop=True)

    n_calls = len(sample) * len(args.kgs) * len(args.variants)
    print(f'\nSample ({len(sample)} pairs):')
    print(sample[['pair_idx', 'drug_name', 'disease_name', 'stratum', 'label']]
          .to_string(index=False))
    print(f'\nWill make {n_calls} LLM calls '
          f'({len(args.variants)} variants × {len(args.kgs)} KGs × {len(sample)} pairs)')

    strat = get_strategy('llm_prompt')
    xw    = load_crosswalks()
    rows  = []
    t_start = time.time()

    for kg_name in args.kgs:
        print(f'\n=== {kg_name} ===')
        try:
            kg_df, nodes_df = load_kg(kg_name, config)
        except Exception as e:
            print(f'  [skip] {e}')
            continue

        mp = build_lookup_maps(nodes_df)
        type_map, name_map = mp['node_type_map'], mp['node_name_map']
        idx_to_type = dict(zip(nodes_df['idx'].astype(int),
                               nodes_df['type'].astype(str)))

        et       = config['knowledge_graphs'][kg_name]['entity_types']
        drug_t   = et.get('Drug')
        disease_t = et.get('Disease')
        gene_t   = et.get('Gene/Protein')

        # If this KG compound-joins equivalent disease IDs (e.g. PrimeKG
        # joins MONDO components with '_'), also index each component of a
        # disease node ID separately so a resolver candidate like '24300'
        # can reach a compound node like '17324_..._44_24300'. Guarded by
        # node type so non-disease IDs with underscores aren't affected.
        disease_sep = config['knowledge_graphs'][kg_name].get('disease_id_separator')
        id_to_idx = defaultdict(list)
        for raw, idx in zip(nodes_df['id'].astype(str),
                            nodes_df['idx'].astype(int)):
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

        to_kg_drug_ids, to_kg_disease_ids = make_resolvers(config, kg_name, xw)
        edge_index = build_edge_index(kg_df)

        for _, row in sample.iterrows():
            pi    = int(row['pair_idx'])
            d_idx = _tc(to_kg_drug_ids(row['drug_id']),       drug_t)
            x_idx = _tc(to_kg_disease_ids(row['disease_id']), disease_t)
            drug_in = d_idx is not None
            dis_in  = x_idx is not None

            # Typed edges (head, tail, rel, tail_type), leakage-filtered
            d_edges, x_edges = [], []
            if drug_in:
                for v, e_rel in edge_index.get(d_idx, []):
                    if v == x_idx:
                        continue
                    d_edges.append((d_idx, v, e_rel, idx_to_type.get(v)))
            if dis_in:
                for v, e_rel in edge_index.get(x_idx, []):
                    if v == d_idx:
                        continue
                    x_edges.append((x_idx, v, e_rel, idx_to_type.get(v)))

            ctx = {
                'drug_name':     row['drug_name'],
                'disease_name':  row['disease_name'],
                'drug_in_kg':    drug_in,
                'disease_in_kg': dis_in,
                'drug_edges':    d_edges,
                'disease_edges': x_edges,
                'name_map':      name_map,
                'gene_type':     gene_t,
            }

            for variant_name in args.variants:
                kg_text = VARIANTS[variant_name](ctx)
                t0 = time.time()
                result = strat.execute(
                    ollama_query, row['drug_name'], row['disease_name'],
                    kg_text, seed=RANDOM_SEED + pi)
                elapsed = time.time() - t0
                rows.append({
                    'pair_idx':         pi,
                    'kg':               kg_name,
                    'variant':          variant_name,
                    'stratum':          row['stratum'],
                    'label':            int(row['label']),
                    'drug_in_kg':       drug_in,
                    'disease_in_kg':    dis_in,
                    'pred':             result['pred'],
                    'confidence':       result['confidence'],
                    'correct':          (result['pred'] == int(row['label']))
                                         if result['pred'] is not None else False,
                    'prompt_chars':     len(kg_text),
                    'n_drug_edges':     len(d_edges),
                    'n_disease_edges':  len(x_edges),
                    'elapsed_s':        round(elapsed, 1),
                    'response':         result['response'],
                    'error':            result['error'],
                })

            print(f'  pi={pi:3d}  {row["drug_name"][:30]:30s} × '
                  f'{row["disease_name"][:30]:30s}  '
                  f'(drug_in_kg={drug_in}, disease_in_kg={dis_in}, '
                  f'd_edges={len(d_edges)}, x_edges={len(x_edges)})')

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    elapsed_total = time.time() - t_start
    print(f'\nSaved: {OUT_CSV}   ({len(df)} rows, {elapsed_total/60:.1f} min)')

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Per-variant accuracy (across all KGs and pairs)')
    print('=' * 60)
    print(df.groupby('variant')['correct']
            .agg(accuracy='mean', n='count').round(3).to_string())

    print('\n=== Per-variant × KG accuracy ===')
    print((df.groupby(['variant', 'kg'])['correct']
             .mean().unstack().round(3).to_string()))

    print('\n=== Prompt size (chars) by variant ===')
    print(df.groupby('variant')['prompt_chars']
            .agg(median='median',
                 p95=lambda s: int(s.quantile(0.95)),
                 max='max')
            .round(0).to_string())

    anch = df[df['drug_in_kg'] & df['disease_in_kg']]
    print(f'\n=== Anchored-only accuracy (both anchors in KG)  '
          f'n={len(anch)}/{len(df)} ===')
    if len(anch):
        print(anch.groupby('variant')['correct']
                  .agg(accuracy='mean', n='count').round(3).to_string())

    print('\n=== Per-stratum × variant accuracy ===')
    print((df.groupby(['stratum', 'variant'])['correct']
             .mean().unstack().round(3).to_string()))

    # Flip analysis: same (pair, kg), different variant → different pred
    print('\n=== Flips: (pair × kg) combos where prediction differs across variants ===')
    pivot = (df[df['pred'].notna()]
                .pivot_table(index=['pair_idx', 'kg'], columns='variant',
                             values='pred', aggfunc='first'))
    flippy = pivot[pivot.nunique(axis=1) > 1]
    if len(flippy):
        print(f'  {len(flippy)} (pair × kg) combos flipped across variants.')
        print('  First 15:')
        print(flippy.head(15).to_string())
    else:
        print('  no flips — all variants agreed everywhere')


if __name__ == '__main__':
    main()
