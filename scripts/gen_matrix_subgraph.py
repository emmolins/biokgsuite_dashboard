#!/usr/bin/env python3
"""
Generate the MATRIX subgraph data for the BioKGSuite dashboard's Subgraph Explorer.

WHERE TO RUN: on a machine that can load the MATRIX KG (e.g. your HPC), from the
repo root, with the conda env active:

    cd ~/biokgsuite
    python scripts/gen_matrix_subgraph.py

It writes `matrix_subgraph.json` (small). Send that file back and it gets spliced
into dashboard.html + docs/dashboard.html (each seed gets a `kgs.matrix` entry).

It uses the repo's own loader, so the graph matches the filtered MATRIX the
benchmark scored, and node names/types are the clean canonical ones.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'src'))
from loading import load_kg, load_config, find_config          # noqa: E402
from graph_utils import build_lookup_maps                      # noqa: E402

# ── the 15 dashboard seeds, resolved to canonical CURIEs present in MATRIX ────
# (human Entrez for genes, CHEBI for drugs, MONDO for diseases). The loader's
# nodes_df 'id' column carries these CURIEs. Fallback: case-insensitive name.
SEEDS = {
    'metformin':  ('Metformin',        'drug',    ['CHEBI:6801']),
    'aspirin':    ('Aspirin',          'drug',    ['CHEBI:15365']),
    'ibuprofen':  ('Ibuprofen',        'drug',    ['CHEBI:5855']),
    'levodopa':   ('Levodopa',         'drug',    ['CHEBI:15765', 'DRUGBANK:DB01235', 'PUBCHEM.COMPOUND:6047', 'UNII:46627O600J']),
    'warfarin':   ('Warfarin',         'drug',    ['CHEBI:10033']),
    'narcolepsy': ('Narcolepsy',       'disease', ['MONDO:0021107']),
    'tourette':   ('Tourette Syndrome','disease', ['MONDO:0007661']),
    'gout':       ('Gout',             'disease', ['MONDO:0005393', 'HPO:0001997']),
    'vitiligo':   ('Vitiligo',         'disease', ['MONDO:0008661', 'HPO:0001045']),
    'celiac':     ('Celiac Disease',   'disease', ['MONDO:0005130']),
    'tp53':       ('TP53',  'gene', ['NCBIGene:7157']),
    'brca1':      ('BRCA1', 'gene', ['NCBIGene:672']),
    'egfr':       ('EGFR',  'gene', ['NCBIGene:1956']),
    'apoe':       ('APOE',  'gene', ['NCBIGene:348']),
    'ace2':       ('ACE2',  'gene', ['NCBIGene:59272']),
}

CAP_H1 = 40      # max 1-hop neighbours kept per seed (viz readability)
CAP_H2 = 30      # max 2-hop neighbours kept per seed

# map loader entity-type strings to the dashboard's NODE_COLORS vocabulary
def map_type(t: str) -> str:
    t = (t or '').lower()
    table = {
        'drug': 'drug', 'compound': 'compound', 'small molecule': 'compound',
        'disease': 'disease', 'phenotypic feature': 'phenotype', 'phenotype': 'phenotype',
        'effect/phenotype': 'phenotype', 'gene': 'gene', 'protein': 'protein',
        'gene/protein': 'gene/protein', 'pathway': 'pathway', 'anatomy': 'anatomy',
        'molecular function': 'molecular_function', 'biological process': 'biological_process',
        'cellular component': 'cellular_component', 'exposure': 'other',
    }
    return table.get(t, t.replace(' ', '_') or 'other')


def main():
    print('Loading MATRIX (this is the slow part) ...', flush=True)
    config = load_config(find_config(BASE))
    kg_df, nodes_df = load_kg('matrix', config)
    mp = build_lookup_maps(nodes_df)
    name_map = mp['node_name_map']                       # idx -> name
    idx_to_type = dict(zip(nodes_df['idx'].astype(int), nodes_df['type'].astype(str)))

    # id (CURIE) -> idx, and lowercase-name -> idx, for seed resolution
    id_to_idx = {}
    name_to_idx = {}
    for idx, _id, nm in zip(nodes_df['idx'].astype(int),
                            nodes_df['id'].astype(str), nodes_df['name'].astype(str)):
        id_to_idx.setdefault(_id, idx)
        name_to_idx.setdefault(nm.lower(), idx)

    # undirected adjacency from the edge table
    print('Building adjacency ...', flush=True)
    adj = defaultdict(list)
    for x, y in zip(kg_df['x_index'].astype(int), kg_df['y_index'].astype(int)):
        adj[x].append(y)
        adj[y].append(x)

    def resolve(curies, name):
        for c in curies:
            if c in id_to_idx:
                return id_to_idx[c]
        nl = name.lower()
        if nl in name_to_idx:
            return name_to_idx[nl]
        # last resort: shortest node name that contains the label
        hits = sorted((k for k in name_to_idx if nl in k), key=len)
        return name_to_idx[hits[0]] if hits else None

    def node_obj(idx, hop):
        return {'i': str(idx), 't': map_type(idx_to_type.get(idx, 'other')),
                'l': name_map.get(idx, str(idx)), 'h': hop}

    out = {}
    for sk, (label, typ, curies) in SEEDS.items():
        s = resolve(curies, label)
        if s is None:
            print(f'  WARN {sk}: not found in MATRIX (skipped)')
            continue
        h1 = list(dict.fromkeys(adj.get(s, [])))          # dedup, keep order
        h1 = [n for n in h1 if n != s][:CAP_H1]
        seen = {s, *h1}
        h2 = []
        for n in h1:
            for nn in adj.get(n, []):
                if nn not in seen:
                    seen.add(nn); h2.append(nn)
                    if len(h2) >= CAP_H2:
                        break
            if len(h2) >= CAP_H2:
                break
        out[sk] = {'1': {'n': [node_obj(n, 1) for n in h1]},
                   '2': {'n': [node_obj(n, 2) for n in h2]}}
        print(f'  {sk:12} seed={s}  h1={len(h1)}  h2={len(h2)}')

    Path('matrix_subgraph.json').write_text(json.dumps(out))
    print(f'\nwrote matrix_subgraph.json  ({len(out)}/{len(SEEDS)} seeds resolved)')


if __name__ == '__main__':
    main()
