#!/usr/bin/env python3
"""Run a single embedding model on a single KG. Saves to results/cache/."""
import json, sys, time, numpy as np, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.embedding import (TransE, RotatE, build_train_triples, kg_to_triples,
                           compute_embedding_metrics)
from src.negative_sampling import generate_negatives


def prepare_kg(kg_name, seed=42, neg_ratio=5):
    """Load and prepare KG data, cache to pkl."""
    cache = ROOT / 'results' / 'cache' / f'{kg_name}_prep.pkl'
    if cache.exists():
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        if data is not None and isinstance(data, dict) and 'train_triples' in data:
            return data

    from src.loading import load_config, load_kg
    config = load_config(str(ROOT / 'config.yaml'))
    kg_df, nodes_df = load_kg(kg_name, config)
    kg_cfg = config['knowledge_graphs'][kg_name]
    etypes = kg_cfg['entity_types']
    type_map = dict(zip(nodes_df['idx'], nodes_df['type']))
    drug_idx = {i for i, t in type_map.items() if t == etypes.get('Drug', 'Drug')}
    disease_idx = {i for i, t in type_map.items() if t == etypes.get('Disease', 'Disease')}
    gene_idx = {i for i, t in type_map.items() if t == etypes.get('Gene/Protein', 'Gene')}

    dd = kg_cfg.get('relations', {}).get('drug_disease', {})
    ind_rels = [dd['relation']] if 'relation' in dd else dd.get('relations', [])
    dt = kg_cfg.get('relations', {}).get('drug_target', {})
    dt_rels = [dt['relation']] if 'relation' in dt else dt.get('relations', [])

    mask = kg_df['relation'].isin(ind_rels)
    pairs = set()
    for _, row in kg_df[mask].iterrows():
        h, t = int(row['x_index']), int(row['y_index'])
        if h in drug_idx and t in disease_idx: pairs.add((h, t))
        elif t in drug_idx and h in disease_idx: pairs.add((t, h))
    pairs = list(pairs)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pairs))
    split = int(0.9 * len(pairs))
    test_pos = [pairs[i] for i in perm[split:]]
    all_pos = set(pairs)

    train_triples, rel_to_idx, idx_to_rel = build_train_triples(
        kg_df, set(test_pos), ind_rels)
    n_ent = int(nodes_df['idx'].max()) + 1
    node_name_map = dict(zip(nodes_df['idx'], nodes_df['name']))

    drug_targets = {}
    dt_mask = kg_df['relation'].isin(dt_rels)
    for _, row in kg_df[dt_mask].iterrows():
        h, t = int(row['x_index']), int(row['y_index'])
        if h in drug_idx and t in gene_idx: drug_targets.setdefault(h, set()).add(t)
        elif t in drug_idx and h in gene_idx: drug_targets.setdefault(t, set()).add(h)

    neg_pairs = generate_negatives(test_pos, len(test_pos) * neg_ratio,
        'type-constrained', drug_idx, disease_idx, drug_targets,
        node_name_map, all_pos, rng)

    rel_idx = rel_to_idx[ind_rels[0]]
    inv_name = f'{ind_rels[0]}_inv'
    rel_idx_inv = rel_to_idx.get(inv_name)

    prep = {'train_triples': train_triples, 'rel_to_idx': rel_to_idx,
            'n_ent': n_ent, 'n_rels': len(rel_to_idx),
            'test_pos': test_pos, 'neg_pairs': neg_pairs,
            'rel_idx': rel_idx, 'rel_idx_inv': rel_idx_inv,
            'n_train': split, 'n_test': len(test_pos)}

    with open(cache, 'wb') as f:
        pickle.dump(prep, f)
    return prep


def run_model(kg_name, model_name, n_epochs=5, dim=32, seed=42):
    p = prepare_kg(kg_name, seed=seed)
    print(f'{kg_name}/{model_name}: {len(p["train_triples"])} triples, '
          f'{p["n_ent"]} entities, {p["n_rels"]} rels', flush=True)

    Cls = TransE if model_name == 'TransE' else RotatE
    kw = dict(n_entities=p['n_ent'], n_relations=p['n_rels'],
              dim=dim, seed=seed, lr=0.01)
    if model_name == 'RotatE':
        kw['dim'] = max(dim // 2, 8)
        kw['margin'] = 6.0
    else:
        kw['margin'] = 1.0

    t0 = time.time()
    model = Cls(**kw)
    model.fit(p['train_triples'], n_epochs=n_epochs, batch_size=32768, verbose=True)
    ts = time.time() - t0
    print(f'{model_name} trained in {ts:.1f}s', flush=True)

    m = compute_embedding_metrics(model, p['test_pos'], p['neg_pairs'],
                                  p['rel_idx'], rel_idx_inv=p['rel_idx_inv'])
    m['train_time_s'] = ts
    m['n_epochs'] = n_epochs
    m['dim'] = dim
    print(f'AUROC={m["auroc"]:.4f} AUPRC={m["auprc"]:.4f} MRR={m["mrr"]:.4f} '
          f'H@10={m["hits@10"]:.4f} H@50={m["hits@50"]:.4f} H@100={m["hits@100"]:.4f}',
          flush=True)

    # Save/update JSON
    out = ROOT / 'results' / 'cache' / f'embedding_{kg_name}.json'
    data = {}
    if out.exists():
        with open(out) as f:
            data = json.load(f)
    data.setdefault('kg', kg_name)
    data.setdefault('models', {})
    data['models'][model_name] = m
    data['n_test'] = len(p['test_pos'])
    data['n_neg'] = len(p['neg_pairs'])
    data['n_entities'] = p['n_ent']
    data['n_relations'] = p['n_rels']
    data['n_train_triples'] = len(p['train_triples'])
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved to {out}', flush=True)
    return m


if __name__ == '__main__':
    kg = sys.argv[1]
    model = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    dim = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    run_model(kg, model, n_epochs=epochs, dim=dim)
