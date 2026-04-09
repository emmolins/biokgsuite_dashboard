"""Topology-based metrics for KG robustness evaluation.

Implements differential resilience (DR) metric for measuring how knowledge graphs
degrade under incremental edge dropout, comparing gold-standard therapeutic pairs
against random negative pairs.
"""

import numpy as np
import sys
import time as _time


def _build_indexed_adj_topo(G):
    """Build indexed adjacency representation for efficient dropout.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph.

    Returns
    -------
    tuple of (list, ndarray, dict)
        (node_list, edges_arr, node_to_idx) for efficient indexing.
    """
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges_arr = np.array([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()],
                         dtype=np.int32)
    return node_list, edges_arr, node_to_idx


def _bfs_reach_topo(adj, src, tgt, max_hops=2):
    """BFS reachability check on adjacency dict (indexed nodes).

    Parameters
    ----------
    adj : dict
        Adjacency dict: int → set(int).
    src : int
        Source node index.
    tgt : int
        Target node index.
    max_hops : int
        Maximum hops to search.

    Returns
    -------
    bool
        True if target is reachable from source within max_hops.
    """
    if src == tgt:
        return True
    visited = {src}
    queue = [(src, 0)]
    qi = 0
    while qi < len(queue):
        node, d = queue[qi]
        qi += 1
        if d >= max_hops:
            continue
        for nbr in adj.get(node, ()):
            if nbr == tgt:
                return True
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, d + 1))
    return False


def _dropout_recovery_topo(edges_arr, idx_pairs, drop_rate, rng, max_hops=2, n_trials=3):
    """Measure recovery of pair reachability after edge dropout.

    Parameters
    ----------
    edges_arr : ndarray
        Edge array (n_edges, 2) of indexed node pairs.
    idx_pairs : list of (int, int) tuples
        Node index pairs to test.
    drop_rate : float
        Fraction of edges to drop (0.0 to 1.0).
    rng : np.random.RandomState
        RNG for reproducibility.
    max_hops : int
        Maximum hops for BFS reachability.
    n_trials : int
        Number of repeated trials for averaging.

    Returns
    -------
    float
        Mean recovery fraction across trials.
    """
    if not idx_pairs:
        return 0.0
    n_e = len(edges_arr)
    n_rem = int(round(drop_rate * n_e))
    trials = []
    for _ in range(n_trials):
        if n_rem > 0:
            mask = np.zeros(n_e, dtype=bool)
            mask[rng.choice(n_e, size=n_rem, replace=False)] = True
            kept = edges_arr[~mask]
        else:
            kept = edges_arr
        adj_d = {}
        for u, v in kept:
            adj_d.setdefault(int(u), set()).add(int(v))
            adj_d.setdefault(int(v), set()).add(int(u))
        hits = sum(1 for si, ti in idx_pairs if _bfs_reach_topo(adj_d, si, ti, max_hops))
        trials.append(hits / len(idx_pairs))
    return float(np.mean(trials))


def _ri_topo(drop_rates, recoveries):
    """Compute resilience index from dropout curve via AUC.

    Parameters
    ----------
    drop_rates : list of float
        Drop rates tested.
    recoveries : list of float
        Recovery fractions at each drop rate.

    Returns
    -------
    float
        Normalized resilience index (0 to 1).
    """
    dr = np.array(drop_rates, dtype=float)
    rc = np.array(recoveries, dtype=float)
    auc = float(np.trapz(rc, dr))
    max_auc = float(dr[-1] - dr[0])
    return round(auc / max_auc, 4) if max_auc > 0 else 0.0


def compute_differential_resilience(G, therapeutic_pairs, neg_builder, kg_name,
                                     rng_dr, params, verbose=True):
    """Compute differential resilience (DR) metric for a KG.

    DR score = (RI_gold - RI_rand) / (1 - RI_rand), where RI is resilience index
    measured from edge dropout curves on therapeutic pairs (gold) vs. random pairs.

    Parameters
    ----------
    G : networkx.Graph
        Undirected knowledge graph.
    therapeutic_pairs : list of (src, tgt) tuples
        Gold-standard therapeutic pairs.
    neg_builder : callable
        Function build_negative_pairs(name, pos_set, n_neg, rng) returning list of pairs.
    kg_name : str
        KG name (for verbose output).
    rng_dr : np.random.RandomState
        RNG for reproducibility.
    params : dict
        Config with 'random_seed' and other settings.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict
        Results dict with keys: drop_rates, gold_curve, rand_curve, ri_gold,
        ri_rand, dr_score, n_gold, n_rand, timing info.
    """
    tp = therapeutic_pairs
    if not tp:
        if verbose:
            print(f'{kg_name}: no therapeutic pairs - skipping DR')
        return None

    N_DR_SAMPLE = 300
    DR_RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    N_DR_TRIALS = 3

    pos_set = set(tp)
    t0 = _time.time()
    if verbose:
        print(f'{kg_name}: building indexed adjacency...')
    node_list, edges_arr, node_to_idx = _build_indexed_adj_topo(G)

    n_pos = min(N_DR_SAMPLE, len(tp))
    sampled_pos = [tp[i] for i in rng_dr.choice(len(tp), size=n_pos, replace=False)]
    sampled_neg = neg_builder(kg_name, pos_set, n_pos, rng_dr)

    gold_idx = [(node_to_idx[s], node_to_idx[t])
                for s, t in sampled_pos if s in node_to_idx and t in node_to_idx]
    rand_idx = [(node_to_idx[s], node_to_idx[t])
                for s, t in sampled_neg if s in node_to_idx and t in node_to_idx]

    if verbose:
        print(f'  gold={len(gold_idx)}  random={len(rand_idx)}  edges={len(edges_arr):,}')

    gold_curve, rand_curve = [], []
    for rate in DR_RATES:
        t1 = _time.time()
        r_g = _dropout_recovery_topo(edges_arr, gold_idx, rate, rng_dr, 2, N_DR_TRIALS)
        r_r = _dropout_recovery_topo(edges_arr, rand_idx, rate, rng_dr, 2, N_DR_TRIALS)
        gold_curve.append(r_g)
        rand_curve.append(r_r)
        if verbose:
            print(f'\r  rate={rate:.2f}  gold={r_g:.3f}  random={r_r:.3f}  '
                  f'[{_time.time()-t1:.1f}s]', end='')
            sys.stdout.flush()
    if verbose:
        print()

    ri_gold = _ri_topo(DR_RATES, gold_curve)
    ri_rand = _ri_topo(DR_RATES, rand_curve)
    headroom = max(1e-6, 1.0 - ri_rand)
    dr_score = round((ri_gold - ri_rand) / headroom, 4)

    result = {
        'drop_rates': DR_RATES,
        'gold_curve': gold_curve,
        'rand_curve': rand_curve,
        'ri_gold': ri_gold,
        'ri_rand': ri_rand,
        'dr_score': dr_score,
        'n_gold': len(gold_idx),
        'n_rand': len(rand_idx),
    }
    if verbose:
        print(f'  RI_gold={ri_gold:.4f}  RI_rand={ri_rand:.4f}  DR={dr_score:.4f}  '
              f'[total {_time.time()-t0:.0f}s]')
    return result
