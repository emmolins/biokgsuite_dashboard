"""Path enumeration and analysis utilities for Knowledge Graphs.

Provides BFS and path counting utilities for analyzing connectivity
between drug-disease pairs in knowledge graphs.
"""

from collections import deque, Counter
import networkx as nx


# Configuration constants
MAX_PATH_LEN = 3        # Default max hops; balances mechanistic coverage with tractability
BFS_NODE_CAP = 20_000   # Deterministic budget for reachability pre-check
PATH_ENUM_CAP = 100     # Max paths per pair (sufficient for Mann-Whitney)


def bfs_depth_limited(G, src, tgt, max_depth, node_cap=BFS_NODE_CAP):
    """Return (reachable, budget_exceeded).

    BFS to max_depth hops; exits after node_cap visited nodes.
    Budget-exceeded pairs are excluded from analysis rather than
    silently treated as unreachable (avoids bias toward sparse regions).

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph.
    src : node
        Source node.
    tgt : node
        Target node.
    max_depth : int
        Maximum search depth (hops).
    node_cap : int
        Maximum number of visited nodes before budge exceeded.

    Returns
    -------
    reachable : bool
        True if target is reachable within max_depth and node_cap.
    budget_exceeded : bool
        True if node_cap was hit (pair excluded from analysis).
    """
    if src == tgt:
        return True, False
    visited, queue = {src}, deque([(src, 0)])
    while queue:
        if len(visited) >= node_cap:
            return False, True
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nbr in G.neighbors(node):
            if nbr == tgt:
                return True, False
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, depth + 1))
    return False, False


def count_paths(G, src, tgt, cutoff, ntm, cap=PATH_ENUM_CAP):
    """Count simple paths src→tgt up to cutoff hops, capped at cap.

    BFS pre-check skips pairs with no short path (avoids combinatorial
    explosion in the no-path case). Timeout checked every iteration.

    Parameters
    ----------
    G : networkx.Graph
        Undirected graph.
    src : node
        Source node.
    tgt : node
        Target node.
    cutoff : int
        Maximum path length (hops).
    ntm : dict
        Node-to-type mapping for intermediate node labeling.
    cap : int
        Maximum number of paths to enumerate before capping.

    Returns
    -------
    n_paths : int
        Number of paths found (up to cap).
    sigs : Counter
        Counter of intermediate node type signatures.
    budget_exceeded : bool
        True if path enumeration hit the cap.
    """
    reachable, exceeded = bfs_depth_limited(G, src, tgt, cutoff)
    if not reachable:
        return 0, Counter(), exceeded
    n_paths, sigs = 0, Counter()
    try:
        for path in nx.all_simple_paths(G, src, tgt, cutoff=cutoff):
            n_paths += 1
            sig = tuple(ntm.get(v, '?') for v in path[1:-1])
            sigs[sig] += 1
            if n_paths >= cap:
                break
    except nx.NetworkXError:
        pass
    return n_paths, sigs, False


def analyze_path_diversity(G, therapeutic_pairs, neg_builder, maps, config,
                           rng, verbose=True):
    """Analyze path diversity between therapeutic vs. random drug-disease pairs.

    Parameters
    ----------
    G : networkx.Graph
        Undirected knowledge graph.
    therapeutic_pairs : list of (src, tgt) tuples
        Positive gold-standard pairs.
    neg_builder : callable
        Function build_negative_pairs(name, pos_set, n_neg, rng).
    maps : dict
        Mapping from node ID to type ('node_type_map' key).
    config : dict
        Configuration dict with 'n_sample' and path_enum_cap'.
    rng : np.random.RandomState
        RNG for reproducibility.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    dict
        Results including 'pos', 'neg' arrays, 'sigs', 'p_val', 'effect', 'n_skipped'.
    """
    import numpy as np
    from scipy import stats
    from tqdm.notebook import tqdm

    tp = therapeutic_pairs
    if not tp:
        if verbose:
            print('No therapeutic pairs — skipping');
        return None

    ntm = maps.get('node_type_map', {})
    pos_set = set(tp)

    sampled_pos = [tp[i] for i in rng.choice(len(tp), size=min(config['n_sample'], len(tp)), replace=False)]
    sampled_neg = neg_builder(pos_set, len(sampled_pos), rng)

    pos_counts, neg_counts, all_sigs = [], [], Counter()
    n_skipped = 0

    total_pairs = len(sampled_pos) + len(sampled_neg)
    with tqdm(total=total_pairs, unit='pair',
              bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

        for pairs, store in [(sampled_pos, pos_counts), (sampled_neg, neg_counts)]:
            label = 'pos' if store is pos_counts else 'neg'
            for src, tgt in pairs:
                cnt, sigs, exceeded = count_paths(G, src, tgt, config.get('max_path_len', MAX_PATH_LEN),
                                                   ntm, cap=config.get('path_enum_cap', PATH_ENUM_CAP))
                if exceeded:
                    n_skipped += 1
                else:
                    store.append(cnt)
                    if label == 'pos':
                        for sig, c in sigs.items():
                            readable = ('drug → ' + ' → '.join(sig) + ' → disease') if sig else 'direct'
                            all_sigs[readable] += c
                pbar.update(1)
                pbar.set_postfix(pos=len(pos_counts), neg=len(neg_counts), skipped=n_skipped)

    if not pos_counts or not neg_counts:
        if verbose:
            print('Insufficient resolved pairs — skipping')
        return None

    pos_arr, neg_arr = np.array(pos_counts), np.array(neg_counts)
    u, p_val = stats.mannwhitneyu(pos_arr, neg_arr, alternative='greater')
    effect = 1 - (2 * u) / (len(pos_arr) * len(neg_arr))

    return {
        'pos': pos_arr, 'neg': neg_arr, 'sigs': all_sigs,
        'p_val': p_val, 'effect': effect, 'n_skipped': n_skipped
    }
