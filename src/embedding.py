"""Lightweight TransE and RotatE implementations for KG link prediction.

Pure NumPy implementations that mirror the standard formulations:

TransE:
    Bordes, A. et al. "Translating embeddings for modeling multi-relational
    data." NeurIPS 2013.
    Score: -||h + r - t||

RotatE:
    Sun, Z. et al. "RotatE: Knowledge Graph Embedding by Relational Rotation
    in Complex Space." ICLR 2019.
    Score: -||h o r - t||  (complex Hadamard product)

Both use margin-based ranking loss with uniform negative sampling and
vanilla SGD with per-entity updates for speed on CPU.
"""

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def _l2_normalize_rows(x, eps=1e-12):
    """L2-normalise each row of x in-place."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms
    return x


# ── TransE ───────────────────────────────────────────────────────────────────

class TransE:
    """TransE with vanilla SGD and per-entity updates.

    Parameters
    ----------
    n_entities, n_relations : int
    dim : int
        Embedding dimension.
    margin : float
    lr : float
        Learning rate.
    seed : int
    """

    name = 'TransE'

    def __init__(self, n_entities, n_relations, dim=128, margin=1.0,
                 lr=0.01, seed=42):
        self.dim = dim
        self.margin = margin
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self.n_entities = n_entities
        self.n_relations = n_relations

        bound = 6.0 / np.sqrt(dim)
        self.ent_emb = self.rng.uniform(-bound, bound,
                                        (n_entities, dim)).astype(np.float32)
        self.rel_emb = self.rng.uniform(-bound, bound,
                                        (n_relations, dim)).astype(np.float32)
        _l2_normalize_rows(self.ent_emb)

    def score(self, h_idx, r_idx, t_idx):
        """Higher = more plausible."""
        h = self.ent_emb[h_idx]
        r = self.rel_emb[r_idx]
        t = self.ent_emb[t_idx]
        return -np.linalg.norm(h + r - t, axis=-1)

    def score_pairs(self, pairs, rel_idx):
        pairs = np.asarray(pairs)
        h_idx, t_idx = pairs[:, 0], pairs[:, 1]
        if np.isscalar(rel_idx):
            rel_idx = np.full(len(pairs), rel_idx, dtype=int)
        return self.score(h_idx, rel_idx, t_idx)

    def fit(self, triples, n_epochs=50, batch_size=16384, verbose=True):
        """Train with margin ranking loss and direct SGD updates."""
        triples = np.asarray(triples, dtype=np.int32)
        n = len(triples)
        lr = self.lr

        for epoch in range(n_epochs):
            perm = self.rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = triples[perm[start:start + batch_size]]
                bs = len(batch)
                heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]

                # Corrupt head or tail
                neg_heads = heads.copy()
                neg_tails = tails.copy()
                flip = self.rng.random(bs) < 0.5
                n_h = flip.sum()
                if n_h > 0:
                    neg_heads[flip] = self.rng.randint(0, self.n_entities, n_h)
                if bs - n_h > 0:
                    neg_tails[~flip] = self.rng.randint(
                        0, self.n_entities, bs - n_h)

                # Forward
                h = self.ent_emb[heads]
                r = self.rel_emb[rels]
                t = self.ent_emb[tails]
                nh = self.ent_emb[neg_heads]
                nt = self.ent_emb[neg_tails]

                dp = h + r - t
                dn = nh + r - nt
                d_pos = np.linalg.norm(dp, axis=1)
                d_neg = np.linalg.norm(dn, axis=1)

                violation = self.margin + d_pos - d_neg
                mask = violation > 0
                n_active = mask.sum()
                if n_active == 0:
                    n_batches += 1
                    continue

                epoch_loss += violation[mask].mean()
                n_batches += 1

                # Gradient direction (unit vectors)
                dp_norm = d_pos[mask, None] + 1e-12
                dn_norm = d_neg[mask, None] + 1e-12
                gp = dp[mask] / dp_norm  # d(d_pos)/d(h+r-t)
                gn = dn[mask] / dn_norm

                # Direct SGD updates on touched entities only
                # Positive: push h+r closer to t
                self.ent_emb[heads[mask]] -= lr * gp
                self.ent_emb[tails[mask]] += lr * gp
                # Negative: push nh+r away from nt
                self.ent_emb[neg_heads[mask]] += lr * gn
                self.ent_emb[neg_tails[mask]] -= lr * gn
                # Relation
                self.rel_emb[rels[mask]] -= lr * (gp - gn)

            # Re-normalise entity embeddings once per epoch
            _l2_normalize_rows(self.ent_emb)

            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f'  Epoch {epoch+1:>4d}/{n_epochs}  loss={avg:.4f}')


# ── RotatE ───────────────────────────────────────────────────────────────────

class RotatE:
    """RotatE with vanilla SGD and per-entity updates.

    Parameters
    ----------
    n_entities, n_relations : int
    dim : int
        Complex embedding dimension (each entity has 2*dim float params).
    margin : float
    lr : float
    seed : int
    """

    name = 'RotatE'

    def __init__(self, n_entities, n_relations, dim=64, margin=6.0,
                 lr=0.01, seed=42):
        self.dim = dim
        self.margin = margin
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self.n_entities = n_entities
        self.n_relations = n_relations

        bound = 6.0 / np.sqrt(dim)
        self.ent_re = self.rng.uniform(-bound, bound,
                                       (n_entities, dim)).astype(np.float32)
        self.ent_im = self.rng.uniform(-bound, bound,
                                       (n_entities, dim)).astype(np.float32)
        self.rel_phase = self.rng.uniform(-np.pi, np.pi,
                                          (n_relations, dim)).astype(np.float32)

    def _dist(self, h_re, h_im, r_phase, t_re, t_im):
        """||h o r - t|| in complex space."""
        r_re = np.cos(r_phase)
        r_im = np.sin(r_phase)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        d_re = hr_re - t_re
        d_im = hr_im - t_im
        return np.sqrt((d_re ** 2 + d_im ** 2).sum(axis=-1) + 1e-12)

    def score(self, h_idx, r_idx, t_idx):
        return -self._dist(
            self.ent_re[h_idx], self.ent_im[h_idx],
            self.rel_phase[r_idx],
            self.ent_re[t_idx], self.ent_im[t_idx])

    def score_pairs(self, pairs, rel_idx):
        pairs = np.asarray(pairs)
        h_idx, t_idx = pairs[:, 0], pairs[:, 1]
        if np.isscalar(rel_idx):
            rel_idx = np.full(len(pairs), rel_idx, dtype=int)
        return self.score(h_idx, rel_idx, t_idx)

    def fit(self, triples, n_epochs=50, batch_size=16384, verbose=True):
        """Train with margin ranking loss and direct SGD updates."""
        triples = np.asarray(triples, dtype=np.int32)
        n = len(triples)
        lr = self.lr

        for epoch in range(n_epochs):
            perm = self.rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = triples[perm[start:start + batch_size]]
                bs = len(batch)
                heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]

                neg_heads = heads.copy()
                neg_tails = tails.copy()
                flip = self.rng.random(bs) < 0.5
                n_h = flip.sum()
                if n_h > 0:
                    neg_heads[flip] = self.rng.randint(0, self.n_entities, n_h)
                if bs - n_h > 0:
                    neg_tails[~flip] = self.rng.randint(
                        0, self.n_entities, bs - n_h)

                # Forward
                h_re = self.ent_re[heads]; h_im = self.ent_im[heads]
                t_re = self.ent_re[tails]; t_im = self.ent_im[tails]
                nh_re = self.ent_re[neg_heads]; nh_im = self.ent_im[neg_heads]
                nt_re = self.ent_re[neg_tails]; nt_im = self.ent_im[neg_tails]
                rp = self.rel_phase[rels]

                d_pos = self._dist(h_re, h_im, rp, t_re, t_im)
                d_neg = self._dist(nh_re, nh_im, rp, nt_re, nt_im)

                violation = self.margin + d_pos - d_neg
                mask = violation > 0
                n_active = mask.sum()
                if n_active == 0:
                    n_batches += 1
                    continue

                epoch_loss += violation[mask].mean()
                n_batches += 1

                # Compute rotation components for active samples
                m = mask
                r_re_m = np.cos(rp[m]); r_im_m = np.sin(rp[m])

                # Positive triple
                hr_re_p = h_re[m]*r_re_m - h_im[m]*r_im_m
                hr_im_p = h_re[m]*r_im_m + h_im[m]*r_re_m
                diff_re_p = hr_re_p - t_re[m]
                diff_im_p = hr_im_p - t_im[m]
                dp = d_pos[m, None] + 1e-12

                # Negative triple
                hr_re_n = nh_re[m]*r_re_m - nh_im[m]*r_im_m
                hr_im_n = nh_re[m]*r_im_m + nh_im[m]*r_re_m
                diff_re_n = hr_re_n - nt_re[m]
                diff_im_n = hr_im_n - nt_im[m]
                dn = d_neg[m, None] + 1e-12

                # Positive gradients -> push d_pos down
                g_h_re = (diff_re_p * r_re_m + diff_im_p * r_im_m) / dp
                g_h_im = (-diff_re_p * r_im_m + diff_im_p * r_re_m) / dp
                g_t_re = -diff_re_p / dp
                g_t_im = -diff_im_p / dp

                self.ent_re[heads[m]] -= lr * g_h_re
                self.ent_im[heads[m]] -= lr * g_h_im
                self.ent_re[tails[m]] -= lr * g_t_re
                self.ent_im[tails[m]] -= lr * g_t_im

                # Negative gradients -> push d_neg up
                g_nh_re = (diff_re_n * r_re_m + diff_im_n * r_im_m) / dn
                g_nh_im = (-diff_re_n * r_im_m + diff_im_n * r_re_m) / dn
                g_nt_re = -diff_re_n / dn
                g_nt_im = -diff_im_n / dn

                self.ent_re[neg_heads[m]] += lr * g_nh_re
                self.ent_im[neg_heads[m]] += lr * g_nh_im
                self.ent_re[neg_tails[m]] += lr * g_nt_re
                self.ent_im[neg_tails[m]] += lr * g_nt_im

                # Relation phase gradient
                g_r_p = (diff_re_p*(-h_re[m]*r_im_m - h_im[m]*r_re_m)
                         + diff_im_p*(h_re[m]*r_re_m - h_im[m]*r_im_m)
                         ) / dp
                g_r_n = (diff_re_n*(-nh_re[m]*r_im_m - nh_im[m]*r_re_m)
                         + diff_im_n*(nh_re[m]*r_re_m - nh_im[m]*r_im_m)
                         ) / dn
                self.rel_phase[rels[m]] -= lr * (g_r_p - g_r_n)

            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f'  Epoch {epoch+1:>4d}/{n_epochs}  loss={avg:.4f}')


# ── Triple preparation ───────────────────────────────────────────────────────

def kg_to_triples(kg_df, add_inverse=True):
    """Convert a BioKGSuite edge DataFrame to integer-indexed triples.

    Parameters
    ----------
    kg_df : pd.DataFrame
        Must have columns: x_index, y_index, relation.
    add_inverse : bool
        If True, add inverse triples (t, r_inv, h) for each (h, r, t).
        This mirrors the undirected-graph assumption of heuristic scorers
        and makes TransE/RotatE direction-agnostic.

    Returns
    -------
    triples : np.ndarray of shape (N, 3) — (head, relation, tail)
    rel_to_idx : dict
    idx_to_rel : dict
    """
    relations = list(kg_df['relation'].unique())
    if add_inverse:
        inv_relations = [f'{r}_inv' for r in relations]
        all_relations = relations + inv_relations
    else:
        all_relations = relations
    rel_to_idx = {r: i for i, r in enumerate(all_relations)}
    idx_to_rel = {i: r for r, i in rel_to_idx.items()}

    h_vals = kg_df['x_index'].values
    r_vals = kg_df['relation'].map(rel_to_idx).values
    t_vals = kg_df['y_index'].values
    fwd = np.column_stack([h_vals, r_vals, t_vals]).astype(np.int32)

    if add_inverse:
        r_inv_vals = kg_df['relation'].map(
            lambda r: rel_to_idx[f'{r}_inv']).values
        rev = np.column_stack([t_vals, r_inv_vals, h_vals]).astype(np.int32)
        triples = np.vstack([fwd, rev])
    else:
        triples = fwd

    return triples, rel_to_idx, idx_to_rel


def build_train_triples(kg_df, test_pairs_set, indication_rels):
    """Build training triples by removing held-out drug-disease test edges.

    Parameters
    ----------
    kg_df : pd.DataFrame
        Full edge table.
    test_pairs_set : set of (int, int)
        Drug-disease pairs held out for testing.
    indication_rels : list of str
        Relation names corresponding to drug-disease indications.

    Returns
    -------
    train_triples : np.ndarray of shape (M, 3)
    rel_to_idx : dict
    idx_to_rel : dict
    """
    is_indication = kg_df['relation'].isin(set(indication_rels))
    x_vals = kg_df['x_index'].values
    y_vals = kg_df['y_index'].values

    test_both = test_pairs_set | {(b, a) for a, b in test_pairs_set}

    ind_idx = np.where(is_indication)[0]
    drop_mask = np.zeros(len(kg_df), dtype=bool)
    if len(ind_idx) > 0:
        pairs_check = list(zip(x_vals[ind_idx].astype(int),
                               y_vals[ind_idx].astype(int)))
        is_test = np.array([p in test_both for p in pairs_check])
        drop_mask[ind_idx[is_test]] = True

    train_df = kg_df[~drop_mask]
    return kg_to_triples(train_df)


def compute_embedding_metrics(model, test_pairs, neg_pairs, rel_idx,
                              rel_idx_inv=None, ks=(10, 50, 100)):
    """Evaluate embedding model on test pairs with negative sampling.

    Scores each pair using both the forward relation and its inverse
    (if provided), taking the maximum score. This handles KGs where
    indication edges may be stored in either direction.

    Parameters
    ----------
    model : TransE or RotatE instance
    test_pairs : list of (head, tail) tuples — positive pairs
    neg_pairs : list of (head, tail) tuples — negative pairs
    rel_idx : int
        Forward relation index for scoring (drug, rel, disease).
    rel_idx_inv : int, optional
        Inverse relation index. If given, also scores (disease, rel_inv, drug)
        and takes element-wise max.
    ks : tuple of int
        Values of K for Hits@K.

    Returns
    -------
    dict with: auroc, auprc, mrr, hits@K for each K
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    all_pairs = np.array(test_pairs + neg_pairs)
    labels = np.array([1] * len(test_pairs) + [0] * len(neg_pairs))

    # Score in forward direction: (drug, rel, disease)
    scores_fwd = model.score_pairs(all_pairs, rel_idx)

    if rel_idx_inv is not None:
        # Also score (drug, rel_inv, disease) — the inverse relation was
        # learned from reversed triples so it captures the other direction.
        scores_inv = model.score_pairs(all_pairs, rel_idx_inv)
        scores = np.maximum(scores_fwd, scores_inv)
    else:
        scores = scores_fwd

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    pos_scores = scores[:len(test_pairs)]
    results = {'auroc': float(auroc), 'auprc': float(auprc)}

    reciprocal_ranks = []
    hits_at_k = {k: [] for k in ks}
    for ps in pos_scores:
        rank = int(np.sum(scores >= ps))
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        for k in ks:
            hits_at_k[k].append(1 if rank <= k else 0)

    results['mrr'] = float(np.mean(reciprocal_ranks))
    for k in ks:
        results[f'hits@{k}'] = float(np.mean(hits_at_k[k]))

    return results


# ── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    'TransE': TransE,
    'RotatE': RotatE,
}
