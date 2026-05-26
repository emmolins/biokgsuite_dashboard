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
from pathlib import Path
import os


# ── Helpers ──────────────────────────────────────────────────────────────────

def _l2_normalize_rows(x, eps=1e-12):
    """L2-normalise each row of x in-place."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms
    return x


def _save_ckpt_atomic(path, **arrays):
    """Atomic-rename save so a crash mid-write doesn't corrupt the checkpoint.

    Note: np.savez auto-appends '.npz' to a *path string*; pass a binary
    file object instead so the temp filename stays exactly what we set.
    """
    path = Path(path)
    tmp  = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)           # uncompressed for speed; ~5 GB/30s on M4
    os.replace(tmp, path)               # atomic on POSIX


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

    def fit(self, triples, n_epochs=50, batch_size=16384, verbose=True,
            checkpoint_path=None, checkpoint_every=10):
        """Train with margin ranking loss and direct SGD updates.

        If ``checkpoint_path`` is set, the embedding state is saved after
        every ``checkpoint_every`` epochs. If a matching checkpoint already
        exists at that path with the same shapes, training resumes from the
        epoch recorded there. A crash + restart therefore re-uses prior
        progress instead of starting over from epoch 0.
        """
        triples = np.asarray(triples, dtype=np.int32)
        n = len(triples)
        lr = self.lr

        # Resume from checkpoint if compatible
        start_epoch = 0
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                ckpt = np.load(checkpoint_path)
                if (ckpt["ent_emb"].shape == self.ent_emb.shape
                    and ckpt["rel_emb"].shape == self.rel_emb.shape):
                    self.ent_emb = ckpt["ent_emb"].astype(np.float32, copy=True)
                    self.rel_emb = ckpt["rel_emb"].astype(np.float32, copy=True)
                    start_epoch = int(ckpt["epoch_completed"])
                    if verbose:
                        print(f"  Resumed from checkpoint at epoch {start_epoch}/{n_epochs}")
                else:
                    if verbose:
                        print("  Checkpoint shape mismatch — starting fresh")
            except Exception as e:
                if verbose:
                    print(f"  Checkpoint load failed ({e}) — starting fresh")

        for epoch in range(start_epoch, n_epochs):
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

            # Periodic checkpoint
            if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
                _save_ckpt_atomic(checkpoint_path,
                                  ent_emb=self.ent_emb,
                                  rel_emb=self.rel_emb,
                                  epoch_completed=np.int32(epoch + 1))


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

    def fit(self, triples, n_epochs=50, batch_size=16384, verbose=True,
            checkpoint_path=None, checkpoint_every=10):
        """Train with margin ranking loss and direct SGD updates.

        If ``checkpoint_path`` is set, the complex embedding state
        (ent_re, ent_im, rel_phase) is saved every ``checkpoint_every``
        epochs and auto-resumed on a matching restart.
        """
        triples = np.asarray(triples, dtype=np.int32)
        n = len(triples)
        lr = self.lr

        # Resume from checkpoint if compatible
        start_epoch = 0
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                ckpt = np.load(checkpoint_path)
                if (ckpt["ent_re"].shape == self.ent_re.shape
                    and ckpt["ent_im"].shape == self.ent_im.shape
                    and ckpt["rel_phase"].shape == self.rel_phase.shape):
                    self.ent_re    = ckpt["ent_re"].astype(np.float32, copy=True)
                    self.ent_im    = ckpt["ent_im"].astype(np.float32, copy=True)
                    self.rel_phase = ckpt["rel_phase"].astype(np.float32, copy=True)
                    start_epoch = int(ckpt["epoch_completed"])
                    if verbose:
                        print(f"  Resumed from checkpoint at epoch {start_epoch}/{n_epochs}")
                else:
                    if verbose:
                        print("  Checkpoint shape mismatch — starting fresh")
            except Exception as e:
                if verbose:
                    print(f"  Checkpoint load failed ({e}) — starting fresh")

        for epoch in range(start_epoch, n_epochs):
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

            # Periodic checkpoint
            if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
                _save_ckpt_atomic(checkpoint_path,
                                  ent_re=self.ent_re,
                                  ent_im=self.ent_im,
                                  rel_phase=self.rel_phase,
                                  epoch_completed=np.int32(epoch + 1))


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

    # Expose per-pair (y_true, y_score) so downstream bootstrap CIs can
    # resample test pairs without retraining the embedding model.
    results['scores'] = scores.astype(float).tolist()
    results['labels'] = labels.astype(int).tolist()

    return results


# ── Text-embedding baseline: EmbeddingGemma (word-priors only) ───────────────

class GemmaNameEmbedder:
    """Score drug-disease pairs using ONLY a text encoder's prior over entity names.

    This is intentionally NOT a KG embedding method. It exists to answer a
    specific question: "how much of the drug-disease indication signal is
    already latent in a pretrained language model's word priors, with zero
    knowledge of the graph?"

    The model embeds each entity's bare name (no type prefix, no task prefix,
    no neighborhood context) and scores a (drug, disease) pair by cosine
    similarity of their two embeddings. There are no relation parameters.

    Designed to plug into the same eval harness as TransE/RotatE — exposes
    `score_pairs(pairs, rel_idx)` (rel_idx is ignored) and a `name` attr,
    so ``compute_embedding_metrics`` works unchanged.

    Parameters
    ----------
    n_entities : int
        Size of the entity space (so we can allocate the embedding matrix).
    n_relations : int
        Accepted but ignored — keeps the constructor signature symmetric
        with TransE/RotatE.
    dim : int
        Matryoshka-truncated embedding dimension (EmbeddingGemma supports
        128, 256, 512, 768). Lower = less RAM, slightly lower fidelity.
    model_name : str
        HuggingFace model id. Default 'google/embeddinggemma-300m'.
    batch_size : int
        Encoding batch size on CPU/GPU.
    seed : int
        Accepted for interface parity; the encoder itself is deterministic.
    """

    name = 'EmbeddingGemma-300m'

    def __init__(self, n_entities, n_relations=0, dim=768,
                 model_name='google/embeddinggemma-300m',
                 batch_size=64, seed=42, token=None):
        """
        token : str or None
            HuggingFace token for accessing gated models. If None, the
            sentence-transformers library falls back to env vars
            ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` or the cached
            credential at ``~/.cache/huggingface/token``.
        """
        self.n_entities = n_entities
        self.n_relations = n_relations  # ignored
        self.dim = dim
        self.model_name = model_name
        self.batch_size = batch_size
        self.seed = seed
        self.token = token
        self._encoder = None
        # Lazy-allocate; encode_entities fills this in
        self.ent_emb = None  # (n_entities, dim) float32, L2-normalized

    def _load_encoder(self):
        if self._encoder is not None:
            return
        import os
        from sentence_transformers import SentenceTransformer
        import torch
        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if getattr(torch.backends, 'mps', None)
                                  and torch.backends.mps.is_available()
                  else 'cpu')
        # truncate_dim uses Matryoshka representation if dim < 768
        kwargs = {}
        if self.dim < 768:
            kwargs['truncate_dim'] = self.dim
        # Resolve token: explicit arg → env var → cached credential (default)
        token = (self.token
                 or os.environ.get('HF_TOKEN')
                 or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
        if token is not None:
            kwargs['token'] = token
        self._encoder = SentenceTransformer(self.model_name, device=device, **kwargs)
        # Set static seed for any internal stochasticity (none expected, but safe)
        try:
            torch.manual_seed(self.seed)
        except Exception:
            pass

    def encode_entities(self, names, verbose=True):
        """Encode an ordered list of names into self.ent_emb.

        Parameters
        ----------
        names : list of str
            Length must equal n_entities. Order corresponds to entity idx.
            Empty / NaN names are replaced with the literal "" to keep
            indices aligned (those entities will get the model's default
            "empty string" embedding — distinguishable from missing).

        Returns
        -------
        self.ent_emb : np.ndarray of shape (n_entities, dim), L2-normalized.
        """
        assert len(names) == self.n_entities, (
            f"got {len(names)} names but n_entities={self.n_entities}")
        self._load_encoder()
        # Normalize input — never pass NaN/None to the encoder
        clean = [('' if (n is None or (isinstance(n, float) and np.isnan(n)))
                  else str(n)) for n in names]
        if verbose:
            print(f'  Encoding {len(clean)} names with {self.model_name} '
                  f'(dim={self.dim}, batch={self.batch_size})...', flush=True)
        emb = self._encoder.encode(
            clean,
            batch_size=self.batch_size,
            show_progress_bar=verbose,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine = dot product
        ).astype(np.float32)
        self.ent_emb = emb
        return self.ent_emb

    def score(self, h_idx, r_idx, t_idx):
        """Higher = more plausible. Ignores r_idx (no relation concept)."""
        if self.ent_emb is None:
            raise RuntimeError("Call encode_entities(names) before scoring.")
        h = self.ent_emb[h_idx]
        t = self.ent_emb[t_idx]
        # Embeddings are L2-normalized; cosine sim = elementwise dot.
        return (h * t).sum(axis=-1)

    def score_pairs(self, pairs, rel_idx):
        """Score a list of (head, tail) pairs. rel_idx is accepted but ignored."""
        pairs = np.asarray(pairs)
        return self.score(pairs[:, 0], None, pairs[:, 1])

    def fit(self, *args, **kwargs):
        """No-op: there's nothing to train. encode_entities() does the work."""
        return self

    # ── Cache helpers ──────────────────────────────────────────────────────
    def save_embeddings(self, path):
        """Save self.ent_emb to disk (atomic)."""
        if self.ent_emb is None:
            raise RuntimeError("Nothing to save — encode_entities first.")
        _save_ckpt_atomic(path, ent_emb=self.ent_emb,
                          dim=np.int32(self.dim),
                          n_entities=np.int32(self.n_entities))

    def load_embeddings(self, path):
        """Load self.ent_emb from a prior save_embeddings() call."""
        ckpt = np.load(path)
        if int(ckpt['n_entities']) != self.n_entities:
            raise ValueError(
                f"cache n_entities={int(ckpt['n_entities'])} != "
                f"current n_entities={self.n_entities}")
        if int(ckpt['dim']) != self.dim:
            raise ValueError(
                f"cache dim={int(ckpt['dim'])} != current dim={self.dim}")
        self.ent_emb = ckpt['ent_emb'].astype(np.float32, copy=True)
        return self.ent_emb


# ── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    'TransE': TransE,
    'RotatE': RotatE,
    'Gemma':  GemmaNameEmbedder,
}
