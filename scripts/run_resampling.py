#!/usr/bin/env python3
"""Standalone resampling run for nb08 — robust, incremental, no notebook.

Why this exists: running the full nb08 headless (papermill/nbconvert) re-does the
single-run + Gemma encode every time and dies on any one fragile cell after hours
(e.g. the Gemma re-score cell's `for kg in KG_NAMES` KeyError on 'matrix'),
losing all the resampling work because the CSV is only written at the very end.

This script does ONLY the resampling, using the FIXED src/embedding.py
(.sum() loss). It writes results/embedding_comparison_resampled.csv after every
KG, so a crash/timeout never costs more than the KG in progress. Gemma is
optional and guarded — it only runs for KGs whose encoding cache already exists,
and per-KG failures are caught, never aborting the run.

Usage (on a GPU node, venv active):
    python scripts/run_resampling.py                  # 3 reruns, no matrix, no gemma
    python scripts/run_resampling.py --reruns 3 --batch 4096
    python scripts/run_resampling.py --kgs biokg,drkg,hetionet,openbilink,primekg
    python scripts/run_resampling.py --gemma          # also re-score Gemma where cached
    python scripts/run_resampling.py --kgs matrix     # add matrix explicitly (slow)
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

BASE = Path(__file__).resolve().parent.parent
import sys
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.loading import find_config, load_config, load_kg
from src.embedding import (TransE, RotatE, GemmaNameEmbedder,
                           build_train_triples, compute_embedding_metrics)
from src.negative_sampling import generate_negatives

CACHE = BASE / "results" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_CSV = BASE / "results" / "embedding_comparison_resampled.csv"

# ── Hyperparameters (match nb08 cell 6) ──────────────────────────────────────
EMB_DIM       = 128
N_EPOCHS      = 100
LR            = 0.01
MARGIN_TRANSE = 1.0
MARGIN_ROTATE = 6.0
SEED          = 42
NEG_RATIO     = 5
STRATEGIES    = ["random", "type-constrained", "shared-target"]
KG_ORDER      = ["primekg", "hetionet", "drkg", "openbilink", "biokg", "matrix"]
GEMMA_DIM     = 768
GEMMA_MODEL   = "google/embeddinggemma-300m"
GEMMA_BATCH   = 64

config = load_config(find_config())


def prepare_kg_resampled(kg_name, rerun_idx, rerun_seeds, neg_ratio=NEG_RATIO):
    """Per-rerun prep: different random 10% held out; cached per (kg, rerun)."""
    rerun_seed = rerun_seeds[rerun_idx]
    cache_path = CACHE / f"{kg_name}_prep_rerun{rerun_idx}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data is not None and "neg_by_strategy" in data:
            return data

    kg_df, nodes_df = load_kg(kg_name, config)
    kg_cfg = config["knowledge_graphs"][kg_name]
    etypes = kg_cfg["entity_types"]
    type_map = dict(zip(nodes_df["idx"], nodes_df["type"]))

    drug_idx    = {i for i, t in type_map.items() if t == etypes.get("Drug", "Drug")}
    disease_idx = {i for i, t in type_map.items() if t == etypes.get("Disease", "Disease")}
    gene_idx    = {i for i, t in type_map.items() if t == etypes.get("Gene/Protein", "Gene")}

    dd = kg_cfg.get("relations", {}).get("drug_disease", {})
    ind_rels = [dd["relation"]] if "relation" in dd else dd.get("relations", [])
    dt = kg_cfg.get("relations", {}).get("drug_target", {})
    dt_rels = [dt["relation"]] if "relation" in dt else dt.get("relations", [])

    mask = kg_df["relation"].isin(ind_rels)
    sub = kg_df.loc[mask, ["x_index", "y_index"]].astype("int64")
    h, t = sub["x_index"].values, sub["y_index"].values
    drug_arr    = np.fromiter(drug_idx, dtype="int64", count=len(drug_idx))
    disease_arr = np.fromiter(disease_idx, dtype="int64", count=len(disease_idx))
    drug_set, disease_set = set(drug_arr.tolist()), set(disease_arr.tolist())
    fwd_mask = np.array([x in drug_set for x in h]) & np.array([x in disease_set for x in t])
    rev_mask = np.array([x in disease_set for x in h]) & np.array([x in drug_set for x in t])
    pairs_set = set()
    pairs_set.update(zip(h[fwd_mask].tolist(), t[fwd_mask].tolist()))
    pairs_set.update(zip(t[rev_mask].tolist(), h[rev_mask].tolist()))
    pairs = list(pairs_set)

    rng = np.random.RandomState(rerun_seed)
    perm = rng.permutation(len(pairs))
    split = int(0.9 * len(pairs))
    test_pos = [pairs[i] for i in perm[split:]]
    all_pos = set(pairs)

    train_triples, rel_to_idx, idx_to_rel = build_train_triples(kg_df, set(test_pos), ind_rels)
    n_ent = int(nodes_df["idx"].max()) + 1
    node_name_map = dict(zip(nodes_df["idx"], nodes_df["name"]))

    drug_targets = {}
    _dt_sub = kg_df.loc[kg_df["relation"].isin(dt_rels), ["x_index", "y_index"]]
    if not _dt_sub.empty:
        _h = _dt_sub["x_index"].astype("int64").to_numpy()
        _t = _dt_sub["y_index"].astype("int64").to_numpy()
        _gene_arr = np.fromiter(gene_idx, dtype="int64", count=len(gene_idx))
        _fwd = np.isin(_h, drug_arr) & np.isin(_t, _gene_arr)
        _rev = np.isin(_t, drug_arr) & np.isin(_h, _gene_arr)
        _keep = _fwd | _rev
        _drugs = np.where(_fwd, _h, _t)[_keep]
        _genes = np.where(_fwd, _t, _h)[_keep]
        if _drugs.size:
            _pairs_df = pd.DataFrame({"drug": _drugs, "gene": _genes})
            drug_targets = {int(d): set(int(x) for x in g)
                            for d, g in _pairs_df.groupby("drug")["gene"]}

    n_neg = len(test_pos) * neg_ratio
    neg_by_strategy = {s: generate_negatives(test_pos, n_neg, s, drug_idx, disease_idx,
                                             drug_targets, node_name_map, all_pos, rng)
                       for s in STRATEGIES}

    rel_idx = rel_to_idx[ind_rels[0]]
    rel_idx_inv = rel_to_idx.get(f"{ind_rels[0]}_inv")
    prep = {"train_triples": train_triples, "rel_to_idx": rel_to_idx,
            "n_ent": n_ent, "n_rels": len(rel_to_idx),
            "test_pos": test_pos, "neg_by_strategy": neg_by_strategy,
            "rel_idx": rel_idx, "rel_idx_inv": rel_idx_inv,
            "n_test": len(test_pos), "rerun_idx": rerun_idx, "rerun_seed": rerun_seed}
    with open(cache_path, "wb") as f:
        pickle.dump(prep, f)
    return prep


def train_one(model_name, prep, seed, epochs, batch):
    Cls = TransE if model_name == "TransE" else RotatE
    if model_name == "RotatE":
        model = Cls(n_entities=prep["n_ent"], n_relations=prep["n_rels"],
                    dim=max(EMB_DIM // 2, 8), margin=MARGIN_ROTATE, lr=LR, seed=seed)
    else:
        model = Cls(n_entities=prep["n_ent"], n_relations=prep["n_rels"],
                    dim=EMB_DIM, margin=MARGIN_TRANSE, lr=LR, seed=seed)
    t0 = time.time()
    for _ in tqdm(range(epochs), leave=False, unit="ep",
                  desc=f"{model_name} seed{seed}"):
        model.fit(prep["train_triples"], n_epochs=1, batch_size=batch, verbose=False)
    return model, time.time() - t0


def eval_rows(model, prep, kg, model_name, rerun_idx, rerun_seed, train_s, rel_inv=True):
    out = []
    inv = prep["rel_idx_inv"] if rel_inv else None
    for strat in STRATEGIES:
        m = compute_embedding_metrics(model, prep["test_pos"],
                                      prep["neg_by_strategy"][strat],
                                      prep["rel_idx"], rel_idx_inv=inv)
        out.append({"kg": kg, "model": model_name, "strategy": strat,
                    "rerun": rerun_idx, "rerun_seed": rerun_seed,
                    "auroc": m["auroc"], "auprc": m["auprc"], "mrr": m.get("mrr"),
                    "hits@10": m.get("hits@10"), "hits@100": m.get("hits@100"),
                    "train_time_s": train_s})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reruns", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096,
                    help="train batch size (bigger=faster on GPU; .sum() loss)")
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--kgs", default=",".join(k for k in KG_ORDER if k != "matrix"),
                    help="comma-separated KGs (matrix excluded by default — slow)")
    ap.add_argument("--gemma", action="store_true",
                    help="also re-score Gemma where an encoding cache exists")
    args = ap.parse_args()

    rerun_seeds = [SEED + 1000 * i for i in range(args.reruns)]
    kgs = [k.strip() for k in args.kgs.split(",") if k.strip()]
    print(f"Resampling: reruns={args.reruns} seeds={rerun_seeds} batch={args.batch} "
          f"epochs={args.epochs}\nKGs: {kgs}  Gemma: {args.gemma}")

    # Resume: load any existing CSV and skip (kg, model, rerun) combos already done,
    # so e.g. `--kgs matrix` appends to the existing rows instead of overwriting them.
    rows, done = [], set()
    if OUT_CSV.exists():
        _prev = pd.read_csv(OUT_CSV)
        rows = _prev.to_dict("records")
        done = {(r["kg"], r["model"], int(r["rerun"])) for r in rows}
        print(f"Resuming — {len(rows)} existing rows "
              f"({len(done)} kg/model/rerun combos) in {OUT_CSV.name}")

    def flush():
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    for kg in kgs:
        print(f"\n=== {kg} ===", flush=True)
        for rerun_idx in range(args.reruns):
            prep = prepare_kg_resampled(kg, rerun_idx, rerun_seeds)
            seed = rerun_seeds[rerun_idx]
            for model_name in ("TransE", "RotatE"):
                if (kg, model_name, rerun_idx) in done:
                    print(f"  {kg}/{model_name}/rerun{rerun_idx}: cached, skip", flush=True)
                    continue
                model, train_s = train_one(model_name, prep, seed, args.epochs, args.batch)
                rs = eval_rows(model, prep, kg, model_name, rerun_idx, seed, train_s)
                rows.extend(rs)
                tc = next(r["auroc"] for r in rs if r["strategy"] == "type-constrained")
                print(f"  {kg}/{model_name}/rerun{rerun_idx}: AUROC(tc)={tc:.4f} "
                      f"({train_s:.0f}s)", flush=True)
                del model

            # Gemma — guarded: only if an encoding cache exists for this KG
            if args.gemma and (kg, "Gemma", rerun_idx) not in done:
                emb_cache = CACHE / f"gemma_emb_{kg}_d{GEMMA_DIM}.npz"
                if emb_cache.exists():
                    try:
                        g = GemmaNameEmbedder(n_entities=prep["n_ent"],
                                              n_relations=prep["n_rels"], dim=GEMMA_DIM,
                                              model_name=GEMMA_MODEL,
                                              batch_size=GEMMA_BATCH, seed=SEED)
                        g.load_embeddings(emb_cache)
                        rows.extend(eval_rows(g, prep, kg, "Gemma", rerun_idx, seed,
                                              0.0, rel_inv=False))
                        del g
                    except Exception as e:
                        print(f"  {kg}/Gemma/rerun{rerun_idx}: skipped ({e})", flush=True)
                else:
                    print(f"  {kg}/Gemma: no encoding cache ({emb_cache.name}); skipping",
                          flush=True)
        flush()  # save after every KG — crash/timeout never loses completed KGs
        print(f"  wrote {OUT_CSV.name} ({len(rows)} rows so far)", flush=True)

    flush()
    print(f"\nDone. {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
