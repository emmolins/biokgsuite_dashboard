"""
Stream all 86.7M Matrix edges and emit predicate-frequency tables for:
- overall predicate distribution
- (entity_type_subject, predicate, entity_type_object) triples for the four
  drug-centric subgraphs we care about (drug-drug, drug-disease, drug-gene)

Reads node categories from data/matrix/nodes.tsv first into a dict keyed by
node ID, then streams data/matrix/edges.tsv. Memory: ~node-count * ~80B.

Usage:  python matrix_predicate_sweep.py
Writes: results/matrix_predicate_audit.json
"""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path("/sessions/ecstatic-great-hawking/mnt/biokgsuite")
NODES = ROOT / "data/matrix/nodes.tsv"
EDGES = ROOT / "data/matrix/edges.tsv"
OUT   = ROOT / "results/matrix_predicate_audit.json"


# Map biolink categories to BioKGSuite canonical types.
CAT_MAP = {
    "biolink:Drug":            "Drug",
    "biolink:SmallMolecule":   "Drug",
    "biolink:MolecularMixture":"Drug",
    "biolink:ChemicalEntity":  "Drug",
    "biolink:Disease":         "Disease",
    "biolink:PhenotypicFeature":"Disease",   # often used interchangeably
    "biolink:Gene":            "Gene/Protein",
    "biolink:Protein":         "Gene/Protein",
    "biolink:Pathway":         "Pathway",
}


def map_category(raw: str) -> str | None:
    return CAT_MAP.get(raw)


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] indexing nodes...", flush=True)
    node_type: dict[str, str] = {}
    with NODES.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(r, 1):
            t = map_category(row["category"])
            if t is not None:
                node_type[row["id"]] = t
            if i % 5_000_000 == 0:
                print(f"  ... {i:,} nodes ({len(node_type):,} typed)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] indexed {len(node_type):,} typed nodes", flush=True)

    preds_overall = Counter()
    triple_counts = Counter()  # (subject_type, predicate, object_type)
    n = 0
    with EDGES.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n += 1
            p = row["predicate"]
            preds_overall[p] += 1
            s_t = node_type.get(row["subject"])
            o_t = node_type.get(row["object"])
            if s_t and o_t:
                triple_counts[(s_t, p, o_t)] += 1
            if n % 10_000_000 == 0:
                print(f"  ... {n:,} edges processed", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] processed {n:,} edges", flush=True)

    # Filter triples to the canonical drug-centric subgraphs.
    drug_drug    = Counter({k[1]: v for k, v in triple_counts.items() if k[0]=="Drug"         and k[2]=="Drug"})
    drug_disease = Counter({k[1]: v for k, v in triple_counts.items() if k[0]=="Drug"         and k[2]=="Disease"})
    disease_drug = Counter({k[1]: v for k, v in triple_counts.items() if k[0]=="Disease"      and k[2]=="Drug"})
    drug_gene    = Counter({k[1]: v for k, v in triple_counts.items() if k[0]=="Drug"         and k[2]=="Gene/Protein"})
    gene_drug    = Counter({k[1]: v for k, v in triple_counts.items() if k[0]=="Gene/Protein" and k[2]=="Drug"})

    payload = {
        "total_edges": n,
        "total_nodes_typed": len(node_type),
        "predicates_overall_top50": preds_overall.most_common(50),
        "drug_drug_top30":    drug_drug.most_common(30),
        "drug_disease_top30": drug_disease.most_common(30),
        "disease_drug_top30": disease_drug.most_common(30),
        "drug_gene_top30":    drug_gene.most_common(30),
        "gene_drug_top30":    gene_drug.most_common(30),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
