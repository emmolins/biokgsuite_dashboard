# BioKGSuite

A reproducible benchmark for biomedical knowledge graphs applied to drug repurposing. Five public KGs — **PrimeKG**, **Hetionet**, **DRKG**, **OpenBioLink**, and **BioKG** — are evaluated across 18 metrics spanning eight dimensions: coverage, annotation accuracy, trustworthiness, topology, stability, task performance, generalisation, and embedding model comparison.

[**Interactive dashboard**](https://emmolins.github.io/biokgsuite_dashboard/dashboard.html)

## Evaluation dimensions

| # | Notebook | Dimension | Metrics |
|---|---|---|---|
| 01 | `01_coverage` | Coverage | Entity coverage, relation coverage |
| 02 | `02_annotation_accuracy` | Annotation accuracy | Entity validity, relational consistency |
| 03 | `03_trustworthiness` | Trustworthiness | Edge traceability, uncertainty quantification |
| 04 | `04_topology` | Topology | Connectedness, small-world, reachability, community purity |
| 05 | `05_stability` | Stability | Random dropout, peripheral dropout |
| 06 | `06_task_performance` | Task performance | Link prediction, neighbourhood retrieval, multi-hop reasoning |
| 07 | `07_generalization` | Generalisation | Data-sparse, cross-domain, prospective |
| 08 | `08_embedding_validation` | Embedding models | TransE vs. RotatE comparison |

Notebook `00_benchmark_summary` aggregates all checkpoints into the final summary. Run `01` through `08`, then `00`.

## Quick start

```bash
git clone https://github.com/emmolins/biokgsuite_dashboard.git
cd biokgsuite_dashboard
conda env create -f environment.yml
conda activate biokgsuite
pip install -e .
```

Download each KG to the path declared in `config.yaml`:

```
data/primekg/primekg.csv        data/drkg/drkg.tsv
data/hetionet/nodes.tsv         data/openbilink/edges.csv
data/hetionet/edges.tsv         data/biokg/biokg.links.tsv
```

Gold-standard references go under `data/gold_standards/` (sources in [Data availability](#data-availability)).

Run all notebooks:

```bash
cd eval_notebooks
for nb in 01_coverage 02_annotation_accuracy 03_trustworthiness \
          04_topology 05_stability 06_task_performance \
          07_generalization 08_embedding_validation \
          00_benchmark_summary; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "${nb}.ipynb"
done
```

Notebook 08 reads embedding caches produced by `run_emb_model.py`. To regenerate: `python run_emb_model.py`.

Outputs: figures in `results/figures/` (PDF + PNG), per-notebook checkpoints in `results/checkpoints/`, final table in `results/benchmark_summary.csv`.

## Repository layout

```
eval_notebooks/          9 Jupyter notebooks (00–08)
src/                     Shared modules: loaders, embeddings, evaluation, plotting
run_emb_model.py         Standalone TransE/RotatE runner
config.yaml              KG paths and analysis parameters
results/
  benchmark_summary.csv  18 metrics x 5 KGs
  embedding_comparison.csv
  relation_conflicts.csv
  *.md                   Gold-standard vetting and replication reports
docs/dashboard.html      Interactive dashboard (GitHub Pages)
environment.yml          Conda environment (Python 3.11)
```

Tested on macOS 14 (Apple Silicon) and Ubuntu 22.04. Python 3.10–3.12 expected to work.

## Data availability

| Dataset | Source | Version |
|---|---|---|
| PrimeKG | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) | v2.0, 2023-01 |
| Hetionet | [GitHub](https://github.com/hetio/hetionet) | v1.0, 2017 |
| DRKG | [GitHub](https://github.com/gnn4dr/DRKG) | 2020-09 |
| OpenBioLink | [GitHub](https://github.com/openbiolink/openbiolink) | HQ, 2020 |
| BioKG | [GitHub](https://github.com/dsi-bdi/biokg) | 2021 |
| DrugBank | [DrugBank](https://go.drugbank.com/releases/latest) | v5.1.12 |
| UniProt | [UniProt](https://www.uniprot.org/proteomes/UP000005640) | 2024-06 |
| Disease Ontology | [DO](https://disease-ontology.org/) | 2024-04 |
| Reactome | [Reactome](https://reactome.org/download-data) | v88 |
| Open Targets | [Open Targets](https://platform.opentargets.org/downloads) | 24.06 |
| CTD | [CTD](http://ctdbase.org/) | 2024 |

All datasets are open access except DrugBank (free academic account required).

## Citation

```bibtex
@software{molins_biokgsuite_2026,
  author  = {Molins, Emily},
  title   = {{BioKGSuite}: A systematic evaluation framework for biomedical
             knowledge graphs for drug-repurposing applications},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/emmolins/biokgsuite_dashboard},
  license = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

[MIT](LICENSE)
