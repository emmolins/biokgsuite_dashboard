# BioKGSuite

A reproducible evaluation framework for biomedical knowledge graphs, applied to drug repurposing. BioKGSuite benchmarks five public KGs — PrimeKG, Hetionet, DRKG, OpenBioLink, and BioKG — across seven evaluation dimensions and 18 metrics. All analyses are implemented as self-contained Jupyter notebooks that produce every figure and result table in the associated manuscript.

**Interactive dashboard:** https://emmolins.github.io/biokgsuite_dashboard/dashboard.html

## Evaluation dimensions

| Notebook | Dimension | Metrics |
|---|---|---|
| `01_coverage.ipynb` | Coverage | Entity coverage, relation coverage |
| `02_annotation_accuracy.ipynb` | Annotation accuracy | Entity validity, relational consistency |
| `03_trustworthiness.ipynb` | Trustworthiness | Edge traceability, uncertainty quantification |
| `04_topology.ipynb` | Topology | Connectedness, small-world, reachability, community purity |
| `05_stability.ipynb` | Stability | Random dropout, peripheral dropout |
| `06_task_performance.ipynb` | Task performance | Link prediction, neighbourhood retrieval, multi-hop reasoning |
| `07_generalization.ipynb` | Generalisation | Data-sparse, cross-domain, prospective |
| `00_benchmark_summary.ipynb` | Summary | Cross-dimension aggregate of all 18 metrics |

Notebooks 01 – 07 produce per-dimension results; notebook 00 reads their checkpoints to assemble the final summary. Run in order `01 → 07`, then `00`.

## Repository layout

```
biokgsuite/
├── docs/dashboard.html       # Interactive dashboard (served by GitHub Pages)
├── eval_notebooks/           # Eight evaluation notebooks (00–07)
├── src/                      # Shared modules (loaders, scorers, plotting)
├── data/                     # KG files + gold-standard references (gitignored)
├── results/
│   ├── benchmark_summary.csv # 18 metrics × 5 KGs (produced by 00)
│   ├── figures/              # PDF + PNG for every manuscript figure
│   └── checkpoints/          # Per-notebook serialised results (.pkl)
├── config.yaml               # Data paths and analysis parameters
├── environment.yml           # Conda environment
├── pyproject.toml            # Editable install
├── CITATION.cff              # Machine-readable citation
└── LICENSE                   # MIT
```

## Reproducing the benchmark

**1. Clone and install.**

```bash
git clone https://github.com/emmolins/biokgsuite_dashboard.git
cd biokgsuite_dashboard
conda env create -f environment.yml
conda activate biokgsuite
pip install -e .
```

Python 3.11 is tested on macOS 14 (Apple Silicon) and Ubuntu 22.04 (x86-64). Python 3.10 and 3.12 are expected to work.

**2. Download input data.** Place each knowledge graph at the path declared in `config.yaml` (sources and versions in [Data availability](#data-availability)). The six required files are:

```
data/primekg/primekg.csv
data/hetionet/nodes.tsv
data/hetionet/edges.tsv
data/drkg/drkg.tsv
data/openbilink/edges.csv
data/biokg/biokg.links.tsv
```

Gold-standard reference files under `data/gold_standards/` are regenerated from the sources listed in [Data availability](#data-availability). To tune analysis parameters (random seed, negative-sampling ratio, dropout rates, etc.), edit `config.yaml` under `analysis_params`.

**3. Run the notebooks.** Interactively via `jupyter lab` (run in order `01 → 07`, then `00`), or non-interactively:

```bash
cd eval_notebooks
for nb in 01_coverage 02_annotation_accuracy 03_trustworthiness \
          04_topology 05_stability 06_task_performance \
          07_generalization 00_benchmark_summary; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "${nb}.ipynb"
done
```

Figures are written to `results/figures/` as both `.pdf` (for the manuscript) and `.png` (for inspection). Per-notebook intermediate results are serialised to `results/checkpoints/` as `.pkl` files. The final cross-dimension summary is `results/benchmark_summary.csv`.

Notebook 02 validates entity identifiers against external APIs (NCBI, EBI QuickGO, EBI OLS4, DrugBank, Reactome, PubChem). API responses are cached on first run; subsequent runs are offline.

## Data availability

| Dataset | Source | Version | Access |
|---|---|---|---|
| PrimeKG | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) | v2.0, 2023-01 | Open |
| Hetionet | [GitHub](https://github.com/hetio/hetionet) | v1.0, 2017 | Open |
| DRKG | [GitHub](https://github.com/gnn4dr/DRKG) | 2020-09 | Open |
| OpenBioLink | [GitHub](https://github.com/openbiolink/openbiolink) | HQ, 2020 | Open |
| BioKG | [GitHub](https://github.com/dsi-bdi/biokg) | 2021 | Open |
| DrugBank | [DrugBank](https://go.drugbank.com/releases/latest) | v5.1.12 | Free academic account |
| UniProt (human) | [UniProt](https://www.uniprot.org/proteomes/UP000005640) | 2024-06 | Open |
| Disease Ontology | [Disease Ontology](https://disease-ontology.org/) | 2024-04 | Open |
| Reactome | [Reactome](https://reactome.org/download-data) | v88 | Open |
| Open Targets | [Open Targets](https://platform.opentargets.org/downloads) | 24.06 | Open |
| CTD | [CTD](http://ctdbase.org/) | 2024 | Open |

## Citing

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

Machine-readable metadata in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
