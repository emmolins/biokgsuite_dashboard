# BioKGSuite

A reproducible benchmark for biomedical knowledge graphs applied to drug repurposing. Six public KGs — **PrimeKG**, **Hetionet**, **DRKG**, **OpenBioLink**, **BioKG**, and **MATRIX** (Every Cure) — are scored across **18 metrics in seven quality dimensions**: coverage, annotation accuracy, trustworthiness, topology, stability, task performance, and generalisation. A supplementary notebook validates KG embeddings (TransE vs. RotatE vs. an EmbeddingGemma name-prior baseline, with multi-rerun resampling).

[**Interactive dashboard**](https://emmolins.github.io/biokgsuite_dashboard/dashboard.html) - Last updated: May 2026.

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

`00_benchmark_summary` aggregates the seven dimensions into the headline summary — run `01`–`07`, then `00`. `08_embedding_validation` is an independent supplement comparing TransE, RotatE, and the EmbeddingGemma-300m name-prior baseline on drug–disease link prediction, with stability across resampled reruns (`N_RERUNS`, default 3).

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
data/matrix/nodes.tsv           data/matrix/edges.tsv
```

MATRIX is large (~5 GB nodes, ~14 GB edges); the loader streams it in chunks and filters to the canonical drug/disease/gene/pathway/phenotype subset (`matrix.keep_categories`). Its disease nodes use mixed schemes (UMLS, OMIM, Orphanet, MONDO, DOID, MESH, …); with `disease_id_scheme: mondo` the loader bridges them via DO/MESH crosswalks and the MONDO SSSOM table — run `bash scripts/download_mondo_sssom.sh` once to fetch the SSSOM file (~30 MB) into `data/gold_standards/` (optional; the loader degrades gracefully without it). Gold-standard references live in `data/gold_standards/` (sources below).

Run the benchmark:

```bash
cd eval_notebooks
for nb in 01_coverage 02_annotation_accuracy 03_trustworthiness \
          04_topology 05_stability 06_task_performance \
          07_generalization 00_benchmark_summary; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "${nb}.ipynb"
done
```

**Notebook 08** is idempotent: it reuses committed embedding caches and the committed Gemma metrics (`results/tables/08_embedding_comparison.csv`) by default. To re-encode the gated `google/embeddinggemma-300m`, set `GEMMA_FORCE_REENCODE = True` and `export HF_TOKEN=hf_...`. To regenerate from scratch:

```bash
python scripts/run_emb_model.py TransE   # one model, one KG (CLI: <kg> <model> [epochs] [dim])
bash scripts/run_gemma_benchmark.sh      # Gemma name-prior baseline (needs HF_TOKEN)
bash scripts/run_resampled_nb08.sh       # nb08 end-to-end, including resampling
```

Outputs: figures in `results/figures/` (PDF+PNG), checkpoints in `results/checkpoints/`, tables in `results/tables/`, headline summary in `results/tables/00_benchmark_summary.csv`.

## Repository layout

```
eval_notebooks/   00–07 (main) + 08 (supplementary)
src/              embedding.py (TransE/RotatE/GemmaNameEmbedder), loading.py,
                  graph_utils.py, plotting.py, scoring.py, and more
config.yaml       KG paths and analysis parameters
scripts/          embedding runners (run_emb_model.py, run_gemma_*.{sh,py},
                  run_resampled_nb08.sh), ddi_gap_audit.py, hpc/ (SLURM)
results/          tables/ and figures/, prefixed by notebook number;
                  checkpoints/ consumed by nb00
docs/             dashboard.html (GitHub Pages)
environment.yml   Conda environment (Python 3.11)
```

Tested on macOS 14 (Apple Silicon) and Ubuntu 22.04; Python 3.10–3.12.

## Data availability

| Dataset | Source | Version |
|---|---|---|
| PrimeKG | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) | v2.0, 2023-01 |
| Hetionet | [GitHub](https://github.com/hetio/hetionet) | v1.0, 2017 |
| DRKG | [GitHub](https://github.com/gnn4dr/DRKG) | 2020-09 |
| OpenBioLink | [GitHub](https://github.com/openbiolink/openbiolink) | HQ, 2020 |
| BioKG | [GitHub](https://github.com/dsi-bdi/biokg) | 2021 |
| MATRIX | [Every Cure / Hugging Face](https://huggingface.co/datasets/everycure/matrix-kg) | 2025 |
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
  title   = {{BioKGSuite}: A Multidimensional Framework for the Systematic Evaluation of Biomedical Knowledge Graphs},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/emmolins/biokgsuite_dashboard},
  license = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

[MIT](LICENSE)
