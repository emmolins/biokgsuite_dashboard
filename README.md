# BioKGSuite

A reproducible benchmark for biomedical knowledge graphs applied to drug repurposing. Six public KGs — **PrimeKG**, **Hetionet**, **DRKG**, **OpenBioLink**, **BioKG**, and **MATRIX** (Every Cure) — are evaluated across **18 metrics spanning seven quality dimensions**: coverage, annotation accuracy, trustworthiness, topology, stability, task performance, and generalisation. Two supplementary notebooks extend the analysis: an embedding-validation notebook (TransE vs. RotatE vs. EmbeddingGemma word-priors, with multi-rerun resampling for stability) and a KG-augmented LLM prompting notebook (five literature-grounded prompting strategies).

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

**Supplementary notebooks** (not part of the 7-dimension aggregate):
- `08_embedding_validation` — compares TransE, RotatE, and the EmbeddingGemma-300m word-priors baseline on drug–disease link prediction; reports stability under multi-rerun resampling (`N_RERUNS=5`). See [`docs/resampling_methodology.md`](docs/resampling_methodology.md).
- `09_llm_integration` — KG-augmented LLM prompting for drug–disease plausibility; evaluates five literature-grounded strategies (2 restricted + 3 unrestricted: `zero_shot_direct`, `structured_json`, `zero_shot_cot`, `multi_expert_rot`, `cisc`). See [`docs/llm_prompting_strategies.md`](docs/llm_prompting_strategies.md).

Notebook `00_benchmark_summary` aggregates the seven main dimensions into the final summary. Run `01` through `07`, then `00`. Notebooks `08` and `09` are independent.

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

Note: MATRIX is large (~5 GB nodes, ~14 GB edges). The loader streams in chunks and filters to the canonical drug/disease/gene/pathway/phenotype subset declared in `config.yaml :: matrix.keep_categories` to stay apples-to-apples with the other KGs.

MATRIX disease nodes are heterogeneously identified (UMLS, OMIM, Orphanet, ICD9, NCIT, MONDO, DOID, MESH, ...). With `disease_id_scheme: mondo`, the loader bridges through three crosswalks in cascade — DOID→MONDO (`do_diseases.csv`), MESH→DOID→MONDO (`mesh_to_doid.csv`), and the broad MONDO SSSOM mapping table for the long-tail UMLS/OMIM/Orphanet/ICD9/NCIT cases. Run `bash scripts/download_mondo_sssom.sh` once to fetch the SSSOM file (~30 MB) into `data/gold_standards/`; the loader degrades gracefully if it isn't present.

Gold-standard references go under `data/gold_standards/` (sources in [Data availability](#data-availability)).

Run all notebooks:

```bash
cd eval_notebooks
for nb in 01_coverage 02_annotation_accuracy 03_trustworthiness \
          04_topology 05_stability 06_task_performance \
          07_generalization 00_benchmark_summary; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "${nb}.ipynb"
done
```

**Notebook 08** reads embedding caches produced by `run_emb_model.py`. To regenerate the full embedding analysis (TransE + RotatE + EmbeddingGemma word-priors + 5-rerun resampling stability):

```bash
# 1. Train TransE / RotatE (the original analysis)
python run_emb_model.py

# 2. (Optional) Generate the EmbeddingGemma word-priors baseline
#    Requires: pip install torch transformers sentence-transformers
#    Requires: export HF_TOKEN=hf_...  (EmbeddingGemma is gated)
bash scripts/run_gemma_benchmark.sh

# 3. Execute the notebook end-to-end (includes the resampling section)
bash scripts/run_resampled_nb08.sh
```

**Notebook 09** requires a local Ollama server with `llama3.1:8b` pulled. To run the prompting-strategy pilot:

```bash
ollama serve &
ollama pull llama3.1:8b
bash scripts/run_prompting_pilot.sh
```

Both notebooks resume from per-row / per-rerun caches if interrupted.

Outputs: figures in `results/figures/` (PDF + PNG), per-notebook checkpoints in `results/checkpoints/`, tabular outputs in `results/tables/`, final summary in `results/benchmark_summary.csv`.

## Repository layout

```
eval_notebooks/          10 Jupyter notebooks (00–07 main, 08–09 supplementary)
src/
  embedding.py           TransE, RotatE, and GemmaNameEmbedder
  prompting_strategies.py  9 LLM prompting strategies for nb09
  loading.py, graph_utils.py, plotting.py, scoring.py, …
run_emb_model.py         Standalone TransE / RotatE / Gemma runner
config.yaml              KG paths and analysis parameters
scripts/
  run_gemma_benchmark.sh   Word-priors baseline for the 5 small KGs (nb08)
  run_gemma_matrix.py      Word-priors for MATRIX (subsampled or full)
  run_resampled_nb08.sh    Multi-rerun stability analysis end-to-end (nb08)
  run_prompting_pilot.sh   Prompting-strategy pilot via Ollama (nb09)
  _inject_*.py             Internal build scripts (regenerate nb08 / nb09 cells from clean upstream)
results/
  benchmark_summary.csv  18 metrics × 6 KGs (headline)
  embedding_comparison.csv             Single-run TransE / RotatE / Gemma (nb08)
  embedding_comparison_resampled.csv   Multi-rerun stability (nb08)
  tables/                Per-notebook data outputs (.csv, .tsv, .md, .json)
  figures/               Per-notebook charts (.pdf, .png)
  checkpoints/           Per-notebook .pkl files consumed by nb00
docs/
  dashboard.html         Interactive dashboard (GitHub Pages)
  llm_prompting_strategies.md      Per-strategy rationale + literature
  llm_prompting_analysis_outline.md  How to read nb09 pilot results
  resampling_methodology.md        Why N=5 reruns instead of bootstrap CIs
  gemma_name_resolution_followup.md  KGs where Gemma needs external name lookups
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
