# BioKGSuite

A reproducible benchmark for biomedical knowledge graphs applied to drug repurposing. Six public KGs (**PrimeKG**, **Hetionet**, **DRKG**, **OpenBioLink**, **BioKG**, and **MATRIX** from Every Cure) are evaluated across **18 metrics spanning seven quality dimensions**: coverage, annotation accuracy, trustworthiness, topology, stability, task performance, and generalisation. Two supplementary notebooks extend the analysis: an embedding-validation notebook (TransE vs. RotatE vs. an EmbeddingGemma name-prior baseline, with multi-rerun resampling for stability) and a KG-augmented LLM notebook (does knowledge-graph context help an LLM rank the true repurposing candidate higher, and which KG helps most).

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

- `08_embedding_validation` compares TransE, RotatE, and the EmbeddingGemma-300m name-prior baseline on drug-disease link prediction, with stability reported across multiple resampled reruns (`N_RERUNS`, default 3). Three figures: resampled AUROC per KG, lift over the Gemma name prior, and heuristic vs. embedding AUROC.
- `09_llm_integration` poses a realistic repurposing task: for a target disease, the LLM ranks a pool of candidate drugs, with the true post-cutoff drug hidden among distractors drawn from a prospective (time-split) gold standard. Each candidate carries a balanced, query-independent KG dossier (targets, pathways, indications, side-effects) and the disease carries one profile (associated genes, phenotypes); the model must connect them itself (no pre-computed drug→disease bridge). It reports MRR and hits@k per (model, KG) for a no-KG baseline vs. the KG arm, plus reliance fields (whether the model used the KG and over-trusted it), across a slate of local Ollama models and hosted APIs. KG predicates are mapped to canonical slots in `data/kg_slot_maps.yaml`.

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

Note: MATRIX is large (~5 GB nodes, ~14 GB edges). The loader streams in chunks and filters to the canonical drug/disease/gene/pathway/phenotype subset declared in `config.yaml` (`matrix.keep_categories`) to stay apples-to-apples with the other KGs.

MATRIX disease nodes are heterogeneously identified (UMLS, OMIM, Orphanet, ICD9, NCIT, MONDO, DOID, MESH, and more). With `disease_id_scheme: mondo`, the loader bridges through three crosswalks in cascade: DOID to MONDO (`do_diseases.csv`), MESH to DOID to MONDO (`mesh_to_doid.csv`), and the broad MONDO SSSOM mapping table for the long-tail UMLS/OMIM/Orphanet/ICD9/NCIT cases. Run `bash scripts/download_mondo_sssom.sh` once to fetch the SSSOM file (~30 MB) into `data/gold_standards/`; the loader degrades gracefully if it isn't present.

Gold-standard references go under `data/gold_standards/` (sources in [Data availability](#data-availability)).

Run the main benchmark:

```bash
cd eval_notebooks
for nb in 01_coverage 02_annotation_accuracy 03_trustworthiness \
          04_topology 05_stability 06_task_performance \
          07_generalization 00_benchmark_summary; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "${nb}.ipynb"
done
```

**Notebook 08** is idempotent. It loads embedding caches if present and only retrains when they are missing, so a clean run reuses committed results and is skipped where possible. The single-run Gemma metrics are reused from `results/tables/08_embedding_comparison.csv` by default; to re-encode the gated `google/embeddinggemma-300m` model, set `GEMMA_FORCE_REENCODE = True` and `export HF_TOKEN=hf_...`. Helper scripts to regenerate from scratch:

```bash
python scripts/run_emb_model.py TransE   # train one model on one KG (CLI: <kg> <model> [epochs] [dim])
bash scripts/run_gemma_benchmark.sh      # optional Gemma name-prior baseline (needs HF_TOKEN)
bash scripts/run_resampled_nb08.sh       # execute nb08 end-to-end, including resampling
```

**Notebook 09** runs the ranking task across a configurable model slate. It defaults to `MODE = 'mock'` (no Ollama, no KG load) so the whole pipeline — pools, prompt, shuffles, parsing, scoring — runs anywhere as a smoke test. Set `MODE = 'real'` and edit the `MODELS` list to use local Ollama models (pull them first) and/or hosted APIs (each provider reads its key from the matching env var: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`). The loop is idempotent: results are cached to `results/tables/09_llm_runs/09_ranking.csv` and reused unless `FORCE_RERUN = True`.

```bash
# local models (optional — APIs work without these)
ollama serve &
ollama pull llama3.3:70b
jupyter nbconvert --to notebook --execute --inplace eval_notebooks/09_llm_integration.ipynb
```

Both supplementary notebooks resume from per-row and per-rerun caches if interrupted.

Outputs: figures in `results/figures/` (PDF + PNG), per-notebook checkpoints in `results/checkpoints/`, tabular outputs in `results/tables/`, final summary in `results/tables/00_benchmark_summary.csv`.

## Repository layout

```
eval_notebooks/          10 Jupyter notebooks (00 to 07 main, 08 and 09 supplementary)
src/
  embedding.py           TransE, RotatE, and GemmaNameEmbedder
  prompting_strategies.py  LLMPrompt strategy used by the nb09 pilot scripts
  loading.py, graph_utils.py, plotting.py, scoring.py, and more
config.yaml              KG paths and analysis parameters
data/kg_slot_maps.yaml   Semantic slot to relation maps per KG (nb09)
scripts/
  run_emb_model.py         Standalone single-model / single-KG embedding runner
  run_gemma_benchmark.sh   Name-prior baseline for the 5 small KGs (nb08)
  run_gemma_matrix.py      Name-prior for MATRIX (subsampled or full)
  run_resampled_nb08.sh    Multi-rerun stability analysis end-to-end (nb08)
  run_prompting_pilot.sh   nb09 pilot run via Ollama
  pilot_ranking.py         nb09 ranking pilot (KG-quality -> LLM repurposing)
  pilot_packaging.py       KG-dossier builders + crosswalk resolvers used by pilot_ranking.py
  ddi_gap_audit.py         One-off audit behind results/tables/09_ddi_gap_audit.json
  hpc/                     SLURM batch scripts for the HPC runs
results/
  tables/                Per-notebook data outputs (.csv, .tsv, .md, .json), prefixed by notebook number
    00_benchmark_summary.csv              18 metrics x 6 KGs (headline)
    08_embedding_comparison.csv           Single-run TransE / RotatE / Gemma (nb08)
    08_embedding_comparison_resampled.csv Multi-rerun stability (nb08)
  figures/               Per-notebook charts (.pdf, .png)
  checkpoints/           Per-notebook .pkl files consumed by nb00
docs/
  dashboard.html         Interactive dashboard (GitHub Pages)
environment.yml          Conda environment (Python 3.11)
```

Tested on macOS 14 (Apple Silicon) and Ubuntu 22.04. Python 3.10 to 3.12 expected to work.

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
