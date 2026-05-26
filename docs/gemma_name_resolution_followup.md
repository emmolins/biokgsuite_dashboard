# Follow-up: name resolution for DRKG, OpenBioLink, BioKG

For the EmbeddingGemma word-priors experiment (notebook 08) to be
meaningful on a given KG, `nodes_df['name']` must contain human-readable
strings — e.g. `"Aspirin"`, `"Type 2 diabetes mellitus"`, `"COX1"` —
not opaque identifiers.

## Current state per KG

Audited 2026-05-26 against `src/loading.py` at HEAD.

| KG | name source | sample names | usable for Gemma? |
|---|---|---|---|
| Hetionet | `nodes.tsv` `name` column | `"uterine cervix"`, `"nose"`, `"islet of Langerhans"` | ✓ |
| PrimeKG  | `x_name` / `y_name` columns | `"PHYHIP"`, `"KIF15"`, `"Aspirin"` | ✓ |
| MATRIX   | nodes file `name` column (heterogeneous) | mostly real names; some `MONDO:...` | ✓ (partial) |
| DRKG     | suffix of `Type::Identifier` | `"2157"` for `Gene::2157`, `"DB00001"` for `Compound::DB00001` | ✗ |
| OpenBioLink | suffix of `Prefix:Identifier` | `"0000001"` for `CL:0000001`, `"DB00001"` for `PUBCHEM.COMPOUND:...` | ✗ |
| BioKG    | metadata files don't carry display names | `"DB00001"`, `"D000006"`, `"P12345"` | ✗ |

The three ✗ KGs need external name resolution. The lookups below produce
a `(id → name)` dict per entity type; the loader can then populate
`nodes_df['name']` from it.

## What's needed per entity type

### Genes / Proteins

**Source (preferred):** NCBI Gene Info table
`ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/All_Data.gene_info.gz`
(~600 MB compressed). Columns of interest: `GeneID`, `Symbol`,
`description`, `Synonyms`.

**Alt (for proteins indexed by UniProt accession — BioKG):** UniProt
ID-mapping `idmapping.dat.gz` from
`https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/`
to get protein names. The `uniprot_sprot.dat.gz` file has the full
canonical name (`DE Recommended Name`) but is ~3 GB. For BioKG you really
only need the ~20K human-relevant proteins; the API
`https://www.uniprot.org/uniprot/{accession}.json` works for one-off
lookups.

**Effort:** ~1 day. Download once, build `gene_id → symbol` dict, apply
in the loader.

### Drugs / Compounds

**Source (DrugBank IDs, e.g. BioKG, DRKG):** DrugBank `vocabulary.csv`
(part of the DrugBank Open Data distribution at
`https://go.drugbank.com/releases/latest`). Columns: `DrugBank ID`,
`Common name`, `Synonyms`. Free for academic use, registration required.

**Source (PubChem CIDs, e.g. OpenBioLink):** PubChem CID-to-name mapping.
`https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Title.gz`
(~250 MB). Plain text: `CID<TAB>Title`.

**Effort:** ~1 day per source. Both are flat files with id→name; trivial
to integrate.

### Diseases

**Source (MeSH IDs, e.g. BioKG `D000006`):** MeSH descriptor file
`https://www.nlm.nih.gov/databases/download/mesh.html` — XML, ~300 MB.
Need to parse `<DescriptorRecord>` blocks for the `DescriptorUI`
(`D000006`) and `DescriptorName` (`"Abdomen"`). The
`pymesh-parser` package handles this in a few lines.

**Source (DOID, e.g. OpenBioLink, Hetionet):** DOID OBO file from
`http://purl.obolibrary.org/obo/doid.obo`. Already used in this repo
(`data/gold_standards/do_diseases.csv`) so the parser exists.

**Source (MONDO, e.g. MATRIX):** MONDO SSSOM already integrated (see
`scripts/download_mondo_sssom.sh`). MONDO OBO at
`http://purl.obolibrary.org/obo/mondo.obo` gives the canonical names.

**Effort:** ~0.5–1 day for MeSH parsing; DOID/MONDO are essentially free
since the parsers already exist.

### Pathways

**Source (Reactome `R-HSA-*`):** Reactome name dump
`https://reactome.org/download/current/ReactomePathways.txt` — flat
`stable_id<TAB>name<TAB>species`. Already used by
`05_stability.ipynb` for community labels.

**Source (KEGG `hsa*`):** KEGG REST `https://rest.kegg.jp/list/pathway/hsa`
returns flat `id<TAB>name`. Rate-limited but usable for one-off builds.

**Effort:** ~0.5 day.

### Other types (Cell, Anatomy, Phenotype, etc.)

Mostly not used as drug or disease nodes in the test pairs, so name
quality there doesn't affect the indication-prediction AUROC. Skip
unless extending the experiment beyond drug-disease.

## Suggested implementation order

1. **Reactome pathways** (~0.5 day) — easiest, file already on disk
   for some KGs, parser exists.
2. **DOID + MONDO diseases** (~0.5 day) — repo already has tooling.
3. **PubChem CID → Title** (~0.5 day) — unlocks OpenBioLink drugs.
4. **DrugBank vocabulary** (~0.5 day) — unlocks BioKG and DRKG drugs.
5. **NCBI Gene Info** (~1 day) — unlocks all three KGs' gene nodes.
6. **MeSH descriptors** (~1 day) — unlocks BioKG diseases.

Total estimate: **~3–4 days of focused work** to bring DRKG, OpenBioLink,
and BioKG to the same name-quality baseline as Hetionet and PrimeKG.

## Where to wire it in

In `src/loading.py`, each of `load_drkg`, `load_openbilink`, `load_biokg`
currently has a `_name_from_id` (or equivalent) function that returns the
identifier suffix. Replace with a lookup against the resolved-name dicts
built from the sources above. The rest of the codebase (which only reads
`nodes_df['name']`) needs no changes.

Once resolution is in place, re-run `scripts/run_gemma_benchmark.sh` —
the embedding cache files (`results/cache/gemma_emb_*.npz`) include the
KG name and dim in their filename, so old caches naturally won't be
reused after the loader change.

## Expected impact on the slide-3 chart

Currently the chart shows Gemma at ~random AUROC for DRKG, OpenBioLink,
BioKG, marked with `†`. After name resolution, those three bars should
move up into the same range as Hetionet and PrimeKG — turning a "2 of 6"
result into a defensible "5 of 6 small KGs, plus MATRIX" claim with no
asterisks.
