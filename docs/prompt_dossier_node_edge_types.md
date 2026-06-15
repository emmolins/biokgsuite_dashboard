# What gets pulled into the LLM prompt, per KG

How the drug/disease dossiers are built (`make_kg_block_fn`, notebook 09), and exactly which node and edge types each evaluable KG can contribute. Counts are total edges of that type in the graph (before the per-query `cap` and most-specific-first selection).

## Selection policy (same for every KG)

Edges are kept by the **entity type of the neighbour**, not by relation name. From a node, only neighbours of type **gene**, **pathway**, or **disease** can render, via a fixed phrase:

| Subject | Neighbour | Rendered phrase |
|---|---|---|
| Drug | Gene/Protein | `targets` |
| Drug | Pathway | `modulates pathway` |
| Disease | Gene/Protein | `is associated with gene` |
| Disease | Pathway | `is associated with pathway` |
| Disease | Disease | `is related to` |

Everything else is dropped, by rule:
- **Drug ↔ Disease edges are never rendered** (structural leakage block — this is the answer being hidden).
- **Drug ↔ Drug edges** have no phrase → dropped.
- **ADME gene edges** (enzyme / carrier / transporter) are dropped via `PK_EXCLUDE` so "targets" means mechanistic targets only.
- **Unmapped numeric gene IDs** (no HGNC symbol) are dropped as unreadable.
- **Any neighbour type not mapped to gene/pathway/disease** (phenotype, anatomy, side-effect, ATC, exposure, genetic-disorder) is dropped — these exist in the graphs but never reach the prompt.
- Per phrase, neighbours are deduped and sorted **most-specific first** (lowest degree), then truncated to `cap` (=8).

## Per-KG breakdown

### primekg  — nodes in prompt: Drug, Gene/Protein, Disease
- **Renders:** `targets` (drug→gene, 51,306 · `drug_protein`); `is associated with gene` (disease→gene, 160,822 · `disease_protein`); `is related to` (disease→disease, 128,776).
- **No pathway edges render** (drug/disease have no direct pathway neighbours).
- **Excluded — leakage:** drug↔disease 85,262 (`contraindication` 61,350, `indication` 18,776, `off-label use` 5,136). **Other excluded:** drug↔drug 5,345,256.
- **Present but never shown:** disease→effect/phenotype 303,020; drug→effect/phenotype 129,568; disease→exposure 4,608.

### drkg  — nodes in prompt: Compound(drug), Gene, Disease
- **Renders:** `targets` (drug→gene, 205,151 rendered; +5,643 ADME excluded; sources: GNBR, bioarx, Hetionet, DrugBank); `is associated with gene` (disease→gene, 123,837); `is related to` (disease→disease, **1,086** — negligible).
- **No pathway edges render.**
- **Excluded — leakage:** drug↔disease 83,895 (`GNBR::T` 54,020, `GNBR::Sa` 16,923, `DRUGBANK::treats` 4,968). **Other:** drug↔drug 2,771,514 (mostly DDI).
- **Present but never shown:** drug→Side-Effect 138,944; drug→ATC 15,750; drug→Pharmacologic-Class 1,029; disease→Anatomy 3,602; disease→Symptom 3,357.

### biokg  — nodes in prompt: Drug, Gene/Protein, Disease, **Pathway**
- **Richest bridge — the only KG that renders pathways.** `targets` (drug→gene, 43,264 rendered; +8,921 ADME excluded); `modulates pathway` (drug→pathway, 5,114); `is associated with gene` (disease→gene, 109,276); `is associated with pathway` (disease→pathway, 3,544).
- **No disease→disease.**
- **Excluded — leakage:** drug↔disease 66,867 (`DRUG_DISEASE_ASSOCIATION`). **Other:** drug↔drug 2,668,170 (DDI).
- **Present but never shown:** disease→GeneticDisorder 4,882.

### openbilink  — nodes in prompt: Drug, Gene, Disease
- **Renders:** `targets` (drug→gene, 456,423 — largest; GENE_DRUG, DRUG_BINDING_GENE, …); `is associated with gene` (disease→gene, 93,885); `is related to` (disease→disease, 23,934 — but **all `IS_A`**, i.e. pure taxonomy, not biology).
- **No pathway edges render.**
- **Excluded — leakage:** drug↔disease 7,164 (`DIS_DRUG`).
- **Present but never shown:** drug→Phenotype 89,096; disease→Phenotype 49,556.

## Cross-KG comparison

| Renderable relation | primekg | drkg | biokg | openbilink |
|---|---|---|---|---|
| drug → gene (`targets`) | ✅ 51k | ✅ 205k | ✅ 43k | ✅ 456k |
| disease → gene (`associated`) | ✅ 161k | ✅ 124k | ✅ 109k | ✅ 94k |
| drug → pathway (`modulates`) | — | — | ✅ 5k | — |
| disease → pathway (`associated`) | — | — | ✅ 4k | — |
| disease → disease (`related`) | ✅ 129k | ✅ 1k | — | ✅ 24k (IS_A only) |

## What this means for the experiment

1. **The promised "drug → target → pathway → disease" chain is only fully expressible in biokg** — it is the only KG where pathways connect both sides. In primekg, drkg, and openbilink the bridge is **gene-level only**: drug→gene and disease→gene share the gene vocabulary, but pathways never appear. The prompt header should probably say so, or be softened, since for 3 of 4 KGs it tells the model to expect pathways that are never present.
2. **disease→disease content varies in usefulness:** primekg's is genuine disease relations; openbilink's is pure `IS_A` taxonomy; drkg's is negligible (1,086); biokg has none. This affects how much the disease profile actually helps.
3. **A lot of signal is structurally discarded** — phenotype/effect edges especially (primekg 303k, openbilink 49k+89k). Worth a deliberate decision: is phenotype a useful bridge worth adding to the renderable types, or noise? Right now it is silently dropped everywhere.
4. **These differences are themselves a KG-quality signal** (relational depth / multi-hop capacity) — exactly what BioKGSuite's topology and task-performance dimensions measure. The per-KG bridge richness here likely predicts per-KG LLM lift, which ties straight into the keystone analysis.
