# Methods — construction of the 116-pair prospective drug-repurposing benchmark

## Candidate generation (→ 253)
Drug-repurposing events were drawn from new-indication regulatory approvals (2023–2026; FDA, EMA,
PMDA, TGA/MHRA-Orbis). The 2024–2026 approvals were combined with 96 reconstructed 2023
new-indication approvals, deduplicated to distinct regulatory events, and resolved to ontology
identifiers — drugs to DrugBank/PubChem and diseases to MONDO/MeSH/DOID — through a multi-source
cascade (PubChem PUG-REST, Wikidata, mychem.info for drugs; OLS4, Monarch and NLM MeSH for diseases),
yielding **253 ID-resolved candidate drug–disease pairs**.

## Filtering funnel (253 → 116)
Three successive filters produced the evaluable test set:

1. **Unresolved identifiers (−19 → 234).** Pairs lacking a DrugBank ID (n = 7; chiefly CAR-T cell
   therapies and biologics with no DrugBank entry), lacking any resolved disease identifier (n = 2),
   or carrying a placeholder/not-yet-published indication (n = 12; EMA 2026 "TBD" entries) were
   removed, leaving **234 harness-ready pairs** (strict, prospective repurposing with label dates
   post-2023).

2. **Answer leakage / contamination (−91 → 143).** A pair was excluded if a direct drug→disease edge
   already existed in **any** benchmarked knowledge graph (PrimeKG, DRKG, BioKG, HetioNet,
   OpenBioLink), since such an edge would let a model read the answer directly off the graph. This
   yielded **143 leakage-free pairs**.

3. **Knowledge-graph coverage (−27 → 116).** A pair was retained only if the true drug and its disease
   both resolve to typed nodes in at least one of the three evaluation KGs (PrimeKG, DRKG, BioKG;
   OpenBioLink and HetioNet were excluded from evaluation for weak coverage). This produced the final
   **116 evaluable pairs**, of which 111 are covered by PrimeKG, 90 by DRKG and 78 by BioKG, with 72
   covered by all three. Each pair becomes one anonymized ranking query (the true drug among 8
   candidates: 1 true + 7 Open Targets approved-drug distractors), evaluated across no-KG and the
   three KG arms.

## Reproducibility
The funnel is reproduced from the raw KG files in `scripts/build_116.py` (with the validated
node-resolution logic in `scripts/reconstruct_116.py`). The coverage step is exact: a from-scratch
re-implementation of the typed drug/disease node resolution reproduces the per-KG coverage of the
116 (PrimeKG 111, DRKG 90, BioKG 78) at **100% per-pair agreement**. The unresolved-ID step (−19) is
exact. The leakage step reproduces 84–85 of the 91 excluded pairs; the residual reflects disease-ID
synonym crosswalking (expansion of each disease to equivalent MeSH/MONDO/DOID via `mondo.sssom.tsv`
and `mesh_to_doid.csv`) used in the original audit. Per-stage dropped-pair lists are written to
`results/tables/build_116_dropped_step{1,2,3}_*.csv`; full per-pair flags to
`results/tables/build_116_funnel.csv`. See `data/gold_standards/PROVENANCE_253_candidate_pairs.md`
for the candidate-generation record.
