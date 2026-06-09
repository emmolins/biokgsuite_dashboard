# What goes in the kg_block (nb09 redesign)

The kg_block is the **treatment** in the KG-quality experiment — the only channel
through which a KG's quality reaches the LLM. Governing principle:

> **The block must encode the quality dimensions the benchmark measures, and it
> must not hand the model the answer. Anything it omits is a dimension the
> KG-quality correlation can't detect; anything that pre-computes the link is
> leakage dressed up as evidence.**

## Why not the drug→disease "bridge"

The natural first design extracts the mechanistic path that connects the drug to
the disease (drug → shared gene → disease). We built it, and rejected it, for two
reasons:

1. **It's cherry-picked / leaky.** Extracting "the gene that links this drug to
   this disease" means running the algorithm that *finds the answer* and handing
   the result to the model. No pharma analyst is given a pre-distilled connecting
   path; receiving one is the tell that the evidence was reverse-engineered from
   the label.
2. **It barely exists.** On PrimeKG the strict slot bridge covered only **2/24**
   candidates (1/4 positives): it required the drug's *target gene itself* to be
   directly annotated as a disease gene, which most real mechanisms aren't.

## The design: balanced dual dossiers

Extract what a pharma analyst actually pulls — two neutral, **query-independent**
profiles — and let the model connect them:

- **Drug dossier** (per candidate, shown next to each drug): `targets`,
  `pathways`, `approved/known indications`, `side effects`.
- **Disease profile** (once per query, in the prompt header): `associated genes`,
  `phenotypes`.

The model sees the drug's targets and the disease's biology *separately* and must
notice the overlap (or reason about an indirect link) itself. The KG supplies
ingredients; the reasoning is the model's. This is what we want to score —
"can it use balanced KG context to find the mechanism," not "can it read back the
mechanism we found for it."

**The not-cherry-picked invariant: query-independence + a fixed schema.** Extract
what is true about each entity on its own, with identical fields for every
candidate. Symmetry is the test — if the true drug gets a special connecting path
that distractors don't, it's cherry-picked; if every candidate gets the same
neutral profile, it isn't.

## Handling high-degree nodes (1000+ neighbours)

Do **not** threshold degree (the old `hub_cap` gave 0 coverage at 200, arbitrary
at 3000). Instead **rank by specificity and take top-N**: sort a node's
neighbours by *their own* degree ascending and keep the top `cap` (default 12). A
disease gene linked to 3 diseases is informative; one linked to 900 is noise.
This demotes 1000+-neighbour hubs automatically, bounds the token count, and is
neutral — it never looks at the candidate drug. (Edge confidence or source-count
would be an even better neutral ranker; PrimeKG carries `source`, a later
upgrade.) Capping by "relevance to the candidate" is forbidden — that rebuilds
the bridge.

## Leakage controls

- **Mask the drug's indication for the target disease.** Its *other* indications
  stay as honest repurposing-landscape context. (Verified: venetoclax shows
  "Richter syndrome, ALL", not AML; upadacitinib shows RA, not eczema.)
- **Query-independent extraction** — the disease profile is identical across all
  candidates, so it can't favour the positive.
- The shortcut (drug-target ∩ disease-gene overlap) is *not* eliminated — but it
  is now the model's inference over neutral context, not our retrieval of the
  answer. That relocation is the point.

## Quality-dimension coverage

| Dimension | Carried by |
|---|---|
| Coverage | whether each dossier section is populated (omitted when empty) |
| Annotation accuracy | slot-typed fields (`targets`, `associated genes`, …) via `kg_slot_maps.yaml` |
| Trustworthiness | provenance/confidence tags (planned; PrimeKG `source` available) |
| Topology | breadth of associations surfaced per entity |

Six of the seven canonical slots feed the dossiers, vs. two for the old bridge —
so more quality dimensions can actually move the output.

## Arms

- `full` — drug dossiers + disease profile.
- `mechanism_only` — drug dossiers only, disease profile suppressed (model bridges
  from its own disease knowledge). The gap between the two measures how much the
  KG disease context contributes.

## Excluded

- The drug→disease connecting path (cherry-picked / leaky).
- Hard degree thresholds (replaced by specificity ranking).
- Raw per-KG predicates — normalized through `kg_slot_maps.yaml` so a field means
  the same thing across all six KGs.
- Long anatomy/expression/DDI tails — unmapped in the slot map, left out.

## Invariants

- Identical schema and surface form across all six KGs.
- Short — `cap` per section; long blocks degrade reasoning and create the
  annotation-salience confound.
- Omit empty sections / empty blocks — never the word "none".

## Empirical status (PrimeKG pilot)

- Dual dossier coverage: **24/24** candidates, 4/4 positives (vs. 2/24 for the
  strict bridge).
- Caveat: PrimeKG `drug_protein` mixes pharmacological targets with ADME genes
  (CYP3A4, ABCG2, transporters), so `targets` carries metabolism noise — realistic
  for a dossier, and the true target is always present. Refinable by relation
  sub-typing.

## Open questions

1. Do all six KGs carry usable edge **provenance/confidence**? If only some do,
   provenance becomes a trust-dimension confound and must be reported per KG.
2. Sub-type `drug_target` to separate pharmacological targets from ADME genes?

Implemented in `scripts/pilot_ranking.py` (`make_kg_block_fn`).
