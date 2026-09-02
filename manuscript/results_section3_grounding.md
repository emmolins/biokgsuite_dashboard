# Grounding gains depend on which knowledge graph supplies the evidence

The profiles in Figs. 2 and 3 characterise the six graphs but do not establish that the
characterisation is decision-relevant. Testing that requires a task on which a graph can
measurably help or fail to help, evaluated on cases whose answers are available neither in a
model's training data nor in the graph itself. We therefore constructed a prospective,
leakage-controlled evaluation set from regulatory approvals postdating the release of the
resources under test, and used it to measure how much each graph improves large language model
repurposing predictions.

The evaluation set begins from 253 new indications granted to previously approved drugs by the
FDA, EMA or PMDA. Nineteen pairs were removed for unresolvable identifiers, 81 because the
drug–disease edge was already present in at least one evaluated graph, and 37 because neither
entity appeared in any graph, leaving 116 pairs (Fig. 5a). Seven of these resolve to no single
drug node, six being fixed-dose combination products, so 109 pairs are evaluable. The retained
set spans nine therapeutic areas, led by oncology (48 pairs) and immunology (28). Comparing retained against
discarded pairs across six characteristics found a single difference, retained drugs having
received their original approval a median of two years more recently (6 against 8 years,
P = 0.013, Mann–Whitney); oncology status, biologic modality, multi-agency approval, strict
repurposing status and therapeutic area were all non-significant (Supplementary Table S4).

Three graphs supported the experiment. Hetionet and OpenBioLink cover too few retained pairs to
estimate performance, yielding 6 and 35 positives against 50 for PrimeKG, and MATRIX postdates
the approval window that defines the evaluation set, so including it would not have been
leakage-controlled. PrimeKG, DRKG and BioKG cover 105, 85 and 73 of the 109 evaluable pairs
respectively, and none retains a residual drug–disease edge after filtering.

Each query presents one disease together with a pool of candidate drugs, exactly one of which
received the approval, and asks the model to rank the pool by repurposing plausibility. Arms
differ only in the evidence attached to each candidate. The baseline arm supplies none. Each
graph arm supplies one-hop neighbourhoods rendered as text, retaining gene, pathway and
related-disease neighbours and excluding the drug–disease relation structurally so that no arm
can contain the answer, with named pharmacokinetic relations dropped (Fig. 5b). We evaluated
three models drawn from different families, GPT-4.1-mini, Llama-3.3-70B and
Gemini-3.1-Flash-Lite, across 109 diseases with candidate order shuffled between repeats.

Grounding improves ranking accuracy substantially in every model. With PrimeKG supplying the
evidence, mean reciprocal rank rises from 0.236 to 0.785 for Gemini-3.1-Flash-Lite, from 0.273
to 0.756 for GPT-4.1-mini and from 0.320 to 0.644 for Llama-3.3-70B, and every paired confidence
interval excludes zero (Fig. 6a). That grounding helps is expected. What the evaluation set
makes measurable is how far the benefit depends on which graph is used.

The size of the gain is a property of the graph rather than of the model. PrimeKG produces the
largest gain, BioKG the next largest and DRKG the smallest, and this ordering is identical in
all three models despite their differences in family, scale and ungrounded capability
(Fig. 6a). PrimeKG's advantage over DRKG is 0.127 MRR in GPT-4.1-mini and 0.093 in
Gemini-3.1-Flash-Lite, both with confidence intervals excluding zero (Fig. 6b), and is
comparable to the 0.084 spread in ungrounded performance across the three models. Choosing the
evidence source therefore matters about as much as choosing the model. The separation is
two-tiered rather than strictly ordered, since PrimeKG separates from both other graphs while
BioKG and DRKG do not separate from each other in any model.

PrimeKG's advantage coincides with its coverage advantage. Coverage is the dimension governing
whether a queried pair has retrievable evidence at all, and the only one on which PrimeKG exceeds
the other two by a margin comparable to its margin in grounding gain (0.583 against 0.465 and
0.463). It is not the dimension that determines their overall ranking, since DRKG scores above
BioKG on four of seven dimensions and on the overall mean yet produces the smaller gain. Three
graphs are too few to estimate a relationship between profile and grounding benefit, so this
correspondence is a hypothesis the framework generates rather than one it establishes, and we
return to it in the Discussion.

Two questions that intrinsic metrics alone cannot separate are therefore separable here. Whether
grounding is worth doing is settled by the aggregate lift; how much it is worth depends on the
resource chosen, and is predicted by the dimension of the profile that governs evidence
availability rather than by the profile as a whole.

---

## Display items

**Fig. 5 | Design of the prospective, leakage-controlled evaluation set and the grounding
experiment.**
**a**, Construction of the evaluation set from 253 candidate approvals, showing pairs removed at
each filter and the therapeutic-area composition of the retained set. **b**, Structure of a single
query. One disease is paired with a candidate pool containing exactly one post-cutoff positive.
Arms differ only in the evidence block attached to each candidate, which is empty in the baseline
arm and drawn from one-hop graph neighbourhoods otherwise. The drug–disease relation is excluded
structurally from every evidence block. The example is a retained pair of the evaluation set
(sarilumab, polymyalgia rheumatica). PMR, polymyalgia rheumatica; GCA, giant cell arteritis.

**Fig. 6 | The graph supplying the evidence determines the size of the grounding gain.**
**a**, Mean reciprocal rank by model and evidence arm across 109 diseases. Error bars are 95%
bootstrap confidence intervals over diseases. **b**, Paired between-graph differences in mean
reciprocal rank, by model. Filled markers denote confidence intervals excluding zero; open
markers denote intervals including it.

---

## Notes for revision

- The 253→116 funnel reconciles exactly (253 − 19 − 81 − 37 = 116). Seven of those survivors are
  not flagged `in_final` in `reconstruct_116_audit.csv`, leaving 109, which matches the 109
  diseases actually run. Six are fixed-dose combinations with `n_present = 0`. The seventh,
  Pair 94 (Rituxan, `n_present = 1`), has no stated reason and should be checked.
- Therapeutic-area counts and per-graph coverage in the text are computed from the funnel
  survivors, not the `in116` flag, which disagrees by a few pairs. Figure 5 regenerates both
  from source, so the text and figure cannot diverge.
- Hetionet and OpenBioLink positive counts (6 and 35) come from `09_auroc_ci.csv`, which is the
  earlier binary design. Confirm the same counts hold under the listwise design, or state the
  exclusion threshold prospectively in Methods.
- The across-family comparison (heuristics against embeddings) is deferred pending the nb08
  TransE rerun. RotatE and heuristic columns are unaffected.
