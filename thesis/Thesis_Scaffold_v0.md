# Doctoral Thesis — Working Scaffold (v1)

**Author:** Emily Molins · **Supervisors:** Chris, Agni · **Stage:** DPhil Year 1
**Status:** Draft scaffold for iteration — precursor to the one-page report requested in the 8 Jul 2026 supervisor meeting
**Last updated:** 9 Jul 2026

> Framing decided: a **methodological program** in causal and statistical machine learning for biomedicine, organized as a single pipeline. Neurovascular disease is the **application domain / testbed**, not the organizing principle.

---

## 1. Working title

**"A Causally-Grounded, In-Silico-to-Observational Pipeline for Drug Repurposing"**
*(subtitle option: "Knowledge-graph hypothesis generation, agentic trial-emulation design, and target trial emulation — with applications in neurovascular disease")*

---

## 2. Central thesis argument (the spine)

The thesis advances one methodological pipeline that carries a drug-repurposing hypothesis from **association to intervention** in three stages, each a distinct *kind* of evidence and a distinct methodological contribution:

1. **Hypothesis generation** — a biomedical knowledge graph proposes many candidate drug–disease pairs (broad, cheap, associational).
2. **Hypothesis operationalization** — a feasibility-and-design agent discards candidates the data can't actually test and ranks the rest by expected information, formalizing the survivors into valid target-trial protocols.
3. **Hypothesis testing** — target trial emulation on EHR cohorts estimates causal effects, narrowing to a small set (~10) of interventionally supported candidates.

**Why it holds together:** each stage repairs a failure mode of the others. The KG generates but can't distinguish causation from association; the agent decides what is *emulable* under real data limits but doesn't itself estimate effects; TTE estimates rigorously but is expensive and data-hungry, so it must be pointed only at questions worth asking. The funnel narrows as causal rigor rises.

**The novel core:** most repurposing pipelines rank candidates by *predicted efficacy*. This one adds a stage that ranks by **emulability** — learning which causal questions the available EHR can actually answer — turning the data-access constraint into the research contribution.

*(See `thesis_pipeline_diagram.svg` for the one-figure version.)*

---

## 3. The three stages as co-equal methodological pillars

Each pillar chapter follows the same internal structure: *motivation → method/contribution → neurovascular application vignette → evaluation → status.*

### Pillar I — Hypothesis generation: Knowledge graphs
- **Contribution:** representation and link-prediction methods over biomedical KGs to propose drug–disease candidates.
- **Current state (Year 1):** active — KG embedding work underway (TransE and related; the drug-repurposing notebooks). Most mature pillar; anchors the proof-of-concept.
- **Neurovascular vignette:** repurposing candidates / mechanism hypotheses for a neurovascular indication.
- **Method risks:** embedding-evaluation rigor (resampling/CI issue already logged), negative sampling, train/test leakage.

### Pillar II — Hypothesis operationalization: Feasibility & design agent
- **Contribution:** an agent that, per candidate, (a) checks emulability — definable eligible cohort, cohort size, a valid active comparator, positivity, adequate follow-up, measured confounders, structural red flags like immortal-time bias; (b) estimates expected power/precision; (c) ranks candidates by expected information gain; (d) auto-formalizes survivors into target-trial protocols (eligibility, treatment strategies, estimand).
- **Current state (Year 1):** proposed — the most novel pillar and the intended methodological centerpiece.
- **Neurovascular vignette:** decide which neurovascular repurposing candidates are testable in the available EHR before spending analytic budget.
- **Method risks:** validating the feasibility predictions themselves; generative/agent hallucination; encoding causal-design rules faithfully.

### Pillar III — Hypothesis testing: Target trial emulation
- **Contribution:** causal frameworks (target trial emulation, causal ML) estimating treatment effects from observational EHR data.
- **Current state (Year 1):** proposed — most data-dependent pillar; coupled to the Truveta timeline and alternative data streams.
- **Neurovascular vignette:** emulate a trial for a neurovascular treatment decision on an EHR-derived cohort.
- **Method risks:** identifiability, confounding, the data-access constraints raised in the 8 Jul meeting.

**Optional side-screen — Mendelian randomization:** a genetics-based causal pre-screen on the drug *target* axis that needs no EHR. Kept as an optional parallel filter and an explicit Truveta hedge; not part of the core funnel unless scope allows.

---

## 4. Neurovascular disease as application domain
- Serves as the **shared testbed** demonstrating the pipeline end-to-end on a real, hard clinical problem — not the organizing principle.
- Framing to state explicitly (pre-empting the "personal interest vs. coherent thesis" tension): one coherent application domain across all three stages is a strength — the synthesis chapter shows the stages composing on one problem.
- Domain results are framed as *validation*, not as the novel claim.

---

## 5. Data strategy (given the 8 Jul constraints)
Build so **no single stage dies if one dataset disappears.**
- **Truveta:** unavailable after Dec 2026; no imaging, no angiograms, uncertain access to extracted note "concepts." *Dependency, not foundation.* Pending answers from Chris/Annie: retain a small cohort (10k–50k) past year-end, concept-extraction support, post-submission access for edits.
- **Diversify streams (per meeting):** Agni → MGH neurology; Emily → Stanford neuro-IR; Chris → BDI / OUH records. Each stage names a *primary* and a *fallback* source.
- **Design principle:** favor methods that degrade gracefully under limited data — a reason Stage II (emulability screening) is valuable precisely *because* data is scarce.

---

## 6. Provisional chapter map
1. Introduction — the association→intervention pipeline; why these three stages.
2. Background — biomedical ML, KGs, causal inference / TTE, agentic methods; the neurovascular domain.
3. Pillar I — Knowledge graphs (hypothesis generation).
4. Pillar II — Feasibility & design agent (hypothesis operationalization).
5. Pillar III — Target trial emulation (hypothesis testing).
6. Synthesis — the full pipeline run end-to-end on a neurovascular repurposing problem.
7. Discussion, limitations, future work.

---

## 7. Year-1 plan & milestones
- **Now → autumn:** consolidate Pillar I into a submittable result; lock KG embedding evaluation.
- **Parallel:** resolve Truveta terms (Chris ↔ Annie) before committing Pillar III scope.
- **Home base:** secure external research community — BDI sit-in (via Chris), recurring meetings with Agni's clinical-research supervisees, OUH interventional-radiology observership. (EIT no longer offers a clinical-research peer group after the AIR/PJFM changes.)
- **Deliverable this cycle:** the one-page thesis-structure report (distilled from this scaffold).

---

## 8. Open questions for supervisors
1. Is drug repurposing the intended flagship application, or a warm-up before a neurovascular-specific KG?
2. How much should the thesis commit to Truveta before access terms are known?
3. Does Stage II stand alone as a methodological contribution, or is it evaluated only in-pipeline?
4. Keep Mendelian randomization in scope as a hedge, or cut for focus?
5. Does one application domain across all three stages read as coherence or as constraint to examiners?

---

*Next step: distill this into the one-page report for Chris and Agni, with the pipeline diagram as the single figure.*
