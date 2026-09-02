# Doctoral Thesis — Extended Scaffold (v2)

**Author:** Emily Molins · **Supervisors:** Chris, Agni · **Stage:** DPhil Year 1
**Framing:** A methodological program in causal/statistical ML for biomedicine, organized as one pipeline (hypothesis generation → operationalization → testing). Neurovascular disease is the application testbed, not the organizing principle.
**Last updated:** 11 Jul 2026

---

## 0. The one thing an examiner will ask: *what is new?*

**Honest finding from a hard literature pass (Jul 2026):** the pipeline is *not* novel, and — importantly — neither is "optimal-design/active selection for causal estimation," which is a fast-moving area right now. So the novelty flag is planted deliberately on the axis with the least traffic and where BioKGSuite already gives an advantage:

> **The thesis's primary contribution is a methodology for *honest evaluation of biomedical discovery under realistic constraints* — refusing to overstate what has actually been shown. Two instruments instantiate it: (i) contamination/leakage-controlled *prospective* evaluation of KG-generated hypotheses (BioKGSuite); and (ii) *emulability screening* — a quantitative triage of which causal questions are identifiable and adequately powered from the *fixed, available* EHR, before any effect is estimated. Neurovascular medicine is the domain. The through-line is a methodological stance + instruments, which is far harder to scoop than a single algorithm.**

What the thesis explicitly does **not** claim: a novel pipeline; a new causal-ML estimator; or a new active-learning optimizer for causal effects (that ground is being actively taken — see threat log). The contribution is the *evaluation methodology*, not new machinery.

**Why emulability-screening is still defensible despite the active-causal-learning boom:** budgeted active-experimentation methods (Gao 2026 [4]; ActiveCQ 2025 [5]) assume you can *collect more data* to sharpen an estimate. Emulability screening is the opposite, more realistic regime: the data is *fixed and shrinking* (Truveta ends Dec 2026), so the question is *which of many candidate questions are even identifiable/powered from what already exists.* That is a screening/feasibility problem, not an active-sampling one — and the closest work (Wang 2026 [6]) frames it only *qualitatively*.

### The unifying meta-spine: fitness-for-purpose evaluation at every handoff
The deeper thread that makes the three stages one thesis (not three papers): **at each stage the contribution is not just an output but a measure of whether that output is fit for the next stage — and fitness is defined by the downstream use, not intrinsically.**

- **Stage I quality — is the KG good, and *good for what?*** This is exactly what **BioKGSuite** (Emily's existing work) establishes: a 7-dimension / 18-metric quality framework over six biomedical KGs, plus a prospective, leakage-controlled LLM+KG repurposing benchmark, whose headline finding is that **intrinsic KG quality does not cleanly predict downstream LLM utility** (intrinsic winner ≠ downstream winner). Nature Communications manuscript in progress. *One chapter is already substantially done.*
- **Stage II quality — is the candidate *emulable*, and worth testing?** The emulability/optimal-design layer applies the *same* intrinsic-vs-extrinsic logic to hypotheses: a candidate's value is defined by what the EHR can actually test, not by its KG-embedding score.
- **Stage III quality — is the causal estimate *trustworthy?*** TTE validity: identifiability, negative-control calibration, sensitivity analysis.

The intrinsic≠extrinsic result from BioKGSuite is therefore not a one-off benchmark finding — it is the **philosophical claim of the whole thesis**, first proven for knowledge graphs and then extended to hypothesis selection and causal estimation. This elevates Stage I from "background" to a load-bearing chapter and de-risks Year 1.

### Prior-art threat log (viva-ready; be able to differentiate from each)

*Pipeline / generation — occupied:*
- KG → emulated-trials-on-EHR at scale, screening many candidates: Liu 2021 (Nat Mach Intell); Zang 2023 (Nat Comms, thousands of drugs / 170M patients); Yan 2024 (GenAI proposes → EHR-validates). *The pipeline is not novel.*
- KG repurposing is a mature, reviewed field (Perdomo-Quinteiro 2024; Wei 2025).

*Instrument (i) — contamination-controlled KG repurposing evaluation — has a close neighbor:*
- Brière et al. 2025 (bioRxiv): benchmarks **data leakage in biomedical KGE link prediction**, builds redundancy-removal control, shows collapse on a real-world (Orphanet) inference set vs. random splits — your leakage argument, for KGE.
- Réda et al. 2025 (Sci Rep): leakage/extrapolation warnings for repurposing benchmarks. Singh et al. 2024: LLM eval-contamination metrics (ConTAM).
- **BioKGSuite's remaining edge:** the multi-dimensional quality framework (7×18 across 6 KGs) + the *prospective, approval-date LLM-grounding* task + the intrinsic≠extrinsic finding. Real but **incremental** benchmark contribution.

*Instrument (ii) — emulability screening — has adjacent work, but the specific framing is open:*
- Kamdje Wabo et al. 2024 (JMIR Med Inform): "**fitness-for-purpose of EHR data**" + pathway to automation — but *generic data quality for use*, not causal-question-specific.
- EU-PEARL / Lombardo 2023 (JBI): **EHR protocol feasibility** (cohort/eligibility readiness) — but for *prospective-trial recruitment*, not observational-emulation identifiability/power.
- Wang 2026 (NPJ Dig Med): emulability, but **qualitative** checklist.
- Gao 2026 (budgeted active experimentation) / ActiveCQ 2025: active selection assuming you can **collect more data** — opposite of the fixed-shrinking-data regime.

*Domain — open:*
- Trial emulation concentrated in cardiology/ID/oncology (Scola 2023); neurovascular KG→emulability→TTE is unclaimed.

### The gap, stated crisply (honest, narrowed)
No one has connected KG hypothesis-quality evaluation, EHR data-fitness, and trial-emulation feasibility into a **causal-question-level emulability triage** — a quantitative predictor that ranks which KG-generated candidate questions are *identifiable and adequately powered from a fixed EHR* to decide the emulation order. The closest works are either qualitative (Wang 2026), generic-data-quality (Kamdje Wabo 2024), recruitment-feasibility (EU-PEARL), or assume more data can be collected (Gao 2026). **Novelty type = problem-formulation + integration + domain, not a landmark new algorithm** — sufficient for a DPhil, but defended as "operationalized and connected what others left qualitative/disconnected," with BioKGSuite as the already-in-hand anchor.

---

## 1. Working title
**"Causally-Grounded Biomedical Discovery: Hypothesis Generation, Operationalization, and Testing from In-Silico to Observational Evidence — with Applications in Neurovascular Medicine."**

*Framing:* thesis organized **by method** (the three phases), with **neurovascular medicine as the single unifying application domain**. Generality is claimed not across diseases but across **intervention types within neurovascular medicine** (drug efficacy, drug safety, procedure/device, screening) — the same three fitness gates holding across all of them is the generalization result.

*(Alt / punchier internal handle for the Stage II contribution: "Learning What Can Be Tested.")*

---

## 2. Extended pipeline (methods to cover, by stage)

Three stages, each a chapter. Stage II is the centerpiece; Stages I and III are competent applications of known methods that set it up and cash it out. A feedback loop makes the whole thing an active-learning system rather than a one-shot funnel.

### Stage I — Hypothesis generation & KG quality (BioKGSuite) — *already underway; higher novelty than first assumed*
*Role: produce candidate drug–disease pairs AND establish how good the generating substrate is, and good for what.*
- **BioKGSuite (existing):** 7-dimension / 18-metric quality benchmark across 6 KGs (PrimeKG, Hetionet, DRKG, OpenBioLink, BioKG, MATRIX); prospective, leakage-controlled 116-pair LLM+KG repurposing benchmark; grounding scaling-law (Llama 1B→405B, saturates ~70B); embedding validation (TransE/RotatE vs. name-prior baseline). Nature Comms manuscript in progress.
- **The chapter's contribution:** intrinsic quality metrics vs. downstream utility — the intrinsic-vs-extrinsic result that seeds the thesis meta-spine. Open gap to close (per manuscript outline §7): run the LLM task on all 6 KGs so the intrinsic-vs-extrinsic claim is testable.
- **Handoff to Stage II:** output is a **calibrated hypothesis prior with uncertainty**, plus a *fitness-annotated* KG (which KG/subgraph is trustworthy for which candidate) — this is what the emulability layer consumes.

### Stage II — Hypothesis operationalization (Emulability & design layer) — *the contribution*
*Role: decide which candidates are testable and worth testing; formalize the survivors.*
- **Identifiability screening:** encode each candidate as a target-trial protocol (DAG/SWIG); check identification (do-calculus / ID algorithm); flag structural threats (immortal-time, no clean index date, no valid active comparator).
- **Feasibility & power prediction (learnable):** predict, from the EHR, definable-cohort size, positivity/overlap, follow-up adequacy, confounder measurability — and a **calibrated estimate of the precision/power** of the eventual effect estimate. Train/validate these predictors against realized emulation outcomes (this is a core empirical chapter).
- **Optimal-design ranking:** rank candidates by **expected information gain / value of information** under a limited analytic budget (adapt Bayesian OED [7] to observational-emulation selection). Objective: maximize expected number of *correctly resolved* causal questions per unit budget.
- **Active-learning loop:** feed realized feasibility/effect results back to update the KG prior and the feasibility predictor — the funnel becomes sequential and self-improving.
- **Agentic implementation (optional layer):** an LLM agent that assembles protocols and emulation code (à la [4]) *conditioned on* the Stage-II decision — positioned as engineering, not the novel claim.

### Stage III — Hypothesis testing (Target Trial Emulation)
*Role: rigorously estimate effects for the prioritized survivors; established methods.*
- TTE execution: clone-censor-weight, sequential trial emulation, landmark analysis (Hernán framework [9]).
- Estimation: doubly-robust / semiparametric (TMLE, AIPW), double/debiased ML; CATE learners for heterogeneity where warranted.
- Robustness/validity: E-values and negative-control/proximal methods for unmeasured confounding; conformal / influence-function inference; multiplicity control (FDR) since many candidates are tested.
- Multi-site: federated or meta-analytic estimation across MGH / Stanford / OUH streams to address the Truveta cutoff.

### Optional side-screen — Mendelian randomization
Genetics-based causal pre-screen on the drug *target* axis; needs no EHR; explicit Truveta hedge. In scope only if time allows.

---

## 3. Neurovascular medicine as the unifying application domain
Neurovascular medicine is the single clinical domain across all three method chapters — providing depth, clinical collaborators, and a home for the IR observership and the MGH/Stanford/OUH data streams. **Generality is demonstrated across hypothesis *types* (forms of causal question) within neurovascular, not across diseases.**

- **Domain white space:** trial-emulation methodology is concentrated in cardiology/ID/oncology [8]; neurovascular (ischemic stroke, aneurysmal SAH, cerebral aneurysm growth/rupture, vascular malformations) is comparatively unexplored — so the *application* results are publishable in their own right.

**Hypothesis types (the generality claim = same fitness gates across every hypothesis *form*):**

| Hypothesis type | Example neurovascular causal question | Enters via | Note |
|---|---|---|---|
| Efficacy — established indication (validation) | Does nimodipine reduce delayed cerebral ischemia after SAH? | Stage I (KG) | known-effect sanity check; emulation-vs-RCT |
| Efficacy — repurposing / novel indication ★ *flagship* | Does statin / SGLT2-inhibitor exposure slow unruptured aneurysm growth or rupture? | Stage I (KG) | KG-generated + leakage-controlled; the flagship |
| Drug safety / harm | DOAC vs. warfarin: intracranial hemorrhage risk in AF? | Stage I (KG) | safety outcomes well-recorded → *highly emulable* |
| Comparative effectiveness (procedure/device) | Flow-diverter vs. coiling for unruptured aneurysm: occlusion + complications? | Stage II (non-drug) | procedure TTE; IR domain |
| Care-delivery / process-of-care | Mobile stroke unit / shorter door-to-groin time → 90-day functional outcome? | Stage II (non-drug) | systems TTE; highly emulable |
| Management strategy / timing | Intervene vs. conservatively manage small unruptured aneurysms (ARUBA-style for AVM); anticoagulation timing after ICH | Stage I or II | strategy TTE; watchful-waiting comparator |
| Screening / surveillance | Aneurysm surveillance interval (6 vs. 12 mo) → rupture-free survival | Stage II (non-drug) | screening-strategy estimand |
| Effect heterogeneity / precision | Does thrombectomy benefit vary by collateral status / core volume? | cross-cuts | CATE within TTE |

*Enters via:* **Stage I (KG)** = the biomedical knowledge graph generates the hypothesis (drug-centric); **Stage II (non-drug)** = procedures, care-delivery, and screening hypotheses come from clinical literature / registries / expert priors and enter directly at the emulability layer, since the KG doesn't generate them. This keeps the BioKGSuite generation story drug-focused while demonstrating that Stages II–III generalize beyond drugs.

*Illustrative — vet with IR collaborators. Several (ARUBA/AVM, thrombectomy heterogeneity) have RCT evidence, useful as emulation-vs-trial validation targets.*

- **Stage I (general method, neurovascular instantiation):** BioKGSuite stays a *general* KG-quality contribution; its worked example is neurovascular repurposing-candidate generation.
- **Stage II demonstration:** many appealing neurovascular questions are *un-emulable* in the available EHR (small cohorts, missing severity/imaging concepts — the Truveta imaging limitation); the emulability ranking reallocates effort to answerable ones. **The most persuasive demonstration of the contribution** because the data scarcity is real.
- **Stage III demonstration:** emulate the surviving neurovascular questions; where a real RCT exists, report emulation-vs-trial concordance as validation.

---

## 4. Validation strategy (how each novel claim is defended)
1. **Stage II predictors are calibrated:** hold-out evaluation of feasibility/power predictions against realized emulation outcomes (calibration curves, not just AUC).
2. **The ranking is better than baselines:** compare emulability-ranked selection vs. (a) efficacy-ranked selection [2,3], (b) random/greedy selection, (c) the Wang-2026 qualitative checklist [6] — on "correct causal questions resolved per budget."
3. **The loop helps:** ablate the active-learning feedback; show sequential selection beats one-shot.
4. **End-to-end sanity:** on a domain with known answers, recover established effects; report emulation-vs-RCT concordance where an RCT exists.
5. **Robustness:** negative-control calibration and sensitivity analysis on all reported Stage III effects.

---

## 5. Provisional chapter map (with per-chapter novelty)
1. **Introduction** — the "learning what can be tested" thesis; why the decision layer, not the pipeline, is the contribution.
2. **Background & prior art** — KGs, TTE [9], OED/VoI [7], the emulability question [6]; explicit gap statement.
3. **Stage I — Knowledge graphs & KG quality (BioKGSuite)** *(novelty: high — quality framework + intrinsic-vs-extrinsic finding; Nature Comms manuscript in progress; seeds the fitness-for-purpose meta-spine).*
4. **Stage II — Emulability & optimal-design layer** *(novelty: high — the core methodological chapter; likely 2 chapters: 4a feasibility/identifiability prediction, 4b optimal-design ranking + active-learning loop).*
5. **Stage III — Target trial emulation** *(novelty: modest — rigorous application; multi-site estimation under data constraints).*
6. **Synthesis** — full pipeline on a neurovascular repurposing problem; the flagship end-to-end result.
7. **Discussion, limitations, future work.**

---

## 6. Data strategy (given 8 Jul constraints)
Build so no stage dies if one dataset disappears. Truveta unavailable after Dec 2026 (no imaging/angiograms, uncertain note "concepts") — treat as *dependency, not foundation*; the emulability layer is *more* valuable precisely because data is scarce. Diversify: Agni→MGH neurology, Emily→Stanford neuro-IR, Chris→BDI/OUH; each stage names a primary + fallback source. Pending: Chris↔Annie on retaining a 10k–50k cohort past year-end, concept-extraction support, post-submission access.

---

## 7. Open questions for supervisors
1. Is the emulability layer (Stage II) accepted as *the* thesis contribution, with I and III as supporting?
2. How hard to differentiate from Wang et al. 2026 [6] — is "qualitative framework → learnable optimization" enough daylight, and how fast is that space moving?
3. Neurovascular indication to anchor the vignettes (stroke vs. aneurysm/SAH vs. malformations) — driven by which cohort is actually emulable?
4. Commit analytic budget to Truveta before access terms are known, or design Stage II to be dataset-agnostic and demonstrate on MGH/Stanford first?
5. Keep Mendelian randomization in scope as a hedge, or cut for focus?

---

*Refs [1]–[9] correspond to the literature list in the accompanying chat message. Next step: distill the one-pager for Chris and Agni, and (optionally) extend the diagram to show the active-learning feedback loop that makes Stage II the star.*
