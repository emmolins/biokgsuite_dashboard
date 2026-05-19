# Prospective Gold Standard Vetting Report

**Date:** 2026-04-15  
**File vetted:** `src/regulatory_approvals.py`  
**Total records:** 107 (40 FDA, 45 EMA, 11 PMDA, 11 TGA) → 54 unique (drug, disease) pairs

---

## 1. Curation Methodology (as originally implemented)

The gold standard was curated from AI training knowledge (Claude's knowledge through early 2025), NOT from direct database queries. The method was:

1. **Drug selection:** Drugs present in ≥1 of the 5 benchmark KGs (Hetionet, DRKG, PrimeKG, BioKG, OpenBioLink)
2. **Indication filter:** New indication approvals after the KG construction cutoff (~2020–2022), so the KG could not have encoded the relationship at training time
3. **Entity normalization:** DrugBank accessions for drugs, Disease Ontology DOIDs for diseases
4. **Multi-agency expansion:** Extended from an initial ~41 FDA-only set to include EMA EPARs, PMDA, and TGA approvals

**Critical limitation:** Because the curation relied on AI-recalled knowledge rather than systematic database queries, it is susceptible to hallucinated entries, incorrect identifiers, and wrong dates. The vetting below confirms this concern.

---

## 2. DrugBank ID Errors (9 of 33 unique drugs are WRONG)

| Drug | ID in File | Correct DrugBank ID | Source |
|------|-----------|---------------------|--------|
| trastuzumab deruxtecan | DB15687 | **DB14962** | go.drugbank.com/drugs/DB14962 |
| glofitamab | DB16957 | **DB16371** | go.drugbank.com/drugs/DB16371 |
| epcoritamab | DB17098 | **DB16672** | go.drugbank.com/drugs/DB16672 |
| talquetamab | DB17191 | **DB16678** | go.drugbank.com/drugs/DB16678 |
| teclistamab | DB16870 | **DB16655** | go.drugbank.com/drugs/DB16655 |
| sacituzumab govitecan | DB15773 | **DB12893** | go.drugbank.com/drugs/DB12893 |
| capivasertib | DB15947 | **DB12218** | go.drugbank.com/drugs/DB12218 |
| zanubrutinib | DB14960 | **DB15035** | go.drugbank.com/drugs/DB15035 (DB14960 = somatrogon!) |
| tucatinib | DB15262 | **DB11652** | go.drugbank.com/drugs/DB11652 |

**Impact:** These wrong IDs mean the drugs will NOT map to KG entities during the prospective evaluation. Any pair using these drugs would fail to match and be silently dropped, biasing results. **This is the most critical finding.**

### Verified Correct DrugBank IDs (24 of 33)

DB09037 (pembrolizumab), DB09035 (nivolumab), DB11714 (durvalumab), DB12159 (dupilumab), DB13928 (semaglutide), DB15091 (upadacitinib), DB14762 (risankizumab), DB11817 (baricitinib), DB09074 (olaparib), DB06292 (dapagliflozin), DB09038 (empagliflozin), DB08877 (ruxolitinib), DB08879 (belimumab), DB08875 (cabozantinib), DB11595 (atezolizumab), DB00051 (adalimumab), DB06273 (tocilizumab), DB09053 (ibrutinib), DB11581 (venetoclax), DB11703 (acalabrutinib), DB00072 (trastuzumab), DB09078 (lenvatinib), DB01590 (everolimus), DB09331 (daratumumab)

---

## 3. Disease Ontology (DOID) Errors

| Disease in File | DOID in File | Issue | Correct DOID |
|----------------|-------------|-------|-------------|
| endometrial carcinoma | DOID:2871 | DOID:2871 = "endometrial stromal tumor" | **DOID:2870** (endometrial adenocarcinoma) or use parent DOID:1380 (endometrial cancer) |
| gastric cancer | DOID:10534 | DO canonical name is "stomach cancer" | Acceptable synonym — ID is correct |
| atopic eczema | DOID:3310 | DO canonical name is "atopic dermatitis" | Acceptable synonym — ID is correct |
| cardiovascular disease | DOID:1287 | DO canonical name is "cardiovascular system disease" | Acceptable synonym — ID is correct |

**Critical error:** DOID:2871 maps to endometrial stromal tumor (a rare mesenchymal neoplasm), NOT endometrial carcinoma (the epithelial cancer that pembrolizumab/lenvatinib treat). This affects 3 records (pembrolizumab→endometrial carcinoma via EMA/FDA/TGA, and lenvatinib→endometrial cancer).

---

## 4. Approval Date / Factual Errors

### Entries that should be REMOVED (fabricated or pre-2022)

| # | Drug | Disease | Date in File | Issue |
|---|------|---------|-------------|-------|
| 1 | atezolizumab | hepatocellular carcinoma | 2023-10-15 | **Original approval was May 2020** — pre-2022, should not be in post-2022 gold standard. No 2023 expanded HCC indication found. |
| 2 | ibrutinib | follicular lymphoma | 2024-03-29 | **Never approved for FL.** Ibrutinib's accelerated approvals for MCL and MZL were actually *withdrawn* in 2023. This entry is fabricated. |
| 3 | venetoclax | myelodysplastic syndrome | 2024-06-14 | **Not approved for MDS.** Only has Breakthrough Therapy Designation (2021). Clinical trials ongoing. |
| 4 | trastuzumab | gastric cancer (updated regimen) | 2024-06-13 | **Original trastuzumab gastric approval was 2010.** The 2024 gastric activity was pembrolizumab combination approvals, not a new trastuzumab indication. |

### Entries with uncertain dates (could not confirm exact date)

| Drug | Disease | Date in File | Notes |
|------|---------|-------------|-------|
| ibrutinib | CLL | 2023-12-01 (FDA) | Original CLL approval was 2014. No clear evidence of a December 2023 supplemental expansion found. May refer to a formulation change (oral suspension label expansion). |

### Spot-checked and CONFIRMED correct

| Drug | Disease | Date | Agency |
|------|---------|------|--------|
| dupilumab | COPD | 2024-09-27 | FDA ✓ |
| semaglutide | CV risk reduction | 2024-03-08 | FDA ✓ |

---

## 5. Potentially Missed Post-2022 Approvals

Based on FDA new indications archives (drugs.com 2023/2024), several notable expanded indications for drugs likely present in the KGs are absent:

| Drug | New Indication | Approval | Notes |
|------|---------------|----------|-------|
| pembrolizumab | malignant pleural mesothelioma | 2024 FDA | Not in our gold standard |
| nivolumab | urothelial carcinoma (with cisplatin/gemcitabine) | 2024 FDA | Not in our gold standard |
| dupilumab | eosinophilic esophagitis (EoE) | 2024 FDA | Age expansion — potentially relevant |
| durvalumab | neoadjuvant/adjuvant resectable NSCLC | 2024 FDA | Not in our gold standard |
| ribociclib | adjuvant HR+/HER2- breast cancer | 2024 FDA | Major approval, not included |
| amivantamab | NSCLC with EGFR exon 20 insertions | 2024 FDA | May not be in older KGs |

---

## 6. Summary of Required Corrections

### Critical (must fix before publication)

1. **Fix 9 wrong DrugBank IDs** — these cause entity mapping failures
2. **Fix DOID:2871 → DOID:2870** for endometrial carcinoma entries
3. **Remove 4 fabricated/invalid entries:**
   - atezolizumab → HCC (pre-2022 approval)
   - ibrutinib → follicular lymphoma (never approved)
   - venetoclax → MDS (not approved)
   - trastuzumab → gastric cancer "updated regimen" (not a new indication)

### Recommended

4. **Investigate ibrutinib → CLL date** (2023-12-01) — may be a formulation change, not a true new indication
5. **Add missed approvals** from the list in Section 5 if the drugs map to KG entities
6. **Re-run notebook 07** after corrections to get accurate prospective evaluation

### Net effect on gold standard size

- Current: 54 unique pairs
- After removing 4 invalid: ~50 pairs
- After adding missed approvals: potentially 53–56 pairs
- After fixing DrugBank IDs: more pairs will successfully map to KG entities, improving statistical power

---

## 7. Recommendation for the Paper

Given that ~27% of DrugBank IDs were wrong and 4 entries were fabricated, **every entry should be independently verified before publication**. The spot-checks above covered the most suspicious entries, but the EMA/PMDA/TGA approval dates were NOT individually verified against official agency databases (EPARs, PMDA approval notices, TGA ARTG) due to access limitations.

For a Nature Communications submission, I recommend:

1. Cross-reference all DrugBank IDs against the official DrugBank database (free academic access)
2. Verify all approval dates against primary sources (FDA Drugs@FDA, EMA EPARs, PMDA approval list)
3. Have a domain expert (pharmacologist or regulatory affairs specialist) review the final list
4. Include the curation methodology and verification steps in the Supplementary Methods
