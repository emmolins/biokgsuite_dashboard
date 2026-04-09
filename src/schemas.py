"""Knowledge Graph schema definitions for type conformance validation.

Defines expected entity type pairs for each relation in PrimeKG, Hetionet,
DRKG, and MATRIX knowledge graphs, and provides utilities for schema validation.
"""

# PrimeKG: declared (x_type, y_type) pairs per relation
PRIMEKG_SCHEMA = {
    'drug_protein':               {('drug','gene/protein'), ('gene/protein','drug')},
    'indication':                 {('drug','disease'), ('disease','drug')},
    'contraindication':           {('drug','disease'), ('disease','drug')},
    'off-label use':              {('drug','disease'), ('disease','drug')},
    'drug_drug':                  {('drug','drug')},
    'drug_effect':                {('drug','effect/phenotype'), ('effect/phenotype','drug')},
    'disease_protein':            {('disease','gene/protein'), ('gene/protein','disease')},
    'disease_phenotype_positive': {('disease','effect/phenotype'), ('effect/phenotype','disease')},
    'disease_phenotype_negative': {('disease','effect/phenotype'), ('effect/phenotype','disease')},
    'protein_protein':            {('gene/protein','gene/protein')},
    'bioprocess_protein':         {('biological_process','gene/protein'), ('gene/protein','biological_process')},
    'molfunc_protein':            {('molecular_function','gene/protein'), ('gene/protein','molecular_function')},
    'cellcomp_protein':           {('cellular_component','gene/protein'), ('gene/protein','cellular_component')},
    'exposure_protein':           {('exposure','gene/protein'), ('gene/protein','exposure')},
    'exposure_disease':           {('exposure','disease'), ('disease','exposure')},
    'exposure_bioprocess':        {('exposure','biological_process'), ('biological_process','exposure')},
    'exposure_molfunc':           {('exposure','molecular_function'), ('molecular_function','exposure')},
    'exposure_cellcomp':          {('exposure','cellular_component'), ('cellular_component','exposure')},
    'anatomy_protein_present':    {('anatomy','gene/protein'), ('gene/protein','anatomy')},
    'anatomy_protein_absent':     {('anatomy','gene/protein'), ('gene/protein','anatomy')},
    'disease_disease':            {('disease','disease')},
    'anatomy_anatomy':            {('anatomy','anatomy')},
    'pathway_protein':            {('pathway','gene/protein'), ('gene/protein','pathway')},
    'pathway_pathway':            {('pathway','pathway')},
    'phenotype_protein':          {('effect/phenotype','gene/protein'), ('gene/protein','effect/phenotype')},
    'phenotype_phenotype':        {('effect/phenotype','effect/phenotype')},
    'bioprocess_bioprocess':      {('biological_process','biological_process')},
    'molfunc_molfunc':            {('molecular_function','molecular_function')},
    'cellcomp_cellcomp':          {('cellular_component','cellular_component')},
    'exposure_exposure':          {('exposure','exposure')},
}

# Hetionet: declared schema based on known metaedge definitions
HETIONET_SCHEMA = {
    'CbG':  {('Compound','Gene'),               ('Gene','Compound')},
    'CcSE': {('Compound','Side Effect'),         ('Side Effect','Compound')},
    'CdG':  {('Compound','Gene'),               ('Gene','Compound')},
    'CpD':  {('Compound','disease'),            ('disease','Compound')},
    'CrC':  {('Compound','Compound')},
    'CtD':  {('Compound','disease'),            ('disease','Compound')},
    'CuG':  {('Compound','Gene'),               ('Gene','Compound')},
    'DaG':  {('disease','Gene'),                ('Gene','disease')},
    'DdG':  {('disease','Gene'),                ('Gene','disease')},
    'DlA':  {('disease','Anatomy'),             ('Anatomy','disease')},
    'DpS':  {('disease','Symptom'),             ('Symptom','disease')},
    'DrD':  {('disease','disease')},
    'DuG':  {('disease','Gene'),                ('Gene','disease')},
    'GcG':  {('Gene','Gene')},
    'GiG':  {('Gene','Gene')},
    'GpBP': {('Gene','Biological Process'),      ('Biological Process','Gene')},
    'GpCC': {('Gene','Cellular Component'),      ('Cellular Component','Gene')},
    'GpMF': {('Gene','Molecular Function'),      ('Molecular Function','Gene')},
    'GpPW': {('Gene','Pathway'),                ('Pathway','Gene')},
    'GrG':  {('Gene','Gene')},
    'Gr>G': {('Gene','Gene')},
    'PCiC': {('Pharmacologic Class','Compound'), ('Compound','Pharmacologic Class')},
    'AeG':  {('Anatomy','Gene'),                ('Gene','Anatomy')},
    'AuG':  {('Anatomy','Gene'),                ('Gene','Anatomy')},
    'AdG':  {('Anatomy','Gene'),                ('Gene','Anatomy')},
}


def drkg_expected_types(rel_name):
    """Auto-derive allowed (x_type, y_type) pairs from DRKG relation name.

    DRKG format: SOURCE::rel::XType:YType — the type pair is always the last segment.
    Returns a set of allowed pairs (both directions), or None if format is unrecognised.

    Parameters
    ----------
    rel_name : str
        DRKG relation name in format SOURCE::rel::XType:YType.

    Returns
    -------
    set of tuples or None
        Set of allowed (x_type, y_type) pairs, or None if format unrecognized.
    """
    parts = rel_name.split('::')
    if len(parts) >= 3 and ':' in parts[-1]:
        x, y = parts[-1].split(':', 1)
        return {(x, y), (y, x)} if x != y else {(x, y)}
    return None


# MATRIX: schema based on actual biolink predicates used in MATRIX KG.
# Uses canonical types assigned by load_matrix() (drug, disease, gene/protein,
# pathway, anatomy, effect/phenotype, biological_process, molecular_function,
# cellular_component). Predicates absent from this dict are skipped; only those
# listed here are checked for type conformance.
MATRIX_SCHEMA = {
    # ── Drug → Disease ────────────────────────────────────────────────────────
    'treats': {
        ('drug', 'disease'), ('disease', 'drug')},
    'treats_or_applied_or_studied_to_treat': {
        ('drug', 'disease'), ('disease', 'drug')},
    'applied_to_treat': {
        ('drug', 'disease'), ('disease', 'drug')},
    'ameliorates_condition': {
        ('drug', 'disease'), ('disease', 'drug')},
    'contraindicated_in': {
        ('drug', 'disease'), ('disease', 'drug')},
    'prevented_by': {
        ('disease', 'drug'), ('drug', 'disease')},
    'prevents': {
        ('drug', 'disease'), ('disease', 'drug')},
    'clinically_tested_for': {
        ('drug', 'disease'), ('disease', 'drug')},
    'has_not_completed_clinical_trials_for': {
        ('drug', 'disease'), ('disease', 'drug')},
    'causes_adverse_event': {
        ('drug', 'disease'),          ('disease', 'drug'),
        ('drug', 'effect/phenotype'), ('effect/phenotype', 'drug')},
    # ── Drug → Gene/Protein ───────────────────────────────────────────────────
    'affects': {
        ('drug', 'gene/protein'),         ('gene/protein', 'drug'),
        ('drug', 'biological_process'),   ('biological_process', 'drug'),
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein')},
    'has_target': {
        ('drug', 'gene/protein'), ('gene/protein', 'drug')},
    'increases_synthesis_of': {
        ('drug', 'gene/protein'),    ('gene/protein', 'drug'),
        ('gene/protein', 'gene/protein')},
    'decreases_synthesis_of': {
        ('drug', 'gene/protein'),    ('gene/protein', 'drug'),
        ('gene/protein', 'gene/protein')},
    'increases_transport_of': {
        ('drug', 'gene/protein'),    ('gene/protein', 'drug'),
        ('gene/protein', 'gene/protein')},
    'decreases_transport_of': {
        ('drug', 'gene/protein'),    ('gene/protein', 'drug'),
        ('gene/protein', 'gene/protein')},
    'increases_uptake_of': {
        ('drug', 'gene/protein'),    ('gene/protein', 'drug'),
        ('gene/protein', 'gene/protein')},
    'decreases_uptake_of': {
        ('drug', 'gene/protein'),    ('gene/protein', 'drug'),
        ('gene/protein', 'gene/protein')},
    'disrupts': {
        ('drug', 'gene/protein'),       ('gene/protein', 'drug'),
        ('drug', 'biological_process'), ('biological_process', 'drug')},
    # ── Drug–Drug / Gene–Gene interaction ────────────────────────────────────
    # interacts_with covers both DDI and PPI; drug→gene/protein also allowed in MATRIX
    'interacts_with': {
        ('drug', 'drug'),
        ('gene/protein', 'gene/protein'),
        ('drug', 'gene/protein'), ('gene/protein', 'drug')},
    # ── Gene/Protein regulation ───────────────────────────────────────────────
    'positively_regulates': {
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein'),
        ('gene/protein', 'molecular_function'),
        ('biological_process', 'biological_process')},
    'negatively_regulates': {
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein'),
        ('gene/protein', 'molecular_function'),
        ('biological_process', 'biological_process')},
    'directly_positively_regulates': {
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein')},
    'in_complex_with': {
        ('gene/protein', 'gene/protein')},
    'in_pathway_with': {
        ('gene/protein', 'gene/protein')},
    'homologous_to': {
        ('gene/protein', 'gene/protein'),
        ('disease', 'disease')},
    'has_gene_product': {
        ('gene/protein', 'gene/protein')},
    'gene_product_of': {
        ('gene/protein', 'gene/protein')},
    'coexists_with': {
        ('gene/protein', 'gene/protein'),
        ('disease', 'disease'),
        ('gene/protein', 'disease'), ('disease', 'gene/protein')},
    # ── Gene/Protein → Disease ────────────────────────────────────────────────
    'gene_associated_with_condition': {
        ('gene/protein', 'disease'), ('disease', 'gene/protein')},
    'disease_has_basis_in': {
        ('disease', 'gene/protein'), ('gene/protein', 'disease')},
    'biomarker_for': {
        ('gene/protein', 'disease'), ('disease', 'gene/protein')},
    'contributes_to': {
        ('gene/protein', 'disease'),          ('disease', 'gene/protein'),
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein')},
    'causes': {
        ('gene/protein', 'disease'), ('disease', 'gene/protein'),
        ('disease', 'disease')},
    'complicates': {
        ('disease', 'disease'),
        ('disease', 'gene/protein'), ('gene/protein', 'disease')},
    # ── Disease → Phenotype ───────────────────────────────────────────────────
    'has_phenotype': {
        ('disease', 'effect/phenotype'),      ('effect/phenotype', 'disease'),
        ('gene/protein', 'effect/phenotype'), ('effect/phenotype', 'gene/protein')},
    'manifestation_of': {
        ('effect/phenotype', 'disease'), ('disease', 'effect/phenotype')},
    'has_mode_of_inheritance': {
        ('disease', 'gene/protein'), ('gene/protein', 'disease')},
    # ── Gene/Protein → Anatomy / GO terms ────────────────────────────────────
    'expressed_in': {
        ('gene/protein', 'anatomy'),            ('anatomy', 'gene/protein'),
        ('gene/protein', 'cellular_component'), ('cellular_component', 'gene/protein')},
    'expresses': {
        ('anatomy', 'gene/protein'),  ('gene/protein', 'anatomy')},
    'located_in': {
        ('gene/protein', 'cellular_component'), ('cellular_component', 'gene/protein'),
        ('gene/protein', 'anatomy'),            ('anatomy', 'gene/protein'),
        ('biological_process', 'cellular_component')},
    'occurs_in': {
        ('biological_process', 'cellular_component'), ('cellular_component', 'biological_process'),
        ('biological_process', 'anatomy'),             ('anatomy', 'biological_process')},
    'has_participant': {
        ('pathway', 'gene/protein'),            ('gene/protein', 'pathway'),
        ('biological_process', 'gene/protein'), ('gene/protein', 'biological_process')},
    'capable_of': {
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein'),
        ('gene/protein', 'molecular_function')},
    # ── Correlation ───────────────────────────────────────────────────────────
    'positively_correlated_with': {
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'disease'),  ('disease', 'gene/protein'),
        ('drug', 'gene/protein'),     ('gene/protein', 'drug'),
        ('drug', 'disease'),          ('disease', 'drug')},
    'negatively_correlated_with': {
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'disease'),  ('disease', 'gene/protein'),
        ('drug', 'gene/protein'),     ('gene/protein', 'drug'),
        ('drug', 'disease'),          ('disease', 'drug')},
    'correlated_with': {
        ('gene/protein', 'gene/protein'),
        ('gene/protein', 'disease'),  ('disease', 'gene/protein'),
        ('drug', 'gene/protein'),     ('gene/protein', 'drug')},
    # ── Hierarchical / structural relations ───────────────────────────────────
    'subclass_of': {
        ('disease', 'disease'),
        ('biological_process', 'biological_process'),
        ('molecular_function', 'molecular_function'),
        ('cellular_component', 'cellular_component'),
        ('effect/phenotype', 'effect/phenotype'),
        ('anatomy', 'anatomy'),
        ('pathway', 'pathway'),
        ('drug', 'drug')},
    'superclass_of': {
        ('disease', 'disease'),
        ('biological_process', 'biological_process'),
        ('molecular_function', 'molecular_function'),
        ('cellular_component', 'cellular_component'),
        ('effect/phenotype', 'effect/phenotype'),
        ('anatomy', 'anatomy')},
    'part_of': {
        ('anatomy', 'anatomy'),
        ('cellular_component', 'cellular_component'),
        ('biological_process', 'biological_process'),
        ('pathway', 'pathway'),
        ('gene/protein', 'gene/protein'),
        ('disease', 'disease')},
    'overlaps_with': {
        ('anatomy', 'anatomy'),
        ('biological_process', 'biological_process'),
        ('pathway', 'pathway')},
    'derives_from': {
        ('drug', 'drug'),
        ('gene/protein', 'gene/protein'),
        ('disease', 'disease')},
    'similar_to': {
        ('drug', 'drug'),
        ('disease', 'disease'),
        ('gene/protein', 'gene/protein')},
    'same_as': {
        ('drug', 'drug'),        ('disease', 'disease'),
        ('gene/protein', 'gene/protein'),
        ('pathway', 'pathway'),  ('anatomy', 'anatomy')},
    'related_to': {
        ('gene/protein', 'gene/protein'),
        ('disease', 'disease'),
        ('drug', 'drug'),
        ('drug', 'disease'),         ('disease', 'drug'),
        ('gene/protein', 'disease'), ('disease', 'gene/protein'),
        ('drug', 'gene/protein'),    ('gene/protein', 'drug')},
    'associated_with': {
        ('disease', 'gene/protein'),  ('gene/protein', 'disease'),
        ('disease', 'pathway'),       ('pathway', 'disease'),
        ('gene/protein', 'pathway'),  ('pathway', 'gene/protein'),
        ('drug', 'disease'),          ('disease', 'drug'),
        ('gene/protein', 'gene/protein'),
        ('drug', 'gene/protein'),     ('gene/protein', 'drug'),
        ('drug', 'drug'),
        ('disease', 'disease'),
        ('drug', 'biological_process'),   ('biological_process', 'drug'),
        ('gene/protein', 'biological_process'), ('biological_process', 'gene/protein')},
    # ── Temporal ──────────────────────────────────────────────────────────────
    'preceded_by': {
        ('biological_process', 'biological_process'),
        ('disease', 'disease')},
    'precedes': {
        ('biological_process', 'biological_process'),
        ('disease', 'disease')},
    # ── Abundance changes ─────────────────────────────────────────────────────
    'has_increased_amount': {
        ('disease', 'gene/protein'), ('gene/protein', 'disease')},
    'has_decreased_amount': {
        ('disease', 'gene/protein'), ('gene/protein', 'disease')},
    # ── Drug properties ───────────────────────────────────────────────────────
    'has_route_of_administration': {
        ('drug', 'drug')},
}

# OpenBioLink: schema derived from relation names.
# Relation names encode the entity type pair, e.g. DRUG_BINDING_GENE → (Drug, Gene).
# All declared pairs allow both orientations (undirected).
OPENBILINK_SCHEMA = {
    # ── Gene–Gene ──────────────────────────────────────────────────────────────
    'GENE_GENE':                {('Gene', 'Gene')},
    'GENE_BINDING_GENE':        {('Gene', 'Gene')},
    'GENE_REACTION_GENE':       {('Gene', 'Gene')},
    'GENE_CATALYSIS_GENE':      {('Gene', 'Gene')},
    'GENE_ACTIVATION_GENE':     {('Gene', 'Gene')},
    'GENE_INHIBITION_GENE':     {('Gene', 'Gene')},
    'GENE_PTMOD_GENE':          {('Gene', 'Gene')},
    'GENE_EXPRESSION_GENE':     {('Gene', 'Gene')},
    'GENE_BINDACT_GENE':        {('Gene', 'Gene')},
    'GENE_BINDINH_GENE':        {('Gene', 'Gene')},
    # ── Drug–Gene ──────────────────────────────────────────────────────────────
    'GENE_DRUG':                {('Gene', 'Drug'), ('Drug', 'Gene')},
    'DRUG_BINDING_GENE':        {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_PREDBIND_GENE':       {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_ACTIVATION_GENE':     {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_INHIBITION_GENE':     {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_EXPRESSION_GENE':     {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_CATALYSIS_GENE':      {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_REACTION_GENE':       {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_BINDACT_GENE':        {('Drug', 'Gene'), ('Gene', 'Drug')},
    'DRUG_BINDINH_GENE':        {('Drug', 'Gene'), ('Gene', 'Drug')},
    # ── Gene–Anatomy/Cell ──────────────────────────────────────────────────────
    'GENE_EXPRESSED_ANATOMY':   {('Gene', 'Anatomy'), ('Anatomy', 'Gene'),
                                  ('Gene', 'Cell'),    ('Cell', 'Gene')},
    'GENE_OVEREXPRESSED_ANATOMY': {('Gene', 'Anatomy'), ('Anatomy', 'Gene'),
                                    ('Gene', 'Cell'),    ('Cell', 'Gene')},
    'GENE_UNDEREXPRESSED_ANATOMY': {('Gene', 'Anatomy'), ('Anatomy', 'Gene'),
                                     ('Gene', 'Cell'),    ('Cell', 'Gene')},
    # ── Gene–GO ────────────────────────────────────────────────────────────────
    'GENE_GO':                  {('Gene', 'GO'), ('GO', 'Gene')},
    # ── Gene–Pathway ───────────────────────────────────────────────────────────
    'GENE_PATHWAY':             {('Gene', 'Pathway'), ('Pathway', 'Gene')},
    # ── Gene–Phenotype ─────────────────────────────────────────────────────────
    'GENE_PHENOTYPE':           {('Gene', 'Phenotype'), ('Phenotype', 'Gene')},
    # ── Gene–Disease ───────────────────────────────────────────────────────────
    'GENE_DIS':                 {('Gene', 'Disease'), ('Disease', 'Gene')},
    # ── Drug–Phenotype ─────────────────────────────────────────────────────────
    'DRUG_PHENOTYPE':           {('Drug', 'Phenotype'), ('Phenotype', 'Drug')},
    # ── Disease–Drug ───────────────────────────────────────────────────────────
    'DIS_DRUG':                 {('Disease', 'Drug'), ('Drug', 'Disease')},
    # ── Disease–Phenotype ──────────────────────────────────────────────────────
    'DIS_PHENOTYPE':            {('Disease', 'Phenotype'), ('Phenotype', 'Disease')},
    # ── Ontology hierarchy ─────────────────────────────────────────────────────
    'IS_A':                     {
        ('Disease', 'Disease'), ('GO', 'GO'), ('Anatomy', 'Anatomy'),
        ('Cell', 'Cell'), ('Phenotype', 'Phenotype'),
    },
    'PART_OF':                  {
        ('Anatomy', 'Anatomy'), ('Cell', 'Anatomy'), ('Anatomy', 'Cell'),
        ('GO', 'GO'),
    },
}


# BioKG: schema based on predicate → entity type pairs.
# Entity types are assigned by load_biokg() using the _PRED_TYPES mapping.
BIOKG_SCHEMA = {
    # ── Protein–Protein ──────────────────────────────────────────────────────
    'PPI':                         {('Gene/Protein', 'Gene/Protein')},
    # ── Drug–Protein ─────────────────────────────────────────────────────────
    'DPI':                         {('Drug', 'Gene/Protein'), ('Gene/Protein', 'Drug')},
    'DRUG_TARGET':                 {('Drug', 'Gene/Protein'), ('Gene/Protein', 'Drug')},
    'DRUG_CARRIER':                {('Drug', 'Gene/Protein'), ('Gene/Protein', 'Drug')},
    'DRUG_ENZYME':                 {('Drug', 'Gene/Protein'), ('Gene/Protein', 'Drug')},
    'DRUG_TRANSPORTER':            {('Drug', 'Gene/Protein'), ('Gene/Protein', 'Drug')},
    # ── Drug–Drug ────────────────────────────────────────────────────────────
    'DDI':                         {('Drug', 'Drug')},
    # ── Protein–Disease ──────────────────────────────────────────────────────
    'PROTEIN_DISEASE_ASSOCIATION': {('Gene/Protein', 'Disease'), ('Disease', 'Gene/Protein')},
    # ── Drug–Disease ─────────────────────────────────────────────────────────
    'DRUG_DISEASE_ASSOCIATION':    {('Drug', 'Disease'), ('Disease', 'Drug')},
    # ── Protein–Pathway ──────────────────────────────────────────────────────
    'PROTEIN_PATHWAY_ASSOCIATION': {('Gene/Protein', 'Pathway'), ('Pathway', 'Gene/Protein')},
    # ── Drug–Pathway ─────────────────────────────────────────────────────────
    'DRUG_PATHWAY_ASSOCIATION':    {('Drug', 'Pathway'), ('Pathway', 'Drug')},
    # ── Disease–Pathway ──────────────────────────────────────────────────────
    'DISEASE_PATHWAY_ASSOCIATION': {('Disease', 'Pathway'), ('Pathway', 'Disease')},
    # ── Complex membership ───────────────────────────────────────────────────
    'MEMBER_OF_COMPLEX':           {('Gene/Protein', 'Complex'), ('Complex', 'Gene/Protein')},
    'MEMBER_OF_PATHWAY':           {('Complex', 'Pathway'), ('Pathway', 'Complex')},
    'MEMBER_OF_TOP_LEVEL_PATHWAY': {('Complex', 'Pathway'), ('Pathway', 'Complex')},
    # ── Complex–Pathway ──────────────────────────────────────────────────────
    'COMPLEX_IN_PATHWAY':          {('Complex', 'Pathway'), ('Pathway', 'Complex')},
    'COMPLEX_TOP_LEVEL_PATHWAY':   {('Complex', 'Pathway'), ('Pathway', 'Complex')},
    # ── Pathway hierarchy ────────────────────────────────────────────────────
    'HAS_PARENT_PATHWAY':          {('Pathway', 'Pathway')},
    # ── Disease–GeneticDisorder ──────────────────────────────────────────────
    'DISEASE_GENETIC_DISORDER':    {('Disease', 'GeneticDisorder'), ('GeneticDisorder', 'Disease')},
    'RELATED_GENETIC_DISORDER':    {('Gene/Protein', 'GeneticDisorder'), ('GeneticDisorder', 'Gene/Protein')},
    # ── Additional link types ────────────────────────────────────────────────
    'DISEASE_SUPERGRP':            {('Disease', 'DiseaseCategory'), ('DiseaseCategory', 'Disease')},
    'DRUG_SIDEEFFECT_ASSOCIATION': {('Drug', 'SideEffect'), ('SideEffect', 'Drug')},
    'DRUG_INDICATION_ASSOCIATION': {('Drug', 'Disease'), ('Disease', 'Drug')},
    'DRUG_ATC_CODE':               {('Drug', 'ATC'), ('ATC', 'Drug')},
    'PROTEIN_EXPRESSED_IN':        {('Gene/Protein', 'Tissue'), ('Tissue', 'Gene/Protein')},
    'PART_OF_TISSUE':              {('Cell', 'Tissue'), ('Tissue', 'Cell')},
}

KG_SCHEMAS = {
    'primekg':    PRIMEKG_SCHEMA,
    'hetionet':   HETIONET_SCHEMA,
    'drkg':       None,   # auto-derived from relation name
    'openbilink': OPENBILINK_SCHEMA,
    'matrix':     MATRIX_SCHEMA,
    'biokg':      BIOKG_SCHEMA,
}
