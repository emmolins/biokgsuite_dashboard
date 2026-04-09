"""Relation identifiers and edge filtering utilities for KGs.

Defines relation specifications for different KGs and provides utilities
for filtering edges by relation type, normalizing disease IDs, and
extracting entity pairs for analysis.
"""

import pandas as pd


# Relation identifiers per KG for each drug-repurposing relation type
KG_RELATIONS = {
    'primekg': {
        'drug_target':  {'relation': 'drug_protein', 'display_relation': 'target'},
        'drug_disease': {'relation': 'indication'},
        'drug_drug':    {'relation': 'drug_drug'},
    },
    'hetionet': {
        'drug_target':  {'relation': 'CbG'},   # Compound binds Gene
        'drug_disease': {'relation': 'CtD'},   # Compound treats Disease
        'drug_drug':    None,                  # CrC = structural similarity, not DDI
    },
    'drkg': {
        'drug_target':  {'relation': 'DRUGBANK::target::Compound:Gene'},
        'drug_disease': {'relation': 'DRUGBANK::treats::Compound:Disease'},
        'drug_drug':    {'relation': 'DRUGBANK::ddi-interactor-in::Compound:Compound'},
    },
    'openbilink': {
        'drug_target':  {'relations': ['DRUG_BINDACT_GENE', 'DRUG_BINDING_GENE']},
        'drug_disease': {'relation': 'DIS_DRUG'},
        'drug_drug':    None,                  # no clinical DDI edges in OpenBioLink
    },
    'matrix': {
        'drug_target':  {'relation': 'binds'},   # Biolink predicate: binds
        'drug_disease': {'relations': ['treats', 'treats_or_applied_or_studied_to_treat']},  # Biolink predicates
        'drug_drug':    None,                  # Special handling: all drug-drug edges regardless of predicate
    },
    'biokg': {
        'drug_target':  {'relation': 'DPI'},                    # aggregated drug–protein interactions
        'drug_disease': {'relation': 'DRUG_DISEASE_ASSOCIATION'},
        'drug_drug':    {'relation': 'DDI'},
    },
}


def get_edges(kg, rel_spec):
    """Filter edges by relation (and optional display_relation).

    rel_spec may contain either 'relation' (single string) or 'relations' (list of strings).
    """
    if 'relations' in rel_spec:
        mask = kg['relation'].isin(rel_spec['relations'])
    else:
        mask = kg['relation'] == rel_spec['relation']
    if 'display_relation' in rel_spec:
        mask &= kg['display_relation'] == rel_spec['display_relation']
    return kg[mask]


def _strip_prefix(id_str):
    """Strip Hetionet/DRKG '::' prefix  e.g. 'Compound::DB00001' -> 'DB00001'."""
    s = str(id_str).strip()
    return s.split('::')[-1] if '::' in s else s


def _disease_id_to_mondo(id_str, doid_to_mondo, mesh_to_doid):
    """DOID:*, MESH:*, MONDO:*, or bare MeSH ID -> MONDO numeric (for Open Targets comparison)."""
    s = str(id_str).strip()
    if s.startswith('DOID:'):
        return doid_to_mondo.get(s.replace('DOID:', '').lstrip('0'))
    if s.startswith('MONDO:'):
        # MATRIX: MONDO CURIE -- strip prefix to get MONDO numeric
        return s.replace('MONDO:', '').lstrip('0')
    if s.startswith('MESH:'):
        doid_num = mesh_to_doid.get(s)
        return doid_to_mondo.get(doid_num) if doid_num else None
    # BioKG: bare MeSH IDs (D* or C*) — try bridging via MESH: prefix
    if s and (s[0] in ('D', 'C')) and any(c.isdigit() for c in s):
        doid_num = mesh_to_doid.get(f'MESH:{s}')
        return doid_to_mondo.get(doid_num) if doid_num else None
    return None


def _openbilink_drug_to_db_id(drug_id, obl_db_lookup):
    """Map an OpenBioLink PUBCHEM.COMPOUND:* drug ID to a bare DrugBank accession.

    Uses the pre-built PubChem→DrugBank mapping.
    Returns None if no DrugBank ID can be determined.
    """
    return obl_db_lookup.get(str(drug_id))


def build_openbilink_drug_lookup(obl_nodes_df):
    """Build Drug → DrugBank ID lookup from OpenBioLink nodes (PubChem→DrugBank).

    Parameters
    ----------
    obl_nodes_df : pd.DataFrame
        OpenBioLink node table with columns id, type, drugbank_id.

    Returns
    -------
    dict
        Mapping from PUBCHEM.COMPOUND:* IDs to bare DrugBank accessions.
    """
    if obl_nodes_df.empty or 'drugbank_id' not in obl_nodes_df.columns:
        return {}
    drug_rows = obl_nodes_df[
        (obl_nodes_df['type'] == 'Drug') & (obl_nodes_df['drugbank_id'] != '')
    ][['id', 'drugbank_id']]
    return dict(zip(drug_rows['id'].astype(str), drug_rows['drugbank_id'].astype(str)))


def extract_pairs(kg, rel_spec, doid_to_mondo=None, mesh_to_doid=None,
                  normalise_disease=False, kg_name=None,
                  x_type_filter=None, y_type_filter=None):
    """Return (x_id, y_id) pairs for a relation, stripping entity-type prefixes.

    Parameters
    ----------
    kg : pd.DataFrame
        Edge list with columns x_id, y_id, x_type, y_type, relation.
    rel_spec : dict
        Relation specification (from KG_RELATIONS).
    doid_to_mondo : dict, optional
        DOID to MONDO mapping (required if normalise_disease=True).
    mesh_to_doid : dict, optional
        MESH to DOID mapping (required if normalise_disease=True).
    normalise_disease : bool
        If True, y-side disease IDs are mapped to MONDO numeric.
    kg_name : str, optional
        Name of KG (primekg, hetionet, drkg, matrix) for disease normalization.
    x_type_filter : str, optional
        Filter rows where x_type equals the given string.
    y_type_filter : str, optional
        Filter rows where y_type equals the given string.

    Returns
    -------
    set of (x_id, y_id) tuples
    """
    edges = get_edges(kg, rel_spec)
    if x_type_filter is not None:
        edges = edges[edges['x_type'] == x_type_filter]
    if y_type_filter is not None:
        edges = edges[edges['y_type'] == y_type_filter]
    x_ids = edges['x_id'].astype(str).map(_strip_prefix)
    y_ids = edges['y_id'].astype(str).map(_strip_prefix)
    if normalise_disease and kg_name in ('hetionet', 'drkg', 'openbilink', 'matrix', 'biokg'):
        y_ids = y_ids.map(lambda id_: _disease_id_to_mondo(id_, doid_to_mondo, mesh_to_doid))
    mask = x_ids.notna() & y_ids.notna()
    return set(zip(x_ids[mask], y_ids[mask]))


# ── MATRIX drug-ID normalisation lookups ─────────────────────────────────────
# MATRIX edges store drug primary IDs (PUBCHEM.COMPOUND:*, CHEBI:*, UNII:*, etc.)
# rather than DrugBank IDs.  load_matrix() extracts DrugBank accessions from
# the equivalent_identifiers column.
# _matrix_db_lookup      : primary_id -> first DrugBank accession (backward-compat)
# _matrix_db_lookup_multi: primary_id -> list of ALL DrugBank accessions
def build_matrix_drug_lookups(matrix_nodes_df):
    """Build drug → DrugBank ID lookup tables from MATRIX nodes."""
    _matrix_db_lookup = {}
    _matrix_db_lookup_multi = {}
    if matrix_nodes_df.empty:
        return _matrix_db_lookup, _matrix_db_lookup_multi

    if 'drugbank_ids_all' in matrix_nodes_df.columns:
        # Filter to drug nodes with at least one DrugBank accession (fast vectorised path)
        _drug_db_rows = matrix_nodes_df[
            (matrix_nodes_df['type'] == 'drug') & (matrix_nodes_df['drugbank_ids_all'] != '')
        ][['id', 'drugbank_ids_all']]
        for pid, ids_str in zip(_drug_db_rows['id'].astype(str),
                                _drug_db_rows['drugbank_ids_all'].astype(str)):
            all_ids = [x for x in ids_str.split('|') if x]
            if all_ids:
                _matrix_db_lookup_multi[pid] = all_ids
                _matrix_db_lookup[pid] = all_ids[0]
    elif 'drugbank_id' in matrix_nodes_df.columns:
        _drug_rows = matrix_nodes_df[
            (matrix_nodes_df['type'] == 'drug') & (matrix_nodes_df['drugbank_id'] != '')
        ][['id', 'drugbank_id']]
        _matrix_db_lookup = dict(zip(_drug_rows['id'].astype(str),
                                      _drug_rows['drugbank_id'].astype(str)))
        _matrix_db_lookup_multi = {k: [v] for k, v in _matrix_db_lookup.items()}

    return _matrix_db_lookup, _matrix_db_lookup_multi


def build_matrix_gene_lookup(matrix_nodes_df):
    """Build gene/protein → NCBIGene ID lookup from MATRIX nodes."""
    _matrix_gene_equiv_lookup = {}
    if not matrix_nodes_df.empty and 'ncbigene_id' in matrix_nodes_df.columns:
        _gene_rows = matrix_nodes_df[
            (matrix_nodes_df['type'] == 'gene/protein') & (matrix_nodes_df['ncbigene_id'] != '')
        ][['id', 'ncbigene_id']]
        _matrix_gene_equiv_lookup = dict(zip(_gene_rows['id'].astype(str),
                                              _gene_rows['ncbigene_id'].astype(str)))
    return _matrix_gene_equiv_lookup


def _matrix_drug_to_db_id(drug_id, _matrix_db_lookup):
    """Map a MATRIX primary drug node ID to a bare DrugBank accession (first match).

    Tries the pre-built lookup first (covers PUBCHEM/CHEBI/UNII primary IDs whose
    DrugBank equivalent was extracted from the equivalent_identifiers column).
    Falls back to stripping the 'DRUGBANK:' prefix for nodes whose primary ID is
    already a DrugBank CURIE.
    Returns None if no DrugBank ID can be determined.
    """
    db = _matrix_db_lookup.get(str(drug_id))
    if db:
        return db
    s = str(drug_id)
    if s.startswith('DRUGBANK:'):
        return s[len('DRUGBANK:'):]
    return None


def _matrix_drug_to_db_ids(drug_id, _matrix_db_lookup_multi):
    """Map a MATRIX primary drug node ID to ALL bare DrugBank accessions (list).

    Returns an empty list if no DrugBank ID can be determined.
    Handles nodes with multiple DrugBank accessions in equivalent_identifiers.
    """
    ids = _matrix_db_lookup_multi.get(str(drug_id))
    if ids:
        return ids
    s = str(drug_id)
    if s.startswith('DRUGBANK:'):
        return [s[len('DRUGBANK:'):]]
    return []


def _matrix_gene_to_entrez(gene_id, _matrix_gene_equiv_lookup, uniprot_to_entrez=None):
    """Map a MATRIX gene node ID to an Entrez ID string (or '' if not resolvable).

    Handles NCBIGene: CURIEs directly, UniProtKB: via uniprot_to_entrez,
    and other prefixes (PR:, GeneFamily, etc.) via _matrix_gene_equiv_lookup.
    """
    if uniprot_to_entrez is None:
        uniprot_to_entrez = {}
    s = str(gene_id)
    if s.startswith('NCBIGene:'):
        return s[len('NCBIGene:'):]
    if s.startswith('UniProtKB:'):
        return uniprot_to_entrez.get(s[len('UniProtKB:'):], '')
    # Fallback: try equivalent_identifiers-derived NCBIGene ID
    return _matrix_gene_equiv_lookup.get(s, '')
