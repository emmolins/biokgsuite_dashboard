"""Entity coverage and normalization utilities for Knowledge Graph evaluation.

Provides helpers for extracting entity sets from KGs, normalizing disease IDs
across ontologies (DOID, MESH, MONDO), and computing coverage metrics.
"""

import pandas as pd


def _as_str_set(series):
    """Series -> normalised string set (strips whitespace, drops nulls/'')."""
    return set(series.dropna().astype(str).str.strip()) - {''}


def _strip_prefix(id_str):
    """Strip Hetionet/DRKG '::' prefix  e.g. 'Compound::DB00001' -> 'DB00001'."""
    s = str(id_str).strip()
    return s.split('::')[-1] if '::' in s else s


def entity_set_from_kg(kg, entity_type):
    """Unique node IDs for entity_type from a BioKGBench-schema edge list."""
    raw = _as_str_set(pd.concat([
        kg.loc[kg['x_type'] == entity_type, 'x_id'],
        kg.loc[kg['y_type'] == entity_type, 'y_id'],
    ], ignore_index=True))
    return {_strip_prefix(i) for i in raw}


def entity_set_from_nodes(nodes_df, entity_type):
    """Unique node IDs for entity_type directly from the node table.

    Preferred over entity_set_from_kg for KGs such as MATRIX where many
    valid nodes are absent from the loaded edge table (because edge partners
    are missing from the node file and those edges are dropped on load).
    """
    return _as_str_set(nodes_df.loc[nodes_df['type'] == entity_type, 'id'])


def calc_coverage(kg_set, gold_set):
    """Coverage = |KG intersect Gold| / |Gold|."""
    matched = kg_set & gold_set
    n_gold  = len(gold_set)
    return {
        'kg_n':         len(kg_set),
        'gold_n':       n_gold,
        'overlap_n':    len(matched),
        'coverage_pct': round(100 * len(matched) / n_gold, 2) if n_gold else float('nan'),
        'matched':      matched,
        'unmatched':    gold_set - kg_set,
    }


# ── Entity-type name maps and disease normalisation ───────────────────────────
# KG entity-type names -> canonical labels.
# MATRIX already stores canonical types (drug, disease, gene/protein ...)
# after the biolink-category normalisation in load_matrix(), so its map is identity.
ENTITY_TYPE_MAP = {
    'primekg':    {'Drug': 'drug',     'Disease': 'disease',  'Gene/Protein': 'gene/protein', 'Pathway': 'pathway'},
    'hetionet':   {'Drug': 'Compound', 'Disease': 'Disease',  'Gene/Protein': 'Gene',         'Pathway': 'Pathway'},
    'drkg':       {'Drug': 'Compound', 'Disease': 'Disease',  'Gene/Protein': 'Gene',         'Pathway': 'Pathway'},
    'openbilink': {'Drug': 'Drug',     'Disease': 'Disease',  'Gene/Protein': 'Gene',         'Pathway': 'Pathway'},
    'matrix':     {'Drug': 'drug',     'Disease': 'disease',  'Gene/Protein': 'gene/protein', 'Pathway': 'pathway'},
    'biokg':      {'Drug': 'Drug',     'Disease': 'Disease',  'Gene/Protein': 'Gene/Protein', 'Pathway': 'Pathway'},
}


def get_openbilink_entity_set(kg, nodes_df, canonical_type):
    """Return normalised entity IDs for an OpenBioLink entity type.

    OpenBioLink stores entities with CURIE-style prefixes that must be stripped
    or remapped before comparison with gold standards:

    - Drug        PUBCHEM.COMPOUND:* -> DrugBank accession (DB*) via nodes_df['drugbank_id']
    - Gene        NCBIGENE:*         -> bare Entrez ID (strip prefix)
    - Pathway     REACTOME:* / KEGG:* -> bare pathway ID (strip prefix)
    - Disease     handled separately by get_disease_ids_doid (DOID: prefix)

    Parameters
    ----------
    kg : pd.DataFrame
        OpenBioLink edge table in BioKGBench schema.
    nodes_df : pd.DataFrame
        Node table returned by load_openbilink (must contain 'drugbank_id' column).
    canonical_type : str
        One of 'Drug', 'Gene/Protein', 'Pathway', 'Disease'.

    Returns
    -------
    set of str
    """
    # Internal type names used in OpenBioLink edge table
    _INTERNAL = {'Drug': 'Drug', 'Gene/Protein': 'Gene', 'Pathway': 'Pathway', 'Disease': 'Disease'}
    kg_type = _INTERNAL.get(canonical_type, canonical_type)
    raw = entity_set_from_kg(kg, kg_type)

    if canonical_type == 'Drug':
        # Build PubChem -> DrugBank mapping from the node table
        drug_mask = (nodes_df['type'] == 'Drug') & nodes_df['drugbank_id'].str.startswith('DB', na=False)
        pc_to_db  = dict(zip(nodes_df.loc[drug_mask, 'id'],
                             nodes_df.loc[drug_mask, 'drugbank_id']))
        return {pc_to_db[i] for i in raw if i in pc_to_db}

    if canonical_type == 'Gene/Protein':
        # NCBIGENE:1234 -> '1234'
        return {i.split(':', 1)[1] if i.startswith('NCBIGENE:') else i for i in raw}

    if canonical_type == 'Pathway':
        # REACTOME:R-HSA-* / KEGG:* -> strip namespace prefix
        return {i.split(':', 1)[1] if ':' in i else i for i in raw}

    # Disease or unknown - return as-is (disease handled by get_disease_ids_doid)
    return raw


def get_disease_ids_doid(kg, kg_name, mondo_to_doid, mesh_to_doid):
    """Return disease IDs normalised to DOID numeric strings.

    PrimeKG:  MONDO numeric -> DOID via cross-ref; compound IDs split first.
    Hetionet: DOID IDs directly.
    DRKG:     DOID entries directly; MESH entries bridged via mesh_to_doid.
    MATRIX:   MONDO CURIE ('MONDO:*') -> DOID via mondo_to_doid.
              MESH CURIE ('MESH:*') bridged via mesh_to_doid (~56% coverage).
    """
    raw = entity_set_from_kg(kg, ENTITY_TYPE_MAP[kg_name]['Disease'])
    if kg_name == 'primekg':
        return {
            mondo_to_doid[part.strip().lstrip('0')]
            for i in raw for part in str(i).split('_')
            if part.strip().lstrip('0') in mondo_to_doid
        }
    if kg_name in ('hetionet', 'openbilink'):
        return {i.replace('DOID:', '').lstrip('0') for i in raw if i.startswith('DOID:')}
    if kg_name == 'matrix':
        # MATRIX uses full CURIE IDs -- no bare IDs without namespace prefix.
        result = set()
        for i in raw:
            s = str(i).strip()
            if s.startswith('MONDO:'):
                num = s.replace('MONDO:', '').lstrip('0')
                doid = mondo_to_doid.get(num)
                if doid:
                    result.add(doid)
            elif s.startswith('DOID:'):
                result.add(s.replace('DOID:', '').lstrip('0'))
            elif s.startswith('MESH:'):
                doid = mesh_to_doid.get(s)
                if doid:
                    result.add(doid)
        return result
    if kg_name == 'biokg':
        # BioKG uses bare MeSH IDs (D*, C*) — bridge to DOID via mesh_to_doid
        result = set()
        for i in raw:
            s = str(i).strip()
            # Try with MESH: prefix for lookup
            mesh_key = f'MESH:{s}' if not s.startswith('MESH:') else s
            doid = mesh_to_doid.get(mesh_key)
            if doid:
                result.add(doid)
        return result
    # drkg: direct DOID + MESH-bridged
    return (
        {i.replace('DOID:', '').lstrip('0') for i in raw if i.startswith('DOID:')} |
        {mesh_to_doid[i] for i in raw if i.startswith('MESH:') and i in mesh_to_doid}
    )


def _disease_bridge_note(kg, kg_name, mondo_to_doid, mesh_to_doid):
    """One-line summary of disease ID normalisation losses for a KG."""
    dtype    = ENTITY_TYPE_MAP[kg_name]['Disease']
    raw_ids  = {_strip_prefix(i) for i in _as_str_set(pd.concat([
        kg.loc[kg['x_type'] == dtype, 'x_id'],
        kg.loc[kg['y_type'] == dtype, 'y_id'],
    ], ignore_index=True))}
    n_raw    = len(raw_ids)
    n_norm   = len(get_disease_ids_doid(kg, kg_name, mondo_to_doid, mesh_to_doid))
    if kg_name == 'primekg':
        parts  = {p.strip().lstrip('0') for i in raw_ids
                  for p in str(i).split('_') if p.strip().lstrip('0')}
        n_lost = len(parts) - n_norm
        return (f'{n_raw:,} nodes -> {len(parts):,} MONDO parts after split -> '
                f'{n_norm:,} mapped to DOID  ({n_lost:,} MONDO parts with no cross-ref)')
    if kg_name in ('hetionet', 'openbilink'):
        return f'{n_raw:,} DOID nodes -> {n_norm:,} normalised'
    if kg_name == 'matrix':
        mondo_ids = [i for i in raw_ids if str(i).startswith('MONDO:')]
        mesh_ids  = [i for i in raw_ids if str(i).startswith('MESH:')]
        n_mondo_mapped = sum(1 for i in mondo_ids
                             if mondo_to_doid.get(i.replace('MONDO:', '').lstrip('0')))
        n_mesh_mapped  = sum(1 for i in mesh_ids if i in mesh_to_doid)
        n_lost = n_raw - n_norm
        return (f'{n_raw:,} disease nodes -> '
                f'{n_mondo_mapped:,}/{len(mondo_ids):,} MONDO bridged, '
                f'{n_mesh_mapped:,}/{len(mesh_ids):,} MESH bridged -> '
                f'{n_norm:,} DOID  ({n_lost:,} lost: OMIM/unmapped)')
    if kg_name == 'biokg':
        # BioKG bare MeSH IDs
        n_mapped = sum(1 for i in raw_ids
                       if mesh_to_doid.get(f'MESH:{i}' if not str(i).startswith('MESH:') else str(i)))
        n_lost = n_raw - n_norm
        return (f'{n_raw:,} MeSH disease nodes -> '
                f'{n_mapped:,}/{n_raw:,} bridged via MESH→DOID -> '
                f'{n_norm:,} DOID  ({n_lost:,} lost: MeSH scope gap)')
    # drkg
    mesh_ids      = [i for i in raw_ids if i.startswith('MESH:')]
    n_mesh        = len(mesh_ids)
    n_mesh_mapped = sum(1 for i in mesh_ids if i in mesh_to_doid)
    n_doid_direct = sum(1 for i in raw_ids if i.startswith('DOID:'))
    n_lost        = n_raw - n_norm
    return (f'{n_raw:,} nodes -> {n_doid_direct:,} direct DOID + '
            f'{n_mesh_mapped:,}/{n_mesh:,} MESH bridged ({100*n_mesh_mapped/max(n_mesh,1):.0f}%) -> '
            f'{n_norm:,} DOID  ({n_lost:,} lost: MESH scope gap + OMIM)')
