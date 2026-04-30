"""Knowledge graph loaders.

Each loader returns a standardized tuple:
    (kg_df, nodes_df)

where:
  - kg_df has columns: relation, x_index, x_id, x_type, x_name, y_index, y_id, y_type, y_name
    (plus optional: display_relation, x_source, y_source)
  - nodes_df has columns: idx, id, type, name
"""

import gzip
import re
import pandas as pd
import yaml
from pathlib import Path


def find_config(start: Path = None) -> Path:
    """Locate config.yaml by searching upward from *start* (default: cwd).

    Searches the start directory and its immediate parent so the notebooks
    work correctly whether launched from ``eval_notebooks/`` (interactive
    JupyterLab) or from the repo root (``nbconvert --execute``).

    Parameters
    ----------
    start : Path, optional
        Directory to begin the search. Defaults to ``Path.cwd()``.

    Returns
    -------
    Path — resolved path to config.yaml.

    Raises
    ------
    FileNotFoundError if config.yaml is not found in start or its parent.
    """
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start / 'config.yaml', start.parent / 'config.yaml']:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'config.yaml not found in {start} or {start.parent}. '
        'Run notebooks from the repo root or the eval_notebooks/ directory.'
    )


def load_config(config_path='config.yaml'):
    """Load the YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to config.yaml (absolute, or relative to cwd).

    Returns
    -------
    dict — includes '_base_dir' key with the resolved config parent directory.
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['_base_dir'] = config_path.parent
    return cfg


def load_primekg(path):
    """Load PrimeKG from its CSV file.

    Parameters
    ----------
    path : str or Path
        Path to primekg.csv.

    Returns
    -------
    kg : pd.DataFrame
        Edge table with standardized columns.
    nodes : pd.DataFrame
        Node table with columns [idx, id, type, name].
    """
    cols = {
        'relation', 'display_relation',
        'x_index', 'x_id', 'x_type', 'x_name', 'x_source',
        'y_index', 'y_id', 'y_type', 'y_name', 'y_source',
    }
    # Chunked read prevents peak-memory spikes on the ~8 M-row CSV.
    # pandas still parses full rows before filtering, so low_memory=False on
    # the full file routinely exceeds 3 GB; 200 k-row chunks stay comfortably
    # below that ceiling.
    chunks = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in cols,
                              dtype={'x_id': str, 'y_id': str},
                              low_memory=True, chunksize=200_000):
        chunks.append(chunk)
    kg = pd.concat(chunks, ignore_index=True)
    del chunks

    nodes = _build_node_table(kg)
    return kg, nodes


def load_hetionet(nodes_path, edges_path):
    """Load Hetionet from TSV files.

    Parameters
    ----------
    nodes_path : str or Path
        Path to hetionet nodes TSV.
    edges_path : str or Path
        Path to hetionet edges TSV.

    Returns
    -------
    kg : pd.DataFrame   (standardized columns)
    nodes : pd.DataFrame (columns: idx, id, type, name)
    """
    raw_nodes = pd.read_csv(nodes_path, sep='\t')
    raw_edges = pd.read_csv(edges_path, sep='\t')

    # Hetionet nodes: id, name, kind
    raw_nodes = raw_nodes.rename(columns={'kind': 'type'})
    raw_nodes['idx'] = range(len(raw_nodes))
    id_to_idx = dict(zip(raw_nodes['id'], raw_nodes['idx']))

    # Hetionet edges: source, metaedge, target
    kg = pd.DataFrame({
        'relation': raw_edges['metaedge'],
        'x_index': raw_edges['source'].map(id_to_idx),
        'x_id': raw_edges['source'].astype(str),
        'y_index': raw_edges['target'].map(id_to_idx),
        'y_id': raw_edges['target'].astype(str),
    })

    # Merge node metadata
    node_meta = dict(zip(raw_nodes['id'], zip(raw_nodes['type'], raw_nodes['name'])))
    kg['x_type'] = raw_edges['source'].map(lambda x: node_meta.get(x, ('', ''))[0])
    kg['x_name'] = raw_edges['source'].map(lambda x: node_meta.get(x, ('', ''))[1])
    kg['y_type'] = raw_edges['target'].map(lambda x: node_meta.get(x, ('', ''))[0])
    kg['y_name'] = raw_edges['target'].map(lambda x: node_meta.get(x, ('', ''))[1])

    kg = kg.dropna(subset=['x_index', 'y_index'])
    kg['x_index'] = kg['x_index'].astype(int)
    kg['y_index'] = kg['y_index'].astype(int)

    nodes_df = raw_nodes[['idx', 'id', 'type', 'name']].copy()
    return kg, nodes_df


def load_drkg(path):
    """Load DRKG from its TSV file.

    DRKG format: head \\t relation \\t tail  (no header).
    Entity IDs encode their type, e.g. "Gene::1234", "Compound::DB00001".

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    kg : pd.DataFrame   (standardized columns)
    nodes : pd.DataFrame (columns: idx, id, type, name)
    """
    raw = pd.read_csv(path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    # Extract entity type from ID prefix
    def _parse_entity(eid):
        parts = eid.split('::', 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ('unknown', eid)

    all_entities = set(raw['head']) | set(raw['tail'])
    entity_list = sorted(all_entities)
    id_to_idx = {eid: i for i, eid in enumerate(entity_list)}

    nodes_records = []
    for eid in entity_list:
        etype, ename = _parse_entity(eid)
        nodes_records.append({
            'idx': id_to_idx[eid],
            'id': eid,
            'type': etype,
            'name': ename,
        })
    nodes_df = pd.DataFrame(nodes_records)
    type_map = dict(zip(nodes_df['id'], nodes_df['type']))
    name_map = dict(zip(nodes_df['id'], nodes_df['name']))

    kg = pd.DataFrame({
        'relation': raw['relation'],
        'x_index': raw['head'].map(id_to_idx),
        'x_id': raw['head'],
        'x_type': raw['head'].map(type_map),
        'x_name': raw['head'].map(name_map),
        'y_index': raw['tail'].map(id_to_idx),
        'y_id': raw['tail'],
        'y_type': raw['tail'].map(type_map),
        'y_name': raw['tail'].map(name_map),
    })

    return kg, nodes_df


def _openbilink_hq_mask(score_series, source_series):
    """Return boolean mask selecting high-quality OpenBioLink edges.

    Quality criteria (per the OpenBioLink benchmark specification):
    * STRING / STITCH (numeric confidence scores): keep score >= 700
    * Bgee (expression quality labels): keep 'gold quality' or 'high quality'
    * All other sources (curated): keep unconditionally

    This reduces the ~29 M-row ALL_DIR dump to ~5 M HQ edges while
    retaining *all* entity types and curated relation classes.
    """
    score  = score_series.fillna('')
    source = source_series.fillna('')
    # Numeric-score sources: STRING / STITCH
    numeric_src = source.isin(['STRING', 'STITCH'])
    score_num   = pd.to_numeric(score, errors='coerce')
    num_pass    = numeric_src & (score_num >= 700)
    # Bgee quality labels
    bgee_src    = source == 'Bgee'
    bgee_pass   = bgee_src & score.isin(['gold quality', 'high quality'])
    # All other sources pass unconditionally
    other_src   = ~numeric_src & ~bgee_src
    return num_pass | bgee_pass | other_src


def load_openbilink(path, pubchem_to_drugbank_path=None, hq_filter=True):
    """Load OpenBioLink from its TSV edges file.

    OpenBioLink format (TSV, no header, 5 columns):
        subject \\t relation \\t object \\t score \\t source

    Entity types are inferred from ID prefixes:
        NCBIGENE:*          → Gene
        PUBCHEM.COMPOUND:*  → Drug
        DOID:*              → Disease
        GO:*                → GO
        UBERON:*            → Anatomy
        HP:*                → Phenotype
        CL:*                → Cell
        REACTOME:*          → Pathway
        KEGG:*              → Pathway

    Parameters
    ----------
    path : str or Path
        Path to edges.csv (tab-separated, no header).
    pubchem_to_drugbank_path : str or Path, optional
        Path to pubchem_to_drugbank.csv mapping file.  When provided the
        mapping is attached to the nodes DataFrame as a 'drugbank_id' column
        on Drug nodes (empty string for unmapped nodes).
    hq_filter : bool, default True
        Apply the OpenBioLink high-quality filter (STRING/STITCH >= 700,
        Bgee gold/high quality, all curated kept).  Reduces ~29 M edges
        to ~5 M.  Set False only if sufficient RAM is available.

    Returns
    -------
    kg : pd.DataFrame   (standardized columns)
    nodes : pd.DataFrame (columns: idx, id, type, name, drugbank_id)
    """
    import gc

    # ── ID-prefix → canonical type mapping ─────────────────────────────────────
    _PREFIX_TO_TYPE = {
        'NCBIGENE':          'Gene',
        'PUBCHEM.COMPOUND':  'Drug',
        'DOID':              'Disease',
        'GO':                'GO',
        'UBERON':            'Anatomy',
        'HP':                'Phenotype',
        'CL':                'Cell',
        'REACTOME':          'Pathway',
        'KEGG':              'Pathway',
    }

    def _type_from_id(eid):
        colon_pos = str(eid).find(':')
        if colon_pos == -1:
            return 'unknown'
        return _PREFIX_TO_TYPE.get(str(eid)[:colon_pos], str(eid)[:colon_pos])

    def _name_from_id(eid):
        s = str(eid)
        colon_pos = s.find(':')
        return s[colon_pos + 1:] if colon_pos != -1 else s

    # ── Read, apply HQ filter, and collect entity IDs in one pass ─────────────
    all_entities = set()
    filtered_chunks = []
    for chunk in pd.read_csv(path, sep='\t', header=None,
                             names=['head', 'relation', 'tail', 'score', 'source'],
                             dtype=str, chunksize=2_000_000,
                             encoding='utf-8', lineterminator='\n'):
        # Strip Windows \r from all string columns
        for col in chunk.columns:
            chunk[col] = chunk[col].str.rstrip('\r')
        chunk = chunk.fillna('')

        if hq_filter:
            mask = _openbilink_hq_mask(chunk['score'], chunk['source'])
            chunk = chunk[mask].copy()

        if chunk.empty:
            continue

        all_entities.update(chunk['head'])
        all_entities.update(chunk['tail'])
        # Keep only columns needed for the edge table
        filtered_chunks.append(
            chunk[['head', 'relation', 'tail', 'source']].copy()
        )
        gc.collect()

    # ── Build node table ───────────────────────────────────────────────────────
    entity_list = sorted(all_entities)
    del all_entities
    id_to_idx = {eid: i for i, eid in enumerate(entity_list)}

    ids_series = pd.Series(entity_list, dtype='object')
    nodes_df = pd.DataFrame({
        'idx':  range(len(entity_list)),
        'id':   ids_series,
        'type': ids_series.map(_type_from_id),
        'name': ids_series.map(_name_from_id),
    })
    del ids_series, entity_list

    # ── Attach DrugBank IDs via PubChem→DrugBank mapping ──────────────────────
    nodes_df['drugbank_id'] = ''
    if pubchem_to_drugbank_path is not None:
        pcdb = pd.read_csv(pubchem_to_drugbank_path, dtype=str)
        pc_map = dict(zip(
            'PUBCHEM.COMPOUND:' + pcdb['pubchem_cid'].astype(str),
            pcdb['drugbank_id'].astype(str),
        ))
        drug_mask = nodes_df['type'] == 'Drug'
        nodes_df.loc[drug_mask, 'drugbank_id'] = (
            nodes_df.loc[drug_mask, 'id'].map(pc_map).fillna('')
        )

    # ── Build fast lookup arrays ───────────────────────────────────────────────
    _type_arr = nodes_df['type'].values
    _name_arr = nodes_df['name'].values

    # ── Build edge table from filtered chunks ─────────────────────────────────
    kg_parts = []
    for fchunk in filtered_chunks:
        xi = fchunk['head'].map(id_to_idx)
        yi = fchunk['tail'].map(id_to_idx)
        valid = xi.notna() & yi.notna()
        xi = xi[valid].astype('int32')
        yi = yi[valid].astype('int32')

        part = pd.DataFrame({
            'relation': fchunk.loc[valid, 'relation'].values,
            'x_index':  xi.values,
            'x_id':     fchunk.loc[valid, 'head'].values,
            'x_type':   _type_arr[xi.values],
            'x_name':   _name_arr[xi.values],
            'y_index':  yi.values,
            'y_id':     fchunk.loc[valid, 'tail'].values,
            'y_type':   _type_arr[yi.values],
            'y_name':   _name_arr[yi.values],
            'x_source': fchunk.loc[valid, 'source'].values,
            'y_source': fchunk.loc[valid, 'source'].values,
        })
        kg_parts.append(part)

    del filtered_chunks, id_to_idx, _type_arr, _name_arr
    gc.collect()

    kg = pd.concat(kg_parts, ignore_index=True)
    del kg_parts
    gc.collect()

    return kg, nodes_df


def load_biokg(links_path, metadata_dir=None):
    """Load BioKG from its consolidated links TSV file.

    BioKG format: subject \\t predicate \\t object  (no header, 3 columns).
    Entity types are inferred from the predicate (e.g. PPI → Gene/Protein ×
    Gene/Protein, DPI → Drug × Gene/Protein).  Entity names are resolved from
    optional metadata TSV files when *metadata_dir* is given.

    Parameters
    ----------
    links_path : str or Path
        Path to ``biokg.links.tsv`` (or ``.tsv.gz``).
    metadata_dir : str or Path, optional
        Directory containing ``biokg.metadata.*.tsv`` (or ``.tsv.gz``) files
        used for entity-name lookups.  If *None*, entity IDs are used as names.

    Returns
    -------
    kg : pd.DataFrame   (standardised columns)
    nodes : pd.DataFrame (columns: idx, id, type, name)
    """
    links_path = Path(links_path)

    # ── Predicate → (subject_type, object_type) ────────────────────────────
    _PRED_TYPES = {
        'PPI':                         ('Gene/Protein', 'Gene/Protein'),
        'DPI':                         ('Drug',         'Gene/Protein'),
        'DDI':                         ('Drug',         'Drug'),
        'DRUG_TARGET':                 ('Drug',         'Gene/Protein'),
        'DRUG_CARRIER':                ('Drug',         'Gene/Protein'),
        'DRUG_ENZYME':                 ('Drug',         'Gene/Protein'),
        'DRUG_TRANSPORTER':            ('Drug',         'Gene/Protein'),
        'PROTEIN_DISEASE_ASSOCIATION': ('Gene/Protein', 'Disease'),
        'DRUG_DISEASE_ASSOCIATION':    ('Drug',         'Disease'),
        'PROTEIN_PATHWAY_ASSOCIATION': ('Gene/Protein', 'Pathway'),
        'DRUG_PATHWAY_ASSOCIATION':    ('Drug',         'Pathway'),
        'DISEASE_PATHWAY_ASSOCIATION': ('Disease',      'Pathway'),
        'MEMBER_OF_COMPLEX':           ('Gene/Protein', 'Complex'),
        'MEMBER_OF_PATHWAY':           ('Complex',      'Pathway'),
        'MEMBER_OF_TOP_LEVEL_PATHWAY': ('Complex',      'Pathway'),
        'COMPLEX_IN_PATHWAY':          ('Complex',      'Pathway'),
        'COMPLEX_TOP_LEVEL_PATHWAY':   ('Complex',      'Pathway'),
        'HAS_PARENT_PATHWAY':          ('Pathway',      'Pathway'),
        'DISEASE_GENETIC_DISORDER':    ('Disease',      'GeneticDisorder'),
        'RELATED_GENETIC_DISORDER':    ('Gene/Protein', 'GeneticDisorder'),
        'DISEASE_SUPERGRP':            ('Disease',      'DiseaseCategory'),
        'DRUG_SIDEEFFECT_ASSOCIATION': ('Drug',         'SideEffect'),
        'DRUG_INDICATION_ASSOCIATION': ('Drug',         'Disease'),
        'DRUG_ATC_CODE':               ('Drug',         'ATC'),
        'PROTEIN_EXPRESSED_IN':        ('Gene/Protein', 'Tissue'),
        'PART_OF_TISSUE':              ('Cell',         'Tissue'),
    }

    # Type priority: higher-priority types override lower-priority ones when
    # the same entity is seen in multiple predicates (e.g. an R-HSA-* node
    # first seen as Complex target in MEMBER_OF_COMPLEX, later as Pathway
    # target in PROTEIN_PATHWAY_ASSOCIATION).
    _TYPE_PRIORITY = {
        'Drug': 5, 'Disease': 5, 'Gene/Protein': 5, 'Pathway': 5,
        'GeneticDisorder': 4, 'SideEffect': 4, 'ATC': 4,
        'DiseaseCategory': 3, 'Tissue': 3, 'Cell': 3,
        'Complex': 2,
        'unknown': 0,
    }

    # ── Read triples ────────────────────────────────────────────────────────
    open_fn = gzip.open if str(links_path).endswith('.gz') else open
    with open_fn(links_path, 'rt') as fh:
        raw = pd.read_csv(fh, sep='\t', header=None,
                          names=['head', 'relation', 'tail'], dtype=str)
    raw = raw.dropna(subset=['head', 'relation', 'tail'])

    # ── Assign entity types from predicate ──────────────────────────────────
    x_types = raw['relation'].map(lambda r: _PRED_TYPES.get(r, ('unknown', 'unknown'))[0])
    y_types = raw['relation'].map(lambda r: _PRED_TYPES.get(r, ('unknown', 'unknown'))[1])

    # ── Build entity → type mapping (highest-priority wins) ─────────────────
    entity_type = {}
    for eid, etype in zip(raw['head'], x_types):
        if etype == 'unknown':
            continue
        cur = entity_type.get(eid, 'unknown')
        if _TYPE_PRIORITY.get(etype, 0) > _TYPE_PRIORITY.get(cur, 0):
            entity_type[eid] = etype
    for eid, etype in zip(raw['tail'], y_types):
        if etype == 'unknown':
            continue
        cur = entity_type.get(eid, 'unknown')
        if _TYPE_PRIORITY.get(etype, 0) > _TYPE_PRIORITY.get(cur, 0):
            entity_type[eid] = etype

    # ── Load optional metadata for entity names ─────────────────────────────
    id_to_name = {}
    if metadata_dir is not None:
        metadata_dir = Path(metadata_dir)
        for meta_file in sorted(metadata_dir.glob('biokg.metadata.*.tsv*')):
            mopen = gzip.open if str(meta_file).endswith('.gz') else open
            with mopen(meta_file, 'rt') as fh:
                for line in fh:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 3 and parts[1] == 'NAME':
                        eid, _, name = parts[0], parts[1], parts[2]
                        if eid not in id_to_name:
                            id_to_name[eid] = name

    # ── Build node table ────────────────────────────────────────────────────
    # BioKG uses bare IDs (e.g. "DB00001", "D000006", "P21917").  While
    # current ID namespaces don't collide in practice, we add a type-
    # qualified ``uid`` column for safe lookups, consistent with PrimeKG.
    all_entities = sorted(set(raw['head']) | set(raw['tail']))
    id_to_idx = {eid: i for i, eid in enumerate(all_entities)}

    _types_list = [entity_type.get(eid, 'unknown') for eid in all_entities]
    nodes_df = pd.DataFrame({
        'idx':  range(len(all_entities)),
        'id':   all_entities,
        'type': _types_list,
        'name': [id_to_name.get(eid, eid) for eid in all_entities],
    })
    nodes_df['uid'] = nodes_df['type'].astype(str) + '/' + nodes_df['id'].astype(str)

    # ── Build edge table ────────────────────────────────────────────────────
    type_arr = nodes_df['type'].values
    name_arr = nodes_df['name'].values

    xi = raw['head'].map(id_to_idx)
    yi = raw['tail'].map(id_to_idx)

    kg = pd.DataFrame({
        'relation': raw['relation'].values,
        'x_index':  xi.values,
        'x_id':     raw['head'].values,
        'x_type':   type_arr[xi.values.astype(int)],
        'x_name':   name_arr[xi.values.astype(int)],
        'y_index':  yi.values,
        'y_id':     raw['tail'].values,
        'y_type':   type_arr[yi.values.astype(int)],
        'y_name':   name_arr[yi.values.astype(int)],
    })

    kg['x_index'] = kg['x_index'].astype(int)
    kg['y_index'] = kg['y_index'].astype(int)

    return kg, nodes_df


def load_kg(kg_name, config):
    """Load a knowledge graph by name using config settings.

    Parameters
    ----------
    kg_name : str
        One of 'primekg', 'hetionet', 'drkg', 'matrix'.
    config : dict
        Parsed config.yaml (from load_config).

    Returns
    -------
    kg : pd.DataFrame
    nodes : pd.DataFrame
    """
    cfg = config['knowledge_graphs'][kg_name]
    base = config.get('_base_dir', Path('.'))

    def _resolve(p):
        """Resolve a config path relative to the repo root."""
        p = Path(p)
        return p if p.is_absolute() else base / p

    if kg_name == 'primekg':
        return load_primekg(_resolve(cfg['path']))
    elif kg_name == 'hetionet':
        return load_hetionet(_resolve(cfg['nodes_path']), _resolve(cfg['edges_path']))
    elif kg_name == 'drkg':
        return load_drkg(_resolve(cfg['path']))
    elif kg_name == 'openbilink':
        pc_path = cfg.get('pubchem_to_drugbank_path')
        if pc_path:
            pc_path = _resolve(pc_path)
        hq = cfg.get('hq_filter', True)
        return load_openbilink(_resolve(cfg['path']), pc_path, hq_filter=hq)
    elif kg_name == 'matrix':
        return load_matrix(_resolve(cfg['nodes_path']), _resolve(cfg['edges_path']))
    elif kg_name == 'biokg':
        meta_dir = cfg.get('metadata_dir')
        if meta_dir:
            meta_dir = _resolve(meta_dir)
        return load_biokg(_resolve(cfg['path']), metadata_dir=meta_dir)
    else:
        raise ValueError(
            f"Unknown KG: {kg_name!r}. "
            f"Valid names: 'primekg', 'hetionet', 'drkg', 'openbilink', 'matrix', 'biokg'"
        )


# Private helpers 

def _build_node_table(kg):
    """Build a deduplicated node table from PrimeKG edge table.

    .. warning::

       PrimeKG external IDs (the ``id`` column) are **not unique across node
       types**.  For example, ``"9891"`` is both gene NUAK1 (NCBI) and disease
       "acquired polycythemia vera" (MONDO).  Any lookup dict built from
       ``id`` alone will silently collide.  Use the ``uid`` column (which
       prefixes the type) or the helper :func:`graph_utils.node_id_lookup`
       for safe lookups.
    """
    nodes = pd.concat([
        kg[['x_index', 'x_id', 'x_type', 'x_name']].rename(
            columns={'x_index': 'idx', 'x_id': 'id', 'x_type': 'type', 'x_name': 'name'}),
        kg[['y_index', 'y_id', 'y_type', 'y_name']].rename(
            columns={'y_index': 'idx', 'y_id': 'id', 'y_type': 'type', 'y_name': 'name'}),
    ]).drop_duplicates(subset=['idx'])
    nodes = nodes.reset_index(drop=True)
    # Type-qualified unique ID — safe for dict lookups across node types.
    nodes['uid'] = nodes['type'].astype(str) + '/' + nodes['id'].astype(str)
    return nodes


def load_matrix(nodes_path, edges_path):
    """Load EveryCure MATRIX knowledge graph from TSV files.

    Expected file formats
    ---------------------
    nodes_path (TSV) — columns from HuggingFace everycure/kg-nodes:
        id                      : CURIE-style identifier, e.g. 'DRUGBANK:DB00001',
                                  'MONDO:0005812', 'NCBIGene:1234', 'UniProtKB:P12345'
        name                    : human-readable label
        category                : Biolink category, e.g. 'biolink:SmallMolecule'
        equivalent_identifiers  : list of alternate CURIEs (string repr of list)
        upstream_data_source    : originating database(s)
        (other columns ignored)

    edges_path (TSV) — columns from HuggingFace everycure/kg-edges:
        subject   : source node CURIE (matches nodes 'id')
        predicate : Biolink predicate, e.g. 'biolink:treats'
        object    : target node CURIE
        primary_knowledge_source / aggregator_knowledge_source (optional)
        (column names are auto-detected; 'biolink:' prefix stripped from predicates)

    Biolink category → canonical type mapping (stored in nodes_df 'type' column):
        biolink:SmallMolecule / Drug / ChemicalEntity  → 'drug'
        biolink:Disease / DiseaseOrPhenotypicFeature   → 'disease'
        biolink:PhenotypicFeature                      → 'effect/phenotype'
        biolink:Gene / Protein / GeneOrGeneProduct     → 'gene/protein'
        biolink:GenomicEntity / NucleicAcidEntity      → 'gene/protein'
        biolink:BiologicalProcess                      → 'biological_process'
        biolink:MolecularActivity                      → 'molecular_function'
        biolink:CellularComponent                      → 'cellular_component'
        biolink:Pathway                                → 'pathway'
        biolink:AnatomicalEntity / Cell                → 'anatomy'
        biolink:OrganismTaxon                          → 'organism'

    ID format notes (CRITICAL)
    --------------------------
    * Drug IDs use CURIE format: 'DRUGBANK:DB00001'.  The regex r'(DB\\d{5,})'
      in extract_drugbank_ids() correctly extracts the DrugBank accession.
      Many nodes use CHEMBL / PUBCHEM as primary ID; look in
      equivalent_identifiers for the DRUGBANK CURIE when needed.
    * Disease IDs use CURIE format: 'MONDO:0005812', 'MESH:D000544'.
      MONDO IDs normalise directly via mondo_to_doid; MESH IDs bridge via
      mesh_to_doid.  No bare (prefix-free) IDs — unlike CKG.
    * Gene/Protein IDs: 'NCBIGene:1234' (extract numeric part for Entrez
      comparison) or 'UniProtKB:P12345' (bridge via uniprot_to_entrez).
      Nodes with NCBIGene: prefix integrate with existing Entrez gold standards
      without any accession bridging.

    Relation predicates
    -------------------
    All 'biolink:' prefixes are stripped so predicates are stored as bare
    strings (e.g. 'treats', 'contraindicated_for', 'affects').

    Parameters
    ----------
    nodes_path : str or Path
    edges_path : str or Path

    Returns
    -------
    kg : pd.DataFrame   (standardised BioKGSuite edge schema)
    nodes : pd.DataFrame (columns: idx, id, type, name)
    """
    # ── Biolink category → BioKGSuite canonical type ──────────────────────────
    _BIOLINK_TO_CANONICAL = {
        # ── Drug / Chemical ───────────────────────────────────────────────
        'SmallMolecule':             'drug',
        'Drug':                      'drug',
        'ChemicalEntity':            'drug',
        'ChemicalMixture':           'drug',
        'MolecularMixture':          'drug',
        'ComplexMolecularMixture':   'drug',
        'ProcessedMaterial':         'drug',
        'ChemicalOrDrugOrTreatment': 'drug',
        'Food':                      'drug',       # food compounds (e.g. vitamins, nutrients)
        # ── Disease / Phenotype ────────────────────────────────────────────
        'Disease':                   'disease',
        'DiseaseOrPhenotypicFeature':'disease',
        'PathologicalProcess':       'disease',    # pathological processes treated as diseases
        'PhenotypicFeature':         'effect/phenotype',
        'BehavioralFeature':         'effect/phenotype',
        # ── Gene / Protein ─────────────────────────────────────────────────
        'Gene':                      'gene/protein',
        'Protein':                   'gene/protein',
        'Polypeptide':               'gene/protein',
        'GeneOrGeneProduct':         'gene/protein',
        'GeneFamily':                'gene/protein',  # gene family nodes (Entrez via equiv IDs)
        'GenomicEntity':             'gene/protein',
        'NucleicAcidEntity':         'gene/protein',
        'Transcript':                'gene/protein',
        'Exon':                      'gene/protein',
        'RNAProduct':                'gene/protein',
        'NoncodingRNAProduct':       'gene/protein',
        'MicroRNA':                  'gene/protein',
        # ── Other biological entities ──────────────────────────────────────
        'BiologicalProcess':         'biological_process',
        'PhysiologicalProcess':      'biological_process',
        'MolecularActivity':         'molecular_function',
        'CellularComponent':         'cellular_component',
        'Pathway':                   'pathway',
        'AnatomicalEntity':          'anatomy',
        'GrossAnatomicalStructure':  'anatomy',
        'Cell':                      'anatomy',
        'CellLine':                  'anatomy',
        'OrganismTaxon':             'organism',
        'SequenceVariant':           'variant',
        'Haplotype':                 'variant',
        'Procedure':                 'procedure',
        'MolecularEntity':           'molecular_entity',
    }

    def _map_biolink(cat: str) -> str:
        """Strip 'biolink:' prefix and map to canonical type.

        Handles both simple strings ('biolink:Disease') and multi-value
        array-like strings produced by numpy repr, e.g.:
            "['biolink:Disease' 'biolink:DiseaseOrPhenotypicFeature' 'biolink:NamedThing']"
        For multi-value strings the first CamelCase token that has a mapping
        in _BIOLINK_TO_CANONICAL is used, so the most-specific declared type
        wins (callers should list the most-specific category first).
        """
        bare = str(cat).replace('biolink:', '').strip()
        # Fast path: clean single-category string
        result = _BIOLINK_TO_CANONICAL.get(bare)
        if result:
            return result
        # Slow path: array-like multi-value string, e.g. "['Disease' 'NamedThing']"
        # Extract capitalised CamelCase tokens and return the first recognised one.
        for token in re.findall(r'[A-Z][a-zA-Z0-9/]*', bare):
            result = _BIOLINK_TO_CANONICAL.get(token)
            if result:
                return result
        return bare.lower()

    # ── Load nodes ────────────────────────────────────────────────────────────
    raw_nodes = pd.read_csv(nodes_path, sep='\t', dtype=str).fillna('')
    raw_nodes['type'] = raw_nodes['category'].apply(_map_biolink)
    raw_nodes['idx']  = range(len(raw_nodes))
    id_to_idx = dict(zip(raw_nodes['id'], raw_nodes['idx']))

    # ── Extract normalised auxiliary IDs from equivalent_identifiers ──────────
    # Drug nodes: primary IDs are often PUBCHEM/CHEBI/UNII; DrugBank accessions
    #   are stored in equivalent_identifiers as 'DRUGBANK:DB*' entries.
    # Pathway nodes: some nodes use SMPDB/PathWhiz as primary ID but list a
    #   human Reactome ID ('REACT:R-HSA-*') in equivalent_identifiers.
    _eq_col = 'equivalent_identifiers' if 'equivalent_identifiers' in raw_nodes.columns else None
    if _eq_col:
        _drug_mask    = raw_nodes['type'] == 'drug'
        _pathway_mask = raw_nodes['type'] == 'pathway'
        _gene_mask    = raw_nodes['type'] == 'gene/protein'
        _disease_mask = raw_nodes['type'] == 'disease'

        # Drug nodes: first DrugBank ID (backward-compat) + all DrugBank IDs
        raw_nodes['drugbank_id'] = ''
        raw_nodes['drugbank_ids_all'] = ''
        if _drug_mask.any():
            raw_nodes.loc[_drug_mask, 'drugbank_id'] = (
                raw_nodes.loc[_drug_mask, _eq_col]
                .str.extract(r'\b(DB\d{5})\b', expand=False)
                .fillna('')
            )
            raw_nodes.loc[_drug_mask, 'drugbank_ids_all'] = (
                raw_nodes.loc[_drug_mask, _eq_col]
                .apply(lambda s: '|'.join(re.findall(r'\b(DB\d{5})\b', str(s))))
            )

        # Pathway nodes: Reactome ID from equivalent_identifiers
        raw_nodes['reactome_id'] = ''
        if _pathway_mask.any():
            raw_nodes.loc[_pathway_mask, 'reactome_id'] = (
                raw_nodes.loc[_pathway_mask, _eq_col]
                .str.extract(r'\b(R-HSA-\d+)\b', expand=False)
                .fillna('')
            )

        # Gene/protein nodes: NCBIGene ID from equivalent_identifiers
        # (bridges PR: and other non-NCBIGene/UniProtKB primary-ID nodes)
        raw_nodes['ncbigene_id'] = ''
        if _gene_mask.any():
            raw_nodes.loc[_gene_mask, 'ncbigene_id'] = (
                raw_nodes.loc[_gene_mask, _eq_col]
                .str.extract(r'\bNCBIGene:(\d+)\b', expand=False)
                .fillna('')
            )

        # Disease nodes: DOID from equivalent_identifiers
        # (captures DOID CURIEs stored alongside MONDO/MESH primary IDs)
        raw_nodes['doid_id'] = ''
        if _disease_mask.any():
            raw_nodes.loc[_disease_mask, 'doid_id'] = (
                raw_nodes.loc[_disease_mask, _eq_col]
                .str.extract(r'\bDOID:(\d+)\b', expand=False)
                .fillna('')
            )
    else:
        raw_nodes['drugbank_id'] = ''
        raw_nodes['drugbank_ids_all'] = ''
        raw_nodes['reactome_id'] = ''
        raw_nodes['ncbigene_id'] = ''
        raw_nodes['doid_id'] = ''

    # Build fast metadata lookup: id → (canonical_type, name)
    node_meta = dict(zip(
        raw_nodes['id'],
        zip(raw_nodes['type'], raw_nodes['name'])
    ))

    # ── Load edges ────────────────────────────────────────────────────────────
    raw_edges = pd.read_csv(edges_path, sep='\t', dtype=str).fillna('')
    ecols = raw_edges.columns.tolist()

    # Auto-detect subject / predicate / object column names
    subj_col = next(
        (c for c in ecols if c.lower() in ('subject', 'source', 'head', 'from')), None
    )
    obj_col  = next(
        (c for c in ecols if c.lower() in ('object', 'target', 'tail', 'to')), None
    )
    pred_col = next(
        (c for c in ecols if c.lower() in ('predicate', 'relation', 'edge_label', 'type')), None
    )
    if subj_col is None or obj_col is None or pred_col is None:
        raise ValueError(
            f"MATRIX edges: cannot detect subject/predicate/object columns. "
            f"Found columns: {ecols}"
        )

    src_ids    = raw_edges[subj_col]
    tgt_ids    = raw_edges[obj_col]
    # Strip 'biolink:' prefix from predicates (store as bare strings, e.g. 'treats')
    predicates = raw_edges[pred_col].str.replace('biolink:', '', regex=False)

    # ── Build standardised edge table ────────────────────────────────────────
    kg = pd.DataFrame({
        'relation': predicates,
        'x_index':  src_ids.map(id_to_idx),
        'x_id':     src_ids,
        'x_type':   src_ids.map(lambda i: node_meta.get(i, ('', ''))[0]),
        'x_name':   src_ids.map(lambda i: node_meta.get(i, ('', ''))[1]),
        'y_index':  tgt_ids.map(id_to_idx),
        'y_id':     tgt_ids,
        'y_type':   tgt_ids.map(lambda i: node_meta.get(i, ('', ''))[0]),
        'y_name':   tgt_ids.map(lambda i: node_meta.get(i, ('', ''))[1]),
    })

    # Forward knowledge-source column for source-diversity analysis (nb03)
    src_attr_col = next(
        (c for c in ecols if 'knowledge_source' in c.lower() or 'data_source' in c.lower()),
        None
    )
    if src_attr_col:
        kg['x_source'] = raw_edges[src_attr_col].values
        kg['y_source'] = raw_edges[src_attr_col].values

    # Drop edges where either endpoint is absent from the node table
    kg = kg.dropna(subset=['x_index', 'y_index'])
    kg['x_index'] = kg['x_index'].astype(int)
    kg['y_index'] = kg['y_index'].astype(int)

    nodes_df = raw_nodes[['idx', 'id', 'type', 'name',
                           'drugbank_id', 'drugbank_ids_all',
                           'reactome_id', 'ncbigene_id', 'doid_id']].copy()
    return kg, nodes_df
