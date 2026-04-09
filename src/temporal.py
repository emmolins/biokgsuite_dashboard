"""Post-cutoff FDA approvals and temporal validation utilities.

Manages hardcoded FDA supplemental approvals (post-knowledge-graph-cutoff)
and utilities for resolving drugs and diseases to KG nodes.
"""

import pandas as pd


# Post-cutoff FDA supplemental approvals; all KGs predate these approvals.
# Sources: FDA Drugs@FDA (Efficacy Supplements), FDA approval letters.
# Format: DrugBank ID → (human-readable name, [(disease term, date, description)])
NEW_INDICATION_APPROVALS = {
    'DB12159': ('dupilumab',       [('chronic obstructive pulmonary disease','2024-09-27','COPD with eosinophilic phenotype'),
                                    ('prurigo nodularis',                    '2023-10-25','Prurigo nodularis in adults')]),
    'DB13928': ('semaglutide',     [('cardiovascular disease', '2024-03-08','CV risk reduction in overweight/obese adults'),
                                    ('obstructive sleep apnea','2024-12-13','Moderate-to-severe OSA with obesity')]),
    'DB09037': ('pembrolizumab',   [('endometrial carcinoma',                   '2024-06-07','Advanced endometrial carcinoma (1L)'),
                                    ('cervical cancer',                         '2024-01-15','Persistent/recurrent/metastatic cervical cancer'),
                                    ('gastric cancer',                          '2024-10-25','HER2+ gastric/GEJ adenocarcinoma (1L)'),
                                    ('gastroesophageal junction adenocarcinoma','2024-10-25','HER2+ GEJ adenocarcinoma'),
                                    ('hepatocellular carcinoma',                '2024-09-03','HCC (1L with lenvatinib)')]),
    'DB09035': ('nivolumab',       [('hepatocellular carcinoma','2024-08-15','HCC (1L with ipilimumab)'),
                                    ('colorectal cancer',       '2024-07-25','MSI-H/dMMR CRC'),
                                    ('esophageal cancer',       '2024-03-22','Resectable esophageal/GEJ cancer (neoadjuvant)')]),
    'DB11714': ('durvalumab',      [('small cell lung carcinoma','2024-12-20','Limited-stage SCLC'),
                                    ('biliary tract neoplasm',   '2024-09-05','Biliary tract cancer (1L)')]),
    'DB11595': ('atezolizumab',    [('hepatocellular carcinoma','2023-10-15','Unresectable HCC (with bevacizumab)')]),
    'DB00051': ('adalimumab',      [('hidradenitis suppurativa','2024-04-30','HS in adolescents 12+ years')]),
    'DB09053': ('ibrutinib',       [('chronic lymphocytic leukemia/small lymphocytic lymphoma','2023-12-01','CLL/SLL (1L and relapsed)'),
                                    ('follicular lymphoma',                                    '2024-03-29','Relapsed/refractory FL')]),
    'DB11581': ('venetoclax',      [('bilineal acute myeloid leukemia','2024-01-12','Newly diagnosed AML (with azacitidine)'),
                                    ('myelodysplastic syndrome',        '2024-06-14','Higher-risk MDS')]),
    'DB11703': ('acalabrutinib',   [('chronic lymphocytic leukemia/small lymphocytic lymphoma','2024-05-20','CLL/SLL (1L and relapsed)')]),
    'DB09074': ('olaparib',        [('prostate cancer, hereditary','2023-12-19','mCRPC with HRR gene mutations'),
                                    ('breast cancer',               '2024-03-11','Adjuvant HER2- high-risk early breast cancer')]),
    'DB00072': ('trastuzumab',     [('gastric cancer','2024-06-13','HER2+ metastatic gastric cancer (updated regimen)')]),
    'DB09078': ('lenvatinib',      [('endometrial cancer','2024-06-07','Advanced endometrial carcinoma (with pembrolizumab)')]),
    'DB06292': ('dapagliflozin',   [('heart failure',            '2023-08-10','Heart failure across LVEF spectrum'),
                                    ('congestive heart failure',  '2023-08-10','HFrEF and HFpEF')]),
    'DB09038': ('empagliflozin',   [('heart failure','2023-09-15','Heart failure across LVEF spectrum')]),
    'DB08877': ('ruxolitinib',     [('alopecia areata','2023-08-14','Alopecia areata (topical)'),
                                    ('vitiligo',       '2023-10-18','Non-segmental vitiligo (topical)')]),
    'DB08879': ('belimumab',       [('lupus nephritis','2024-06-01','Active lupus nephritis (expanded)')]),
    'DB15091': ('upadacitinib',    [('ulcerative colitis',    '2023-08-16','Moderately-to-severely active UC'),
                                    ('Crohn disease',         '2024-05-17','Moderately-to-severely active CD'),
                                    ('ankylosing spondylitis','2024-02-13','Active AS'),
                                    ('atopic eczema',         '2024-01-25','Moderate-to-severe AD (expanded age)')]),
    'DB14762': ('risankizumab',    [('Crohn disease',      '2023-08-02','Moderately-to-severely active Crohn disease'),
                                    ('ulcerative colitis', '2024-09-19','Moderately-to-severely active UC')]),
    'DB06273': ('tocilizumab',     [('systemic lupus erythematosus','2024-03-06','Expanded systemic autoimmune indications')]),
    'DB08875': ('cabozantinib',    [('hepatocellular carcinoma','2024-02-28','Previously treated HCC')]),
    'DB09331': ('daratumumab',     [('plasma cell myeloma','2024-03-25','Newly diagnosed transplant-eligible MM (expanded)')]),
    'DB01590': ('everolimus',      [('gastrointestinal stromal tumor','2024-05-09','Advanced GI NETs (expanded)')]),
}


# Curated aliases for FDA approval terms that differ from DO nomenclature
TEMPORAL_DISEASE_ALIASES = {
    'bilineal acute myeloid leukemia':                         'acute myeloid leukemia',
    'heart failure':                                           'congestive heart failure',
    'prostate cancer, hereditary':                             'prostate cancer',
    'gastroesophageal junction adenocarcinoma':                'esophageal adenocarcinoma',
    'biliary tract neoplasm':                                  'biliary tract cancer',
    'obstructive sleep apnea':                                 'obstructive sleep apnea syndrome',
}


def build_drkg_disease_resolver(do_diseases_path, mesh_to_doid_path):
    """Build resolver for DRKG disease nodes from ontology data.

    Parameters
    ----------
    do_diseases_path : Path or str
        Path to 'do_diseases.csv' with columns: doid, mondo_name.
    mesh_to_doid_path : Path or str
        Path to 'mesh_to_doid.csv' with columns: mesh_id, doid.

    Returns
    -------
    tuple of dicts
        (name_to_doid, doid_to_mesh) for FDA term resolution.
    """
    _do_rev = pd.read_csv(do_diseases_path)
    _mesh_rev = pd.read_csv(mesh_to_doid_path, on_bad_lines='skip')
    _mesh_rev['mesh_clean'] = _mesh_rev['mesh_id'].astype(str).str.split(' ').str[0]

    _doid_to_mesh_t = {}
    for _, _r in _mesh_rev.iterrows():
        _d = str(_r['doid'])
        if _d not in _doid_to_mesh_t:
            _doid_to_mesh_t[_d] = _r['mesh_clean']

    _name_to_doid_t = {}
    for _, _r in _do_rev.iterrows():
        _n = str(_r['mondo_name']).lower().strip()
        if _n not in _name_to_doid_t:
            _name_to_doid_t[_n] = str(_r['doid'])

    return _name_to_doid_t, _doid_to_mesh_t


def resolve_drkg_disease(term, name_to_doid, doid_to_mesh, drkg_dis_rev):
    """Resolve an FDA approval disease term to a DRKG disease node.

    Parameters
    ----------
    term : str
        Disease term from FDA approval.
    name_to_doid : dict
        DO disease name → DOID numeric.
    doid_to_mesh : dict
        DOID → MESH ID mapping.
    drkg_dis_rev : dict
        Reverse map of DRKG disease node names (DOID:* / MESH:*) → indices.

    Returns
    -------
    int or None
        DRKG node index for disease, or None if unresolvable.
    """
    for t in [term.lower(), TEMPORAL_DISEASE_ALIASES.get(term.lower(), '')]:
        if not t:
            continue
        doid = name_to_doid.get(t)
        if doid:
            doid_node = 'DOID:' + doid.split(':')[1] if ':' in doid else None
            if doid_node and doid_node in drkg_dis_rev:
                return drkg_dis_rev[doid_node]
            mesh = doid_to_mesh.get(doid)
            if mesh and mesh in drkg_dis_rev:
                return drkg_dis_rev[mesh]
    return None


def build_disease_name_to_idx_kg(kgs, maps, kg_config, loaded_kgs):
    """Build disease name → node index lookup tables for all loaded KGs.

    Parameters
    ----------
    kgs : dict
        KG data dict with 'primekg', 'hetionet', 'drkg', 'matrix' keys.
    maps : dict
        Node mapping dicts with 'node_name_map' and 'node_type_map' keys.
    kg_config : dict
        KG-specific config with 'disease_type' key.
    loaded_kgs : list of str
        List of loaded KG names to process.

    Returns
    -------
    dict
        Mapping of KG name → {disease_name → node_index}.
    """
    disease_name_to_idx_kg = {}
    for name in loaded_kgs:
        if name == 'drkg':
            # Reverse map: disease node name (DOID:xxx / MESH:Dxxx) → node index
            disease_name_to_idx_kg[name] = {
                v: k for k, v in maps[name]['node_name_map'].items()
                if maps[name]['node_type_map'].get(k) == 'Disease'
            }
            continue
        kg_data = kgs[name]['kg']
        dst = kg_config[name]['disease_type']
        d2i = {}
        for ci, cn, ct in [('x_index','x_name','x_type'), ('y_index','y_name','y_type')]:
            sub = kg_data[kg_data[ct] == dst][[ci, cn]].drop_duplicates(ci)
            for _, row in sub.iterrows():
                d2i[str(row[cn]).lower()] = row[ci]
        disease_name_to_idx_kg[name] = d2i
    return disease_name_to_idx_kg


def find_drug_in_kg(dbid, drug_name, kg_name, maps, kg_config):
    """Find a drug node in a KG by DrugBank ID or name.

    Parameters
    ----------
    dbid : str
        DrugBank ID (bare, e.g., 'DB00001').
    drug_name : str
        Human-readable drug name.
    kg_name : str
        KG name (primekg, hetionet, drkg, matrix).
    maps : dict
        Node mapping dicts.
    kg_config : dict
        KG-specific config.

    Returns
    -------
    int or None
        Node index in the KG, or None if not found.
    """
    nm = maps[kg_name]['node_name_map']
    tm = maps[kg_name]['node_type_map']
    dt = kg_config[kg_name]['drug_type']

    if kg_name == 'primekg':
        return maps[kg_name]['id_to_idx'].get(dbid)  # PrimeKG: bare DrugBank ID
    elif kg_name == 'matrix':
        return maps[kg_name]['id_to_idx'].get('DRUGBANK:' + dbid)  # MATRIX: CURIE 'DRUGBANK:DB*'
    elif kg_name == 'drkg':
        # DRKG: find by name or Compound::DBID format
        return _find_node_by_prefix(dbid, nm, tm, dt, 'Compound::')
    else:
        # Hetionet: readable name
        return _find_node_by_name(drug_name, nm, tm, dt)


def find_disease_in_kg(disease_term, kg_name, disease_name_to_idx_kg):
    """Find a disease node in a KG by term.

    Parameters
    ----------
    disease_term : str
        Disease term (name or ID).
    kg_name : str
        KG name (primekg, hetionet, drkg, matrix).
    disease_name_to_idx_kg : dict
        Pre-built disease name → index mapping for each KG.

    Returns
    -------
    int or None
        Node index in the KG, or None if not found.
    """
    if kg_name == 'drkg':
        # Handled separately with resolver function
        return None
    d2i = disease_name_to_idx_kg.get(kg_name, {})
    term = disease_term.lower()
    if term in d2i:
        return d2i[term]
    cands = sorted([(n, i) for n, i in d2i.items() if term in n], key=lambda x: len(x[0]))
    return cands[0][1] if cands else None


def _find_node_by_name(name, node_name_map, node_type_map, entity_type):
    """Helper: find node by matching name and type."""
    name_lower = str(name).lower()
    for node_id, n_name in node_name_map.items():
        if str(n_name).lower() == name_lower and node_type_map.get(node_id) == entity_type:
            return node_id
    return None


def _find_node_by_prefix(id_str, node_name_map, node_type_map, entity_type, prefix=''):
    """Helper: find node by ID with optional prefix."""
    search_id = prefix + str(id_str) if prefix else str(id_str)
    for node_id in node_name_map:
        if str(node_id) == search_id and node_type_map.get(node_id) == entity_type:
            return node_id
    return None
