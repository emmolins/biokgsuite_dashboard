"""Ontology bridging and disease resolution utilities.

Provides utilities for resolving disease identifiers across ontologies
(DOID, MESH, MONDO) and mapping to therapeutic areas.
"""

import pandas as pd


def build_ontology_lookups(do_diseases_path, mesh_to_doid_path):
    """Build bidirectional DOID/MESH/MONDO lookup tables.

    Parameters
    ----------
    do_diseases_path : Path or str
        Path to 'do_diseases.csv' file with columns: doid, mondo_name.
    mesh_to_doid_path : Path or str
        Path to 'mesh_to_doid.csv' file with columns: mesh_id, doid.

    Returns
    -------
    tuple of dicts
        (doid_to_name, mesh_to_doid, doid_to_mesh, name_to_doid)
    """
    _do_df = pd.read_csv(do_diseases_path)
    _doid_to_name = dict(zip(_do_df['doid'].astype(str), _do_df['mondo_name'].astype(str)))

    _mesh_df = pd.read_csv(mesh_to_doid_path, on_bad_lines='skip')
    _mesh_df['mesh_clean'] = _mesh_df['mesh_id'].astype(str).str.split(' ').str[0]
    _mesh_to_doid = {}
    for _, _row in _mesh_df.iterrows():
        _m = _row['mesh_clean']
        if _m not in _mesh_to_doid:
            _mesh_to_doid[_m] = str(_row['doid'])

    # Reverse: DOID → MESH (for DRKG disease resolution)
    _doid_to_mesh_t = {}
    for _, _r in _mesh_df.iterrows():
        _d = str(_r['doid'])
        if _d not in _doid_to_mesh_t:
            _doid_to_mesh_t[_d] = _r['mesh_clean']

    # Name → DOID (for FDA approval term matching)
    _name_to_doid_t = {}
    for _, _r in _do_df.iterrows():
        _n = str(_r['mondo_name']).lower().strip()
        if _n not in _name_to_doid_t:
            _name_to_doid_t[_n] = str(_r['doid'])

    return _doid_to_name, _mesh_to_doid, _doid_to_mesh_t, _name_to_doid_t


# Curated MeSH descriptor → disease name supplement for high-frequency DRKG nodes.
# Covers MESH:D descriptors not bridged by the mesh_to_doid cross-mapping file.
# Names are keyword-rich to support downstream area classification.
MESH_NAME_SUPPLEMENT = {
    'MESH:D006973': 'hypertension cardiovascular',
    'MESH:D006333': 'heart failure cardiac',
    'MESH:D001171': 'arthritis joint disease',
    'MESH:D017192': 'hepatitis c infection viral',
    'MESH:D006255': 'rhinitis allergic seasonal hay fever',
    'MESH:D010146': 'pain neuropathic chronic',
    'MESH:D003865': 'major depressive disorder',
    'MESH:D006470': 'hemorrhage cardiovascular',
    'MESH:D010003': 'osteoarthritis joint arthritis',
    'MESH:D006356': 'opioid substance dependence',
    'MESH:D020521': 'stroke cerebrovascular',
    'MESH:D003928': 'diabetic nephropathies diabetes',
    'MESH:D012221': 'rhinitis allergic perennial',
    'MESH:D019698': 'hepatitis c chronic infection viral',
    'MESH:D003866': 'depressive disorder',
    'MESH:D000341': 'agitation psychomotor anxiety',
    'MESH:D011537': 'psoriasis skin autoimmune',
    'MESH:D007319': 'leukemia cancer',
    'MESH:D059413': 'heart failure cardiac',
    'MESH:D000077': 'amyotrophic lateral sclerosis neurodegenerat',
    'MESH:D065631': 'thrombosis cardiovascular',
    'MESH:D009103': 'multiple sclerosis neurology',
    'MESH:D009765': 'obesity metabolic',
    'MESH:D006509': 'hepatitis b infection viral',
    'MESH:D007938': 'leukemia cancer',
    'MESH:D009101': 'multiple myeloma cancer',
    'MESH:D008223': 'lymphoma cancer',
    'MESH:D000755': 'anemia hematology',
    'MESH:D001289': 'attention deficit hyperactivity adhd',
    'MESH:D003920': 'diabetes mellitus metabolic',
    'MESH:D006693': 'hodgkin lymphoma cancer',
    'MESH:D003324': 'coronary artery disease cardiac',
    'MESH:D003072': 'cognitive impairment dementia',
    'MESH:D016212': 'follicular lymphoma cancer neoplasm',
    'MESH:D007715': 'kidney failure renal',
    'MESH:D008103': 'liver cirrhosis hepatic',
    'MESH:D018580': 'anxiety disorder psychiatric',
    'MESH:D001714': 'bipolar disorder psychiatric',
    'MESH:D013167': 'ankylosing spondylitis arthritis autoimmune',
    'MESH:D015535': 'psoriatic arthritis autoimmune',
    'MESH:D015658': 'hiv infection',
    'MESH:D001943': 'breast carcinoma cancer',
    'MESH:D014552': 'urinary tract infection',
    'MESH:D000152': 'acne skin',
    'MESH:D003865': 'major depressive disorder',
    'MESH:D001007': 'anxiety disorders psychiatric',
    'MESH:D015179': 'colorectal cancer neoplasm',
    'MESH:D008175': 'lung cancer neoplasm',
    'MESH:D011467': 'prostate cancer neoplasm',
    'MESH:D009422': 'nervous system disease neurology',
    'MESH:D007239': 'infections bacterial',
    'MESH:D003920': 'diabetes mellitus metabolic diabet',
    'MESH:D005355': 'fibrosis pulmonary respiratory',
    'MESH:D008288': 'malaria infection',
    'MESH:D012559': 'schizophrenia psychiatric',
    'MESH:D004194': 'disease broad',
    'MESH:D016212': 'follicular lymphoma cancer',
    'MESH:D016889': 'endometrial cancer carcinoma',
    'MESH:D019337': 'hematologic neoplasm leukemia cancer',
}


def resolve_disease_name(raw_node_name, kg_name, doid_to_name=None, mesh_to_doid=None, doid_to_mesh=None):
    """Resolve a disease node identifier to a human-readable name for keyword matching.

    For PrimeKG / Hetionet the node name already contains the readable name.
    For DRKG the raw name is a DOID or MESH identifier that must be bridged
    through the ontology lookup hierarchy.

    Parameters
    ----------
    raw_node_name : str
        Disease node identifier or name.
    kg_name : str
        Knowledge graph name (primekg, hetionet, drkg, matrix).
    doid_to_name : dict, optional
        DOID numeric → disease name mapping.
    mesh_to_doid : dict, optional
        MESH ID → DOID mapping.
    doid_to_mesh : dict, optional
        DOID → MESH mapping (for DRKG fallback).

    Returns
    -------
    str
        Human-readable disease name in lowercase.
    """
    if kg_name != 'drkg':
        return raw_node_name.lower()

    node_id = raw_node_name  # e.g. 'DOID:10652' or 'MESH:D006973'

    # Layer 3: curated supplement (highest priority for DRKG MESH nodes)
    if node_id in MESH_NAME_SUPPLEMENT:
        return MESH_NAME_SUPPLEMENT[node_id].lower()

    # Layer 1: DOID → name via Disease Ontology
    if node_id.startswith('DOID:'):
        if doid_to_name:
            return doid_to_name.get(node_id, '').lower()
        return ''

    # Layer 2: MESH → DOID → name via cross-mapping
    if node_id.startswith('MESH:'):
        if mesh_to_doid:
            doid = mesh_to_doid.get(node_id)
            if doid and doid_to_name:
                return doid_to_name.get(doid, '').lower()

    return ''


# ── Extended therapeutic area keyword vocabulary ───────────────────────────────
# Additional terms cover variant disease name forms that arise when converting
# ontology identifiers to human-readable labels (e.g., 'diabet' catches
# 'diabetic nephropathy'; 'infect' catches 'bacterial infectious disease').
AREA_KEYWORDS = {
    'Oncology':       ['cancer', 'carcinoma', 'leukemia', 'lymphoma', 'tumor', 'neoplasm',
                       'melanoma', 'sarcoma', 'myeloma', 'glioma', 'blastoma'],
    'Cardiovascular': ['heart', 'cardiac', 'hypertens', 'coronary', 'arrhyth',
                       'atheroscl', 'cardiomyop', 'hemorrhag', 'thrombos', 'ischemi'],
    'Neurology':      ['alzheimer', 'parkinson', 'epilepsy', 'dementia', 'sclerosis',
                       'neuropath', 'huntington', 'migraine', 'seizure', 'cerebrovascular',
                       'neurodegenerat'],
    'Metabolic':      ['diabet', 'obesity', 'metabolic', 'thyroid', 'hyperlipid', 'gout',
                       'insulin', 'hyperglycemi'],
    'Immunology':     ['autoimmune', 'lupus', 'rheumatoid', 'psoriasis', 'crohn',
                       'colitis', 'allerg', 'arthritis', 'spondylitis', 'eczema', 'atopic'],
    'Infectious':     ['infect', 'bacterial', 'viral', 'tuberculosis', 'hiv',
                       'hepatitis', 'malaria', 'pneumonia', 'influenza', 'herpes'],
    'Respiratory':    ['asthma', 'copd', 'pulmonary', 'respiratory', 'fibrosis', 'bronchit'],
    'Psychiatry':     ['schizophren', 'bipolar', 'depress', 'anxiety', 'psycho',
                       'adhd', 'autism', 'dependence', 'substance', 'agitation'],
}


def assign_disease_to_area(disease_name, area_keywords=None):
    """Assign a disease name to a therapeutic area based on keyword matching.

    Parameters
    ----------
    disease_name : str
        Human-readable disease name (lowercase).
    area_keywords : dict, optional
        Custom area-to-keywords mapping; defaults to AREA_KEYWORDS.

    Returns
    -------
    str or None
        Assigned therapeutic area name, or None if no match.
    """
    if area_keywords is None:
        area_keywords = AREA_KEYWORDS
    for area, kws in area_keywords.items():
        if any(k in disease_name for k in kws):
            return area
    return None
