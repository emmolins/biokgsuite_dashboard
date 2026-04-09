"""API-based validation utilities for Knowledge Graph entities.

Validates drug, gene, disease, and pathway nodes against external APIs
with configurable caching and rate limiting.
"""

import urllib.request
import urllib.parse
import urllib.error
import json as _json_mod
import time
import ssl
from pathlib import Path


class APIValidator:
    """Manages API calls with caching and rate limiting."""

    def __init__(self, cache_path, timeout=30, sleep_interval=0.1):
        """Initialize validator with cache settings.

        Parameters
        ----------
        cache_path : Path or str
            Path to persistent JSON cache file.
        timeout : int
            HTTP request timeout in seconds.
        sleep_interval : float
            Sleep duration between API calls in seconds.
        """
        self.cache_path = Path(cache_path) if not isinstance(cache_path, Path) else cache_path
        self.timeout = timeout
        self.sleep = sleep_interval
        self.ctx = ssl.create_default_context()
        self._api_cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk if exists."""
        if self.cache_path.exists():
            return _json_mod.loads(self.cache_path.read_text())
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(_json_mod.dumps(self._api_cache, indent=1))

    def get_json(self, url, headers=None):
        """Fetch JSON from URL with caching.

        Parameters
        ----------
        url : str
            URL to fetch.
        headers : dict, optional
            Additional HTTP headers.

        Returns
        -------
        dict or None
            Parsed JSON response, or None if request failed.
        """
        if url in self._api_cache:
            return self._api_cache[url]
        hdrs = {'Accept': 'application/json', 'User-Agent': 'BioKGSuite/1.0'}
        if headers:
            hdrs.update(headers)
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(url, headers=hdrs), timeout=self.timeout, context=self.ctx)
            result = _json_mod.loads(resp.read())
        except Exception:
            result = None
        self._api_cache[url] = result
        self._save_cache()
        return result

    def get_status(self, url):
        """Check HTTP status of URL with caching.

        Parameters
        ----------
        url : str
            URL to check.

        Returns
        -------
        int or None
            HTTP status code, or None if request failed.
        """
        cache_key = f'__status__{url}'
        if cache_key in self._api_cache:
            return self._api_cache[cache_key]
        try:
            status = urllib.request.urlopen(
                urllib.request.Request(url, headers={'User-Agent': 'BioKGSuite/1.0'}),
                timeout=self.timeout, context=self.ctx).status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception:
            status = None
        self._api_cache[cache_key] = status
        self._save_cache()
        return status


# ── Validator functions (used in notebook cells) ───────────────────────────────

def validate_gene_protein(df, validator, n_samples=200):
    """Validate gene node IDs against NCBI Entrez.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column.
    validator : APIValidator
        Configured validator instance.
    n_samples : int
        Batch size for NCBI queries.

    Returns
    -------
    int
        Count of valid gene IDs.
    """
    ids, valid = df['node_id'].astype(str).tolist(), 0
    for i in range(0, len(ids), n_samples):
        batch = ids[i:i+n_samples]
        data = validator.get_json('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
                                   '?db=gene&id=' + ','.join(batch) + '&retmode=json')
        if data and 'result' in data:
            valid += sum(1 for gid in batch
                         if data['result'].get(gid, {}) and 'error' not in data['result'].get(gid, {}))
        time.sleep(validator.sleep * 3)
    return valid


def validate_go_term(df, validator, expected_aspect):
    """Validate GO term IDs against QuickGO.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column.
    validator : APIValidator
        Configured validator instance.
    expected_aspect : str
        Expected GO aspect (biological_process, molecular_function, cellular_component).

    Returns
    -------
    int
        Count of valid GO terms.
    """
    valid = 0
    for _, row in df.iterrows():
        raw = str(row['node_id']).strip()
        go_id = raw if raw.upper().startswith('GO:') else (
            'GO:' + f'{int(raw):07d}' if raw.isdigit() else None)
        if go_id is None:
            continue
        data = validator.get_json(
            'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/' + urllib.parse.quote(go_id))
        if (data and 'results' in data and data['results']
                and data['results'][0].get('aspect', '').lower().replace(' ', '_') == expected_aspect):
            valid += 1
        time.sleep(validator.sleep)
    return valid


def validate_ols4(df, validator, ontology, prefix, pad=7):
    """Validate ontology terms via OLS4.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column.
    validator : APIValidator
        Configured validator instance.
    ontology : str
        Ontology name (mondo, doid, uberon, hp, etc.).
    prefix : str
        Ontology prefix (MONDO, DOID, UBERON, HP, etc.).
    pad : int
        Padding for numeric IDs.

    Returns
    -------
    int
        Count of valid terms.
    """
    valid = 0
    for _, row in df.iterrows():
        raw = str(row['node_id']).strip()
        if ':' in raw:
            short_form = prefix + '_' + raw.split(':', 1)[1]
        else:
            try:
                short_form = prefix + '_' + f'{int(raw.split("_", 1)[-1]):0{pad}d}'
            except ValueError:
                continue
        data = validator.get_json(
            'https://www.ebi.ac.uk/ols4/api/ontologies/' + ontology + '/terms?short_form=' + short_form)
        if data and data.get('page', {}).get('totalElements', 0) > 0:
            valid += 1
        time.sleep(validator.sleep)
    return valid


def validate_drug(df, validator):
    """Validate drug node IDs against DrugBank.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column.
    validator : APIValidator
        Configured validator instance.

    Returns
    -------
    int
        Count of valid drugs.
    """
    valid = 0
    for _, row in df.iterrows():
        status = validator.get_status('https://go.drugbank.com/drugs/' + str(row['node_id']))
        if status and 200 <= status < 400:
            valid += 1
        time.sleep(validator.sleep)
    return valid


def validate_pathway(df, validator):
    """Validate pathway node IDs against Reactome.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column.
    validator : APIValidator
        Configured validator instance.

    Returns
    -------
    int
        Count of valid pathways.
    """
    valid = 0
    for _, row in df.iterrows():
        data = validator.get_json('https://reactome.org/ContentService/data/query/' + str(row['node_id']))
        if data and 'dbId' in data:
            valid += 1
        time.sleep(validator.sleep)
    return valid


def validate_exposure(df, validator):
    """Validate exposure node IDs against CTD/PubChem or MeSH.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column.
    validator : APIValidator
        Configured validator instance.

    Returns
    -------
    int
        Count of valid exposures.
    """
    valid = 0
    for _, row in df.iterrows():
        eid = str(row['node_id'])
        data = validator.get_json('https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/'
                                   'Comparative%20Toxicogenomics%20Database/' + eid + '/JSON')
        if data and 'PC_Substances' in data:
            valid += 1
        else:
            mesh = validator.get_json('https://id.nlm.nih.gov/mesh/' + eid + '.json')
            if mesh and '@id' in mesh:
                valid += 1
        time.sleep(validator.sleep)
    return valid


def validate_mesh_disease(df, validator):
    """Validate disease node IDs against the NLM MeSH browser.

    Handles both bare MeSH term IDs (e.g. 'D001234', 'C000123') and
    'MESH:'-prefixed CURIEs, normalising either form before querying the
    NLM Linked Data API (``https://id.nlm.nih.gov/mesh/{id}.json``).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'node_id' column containing MeSH IDs.
    validator : APIValidator
        Configured validator instance.

    Returns
    -------
    int
        Count of valid MeSH IDs.
    """
    valid = 0
    for _, row in df.iterrows():
        raw = str(row['node_id']).strip()
        # Normalise MESH: CURIE to bare ID
        eid = raw.split(':', 1)[1] if ':' in raw else raw
        data = validator.get_json('https://id.nlm.nih.gov/mesh/' + eid + '.json')
        if data and '@id' in data:
            valid += 1
        time.sleep(validator.sleep)
    return valid


KG_VALIDATORS = {
    'primekg': {
        'drug':               ('DrugBank', validate_drug),
        'gene/protein':       ('NCBI',     validate_gene_protein),
        'disease':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'mondo',  'MONDO')),
        'biological_process': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'biological_process')),
        'molecular_function': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'molecular_function')),
        'cellular_component': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'cellular_component')),
        'anatomy':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'uberon', 'UBERON')),
        'effect/phenotype':   ('OLS4',     lambda df, v: validate_ols4(df, v, 'hp',     'HP')),
        'pathway':            ('Reactome', validate_pathway),
        'exposure':           ('PubChem',  validate_exposure),
    },
    'hetionet': {
        'Compound':           ('DrugBank', validate_drug),
        'Gene':               ('NCBI',     validate_gene_protein),
        'Disease':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'doid',   'DOID')),
        'Biological Process': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'biological_process')),
        'Molecular Function': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'molecular_function')),
        'Cellular Component': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'cellular_component')),
        'Anatomy':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'uberon', 'UBERON')),
    },
    'drkg': {
        'Compound':           ('DrugBank', validate_drug),
        'Gene':               ('NCBI',     validate_gene_protein),
        'Disease':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'doid',   'DOID')),
        'Biological Process': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'biological_process')),
        'Molecular Function': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'molecular_function')),
        'Cellular Component': ('QuickGO',  lambda df, v: validate_go_term(df, v, 'cellular_component')),
        'Anatomy':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'uberon', 'UBERON')),
    },
    'openbilink': {
        'Drug':               ('DrugBank', validate_drug),       # Validated via drugbank_id column (PubChem→DrugBank)
        'Gene':               ('NCBI',     validate_gene_protein),
        'Disease':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'doid',   'DOID')),
        'GO':                 ('QuickGO',  lambda df, v: validate_go_term(df, v, 'biological_process')),  # Mixed GO terms
        'Anatomy':            ('OLS4',     lambda df, v: validate_ols4(df, v, 'uberon', 'UBERON')),
        'Phenotype':          ('OLS4',     lambda df, v: validate_ols4(df, v, 'hp',     'HP')),
        'Pathway':            ('Reactome', validate_pathway),
    },
}


def make_val_nodes(nodes_df):
    """Prepare node DataFrame for validation by stripping type prefixes.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        Node DataFrame with 'id' column containing 'Type::ID' format strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'node_id' column containing bare IDs/CURIEs.
    """
    df = nodes_df.copy()
    df['node_id'] = df['id'].astype(str).map(
        lambda s: s.split('::', 1)[1] if '::' in s else s)
    return df
