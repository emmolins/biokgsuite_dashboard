"""Ranking pilot for the redesigned nb09 (KG-quality → LLM repurposing output).

Realistic pharma task: for a target disease, rank a pool of candidate drugs by
repurposing plausibility. The true (post-cutoff) repurposed drug is hidden in a
pool of distractors; we score the rank it receives (MRR, hits@k). The only thing
that varies across arms is the KG-derived evidence attached to each candidate —
so any lift over the no-KG baseline is attributable to the knowledge graph.

This is a PILOT: a few diseases, baseline + one KG, the free model slate, small
sample counts — a smoke test for the full design before scaling on the cluster.

What changed vs the old binary nb09
-----------------------------------
  * Task is listwise ranking, not yes/no plausibility.
  * Positives come from the prospective (time-split) gold standard, so the
    answer is not in the model's parametric memory.
  * Prompt is reason-then-JSON (free-text reasoning precedes the JSON block),
    attributes the KG as a source (not ground truth), permits abstention, and
    carries basis / evidence_agreement / confidence fields.
  * No-KG and missing-candidate evidence are OMITTED entirely — never the word
    "none" — so absence is not read as a negative signal.
  * Candidate order is shuffled across runs (position-bias control); the rank is
    aggregated over shuffles.

Modes
-----
  --mock   No Ollama, no KG load. A mock LLM + mock KG coverage exercise the full
           pipeline (pools, prompt, shuffles, parse, scoring). Runs anywhere.
  --real   Calls Ollama for each model in --models and builds KG blocks from the
           real graph via the repo's loaders. Needs `ollama serve` + pulled models.

Usage
-----
  python scripts/pilot_ranking.py --mock                      # validate plumbing
  python scripts/pilot_ranking.py --real --kg primekg \
      --models llama3.1:8b llama3.3:70b txgemma:27b \
      --n-diseases 3 --pool-size 8 --shuffles 3

Outputs
-------
  results/tables/09_llm_runs/09_pilot_ranking.csv   (one row per disease×cond×model×shuffle)
  Summary table (MRR / hits@k per model × condition) to stdout.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
GOLD   = BASE / 'data' / 'gold_standards'
TABLES = BASE / 'results' / 'tables' / '09_llm_runs'
OUT_CSV = TABLES / '09_pilot_ranking.csv'

OLLAMA_URL = 'http://localhost:11434/api/generate'
TAGS_URL   = 'http://localhost:11434/api/tags'
LETTERS    = list(string.ascii_uppercase)

# ── Refined prompt (the larger, research-grounded version) ───────────────────
PROMPT_HEADER = (
    "You are assisting a drug repurposing scientist. For the target disease below,\n"
    "you are given candidate drugs. Some include evidence retrieved from a\n"
    "biomedical knowledge graph, reported as mechanistic links\n"
    "(drug -> target -> pathway -> disease) and related annotations.\n"
)
PROMPT_INSTRUCTIONS = (
    "Assess each candidate's mechanistic plausibility for this disease using BOTH\n"
    "the knowledge-graph evidence and your own pharmacological knowledge. The\n"
    "knowledge graph reports associations; treat them as one source, not as\n"
    "established truth. Where its evidence is sparse, or conflicts with what you\n"
    "know, say so and weight it accordingly -- do not construct a rationale to fit\n"
    "weak evidence. If the evidence is insufficient to judge a candidate, mark it\n"
    "\"uncertain\" rather than forcing a confident rank.\n\n"
    "Respond with ONLY a single JSON object (no text before or after) of the form:\n"
    "{ \"reasoning\": \"<a few sentences weighing the candidates and flagging any "
    "KG-vs-knowledge conflicts>\",\n"
    "  \"ranking\": [ { \"rank\": 1, \"drug\": \"<letter>\",\n"
    "    \"basis\": \"kg\" | \"prior\" | \"both\",\n"
    "    \"evidence_agreement\": \"consistent\" | \"conflicting\" | \"insufficient\",\n"
    "    \"confidence\": 1-5, \"rationale\": \"<one sentence>\" }, ... ] }\n"
    "Rank every candidate. ALL fields are required for EVERY candidate, even when "
    "no evidence is shown: set basis=\"prior\" when you used only your own "
    "knowledge (\"kg\"/\"both\" when you used the provided evidence), "
    "evidence_agreement=\"insufficient\" when no evidence was given; confidence "
    "(1-5) and a one-sentence rationale are ALWAYS required."
)

# JSON schema for enforced structured output (Ollama `format=`, OpenAI json_schema).
# Required fields can't be dropped -> reliance/confidence become measurable.
RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "drug": {"type": "string"},
                    "basis": {"type": "string", "enum": ["kg", "prior", "both"]},
                    "evidence_agreement": {"type": "string",
                                           "enum": ["consistent", "conflicting", "insufficient"]},
                    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                    "rationale": {"type": "string"},
                },
                "required": ["rank", "drug", "basis", "evidence_agreement",
                             "confidence", "rationale"],
            },
        },
    },
    "required": ["reasoning", "ranking"],
}


def build_prompt(disease_name, disease_id, candidates, disease_profile=None):
    """candidates: list of dicts {letter, drug_name, evidence (str or None)}.
    disease_profile: KG disease dossier shown once in the header (KG arms only)."""
    lines = [PROMPT_HEADER,
             f"\nTarget disease: {disease_name} ({disease_id})"]
    if disease_profile:
        lines.append(f"Disease profile (from the knowledge graph): {disease_profile}")
    lines.append("\nCandidates (listed order is arbitrary):")
    for c in candidates:
        lines.append(f"{c['letter']}. {c['drug_name']}")
        if c['evidence']:                       # omit the line entirely if absent
            lines.append(f"   Drug profile (from the knowledge graph): {c['evidence']}")
    lines.append("")
    lines.append(PROMPT_INSTRUCTIONS)
    return "\n".join(lines)


# ── Ranking-response parser ──────────────────────────────────────────────────
def parse_ranking(text, valid_letters):
    """Return an ordered list of candidate letters, best first. [] if unparseable."""
    if not text:
        return []
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            items = obj.get('ranking', [])
            ranked = []
            for it in items:
                d = str(it.get('drug', '')).strip().upper()
                d = d[0] if d and d[0] in valid_letters else None
                rk = it.get('rank')
                if d and d not in [r[0] for r in ranked]:
                    ranked.append((d, rk if isinstance(rk, (int, float)) else len(ranked) + 1))
            if ranked:
                ranked.sort(key=lambda t: t[1])
                return [d for d, _ in ranked]
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    # Fallback: first mention order of "A.", "B)", "drug A", etc.
    seen = []
    for tok in re.findall(r'\b([A-Z])\b', text.upper()):
        if tok in valid_letters and tok not in seen:
            seen.append(tok)
    return seen


def _as_int(x):
    try:
        return max(1, min(5, int(x)))
    except (ValueError, TypeError):
        return None


def parse_response(text, valid_letters):
    """Like parse_ranking, but also return per-letter reliance fields:
    {letter: {basis, evidence_agreement, confidence}}. Lets us measure whether
    the model used the KG (basis) and over-trusted it, not just the ranking."""
    fields = {}
    ordered = []
    if text:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                ranked = []
                for it in obj.get('ranking', []):
                    d = str(it.get('drug', '')).strip().upper()
                    d = d[0] if d and d[0] in valid_letters else None
                    if not d:
                        continue
                    if d not in fields:
                        fields[d] = {
                            'basis': (str(it.get('basis', '')).lower() or None),
                            'evidence_agreement': (str(it.get('evidence_agreement', '')).lower() or None),
                            'confidence': _as_int(it.get('confidence')),
                            'rationale': (str(it.get('rationale', '')).strip() or None),
                        }
                    if d not in [r[0] for r in ranked]:
                        rk = it.get('rank')
                        ranked.append((d, rk if isinstance(rk, (int, float)) else len(ranked) + 1))
                ranked.sort(key=lambda t: t[1])
                ordered = [d for d, _ in ranked]
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
    if not ordered:
        ordered = parse_ranking(text, valid_letters)   # fallback ordering
    return ordered, fields


def rank_of(positive_letter, ordered_letters, pool_size):
    """1-based rank of the positive; pool_size+1 (i.e. miss) if absent/unparsed."""
    if positive_letter in ordered_letters:
        return ordered_letters.index(positive_letter) + 1
    return pool_size + 1


# ── Real Ollama client ───────────────────────────────────────────────────────
def ollama_models_available():
    import requests
    try:
        data = requests.get(TAGS_URL, timeout=3).json()
        return {m['name'] for m in data.get('models', [])}
    except Exception as e:
        raise SystemExit(f"Ollama not reachable at {TAGS_URL}: {e}")


def ollama_rank(model, prompt, seed, temperature):
    import requests
    payload = {
        'model': model, 'prompt': prompt, 'stream': False,
        'format': RANKING_SCHEMA,                       # enforce structured output
        'options': {'temperature': temperature, 'num_predict': _max_tokens(model),
                    'num_ctx': 8192, 'seed': int(seed)},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get('response', '')
    except Exception as e:
        return f'__ERROR__ {e}'


# ── Hosted API clients (no local download) ──────────────────────────────────
# FREE options (no GPU, no download): name a model "provider:model_id" --
#   groq:llama-3.3-70b-versatile               Groq free tier (the 70B, fast)
#   openrouter:deepseek/deepseek-chat:free     OpenRouter free models
#   gemini:gemini-2.0-flash                    Google AI Studio free tier
# PAID (provider inferred from name): claude-haiku-4-5-20251001, gpt-4o-mini
# Each provider reads its own env-var key (GROQ_API_KEY, OPENROUTER_API_KEY,
# GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY).
API_PROVIDERS = {
    'anthropic':  dict(url='https://api.anthropic.com/v1/messages',
                       key='ANTHROPIC_API_KEY', style='anthropic'),
    'openai':     dict(url='https://api.openai.com/v1/chat/completions',
                       key='OPENAI_API_KEY', style='openai'),
    'groq':       dict(url='https://api.groq.com/openai/v1/chat/completions',
                       key='GROQ_API_KEY', style='openai'),
    'openrouter': dict(url='https://openrouter.ai/api/v1/chat/completions',
                       key='OPENROUTER_API_KEY', style='openai'),
    'gemini':     dict(url='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
                       key='GEMINI_API_KEY', style='openai'),
}


def _provider_model(name):
    """(provider, model_id) for an API model, else (None, None) for local Ollama."""
    prefix = name.split(':', 1)[0]
    if prefix in API_PROVIDERS and ':' in name:
        return prefix, name.split(':', 1)[1]
    low = name.lower()
    if low.startswith('claude'):
        return 'anthropic', name
    if low.startswith(('gpt-', 'o1', 'o3', 'o4')):
        return 'openai', name
    return None, None


def is_api_model(name):
    return _provider_model(name)[0] is not None


# Reasoning models emit a long chain before the JSON, so they need a bigger budget
# or the answer truncates and the JSON fails to parse (seen with gpt-oss-120b).
_REASONING_HINTS = ('gpt-oss', 'o1', 'o3', 'o4', '-r1', 'deepseek-r', 'qwq',
                    'reason', 'think')


def _max_tokens(name):
    n = name.lower()
    return 4000 if any(h in n for h in _REASONING_HINTS) else 1500


def _api_key_for(name):
    import os
    prov, _ = _provider_model(name)
    return os.environ.get(API_PROVIDERS[prov]['key']) if prov else None


_API_FORMAT_CACHE = {}   # (provider, model) -> response_format that worked (None = none)


def _post_with_backoff(url, headers, body, timeout=180, max_retries=4):
    """POST, retrying on HTTP 429 (rate limit) with bounded backoff. Fails fast
    (~30s max) rather than burning minutes per call when a budget is exhausted.
    Returns (response | None, error_str | None)."""
    import requests
    import time
    delay = 3.0
    for _ in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            return None, f'__ERROR__ request: {e}'
        if r.status_code == 429:                     # rate limited: wait and retry
            ra = r.headers.get('retry-after', '')
            try:
                wait = float(ra)
            except ValueError:
                wait = delay
            time.sleep(min(wait, 12))
            delay = min(delay * 2, 12)
            continue
        return r, None
    return None, '__ERROR__ 429 rate-limited (gave up after retries)'


def api_rank(name, prompt, temperature):
    """Call a hosted API with 429 backoff. One request per call: prefer
    json_object, fall back to plain once, and cache whichever works per model so
    later calls don't waste requests (which was tripping the rate limit)."""
    import os
    prov, model = _provider_model(name)
    cfg = API_PROVIDERS.get(prov)
    if not cfg:
        return '__ERROR__ not an API model'
    key = os.environ.get(cfg['key'])
    if not key:
        return f'__ERROR__ no key ({cfg["key"]})'
    mx = _max_tokens(name)

    if cfg['style'] == 'anthropic':
        r, err = _post_with_backoff(cfg['url'],
            {'x-api-key': key, 'anthropic-version': '2023-06-01',
             'content-type': 'application/json'},
            {'model': model, 'max_tokens': mx, 'temperature': temperature,
             'messages': [{'role': 'user', 'content': prompt}]})
        if err:
            return err
        if r.status_code >= 400:
            return f'__ERROR__ {r.status_code} {r.text[:160]}'
        try:
            return ''.join(b.get('text', '') for b in r.json().get('content', []))
        except Exception as e:
            return f'__ERROR__ parse {e}'

    # OpenAI-compatible (openai/groq/openrouter/gemini).
    headers = {'Authorization': f'Bearer {key}', 'content-type': 'application/json'}
    base = {'model': model, 'max_tokens': mx, 'temperature': temperature,
            'messages': [{'role': 'user', 'content': prompt}]}
    ck = (prov, model)
    formats = ([_API_FORMAT_CACHE[ck]] if ck in _API_FORMAT_CACHE
               else [{'type': 'json_object'}, None])     # None = no response_format
    last = '__ERROR__ no response'
    for fmt in formats:
        body = {**base, **({'response_format': fmt} if fmt else {})}
        r, err = _post_with_backoff(cfg['url'], headers, body)
        if err:
            return err                                   # 429 exhausted / network error
        if r.status_code >= 400:
            last = f'__ERROR__ {r.status_code} {r.text[:160]}'
            continue                                     # format rejected -> try next
        _API_FORMAT_CACHE[ck] = fmt                      # remember the working format
        try:
            return r.json()['choices'][0]['message']['content']
        except Exception as e:
            return f'__ERROR__ parse {e}'
    return last


# ── Mock LLM (validates plumbing without Ollama) ─────────────────────────────
def mock_rank(candidates, seed):
    """Simulate a discerning model: candidates WITH evidence float up, with noise.
    No-KG arm -> nobody has evidence -> ~random (baseline). KG arm -> evidence
    helps -> positive (more likely to carry evidence) ranks higher."""
    rng = np.random.default_rng(seed)
    scored = []
    for c in candidates:
        base = 1.0 if c['evidence'] else 0.0
        scored.append((base + rng.normal(0, 0.5), c['letter']))
    scored.sort(key=lambda t: -t[0])
    ordered = [l for _, l in scored]
    # Emit a realistic-looking response so the parser is exercised too.
    items = ", ".join(
        f'{{"rank": {i+1}, "drug": "{l}", "basis": "both", '
        f'"evidence_agreement": "consistent", "confidence": 3, "rationale": "mock"}}'
        for i, l in enumerate(ordered))
    return f'Reasoning: mock.\n{{"ranking": [{items}]}}'


# ── KG evidence block (real mode; reuses repo loaders) ───────────────────────
# BALANCED DUAL-DOSSIER version. Instead of
# extracting the drug->disease bridge (which pre-computes the answer == leakage +
# cherry-picking), we extract two neutral, query-INDEPENDENT profiles a pharma
# analyst would actually pull:
#   * a per-candidate DRUG dossier  (targets, known indications, side effects)
#   * one DISEASE profile per query (associated genes, phenotypes), shown once
#     in the prompt header.
# The model must connect them itself. Same fixed schema for every candidate
# (symmetry == not cherry-picked). High-degree nodes are handled by SPECIFICITY
# ranking — neighbours sorted by their OWN degree ascending, top-`cap` kept — so
# 1000+-neighbour hubs sink to the bottom without a brittle threshold. The
# drug's indication for the TARGET disease is masked (no answer leak).
# `bridge_mode='mechanism_only'` suppresses the disease profile (model bridges
# from its own disease knowledge) as a paired arm.
SLOT_LABEL = {
    'drug_target':       'targets',
    'drug_pathway':      'modulates pathways',
    'drug_effect':       'side effects',
    'drug_disease':      'approved/known for',
    'target_disease':    'associated genes',
    'disease_phenotype': 'phenotypes',
}
DRUG_SLOTS_ORDER    = ('drug_target', 'drug_pathway', 'drug_disease', 'drug_effect')
DISEASE_SLOTS_ORDER = ('target_disease', 'disease_phenotype')


def _load_slot_map(kg_name):
    """raw predicate -> canonical slot, from data/kg_slot_maps.yaml."""
    import yaml
    with open(BASE / 'data' / 'kg_slot_maps.yaml') as f:
        full = yaml.safe_load(f) or {}
    pred_to_slot = {}
    for slot, preds in (full.get(kg_name) or {}).items():
        for p in (preds or []):
            pred_to_slot[str(p)] = slot
    return pred_to_slot


def make_kg_block_fn(kg_name, *, bridge_mode='full', cap=12, anonymize=False,
                     anonymize_genes=False):
    """Return (drug_dossier_fn, disease_profile_fn).
      drug_dossier_fn(drug_id, drug_name, disease_id) -> str | None  (per candidate)
      disease_profile_fn(disease_id)                  -> str         (once per query)
    Lazy imports so --mock needs none of this."""
    sys.path.insert(0, str(BASE / 'src'))
    from loading import load_kg, load_config, find_config
    from graph_utils import build_lookup_maps
    sys.path.insert(0, str(BASE / 'scripts'))
    from pilot_packaging import make_resolvers, build_edge_index, load_crosswalks
    from collections import defaultdict

    config = load_config(find_config(BASE))
    kg_df, nodes_df = load_kg(kg_name, config)
    mp = build_lookup_maps(nodes_df)
    name_map = mp['node_name_map']
    idx_to_type = dict(zip(nodes_df['idx'].astype(int), nodes_df['type'].astype(str)))
    et = config['knowledge_graphs'][kg_name]['entity_types']
    drug_t, disease_t = et.get('Drug'), et.get('Disease')
    sep = config['knowledge_graphs'][kg_name].get('disease_id_separator')

    pred_to_slot = _load_slot_map(kg_name)
    if not pred_to_slot:
        print(f"  ⚠  no slot map for '{kg_name}' — KG blocks will all be empty.")

    def slot_of(rel):
        return pred_to_slot.get(str(rel))

    id_to_idx = defaultdict(list)
    for raw, idx in zip(nodes_df['id'].astype(str), nodes_df['idx'].astype(int)):
        nt = idx_to_type.get(int(idx))
        keys = {raw, raw.split('::', 1)[-1] if '::' in raw else raw}
        if sep and nt == disease_t and sep in raw:
            keys.update(p for p in raw.split(sep) if p)
        for k in keys:
            id_to_idx[k].append((int(idx), nt))

    to_drug_ids, to_disease_ids = make_resolvers(config, kg_name, xw=load_crosswalks())
    edge_index = build_edge_index(kg_df)
    degree = {u: len(adj) for u, adj in edge_index.items()}

    def _resolve(cands, want):
        for c in cands:
            for idx, t in id_to_idx.get(c, []):
                if t == want:
                    return idx
        return None

    def _neighbours_by_slot(node):
        out = defaultdict(list)
        for v, r in edge_index.get(node, []):
            s = slot_of(r)
            if s:
                out[s].append(v)
        return out

    def _topn(nodes, exclude=(), recode=None):
        """Top-`cap` neighbours by SPECIFICITY (own-degree ascending). This is the
        neutral, query-independent cap that demotes 1000+-neighbour hubs.
        `recode` (optional) maps each name -> code for gene anonymization."""
        seen, res = set(exclude), []
        for v in sorted(nodes, key=lambda n: degree.get(n, 0)):
            if v in seen:
                continue
            seen.add(v)
            nm = name_map.get(v, str(v))
            res.append(recode(nm) if recode else nm)
            if len(res) >= cap:
                break
        return res

    # Per-disease gene-code maps so the SAME gene gets the SAME code in the drug
    # dossier and the disease profile within a query (preserves the overlap
    # signal while hiding biology that lets the model re-identify the drug).
    _gene_code_maps = defaultdict(dict)

    def _gene_coder(disease_id):
        m = _gene_code_maps[disease_id]
        def code(name):
            if name not in m:
                m[name] = f"Gene-{len(m) + 1}"
            return m[name]
        return code

    def _sections(by_slot, slot_order, exclude_per_slot=None):
        exclude_per_slot = exclude_per_slot or {}
        parts = []
        for slot in slot_order:
            names = _topn(by_slot.get(slot, []), exclude=exclude_per_slot.get(slot, ()))
            if names:
                parts.append(f"{SLOT_LABEL.get(slot, slot)}: {', '.join(names)}")
        return "; ".join(parts) if parts else None

    def drug_dossier(drug_id, drug_name, disease_id):
        d_idx = _resolve(to_drug_ids(drug_id), drug_t)
        if d_idx is None:
            return None
        by = _neighbours_by_slot(d_idx)
        recode = _gene_coder(disease_id) if anonymize_genes else None
        parts = []
        tg = _topn(by.get('drug_target', []), recode=recode)
        if tg:
            parts.append(f"{SLOT_LABEL['drug_target']}: {', '.join(tg)}")
        pw = _topn(by.get('drug_pathway', []))
        if pw:
            parts.append(f"{SLOT_LABEL['drug_pathway']}: {', '.join(pw)}")
        # Indications: kept only when nothing is anonymized; they identify the drug
        # AND are the contamination vector, so anonymize/anonymize_genes drop them.
        if not anonymize and not anonymize_genes:
            x_idx = _resolve(to_disease_ids(disease_id), disease_t)
            ind = _topn(by.get('drug_disease', []),
                        exclude=({x_idx} if x_idx is not None else set()))
            if ind:
                parts.append(f"{SLOT_LABEL['drug_disease']}: {', '.join(ind)}")
        # Side effects can re-identify a drug and aren't part of the matching
        # signal, so drop them under gene anonymization.
        if not anonymize_genes:
            se = _topn(by.get('drug_effect', []))
            if se:
                parts.append(f"{SLOT_LABEL['drug_effect']}: {', '.join(se)}")
        return "; ".join(parts) if parts else None

    def disease_profile(disease_id):
        if bridge_mode == 'mechanism_only':
            return ''
        x_idx = _resolve(to_disease_ids(disease_id), disease_t)
        if x_idx is None:
            return ''
        by = _neighbours_by_slot(x_idx)
        recode = _gene_coder(disease_id) if anonymize_genes else None
        parts = []
        g = _topn(by.get('target_disease', []), recode=recode)
        if g:
            parts.append(f"{SLOT_LABEL['target_disease']}: {', '.join(g)}")
        if not anonymize_genes:                 # phenotypes can name the disease
            ph = _topn(by.get('disease_phenotype', []))
            if ph:
                parts.append(f"{SLOT_LABEL['disease_phenotype']}: {', '.join(ph)}")
        return "; ".join(parts)

    return drug_dossier, disease_profile


def mock_kg_block_fn():
    """Mock coverage: positive + ~40% of distractors get a fake block."""
    def block_for(drug_id, drug_name, disease_id, _is_pos=False):
        return None  # replaced per-candidate in build_query for mock
    return block_for


# ── Candidate pools from the prospective gold standard ───────────────────────
def load_positives(gold_file):
    df = pd.read_csv(gold_file, sep='\t')
    df = df.rename(columns={'mondo_id': 'disease_id'}) if 'disease_id' not in df.columns else df
    need = {'drug_id', 'drug_name', 'disease_name', 'disease_id'}
    if 'drug_id' not in df.columns and 'drugbank_id' in df.columns:
        df = df.rename(columns={'drugbank_id': 'drug_id'})
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{gold_file.name} missing columns: {missing}")
    cols = ['drug_id', 'drug_name', 'disease_name', 'disease_id']
    out = df[cols + (['therapeutic_area'] if 'therapeutic_area' in df.columns else [])].dropna(subset=cols)
    if 'therapeutic_area' not in out.columns:
        out = out.assign(therapeutic_area='unknown')
    return out.reset_index(drop=True)


def build_queries(pos_df, n_diseases, pool_size, seed, *, unique_drug=True, stratify=True):
    """Select query diseases from the gold set.
      unique_drug : each drug is the positive for at most one query (kills the
                    pembrolizumab/upadacitinib repetition).
      stratify    : draw diseases round-robin across therapeutic_area so the
                    sample isn't ~80% oncology.
    """
    rng = np.random.default_rng(seed)
    df = pos_df.copy()
    if unique_drug:
        df = df.drop_duplicates('drug_id')
    df = df.drop_duplicates('disease_name').reset_index(drop=True)
    n = min(n_diseases, len(df))

    if stratify and df['therapeutic_area'].nunique() > 1:
        areas = sorted(df['therapeutic_area'].unique())
        pools = {a: list(rng.permutation(df.index[df['therapeutic_area'] == a].to_numpy()))
                 for a in areas}
        picks = []
        while len(picks) < n and any(pools.values()):
            for a in areas:                       # round-robin: one per area per cycle
                if pools[a]:
                    picks.append(int(pools[a].pop()))
                    if len(picks) >= n:
                        break
        chosen = df.loc[picks]
    else:
        chosen = df.iloc[rng.choice(len(df), size=n, replace=False)]

    all_drugs = df.drop_duplicates('drug_id')[['drug_id', 'drug_name']]
    queries = []
    for _, row in chosen.iterrows():
        distractor_pool = all_drugs[all_drugs['drug_id'] != row['drug_id']]
        k = min(pool_size - 1, len(distractor_pool))
        dis = distractor_pool.iloc[rng.choice(len(distractor_pool), size=k, replace=False)]
        cands = [{'drug_id': row['drug_id'], 'drug_name': row['drug_name'], 'is_pos': True}]
        cands += [{'drug_id': d.drug_id, 'drug_name': d.drug_name, 'is_pos': False}
                  for d in dis.itertuples()]
        queries.append({'disease_name': row['disease_name'], 'disease_id': row['disease_id'],
                        'area': row['therapeutic_area'], 'candidates': cands})
    return queries


def assign_letters_and_evidence(cands, block_for, disease_id, condition, mock, rng,
                                code_map=None):
    """Attach a letter and (for KG conditions) an evidence string to each candidate.
    Returns (candidate_view, positive_letter). Order is shuffled here.
    code_map (drug_id -> "Drug-N") anonymizes the displayed drug name when given."""
    order = list(range(len(cands)))
    rng.shuffle(order)
    view, pos_letter = [], None
    for new_i, orig_i in enumerate(order):
        c = cands[orig_i]
        letter = LETTERS[new_i]
        evidence = None
        if condition != 'no_kg':
            if mock:
                # mock coverage: positive 90%, distractors 40%
                p = 0.9 if c['is_pos'] else 0.4
                evidence = "mechanistic link present" if rng.random() < p else None
            else:
                evidence = block_for(c['drug_id'], c['drug_name'], disease_id)
        if c['is_pos']:
            pos_letter = letter
        name = code_map[c['drug_id']] if code_map else c['drug_name']
        view.append({'letter': letter, 'drug_name': name, 'evidence': evidence})
    return view, pos_letter


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mock', action='store_true', help='No Ollama / no KG load.')
    ap.add_argument('--real', action='store_true', help='Call Ollama + load KG.')
    ap.add_argument('--preflight', action='store_true',
                    help='Load the KG and report disease resolution + path coverage; no model calls.')
    ap.add_argument('--kg', default='primekg')
    ap.add_argument('--kgs', nargs='+', default=None,
                    help='Run several KGs and report per-KG (overrides --kg). '
                         'e.g. --kgs primekg hetionet drkg openbilink biokg matrix. '
                         'KGs are loaded one at a time; no_kg baseline runs once.')
    ap.add_argument('--gold', default=str(GOLD / 'expanded_prospective_gold_standard.tsv'))
    ap.add_argument('--models', nargs='+',
                    default=['llama3.1:8b', 'llama3.3:70b', 'txgemma:27b'])
    ap.add_argument('--conditions', nargs='+', default=['no_kg', 'kg'])
    ap.add_argument('--bridge-mode', choices=['full', 'mechanism_only'], default='full',
                    help='full = drug dossier + disease profile; mechanism_only = drug dossier only.')
    ap.add_argument('--cap', type=int, default=12,
                    help='Max items per dossier section, kept by specificity (own-degree '
                         'ascending) so 1000+-neighbour hubs are demoted, not hard-cut.')
    ap.add_argument('--anonymize', action='store_true',
                    help='Contamination control: replace drug names with codes and drop the '
                         'indications field, so the model must reason from KG mechanism rather '
                         'than recall memorized drug-disease approvals. Genes/disease stay real.')
    ap.add_argument('--anonymize-genes', action='store_true',
                    help='Stricter: also code gene names per-query (consistent across drug and '
                         'disease so the overlap signal is kept) and drop side-effects/phenotypes '
                         '-> pure structural-matching test, closes the gene-signature leak. '
                         'Implies --anonymize.')
    ap.add_argument('--n-diseases', type=int, default=3)
    ap.add_argument('--no-stratify', action='store_true',
                    help='Disable round-robin sampling across therapeutic areas (default: on).')
    ap.add_argument('--allow-repeat-drugs', action='store_true',
                    help='Allow the same drug to be the positive for multiple diseases (default: off).')
    ap.add_argument('--pool-size', type=int, default=8)
    ap.add_argument('--shuffles', type=int, default=3, help='Order permutations / self-consistency.')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--api-delay', type=float, default=1.0,
                    help='Seconds to wait before each hosted-API call (paces requests to '
                         'stay under free-tier rate limits). Raise to ~3-4 if you still see 429s.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(OUT_CSV),
                    help='Output CSV path. Use distinct paths to run models/KGs separately '
                         '(avoids the rate limit) without overwriting, then concatenate.')
    args = ap.parse_args()

    if args.anonymize_genes:
        args.anonymize = True            # gene coding only makes sense with coded drug names
    if not (args.mock or args.real):
        args.mock = True
        print("(no mode given -> defaulting to --mock)\n")
    kg_list = args.kgs if args.kgs else [args.kg]

    pos_df = load_positives(Path(args.gold))
    queries = build_queries(pos_df, args.n_diseases, args.pool_size, args.seed,
                            unique_drug=not args.allow_repeat_drugs,
                            stratify=not args.no_stratify)
    print(f"Pilot: {len(queries)} diseases x {args.conditions} x {len(args.models)} models "
          f"x {args.shuffles} shuffles  (pool={args.pool_size})")
    for q in queries:
        print(f"  - [{q['area']}] {q['disease_name']} ({q['disease_id']})  "
              f"true drug: {q['candidates'][0]['drug_name']}")

    # ── Preflight: KG resolution + dossier coverage per KG, no model calls ────
    if args.preflight:
        print(f"\n=== Preflight: dossier coverage per KG ({len(queries)} diseases) ===")
        print(f"  (bridge_mode={args.bridge_mode}, cap={args.cap}, "
              f"anonymize={args.anonymize}, anonymize_genes={args.anonymize_genes})\n")
        print(f"  {'KG':<12} {'positives':>10} {'candidates':>12}")
        for kg in kg_list:
            block_for, disease_profile_fn = make_kg_block_fn(
                kg, bridge_mode=args.bridge_mode, cap=args.cap, anonymize=args.anonymize,
                anonymize_genes=args.anonymize_genes)
            n_pos_block = n_cand = n_cand_block = 0
            for q in queries:
                pos = q['candidates'][0]
                if block_for(pos['drug_id'], pos['drug_name'], q['disease_id']):
                    n_pos_block += 1
                n_cand += len(q['candidates'])
                n_cand_block += sum(bool(block_for(c['drug_id'], c['drug_name'], q['disease_id']))
                                    for c in q['candidates'])
            print(f"  {kg:<12} {f'{n_pos_block}/{len(queries)}':>10} {f'{n_cand_block}/{n_cand}':>12}")
            del block_for, disease_profile_fn
        print("\nNo model calls made (preflight). Drop --preflight and add --real to run the LLMs.")
        return

    if args.real:
        ollama_wanted = [m for m in args.models if not is_api_model(m)]
        if ollama_wanted:
            avail = ollama_models_available()
            missing = [m for m in ollama_wanted if m not in avail]
            if missing:
                print(f"  ⚠  not pulled, skipping: {missing}  "
                      f"(pull them, then re-run to include them)")
                args.models = [m for m in args.models if is_api_model(m) or m in avail]
        for m in [m for m in args.models if is_api_model(m)]:
            if not _api_key_for(m):
                print(f"  ⚠  no API key for {m} (set the provider key env var); skipping")
                args.models = [x for x in args.models if x != m]
        if not args.models:
            raise SystemExit("No usable models: pull an Ollama model or set an API key.")

    def _code_map(q):
        if not args.anonymize:
            return None
        ids = sorted({c['drug_id'] for c in q['candidates']})
        return {did: f"Drug-{i + 1}" for i, did in enumerate(ids)}

    def run_arm(kg_tag, block_for, disease_profile_fn, conds):
        """Run the model loop for `conds`; rows tagged with kg_tag (or 'none' for no_kg)."""
        out, n_pos_ev, n_cells = [], 0, 0
        rl_streak = 0                                # consecutive rate-limit failures
        for qi, q in enumerate(queries):
            code_map = _code_map(q)
            for cond in conds:
                for model in args.models:
                    for s in range(args.shuffles):
                        rng = np.random.default_rng(args.seed + 1000 * qi + 100 * s
                                                    + hash((cond, model, kg_tag)) % 97)
                        view, pos_letter = assign_letters_and_evidence(
                            q['candidates'], block_for, q['disease_id'], cond, args.mock, rng,
                            code_map=code_map)
                        if cond != 'no_kg':
                            n_cells += 1
                            pos_ev = next(c['evidence'] for c in view if c['letter'] == pos_letter)
                            n_pos_ev += int(bool(pos_ev))
                        dprof = (disease_profile_fn(q['disease_id'])
                                 if (cond != 'no_kg' and disease_profile_fn) else None)
                        prompt = build_prompt(q['disease_name'], q['disease_id'], view,
                                              disease_profile=dprof)
                        valid = [c['letter'] for c in view]
                        if args.mock:
                            resp = mock_rank(view, args.seed + s + qi)
                        elif is_api_model(model):
                            if args.api_delay:
                                import time
                                time.sleep(args.api_delay)   # pace to stay under rate limits
                            resp = api_rank(model, prompt, args.temperature)
                        else:
                            resp = ollama_rank(model, prompt, args.seed + s, args.temperature)
                        err = (resp[9:].strip()[:120] if isinstance(resp, str)
                               and resp.startswith('__ERROR__') else None)
                        rl_streak = rl_streak + 1 if (err and '429' in err) else 0
                        ordered, fields = parse_response(resp, valid)
                        r = rank_of(pos_letter, ordered, args.pool_size)
                        bvals = [v['basis'] for v in fields.values() if v.get('basis')]
                        frac_kg = (sum(b in ('kg', 'both') for b in bvals) / len(bvals)
                                   if bvals else None)
                        confs = [v['confidence'] for v in fields.values()
                                 if isinstance(v.get('confidence'), int)]
                        pf = fields.get(pos_letter, {})
                        out.append({
                            'disease': q['disease_name'],
                            'kg': (kg_tag if cond != 'no_kg' else 'none'),
                            'condition': cond, 'model': model, 'shuffle': s,
                            'pool_size': args.pool_size, 'positive_letter': pos_letter,
                            'rank': r, 'reciprocal_rank': 1.0 / r,
                            'hit@1': int(r <= 1), 'hit@3': int(r <= 3), 'hit@5': int(r <= 5),
                            'parsed': int(bool(ordered)), 'error': err,
                            'pos_basis': pf.get('basis'), 'pos_agreement': pf.get('evidence_agreement'),
                            'pos_confidence': pf.get('confidence'), 'pos_rationale': pf.get('rationale'),
                            'frac_basis_kg': frac_kg,
                            'mean_confidence': (sum(confs) / len(confs) if confs else None),
                        })
                        if rl_streak >= 8:           # budget exhausted -> stop wasting time
                            print("\n⚠  Rate limit exhausted (8 calls failed in a row). "
                                  "Aborting early with partial results — the free-tier budget "
                                  "is likely used up. Retry after it resets, or switch --models "
                                  "to another provider (gemini: / openrouter:).")
                            return out, n_pos_ev, n_cells
        return out, n_pos_ev, n_cells

    rows, coverage = [], {}
    want_kg = any(c != 'no_kg' for c in args.conditions)
    if args.mock:
        rr, npev, ncell = run_arm('mock', None, None, args.conditions)
        rows += rr
        coverage['mock'] = (npev, ncell)
    else:
        if 'no_kg' in args.conditions:                      # shared baseline, runs once
            rr, _, _ = run_arm('none', None, None, ['no_kg'])
            rows += rr
        if want_kg:
            for kg in kg_list:                              # one KG in memory at a time
                print(f"\nLoading KG '{kg}' (cap={args.cap}, anonymize={args.anonymize}, "
                      f"anonymize_genes={args.anonymize_genes}) ...")
                block_for, disease_profile_fn = make_kg_block_fn(
                    kg, bridge_mode=args.bridge_mode, cap=args.cap,
                    anonymize=args.anonymize, anonymize_genes=args.anonymize_genes)
                rr, npev, ncell = run_arm(kg, block_for, disease_profile_fn, ['kg'])
                rows += rr
                coverage[kg] = (npev, ncell)
                del block_for, disease_profile_fn

    res = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    summary = (res.groupby(['model', 'kg', 'condition'])
               .agg(MRR=('reciprocal_rank', 'mean'),
                    hit1=('hit@1', 'mean'), hit3=('hit@3', 'mean'), hit5=('hit@5', 'mean'),
                    reliance=('frac_basis_kg', 'mean'), conf=('mean_confidence', 'mean'),
                    parse_rate=('parsed', 'mean'), n=('rank', 'size'))
               .round(3).reset_index().sort_values(['model', 'condition', 'kg']))

    print("\n=== Pilot results by KG (higher MRR / hits = better) ===")
    print(summary.to_string(index=False))
    if 'error' in res and res['error'].notna().any():
        n_err = int(res['error'].notna().sum())
        print(f"\n⚠  {n_err}/{len(res)} calls errored (counted as misses — they deflate MRR). "
              f"By model:")
        print(res[res['error'].notna()].groupby('model')
              .agg(errors=('error', 'size'), example=('error', 'first')).to_string())
    if any(nc for _, nc in coverage.values()):
        print("\nPositive coverage by KG (cells where the true drug had a dossier):")
        for kg, (npev, ncell) in coverage.items():
            if ncell:
                print(f"  {kg:<12} {npev}/{ncell}")
    print(f"\nWrote {len(res)} rows -> {out_path}")


if __name__ == '__main__':
    main()
