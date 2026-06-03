"""LLM prompting strategy for drug-disease plausibility (notebook 09).

This module exposes a SINGLE prompting strategy, ``LLMPrompt`` (registered
under the name ``'llm_prompt'``), which combines the two restricted-schema
strategies from earlier iterations of notebook 09:

  * ``zero_shot_direct`` — single-call, Yes/No + 1-5 confidence baseline
    (Sivarajkumar et al. 2024, JMIR Med Inf).
  * ``structured_json``  — JSON output enforced via Ollama ``format='json'``
    (DrugReX 2025; DrugReAlign 2024, BMC Biology).

The combined prompt:
  - Enforces JSON output (parse rate ≈ 100%).
  - Emits ``{answer, confidence, reasoning}`` — drops the ``contradictions``
    field that ``structured_json`` previously surfaced, keeping the prompt
    light and matching the field set of ``zero_shot_direct`` plus a brief
    one-sentence rationale.

Strategy.execute() returns a single (pred, confidence, response) tuple per
call so the parquet schema in nb09 stays one row per logical experiment
cell. Strategy.n_calls (used for cost accounting) is 1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


# ── Response parser ──────────────────────────────────────────────────────────

_YES_RE  = re.compile(r'\b(answer|final\s*answer)\s*[:\-]?\s*y(?:es)?\b', re.I)
_NO_RE   = re.compile(r'\b(answer|final\s*answer)\s*[:\-]?\s*no\b',     re.I)
_CONF_RE = re.compile(r'confidence\s*[:\-]?\s*([1-5])', re.I)


def _parse_yes_no_conf(text: str) -> tuple[Optional[int], Optional[int]]:
    """Fallback text parser: 'Answer: Yes, Confidence: 4' style."""
    if not text:
        return None, None
    head = text.strip()[:200].lower()
    pred = (1 if (_YES_RE.search(head) or head.lstrip().startswith('yes'))
            else 0 if (_NO_RE.search(head) or head.lstrip().startswith('no'))
            else None)
    m = _CONF_RE.search(text)
    conf = int(m.group(1)) if m else None
    return pred, conf


def _parse_json_response(text: str) -> tuple[Optional[int], Optional[int]]:
    """Primary parser: extract {answer, confidence} from JSON in response."""
    if not text:
        return None, None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return _parse_yes_no_conf(text)  # graceful fallback
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _parse_yes_no_conf(text)
    ans = str(obj.get('answer', '')).strip().lower()
    pred = 1 if ans in ('yes', 'true', 'y') else 0 if ans in ('no', 'false', 'n') else None
    conf_raw = obj.get('confidence')
    try:
        conf = max(1, min(5, int(conf_raw))) if conf_raw is not None else None
    except (ValueError, TypeError):
        conf = None
    return pred, conf


# ── KG-context block helper ──────────────────────────────────────────────────

def _kg_block(kg_text: str) -> str:
    """Format KG text as a labeled block; explicit empty marker if absent."""
    return (f'Knowledge graph context:\n{kg_text}' if kg_text
            else 'No knowledge graph context provided.')


# ── Base strategy ────────────────────────────────────────────────────────────

@dataclass
class Strategy:
    """Abstract base. Subclass and implement execute()."""
    name: str
    n_calls: int = 1  # informational — used for cost accounting only
    description: str = ''
    hypothesis: str = ''

    def execute(self, query_fn: Callable, drug: str, disease: str,
                kg_text: str, seed: int) -> dict:
        """Run the strategy and return a dict with at least:
          pred (0/1/None), confidence (1-5/None), response (str — the raw
          text), error (str/None), n_calls_made (int).
        ``query_fn`` is a callable (prompt, seed, **decoding_kwargs) → dict
        with keys 'response' and 'error' (matches nb09's ollama_query).
        """
        raise NotImplementedError


# ── The single combined prompting strategy ───────────────────────────────────

class LLMPrompt(Strategy):
    """Combined zero_shot_direct + structured_json prompt.

    JSON output enforced via Ollama ``format='json'``. Schema:
        {
          "answer": "Yes" | "No",
          "confidence": <integer 1-5>,
          "reasoning": "<one sentence>"
        }
    No ``contradictions`` field (dropped from the original structured_json
    schema to keep the prompt lean while preserving a brief rationale).
    """

    def __init__(self):
        super().__init__(
            name='llm_prompt', n_calls=1,
            description=('Single-call JSON-formatted yes/no with 1-5 '
                         'confidence and a one-sentence reasoning. '
                         'Combines zero_shot_direct (Sivarajkumar 2024) '
                         'with the JSON-schema enforcement of '
                         'structured_json (DrugReX 2025, DrugReAlign 2024).'),
            hypothesis=('JSON-enforced output gives ~100% parse rate and '
                        'lets the model emit a brief rationale without the '
                        'verbosity of a contradictions field.'),
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a yes/no question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n\n'
            'Respond ONLY with a JSON object of the form:\n'
            '{\n'
            '  "answer": "Yes" | "No",\n'
            '  "confidence": <integer 1-5>,\n'
            '  "reasoning": "<one sentence>"\n'
            '}\n'
            'Do not include any text outside the JSON.'
        )
        # Ollama supports format='json' to enforce JSON decoding.
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=200,
                       format='json')
        pred, conf = _parse_json_response(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── Registry ─────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, Strategy] = {
    s.name: s for s in [
        LLMPrompt(),
    ]
}

# Default for nb09 — kept as a list for API compatibility with earlier code,
# even though there is only one strategy now.
DEFAULT_STRATEGIES_FOR_NB09 = ['llm_prompt']
DEFAULT_STRATEGY_ORDER = list(STRATEGIES.keys())


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f'Unknown strategy {name!r}. '
                       f'Known: {list(STRATEGIES.keys())}')
    return STRATEGIES[name]


def total_calls_per_cell(strategy_names: list[str]) -> int:
    """Sum of n_calls across the listed strategies — for cost estimation."""
    return sum(get_strategy(n).n_calls for n in strategy_names)
