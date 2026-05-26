"""LLM prompting strategies for drug-disease plausibility (notebook 09).

Each strategy is a self-contained object that knows how to:
  (a) build one or more prompts for a given (drug, disease, kg_text),
  (b) request decoding overrides (temperature, max_tokens, json mode),
  (c) aggregate multi-call responses into a single (pred, confidence).

The main nb09 loop is unchanged in shape — it iterates over
(kg × strategy × condition × pair × reseed), calls
``strategy.execute(query_fn, drug, disease, kg_text, seed)`` once per cell,
and stores the returned tuple as a single row in responses.parquet.

Strategy.execute() is what hides the multi-call complexity:
- self_consistency_5 makes 5 calls internally and votes
- prompt_then_verify makes 2 calls (answer, then critique)
- step_back makes 2 calls (principles, then specific question)
- everything else is 1 call

This keeps the parquet schema simple (one row per logical experiment cell)
while letting us count actual call cost separately via Strategy.n_calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Shared response parsers ──────────────────────────────────────────────────

_YES_RE   = re.compile(r'\b(answer|final\s*answer)\s*[:\-]?\s*y(?:es)?\b', re.I)
_NO_RE    = re.compile(r'\b(answer|final\s*answer)\s*[:\-]?\s*no\b',     re.I)
_CONF_RE  = re.compile(r'confidence\s*[:\-]?\s*([1-5])', re.I)
_PROB_RE  = re.compile(r'probability\s*[:\-]?\s*(\d{1,3})(?:\s*[%/])?', re.I)


def _parse_yes_no_conf(text: str) -> tuple[Optional[int], Optional[int]]:
    """Original parser: 'Answer: Yes, Confidence: 4' style."""
    if not text:
        return None, None
    head = text.strip()[:200].lower()
    pred = (1 if (_YES_RE.search(head) or head.lstrip().startswith('yes'))
            else 0 if (_NO_RE.search(head) or head.lstrip().startswith('no'))
            else None)
    m = _CONF_RE.search(text)
    conf = int(m.group(1)) if m else None
    return pred, conf


def _parse_yes_no_prob(text: str) -> tuple[Optional[int], Optional[int]]:
    """Verbalized-probability parser: 'Probability: 73' → pred from prob>=50."""
    if not text:
        return None, None
    m = _PROB_RE.search(text)
    if not m:
        # Fallback to yes/no parsing
        return _parse_yes_no_conf(text)
    prob = max(0, min(100, int(m.group(1))))
    pred = 1 if prob >= 50 else 0
    # Map 0-100 prob to 1-5 confidence for schema-compat with other strategies
    # (distance from 50 = certainty; 50 → conf 1, 100 or 0 → conf 5)
    cert = abs(prob - 50)              # 0..50
    conf = 1 + int(cert / 12.5)        # 1..5 buckets
    return pred, conf


def _parse_json_response(text: str) -> tuple[Optional[int], Optional[int]]:
    """JSON-output parser: extract {answer, confidence} from JSON in response."""
    if not text:
        return None, None
    # Find the outermost JSON object in the response
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


# ── Few-shot example bank ────────────────────────────────────────────────────
# These pairs are intentionally chosen to be classic textbook examples that
# (a) are NOT in any specific KG's held-out test split (they're not from the
#     Open Targets sampling code in nb09 cell 4), and (b) cover the answer
# distribution and the indication-strength distribution.

FEW_SHOT_EXAMPLES_PLAIN = [
    {
        'drug': 'Metformin', 'disease': 'Type 2 diabetes mellitus',
        'answer': 'Yes', 'confidence': 5,
    },
    {
        'drug': 'Penicillin', 'disease': 'Alzheimer disease',
        'answer': 'No', 'confidence': 5,
    },
    {
        'drug': 'Sildenafil', 'disease': 'Pulmonary arterial hypertension',
        'answer': 'Yes', 'confidence': 4,
    },
]

FEW_SHOT_EXAMPLES_COT = [
    {
        'drug': 'Metformin', 'disease': 'Type 2 diabetes mellitus',
        'reasoning': ('Metformin is a biguanide that decreases hepatic glucose '
                      'production and increases insulin sensitivity. It is the '
                      'WHO-recommended first-line oral therapy for type 2 diabetes.'),
        'answer': 'Yes', 'confidence': 5,
    },
    {
        'drug': 'Penicillin', 'disease': 'Alzheimer disease',
        'reasoning': ('Penicillin is a beta-lactam antibiotic targeting bacterial '
                      'cell-wall synthesis. Alzheimer disease is a neurodegenerative '
                      'disorder of the central nervous system with no known bacterial '
                      'aetiology that penicillin would treat. No plausible mechanism.'),
        'answer': 'No', 'confidence': 5,
    },
    {
        'drug': 'Sildenafil', 'disease': 'Pulmonary arterial hypertension',
        'reasoning': ('Sildenafil is a PDE5 inhibitor originally developed for '
                      'angina, repurposed for erectile dysfunction, then approved '
                      'for pulmonary arterial hypertension because PDE5 inhibition '
                      'causes selective pulmonary vasodilation.'),
        'answer': 'Yes', 'confidence': 4,
    },
]


def _format_few_shot_plain(examples):
    return '\n\n'.join(
        f'Question: Is {ex["drug"]} a plausible treatment for {ex["disease"]}?\n'
        f'Answer: {ex["answer"]}, Confidence: {ex["confidence"]}'
        for ex in examples
    )


def _format_few_shot_cot(examples):
    return '\n\n'.join(
        f'Question: Is {ex["drug"]} a plausible treatment for {ex["disease"]}?\n'
        f'Reasoning: {ex["reasoning"]}\n'
        f'Answer: {ex["answer"]}, Confidence: {ex["confidence"]}'
        for ex in examples
    )


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
          pred (0/1/None), confidence (1-5/None), response (str — the
          concatenated raw text), error (str/None), n_calls_made (int).
        ``query_fn`` is a callable (prompt, seed, **decoding_kwargs) → dict
        with keys 'response' and 'error' (matches nb09's ollama_query).
        """
        raise NotImplementedError


def _kg_block(kg_text: str) -> str:
    """Format KG text as a labeled block; explicit empty marker if absent."""
    return (f'Knowledge graph context:\n{kg_text}' if kg_text
            else 'No knowledge graph context provided.')


# ── 1. Zero-shot direct (baseline; mirrors nb09's original prompt) ──────────

class ZeroShotDirect(Strategy):
    def __init__(self):
        super().__init__(
            name='zero_shot_direct', n_calls=1,
            description='Single-call yes/no with 1-5 confidence. nb09 baseline.',
            hypothesis='Reference. All other strategies are compared against this.',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a yes/no question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n'
            f'Respond on one line: "Answer: <Yes|No>, Confidence: <1-5>".'
        )
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=50)
        pred, conf = _parse_yes_no_conf(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── 2. Zero-shot chain-of-thought ────────────────────────────────────────────

class ZeroShotCoT(Strategy):
    def __init__(self):
        super().__init__(
            name='zero_shot_cot', n_calls=1,
            description='Explicit "think step by step" before answering.',
            hypothesis='Verbal reasoning surfaces mechanism plausibility and '
                       'should improve accuracy on harder pairs (Phase 1-2).',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a yes/no question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n\n'
            "Let's think step by step. Consider the drug's mechanism, the "
            'disease pathophysiology, and whether any plausible biological '
            'pathway connects them. After your reasoning, output a final '
            'line: "Answer: <Yes|No>, Confidence: <1-5>".'
        )
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=400)
        pred, conf = _parse_yes_no_conf(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── 3. Step-back prompting (general principles, then specific) ───────────────

class StepBack(Strategy):
    def __init__(self):
        super().__init__(
            name='step_back', n_calls=2,
            description='Call 1: elicit general repurposing principles. '
                        'Call 2: apply to the specific pair.',
            hypothesis='Forcing the model to articulate principles first '
                       'reduces shallow pattern-matching on the drug/disease '
                       'names alone.',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        # Call 1: general principles
        p1 = (
            f'You are a drug repurposing expert. Without considering any '
            f'specific drug or disease yet, list 3 general principles that '
            f'determine whether a drug is a plausible treatment for a disease '
            f'(e.g. mechanism, target overlap, contraindications). Keep each '
            f'principle to one short sentence.'
        )
        out1 = query_fn(p1, seed=seed, temperature=0.0, max_tokens=300)
        principles = out1.get('response', '')

        # Call 2: apply principles
        p2 = (
            'You are a drug repurposing expert. You previously listed these '
            f'general principles:\n\n{principles}\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Now apply those principles to: Is {drug} a plausible treatment '
            f'for {disease}?\n'
            'Output a final line: "Answer: <Yes|No>, Confidence: <1-5>".'
        )
        out2 = query_fn(p2, seed=seed + 1, temperature=0.0, max_tokens=200)
        pred, conf = _parse_yes_no_conf(out2.get('response', ''))
        combined = f'[principles]\n{principles}\n\n[application]\n{out2.get("response","")}'
        err = out1.get('error') or out2.get('error')
        return {'pred': pred, 'confidence': conf,
                'response': combined, 'error': err, 'n_calls_made': 2}


# ── 4. Few-shot 3 (plain) ────────────────────────────────────────────────────

class FewShot3(Strategy):
    def __init__(self):
        super().__init__(
            name='few_shot_3', n_calls=1,
            description='3 worked examples (answers only) before the question.',
            hypothesis='Examples anchor the output format and confidence '
                       'calibration; expect smaller effect on accuracy than CoT.',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        examples = _format_few_shot_plain(FEW_SHOT_EXAMPLES_PLAIN)
        prompt = (
            'You are a drug repurposing expert. Some worked examples first:\n\n'
            f'{examples}\n\n'
            'Now answer the following in the same format.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n'
            'Answer:'
        )
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=50)
        pred, conf = _parse_yes_no_conf(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── 5. Few-shot 3 + chain-of-thought ─────────────────────────────────────────

class FewShot3CoT(Strategy):
    def __init__(self):
        super().__init__(
            name='few_shot_3_cot', n_calls=1,
            description='3 worked examples with reasoning traces.',
            hypothesis='Best of both worlds — examples + reasoning. Highest '
                       'token cost per call (~600-1200 tokens).',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        examples = _format_few_shot_cot(FEW_SHOT_EXAMPLES_COT)
        prompt = (
            'You are a drug repurposing expert. Some worked examples first '
            '(each with reasoning):\n\n'
            f'{examples}\n\n'
            'Now answer the following in the same format (with brief reasoning).\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n'
            'Reasoning:'
        )
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=500)
        pred, conf = _parse_yes_no_conf(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── 6. Self-consistency (majority vote over 5 samples) ───────────────────────

class SelfConsistency5(Strategy):
    def __init__(self, n_samples: int = 5):
        super().__init__(
            name=f'self_consistency_{n_samples}', n_calls=n_samples,
            description=f'Sample {n_samples} times at T=0.7, majority-vote on Yes/No.',
            hypothesis='Decoding noise averages out, accuracy and calibration '
                       'improve, parse rate goes up. Cost = N× single call.',
        )
        self.n_samples = n_samples

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a yes/no question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n'
            f'Respond on one line: "Answer: <Yes|No>, Confidence: <1-5>".'
        )
        votes = []   # list of (pred, conf) from each sample
        responses = []
        last_err = None
        for i in range(self.n_samples):
            out = query_fn(prompt, seed=seed + i * 101,
                           temperature=0.7, max_tokens=50)
            responses.append(out.get('response', ''))
            if out.get('error'):
                last_err = out['error']
                continue
            p, c = _parse_yes_no_conf(out.get('response', ''))
            if p is not None:
                votes.append((p, c if c is not None else 3))
        # Majority vote
        if not votes:
            return {'pred': None, 'confidence': None,
                    'response': '\n---\n'.join(responses), 'error': last_err,
                    'n_calls_made': self.n_samples}
        n_yes = sum(p for p, _ in votes)
        n_total = len(votes)
        pred = 1 if n_yes >= n_total / 2 else 0
        # Confidence: avg confidence among voters that agreed with majority
        agree_confs = [c for p, c in votes if p == pred]
        conf = round(sum(agree_confs) / len(agree_confs)) if agree_confs else None
        return {'pred': pred, 'confidence': conf,
                'response': '\n---\n'.join(responses), 'error': last_err,
                'n_calls_made': self.n_samples}


# ── 7. Structured JSON output ────────────────────────────────────────────────

class StructuredJSON(Strategy):
    def __init__(self):
        super().__init__(
            name='structured_json', n_calls=1,
            description='Forces JSON output: {answer, confidence, reasoning, '
                        'contradictions}.',
            hypothesis='Schema forces the model to consider contradictions '
                       'explicitly; parse rate ≈ 100%; surfaces failure modes '
                       'in the "contradictions" field.',
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
            '  "reasoning": "<one sentence>",\n'
            '  "contradictions": "<any evidence against your answer, or '
            '\\"none\\">"\n'
            '}\n'
            'Do not include any text outside the JSON.'
        )
        # Ollama supports format='json' to enforce JSON decoding
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=300,
                       format='json')
        pred, conf = _parse_json_response(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── 8. Verbalized probability (0-100) ────────────────────────────────────────

class VerbalizedProbability(Strategy):
    def __init__(self):
        super().__init__(
            name='verbalized_prob', n_calls=1,
            description='Ask for 0-100 probability instead of 1-5 confidence.',
            hypothesis='Continuous probability is better-calibrated than '
                       'discrete 1-5; enables true AUROC computation from '
                       'verbalized score alone.',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: On a scale of 0 to 100, what is the probability that '
            f'{drug} is a plausible treatment for {disease}? '
            '0 = certainly no, 100 = certainly yes.\n'
            'Respond on one line: "Probability: <0-100>".'
        )
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=30)
        pred, conf = _parse_yes_no_prob(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── 9. Prompt-then-verify ────────────────────────────────────────────────────

class PromptThenVerify(Strategy):
    def __init__(self):
        super().__init__(
            name='prompt_then_verify', n_calls=2,
            description='Call 1: initial answer. Call 2: model finds flaws in '
                        'its own answer and revises.',
            hypothesis='Self-critique catches confident-but-wrong answers, '
                       'especially on plausible hard negatives.',
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        # Call 1
        p1 = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a yes/no question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n'
            'Briefly explain your reasoning, then output a final line: '
            '"Answer: <Yes|No>, Confidence: <1-5>".'
        )
        out1 = query_fn(p1, seed=seed, temperature=0.0, max_tokens=300)
        initial = out1.get('response', '')

        # Call 2: critique
        p2 = (
            f'You previously gave the following answer to the question '
            f'"Is {drug} a plausible treatment for {disease}?":\n\n'
            f'{initial}\n\n'
            'Carefully consider what might be wrong with that answer. '
            'What contradicting evidence exists? Are there mechanisms you '
            "overlooked? After considering counterarguments, give your "
            'FINAL answer on one line: "Answer: <Yes|No>, Confidence: <1-5>".'
        )
        out2 = query_fn(p2, seed=seed + 1, temperature=0.0, max_tokens=400)
        pred, conf = _parse_yes_no_conf(out2.get('response', ''))
        combined = f'[initial]\n{initial}\n\n[verified]\n{out2.get("response","")}'
        err = out1.get('error') or out2.get('error')
        return {'pred': pred, 'confidence': conf,
                'response': combined, 'error': err, 'n_calls_made': 2}


# ── 10. CISC: Confidence-Informed Self-Consistency ───────────────────────────
# Direct implementation of Taubenfeld et al. (2025), "Confidence Improves
# Self-Consistency in LLMs" — weighted majority vote where each sample's
# vote is scaled by its verbalized probability. Reported to outperform
# vanilla self-consistency by ~40% on average with the same sample budget.
#
# Also addresses Omar et al. (2025, JMIR Med Informatics) finding that
# worse-performing models have paradoxically higher confidence, and the
# Xiong et al. (2023) framework for verbalized confidence + sampling.

class CISC(Strategy):
    def __init__(self, n_samples: int = 5):
        super().__init__(
            name='cisc', n_calls=n_samples,
            description=(f'Confidence-Informed Self-Consistency. Sample {n_samples}× '
                         f'at T=0.7, each emitting a verbalized 0-100 probability. '
                         f'Final answer = weighted vote (weight = verbalized prob).'),
            hypothesis=('Verbalized-prob-weighted voting beats both single-call and '
                        'plain majority-vote self-consistency; reduces overconfidence; '
                        'gives a continuous score usable for AUROC.'),
        )
        self.n_samples = n_samples

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are a drug repurposing expert. Below is knowledge-graph '
            'context about a drug, followed by a question.\n\n'
            f'{_kg_block(kg_text)}\n\n'
            f'Question: On a scale of 0 to 100, what is the probability that '
            f'{drug} is a plausible treatment for {disease}? '
            '0 = certainly no, 100 = certainly yes.\n'
            'Respond on one line: "Probability: <0-100>".'
        )
        responses = []
        probs = []
        last_err = None
        for i in range(self.n_samples):
            out = query_fn(prompt, seed=seed + i * 101,
                           temperature=0.7, max_tokens=30)
            responses.append(out.get('response', ''))
            if out.get('error'):
                last_err = out['error']
                continue
            m = _PROB_RE.search(out.get('response', '') or '')
            if m:
                probs.append(max(0, min(100, int(m.group(1)))))

        if not probs:
            return {'pred': None, 'confidence': None,
                    'response': '\n---\n'.join(responses),
                    'error': last_err, 'n_calls_made': self.n_samples}

        # CISC weighting: each sample's vote weight = |prob - 50| (i.e. how
        # far from "I don't know"). Samples close to 50 contribute less.
        # Final score = sum(weight_i * (prob_i / 100)) / sum(weight_i).
        # Binarize at 0.5 for pred. Confidence bucketed from |score - 0.5|.
        weights  = [abs(p - 50) + 1 for p in probs]    # +1 avoids zero weight
        weighted = sum(w * (p / 100.0) for w, p in zip(weights, probs))
        norm     = sum(weights)
        score    = weighted / norm                      # 0..1
        pred     = 1 if score >= 0.5 else 0
        cert     = abs(score - 0.5)                     # 0..0.5
        conf     = 1 + int(cert * 10)                   # 1..5 buckets
        conf     = max(1, min(5, conf))
        return {'pred': pred, 'confidence': conf,
                'response': '\n---\n'.join(responses),
                'error': last_err, 'n_calls_made': self.n_samples}


# ── 11. Multi-Expert ROT (Reflection of Thoughts) ───────────────────────────
# Direct implementation of Wang et al. (2024), npj Digital Medicine,
# "Prompt engineering in consistency and reliability with the evidence-based
# guideline for LLMs". Tested 4 prompts across 9 LLMs on AAOS osteoarthritis
# guidelines; gpt-4-Web + ROT was the single best combination (62.9%
# overall consistency, 77.5% on strong-evidence questions).
#
# The mechanism: simulate a panel of 3 experts who reason independently,
# then discuss and BACKTRACK to reach agreement. Crucially this is a
# single-call strategy — the model simulates the deliberation internally.
# Distinct from CISC (which samples N times and aggregates externally).

class MultiExpertROT(Strategy):
    def __init__(self):
        super().__init__(
            name='multi_expert_rot', n_calls=1,
            description=('Reflection of Thoughts: simulate 3 experts who reason '
                         'independently then discuss and backtrack to reach '
                         'consensus. Wang et al. 2024 (npj Digital Medicine) '
                         'showed this was best on gpt-4-Web (77.5% on strong '
                         'evidence-level guideline questions).'),
            hypothesis=('Multi-persona deliberation forces the model to '
                        'consider competing perspectives within a single call, '
                        'reducing single-pass anchoring and surfacing '
                        'disagreements that single-expert prompts hide.'),
        )

    def execute(self, query_fn, drug, disease, kg_text, seed):
        prompt = (
            'You are facilitating a panel of 3 drug-repurposing experts. Below '
            'is knowledge-graph context about a drug, followed by a yes/no '
            f'question.\n\n{_kg_block(kg_text)}\n\n'
            f'Question: Is {drug} a plausible treatment for {disease}?\n\n'
            'Simulate the following deliberation:\n\n'
            'Step 1 — Independent reasoning. Each of the 3 experts independently '
            "analyses the question, considering the drug's known mechanism, the "
            'disease pathophysiology, any evidence in the KG context, and any '
            'evidence against the indication. Each expert gives a preliminary '
            'verdict (Yes/No) and a confidence (1-5) with one sentence of '
            'rationale.\n\n'
            'Step 2 — Discussion. The experts discuss their reasoning. Any expert '
            'whose initial verdict differs from the others is asked to defend it. '
            'Experts BACKTRACK and revise their reasoning if convinced by '
            'counterarguments. Show the discussion.\n\n'
            'Step 3 — Consensus. The panel reaches an agreed verdict and '
            'confidence. Output the final line in this exact format:\n'
            '"Answer: <Yes|No>, Confidence: <1-5>"'
        )
        out = query_fn(prompt, seed=seed, temperature=0.0, max_tokens=800)
        pred, conf = _parse_yes_no_conf(out.get('response', ''))
        return {'pred': pred, 'confidence': conf,
                'response': out.get('response', ''),
                'error': out.get('error'),
                'n_calls_made': 1}


# ── Registry ─────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, Strategy] = {
    s.name: s for s in [
        # Selected for nb09 default run (5 strategies, all literature-grounded):
        ZeroShotDirect(),           # restricted: baseline (Sivarajkumar 2024)
        StructuredJSON(),           # restricted: schema w/ contradictions (DrugReX, Paper 1)
        ZeroShotCoT(),              # unrestricted: 0-COT (Sivarajkumar 2024, Wang 2024)
        MultiExpertROT(),           # unrestricted: 3-expert ROT (Wang 2024 npj Digital Medicine)
        CISC(),                     # unrestricted: confidence-informed self-consistency (Taubenfeld 2025)
        # Available as library code but not in nb09 default run:
        FewShot3CoT(),
        StepBack(),
        FewShot3(),
        SelfConsistency5(),
        VerbalizedProbability(),
        PromptThenVerify(),
    ]
}

# The 5 strategies selected for the nb09 default run. Each has direct support
# from peer-reviewed biomedical-LLM literature. See
# docs/llm_prompting_strategies.md for per-strategy citations.
DEFAULT_STRATEGIES_FOR_NB09 = [
    'zero_shot_direct',      # restricted
    'structured_json',       # restricted
    'zero_shot_cot',         # unrestricted
    'multi_expert_rot',      # unrestricted  ← swapped in from Paper 3
    'cisc',                  # unrestricted
]

DEFAULT_STRATEGY_ORDER = list(STRATEGIES.keys())


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f'Unknown strategy {name!r}. '
                       f'Known: {list(STRATEGIES.keys())}')
    return STRATEGIES[name]


def total_calls_per_cell(strategy_names: list[str]) -> int:
    """Sum of n_calls across the listed strategies — for cost estimation."""
    return sum(get_strategy(n).n_calls for n in strategy_names)
