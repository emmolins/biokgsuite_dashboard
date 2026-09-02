#!/usr/bin/env python3
"""
Regenerate figures from CACHED checkpoints only — no raw-KG loading, no analysis.

Every eval-notebook re-loads the ~20 GB raw KGs and recomputes when you Run All;
that is the slow part. The distilled figure data already lives in
results/checkpoints/*.pkl, so this script redraws the restyled figures straight
from cache in seconds, into the per-notebook results/figures/<folder>/ dirs.

Covered (data is in the pkls):
    02_id_validation
    04_graph_cohesion, 04_clustering_coefficient,
    04_known_pair_recovery, 04_differential_resilience
    05_stability
    06_link_prediction_heatmap

NOT covered (figure-level data was never cached — needs a real notebook run):
    00 (cheap: pure aggregation), 01, 03_source_diversity,
    07 proximity/prospective set, 08 (raw embeddings; resampled JSON caches exist),
    09 family (refresh via their own builders / make_fig1_llm.py / scaling_sweep)

Usage:  python scripts/regenerate_figs.py
"""
import os, sys, pickle
import matplotlib
matplotlib.use('Agg')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import eval_figs as E
from src import eval_figs08 as E8

TABLES = os.path.join(ROOT, 'results', 'tables')

CK   = os.path.join(ROOT, 'results', 'checkpoints')
FIGS = os.path.join(ROOT, 'results', 'figures')


def _dir(name):
    p = os.path.join(FIGS, name)
    os.makedirs(p, exist_ok=True)
    return p


def _load(stem):
    return pickle.load(open(os.path.join(CK, f'{stem}.pkl'), 'rb'))


def _figdata(stem):
    """Load a NN_figdata.pkl, or raise a clear 'run the notebook once' message."""
    p = os.path.join(CK, f'{stem}.pkl')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{stem}.pkl not cached yet — Run All on that notebook once to create it')
    return pickle.load(open(p, 'rb'))


JOBS = []          # (label, callable)
RESULTS = []       # (label, 'OK' | 'SKIP: reason')


def job(label):
    def deco(fn):
        JOBS.append((label, fn))
        return fn
    return deco


# ── 02 ───────────────────────────────────────────────────────────────────────
@job('02_id_validation')
def _():
    d = _load('02_semantic_validity')
    E.fig02_validity(d['val_records'], _dir('02_annotation_accuracy'))


# ── 04 (4 figures) ────────────────────────────────────────────────────────────
@job('04_graph_cohesion')
def _():
    d = _load('04_topology')
    E.fig04_cohesion(d['conn_records'], _dir('04_topology'))


@job('04_clustering_coefficient')
def _():
    d = _load('04_topology')
    E.fig04_clustering(d['clust_records'], _dir('04_topology'))


@job('04_known_pair_recovery')
def _():
    d = _load('04_topology')
    E.fig04_recovery(d['recovery_results'], _dir('04_topology'))


# (04_differential_resilience now comes from 04_figdata.pkl — see below — because the
#  main checkpoint only stored the final scalar, not the per-dropout curves.)


# ── 05 ───────────────────────────────────────────────────────────────────────
@job('05_stability')
def _():
    d = _load('05_stability')
    E.fig05_stability(d['raw_r'], d['dropout_rates'], _dir('05_stability'))


# ── 06 ───────────────────────────────────────────────────────────────────────
@job('06_link_prediction_heatmap')
def _():
    d = _load('06_predictive_performance')
    E.fig06_link_heatmap(d['link_records'], d['primary_heuristic'], _dir('06_task_performance'))


# ── figure-data caches (created on the next notebook run; skip gracefully until then) ──
@job('03_source_diversity')
def _():
    fd = _figdata('03_figdata')
    E.fig03_source_diversity(fd['kg_counts'], fd['panel_config'], _dir('03_trustworthiness'))


@job('04_differential_resilience')
def _():
    fd = _figdata('04_figdata')
    E.fig04_resilience(fd['diversity_results'], _dir('04_topology'))


@job('07_network_proximity')
def _():
    fd = _figdata('07_figdata')
    E.fig07_proximity(fd['all_proximity'], _dir('07_generalization'))


@job('07a_prospective_auroc')
def _():
    fd = _figdata('07_figdata')
    E.fig07a_prospective(fd['baseline_all_kg'], fd['baseline_same_kg'],
                         fd['temporal_metrics_kg'], _dir('07_generalization'))


# ── 08 (driven entirely by the two cached CSV tables — no embeddings retrained) ──
@job('08_resampled_auroc')
def _():
    E8.fig08_resampled_auroc(os.path.join(TABLES, '08_embedding_comparison_resampled.csv'),
                             _dir('08_embedding_validation'))


@job('08_lift_over_gemma')
def _():
    E8.fig08_lift_over_gemma(os.path.join(TABLES, '08_embedding_comparison_resampled.csv'),
                             _dir('08_embedding_validation'))


@job('08_heuristic_vs_embedding_scatter')
def _():
    E8.fig08_heuristic_scatter(os.path.join(TABLES, '08_embedding_comparison.csv'),
                               os.path.join(TABLES, '08_embedding_comparison_resampled.csv'),
                               _dir('08_embedding_validation'))


def main():
    print(f'regenerating from cache -> {FIGS}\n')
    for label, fn in JOBS:
        try:
            fn()
            RESULTS.append((label, 'OK'))
            print(f'  [OK]   {label}')
        except Exception as e:
            RESULTS.append((label, f'SKIP: {type(e).__name__}: {e}'))
            print(f'  [SKIP] {label}  -> {type(e).__name__}: {e}')
    ok = sum(1 for _, r in RESULTS if r == 'OK')
    print(f'\n{ok}/{len(RESULTS)} figures regenerated from cache (PNG+PDF).')
    if ok < len(RESULTS):
        print('Skips above need a real notebook run (their data is not in the pkls).')


if __name__ == '__main__':
    main()
