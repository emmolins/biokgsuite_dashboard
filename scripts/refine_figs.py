#!/usr/bin/env python3
"""Verify + regenerate the 02/04 figures headless from the cached pkls, using the
shared clean plotting in src.eval_figs (the same functions the notebooks call).
03 source-diversity is not cached locally (computed from the raw KGs), so it is
only fixed in the notebook cell and renders on the next notebook run."""
import os, sys, pickle, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import eval_figs as E

CK = os.path.join(ROOT, 'results', 'checkpoints')
F02 = os.path.join(ROOT, 'results', 'figures', '02_annotation_accuracy')
F04 = os.path.join(ROOT, 'results', 'figures', '04_topology')

d2 = pickle.load(open(f'{CK}/02_semantic_validity.pkl', 'rb'))
E.fig02_validity(d2['val_records'], F02)

d4 = pickle.load(open(f'{CK}/04_topology.pkl', 'rb'))
E.fig04_cohesion(d4['conn_records'], F04)
E.fig04_clustering(d4['clust_records'], F04)
E.fig04_recovery(d4['recovery_results'], F04)
print('verified + regenerated 02 + 04 figures via src.eval_figs')
