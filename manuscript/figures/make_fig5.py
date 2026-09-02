#!/usr/bin/env python3
"""
Figure 5 for the BioKGSuite Nature Communications manuscript.

Fig. 5 | Design of the prospective, leakage-controlled evaluation set and
the grounding experiment.

  a  Construction of the evaluation set from 253 candidate approvals,
     with the pairs removed at each filter and the therapeutic-area
     composition of the retained set.
  b  Structure of a single query. One disease, a pool of eight candidate
     drugs containing exactly one post-cutoff positive, and the evidence
     block that is the only thing differing between arms.

Panel a is read from results/tables/ so the counts cannot drift. Panel b is
a schematic; its example pair is a real row of the evaluation set (Pair 7,
sarilumab -> polymyalgia rheumatica, present in all three graphs).

Run:  python manuscript/figures/make_fig5.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import (setup_manuscript_style, panel_label,  # noqa: E402
                          KG_OCEAN_BLUE, KG_RUST, KG_SAGE, KG_PLUM,
                          TEXT_COLOR, GRID_COLOR)

TABLES = BASE / "results" / "tables"
OUT = Path(__file__).resolve().parent

setup_manuscript_style()

MUTED = "#8C8C8C"
FAINT = "#B5B5B5"
PANEL_BG = "#F4F4F2"
POS_COLOR = KG_OCEAN_BLUE
CUT_COLOR = KG_RUST

# ---------------------------------------------------------------- data
audit = pd.read_csv(TABLES / "reconstruct_116_audit.csv")
dropped = pd.read_csv(TABLES / "dropped_253_to_116.csv")

N_TOTAL = len(audit)
stage_counts = dropped.groupby("drop_stage").size().to_dict()
N_UNRES = stage_counts["1_unresolved_ids"]
N_LEAK = stage_counts["2_answer_leakage"]
N_COV = stage_counts["3_no_kg_coverage"]

survivors = audit[~audit.Pair_ID.isin(set(dropped.Pair_ID))]
N_KEPT = len(survivors)
N_EVAL = int(survivors.in_final.sum())

STAGES = [
    ("Candidate approvals", N_TOTAL, None),
    ("Identifiers resolved", N_TOTAL - N_UNRES,
     f"−{N_UNRES}  unresolvable identifiers"),
    ("Leakage removed", N_TOTAL - N_UNRES - N_LEAK,
     f"−{N_LEAK}  drug–disease edge already in a graph"),
    ("Retained pairs", N_KEPT,
     f"−{N_COV}  absent from all three graphs"),
]

ta = survivors.TA_simplified.value_counts()
KG_COV = {k: int(survivors[f"{k}_present"].sum())
          for k in ("primekg", "drkg", "biokg")}

# ---------------------------------------------------------------- canvas
fig = plt.figure(figsize=(7.20, 4.75))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.55, 1.0],
                      hspace=0.52, wspace=0.30)
ax_fun = fig.add_subplot(gs[0, 0])
ax_ta = fig.add_subplot(gs[0, 1])
ax_q = fig.add_subplot(gs[1, :])

for ax in (ax_fun, ax_ta, ax_q):
    ax.grid(False)

# ------------------------------------------------------------ (a1) funnel
ax = ax_fun
BH = 0.52
for i, (label, n, removal) in enumerate(STAGES):
    y = -i
    frac = n / N_TOTAL
    ax.barh(y, frac, height=BH, color=KG_OCEAN_BLUE if i == len(STAGES) - 1
            else "#BFD2DC", zorder=3, edgecolor="white", lw=0.5)
    ax.text(-0.015, y, label, ha="right", va="center",
            fontsize=6.6, color=TEXT_COLOR)
    ax.text(frac + 0.015, y, f"{n}", ha="left", va="center",
            fontsize=6.8, color=TEXT_COLOR, fontweight="bold")
    if removal:
        ax.text(0.012, y + BH / 2 + 0.20, removal, ha="left", va="center",
                fontsize=6.0, color=CUT_COLOR)

ax.text(0.012, -len(STAGES) + 0.60,
        f"of which {N_EVAL} evaluable "
        f"({N_KEPT - N_EVAL} fixed-dose combinations)",
        ha="left", va="center", fontsize=6.0, color=MUTED)

cov = "   ".join(f"{k} {v}" for k, v in
                 zip(("PrimeKG", "DRKG", "BioKG"), KG_COV.values()))
ax.text(0.012, -len(STAGES) + 0.28, f"covered per graph   {cov}",
        ha="left", va="center", fontsize=6.0, color=MUTED)

ax.set_xlim(0, 1.16)
ax.set_ylim(-len(STAGES) + 0.05, 0.62)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# ------------------------------------------------------------ (a2) areas
ax = ax_ta
order = ta.sort_values(ascending=True)
ax.barh(range(len(order)), order.values, height=0.66,
        color=KG_OCEAN_BLUE, zorder=3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order.index, fontsize=6.4)
for i, v in enumerate(order.values):
    ax.text(v + 0.9, i, str(v), va="center", fontsize=6.2, color=TEXT_COLOR)
ax.set_xlabel(f"Retained pairs (n = {N_KEPT})", labelpad=2, fontsize=6.8)
ax.set_xlim(0, max(order.values) * 1.20)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID_COLOR, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# ------------------------------------------------------------ (b) schematic
ax = ax_q
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)


def box(x, y, w, h, fc="white", ec=FAINT, lw=0.6, r=1.4, z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.7, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def stage_head(x, text):
    ax.text(x, 96, text, fontsize=6.8, color=TEXT_COLOR,
            fontweight="bold", ha="left", va="top")


# ---- stage 1 · query
stage_head(0, "1 · Query")
box(0, 68, 25, 15, fc=PANEL_BG, ec=FAINT)
ax.text(1.8, 79.5, "Target disease", fontsize=5.8, color=MUTED, va="top")
ax.text(1.8, 73.8, "Polymyalgia rheumatica", fontsize=6.8,
        color=TEXT_COLOR, va="top")

ax.text(0, 62, "Candidate pool", fontsize=5.8, color=MUTED, va="top")
LETTERS = list("ABCDEFGH")
POS_IDX = 3
for i, L in enumerate(LETTERS):
    xx = (i % 4) * 6.3
    yy = 50 - (i // 4) * 8.4
    is_pos = i == POS_IDX
    box(xx, yy, 5.2, 6.2, fc=POS_COLOR if is_pos else "white",
        ec=POS_COLOR if is_pos else FAINT, lw=0.8 if is_pos else 0.6)
    ax.text(xx + 2.6, yy + 3.1, L, ha="center", va="center", fontsize=6.4,
            color="white" if is_pos else TEXT_COLOR,
            fontweight="bold" if is_pos else "normal")

ax.text(0, 30, "7 distractor drugs and 1 post-cutoff\npositive, order shuffled",
        fontsize=5.9, color=MUTED, va="top", linespacing=1.45)

arrow(27.5, 55, 32.5, 55)

# ---- stage 2 · evidence block
stage_head(34, "2 · Evidence block")
ax.text(34, 89.5, "the only thing differing between arms",
        fontsize=5.9, color=MUTED, ha="left", va="top")

box(34, 70, 32, 13, fc="white", ec=FAINT)
ax.text(35.8, 80.0, "Baseline arm", fontsize=6.2, color=TEXT_COLOR, va="top")
ax.text(35.8, 74.6, "no evidence supplied", fontsize=6.0,
        color=MUTED, va="top", style="italic")

box(34, 8, 32, 55, fc=PANEL_BG, ec=FAINT)
ax.text(35.8, 59.5, "Graph arm", fontsize=6.2, color=TEXT_COLOR, va="top")
ax.text(35.8, 53.6, "one-hop neighbours of every candidate\nand of the disease",
        fontsize=5.8, color=MUTED, va="top", linespacing=1.45)

TRIPLES = [
    ("D  targets  IL6R", False),
    ("D  targets  IL6ST", False),
    ("PMR  associated gene  IL6", False),
    ("PMR  related disease  GCA", False),
    ("D  treats  PMR", True),
]
cut_artists = []
ty = 40
for txt, cut in TRIPLES:
    t = ax.text(36.4, ty, txt, fontsize=5.8,
                color=CUT_COLOR if cut else TEXT_COLOR, va="center")
    if cut:
        cut_artists.append((t, ty))
    ty -= 5.4

ax.text(35.8, 12.2,
        "drug–disease relation excluded structurally,\n"
        "pharmacokinetic relations dropped",
        fontsize=5.7, color=CUT_COLOR, va="center", linespacing=1.45)

arrow(68.5, 55, 73.5, 55)

# ---- stage 3 · ranking
stage_head(75, "3 · Ranked output")
ax.text(75, 89.5, "candidates ordered by plausibility",
        fontsize=5.9, color=MUTED, ha="left", va="top")

RANKED = ["D", "B", "G", "A", "F"]
for i, L in enumerate(RANKED):
    yy = 62 - i * 8.0
    is_pos = L == LETTERS[POS_IDX]
    box(75, yy, 5.4, 6.2, fc=POS_COLOR if is_pos else "white",
        ec=POS_COLOR if is_pos else FAINT, lw=0.8 if is_pos else 0.6)
    ax.text(77.7, yy + 3.1, L, ha="center", va="center", fontsize=6.4,
            color="white" if is_pos else TEXT_COLOR,
            fontweight="bold" if is_pos else "normal")
    ax.text(82.5, yy + 3.1, f"rank {i + 1}", fontsize=5.9,
            color=TEXT_COLOR if is_pos else MUTED, va="center")
ax.text(76.0, 26.0, "⋮", fontsize=8, color=FAINT, ha="center", va="center")

ax.text(75, 17,
        "score is the rank of the positive,\nreported as MRR and hits@k",
        fontsize=5.9, color=MUTED, va="top", linespacing=1.45)

# ------------------------------------------------------------ strike-throughs
# Measured after layout so the rule matches the rendered text exactly.
fig.canvas.draw()
inv = ax.transData.inverted()
for t, yy in cut_artists:
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    ax.plot([x0 - 0.4, x1 + 0.4], [yy, yy], color=CUT_COLOR, lw=0.6, zorder=5)

# ------------------------------------------------------------ panel letters
panel_label(ax_fun, "a", x=-0.155, y=1.16)
panel_label(ax_q, "b", x=-0.058, y=1.10)

fig.savefig(OUT / "Figure5.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Figure5.png", bbox_inches="tight", dpi=600, facecolor="white")

# ------------------------------------------------------------ audit
print(f"total {N_TOTAL} | -{N_UNRES} unresolved | -{N_LEAK} leakage | "
      f"-{N_COV} coverage | retained {N_KEPT} | in_final {N_EVAL}")
print("therapeutic areas:", ta.to_dict())
print("per-graph coverage:", KG_COV)
