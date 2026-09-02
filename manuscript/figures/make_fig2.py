#!/usr/bin/env python3
"""
Figure 2 (schematic) | Benchmark construction and per-KG coverage.

  a  the 253 -> 116 attrition funnel
  b  how many of the 116 each knowledge graph actually covers

!! PROVENANCE WARNING -----------------------------------------------------
The two middle funnel steps are NOT reproducible from the stored tables:

    step        build_116.py docstring   dropped CSV   funnel CSV flags
    leakage               -91                -84            -87
    coverage              -27                -43             --
    final                 116                107            111

Only 253, 234 and the final 116 are independently confirmed (the last by
gold_standard_v4_bigset.tsv = 116 rows and coverage_annotation_v4.csv =
116 x 3). The docstring values are drawn because they are the only set
that reconciles (253-19-91-27 = 116). Re-run scripts/build_116.py,
regenerate the dropped tables, then set VERIFIED = True.
---------------------------------------------------------------------------

Panel b is computed live from coverage_annotation_v4.csv.

Run:  python manuscript/figures/make_fig2.py
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import setup_manuscript_style, save_fig, KG_PALETTE

OUT = Path(__file__).resolve().parent
setup_manuscript_style()

INK, GREY, HAIR = "#1A1A1A", "#767676", "#C8C8C8"
CARD, LOSS, TRACK = "#F5F4F1", "#C0563C", "#EBE9E4"
VERIFIED = False

# ---------------------------------------------------------------- panel b data
cov = pd.read_csv(BASE / "data/gold_standards/coverage_annotation_v4.csv")
wide = cov.pivot(index="Pair_ID", columns="kg", values="covered").fillna(0)
N_TOTAL = len(wide)
COVERAGE = {k: int(wide[k].sum()) for k in ["primekg", "drkg", "biokg"]}
N_ALL3 = int((wide[["primekg", "drkg", "biokg"]].sum(axis=1) == 3).sum())

MM = 1 / 25.4
W_IN, H_IN = 180 * MM, 3.6
fig, ax = plt.subplots(figsize=(W_IN, H_IN))
Y_SPAN = 100 * H_IN / W_IN
ax.set_xlim(0, 100)
ax.set_ylim(-2.2, -2.2 + Y_SPAN)
ax.axis("off")

# --- geometry. Narrower boxes than v1: the funnel needs ~20 units of text,
# the attrition notes needed the width far more.
BX, BW, BH, GAP = 2.0, 38.0, 5.2, 2.6
NUM_R = BX + 8.4                 # numbers right-align here ...
LAB_L = BX + 9.6                 # ... labels left-align here, so the two
HEAD = 47.2                      #     columns run straight down the funnel
TOP = 45.6


def head(x, letter, sub):
    ax.text(x, HEAD, letter, fontsize=8, fontweight="bold", color=INK,
            ha="left", va="baseline")
    ax.text(x + 2.9, HEAD, sub, fontsize=6.6, color=INK, ha="left",
            va="baseline")


# ============================================== a | construction funnel
head(BX, "a", "Benchmark construction")

# Subtitles are capped at ~40 characters so none overruns its box; the
# agency list moved to the caption.
STAGES = [
    ("253", "candidate pairs", "new-indication approvals, 2023–2026", False),
    ("234", "harness-ready pairs", "strict repurposing, time-split", False),
    ("143", "leakage-free pairs", "no drug→disease edge in any KG", False),
    ("116", "evaluable pairs", "identical across every KG and model", True),
    ("116", "ranking queries", "1 approved + 7 distractors, 74 diseases", False),
    ("2,784", "calls per model", "×3 seeds × 2 orders × 4 arms", True),
]
LOSSES = [
    ("−19", "unresolved IDs, or not\nstrict repurposing"),
    ("−91", "answer leakage: drug→disease\nedge already in a KG"),
    ("−27", "drug or disease missing\nfrom all three KGs"),
]
LINKS = ["one anonymised query per pair", "no-KG + 3 KG arms"]

tops = [TOP - i * (BH + GAP) for i in range(len(STAGES))]

for i, ((num, lab, sub, key), t) in enumerate(zip(STAGES, tops)):
    ax.add_patch(FancyBboxPatch((BX, t - BH), BW, BH,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                fc=CARD if key else "white", ec=HAIR,
                                lw=.6, zorder=2))
    if key:
        ax.add_patch(FancyBboxPatch((BX, t - BH), 0.8, BH,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=INK, ec="none", zorder=3))
    # right-align the number, left-align the label: no width guessing
    ax.text(NUM_R, t - 2.05, num, fontsize=8.5, fontweight="bold", color=INK,
            ha="right", va="center", zorder=4)
    ax.text(LAB_L, t - 2.05, lab, fontsize=6.4, color=INK, ha="left",
            va="center", zorder=4)
    ax.text(LAB_L, t - 4.05, sub, fontsize=5.0, color=GREY, ha="left",
            va="center", zorder=4)

    if i + 1 < len(STAGES):
        mid = (t - BH + tops[i + 1]) / 2
        ax.add_patch(FancyArrowPatch((BX + BW / 2, t - BH),
                                     (BX + BW / 2, tops[i + 1]),
                                     arrowstyle="-|>", mutation_scale=6,
                                     color=HAIR, lw=.8, zorder=2))
        if i < 3:
            ax.plot([BX + BW, BX + BW + 2.2], [mid, mid], color=LOSS,
                    lw=.7, ls=(0, (2, 1.6)), zorder=2)
            n, why = LOSSES[i]
            ax.text(BX + BW + 6.4, mid, n, fontsize=6.0, fontweight="bold",
                    color=LOSS, ha="right", va="center")
            ax.text(BX + BW + 7.6, mid, why, fontsize=5.1, color=GREY,
                    ha="left", va="center", linespacing=1.45)
        else:
            ax.text(BX + BW / 2 + 1.4, mid, LINKS[i - 3], fontsize=5.1,
                    color=GREY, ha="left", va="center", style="italic")

if not VERIFIED:
    ax.text(BX, tops[-1] - BH - 2.4,
            "steps 2–3 not reproducible from stored tables — see script header",
            fontsize=4.8, color=LOSS, ha="left", va="center")

# ================================================= b | per-KG coverage
# Vertically centred on the funnel rather than top-aligned, so the
# composition does not sit heavy on the left.
CX = 71.0
head(CX, "b", "Coverage of the 116")

T0, TW = CX + 8.0, 14.0
BAR_Y = [34.0, 28.4, 22.8]
for (nm, key), y in zip([("PrimeKG", "primekg"), ("DRKG", "drkg"),
                         ("BioKG", "biokg")], BAR_Y):
    ax.plot([T0, T0 + TW], [y, y], color=TRACK, lw=3.0,
            solid_capstyle="round", zorder=2)
    ax.plot([T0, T0 + TW * COVERAGE[key] / N_TOTAL], [y, y],
            color=KG_PALETTE[key], lw=3.0, solid_capstyle="round", zorder=3)
    ax.text(CX, y, nm, fontsize=6.0, color=INK, ha="left", va="center")
    ax.text(98.0, y, str(COVERAGE[key]), fontsize=6.0, color=INK,
            ha="right", va="center")
ax.text(98.0, BAR_Y[0] + 4.0, f"of {N_TOTAL}", fontsize=5.1, color=GREY,
        ha="right", va="center")

ax.text(CX, 14.0, f"{N_ALL3}", fontsize=8.5, fontweight="bold",
        color=INK, ha="left", va="center")
ax.text(CX + 5.2, 14.0, "covered by all three", fontsize=6.0, color=INK,
        ha="left", va="center")
ax.text(CX, 10.8, "the common support on which evidence\n"
                  "quality is comparable across graphs",
        fontsize=5.0, color=GREY, ha="left", va="top", linespacing=1.45)

save_fig(fig, OUT, "Figure2", dpi=600)
print(f"  panel b live from coverage_annotation_v4.csv: {COVERAGE}, all three = {N_ALL3}")
