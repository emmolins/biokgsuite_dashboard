#!/usr/bin/env python3
"""
Figure 1 for the BioKGSuite Nature Communications manuscript.

Fig. 1 | The BioKGSuite evaluation framework.

A schematic rather than a taxonomy list. Inputs enter at the top, the seven
dimensions are grouped into the three families that structure the framework,
and a multidimensional profile leaves at the bottom.

Layout note: seven dimension columns across a 183 mm page forces type below
6 pt, so the families are stacked as bands and the dimensions run across
each band. Nature conventions otherwise: flat fills, hairline rules, no drop
shadows or nested containers, palette from src/plotting.py so Fig. 1 matches
Figs. 2-4, and 6-7.5 pt sans-serif type.

Run:  python manuscript/figures/make_fig1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import (setup_manuscript_style,  # noqa: E402
                          KG_OCEAN_BLUE, KG_SAGE, KG_RUST, TEXT_COLOR)

OUT = Path(__file__).resolve().parent
setup_manuscript_style()

MUTED = "#7A7A7A"
FAINT = "#C9C9C9"
BAND = "#F6F6F4"

FAMILIES = [
    ("Content", KG_OCEAN_BLUE, "what the graph contains", [
        ("Coverage", "how complete",
         ["Entity coverage", "Relation coverage"]),
        ("Annotation accuracy", "how accurate",
         ["Entity validity", "Relational consistency"]),
        ("Trustworthiness", "how well evidenced",
         ["Source diversity", "Edge traceability", "Uncertainty quantification"]),
    ]),
    ("Structure", KG_SAGE, "how it is organised", [
        ("Topology", "how interconnected",
         ["Graph cohesion", "Clustering coefficient",
          "Graph reachability", "Community purity"]),
        ("Stability", "how robust",
         ["Random dropout", "Periphery dropout"]),
    ]),
    ("Inference", KG_RUST, "what it supports", [
        ("Task performance", "how reliable",
         ["Link prediction", "Neighbourhood retrieval", "Multi-hop reasoning"]),
        ("Generalisation", "how transferable",
         ["Data-sparse", "Cross-domain", "Prospective"]),
    ]),
]

# ---------------------------------------------------------------- canvas
fig = plt.figure(figsize=(7.20, 5.00))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

GUTTER_X = 0.0          # family label column
DIM_X0 = 19.0           # first dimension column
DIM_W = 26.0
DIM_GAP = 1.6

DIM_Y, SUB_Y, MET_Y0, MET_DY = 4.4, 8.6, 12.8, 3.9   # offsets below band top
BAND_PAD = 2.6
BAND_GAP = 3.0

# Band height follows the deepest dimension in that family, so no band
# carries dead space and none overflows into the next.
BAND_H = [MET_Y0 + (max(len(m) for _, _, m in dims) - 1) * MET_DY + BAND_PAD
          for _, _, _, dims in FAMILIES]
BAND_TOP, _y = [], 90.5
for h in BAND_H:
    BAND_TOP.append(_y)
    _y -= h + BAND_GAP
BAND_BOT_LAST = BAND_TOP[-1] - BAND_H[-1]


def text(x, y, s, **kw):
    kw.setdefault("color", TEXT_COLOR)
    kw.setdefault("va", "center")
    return ax.text(x, y, s, **kw)


def box(x, y, w, h, **kw):
    kw.setdefault("facecolor", "white")
    kw.setdefault("edgecolor", FAINT)
    kw.setdefault("lw", 0.6)
    ax.add_patch(Rectangle((x, y), w, h, zorder=2, **kw))


def arrow(x0, y0, x1, y1, colour=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.7, color=colour,
                                 shrinkA=0, shrinkB=0, zorder=4))


# ---------------------------------------------------------------- inputs
IN = [("Biomedical knowledge graph", 27.0), ("Task reference standards", 73.0)]
for label, cx in IN:
    box(cx - 24.0, 95.0, 48.0, 4.8)
    text(cx, 97.4, label, ha="center", fontsize=7, zorder=3)
ax.plot([27.0, 73.0], [93.4, 93.4], color=FAINT, lw=0.6, zorder=1)
ax.plot([27.0, 27.0], [95.0, 93.4], color=FAINT, lw=0.6, zorder=1)
ax.plot([73.0, 73.0], [95.0, 93.4], color=FAINT, lw=0.6, zorder=1)
arrow(50.0, 93.4, 50.0, 91.3)

# ---------------------------------------------------------------- bands
for (fam, colour, fam_sub, dims), top, h in zip(FAMILIES, BAND_TOP, BAND_H):
    bot = top - h
    BAND_H_THIS = h
    ax.add_patch(Rectangle((GUTTER_X, bot), 100.0 - GUTTER_X, BAND_H_THIS,
                           facecolor=BAND, edgecolor="none", zorder=0))
    ax.plot([GUTTER_X, GUTTER_X], [bot, top], color=colour, lw=2.2,
            solid_capstyle="butt", zorder=3)

    text(GUTTER_X + 2.0, top - DIM_Y, fam.upper(), fontsize=7.6, ha="left",
         fontweight="bold", color=colour, zorder=3)
    ax.text(GUTTER_X + 2.0, top - SUB_Y - 0.6, fam_sub, fontsize=6.3,
            ha="left", va="top", color=MUTED, zorder=3, linespacing=1.35)

    for di, (dim, sub, metrics) in enumerate(dims):
        dx = DIM_X0 + di * (DIM_W + DIM_GAP)
        if di:
            ax.plot([dx - DIM_GAP / 2, dx - DIM_GAP / 2],
                    [bot + 1.8, top - 1.8], color=FAINT, lw=0.5, zorder=1)
        text(dx, top - DIM_Y, dim, fontsize=7, ha="left",
             fontweight="bold", zorder=3)
        text(dx, top - SUB_Y, sub, fontsize=6.2, ha="left", color=MUTED,
             style="italic", zorder=3)
        for mi, m in enumerate(metrics):
            text(dx, top - MET_Y0 - mi * MET_DY, m, fontsize=6.2,
                 ha="left", zorder=3)

# ---------------------------------------------------------------- output
arrow(50.0, BAND_BOT_LAST, 50.0, BAND_BOT_LAST - 3.2)
box(26.0, BAND_BOT_LAST - 8.6, 48.0, 4.8)
text(50.0, BAND_BOT_LAST - 6.2, "Multidimensional quality profile",
     ha="center", fontsize=7, zorder=3)

fig.savefig(OUT / "Figure1.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Figure1.png", bbox_inches="tight", dpi=600, facecolor="white")

d = sum(len(x[3]) for x in FAMILIES)
n = sum(len(m) for x in FAMILIES for _, _, m in x[3])
print(f"families={len(FAMILIES)}  dimensions={d}  metrics={n}")
