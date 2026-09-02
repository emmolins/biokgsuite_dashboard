#!/usr/bin/env python3
"""
BioKGSuite taxonomy | seven quality dimensions in three families.

Drawn in the Figure 6 visual language: rounded cards, hairline borders, a
coloured left spine carrying family identity, black type for names, grey
for secondary text, one hairline weight for the hierarchy connectors.

Laid out as three family columns with dimension cards stacked, rather than
seven columns across. Seven columns leaves ~50 pt per card, which forces
"Task perf." and "Nbhd retrieval"; three columns leave ~142 pt, so every
dimension and metric carries its full published name. Card heights follow
their metric count, so no card contains dead space.

Checkpoint keys for the display names:
  Graph cohesion     [connectedness]      Clustering coefficient [small_world]
  Graph reachability [reachability]       Community purity  [community_purity]
  Random dropout     [random_stability]   Periphery dropout [periphery_stability]
  Uncertainty quant. [uq_coverage]        Prospective       [temporal_gen]

NOTE: 7 dimensions, 19 metrics, verified against the seven checkpoints.
README.md still says "18 metrics" and is stale.

Run:  python manuscript/figures/make_taxonomy.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import (setup_manuscript_style, save_fig,
                          KG_OCEAN_BLUE, KG_SAGE, KG_RUST)

OUT = Path(__file__).resolve().parent
setup_manuscript_style()

INK, GREY, HAIR, CARD = "#1A1A1A", "#6E6E6E", "#C8C8C8", "#F5F4F1"
METRIC = "#454545"

# (dimension, gloss, [metric lines])  -- lines are pre-wrapped to the column
FAMILIES = [
    ("Content", KG_OCEAN_BLUE, [
        ("Coverage", "how complete",
         ["Entity coverage · Relation coverage"]),
        ("Annotation accuracy", "how accurate",
         ["Entity validity · Relational consistency"]),
        ("Trustworthiness", "how well evidenced",
         ["Edge traceability · Source diversity ·",
          "Uncertainty quantification"]),
    ]),
    ("Structure", KG_SAGE, [
        ("Topology", "how interconnected",
         ["Graph cohesion · Clustering coefficient ·",
          "Graph reachability · Community purity"]),
        ("Stability", "how robust",
         ["Random dropout · Periphery dropout"]),
    ]),
    ("Inference", KG_RUST, [
        ("Task performance", "how useful",
         ["Link prediction · Neighbourhood retrieval ·",
          "Multi-hop reasoning"]),
        ("Generalisation", "how transferable",
         ["Data-sparse · Cross-domain · Prospective"]),
    ]),
]

N_DIM = sum(len(d) for _, _, d in FAMILIES)
# join the wrapped lines first: a trailing "·" is a wrap marker, not a metric
N_MET = sum(len([t for t in " ".join(m).split("·") if t.strip()])
            for _, _, d in FAMILIES for *_, m in d)

# ---- grid ------------------------------------------------------------------
MARGIN, COL_W, GAP = 1.55, 30.3, 3.0
SPINE, PAD_L = 0.8, 2.0
LINE_H, CARD_GAP = 2.05, 1.5
HEAD_H, FOOT_H = 4.9, 0.9          # card space above / below the metric lines

Y_ROOT_T, ROOT_H = 42.0, 3.2
Y_JOIN = 36.6
Y_FAM = 34.4
Y_RULE = 33.2
CARD_T = 31.6

MM = 1 / 25.4
W_IN = 180 * MM


def card_h(n_lines):
    return HEAD_H + n_lines * LINE_H + FOOT_H


col_bottoms = []
for _, _, dims in FAMILIES:
    y = CARD_T
    for *_, lines in dims:
        y -= card_h(len(lines)) + CARD_GAP
    col_bottoms.append(y + CARD_GAP)

Y_LO = min(col_bottoms) - 1.2
Y_HI = Y_ROOT_T + 1.6
H_IN = W_IN * (Y_HI - Y_LO) / 100

fig, ax = plt.subplots(figsize=(W_IN, H_IN))
ax.set_xlim(0, 100)
ax.set_ylim(Y_LO, Y_HI)
ax.axis("off")

pending = []          # (name artist, y, gloss) -- placed after first draw
xs = [MARGIN + i * (COL_W + GAP) for i in range(3)]
centres = [x + COL_W / 2 for x in xs]

# ---- root card and hierarchy connectors ------------------------------------
ax.add_patch(FancyBboxPatch((50 - 9.0, Y_ROOT_T - ROOT_H), 18.0, ROOT_H,
                            boxstyle="round,pad=0,rounding_size=0.5",
                            fc=CARD, ec=HAIR, lw=.6, zorder=2))
ax.text(50, Y_ROOT_T - ROOT_H / 2, "BioKGSuite", fontsize=6.6,
        fontweight="bold", color=INK, ha="center", va="center", zorder=3)
ax.plot([50, 50], [Y_ROOT_T - ROOT_H, Y_JOIN], color=HAIR, lw=.7, zorder=1)
ax.plot([centres[0], centres[-1]], [Y_JOIN, Y_JOIN], color=HAIR, lw=.7,
        zorder=1)
for c in centres:
    ax.plot([c, c], [Y_JOIN, Y_FAM + 1.3], color=HAIR, lw=.7, zorder=1)

# ---- families --------------------------------------------------------------
for (fam, colour, dims), x0 in zip(FAMILIES, xs):
    ax.text(x0 + COL_W / 2, Y_FAM, fam, fontsize=6.6, fontweight="bold",
            color=INK, ha="center", va="baseline")
    ax.plot([x0, x0 + COL_W], [Y_RULE, Y_RULE], color=colour, lw=1.4,
            solid_capstyle="butt", zorder=3)

    y = CARD_T
    for name, gloss, lines in dims:
        h = card_h(len(lines))
        ax.add_patch(FancyBboxPatch((x0, y - h), COL_W, h,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    fc="white", ec=HAIR, lw=.6, zorder=2))
        ax.add_patch(FancyBboxPatch((x0, y - h), SPINE, h,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=colour, ec="none", zorder=3))
        tx = x0 + PAD_L
        t = ax.text(tx, y - 2.3, name, fontsize=6.2, fontweight="bold",
                    color=INK, ha="left", va="center", zorder=4)
        # defer the gloss: its x depends on the name's true rendered width,
        # which cannot be guessed from character count
        pending.append((t, y - 2.3, gloss))
        for j, ln in enumerate(lines):
            ax.text(tx, y - HEAD_H - 0.4 - j * LINE_H, ln, fontsize=5.2,
                    color=METRIC, ha="left", va="center", zorder=4)
        y -= h + CARD_GAP

# second pass: now the renderer knows how wide each name actually is
fig.canvas.draw()
inv = ax.transData.inverted()
for t, ty, gloss in pending:
    x1 = inv.transform(t.get_window_extent(fig.canvas.get_renderer()))[1][0]
    ax.text(x1 + 1.3, ty, gloss, fontsize=5.2, color=GREY, ha="left",
            va="center", style="italic", zorder=4)

save_fig(fig, OUT, "Taxonomy", dpi=600)
print(f"  {N_DIM} dimensions, {N_MET} metrics (not drawn) — "
      f"README.md says 18, still stale")
