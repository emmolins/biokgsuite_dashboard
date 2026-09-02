#!/usr/bin/env python3
"""
Figure 6 (schematic) | The contamination-controlled repurposing task.

Built to Nature's scientific-illustration guidance:
  - panel letters + short subheadings carry the hierarchy, no chrome
  - black type; colour reserved for the three KG arms and for the single
    most important element, the approved drug
  - saturation weighted by significance: the approved drug is the most
    saturated mark, distractors are pale, the blinded pool is flat grey
  - noun labels only; prose lives in the caption
  - one arrow weight, one arrow style
  - visual grammar for the blinding:
        filled rust   = the approved drug, revealed
        rust outline  = its identity, known to the scorer but NOT the model
        flat grey     = what the model actually sees

Layout is centred: panel widths are measured, then outer margins and
gutters are solved so the three columns sit symmetrically in 180 mm.

Pure schematic: reads no data, safe to re-run.

Run:  python manuscript/figures/make_fig6.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import setup_manuscript_style, save_fig, KG_PALETTE

OUT = Path(__file__).resolve().parent
setup_manuscript_style()

INK = "#1A1A1A"
GREY = "#767676"
HAIR = "#C8C8C8"
PALE = "#E9E7E2"          # the blinded pool
CARD = "#F5F4F1"
POS = "#C0563C"

# ---- centred layout -------------------------------------------------------
W_A, W_B, W_C = 28.2, 22.4, 27.1          # measured panel content widths
MARGIN = 2.5
GUTTER = (100 - 2 * MARGIN - (W_A + W_B + W_C)) / 2
AX = MARGIN
BX = AX + W_A + GUTTER
CX = BX + W_B + GUTTER

HEAD = 38.0                               # shared heading baseline
Y_TOP, Y_BOT = 39.3, 9.0                  # measured content extents
PAD = 1.55
MM = 1 / 25.4

fig, ax = plt.subplots(figsize=(180 * MM, 2.80))
ax.set_xlim(0, 100)
ax.set_ylim(Y_BOT - PAD, Y_TOP + PAD)
ax.axis("off")

BH, BS = 1.62, 1.98       # candidate bar height / row pitch
TOP = 26.8                # top of the candidate stack
SEC = 29.4                # section-label baseline
MID = 20.7                # shared y for the two inter-panel arrows
FOOT = 9.75               # shared baseline for the three footer labels


# ------------------------------------------------------------------ helpers
def panel(x, letter, sub):
    """Letter and subheading share one baseline; x is the panel's content
    left edge, so the letter aligns with the column beneath it."""
    ax.text(x, HEAD, letter, fontsize=8, fontweight="bold", color=INK,
            ha="left", va="baseline")
    ax.text(x + 2.9, HEAD, sub, fontsize=6.6, color=INK, ha="left",
            va="baseline")


def label(x, y, s, fs=5.6, color=GREY, ha="left"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center")


def bar(x, y, w, h, fc, ec="none", lw=0.0, text="", tc=INK, fs=5.1,
        bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4,
                fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, color=HAIR, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=6, color=color, lw=lw,
                                 zorder=2, shrinkA=0, shrinkB=0))


def neighbourhood(cx, cy, color, s=1.55):
    for dx, dy in [(-s, s * .58), (s, s * .50), (s * .10, -s * .80)]:
        ax.plot([cx, cx + dx], [cy, cy + dy], color=color, lw=0.6,
                alpha=.8, zorder=3)
        ax.add_patch(Circle((cx + dx, cy + dy), .32, fc="white", ec=color,
                            lw=.6, zorder=4))
    ax.add_patch(Circle((cx, cy), .40, fc=color, ec="none", zorder=5))


# ================================================== a | query construction
panel(AX + 0.6, "a", "Query construction")

# stacked cards -- front card (offset 0) drawn last, so it sits on top
for k, off in enumerate((1.6, 0.8, 0.0)):
    ax.add_patch(FancyBboxPatch((AX + 0.6 + off, 32.4 + off), 17.0, 2.9,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc=CARD if off else "white", ec=HAIR,
                                lw=.6, zorder=2 + k))
ax.text(AX + 1.7, 33.85, "Crohn's disease", fontsize=6.0, color=INK,
        va="center", style="italic", zorder=6)
label(AX + 19.6, 33.85, "116 approved\npairs", fs=5.3)

arrow(AX + 9.1, 32.2, AX + 9.1, 30.3)

# --- candidate pool. Worked example is a real row of the 116-pair set:
# upadacitinib, approved 2019, gained its Crohn's disease indication in the
# 2023 window. Distractors are approved drugs with no Crohn's indication --
# checked, because a drug that IS indicated would be a false negative in a
# figure about contamination control.
label(AX + 0.6, SEC, "Candidate pool")
named = ["Upadacitinib", "Metformin", "Warfarin", "Atorvastatin",
         "Sertraline", "Omeprazole", "Cetirizine", "Amlodipine"]
for i, nm in enumerate(named):
    y = TOP - i * BS
    is_pos = (i == 0)
    bar(AX + 0.6, y, 13.2, BH, fc=POS if is_pos else "white",
        ec="none" if is_pos else HAIR, lw=0 if is_pos else .6,
        text=nm, tc="white" if is_pos else GREY, bold=is_pos)

# --- blinded pool: letters in the order the model sees them (A-H).
POS_LETTER = "C"                     # sirolimus becomes Drug C
label(AX + 18.6, SEC, "Relabelled, shuffled")
for i, ltr in enumerate("ABCDEFGH"):
    y = TOP - i * BS
    keyed = (ltr == POS_LETTER)
    bar(AX + 18.6, y, 9.6, BH, fc=PALE,
        ec=POS if keyed else "none", lw=.8 if keyed else 0,
        text=f"Drug {ltr}", tc=INK)

arrow(AX + 14.4, MID, AX + 17.8, MID)
label(AX + 16.1, MID + 1.5, "blind", fs=5.3, color=INK, ha="center")

# the shuffle, made explicit: row 1 on the left becomes row 3 on the right
ax.add_patch(FancyArrowPatch(
    (AX + 13.9, TOP + BH / 2), (AX + 18.5, TOP - 2 * BS + BH / 2),
    arrowstyle="-", color=POS, lw=.7, alpha=.9, zorder=5,
    connectionstyle="arc3,rad=-0.32", shrinkA=1, shrinkB=1))

bar(AX + 0.6, FOOT - 0.75, 2.8, 1.5, fc=POS)
label(AX + 4.1, FOOT, "approved", fs=5.2, color=INK)
bar(AX + 12.0, FOOT - 0.75, 2.8, 1.5, fc="white", ec=HAIR, lw=.6)
label(AX + 15.5, FOOT, "distractor", fs=5.2)

# ============================================================== b | arms
arrow(AX + W_A + 1.2, MID, BX - 1.2, MID)
panel(BX + 0.4, "b", "Arms")

ARM_TOP = 35.3          # align with the top cards of panels a and c
arms = [("No KG", None), ("PrimeKG", KG_PALETTE["primekg"]),
        ("DRKG", KG_PALETTE["drkg"]), ("BioKG", KG_PALETTE["biokg"])]
for i, (nm, col) in enumerate(arms):
    y = ARM_TOP - 4.0 - i * 5.05
    ax.add_patch(FancyBboxPatch((BX + 0.4, y), 22.0, 4.0,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                fc="white" if col else CARD, ec=HAIR,
                                lw=.6, zorder=2))
    if col:
        ax.add_patch(FancyBboxPatch((BX + 0.4, y), 0.85, 4.0,
                                    boxstyle="round,pad=0,rounding_size=0.42",
                                    fc=col, ec="none", zorder=3))
    ax.text(BX + 2.4, y + 2.0, nm, fontsize=6.2, color=INK, va="center",
            zorder=4)
    if col:
        neighbourhood(BX + 18.4, y + 2.0, col)
    else:
        ax.text(BX + 18.4, y + 2.0, "—", fontsize=7, color=GREY,
                ha="center", va="center", zorder=4)

label(BX + 0.4, FOOT, "one-hop gene, pathway and\ndisease neighbourhoods",
      fs=5.3)

# ========================================================== c | readout
arrow(BX + W_B + 1.2, MID, CX - 1.2, MID)
panel(CX + 0.4, "c", "Readout")

ax.add_patch(FancyBboxPatch((CX + 0.4, 32.4), 19.0, 2.9,
                            boxstyle="round,pad=0,rounding_size=0.5",
                            fc=CARD, ec=HAIR, lw=.6, zorder=2))
ax.text(CX + 1.6, 33.85, "reasoning  →  JSON", fontsize=5.9, color=INK,
        va="center", zorder=3)
label(CX + 20.6, 33.85, "failed parse\n= miss", fs=5.3)
arrow(CX + 9.9, 32.2, CX + 9.9, 30.3)

label(CX + 0.4, SEC, "Ranking")
for i, ltr in enumerate("GCAEBHDF"):
    y = TOP - i * BS
    is_pos = (ltr == POS_LETTER)
    ax.text(CX + 2.0, y + BH / 2, str(i + 1), fontsize=5.0, ha="right",
            va="center", color=HAIR)
    bar(CX + 2.6, y, 9.6, BH, fc=POS if is_pos else "white",
        ec="none" if is_pos else HAIR, lw=0 if is_pos else .6,
        text=f"Drug {ltr}", tc="white" if is_pos else GREY, bold=is_pos)
    if is_pos:
        ax.annotate("", (CX + 12.4, y + BH / 2), (CX + 14.4, y + BH / 2),
                    arrowprops=dict(arrowstyle="-", color=POS, lw=.6))
        ax.text(CX + 15.0, y + BH / 2, "rank of the\napproved drug",
                fontsize=5.3, color=INK, va="center", ha="left")

label(CX + 2.6, FOOT, "MRR,  hits@$k$", fs=5.6, color=INK)

save_fig(fig, OUT, "Figure6", dpi=600)
print(f"  layout: margin {MARGIN:.2f}, gutter {GUTTER:.2f}, "
      f"panels at x = {AX:.1f} / {BX:.1f} / {CX:.1f}")
