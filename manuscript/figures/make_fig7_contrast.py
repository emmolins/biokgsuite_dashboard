#!/usr/bin/env python3
"""
Figure 7 (contrast variant) | Levels, then the contrasts that carry the claim.

  a  KG-grounding lift over no-KG, all 116 pairs   -- establishes the ordering
  b  every pairwise KG difference, all 116 vs the 72-pair common support,
     against zero                                   -- tests whether it survives

Why this beats two panels of levels: the claim is about DIFFERENCES between
graphs, and a difference of 0.05 sitting on top of a lift of 0.50 is nearly
invisible when both panels plot levels. Panel b plots the contrast itself
against a zero line, which is the quantity the sentence is about, and shows
each contrast moving toward zero as coverage is held fixed.

Same inputs and estimator as make_fig7.py.

Run:  python manuscript/figures/make_fig7_contrast.py
"""
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import setup_manuscript_style, save_fig, KG_PALETTE

OUT = Path(__file__).resolve().parent
setup_manuscript_style()

INK, GREY, HAIR, GRID = "#1A1A1A", "#6E6E6E", "#C8C8C8", "#E8E6E1"
KGS = ["primekg", "drkg", "biokg"]
KG_LAB = {"primekg": "PrimeKG", "drkg": "DRKG", "biokg": "BioKG"}
MODELS = {"gpt-4.1-mini": "GPT-4.1-mini",
          "gemini:gemini-3.1-flash-lite": "Gemini-3.1-Flash-Lite",
          "llama3.3:70b": "Llama-3.3-70B"}
N_BOOT, SEED = 20000, 42   # 2000 left ~0.004 jitter on CI endpoints

runs = pd.concat([pd.read_csv(f) for f in
                  sorted(glob.glob(str(BASE / "results/tables/09_llm_runs/09_big_*.csv")))])
runs = runs[runs.model.isin(MODELS)].copy()
runs["arm"] = np.where(runs.condition == "no_kg", "no_kg", runs.kg)
runs["M"] = runs.model.map(MODELS)

cov = pd.read_csv(BASE / "data/gold_standards/coverage_annotation_v4.csv")
wide = cov.pivot(index="Pair_ID", columns="kg", values="covered").fillna(0)
meta = cov.drop_duplicates("Pair_ID").set_index("Pair_ID")
COMMON = set(meta.loc[wide.index[wide[KGS].sum(axis=1) == 3], "disease_name"])

rng = np.random.RandomState(SEED)


def per_disease(df):
    cell = df.groupby(["M", "disease", "arm"])["reciprocal_rank"].mean().reset_index()
    return cell.pivot_table(index=["M", "disease"], columns="arm",
                            values="reciprocal_rank")


def boot(d):
    d = np.asarray(d, dtype=float)
    bs = d[rng.randint(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


PV_ALL = per_disease(runs)
PV_COM = per_disease(runs[runs.disease.isin(COMMON)])
ORDER = [MODELS[k] for k in ["gpt-4.1-mini", "gemini:gemini-3.1-flash-lite",
                             "llama3.3:70b"]]
PAIRS = list(itertools.combinations(KGS, 2))

MM = 1 / 25.4
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 3.05),
                         gridspec_kw={"width_ratios": [1.0, 1.05]})
fig.subplots_adjust(wspace=0.62)

# ------------------------------------------------- a | lift levels, all 116
ax = axes[0]
ax.set_axisbelow(True)
ax.grid(True, axis="y", color=GRID, lw=.5)
XS, OFF = np.arange(len(ORDER)), 0.24
for j, kg in enumerate(KGS):
    stats = [boot((PV_ALL.loc[m][kg] - PV_ALL.loc[m]["no_kg"]).dropna())
             for m in ORDER]
    ax.errorbar(XS + (j - 1) * OFF, [s[0] for s in stats],
                yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
                fmt="o", ms=4.0, color=KG_PALETTE[kg], ecolor=KG_PALETTE[kg],
                elinewidth=.9, capsize=2.0, capthick=.9, mec="white", mew=.6,
                zorder=4, label=KG_LAB[kg])
ax.set_xticks(XS)
ax.set_xticklabels([m.replace("-3.1-", "-3.1-\n") for m in ORDER], fontsize=5.8,
                   linespacing=1.35)
ax.set_xlim(-0.55, len(ORDER) - 0.45)
ax.set_ylim(0, 0.68)
ax.set_ylabel("Mean per-pair MRR lift over no-KG", fontsize=6.4, color=INK,
              labelpad=3)
ax.tick_params(axis="both", length=0, pad=3)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(HAIR)
ax.legend(loc="lower left", frameon=False, fontsize=6.0, handletextpad=0.35,
          labelspacing=0.32, borderaxespad=0.3)

# ------------------------------------- b | pairwise contrasts against zero
ax = axes[1]
ax.set_axisbelow(True)
ax.axvline(0, color=INK, lw=.8, zorder=2)

# Rows are real y-ticks, so the labels sit outside the data area and cannot
# collide with the whiskers. Headers are ticks too, set bold afterwards.
ticks, labels, bold, y = [], [], [], 0.0
DY = 0.19                      # vertical offset separating the two analyses
for m in ORDER:
    ticks.append(y); labels.append(m); bold.append(True)
    y -= 1.0
    for a, b in PAIRS:
        ticks.append(y); labels.append(f"{KG_LAB[a]} − {KG_LAB[b]}"); bold.append(False)
        s_all = boot((PV_ALL.loc[m][a] - PV_ALL.loc[m][b]).dropna())
        s_com = boot((PV_COM.loc[m][a] - PV_COM.loc[m][b]).dropna())
        ax.plot([s_all[1], s_all[2]], [y + DY, y + DY], color="#B8B8B8",
                lw=.8, zorder=3)
        ax.plot(s_all[0], y + DY, "o", ms=3.4, mfc="white", mec="#8C8C8C",
                mew=.8, zorder=4)
        excl = not (s_com[1] < 0 < s_com[2])
        ax.plot([s_com[1], s_com[2]], [y - DY, y - DY], color=INK, lw=.9,
                zorder=4)
        ax.plot(s_com[0], y - DY, "o", ms=3.8, mfc=INK if excl else "white",
                mec=INK, mew=.9, zorder=5)
        y -= 1.0
    y -= 0.45

ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=5.7)
for t, is_head in zip(ax.get_yticklabels(), bold):
    t.set_fontweight("bold" if is_head else "normal")
    t.set_fontsize(6.0 if is_head else 5.7)
    t.set_color(INK)
ax.set_ylim(min(ticks) - 0.9, 0.9)
ax.set_xlim(-0.16, 0.16)
ax.set_xlabel("Difference in MRR lift between graphs", fontsize=6.4,
              color=INK, labelpad=3)
ax.tick_params(axis="x", labelsize=6.0, length=2.5, color=HAIR)
ax.tick_params(axis="y", length=0, pad=2)
ax.grid(True, axis="x", color=GRID, lw=.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(HAIR)

h_all = plt.Line2D([], [], marker="o", ls="none", ms=3.4, mfc="white",
                   mec="#8C8C8C", mew=.8, color="#B8B8B8",
                   label="all 116 pairs")
h_com = plt.Line2D([], [], marker="o", ls="none", ms=3.8, mfc="white",
                   mec=INK, mew=.9, color=INK, label="72 covered by all three")
h_sig = plt.Line2D([], [], marker="o", ls="none", ms=3.8, mfc=INK, mec=INK,
                   label="excludes zero")
ax.legend(handles=[h_all, h_com, h_sig], loc="upper center",
          bbox_to_anchor=(0.5, -0.115), ncol=3, frameon=False, fontsize=5.7,
          handletextpad=0.35, columnspacing=1.4, borderaxespad=0.0)

for a_, letter, sub in [(axes[0], "a", "Grounding lift, all 116 pairs"),
                        (axes[1], "b", "Pairwise differences between graphs")]:
    a_.text(0.0, 1.045, letter, transform=a_.transAxes, fontsize=8,
            fontweight="bold", color=INK, ha="left", va="baseline")
    a_.text(0.055, 1.045, sub, transform=a_.transAxes, fontsize=6.6,
            color=INK, ha="left", va="baseline")

save_fig(fig, OUT, "Figure7_contrast", dpi=600)
