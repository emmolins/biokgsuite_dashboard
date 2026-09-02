#!/usr/bin/env python3
"""
Five different ways to present the §3 grounding result, for choosing between.

Each option answers a different question, and they are not interchangeable:

  1  Decomposition      how much of the gain is availability vs quality
  2  Per-pair scatter   is the advantage general or driven by a few pairs
  3  Coverage strata    the mechanism, shown directly
  4  Distribution       where in the rank distribution the gain lives
  5  Rank composition   what a reader would actually experience using the tool

Reads only the derived frames plus the raw runs. Run 09_analysis.ipynb first.

Run:  python manuscript/figures/fig7_options.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))

from plotting import (setup_manuscript_style, KG_PALETTE,  # noqa: E402
                      TEXT_COLOR, GRID_COLOR)

setup_manuscript_style()
OUT = Path(__file__).resolve().parent
DERIVED = BASE / "results" / "tables" / "09_derived"
RUNS = BASE / "results" / "tables" / "09_llm_runs"

INK, MUTED, FAINT = "#1A1A1A", "#8C8C8C", "#C9C9C9"
COL = {"PrimeKG": KG_PALETTE["primekg"], "DRKG": KG_PALETTE["drkg"],
       "BioKG": KG_PALETTE["biokg"], "No KG": "#9E9E9E"}
KGS = ["PrimeKG", "DRKG", "BioKG"]
KEY = {"PrimeKG": "primekg", "DRKG": "drkg", "BioKG": "biokg"}
MODELS = ["GPT-4.1-mini", "Gemini-3.1-Flash-Lite", "Llama-3.3-70B"]
SHORT = {"GPT-4.1-mini": "GPT-4.1-mini", "Gemini-3.1-Flash-Lite": "Gemini-Flash-Lite",
         "Llama-3.3-70B": "Llama-3.3-70B"}
CHANCE = float(np.mean([1 / k for k in range(1, 9)]))

pq = pd.read_csv(DERIVED / "per_query_main.csv")
cov = pd.read_csv(DERIVED / "pair_coverage.csv")
cov["pair_id"] = cov.index + 1
cov["n_cov"] = cov[["primekg", "drkg", "biokg"]].sum(axis=1).astype(int)
covflags = cov[["pair_id", "primekg", "drkg", "biokg", "n_cov", "all3"]].rename(
    columns={"primekg": "cov_primekg", "drkg": "cov_drkg", "biokg": "cov_biokg"})
pq = pq.merge(covflags, on="pair_id")
# MRR columns are primekg/drkg/biokg/none; coverage flags are cov_*.


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID_COLOR, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def head(ax, letter, title, x=-0.16):
    ax.text(x, 1.13, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left", color=INK)
    ax.text(x + 0.05, 1.125, title, transform=ax.transAxes, fontsize=7.2,
            va="top", ha="left", color=INK)


# =====================================================================
# 1 · Decomposition — availability vs quality
# =====================================================================
fig1, axes = plt.subplots(1, 3, figsize=(7.09, 2.5), sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.80, wspace=0.14)

for ax, m in zip(axes, MODELS):
    p = pq[pq.model == m]
    for i, kg in enumerate(KGS):
        k = KEY[kg]
        covered = p[p["cov_" + k] == 1]
        gain_cov = (covered[k] - covered["none"]).mean()
        gain_all = (p[k] - p["none"]).mean()
        frac = len(covered) / len(p)
        # total gain = (fraction covered) x (gain when covered)
        ax.bar(i, gain_all, 0.62, color=COL[kg], zorder=3)
        ax.bar(i, gain_cov, 0.62, facecolor="none", edgecolor=INK, lw=0.7,
               ls=(0, (2, 1.5)), zorder=4)
        ax.text(i, gain_cov + 0.012, f"{frac:.0%}", ha="center",
                fontsize=6.0, color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels(KGS, fontsize=6.6)
    ax.set_xlim(-0.62, 2.62)
    clean(ax)
    ax.set_title(SHORT[m], fontsize=7, pad=5)
axes[0].set_ylabel("MRR gain over no-KG", labelpad=3)
head(axes[0], "1", "Realised gain (filled) against gain when the pair is covered "
                   "(dashed); % = coverage", x=-0.16)
fig1.savefig(OUT / "fig7_opt1_decomposition.pdf", bbox_inches="tight",
             facecolor="white")
fig1.savefig(OUT / "fig7_opt1_decomposition.png", bbox_inches="tight", dpi=300,
             facecolor="white")

# =====================================================================
# 2 · Per-pair scatter — PrimeKG against DRKG, pair by pair
# =====================================================================
fig2, axes = plt.subplots(1, 3, figsize=(7.09, 2.6), sharex=True, sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.80, wspace=0.12)

for ax, m in zip(axes, MODELS):
    p = pq[pq.model == m]
    both = p[p.all3]
    other = p[~p.all3]
    ax.plot([0, 1], [0, 1], color=FAINT, lw=0.7, zorder=1)
    ax.scatter(other.drkg, other.primekg, s=13, facecolor="white",
               edgecolor=MUTED, lw=0.5, zorder=2, label="not in all three")
    ax.scatter(both.drkg, both.primekg, s=13, color=COL["PrimeKG"],
               alpha=0.75, lw=0, zorder=3, label="covered by all three")
    win = (p.primekg > p.drkg).mean()
    ax.text(0.04, 0.95, f"PrimeKG higher on {win:.0%} of pairs",
            transform=ax.transAxes, fontsize=6.2, va="top", color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("DRKG", labelpad=2)
    clean(ax, grid="both")
    ax.set_title(SHORT[m], fontsize=7, pad=5)
axes[0].set_ylabel("PrimeKG", labelpad=3)
axes[2].legend(frameon=False, loc="lower right", fontsize=6, handletextpad=0.2,
               borderaxespad=0.2)
head(axes[0], "2", "Reciprocal rank per pair, PrimeKG against DRKG", x=-0.16)
fig2.savefig(OUT / "fig7_opt2_scatter.pdf", bbox_inches="tight",
             facecolor="white")
fig2.savefig(OUT / "fig7_opt2_scatter.png", bbox_inches="tight", dpi=300,
             facecolor="white")

# =====================================================================
# 3 · Coverage strata — the mechanism, directly
# =====================================================================
fig3, axes = plt.subplots(1, 3, figsize=(7.09, 2.5), sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.80, wspace=0.14)

for ax, m in zip(axes, MODELS):
    p = pq[pq.model == m]
    for i, kg in enumerate(KGS):
        k = KEY[kg]
        for j, (lab, sub) in enumerate((("covered", p[p["cov_" + k] == 1]),
                                        ("not covered", p[p["cov_" + k] == 0]))):
            if not len(sub):
                continue
            x = i + (j - 0.5) * 0.34
            v = sub[k]
            ax.plot([x, x], [v.mean() - v.sem() * 1.96,
                             v.mean() + v.sem() * 1.96],
                    color=COL[kg], lw=1.0, zorder=2)
            ax.plot([x], [v.mean()], marker="o" if j == 0 else "s", ms=4.2,
                    mfc=COL[kg] if j == 0 else "white", mec=COL[kg], mew=1.0,
                    zorder=3)
    ax.axhline(CHANCE, color=FAINT, ls=(0, (2, 2)), lw=0.7, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(KGS, fontsize=6.6)
    ax.set_xlim(-0.6, 2.6)
    clean(ax)
    ax.set_title(SHORT[m], fontsize=7, pad=5)
axes[0].set_ylabel("Mean reciprocal rank", labelpad=3)
axes[0].text(0.02, CHANCE + 0.015, "chance", transform=axes[0].get_yaxis_transform(),
             fontsize=6, color=MUTED)
axes[2].legend(handles=[
    plt.Line2D([], [], marker="o", ls="none", ms=4.2, color=MUTED,
               label="pair is in the graph"),
    plt.Line2D([], [], marker="s", ls="none", ms=4.2, mfc="white",
               mec=MUTED, label="pair is absent")],
    frameon=False, loc="lower right", fontsize=6, handletextpad=0.2)
head(axes[0], "3", "Performance splits on whether the graph contains the pair",
     x=-0.16)
fig3.savefig(OUT / "fig7_opt3_strata.pdf", bbox_inches="tight",
             facecolor="white")
fig3.savefig(OUT / "fig7_opt3_strata.png", bbox_inches="tight", dpi=300,
             facecolor="white")

# =====================================================================
# 4 · Distribution — ECDF of reciprocal rank
# =====================================================================
fig4, axes = plt.subplots(1, 3, figsize=(7.09, 2.5), sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.80, wspace=0.12)

for ax, m in zip(axes, MODELS):
    p = pq[pq.model == m]
    for lab, col in [("No KG", "none"), ("PrimeKG", "primekg"),
                     ("DRKG", "drkg"), ("BioKG", "biokg")]:
        v = np.sort(p[col].values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                color=COL[lab], lw=1.2, zorder=3 if lab != "No KG" else 2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Reciprocal rank", labelpad=2)
    clean(ax, grid="both")
    ax.set_title(SHORT[m], fontsize=7, pad=5)
axes[0].set_ylabel("Cumulative fraction of pairs", labelpad=3)
axes[0].legend(handles=[plt.Line2D([], [], color=COL[k], lw=1.2, label=k)
                        for k in ["No KG"] + KGS],
               frameon=False, loc="upper left", fontsize=6,
               handletextpad=0.5, labelspacing=0.3)
head(axes[0], "4", "Full distribution, not just the mean", x=-0.16)
fig4.savefig(OUT / "fig7_opt4_ecdf.pdf", bbox_inches="tight", facecolor="white")
fig4.savefig(OUT / "fig7_opt4_ecdf.png", bbox_inches="tight", dpi=300,
             facecolor="white")

# =====================================================================
# 5 · Rank composition — where the true drug actually lands
# =====================================================================
FILES = {"GPT-4.1-mini": ["09_big_gpt.csv", "09_big_gpt_s12.csv"],
         "Gemini-3.1-Flash-Lite": ["09_big_glite.csv", "09_big_glite_s12.csv"],
         "Llama-3.3-70B": ["09_big_llama.csv", "09_big_llama_s1.csv",
                           "09_big_llama_s2.csv"]}
BANDS = [(1, 1, "rank 1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, 9, "6+")]
SHADE = ["#2E6B8A", "#7FA8BE", "#C3D4DE", "#EDEDEA"]

fig5, axes = plt.subplots(1, 3, figsize=(7.09, 2.5), sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.80, wspace=0.14)

for ax, m in zip(axes, MODELS):
    d = pd.concat([pd.read_csv(RUNS / f) for f in FILES[m]])
    for i, (lab, k) in enumerate([("No KG", "none"), ("PrimeKG", "primekg"),
                                  ("DRKG", "drkg"), ("BioKG", "biokg")]):
        sub = d[d.kg == k]
        bottom = 0.0
        for (lo, hi, _), colr in zip(BANDS, SHADE):
            frac = ((sub["rank"] >= lo) & (sub["rank"] <= hi)).mean()
            ax.bar(i, frac, 0.66, bottom=bottom, color=colr, zorder=3,
                   edgecolor="white", lw=0.5)
            bottom += frac
    ax.set_xticks(range(4))
    ax.set_xticklabels(["No KG", "PrimeKG", "DRKG", "BioKG"], fontsize=6.4)
    ax.set_ylim(0, 1)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(SHORT[m], fontsize=7, pad=5)
axes[0].set_ylabel("Fraction of queries", labelpad=3)
axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="white")
                        for c in SHADE],
               labels=[b[2] for b in BANDS], frameon=False, fontsize=6,
               loc="lower right", handlelength=1.0, handleheight=0.9,
               labelspacing=0.25, borderaxespad=0.2)
head(axes[0], "5", "Rank the approved drug receives", x=-0.16)
fig5.savefig(OUT / "fig7_opt5_ranks.pdf", bbox_inches="tight", facecolor="white")
fig5.savefig(OUT / "fig7_opt5_ranks.png", bbox_inches="tight", dpi=300,
             facecolor="white")

print("wrote 5 options to", OUT)
for m in MODELS:
    p = pq[pq.model == m]
    print(f"{m:24s} PrimeKG higher than DRKG on "
          f"{(p.primekg > p.drkg).mean():.0%} of pairs")
