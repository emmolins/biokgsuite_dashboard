#!/usr/bin/env python3
"""
Five ways to show the same thing: what coverage control does to the gap
between graphs.

  1  Slopegraph          gaps converging between two conditions
  2  Delta forest        the change itself, with a proper CI
  3  Share remaining     what fraction of the gap survives the control
  4  Diagonal scatter    all nine contrasts on one panel, y = x as reference
  5  Decomposition bars  each gap split into removed and residual

The delta CI is a difference-in-differences and the two subsets are nested,
so it is bootstrapped by resampling the 116 pairs and recomputing both gaps
on each resample. Bootstrapping the two independently would overstate the
uncertainty on the change.

Run:  python manuscript/figures/fig8_options.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
from plotting import setup_manuscript_style, KG_PALETTE, GRID_COLOR  # noqa: E402

setup_manuscript_style()
OUT = Path(__file__).resolve().parent
RUNS = BASE / "results" / "tables" / "09_llm_runs"
DERIVED = BASE / "results" / "tables" / "09_derived"
GOLD = BASE / "data" / "gold_standards"

INK, MUTED, FAINT = "#1A1A1A", "#8C8C8C", "#C9C9C9"
PALE, DARK = "#B9C7CF", KG_PALETTE["primekg"]
WARN = "#B0743F"
RNG = np.random.default_rng(0)
N_BOOT = 5000

MODELS = {"GPT-4.1-mini": ["09_big_gpt.csv", "09_big_gpt_s12.csv"],
          "Gemini-3.1-Flash-Lite": ["09_big_glite.csv", "09_big_glite_s12.csv"],
          "Llama-3.3-70B": ["09_big_llama.csv", "09_big_llama_s1.csv",
                            "09_big_llama_s2.csv"]}
MSHORT = {"GPT-4.1-mini": "GPT", "Gemini-3.1-Flash-Lite": "Gemini",
          "Llama-3.3-70B": "Llama"}
CONTRASTS = [("primekg", "drkg", "PrimeKG − DRKG"),
             ("primekg", "biokg", "PrimeKG − BioKG"),
             ("biokg", "drkg", "BioKG − DRKG")]
CCOL = {"PrimeKG − DRKG": DARK, "PrimeKG − BioKG": "#6E93A6",
        "BioKG − DRKG": "#B9A25C"}

# ---------------------------------------------------------------- data
cov = pd.read_csv(DERIVED / "pair_coverage.csv")
cov["pair_id"] = cov.index + 1
gold = pd.read_csv(GOLD / "gold_standard_v4_bigset.tsv", sep="\t").reset_index(drop=True)
gold["pair_id"] = gold.index + 1
COMMON = set(gold.merge(cov[["pair_id", "all3"]], on="pair_id")
                 .query("all3").disease_name)
N_ALL, N_COMMON = len(gold), int(cov.all3.sum())

top1 = {}
for m, files in MODELS.items():
    d = pd.concat([pd.read_csv(RUNS / f) for f in files])
    t = (d.groupby(["disease", "kg"])["rank"]
           .apply(lambda r: (r == 1).mean()).unstack())
    t["in_common"] = t.index.isin(COMMON)
    top1[m] = t

rows = []
for m, t in top1.items():
    dis = t.index.to_numpy()
    inc = t.in_common.to_numpy()
    for a, b, lab in CONTRASTS:
        diff = (t[a] - t[b]).to_numpy() * 100
        g_all, g_com = diff.mean(), diff[inc].mean()
        # nested bootstrap: resample pairs once, recompute both gaps
        bs_all, bs_com = [], []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, len(diff), len(diff))
            s, si = diff[idx], inc[idx]
            if si.sum() < 5:
                continue
            bs_all.append(s.mean())
            bs_com.append(s[si].mean())
        bs_all, bs_com = np.array(bs_all), np.array(bs_com)
        d_boot = bs_all - bs_com
        rows.append(dict(
            model=m, contrast=lab, gap_all=g_all, gap_common=g_com,
            delta=g_all - g_com,
            d_lo=np.percentile(d_boot, 2.5), d_hi=np.percentile(d_boot, 97.5),
            a_lo=np.percentile(bs_all, 2.5), a_hi=np.percentile(bs_all, 97.5),
            c_lo=np.percentile(bs_com, 2.5), c_hi=np.percentile(bs_com, 97.5),
            share=g_com / g_all if g_all > 0 else np.nan))
res = pd.DataFrame(rows)
res["key"] = res.model.map(MSHORT) + "  " + res.contrast


def finish(ax, grid=None):
    if grid:
        ax.grid(axis=grid, color=GRID_COLOR, lw=0.5)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def head(ax, n, title, x=-0.20):
    ax.text(x, 1.11, str(n), transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left", color=INK)
    ax.text(x + 0.035, 1.105, title, transform=ax.transAxes, fontsize=7.2,
            va="top", ha="left", color=INK)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300,
                facecolor="white")


# =====================================================================
# 1 · Slopegraph
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.9), sharey=True)
plt.subplots_adjust(left=0.09, right=0.985, bottom=0.135, top=0.80, wspace=0.10)
for ax, m in zip(axes, MODELS):
    sub = res[res.model == m]
    for _, r in sub.iterrows():
        c = CCOL[r.contrast]
        ax.plot([0, 1], [r.gap_all, r.gap_common], "-o", ms=4.0, lw=1.3,
                color=c, mec="white", mew=0.6, zorder=3)
        ax.annotate(f"{r.gap_all:.1f}", (0, r.gap_all), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.0, color=c)
        ax.annotate(f"{r.gap_common:.1f}", (1, r.gap_common), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.0, color=c)
    ax.axhline(0, color=FAINT, lw=0.7, zorder=1)
    ax.set_xlim(-0.42, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"all {N_ALL}", f"common {N_COMMON}"], fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=3)
    finish(ax)
    ax.set_title(m, fontsize=7, pad=6)
axes[0].set_ylabel("Gap in rank-1 rate (pp)", labelpad=3)
axes[2].legend(handles=[plt.Line2D([], [], color=c, lw=1.3, label=k)
                        for k, c in CCOL.items()],
               frameon=False, fontsize=6, loc="upper right", labelspacing=0.3)
head(axes[0], 1, "Slopegraph: gaps between the two conditions", x=-0.16)
save(fig, "fig8_opt1_slope")

# =====================================================================
# 2 · Delta forest — the change itself
# =====================================================================
fig, ax = plt.subplots(figsize=(7.09, 2.9))
plt.subplots_adjust(left=0.30, right=0.975, bottom=0.165, top=0.83)
y = 0
for m in MODELS:
    for _, _, lab in CONTRASTS:
        r = res[(res.model == m) & (res.contrast == lab)].iloc[0]
        c = INK if r.delta > 0 else WARN
        ax.plot([r.d_lo, r.d_hi], [y, y], color=c, lw=1.2,
                solid_capstyle="round", zorder=2)
        ax.plot([r.delta], [y], "o", ms=4.4, color=c, mec="white", mew=0.6,
                zorder=3)
        ax.text(-6.6, y, f"{MSHORT[m]}  {lab}", fontsize=6.5, va="center",
                ha="left", color=INK)
        y += 1
    y += 0.5
ax.axvline(0, color=INK, lw=0.8, zorder=1)
ax.set_ylim(y - 0.5, -0.8)
ax.set_yticks([])
ax.set_xlim(-6.8, 8)
ax.set_xlabel("Reduction in the gap under coverage control (pp)", labelpad=3)
finish(ax)
ax.spines["left"].set_visible(False)
ax.text(4.2, -0.5, "gap shrinks →", fontsize=6.2, color=MUTED, ha="center")
head(ax, 2, "The change itself, with a difference-in-differences interval",
     x=-0.30)
save(fig, "fig8_opt2_delta")

# =====================================================================
# 3 · Share of the gap remaining
# =====================================================================
fig, ax = plt.subplots(figsize=(7.09, 2.7))
plt.subplots_adjust(left=0.30, right=0.975, bottom=0.165, top=0.83)
y = 0
for m in MODELS:
    for _, _, lab in CONTRASTS:
        r = res[(res.model == m) & (res.contrast == lab)].iloc[0]
        c = CCOL[lab]
        ax.barh(y, 100, 0.62, color="#F0EFEC", zorder=2)
        ax.barh(y, r.share * 100, 0.62, color=c, zorder=3)
        ax.text(-2, y, f"{MSHORT[m]}  {lab}", fontsize=6.5, va="center",
                ha="right", color=INK)
        ax.text(max(r.share * 100, 0) + 2, y, f"{r.share:.0%} remains",
                fontsize=6.2, va="center", color=MUTED)
        y += 1
    y += 0.5
ax.axvline(100, color=FAINT, lw=0.7, ls=(0, (2, 2)), zorder=1)
ax.set_ylim(y - 0.5, -0.8)
ax.set_yticks([])
ax.set_xlim(0, 132)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of the original gap surviving coverage control (%)",
              labelpad=3)
finish(ax)
ax.spines["left"].set_visible(False)
head(ax, 3, "What fraction of each gap survives", x=-0.30)
save(fig, "fig8_opt3_share")

# =====================================================================
# 4 · Diagonal scatter — all nine on one panel
# =====================================================================
fig, ax = plt.subplots(figsize=(3.6, 3.5))
plt.subplots_adjust(left=0.17, right=0.97, bottom=0.155, top=0.83)
lim = 14
ax.plot([-1, lim], [-1, lim], color=FAINT, lw=0.8, zorder=1)
ax.fill_between([-1, lim], [-1, lim], -1, color="#F5F5F3", zorder=0)
for _, r in res.iterrows():
    mk = {"GPT-4.1-mini": "o", "Gemini-3.1-Flash-Lite": "s",
          "Llama-3.3-70B": "^"}[r.model]
    ax.scatter(r.gap_all, r.gap_common, s=34, marker=mk, color=CCOL[r.contrast],
               edgecolor="white", lw=0.6, zorder=3)
ax.text(11.5, 2.0, "gap shrinks", fontsize=6.3, color=MUTED, ha="center")
ax.text(3.2, 11.0, "gap grows", fontsize=6.3, color=MUTED, ha="center")
ax.set_xlim(-1, lim)
ax.set_ylim(-1, lim)
ax.set_xlabel(f"Gap over all {N_ALL} pairs (pp)", labelpad=3)
ax.set_ylabel(f"Gap over the {N_COMMON} common pairs (pp)", labelpad=3)
finish(ax, grid="both")
ax.legend(handles=[plt.Line2D([], [], marker=k, ls="none", ms=4.4,
                              color=MUTED, label=v)
                   for k, v in [("o", "GPT"), ("s", "Gemini"), ("^", "Llama")]]
                  + [plt.Line2D([], [], marker="o", ls="none", ms=4.4,
                                color=c, label=k) for k, c in CCOL.items()],
          frameon=False, fontsize=5.8, loc="upper left", labelspacing=0.25,
          handletextpad=0.3)
head(ax, 4, "All nine contrasts against y = x", x=-0.22)
save(fig, "fig8_opt4_diagonal")

# =====================================================================
# 5 · Decomposition bars — removed vs residual
# =====================================================================
fig, ax = plt.subplots(figsize=(7.09, 2.7))
plt.subplots_adjust(left=0.30, right=0.975, bottom=0.185, top=0.83)
y = 0
for m in MODELS:
    for _, _, lab in CONTRASTS:
        r = res[(res.model == m) & (res.contrast == lab)].iloc[0]
        resid, removed = r.gap_common, r.delta
        ax.barh(y, resid, 0.62, color=CCOL[lab], zorder=3)
        if removed > 0:
            ax.barh(y, removed, 0.62, left=resid, color="#E4E2DC",
                    edgecolor="white", lw=0.5, zorder=3)
        else:
            ax.barh(y, -removed, 0.62, left=resid + removed, color="none",
                    edgecolor=WARN, lw=0.8, hatch="////", zorder=4)
        ax.text(-0.4, y, f"{MSHORT[m]}  {lab}", fontsize=6.5, va="center",
                ha="right", color=INK)
        y += 1
    y += 0.5
ax.set_ylim(y - 0.5, -0.8)
ax.set_yticks([])
ax.set_xlim(0, 14)
ax.set_xlabel("Gap in rank-1 rate (pp)", labelpad=3)
finish(ax)
ax.spines["left"].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=DARK, ec="none",
                                 label="residual, survives the control"),
                   plt.Rectangle((0, 0), 1, 1, fc="#E4E2DC", ec="white",
                                 label="removed by the control"),
                   plt.Rectangle((0, 0), 1, 1, fc="none", ec=WARN, hatch="////",
                                 label="gap grew")],
          frameon=False, fontsize=6, loc="lower right", handlelength=1.2,
          handleheight=0.9)
head(ax, 5, "Each gap split into removed and residual", x=-0.30)
save(fig, "fig8_opt5_decomposition")

# ---------------------------------------------------------------- audit
print(res[["model", "contrast", "gap_all", "gap_common", "delta",
           "d_lo", "d_hi", "share"]].round(2).to_string(index=False))
print(f"\ndeltas whose interval excludes zero: "
      f"{int(((res.d_lo > 0) | (res.d_hi < 0)).sum())} of {len(res)}")
print("wrote 5 options to", OUT)
