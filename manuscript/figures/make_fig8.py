#!/usr/bin/env python3
"""
Figure 8 | Share of each between-graph gap removed by holding evidence
availability constant.

One signed quantity per row: (gap over all 116 pairs - gap over the 72 pairs
all three graphs contain) / gap over all 116 pairs. Zero means coverage
explains none of the gap, one means it explains all of it, and negative means
the gap widened under the control. A single encoding throughout, no stacking,
no hatching, and uncertainty carried by a bootstrap interval.

Three things this fixes relative to the stacked version:
  - dot-and-interval rather than bars of summaries, so the display is not a
    bar graph obliged to show every underlying point
  - one meaning per x-position on every row
  - colour no longer encodes anything undefined; sign is read off zero

Intervals that leave the axis are drawn as arrows at the boundary. Several
do. The share is a ratio, and for BioKG - DRKG the denominator is 1.6-3.4 pp
and not distinguishable from zero, so the ratio is not meaningfully bounded;
those rows are marked and should be read as uninformative rather than as
evidence of a large negative share.

Run:  python manuscript/figures/make_fig8.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
from plotting import setup_manuscript_style, KG_PALETTE  # noqa: E402

setup_manuscript_style()
OUT = Path(__file__).resolve().parent
RUNS = BASE / "results" / "tables" / "09_llm_runs"
DERIVED = BASE / "results" / "tables" / "09_derived"
GOLD = BASE / "data" / "gold_standards"

INK, MUTED, FAINT = "#1A1A1A", "#8C8C8C", "#C9C9C9"
DOT = KG_PALETTE["primekg"]
RNG = np.random.default_rng(0)
N_BOOT = 10_000
XLO, XHI = -0.55, 1.05

MODELS = {"GPT-4.1-mini": ["09_big_gpt.csv", "09_big_gpt_s12.csv"],
          "Gemini-3.1-Flash-Lite": ["09_big_glite.csv", "09_big_glite_s12.csv"],
          "Llama-3.3-70B": ["09_big_llama.csv", "09_big_llama_s1.csv",
                            "09_big_llama_s2.csv"]}
CONTRASTS = [("primekg", "drkg", "PrimeKG − DRKG"),
             ("primekg", "biokg", "PrimeKG − BioKG"),
             ("biokg", "drkg", "BioKG − DRKG")]

# ---------------------------------------------------------------- data
cov = pd.read_csv(DERIVED / "pair_coverage.csv")
cov["pair_id"] = cov.index + 1
gold = pd.read_csv(GOLD / "gold_standard_v4_bigset.tsv", sep="\t").reset_index(drop=True)
gold["pair_id"] = gold.index + 1
COMMON = set(gold.merge(cov[["pair_id", "all3"]], on="pair_id")
                 .query("all3").disease_name)
N_ALL, N_COMMON = len(gold), int(cov.all3.sum())

rows = []
for m, files in MODELS.items():
    d = pd.concat([pd.read_csv(RUNS / f) for f in files])
    t = (d.groupby(["disease", "kg"])["rank"]
           .apply(lambda r: (r == 1).mean()).unstack())
    inc = t.index.isin(COMMON)
    for a, b, lab in CONTRASTS:
        diff = (t[a] - t[b]).to_numpy() * 100
        g_all, g_com = diff.mean(), diff[inc].mean()
        idx = RNG.integers(0, len(diff), size=(N_BOOT, len(diff)))
        s, si = diff[idx], inc[idx]
        keep = si.sum(axis=1) >= 5
        ba = s[keep].mean(axis=1)
        bc = np.array([r[k].mean() for r, k in zip(s[keep], si[keep])])
        ok = np.abs(ba) > 1e-9
        share_boot = (ba[ok] - bc[ok]) / ba[ok]
        # the interval on the underlying gap tells us whether the ratio is
        # anchored at all: if the denominator can be zero, it is not
        gap_lo, gap_hi = np.percentile(ba, [2.5, 97.5])
        rows.append(dict(
            model=m, contrast=lab, gap_all=g_all, gap_common=g_com,
            share=(g_all - g_com) / g_all,
            lo=np.percentile(share_boot, 2.5),
            hi=np.percentile(share_boot, 97.5),
            gap_established=bool(gap_lo > 0 or gap_hi < 0)))
res = pd.DataFrame(rows)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.09, 3.35))
plt.subplots_adjust(left=0.335, right=0.965, bottom=0.225, top=0.845)

ax.axvspan(XLO, 0, color="#F7F5F2", zorder=0, lw=0)

y, heads = 0.0, []
for m in MODELS:
    heads.append((y - 0.80, m))
    for _, _, lab in CONTRASTS:
        r = res[(res.model == m) & (res.contrast == lab)].iloc[0]
        est = r.gap_established
        colr = DOT if est else MUTED

        lo, hi = max(r.lo, XLO), min(r.hi, XHI)
        ax.plot([lo, hi], [y, y], color=colr, lw=1.1,
                solid_capstyle="butt", zorder=3)
        for bound, edge, direction in ((r.lo, XLO, -1), (r.hi, XHI, 1)):
            if (direction < 0 and bound < XLO) or (direction > 0 and bound > XHI):
                ax.add_patch(FancyArrowPatch(
                    (edge + direction * 0.012, y), (edge + direction * 0.055, y),
                    arrowstyle="-|>", mutation_scale=5, lw=1.1, color=colr,
                    shrinkA=0, shrinkB=0, zorder=3))
            else:
                ax.plot([bound, bound], [y - 0.13, y + 0.13], color=colr,
                        lw=1.1, zorder=3)
        ax.plot([r.share], [y], marker="o", ms=4.6, zorder=4,
                mfc=colr if est else "white", mec=colr, mew=1.0)

        ax.text(XLO - 0.045, y, lab, fontsize=6.6, va="center", ha="right",
                color=INK if est else MUTED)
        ax.text(XHI + 0.055, y, f"{r.gap_all:.1f} pp", fontsize=6.1,
                va="center", ha="left", color=MUTED)
        if not est:
            ax.text(XHI + 0.185, y, "gap not established", fontsize=5.9,
                    va="center", ha="left", color=MUTED, style="italic")
        y += 1.02
    y += 0.66

ax.axvline(0, color=INK, lw=0.8, zorder=2)
ax.axvline(1, color=FAINT, lw=0.7, ls=(0, (2, 2)), zorder=1)
ax.set_ylim(y - 0.66 - 0.42, -1.42)
ax.set_yticks([])
ax.set_xlim(XLO, XHI)
ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["−50", "−25", "0", "25", "50", "75", "100"])
ax.set_xlabel("Share of the between-graph gap removed by holding coverage "
              "constant (%)", labelpad=3)
ax.grid(False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
for yh, m in heads:
    ax.text(XLO - 0.30, yh, m, fontsize=6.9, fontweight="bold", color=INK,
            va="center", ha="left")

ax.text(-0.06, -1.05, "gap widened", fontsize=6.0, color=MUTED, ha="right")
ax.text(0.06, -1.05, "gap narrowed", fontsize=6.0, color=MUTED, ha="left")

ax.text(-0.335, 1.11, "b", transform=ax.transAxes, fontsize=8.5,
        fontweight="bold", va="top", ha="left", color=INK)
ax.text(-0.295, 1.105,
        "Coverage accounts for a minority of each gap, imprecisely estimated",
        transform=ax.transAxes, fontsize=7.2, va="top", ha="left", color=INK)
ax.annotate(f"Points, share removed; bars, 95% bootstrap interval over the "
            f"{N_ALL} pairs; arrows, interval continues beyond the axis.\n"
            "Right-hand figures give the gap over all pairs. Open points mark "
            "contrasts whose underlying gap is not\ndistinguishable from zero, "
            "for which the share is a ratio to an unresolved denominator.",
            xy=(0.5, -0.20), xycoords="axes fraction", ha="center", va="top",
            fontsize=6.0, color=MUTED, linespacing=1.55)

fig.savefig(OUT / "Figure8.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Figure8.png", bbox_inches="tight", dpi=600, facecolor="white")

# ---------------------------------------------------------------- audit
print(res[["model", "contrast", "gap_all", "share", "lo", "hi",
           "gap_established"]].round(2).to_string(index=False))
est = res[res.gap_established]
print(f"\ncontrasts with an established gap: {len(est)} of {len(res)}")
print(f"their share removed: {est.share.min():.0%} to {est.share.max():.0%}, "
      f"mean {est.share.mean():.0%}")
print(f"shares whose interval excludes zero: "
      f"{int(((est.lo > 0) | (est.hi < 0)).sum())} of {len(est)}")
