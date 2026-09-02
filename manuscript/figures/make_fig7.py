#!/usr/bin/env python3
"""
Figure 7 | Rank composition across all 116 pairs.

Where the approved drug lands in a pool of eight.

Uncertainty is reported on the paired differences beneath each panel, not as
marginal intervals on each bar. Every arm sees the same 116 pairs, so the
comparison is paired and between-pair variance cancels; marginal intervals
overlap heavily even where the paired contrast is unambiguous (GPT PrimeKG vs
DRKG: marginal [57, 72] and [45, 61], paired +11.5 pp [4.1, 19.0]). Drawing
them would invite a reader to perform an overlap test that does not apply.

The coverage-controlled comparison is reported in the text rather than as a
figure. Restricting to the 72 pairs all three graphs contain shrinks every
PrimeKG gap, but the share removed is estimated to roughly +/-100 percentage
points at this sample size, so a figure would imply a precision the data do
not support.

Run:  python manuscript/figures/make_fig7.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_BOOT = 10_000

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
from plotting import setup_manuscript_style, GRID_COLOR  # noqa: E402

setup_manuscript_style()
OUT = Path(__file__).resolve().parent
RUNS = BASE / "results" / "tables" / "09_llm_runs"
DERIVED = BASE / "results" / "tables" / "09_derived"
GOLD = BASE / "data" / "gold_standards"

INK, MUTED, FAINT = "#1A1A1A", "#8C8C8C", "#C9C9C9"
ARMS = [("none", "No KG"), ("primekg", "PrimeKG"),
        ("drkg", "DRKG"), ("biokg", "BioKG")]
BANDS = [(1, 1, "1"), (2, 3, "2–3"), (4, 5, "4–5"), (6, 9, "6+")]
SHADE = ["#2E6B8A", "#84A9BF", "#C6D6DF", "#ECEBE7"]
MODELS = {"GPT-4.1-mini": ["09_big_gpt.csv", "09_big_gpt_s12.csv"],
          "Gemini-3.1-Flash-Lite": ["09_big_glite.csv", "09_big_glite_s12.csv"],
          "Llama-3.3-70B": ["09_big_llama.csv", "09_big_llama_s1.csv",
                            "09_big_llama_s2.csv"]}
SHORT = {"GPT-4.1-mini": "GPT-4.1-mini",
         "Gemini-3.1-Flash-Lite": "Gemini-3.1-Flash-Lite",
         "Llama-3.3-70B": "Llama-3.3-70B"}

# ---------------------------------------------------------------- data
cov = pd.read_csv(DERIVED / "pair_coverage.csv")
cov["pair_id"] = cov.index + 1
gold = pd.read_csv(GOLD / "gold_standard_v4_bigset.tsv", sep="\t")
gold = gold.reset_index(drop=True)
gold["pair_id"] = gold.index + 1
common = set(gold.merge(cov[["pair_id", "all3"]], on="pair_id")
                 .query("all3").disease_name)

runs = {}
for m, files in MODELS.items():
    runs[m] = pd.concat([pd.read_csv(RUNS / f) for f in files])

N_ALL = len(gold)
N_COMMON = int(cov.all3.sum())

# ---------------------------------------------------------------- canvas
fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.10), sharey=True,
                         squeeze=False)
plt.subplots_adjust(left=0.095, right=0.985, bottom=0.345, top=0.80,
                    wspace=0.12)

ROWS = [(f"All {N_ALL} pairs", None)]

for r, (row_title, keep) in enumerate(ROWS):
    for c, m in enumerate(MODELS):
        ax = axes[r, c]
        d = runs[m] if keep is None else runs[m][runs[m].disease.isin(keep)]
        for i, (k, lab) in enumerate(ARMS):
            sub = d[d.kg == k]
            bottom = 0.0
            for (lo, hi, _), colr in zip(BANDS, SHADE):
                frac = ((sub["rank"] >= lo) & (sub["rank"] <= hi)).mean()
                ax.bar(i, frac, 0.68, bottom=bottom, color=colr, zorder=3,
                       edgecolor="white", lw=0.6)
                bottom += frac
            top1 = (sub["rank"] == 1).mean()
            ax.text(i, 1.03, f"{top1:.0%}", ha="center", va="bottom",
                    fontsize=6.0, color=INK if k != "none" else MUTED)

        # paired contrasts, which is the uncertainty that applies here
        t1 = (d.groupby(["disease", "kg"])["rank"]
                .apply(lambda r: (r == 1).mean()).unstack())
        lines = []
        for a, b, lab in (("primekg", "drkg", "PrimeKG − DRKG"),
                          ("primekg", "biokg", "PrimeKG − BioKG")):
            v = (t1[a] - t1[b]).to_numpy() * 100
            bidx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
            bt = v[bidx].mean(axis=1)
            lo_c, hi_c = np.percentile(bt, [2.5, 97.5])
            star = "" if (lo_c < 0 < hi_c) else "*"
            lines.append(f"{lab}  {v.mean():+.1f} [{lo_c:.1f}, {hi_c:.1f}]{star}")
        ax.annotate("\n".join(lines), xy=(0.5, -0.175),
                    xycoords="axes fraction", ha="center", va="top",
                    fontsize=5.9, color=INK, linespacing=1.5)

        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([lab for _, lab in ARMS], fontsize=6.4)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.62, len(ARMS) - 0.38)
        ax.grid(False)
        ax.tick_params(axis="x", length=0, pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if r == 0:
            ax.set_title(SHORT[m], fontsize=7.2, pad=14)
    axes[r, 0].set_ylabel("Fraction of queries", labelpad=3)

# row labels, set left of the first panel
for r, (row_title, _) in enumerate(ROWS):
    axes[r, 0].text(-0.30, 1.20, "a", transform=axes[r, 0].transAxes,
                    fontsize=8.5, fontweight="bold", va="top", ha="left",
                    color=INK)
    axes[r, 0].text(-0.245, 1.195, row_title, transform=axes[r, 0].transAxes,
                    fontsize=7.2, va="top", ha="left", color=INK)

handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="white", lw=0.6)
           for c in SHADE]
axes[0, 2].legend(handles, [b[2] for b in BANDS], frameon=False, fontsize=6.2,
                  loc="upper center", bbox_to_anchor=(0.5, -0.545), ncol=4,
                  handlelength=1.0, handleheight=0.9, columnspacing=1.0,
                  handletextpad=0.4, title="rank of the approved drug",
                  title_fontsize=6.2)
axes[0, 0].annotate("Paired differences in the rank-1 rate, percentage points, "
                    "with 95% bootstrap intervals over the 116 pairs.\n"
                    "Asterisks mark intervals excluding zero.",
                    xy=(1.75, -0.42), xycoords="axes fraction", ha="center",
                    va="top", fontsize=5.9, color=MUTED, linespacing=1.5)

fig.savefig(OUT / "Figure7.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Figure7.png", bbox_inches="tight", dpi=600,
            facecolor="white")

# ---------------------------------------------------------------- audit
print(f"{'set':10s}{'model':24s}" + "".join(f"{lab:>10s}" for _, lab in ARMS))
for label, keep in [("all", None), ("common", common)]:
    for m in MODELS:
        d = runs[m] if keep is None else runs[m][runs[m].disease.isin(keep)]
        row = "".join(f"{(d[d.kg == k]['rank'] == 1).mean():10.1%}"
                      for k, _ in ARMS)
        print(f"{label:10s}{m:24s}{row}")
_base = runs["GPT-4.1-mini"]
_none = _base[_base.kg == "none"]
print(f"\nresponses per arm: all {len(_none)}, "
      f"common {len(_none[_none.disease.isin(common)])}")
print(f"pairs: all {N_ALL}, common {N_COMMON}")
