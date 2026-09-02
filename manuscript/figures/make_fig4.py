#!/usr/bin/env python3
"""
Figure 4 for the BioKGSuite Nature Communications manuscript.

Fig. 4 | Task performance depends on the task, the graph's structure, and
the density of the entities queried.

  a  Rank reversal between drug-disease link prediction (AUROC) and
     disease-to-gene retrieval (Recall@100).
  b  Two-hop reachability against link-prediction AUROC — reachability
     screens out a failing graph but does not order the survivors.
  c  Aggregate link-prediction AUROC against AUROC restricted to the
     least-annotated quartile of diseases (Q1).

Every value is read from results/checkpoints/*.pkl, so the figure cannot
drift from the benchmark. Nothing is hardcoded except axis limits.

Run:  python manuscript/figures/make_fig4.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.plotting import (setup_manuscript_style, save_fig, panel_label,
                          KG_PALETTE, TEXT_COLOR, TICK_COLOR, GRID_COLOR)

CKPT = BASE / "results" / "checkpoints"
OUT = Path(__file__).resolve().parent

setup_manuscript_style()

# ---------------------------------------------------------------- constants
KGS = ["primekg", "hetionet", "drkg", "openbilink", "biokg", "matrix"]
LABEL = {
    "primekg": "PrimeKG", "hetionet": "Hetionet", "drkg": "DRKG",
    "openbilink": "OpenBioLink", "biokg": "BioKG", "matrix": "MATRIX",
}
# Canonical per-KG colours from src/plotting.py so Fig. 4 matches Figs. 1-3
# and notebooks 00-08. MATRIX has no canonical entry (it was added after the
# palette was fixed); the codebase convention is KG_PALETTE.get(k, grey).
MATRIX_GREY = "#8A8A8A"
COLOR = {k: KG_PALETTE.get(k, MATRIX_GREY) for k in KGS}

MUTED = "#8C8C8C"      # de-emphasised strokes / secondary annotation
FAINT = "#B5B5B5"      # reference lines


# ---------------------------------------------------------------- data
def _load(name):
    with open(CKPT / name, "rb") as fh:
        return pickle.load(fh)


perf = _load("06_predictive_performance.pkl")
topo = _load("04_topology.pkl")
gen = _load("07_generalization.pkl")

lp = {k: perf["sub_scores"][k]["link_prediction"] for k in KGS}
ret = {k: perf["nbhd_scalars"][k]["recall@100"] for k in KGS}
reach = {k: topo["sub_scores"][k]["reachability"] for k in KGS}
sparse = {k: gen["tier_scalars"][k]["Q1: Sparse"]["auroc"] for k in KGS}
n_sparse = {k: int(gen["tier_scalars"][k]["Q1: Sparse"]["n_pos"]) for k in KGS}
# nb07 cell 12 evaluates on sampled = min(500, len(tier_pos)), NOT on every
# positive in the tier, so n_pos overstates the evidence for large tiers.
# Prefer n_eval once nb07 records it; fall back to the cap until then.
TIER_SAMPLE_CAP = 500
n_eval = {k: int(gen["tier_scalars"][k]["Q1: Sparse"].get(
    "n_eval", min(n_sparse[k], TIER_SAMPLE_CAP))) for k in KGS}

# Q1 bootstrap CIs appear once nb07 cell 12 calls bootstrap_auroc_ci.
# Absent today -> whiskers are simply not drawn.
q1_ci = {k: (gen["tier_scalars"][k]["Q1: Sparse"].get("auroc_ci_lo"),
             gen["tier_scalars"][k]["Q1: Sparse"].get("auroc_ci_hi")) for k in KGS}
HAS_Q1_CI = all(v[0] is not None and v[1] is not None for v in q1_ci.values())

rank_lp = {k: r for r, k in enumerate(sorted(KGS, key=lambda x: -lp[x]), 1)}
rank_ret = {k: r for r, k in enumerate(sorted(KGS, key=lambda x: -ret[x]), 1)}
drop = {k: 100 * (lp[k] - sparse[k]) / lp[k] for k in KGS}

# ---------------------------------------------------------------- canvas
# 183 mm double-column (DOUBLE_COL_W). Height set by the 6-row panels.
fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.75))
# Only wspace survives savefig(bbox_inches='tight') — the margin arguments
# are recomputed away by the tight bbox, so they are not set here.
fig.subplots_adjust(wspace=0.62)

for ax in axes:
    ax.grid(False)

# ------------------------------------------------------- (a) rank slopegraph
ax = axes[0]
X0, X1 = 0.40, 0.86

for k in KGS:
    moved = rank_lp[k] != rank_ret[k]
    # Emphasis on the graphs that CHANGE rank — that is the panel's claim.
    ax.plot([X0, X1], [rank_lp[k], rank_ret[k]],
            color=COLOR[k] if moved else FAINT,
            lw=1.6 if moved else 0.9,
            ls="-" if moved else (0, (1.5, 1.5)),
            marker="o", ms=3.6,
            mfc=COLOR[k], mec="white", mew=0.6,
            zorder=3 if moved else 2)
    ax.annotate(f"{LABEL[k]}  {lp[k]:.3f}", (X0, rank_lp[k]),
                xytext=(-5, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6.2, color=COLOR[k])
    ax.annotate(f"{ret[k]:.3f}", (X1, rank_ret[k]),
                xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6.2, color=COLOR[k])

ax.set_xlim(0.0, 1.0)
ax.set_ylim(6.7, 0.3)
ax.set_yticks([])
ax.set_xticks([X0, X1])
ax.set_xticklabels(["Link\nprediction\n(AUROC)", "Gene\nretrieval\n(Recall@100)"])
ax.tick_params(axis="x", length=0, pad=3)
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
# y is rank, and there is no y-axis to say so.
ax.text(-0.19, 0.5, "Rank (1 = best)", transform=ax.transAxes, rotation=90,
        ha="center", va="center", fontsize=6.2, color=MUTED)

# ------------------------------------------- (b) reachability vs LP AUROC
ax = axes[1]
XLIM_B = (0.42, 1.02)

# Above ~0.8 reachability, AUROC spans almost its whole range: the metric
# separates the one failing graph and then stops discriminating.
ax.axvspan(0.80, XLIM_B[1], color="#F2F2F2", zorder=0, lw=0)
ax.annotate("no ordering\nabove 0.8", (0.91, 0.735), ha="center", va="center",
            fontsize=6, color=MUTED, linespacing=1.35)

for k in KGS:
    ax.scatter(reach[k], lp[k], s=34, color=COLOR[k],
               edgecolor="white", lw=0.6, zorder=3)

# Hand-placed to avoid the primekg/hetionet and matrix/biokg collisions.
OFFSETS_B = {                       # (dx, dy) in points, ha
    "primekg":    (7, 3.5, "left"),
    "hetionet":   (7, -4, "left"),
    "drkg":       (-7, 0, "right"),
    "openbilink": (7, 0, "left"),
    "biokg":      (7, -4.5, "left"),
    "matrix":     (-7, 0, "right"),
}
for k in KGS:
    dx, dy, ha = OFFSETS_B[k]
    ax.annotate(LABEL[k], (reach[k], lp[k]), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.2,
                color=COLOR[k], ha=ha, va="center")

ax.set_xlim(*XLIM_B)
ax.set_ylim(0.70, 1.03)
ax.set_xlabel("Two-hop reachability", labelpad=2)
ax.set_ylabel("Link-prediction AUROC", labelpad=2)
ax.grid(True, color=GRID_COLOR, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ------------------------------------------------ (c) all vs sparse dumbbell
ax = axes[2]
order = sorted(KGS, key=lambda k: lp[k])
XLIM_C = (0.50, 1.02)

for i, k in enumerate(order):
    ax.plot([sparse[k], lp[k]], [i, i], color=COLOR[k], lw=1.3, zorder=2,
            solid_capstyle="round")
    ax.scatter([lp[k]], [i], s=28, color=COLOR[k], zorder=3,
               edgecolor="white", lw=0.5)
    if HAS_Q1_CI:
        lo, hi = q1_ci[k]
        ax.plot([lo, hi], [i, i], color=COLOR[k], lw=0.7, alpha=0.85,
                solid_capstyle="butt", zorder=2.5)
        for b in (lo, hi):
            ax.plot([b, b], [i - 0.13, i + 0.13], color=COLOR[k], lw=0.7,
                    zorder=2.5)
    ax.scatter([sparse[k]], [i], s=28, facecolor="white", zorder=3,
               edgecolor=COLOR[k], lw=1.1)
    # Drop labels in a fixed right-hand column (axes fraction on x, data on
    # y) so they align instead of tracking the marker.
    ax.text(1.03, i, f"−{drop[k]:.0f}%", transform=ax.get_yaxis_transform(),
            fontsize=6.2, va="center", ha="left", color=MUTED)

ax.axvline(0.5, color=FAINT, ls=(0, (2, 2)), lw=0.7, zorder=1)
ax.annotate("chance", (0.5, -0.42), xytext=(3, 0),
            textcoords="offset points", fontsize=6, color=MUTED, va="center")

ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{LABEL[k]}\nn = {n_eval[k]:,}" for k in order])
for t, k in zip(ax.get_yticklabels(), order):
    t.set_color(COLOR[k])
ax.set_xlim(*XLIM_C)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Link-prediction AUROC", labelpad=2)
ax.tick_params(axis="y", length=0)
ax.grid(True, axis="x", color=GRID_COLOR, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

h_all = plt.Line2D([], [], marker="o", ls="none", ms=4.2, color=MUTED,
                   label="All diseases")
h_q1 = plt.Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white",
                  mec=MUTED, mew=1.1, color=MUTED, label="Sparse quartile (Q1)")
ax.legend(handles=[h_all, h_q1], loc="upper center", frameon=False,
          bbox_to_anchor=(0.45, -0.20), ncol=2, handletextpad=0.3,
          columnspacing=1.1, borderaxespad=0.0)

# ---------------------------------------------------------- panel furniture
TITLES = ["Ranking depends on the task",
          "Reachability screens, not ranks",
          "Sparse diseases reorder"]
PANEL_X = {"a": -0.30, "b": -0.24, "c": -0.30}
for ax, letter, title in zip(axes, "abc", TITLES):
    panel_label(ax, letter, x=PANEL_X[letter], y=1.22, fontsize=9)
    ax.set_title(title, fontsize=7.2, fontweight="bold", color=TEXT_COLOR,
                 loc="left", pad=6)

save_fig(fig, OUT, "Figure4", dpi=600)

# ---------------------------------------------------------------- audit
print()
print(f"{'KG':<12}{'LP':>9}{'rk':>4}{'R@100':>9}{'rk':>4}{'Reach':>8}"
      f"{'Q1':>9}{'n_pos':>7}{'drop%':>7}")
for k in sorted(KGS, key=lambda x: -lp[x]):
    print(f"{LABEL[k]:<12}{lp[k]:>9.4f}{rank_lp[k]:>4}{ret[k]:>9.4f}"
          f"{rank_ret[k]:>4}{reach[k]:>8.4f}{sparse[k]:>9.4f}"
          f"{n_sparse[k]:>7d}{drop[k]:>7.1f}")

moved = [k for k in KGS if rank_lp[k] != rank_ret[k]]
print(f"\nRank changes between tasks: {len(moved)}/6 "
      f"({', '.join(LABEL[k] for k in moved)})")
print(f"Aggregate AUROC span {max(lp.values()) - min(lp.values()):.4f} | "
      f"Q1 span {max(sparse.values()) - min(sparse.values()):.4f}")
print(f"Degradation {min(drop.values()):.1f}% to {max(drop.values()):.1f}%")
thin = [k for k in KGS if n_eval[k] < 50]
if thin:
    print(f"CAUTION: Q1 AUROC from <50 evaluated positives for "
          f"{', '.join(f'{LABEL[k]} (n={n_eval[k]})' for k in thin)}")
capped = [k for k in KGS if n_sparse[k] > n_eval[k]]
if capped:
    print("NOTE: tier sampling cap applied to "
          + ", ".join(f"{LABEL[k]} ({n_sparse[k]}->{n_eval[k]})" for k in capped))
print(f"Q1 bootstrap CIs in checkpoint: {HAS_Q1_CI}"
      f"{'' if HAS_Q1_CI else '  (re-run nb07 cell 12 to populate)'}")
