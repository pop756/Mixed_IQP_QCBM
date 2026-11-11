#!/usr/bin/env python3
r"""Branch-diversity and mode-separation vs test-MMD^2 scatter, 2 x 4 (figure*,
laid out like fig_train_results). Columns are the four benchmarks (blobs, Ising,
D-Wave, MNIST); every point is one trained model = one (initialization scheme x
training seed).
  top row    (a)-(d): x = D^2_branch = sum_A w_A Var_pi[z_A]  (the exact/surrogate
                      branch-diversity term of Proposition (diversity), Eq. dbranch)
  bottom row (e)-(h): x = Lambda, the branch necessity (Eq. necessity): the pi-weighted geometric-mean factor by
                      which the test MMD^2 grows when one branch is removed, log axis, gray line at Lambda = 1
  y (both rows, shared per column) = held-out test MMD^2 of the full mixture,
both at the panel's headline bandwidth sigma. The diversity-floor relation
L_mix = sum_l pi_l L_l - D^2_branch means a larger D^2_branch (branches spread apart)
buys a lower mixture MMD^2; Lambda additionally asks whether the target needs each branch. One point per training seed makes the seed spread and the scheme
separation directly visible.

Data (read-only): experiments/cosine_rerun/results/branch_diversity.csv (D2, mmd2), branch_necessity.csv (Lambda),
diversity_refs.csv (bandwidth, train-test sampling reference), all from analyze.diversity().
Run:  python make_fig_diversity_scatter.py  -> fig_diversity_scatter.pdf
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()

HERE = Path(__file__).resolve().parent
# the cosine rerun: results/branch_diversity.csv, mode_index.csv, diversity_refs.csv (experiments/cosine_rerun)
from _seed_agg import RESULTS, STEM_OF, scatter_rows  # noqa: E402
OUT = HERE / "fig_diversity_scatter.pdf"

# panel order + display + n (matches the four result sections)
PANELS = [("blobs", "blobs", 16), ("ising", "Ising", 16),
          ("dwave", "D-Wave", 484), ("mnist", "MNIST", 784)]

# scheme cache-key (s*_ prefix in the CSV) -> (display, colour, marker)
SCHEME = {
    "s1_visible":  ("global",       "#666666", "o"),
    "s4_unbiased": ("data-agnostic", "#1f77b4", "D"),
    "s2_cluster":  ("cluster-initialized",  "#2ca02c", "^"),
    "s2_label":    ("label-routed", "#d62728", "s"),
}
ORDER = ["s1_visible", "s4_unbiased", "s2_cluster"]

FS = 8.5
TAGS = "abcdefgh"


def load_scatter():
    """{dataset: {scheme: [(D2, mmd2, Lambda), ...]}}, {dataset: sigma}, {dataset: L}."""
    pts, sig, L = defaultdict(lambda: defaultdict(list)), {}, {}
    for r in csv.DictReader(open(RESULTS / "diversity_refs.csv")):
        sig[r["dataset"]], L[r["dataset"]] = float(r["sigma"]), 2 ** int(r["a"])
    for ds, _disp, _n in PANELS:
        for init, _seed, d2, m2, nm in scatter_rows(ds):
            pts[ds][STEM_OF[init]].append((d2, m2, nm))
    return pts, sig, L


def floor_at(ds, sigma):
    """Train-test sampling reference with the same weights / subsets as the plotted test MMD^2."""
    for r in csv.DictReader(open(RESULTS / "diversity_refs.csv")):
        if r["dataset"] == ds and abs(float(r["sigma"]) - sigma) < 1e-3:
            return float(r["ref_train_vs_test"])
    return None


def print_trend_stats(pts, col=0, name="D2", n_boot=10_000, seed=12345):
    """Spearman rho (x = column `col` vs test MMD^2) + percentile-bootstrap 95% CI
    per panel (quoted in the text)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    for ds, _disp, _n in PANELS:
        xy = np.array([p for sc in pts.get(ds, {}).values() for p in sc])
        rho = spearmanr(xy[:, col], xy[:, 1]).statistic
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(xy), len(xy))
            b = spearmanr(xy[idx, col], xy[idx, 1]).statistic
            if not np.isnan(b):
                boots.append(b)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"[trend {name}] {ds:6s} n={len(xy):2d}  rho={rho:+.2f}  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]")


def main():
    pts, sig, L = load_scatter()
    print_trend_stats(pts, 0, "D2")
    print_trend_stats(pts, 2, "Lambda")
    fig, axes = plt.subplots(2, 4, figsize=(7.057, 3.65), sharey="col")
    handles = {}
    for row, (col, xlabel) in enumerate(((0, r"$D^{2}_{\mathrm{branch}}$"),
                                         (2, r"branch necessity $\Lambda$"))):
        for pi, (ax, (ds, disp, n)) in enumerate(zip(axes[row], PANELS)):
            d = pts.get(ds, {})
            for sc in ORDER:
                if sc not in d:
                    continue
                label, colour, marker = SCHEME[sc]
                xy = np.asarray(d[sc], dtype=float)
                # routed schemes (cluster/label) often coincide at the floor; draw
                # them hollow + on top so an overlap reads as two rings, not a blob.
                routed = sc in ("s2_cluster", "s2_label")
                h = ax.scatter(xy[:, col], xy[:, 1], s=15 if routed else 12,
                               facecolors="none" if routed else colour,
                               edgecolors=colour,
                               linewidths=0.9 if routed else 0.4, marker=marker,
                               alpha=0.95, zorder=4 if routed else 3, label=label)
                handles[sc] = h
            fl = floor_at(ds, sig.get(ds, np.nan))
            if fl is not None and fl > 0:
                handles["floor"] = ax.axhline(fl, color="black", ls=":",
                                              lw=1.0, zorder=1,
                                              label="train–test ref.")
            ax.set_yscale("log")
            head = rf"$\bf{{({TAGS[4 * row + pi]})}}$" + (f"  {disp}" if row == 0 else "")
            ax.set_title(head, fontsize=FS, pad=3, loc="left")
            ax.tick_params(labelsize=FS - 1, length=2.6, width=0.6, pad=2)
            ax.tick_params(which="minor", length=1.5, width=0.5)
            ax.grid(alpha=0.25, which="both", lw=0.5)
            ax.set_xlabel(xlabel, fontsize=FS, labelpad=1.5)
            xs = [p[col] for sc in d.values() for p in sc]
            if row == 0:
                # every panel starts at zero diversity; the upper end follows its data
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
                ax.set_xlim(0.0, max(xs) * 1.12)
            else:
                ax.set_xscale("log")
                ax.axvline(1.0, color="0.6", lw=0.8, zorder=1)
                ax.set_xlim(0.8, max(xs) * 1.6)
                top = max(xs) * 1.6
                ticks = [v for v in ((1, 2, 3) if top < 5 else (1, 2, 5, 10) if top < 20 else (1, 10, 100))
                         if v <= top]
                ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
                ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(v) for v in ticks]))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[row, 0].set_ylabel(r"test MMD$^2$", fontsize=FS, labelpad=1.5)

    # keep the vertical axis on labelled decades: a panel whose data span less
    # than one decade (the Ising) otherwise falls back to 2x, 3x, ... (y is
    # shared per column, so fixing the top row fixes the bottom one)
    for ax in axes[0]:
        lo, hi = ax.get_ylim()
        interior = [k for k in range(int(np.floor(np.log10(lo))),
                                     int(np.ceil(np.log10(hi))) + 1)
                    if lo < 10.0 ** k < hi]
        if len(interior) < 2:
            ax.set_ylim(10.0 ** np.floor(np.log10(lo)),
                        10.0 ** np.ceil(np.log10(hi)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # one shared legend under the grid
    order = [k for k in ORDER if k in handles] + (["floor"] if "floor" in handles else [])
    labels = [SCHEME[k][0] if k in SCHEME else "train–test ref." for k in order]
    fig.legend([handles[k] for k in order], labels, frameon=False,
               fontsize=FS - 0.5, loc="lower center", ncol=len(order),
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4,
               columnspacing=1.3, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.065, 1, 1), w_pad=0.9, h_pad=1.1)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
