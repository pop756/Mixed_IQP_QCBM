#!/usr/bin/env python3
r"""Fig. modesep: mode separation across three targets in ONE full-width figure.

Replaces the former fig_branches + fig_ising_branch + fig_mnist_branches, which
made the same point (branches specialize onto the target's modes, and only the
cluster-initialized start does so reliably) on three targets in three figures.

  (a) binary blobs (n=16, a=3 -> L=8): 3 rows (init scheme) x 8 branch columns,
      each cell the per-bit mean of PER_IMG exact-IQP samples of one branch as a
      4x4 grayscale image (dark = 1). Row labels carry the seed-averaged mode
      coverage, so the picture and the number are read together.
  (b) 2D Ising (n=16, a=3 -> L=8), cluster-initialized: per-branch EXACT energy
      (top) and magnetization (bottom) densities, branches sorted by <M>, with
      the train data underlaid in gray.
  (c) MNIST (n=784, a=4 -> L=16): one fixed branch (bits "0000"), its one-body
      marginal image at initialization (top) and after training (bottom), for
      the three initializations.

REPRESENTATIVE SEED (important). The former fig_branches plotted seed 1234,
which is the ONE seed of {7,1234,2025} at which the data-agnostic start happens
to reach full 8/8 mode coverage (the other two give 6/8), and at which the
global start reaches its own best (7/8). That panel therefore showed both
baselines at their most favorable draw. This figure shows EVERY scheme at its
own WORST draw: for each (dataset, scheme) the training seed with the smallest
branch-diversity term D^2_branch in branch_diversity_scatter.csv (the Fig.-4
statistic). The per-seed coverage table is still printed so the choice stays
auditable. The Ising a=2 cache (prep_modesep_ising_a2.py) already stores the
same per-scheme lowest-diversity selection.

Run:  python figures/make_fig_modesep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()

from _seed_agg import RESULTS, lowest_d2_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
# MODESEP_FULL=1 renders every branch in the two-row appendix layout;
# by default the main-text figure shows the first NSHOW branches in one row.
FULL = os.environ.get("MODESEP_FULL", "0") == "1"
NSHOW = int(os.environ.get("MODESEP_NSHOW", "4"))
OUT = HERE / ("fig_modesep_full.pdf" if FULL else "fig_modesep.pdf")

# Data: exact caches of the cosine rerun (experiments/cosine_rerun/caches.py, analyze.py). Per-scheme
# representative seed: the seed with the SMALLEST branch diversity D^2 of that (dataset, scheme) in
# results/branch_diversity.csv (the Fig.-4 statistic, worst draw).
INIT_OF_STEM = {"s1_visible": "global", "s4_unbiased": "agnostic", "s2_cluster": "cluster"}


def worst_seed(dataset: str, stem: str) -> int:
    return lowest_d2_seed(dataset, INIT_OF_STEM[stem])


A = 3
L = 2 ** A                       # 8 branches
SIDE = 4
NI = 16                          # Ising spins (magnetization axis -NI..NI)
GREEN = "#2ca02c"
ROW_COLORS = ["#666666", "#1f77b4", "#2ca02c"]   # global / agnostic / cluster

SCHEMES = [
    ("global", "global", "s1_visible"),
    ("agnostic", "agnostic", "s4_unbiased"),
    ("cluster-initialized", "cluster-\ninitialized", "s2_cluster"),
]


# ---------------------------------------------------------------- panel (a)
def blobs_panel():
    """Per-scheme exact branch-mean images at the lowest-D2 seed + seed statistics of D-bar (mean pairwise branch
    MMD^2 at m-bar = 2) and blob-peak coverage over all ten seeds."""
    z = np.load(RESULTS / "modesep_blobs_a3.npz")
    for k, (name, _lab, _stem) in enumerate(SCHEMES):
        d, c = z["dbar"][k], z["coverage"][k]
        print(f"[modesep-a] {name:20s} Dbar={d.mean():.3f}+-{d.std(ddof=1):.3f}  "
              f"coverage={c.mean():.2f}+-{c.std(ddof=1):.2f}  per-seed={list(map(int, c))}"
              f"  (seeds {list(z['seeds'])}; plotted lowest-D2 seed {int(z['rep_seed'][k])})")
    return list(z["rep_means"]), (z["dbar"], z["coverage"])


# ---------------------------------------------------------------- panel (b)
def ising_panel():
    """All three schemes' exact per-branch magnetization densities: a = 2 at each scheme's lowest-D-bar seed
    (main text), a = 3 at seed 7 (appendix layout)."""
    if not FULL:
        z = np.load(RESULTS / "modesep_ising_a2.npz")
        print("[modesep-b] Ising a=2, per-scheme lowest-D-bar seeds "
              f"{list(np.atleast_1d(z['seed']))}")
        return dict(hists=z["hists"], means=z["means"],
                    mbins=z["mbins"], data=z["data"])
    z = np.load(RESULTS / "modesep_ext_cache.npz")
    return dict(hists=z["ising_hists"], means=z["ising_means"],
                mbins=z["ising_mbins"], data=z["ising_data"])


# ---------------------------------------------------------------- panel (c)
MNIST_COLS = 16                  # all branches of the a=4 model


def mnist_panel():
    """Per-branch one-body marginal images of the trained MNIST a=4 models,
    each scheme at its own lowest-diversity seed."""
    z = np.load(RESULTS / "mnist_branch_marginals.npz")
    rows = []
    for _, _, stem in SCHEMES:
        init = INIT_OF_STEM[stem]
        seed = worst_seed("mnist", stem)
        rows.append(np.asarray(z[f"{init}_seed{seed}"])[:MNIST_COLS])
        print(f"[modesep-c] mnist {stem}: lowest-D2 seed {seed}")
    return np.stack(rows), 28


# ---------------------------------------------------------------- panel (d)
def dwave_panel():
    """Per-branch overlap mean and width (exact closed form, D-Wave a=3, seed 7)."""
    z = np.load(RESULTS / "dwave_overlap.npz")
    return dict(q=[z["q_global"], z["q_agnostic"], z["q_cluster_initialized"]],
                s=[z["s_global"], z["s_agnostic"], z["s_cluster_initialized"]],
                cluster=z["cluster_qbar"], full=float(z["full_qbar"]))


# ---------------------------------------------------------------- figure
def main_compact(rep_means, isg, mmeans, mside):
    """Main-text figure: the first NSHOW branches of each dataset, three
    panels in a single row."""
    nb = min(NSHOW, L)
    # branches evenly spaced across the index range rather than the first few
    cols_s = np.linspace(0, L - 1, nb).round().astype(int)
    n_ising = isg["hists"].shape[1]
    cols_i = np.linspace(0, n_ising - 1, min(nb, n_ising)).round().astype(int)
    cols_m = np.linspace(0, MNIST_COLS - 1, nb).round().astype(int)
    fig = plt.figure(figsize=(7.057, 2.05))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.20,
                             left=0.115, right=0.99, top=0.80, bottom=0.03)

    ga = outer[0].subgridspec(len(SCHEMES), nb, wspace=0.10, hspace=0.16)
    blob_axes = [[] for _ in SCHEMES]
    for r in range(len(SCHEMES)):
        for c, b in enumerate(cols_s):
            ax = fig.add_subplot(ga[r, c])
            ax.imshow(rep_means[r][b].reshape(SIDE, SIDE), cmap="gray_r",
                      vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title(rf"$\ell={b}$", fontsize=6, pad=1.5)
            blob_axes[r].append(ax)

    gb = outer[1].subgridspec(len(SCHEMES), nb, wspace=0.10, hspace=0.16)
    MC = 0.5 * (isg["mbins"][:-1] + isg["mbins"][1:])
    ymax = max(float(isg["data"].max()), float(isg["hists"].max())) * 1.1
    ising_axes = [[] for _ in SCHEMES]
    for r in range(len(SCHEMES)):
        for c, b in enumerate(cols_i):
            ax = fig.add_subplot(gb[r, c])
            ax.fill_between(MC, isg["data"], step="mid", color="0.85", lw=0)
            ax.fill_between(MC, isg["hists"][r, b], step="mid",
                            color=ROW_COLORS[r], alpha=0.9, lw=0)
            ax.set_ylim(0, ymax); ax.set_xlim(-NI, NI)
            ax.set_yticks([])
            if r == len(SCHEMES) - 1 and c == 0:
                ax.set_xticks([-NI, 0, NI])
                ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            else:
                ax.set_xticks([])
            if r == 0:
                ax.set_title(rf"$\ell={b}$", fontsize=6, pad=1.5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.6")
            ising_axes[r].append(ax)

    gc = outer[2].subgridspec(len(SCHEMES), nb, wspace=0.06, hspace=0.16)
    mnist_axes = [[] for _ in SCHEMES]
    for r in range(len(SCHEMES)):
        for c, b in enumerate(cols_m):
            ax = fig.add_subplot(gc[r, c])
            ax.imshow(mmeans[r, b].reshape(mside, mside), cmap="gray_r",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.7")
            if r == 0:
                ax.set_title(rf"$\ell={b}$", fontsize=6, pad=1.5)
            mnist_axes[r].append(ax)

    # square the image cells, then match the Ising rows to them
    fig.canvas.draw()
    for r in range(len(SCHEMES)):
        for ax in blob_axes[r] + mnist_axes[r]:
            ax.apply_aspect()
        y0 = min(a.get_position(original=False).y0 for a in blob_axes[r])
        y1 = max(a.get_position(original=False).y1 for a in blob_axes[r])
        for ax in ising_axes[r]:
            b = ax.get_position()
            ax.set_position([b.x0, y0, b.width, y1 - y0])

    for r, (_, rowlab, _) in enumerate(SCHEMES):
        b = blob_axes[r][0].get_position(original=False)
        fig.text(0.108, 0.5 * (b.y0 + b.y1), rowlab, fontsize=7,
                 ha="right", va="center")
    for tag, axes_of in (("(a) blobs", blob_axes[0][0]),
                         ("(b) Ising", ising_axes[0][0]),
                         ("(c) MNIST", mnist_axes[0][0])):
        b = axes_of.get_position(original=False)
        fig.text(b.x0, b.y1 + 0.075, tag, fontsize=8, fontweight="bold")
    bi = ising_axes[-1][0].get_position()
    be = ising_axes[-1][-1].get_position()
    fig.text(0.5 * (bi.x0 + be.x1), bi.y0 - 0.035,
             r"magnetization $M$", ha="center", va="top", fontsize=6.5)

    fig.savefig(OUT)
    fig.savefig(HERE / "fig_modesep_preview.png", dpi=200)
    print("wrote", OUT)


def main():
    rep_means, cov = blobs_panel()
    isg = ising_panel()
    mmeans, mside = mnist_panel()
    if not FULL:
        return main_compact(rep_means, isg, mmeans, mside)

    fig = plt.figure(figsize=(7.057, 4.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.00, 1.04],
                             hspace=0.36, left=0.115, right=0.985,
                             top=0.935, bottom=0.045)

    # --- row 1: (a) blobs 3x8 | (b) Ising 3x8, identical cell geometry -----
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.14)
    ga = top[0].subgridspec(len(SCHEMES), L, wspace=0.10, hspace=0.16)
    blob_axes = [[] for _ in SCHEMES]
    for r, (_, rowlab, _) in enumerate(SCHEMES):
        for c in range(L):
            ax = fig.add_subplot(ga[r, c])
            ax.imshow(rep_means[r][c].reshape(SIDE, SIDE), cmap="gray_r",
                      vmin=0, vmax=1)
            blob_axes[r].append(ax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title(rf"$\ell={c}$", fontsize=6, pad=1.5)

    gb = top[1].subgridspec(len(SCHEMES), L, wspace=0.10, hspace=0.16)
    MC = 0.5 * (isg["mbins"][:-1] + isg["mbins"][1:])
    ymax = max(float(isg["data"].max()), float(isg["hists"].max())) * 1.1
    ising_axes = [[] for _ in SCHEMES]
    for r in range(len(SCHEMES)):
        for c in range(L):
            ax = fig.add_subplot(gb[r, c])
            ax.fill_between(MC, isg["data"], step="mid", color="0.85", lw=0)
            ax.fill_between(MC, isg["hists"][r, c], step="mid",
                            color=ROW_COLORS[r], alpha=0.9, lw=0)
            ax.set_ylim(0, ymax); ax.set_xlim(-NI, NI)
            ax.set_yticks([])
            if r == len(SCHEMES) - 1 and c == 0:
                ax.set_xticks([-NI, 0, NI])
                ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            else:
                ax.set_xticks([])
            if r == 0:
                ax.set_title(rf"$\ell={c}$", fontsize=6, pad=1.5)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.6")
            ising_axes[r].append(ax)

    # align the Ising rows with the (square) blobs rows
    fig.canvas.draw()
    for r in range(len(SCHEMES)):
        for ax in blob_axes[r]:
            ax.apply_aspect()
        y0 = min(a.get_position(original=False).y0 for a in blob_axes[r])
        y1 = max(a.get_position(original=False).y1 for a in blob_axes[r])
        for ax in ising_axes[r]:
            b = ax.get_position()
            ax.set_position([b.x0, y0, b.width, y1 - y0])
    ising_last = ising_axes[-1]
    b0, b1 = ising_last[0].get_position(), ising_last[-1].get_position()
    fig.text(0.5 * (b0.x0 + b1.x1), b0.y0 - 0.020,
             r"magnetization $M$", ha="center", va="top", fontsize=7)

    # --- row 2: (c) MNIST 3x16 -------------------------------------------
    gcm = outer[1].subgridspec(len(SCHEMES), MNIST_COLS, wspace=0.06,
                               hspace=0.10)
    mnist_first = {}
    for r, (_, rowlab, _) in enumerate(SCHEMES):
        for c in range(MNIST_COLS):
            ax = fig.add_subplot(gcm[r, c])
            ax.imshow(mmeans[r, c].reshape(mside, mside), cmap="gray_r",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.7")
            if r == 0:
                ax.set_title(rf"$\ell={c}$", fontsize=6, pad=1.2)
            if c == 0:
                mnist_first[r] = ax

    # one shared row-label column, right-aligned at the figure's left margin
    LABX = 0.105
    for r, (_, rowlab, _) in enumerate(SCHEMES):
        for axes_of in (blob_axes[r][0], mnist_first[r]):
            b = axes_of.get_position(original=False)
            fig.text(LABX, 0.5 * (b.y0 + b.y1), rowlab, fontsize=7,
                     ha="right", va="center")

    # panel tags anchored to the panels themselves
    ba = blob_axes[0][0].get_position(original=False)
    fig.text(0.004, ba.y1 + 0.035, "(a) blobs", fontsize=8,
             fontweight="bold")
    bi = ising_axes[0][0].get_position()
    fig.text(bi.x0 - 0.035, bi.y1 + 0.035, "(b) Ising", fontsize=8,
             fontweight="bold")
    bm = mnist_first[0].get_position(original=False)
    fig.text(0.004, bm.y1 + 0.035, "(c) MNIST", fontsize=8, fontweight="bold")
    fig.savefig(OUT)
    fig.savefig(HERE / ("fig_modesep_full_preview.png" if FULL
                        else "fig_modesep_preview.png"), dpi=170)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
