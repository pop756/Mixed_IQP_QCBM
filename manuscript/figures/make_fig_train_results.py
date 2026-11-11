#!/usr/bin/env python3
r"""Two 1x4 figures (figure*), columns the four benchmarks (blobs, Ising, D-Wave, MNIST):
  fig_train_results   (a)-(d): test MMD^2 against the ancilla count a, seed mean with the seed standard deviation,
                               against the train-test sampling reference and the trained one-body mixture (main text);
  fig_app_train_curves (a)-(d): training MMD^2 against the training step at each dataset's headline mixture size,
                               seed mean (smoothed bold, raw faint) with the seed band (appendix).

Data: the cosine rerun (experiments/cosine_rerun, via _seed_agg): training loss averaged over blocks of 50 steps,
final test MMD^2 averaged over m-bar = 1..6, and the trained one-body mixtures (variant fc1). Seed sets are
dataset-specific (_seed_agg.seeds_of): ten for the n=16 datasets, five for MNIST and D-Wave.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import SymmetricalLogLocator
import matplotlib.pyplot as plt

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()

from _paths import PAPER_OURS
from _seed_agg import LOSS_STRIDE, aggregate_losses, final_means, seeds_of

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "fig_train_results.pdf"
OUT_CURVES = HERE / "fig_app_train_curves.pdf"

order = ["S1_visible", "S4", "S2_cluster"]
labels = {"S1_visible": "global", "S4": "data-agnostic",
          "S2_cluster": "cluster-initialized"}
colors = {"S1_visible": "#666666", "S4": "#1f77b4", "S2_cluster": "#2ca02c"}
markers = {"S1_visible": "o", "S4": "D", "S2_cluster": "^"}
HOLLOW = {"S2_cluster"}

# dataset, title, headline ancilla, ancilla ladder, RBM width key
PANELS = [
    ("blobs", "blobs",  3, [0, 1, 2, 3], "RBM_h64"),
    ("ising", "Ising",  3, [0, 1, 2, 3], "RBM_h64"),
    ("dwave", "D-Wave", 3, [0, 1, 2, 3], "RBM_h230"),
    ("mnist", "MNIST",  4, [0, 1, 2, 3, 4], "RBM_h896"),
]
LINTHRESH = 1e-4
FS = 8.5
TAGS = "abcdefgh"


def smooth(x, w=25):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode="valid")


def seed_sweep_means(ds):
    """{(scheme, a): (mean, std)} over training seeds of the final test MMD^2 averaged over m-bar = 1..6; "True"
    rows carry the sweep-averaged train-test sampling reference (cosine rerun, see _seed_agg)."""
    return final_means(ds)


def rbm_value(ds, width):
    try:
        bdf = pd.read_csv(PAPER_OURS / "baselines" / ds
                          / "mmd_sigma_sweep_baselines.csv")
        return float(bdf[bdf.scheme == width].mmd2_mean.mean())
    except Exception as exc:                                # pragma: no cover
        print(f"[warn] no RBM value for {ds}: {exc}")
        return None


def _tag(ax, i, title=""):
    """Left-aligned panel heading: bold tag, then the optional title."""
    head = rf"$\bf{{({TAGS[i]})}}$" + (f"  {title}" if title else "")
    ax.set_title(head, fontsize=FS, pad=3, loc="left")


figc, axc = plt.subplots(1, 4, figsize=(7.057, 1.95))
fig, axt = plt.subplots(1, 4, figsize=(7.057, 2.05))

# ---------------------------------------------------------------- top row
for i, (ax, (ds, title, a, _ladder, _rbm)) in enumerate(zip(axc, PANELS)):
    hi, lo, lo_pos = 1e-3, 0.0, np.inf
    for key in order:
        res = aggregate_losses(ds, key, a)
        if res is None:
            continue
        Lm, Ls = res
        x = (np.arange(len(Lm)) + 1) * LOSS_STRIDE
        ax.plot(x, Lm, color=colors[key], alpha=0.12, lw=0.5, rasterized=True)
        # the closed-form training loss is positive: a log axis, the band clipped at 5 % of the mean
        ax.fill_between(x, np.maximum(Lm - Ls, 0.05 * Lm), Lm + Ls, color=colors[key], alpha=0.15,
                        lw=0, rasterized=True)
        sm = smooth(Lm, 5)
        ax.plot(x[(len(Lm) - len(sm)) // 2:][:len(sm)], sm,
                color=colors[key], lw=1.1, label=labels[key], rasterized=True)
        hi = max(hi, float(np.nanmax(Lm)))
        lo_pos = min(lo_pos, float(np.nanmin(np.maximum(Lm - Ls, 0.05 * Lm))))
    # each panel keeps its own range: a symlog window with the negative branch
    # only where the finite-sample traces actually dip below zero, plain log
    # otherwise, so no panel is anchored at zero
    if lo < 0:
        ax.axhline(0.0, color="0.6", lw=0.5, zorder=1)
        ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
        ax.yaxis.set_minor_locator(SymmetricalLogLocator(
            base=10.0, linthresh=1e-4, subs=list(range(2, 10))))
        ax.set_ylim(lo * 1.6, hi * 1.15)
    else:
        ax.set_yscale("log")
        ax.set_ylim(max(lo_pos, 1e-12) * 0.7, hi * 1.15)

    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.tick_params(labelsize=FS - 1, length=2.6, width=0.6, pad=2)
    ax.tick_params(which="minor", length=1.5, width=0.5)
    ax.set_xlabel("training step", fontsize=FS, labelpad=1.5)
    _tag(ax, i, title)
axc[0].set_ylabel(r"training MMD$^2$", fontsize=FS, labelpad=1.5)

# ------------------------------------------------------------- bottom row
for i, (ax, (ds, _title, _a, ancillas, rbm_w)) in enumerate(zip(axt, PANELS)):
    agg = seed_sweep_means(ds)
    tvals = [agg[k][0] for k in agg if k[0] == "True"]
    if tvals:
        ax.axhline(float(np.mean(tvals)), color="black", ls="--", lw=0.9,
                   zorder=2, label="train–test ref.")
    pa = sorted(a for (k, a) in agg if k == "product")
    pv = np.array([agg[("product", a)][0] for a in pa])
    ax.plot(pa, pv, color="#999999", ls="--", lw=1.0, marker="v",
            ms=2.4, zorder=2, label="one-body mixture (trained)")
    hi, lo = 0.0, np.inf
    dodge = {k: d for k, d in zip(order, (-0.13, 0.0, 0.13))}
    for key in order:
        xs, ys, es = [], [], []
        for a in ancillas:
            if (key, a) not in agg:
                continue
            m, e = agg[(key, a)]
            xs.append(a); ys.append(m); es.append(e)
        if not xs:
            continue
        xs, ys, es = (np.asarray(v, dtype=float) for v in (xs, ys, es))
        ax.errorbar(xs + dodge[key], ys, yerr=es, color=colors[key],
                    marker=markers[key], ls="none", markersize=3.4,
                    mfc="none" if key in HOLLOW else colors[key],
                    capsize=1.8, elinewidth=0.8, mew=0.8,
                    label=labels[key], zorder=4)
        hi = max(hi, float(np.max(ys + es)))
        pos = (ys - es)[(ys - es) > 0]
        lo = min(lo, float(pos.min()) if pos.size else float(ys.min()) * 0.8)
    lo = min([lo, float(pv.min())] + ([float(np.mean(tvals))] if tvals else []))
    ax.set_yscale("log")
    ax.set_ylim(lo * 0.6, max(hi, float(pv.max())) * 2.2)
    ax.set_xticks(ancillas)
    ax.set_xticklabels([fr"${a}$" for a in ancillas])
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.tick_params(labelsize=FS - 1, length=2.6, width=0.6, pad=2)
    ax.tick_params(which="minor", length=1.5, width=0.5)
    ax.set_xlabel(r"ancilla count $a$", fontsize=FS, labelpad=1.5)
    _tag(ax, i, _title)
axt[0].set_ylabel(r"test MMD$^2$", fontsize=FS, labelpad=1.5)

handles, lbls = [], []
for ax in (axt[0],):
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in lbls:
            handles.append(h)
            lbls.append(l)
fig.legend(handles, lbls, frameon=False, fontsize=FS - 0.5, ncol=6,
           loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=1.5,
           columnspacing=1.2, handletextpad=0.4)
fig.tight_layout(rect=(0.0, 0.12, 1, 1), w_pad=0.9)
fig.savefig(OUT_PDF, dpi=600)
print("wrote", OUT_PDF)

hc, lc = axc[0].get_legend_handles_labels()
figc.legend(hc, lc, frameon=False, fontsize=FS - 0.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0),
            handlelength=1.5, columnspacing=1.2, handletextpad=0.4)
figc.tight_layout(rect=(0.0, 0.1, 1, 1), w_pad=0.9)
figc.savefig(OUT_CURVES, dpi=600)
print("wrote", OUT_CURVES)
