#!/usr/bin/env python3
r"""Trainability at scale and escape from coincidence (Sec. V B), full width, four panels:
(a), (b) loss variance on the initialization patch against n: binary blobs(n) and D-Wave restricted to its first n
    spins. Mixed IQP-QCBM with a = 2, all-pairs gates; the three centres with the campaign's uniform patch of
    half-width pi/(2 sqrt(LG)) on every compiled angle, and fully random branch angles. Random-start variances below
    1e-30 (beyond the double-precision resolution of the loss) are not drawn. A dotted n^-3 guide is shown.
(c) the same variance at n = 64 against the branch count L = 2^a, a = 0..5, both datasets (blobs solid, D-Wave
    dashed), with an L^-1 guide; the random-angle reference (1e-27 to 1e-24 at every L) is off the axis.
(d) escape step T_sep against delta for every (dataset, start): GD (filled) with the linearized prediction (lines),
    Adam (open); delta = 0 never escapes.

Data (read-only): experiments/variance_scaling/results/scaling_{blobs,dwave}.json (scaling.py),
                  experiments/variance_scaling/results/ancilla_{blobs,dwave}.json (ancilla.py),
                  experiments/gd_escape/results/escape_<ds>_<scheme>.json (escape.py)
Run:  python make_fig_landscape.py -> fig_landscape.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()
HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
VAR = EXP / "variance_scaling" / "results"
ESC = EXP / "gd_escape" / "results"
OUT = HERE / "fig_landscape.pdf"
STYLE = {"global": ("global", "#666666", "o"), "agnostic": ("data-agnostic", "#1f77b4", "D"),
         "cluster": ("cluster-initialized", "#2ca02c", "^"), "random": ("random angles", "#d62728", "s")}
# (d): colour = start (as in (a)-(c)), marker = dataset
ESC_CONFIGS = [("ising", "global", "#666666", "o"), ("ising", "agnostic", "#1f77b4", "o"),
               ("blobs", "global", "#666666", "s"), ("blobs", "agnostic", "#1f77b4", "s")]


def variance_panel(ax, ds, title, ylabel):
    r = json.load(open(VAR / f"scaling_{ds}.json"))
    for s in ("agnostic", "global", "cluster", "random"):
        n = np.array([p["n"] for p in r["schemes"][s]])
        v = np.array([p["var"] for p in r["schemes"][s]])
        keep = v > 1e-30
        lab, col, mk = STYLE[s]
        ax.plot(n[keep], v[keep], marker=mk, ms=2.6, lw=0.8, color=col, label=lab,
                mfc=col if s != "agnostic" else "none")
    n0 = np.array([16, 484])
    ax.plot(n0, 3e-4 * (n0 / 16.0) ** -3, ls=":", color="k", lw=0.7)
    ax.text(200, 3e-4 * (200 / 16.0) ** -3 * 3, r"$n^{-3}$", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 1e-3)
    ax.set_xlim(10, 560)
    ax.set_yticks([1e-30, 1e-20, 1e-10])
    ax.set_xlabel(r"$n$")
    if ylabel:
        ax.set_ylabel(r"$\mathrm{Var}_\Theta[\mathcal{L}]$")
    ax.set_title(title, loc="left", fontsize=8.5)


def ancilla_panel(ax):
    for ds, ls, mfc_open in (("blobs", "-", False), ("dwave", "--", True)):
        r = json.load(open(VAR / f"ancilla_{ds}.json"))
        for s in ("agnostic", "global", "cluster"):
            L = np.array([2 ** p["a"] for p in r["schemes"][s]])
            v = np.array([p["var"] for p in r["schemes"][s]])
            keep = v > 1e-30
            _, col, mk = STYLE[s]
            ax.plot(L[keep], v[keep], marker=mk, ms=2.6, lw=0.8, ls=ls, color=col,
                    mfc="none" if (mfc_open or s == "agnostic") else col)
    L0 = np.array([1, 32])
    ax.plot(L0, 1.5e-6 / L0, ls=":", color="k", lw=0.7)
    ax.text(5, 1.5e-6 / 5 * 2.2, r"$L^{-1}$", fontsize=7.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(5e-11, 3e-6)
    ax.set_xlabel(r"branch count $L=2^a$")
    ax.set_title(r"(c) $n=64$", loc="left", fontsize=8.5)


def escape_time_panel(ax):
    from matplotlib.lines import Line2D
    for ds, sc, col, mk in ESC_CONFIGS:
        r = json.load(open(ESC / f"escape_{ds}_{sc}.json"))
        d = np.array(r["deltas"])
        ok = d > 0
        for kind, fill in (("gd", True), ("adam", False)):
            T = np.array([np.nan if v is None else v for v in r[kind]["T_sep"]], dtype=float)
            ax.plot(d[ok], T[ok], ls="none", marker=mk, ms=3, color=col, mfc=col if fill else "none", mew=0.8)
        Tp = np.array([np.nan if v is None else v for v in r["gd"]["T_sep_pred"]], dtype=float)
        ax.plot(d[ok], Tp[ok], ls="-", lw=0.7, color=col)
    ax.set_xscale("log")
    ax.set_xlabel(r"initial separation $\delta$")
    ax.set_ylabel(r"escape step $T_{\mathrm{sep}}$")
    ax.set_title("(d) escape from coincidence", loc="left", fontsize=8.5)
    hs = [Line2D([], [], ls="none", marker="o", ms=3, color="k", label="Ising"),
          Line2D([], [], ls="none", marker="s", ms=3, color="k", label="blobs"),
          Line2D([], [], ls="-", marker="o", ms=3, lw=0.7, color="0.4", label="GD, prediction"),
          Line2D([], [], ls="none", marker="o", ms=3, mfc="none", color="0.4", label="Adam")]
    ax.legend(handles=hs, fontsize=6.3, loc="upper right", handlelength=1.4, labelspacing=0.25,
              borderaxespad=0.3, frameon=False)


def main():
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.25))
    variance_panel(axes[0], "blobs", "(a) blobs", True)
    variance_panel(axes[1], "dwave", "(b) D-Wave", False)
    ancilla_panel(axes[2])
    escape_time_panel(axes[3])
    hs = []
    for key in ("global", "agnostic", "cluster", "random"):
        lab, col, mk = STYLE[key]
        hs.append(Line2D([], [], color=col, marker=mk, ms=3, lw=0.8, mfc="none" if key == "agnostic" else col,
                         label=lab))
    hs += [Line2D([], [], color="k", lw=0.8, ls="-", label="blobs (c)"),
           Line2D([], [], color="k", lw=0.8, ls="--", label="D-Wave (c)")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, fontsize=7.0, handlelength=1.8,
               columnspacing=1.1, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.2, w_pad=0.9, rect=(0, 0.1, 1, 1))
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
