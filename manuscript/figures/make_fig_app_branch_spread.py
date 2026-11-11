#!/usr/bin/env python3
r"""fig_app_branch_spread -- the Theorem-3 sweep at every ancilla count.

Appendix companion to Fig. fig_theory_check(c): the exact ancilla-coupling
gradient norm at a branch-coincident start against the branch spread delta, for
both n=16 targets and a = 1, 2, 3; the vertical line is the spread of the
initialization the runs use.

Data (read-only): analysis_ancilla_angles/noise_grad_<ds>_a<a>_exact.csv
Run: python make_fig_app_branch_spread.py -> fig_app_branch_spread.pdf
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
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig_app_branch_spread.pdf"

_CAND = [
    HERE.parents[1] / "legacy" / "experiments" / "paper_ours" / "analysis_ancilla_angles",
]
for _c in _CAND:
    if _c.exists():
        CSVDIR = _c
        break
else:
    raise FileNotFoundError("analysis_ancilla_angles not found")

FS = 8.5
MS = 3.6
SCHEME = {"S1_visible": ("global", "#666666", "o"),
          "S4": ("data-agnostic", "#1f77b4", "D")}
DATASETS = [("blobs", "blobs"), ("ising", "Ising")]


def _agg(ds, a, mode, scheme):
    g = defaultdict(list)
    with (CSVDIR / f"noise_grad_{ds}_a{a}_{mode}.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["scheme"] == scheme:
                g[float(r["scale"])].append((float(r["delta"]),
                                             float(r["gnorm_anc"])))
    cs = sorted(g)
    return (np.array([np.mean([v[0] for v in g[c]]) for c in cs]),
            np.array([np.mean([v[1] for v in g[c]]) for c in cs]),
            cs)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.057, 4.3), sharey="row")
    for i, (ds, disp) in enumerate(DATASETS):
        for j, a in enumerate((1, 2, 3)):
            ax = axes[i, j]
            for key, (name, col, mk) in SCHEME.items():
                xe, ye, cs = _agg(ds, a, "exact", key)
                m = ye > 0
                ax.plot(xe[m], ye[m], "-", marker=mk, ms=MS, lw=1.0, color=col)
                if key == "S1_visible":
                    ax.axvline(xe[list(cs).index(1.0)], color="black", lw=1.1,
                               ls=(0, (4, 1.5, 1, 1.5)))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both", lw=0.5)
            ax.set_title(rf"$\bf{{({chr(97 + 3 * i + j)})}}$  {disp}, $a={a}$",
                         pad=3, loc="left")
            if i == 1:
                ax.set_xlabel(r"branch spread $\delta$",
                              labelpad=1)
            if j == 0:
                ax.set_ylabel(r"$\Vert\nabla_{\mathrm{anc}}\mathcal{L}\Vert$", labelpad=1)
    handles = [Line2D([0], [0], color=c, marker=m, ms=MS, lw=1.0, label=d)
               for d, c, m in SCHEME.values()] + [
        Line2D([0], [0], color="black", ls=(0, (4, 1.5, 1, 1.5)), lw=1.1,
               label="initialization used"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False, handlelength=1.5,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0), pad=0.35, w_pad=0.8, h_pad=1.1)
    fig.savefig(OUT)
    fig.savefig(HERE / "fig_app_branch_spread_preview.png", dpi=190)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
