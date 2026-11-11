#!/usr/bin/env python3
"""Full test-MMD^2 kernel sweeps for the four numerical benchmarks (appendix).

X axis: the mean Pauli weight m-bar = n p_sigma of the evaluation kernel
(Sec. II B), the quantity by which the paper indexes its kernels; one point per
evaluated bandwidth. Rendered at print size with the shared _paperstyle fonts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()
import numpy as np
import pandas as pd

from _paths import PAPER_OURS
from _seed_agg import N_OF, aggregate_sweep, mbar_of


HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "fig_app_sigma_sweeps.pdf"
OUT_PNG = HERE / "fig_app_sigma_sweeps_preview.png"

PANELS = [
    ("blobs", "blobs", 3, "RBM_h64"),
    ("ising", "Ising", 3, "RBM_h64"),
    ("dwave", "D-Wave", 3, "RBM_h230"),
    ("mnist", "MNIST", 4, "RBM_h896"),
]

SERIES = [
    ("single circuit", "S1_visible", 0, "#111111", "X"),
    ("global", "S1_visible", None, "#777777", "o"),
    ("data-agnostic", "S4", None, "#1f77b4", "D"),
    ("cluster-initialized", "S2_cluster", None, "#2ca02c", "^"),
]

FS = 8.5


def baseline_frame(dataset: str) -> pd.DataFrame:
    path = PAPER_OURS / "baselines" / dataset / "mmd_sigma_sweep_baselines.csv"
    return pd.read_csv(path)


fig, axes = plt.subplots(2, 2, figsize=(7.057, 5.35))
tags = "abcd"

for tag, ax, (dataset, title, a_star, rbm_scheme) in zip(tags, axes.ravel(), PANELS):
    sweep, _, _ = aggregate_sweep(dataset)
    base = baseline_frame(dataset) if dataset in ("blobs", "ising") else None

    series = SERIES + ([("one-body mixture (trained)", "product", None, "#999999", "v")]
                       if dataset in ("dwave", "mnist") else [])
    for label, key, fixed_a, color, marker in series:
        ancilla = a_star if fixed_a is None else fixed_a
        sigma, mean, std = sweep[key][ancilla]
        m = np.asarray([mbar_of(N_OF[dataset], x) for x in sigma], float)
        order = np.argsort(m)
        m, mean, std = m[order], np.asarray(mean, float)[order], np.asarray(std, float)[order]
        ax.plot(m, mean, color=color, marker=marker, ms=3.6, lw=1.2,
                ls="--" if key == "product" else "-", label=label, zorder=4)
        positive = mean[mean > 0]
        floor = positive.min() * 0.15 if positive.size else 1e-8
        ax.fill_between(m, np.maximum(mean - std, floor), mean + std,
                        color=color, alpha=0.14, linewidth=0, zorder=2)

    sigma, floor_mean, _ = sweep["True"][0]
    m = np.asarray([mbar_of(N_OF[dataset], x) for x in sigma], float)
    order = np.argsort(m)
    fm = np.asarray(floor_mean, float)[order]
    if (fm > 0).all():
        ax.plot(m[order], fm, color="#333333", ls="--", lw=1.2, label="train–test ref.", zorder=3)

    if dataset in ("blobs", "ising"):     # the RBM is fitted to the n = 16 datasets only
        rows = base[base.scheme == rbm_scheme].sort_values("pauli")
        ax.plot(rows.pauli.to_numpy(float), rows.mmd2_mean.to_numpy(float),
                color="#d62728", ls="-.", lw=1.3, label="RBM", zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(range(1, 7))
    ax.tick_params(labelsize=FS - 0.5, length=3, pad=1.5)
    ax.grid(alpha=0.28, which="both", lw=0.35)
    ax.set_title(rf"$\bf{{({tag})}}$  {title}, $a={a_star}$",
                 fontsize=FS, pad=4, loc="left")
    ax.set_xlabel(r"mean Pauli weight $\bar m$ of the kernel", fontsize=FS,
                  labelpad=1.5)

for ax in axes[:, 0]:
    ax.set_ylabel(r"test MMD$^2$", fontsize=FS)

handles, labels = [], []
for ax in (axes[0, 0], axes[1, 1]):
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=FS, handlelength=1.9, columnspacing=1.0,
           handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0), h_pad=1.6, w_pad=1.4)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=180)
print("wrote", OUT_PDF)
