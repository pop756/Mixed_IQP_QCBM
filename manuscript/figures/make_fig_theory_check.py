#!/usr/bin/env python3
r"""fig_theory_check -- exact numerical check of the three local statements.

Three panels, one per theorem, on the two n=16 datasets (blobs, Ising), where
the (n+a)-qubit Born distribution is available exactly:

 (a) Theorem 1 [Eq. rand-curv]: the center curvature of a single branch's
     one-body angle at the data-agnostic center against the branch count,
     measured against 8 w_{alpha} / L^2.
 (b) Theorem 2 [Eq. datadep-curv-main]: the exact witness curvature at the
     global and cluster centers against the certified bound c_lead - R_n, with
     c_lead = 8 w_{alpha} pi_{l*}^2 (1 - (t_alpha^{(l*)})^2) at the witness
     coordinate and R_n the exact center residual; the theorem places every
     point on or above the diagonal (the same 14 rows as SM Table SII).
 (c) Theorem 3 [Eq. grad-suppress-main]: the exact ancilla-coupling gradient
     norm against the branch spread delta (a=1), blobs and Ising in two
     side-by-side axes; the dash-dotted vertical line is the spread of the run
     initialization, as in fig_app_branch_spread.

Encoding follows the other result figures: colour and marker name the scheme
(global grey circle, data-agnostic blue diamond, cluster-initialized green
triangle), a filled marker is the blobs and an open marker the Ising.  Panel (a) tells its two directions apart by the
reference line (dotted single branch, dashed common mode) and direct labels;
panel (b) shades the half-plane the theorem certifies.  Authored at final print
size with the shared _paperstyle typography.

Data (read-only):
  analysis_ancilla_angles/: curvature_L_scaling.csv,
    noise_grad_{blobs,ising}_a1_exact.csv
  experiments/paper_ours/referee_2026_10/theory_numerics/:
    item1_theorem2_remainder.csv (panel b), item2_common_mode.csv (panel a)
Run:  python make_fig_theory_check.py -> fig_theory_check.pdf
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

TITLE_FS = 22.62
TAG_FS = 23.4
ANNOT_FS = 17.68
XTICK_FS = 19.24
YTICK_FS = 18.46
YLABEL_FS = 21.06
LEGEND_FS = 20.28
XLABEL_FS = 20.54

_apply_paper_style(base=ANNOT_FS)
matplotlib.rcParams.update({
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.minor.width": 1.1,
    "ytick.minor.width": 1.1,
})
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig_theory_check.pdf"

_CAND = [
    HERE.parents[1] / "legacy" / "experiments" / "paper_ours" / "analysis_ancilla_angles",
    HERE.parents[1] / "scaling-gqml" / "paper" / "paper_ours" / "analysis_ancilla_angles",
]
for _c in _CAND:
    if _c.exists():
        CSVDIR = _c
        break
else:
    raise FileNotFoundError("analysis_ancilla_angles not found")
# exact Theorem-2 remainder and common-mode curvature (referee revision)
NUMDIR = HERE.parents[1] / "legacy" / "experiments" / "paper_ours" / "referee_2026_10" / "theory_numerics"

MS = 9.6          # marker size before manuscript scaling
LW = 2.6          # data line width before manuscript scaling
LW_REF = 2.4      # reference line width before manuscript scaling

# scheme -> (display, colour, marker); identical to make_fig_results_all.py
SCHEME = {
    "global": ("global", "#666666", "o"),
    "data-agnostic": ("data-agnostic", "#1f77b4", "D"),
    "cluster": ("cluster-initialized", "#2ca02c", "^"),
}
# per-dataset shade of each scheme colour: blobs saturated, Ising light
SHADE = {("global", "blobs"): "#4d4d4d", ("global", "ising"): "#a6a6a6",
         ("data-agnostic", "blobs"): "#1f77b4",
         ("data-agnostic", "ising"): "#7fb8dd",
         ("cluster", "blobs"): "#2ca02c", ("cluster", "ising"): "#88d18a"}
DS_LABEL = {"blobs": "blobs", "ising": "Ising"}
CERT = "#fbf5e0"    # fill of the region Theorem 2 certifies


def rows(name, base=None):
    with ((base or CSVDIR) / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _mk(ax, x, y, center, dataset, **kw):
    """Plot one series: marker = scheme, colour shade and fill = dataset."""
    _disp, _c, mk = SCHEME[center]
    col = SHADE[(center, dataset)]
    ax.plot(x, y, marker=mk, color=col, ms=MS, mew=0.9,
            mfc=col if dataset == "blobs" else "none", **kw)


def _finish_panel(ax, tag, title=""):
    # the panels carry only their letter; what each one shows is stated in the
    # caption, which also names the theorem it tests
    head = rf"$\bf{{({tag})}}$" + (f"  {title}" if title else "")
    ax.set_title(head, fontsize=TITLE_FS, pad=9.5, loc="left")
    ax.tick_params(axis="x", labelsize=XTICK_FS, length=6.2, width=1.4, pad=4.8)
    ax.tick_params(axis="y", labelsize=YTICK_FS, length=6.2, width=1.4, pad=4.8)
    ax.tick_params(which="minor", length=3.6, width=1.1)
    ax.grid(alpha=0.25, which="both", lw=1.2)


def panel_a(ax):
    """Theorem 1: measured curvature vs 8 w/L^2 at the data-agnostic center."""
    seen = {}
    for r in rows("curvature_L_scaling.csv"):
        if r["center"] != "data-agnostic" or float(r["step"]) != 0.0:
            continue
        seen[(r["dataset"], int(r["L"]))] = (float(r["curv_measured"]),
                                             float(r["curv_pred_8w_L2"]))
    Ls = sorted({k[1] for k in seen})
    ref = [seen[("blobs", L)][1] for L in Ls]
    ax.plot(Ls, ref, ls=":", color="black", lw=LW_REF, zorder=1)
    # the two datasets carry identical values here (the center uses no data),
    # so they are dodged slightly in L to stay individually visible
    for ds, dodge in (("blobs", 1 / 1.05), ("ising", 1.05)):
        _mk(ax, [L * dodge for L in Ls], [seen[(ds, L)][0] for L in Ls],
            "data-agnostic", ds, ls="none", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_ylim(7.5e-4, 9.0e-2)   # keep the top decade off the axis
    ax.set_xlabel(r"branch count $L=2^a$", fontsize=XLABEL_FS, labelpad=3.6)
    ax.set_ylabel(r"center curvature $c$", fontsize=YLABEL_FS, labelpad=3.6)
    ax.text(1.12, 2.2e-3, "single branch angle\n" + r"$8w_{\{j\}}/L^{2}$",
            fontsize=ANNOT_FS, ha="left", va="center", linespacing=1.15)
    # compiled common-mode angle: the single-circuit value 8 w_j for every L
    cm = {(r["dataset"], int(r["L"])): float(r["c_cm"])
          for r in rows("item2_common_mode.csv", NUMDIR)
          if r["center"] == "agnostic"}
    ax.axhline(cm[("blobs", 1)], ls="--", color="black", lw=LW_REF * 0.8,
               zorder=1)
    # same diamonds as the single-branch points: the marker names the center,
    # the reference line and its label name the direction
    for ds, dodge in (("blobs", 1 / 1.05), ("ising", 1.05)):
        _mk(ax, [L * dodge for L in Ls], [cm[(ds, L)] for L in Ls],
            "data-agnostic", ds, ls="none", zorder=3)
    ax.text(8.4, cm[("blobs", 1)] / 1.55, "common-mode angle\n" + r"$8w_{\{j\}}$",
            fontsize=ANNOT_FS, ha="right", va="top", linespacing=1.15)
    _finish_panel(ax, "a")


def panel_b(ax):
    """Theorem 2: exact witness curvature against the bound c_lead - R_n."""
    best = {}
    for r in rows("item1_theorem2_remainder.csv", NUMDIR):
        key = (r["dataset"], r["center"], int(r["L"]))
        if key not in best or float(r["c_lead"]) > float(best[key]["c_lead"]):
            best[key] = r          # witness = maximizer of c_lead
    d = {k: (float(r["lead_minus_Rn"]), float(r["c"])) for k, r in best.items()}
    equality = np.geomspace(5.0e-5, 2.0e-1, 200)
    # Theorem 2 places every point on or above the diagonal
    ax.fill_between(equality, equality, 1.0, color=CERT, lw=0, zorder=0)
    ax.plot(equality, equality, ls=":", color="black", lw=LW_REF, zorder=1)
    for ds in ("blobs", "ising"):
        for center in ("global", "cluster"):
            Ls = sorted(L for (a, c, L) in d if a == ds and c == center)
            if not Ls:
                continue
            _mk(ax,
                [d[(ds, center, L)][0] for L in Ls],
                [d[(ds, center, L)][1] for L in Ls],
                center, ds, ls="none", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6.0e-5, 1.6e-1)
    ax.set_ylim(6.0e-5, 1.6e-1)
    ax.set_xlabel(r"certified bound $c_{\mathrm{lead}}-R_n$",
                  fontsize=XLABEL_FS, labelpad=3.6)
    ax.set_ylabel(r"exact curvature $c$", fontsize=YLABEL_FS, labelpad=3.6)
    ax.text(8.0e-5, 1.15e-1, "certified region\n" + r"$c\geq c_{\mathrm{lead}}-R_n$",
            fontsize=ANNOT_FS, ha="left", va="top", linespacing=1.15)
    ax.text(1.25e-1, 8.0e-5, r"$c<c_{\mathrm{lead}}-R_n$" + "\nexcluded",
            fontsize=ANNOT_FS, ha="right", va="bottom", linespacing=1.15,
            color="#555555")
    _finish_panel(ax, "b")


def _agg(name, scheme):
    g = defaultdict(list)
    for r in rows(name):
        if r["scheme"] == scheme:
            g[float(r["scale"])].append((float(r["delta"]), float(r["gnorm_anc"])))
    xs = [np.mean([v[0] for v in g[s]]) for s in sorted(g)]
    ys = [np.mean([v[1] for v in g[s]]) for s in sorted(g)]
    return np.asarray(xs), np.asarray(ys)


def panel_c(axes):
    """Theorem 3: exact ancilla-coupling gradient against the branch spread, one
    axis per dataset."""
    runs = []
    correlations = []
    for ax, ds in zip(axes, ("blobs", "ising")):
        for key, center in (("S1_visible", "global"), ("S4", "data-agnostic")):
            xe, ye = _agg(f"noise_grad_{ds}_a1_exact.csv", key)
            correlations.append(float(np.corrcoef(xe[:-1], ye[:-1])[0, 1]))
            xe = np.where(ye == 0.0, 0.0, xe)      # the coincident start: spread and gradient exactly zero
            _mk(ax, xe, ye, center, ds, ls="-", lw=LW, zorder=3)
            runs.append(xe[-2])
    run_init = float(np.mean(runs))
    for ax, ds in zip(axes, ("blobs", "ising")):
        ax.axvline(run_init, ls="-.", color="black", lw=LW_REF * 0.8, zorder=1)
        # symmetric-log axes, linear below the thresholds, so that the coincident start (delta = 0,
        # gradient exactly 0) sits at the origin and the linear growth above it stays visible on log scales
        ax.set_xscale("symlog", linthresh=1e-4, linscale=0.6)
        ax.set_yscale("symlog", linthresh=1e-5, linscale=0.6)
        ax.set_xlim(0.0, 0.6)
        ax.set_ylim(0.0, 5.0e-2)
        ax.set_xticks([0, 1e-3, 1e-1])
        ax.set_yticks([0, 1e-4, 1e-3, 1e-2])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(r"branch spread $\delta$", fontsize=XLABEL_FS, labelpad=3.6)
        ax.text(0.5, 0.97, DS_LABEL[ds], transform=ax.transAxes,
                fontsize=ANNOT_FS + 1, ha="center", va="top")
    # the run line is named once, in the blobs axis
    axes[0].text(run_init / 1.25, 2.0e-5, "run initialization", rotation=90,
                 fontsize=ANNOT_FS, ha="right", va="bottom")
    axes[0].set_ylabel(r"$\Vert\nabla_{\mathrm{anc}}\mathcal{L}\Vert$",
                       fontsize=YLABEL_FS, labelpad=3.6)
    axes[1].tick_params(labelleft=False)
    print(f"[theory-c] min Pearson r = {min(correlations):.4f}, "
          f"run init delta = {run_init:.4f}")
    _finish_panel(axes[0], "c")
    _finish_panel(axes[1], "c")
    axes[1].set_title("", loc="left")   # one panel letter for the pair


def main():
    # Authored 2.4x larger than print size so the requested source font sizes
    # reduce to the manuscript's established final-size typography.
    fig = plt.figure(figsize=(16.94, 6.12))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.55])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    gc = gs[2].subgridspec(1, 2, wspace=0.08)
    ax_c0 = fig.add_subplot(gc[0])
    ax_c1 = fig.add_subplot(gc[1], sharey=ax_c0)
    axes = [ax_a, ax_b, ax_c0, ax_c1]
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c([ax_c0, ax_c1])
    handles = []
    for center, (disp, _c, mk) in SCHEME.items():
        for ds in ("blobs", "ising"):
            col = SHADE[(center, ds)]
            handles.append(Line2D(
                [0], [0], color=col, marker=mk, ls="none", ms=MS, mew=0.9,
                mfc=col if ds == "blobs" else "none",
                label=f"{disp}, {DS_LABEL[ds]}"))
    # matplotlib fills legends column-first: with ncol=3 this puts one scheme
    # per column, blobs above Ising
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
                     handlelength=1.4, columnspacing=1.8, handletextpad=0.45,
                     labelspacing=0.28, fontsize=LEGEND_FS)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.36, top=0.90,
                        wspace=0.40)
    # Anchor the legend directly under the panels; the tight bounding box then
    # trims the canvas left underneath, so no blank band separates the two.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y0 = min(inv.transform(ax.get_tightbbox(rend).p0)[1] for ax in axes)
    lh = inv.transform_bbox(leg.get_window_extent(rend)).height
    leg.set_bbox_to_anchor((0.5, y0 - 0.048 - lh), transform=fig.transFigure)
    fig.savefig(OUT)
    fig.savefig(HERE / "fig_theory_check_preview.png", dpi=220)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
