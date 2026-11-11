#!/usr/bin/env python3
r"""fig_app_assumption -- finite-size grouped-data residual diagnostics.

At body order k, the group-weighted residual summary eps_k is reported as the
finite-size effective constant

    C_{k,eff} = n eps_k^{2/k}.

Columns are the four benchmarks; C_{k,eff} runs against the number of groups
L = 2^a of the cluster initialization, with L = 1 the ungrouped training set.
The line C_{k,eff}=n marks q_{k,eff}=1. These median and 95th-percentile
summaries diagnose typical finite-size behavior; they are not the uniform C_n
of Assumption 1 and do not establish its scaling with n.

  top row    (a)-(d): C_{k,eff} from the group-weighted median residual
  bottom row (e)-(h): C_{k,eff} from its 95th percentile over subsets

Dashed lines of the same colour are the finite-sample floor of each statistic:
the same quantity measured on independent bits with the group's own marginals and
size, so a curve at its floor is indistinguishable from a product distribution at
that sample size.

Data (read-only): experiments/cosine_rerun/results/factorization_residuals.csv
Run: python make_fig_app_assumption.py -> fig_app_assumption.pdf
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()

from _paths import PAPER_OURS

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "fig_app_assumption.pdf"
OUT_PNG = HERE / "fig_app_assumption_preview.png"
# all four benchmarks on the rerun's full splits and groups (experiments/cosine_rerun/factorization.py)
from _seed_agg import RESULTS  # noqa: E402
CSV = RESULTS / "factorization_residuals.csv"

PANELS = [("blobs", "blobs"), ("ising", "Ising"),
          ("mnist", "MNIST"), ("dwave", "D-Wave")]
BODIES = [(2, "#440154", "o"), (3, "#414487", "s"), (4, "#2a788e", "^"),
          (5, "#22a884", "v"), (6, "#5ec962", "D")]
ROWS = [("rho", "rho_fl", "median"), ("rho95", "rho95_fl", r"$95$th percentile")]
TAGS = "abcdefgh"

# Requested manuscript-figure font scale. The wide canvas is reduced to
# \textwidth in LaTeX, giving journal-sized lettering after scaling.
TITLE_FONTSIZE = 22.62
TAG_FONTSIZE = 23.4
ALGO_FONTSIZE = 17.68
XTICK_FONTSIZE = 19.24
YTICK_FONTSIZE = 18.46
YLABEL_FONTSIZE = 21.06
LEGEND_FONTSIZE = 20.28
SUPXLABEL_FONTSIZE = 20.54


def main() -> None:
    df = pd.read_csv(CSV)
    e = 2.0 / df.k
    df["rho"] = df.n * df.eps ** e
    df["rho_fl"] = df.n * df.eps_null ** e
    df["rho95"] = df.n * df.eps_q95 ** e
    df["rho95_fl"] = df.n * df.eps_null_q95 ** e

    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.6), sharey=True)
    lo = float(df[["rho", "rho_fl", "rho95", "rho95_fl"]].to_numpy().min())
    hi = max(float(df.n.max()),
             float(df[["rho", "rho_fl", "rho95", "rho95_fl"]].to_numpy().max()))

    for ri, (key, fkey, rlab) in enumerate(ROWS):
        for pi, (ax, (ds, disp)) in enumerate(zip(axes[ri], PANELS)):
            d = df[df.dataset == ds].sort_values("L")
            n = float(d.n.iloc[0])
            ax.axhline(n, color="black", ls=":", lw=2.0, zorder=2)
            for k, col, mk in BODIES:
                r = d[d.k == k]
                ax.plot(r.L, r[key], color=col, marker=mk, ms=7.5, lw=2.7,
                        mfc=col, zorder=4)
                ax.plot(r.L, r[fkey], color=col, ls="--", lw=2.0, alpha=0.75,
                        zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            Ls = sorted(d.L.unique())
            ax.set_xticks(Ls)
            ax.set_xticklabels([rf"${int(v)}$" for v in Ls])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(Ls[0] / 1.5, Ls[-1] * 1.5)
            ax.set_ylim(lo * 0.55, hi * 1.65)
            ax.grid(alpha=0.25, which="both", lw=1.0)
            ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE,
                           length=6.0, width=1.3, pad=4)
            ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE,
                           length=6.0, width=1.3, pad=4)
            ax.tick_params(which="minor", length=3.5, width=1.0)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.text(0.0, 1.04, rf"$\bf{{({TAGS[4 * ri + pi]})}}$",
                    transform=ax.transAxes, fontsize=TAG_FONTSIZE,
                    va="bottom", ha="left")
            ax.text(0.18, 1.04, disp, transform=ax.transAxes,
                    fontsize=TITLE_FONTSIZE, va="bottom", ha="left")
            ax.text(0.97, n * 1.08, r"$C_{k,\mathrm{eff}}=n$",
                    transform=ax.get_yaxis_transform(), fontsize=ALGO_FONTSIZE,
                    va="bottom", ha="right")
            if pi == 0:
                ax.text(0.03, 0.94, rlab, transform=ax.transAxes,
                        fontsize=ALGO_FONTSIZE, va="top", ha="left")
        axes[ri, 0].set_ylabel(r"$C_{k,\mathrm{eff}}=n\epsilon_k^{2/k}$",
                               fontsize=YLABEL_FONTSIZE, labelpad=7)
    supx = fig.supxlabel(r"number of groups $L$", fontsize=SUPXLABEL_FONTSIZE)

    handles = [Line2D([0], [0], color=c, marker=m, ms=7.5, lw=2.7,
                      label=rf"$k={k}$") for k, c, m in BODIES]
    handles.append(Line2D([0], [0], color="0.35", ls="--", lw=2.0,
                          label="independent-bit reference"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=6,
                     bbox_to_anchor=(0.5, 0.01), frameon=False,
                     fontsize=LEGEND_FONTSIZE, handlelength=2.0,
                     columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0), pad=1.0,
                     w_pad=1.5, h_pad=2.2)
    # Anchor the shared label and the legend directly under the bottom row; the
    # tight bounding box then trims the canvas left underneath, so no blank
    # band survives between the panels and the legend.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    y0 = min(inv.transform(ax.get_tightbbox(fig.canvas.get_renderer()).p0)[1]
             for ax in axes[1])
    supx.set_y(y0 - 0.040)
    leg.set_bbox_to_anchor((0.5, y0 - 0.102), transform=fig.transFigure)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)

    for ds, _ in PANELS:
        d = df[df.dataset == ds]
        top = d[d.L == d.L.max()]
        arg = top.loc[top.rho.idxmax()]
        g = d[(d.k == 2) & (d.L == 1)].iloc[0]
        t = top[top.k == 2].iloc[0]
        print(f"[{ds}] L={int(top.L.iloc[0])}: "
              f"max_k C_eff={top.rho.max():.3f} "
              f"(k={int(arg.k)}), q95 {top.rho95.max():.3f}; "
              f"eps_2 x{g.eps / t.eps:.1f}; "
              f"worst sampled C_eff "
              f"{max(r.n * r.eps_max ** (2 / r.k) for _, r in top.iterrows()):.3f}")
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
