#!/usr/bin/env python3
r"""Fig. 1 -- from circuit to branches to branch diversity: three framed panels, (a) and (c) stacked inside.

(a) Top: ancilla-extended IQP circuit H D~_Z H on |0>^{n+a} in register notation (system measured, ancillas
    traced out). Below: D~_Z in the ancilla basis, one block per branch, and the Walsh map of the angles.
(b) Distributions as points in kernel-weighted correlator space: branches on the gray band of single ancilla-free
    circuits, mixture = weighted mean, p_data off the band at distance Delta_anc-free; Proposition 2 as the caption.
(c) Three boxed rows: coincident start (D_branch = 0, no separating gradient), branches that differ but not toward
    the target (Lambda ~ 1), branches aligned with the target (Lambda >> 1).
Schematic, no data. Drawing coordinates are inches.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge
import numpy as np

from _paperstyle import apply as _apply_paper_style

_apply_paper_style()
matplotlib.rcParams["savefig.bbox"] = "standard"        # keep the exact page width

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "fig_1_branch_decomposition.pdf"
OUT_PNG = HERE / "fig_1_preview.png"
W, H = 7.057, 2.35
FS, FS_S, FS_XS = 8.0, 7.2, 6.8
BLUE, GREEN, ORANGE, PURPLE = "#1565a8", "#23823b", "#c46812", "#7b3fa0"
GRAY, TEXT, MUTED = "#8f9499", "#202020", "#5f6368"
BR = [BLUE, GREEN, ORANGE, PURPLE]                        # branch l = 0..3 (a = 2, L = 4)
BAND, FRAME, TILE = "#e3e5e7", "#c4c7cb", "#f4f5f6"
M, GAP = 0.01, 0.08                                       # outer margin, gap between frames
WA, WB = 2.12, 2.35
WC = W - 2 * M - 2 * GAP - WA - WB

fig = plt.figure(figsize=(W, H))


def panel(x0, w):
    """Framed panel; data coordinates are inches from the frame's lower-left corner."""
    fig.add_artist(FancyBboxPatch((x0, M), w, H - 2 * M, boxstyle="round,pad=0,rounding_size=0.06",
                                  transform=fig.dpi_scale_trans, fc="none", ec=FRAME, lw=0.6))
    ax = fig.add_axes([x0 / W, 0, w / W, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, H); ax.set_axis_off()
    return ax


def heading(ax, tag, title):
    ax.text(0.07, H - 0.07, rf"$\bf{{({tag})}}$  {title}", fontsize=FS, ha="left", va="top")


def tile(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=TILE, ec="none",
                                zorder=0))


def arrow(ax, a, b, color=TEXT, lw=0.9, ms=6, z=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def box(ax, x, y, s, label, fs=FS_XS):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc="white", ec=TEXT, lw=0.6, zorder=3))
    ax.text(x, y, label, fontsize=fs, ha="center", va="center", zorder=4)


def meter(ax, x, y):
    ax.add_patch(Rectangle((x - 0.085, y - 0.075), 0.17, 0.15, fc="white", ec=TEXT, lw=0.6, zorder=3))
    t = np.linspace(0.15 * np.pi, 0.85 * np.pi, 30)
    ax.plot(x + 0.058 * np.cos(t), y - 0.04 + 0.065 * np.sin(t), color=TEXT, lw=0.5, zorder=4)
    ax.plot([x, x + 0.042], [y - 0.04, y + 0.05], color=TEXT, lw=0.5, zorder=4)


# ============================================================ (a) circuit over block matrix
A = panel(M, WA)
heading(A, "a", "Ancilla-extended IQP circuit")
YS, YA = 1.90, 1.56                                       # system and ancilla registers
X0, XH1, XD0, XD1, XH2, XEND = 0.60, 0.84, 0.98, 1.26, 1.40, 1.54
for y, reg, lab in ((YS, "n", "system"), (YA, "a", "ancillas")):
    A.text(0.07, y, lab, fontsize=FS_XS, ha="left", va="center", color=MUTED)
    A.text(X0 - 0.03, y, r"$|0\rangle$", fontsize=FS_S, ha="right", va="center")
    A.text(X0 + 0.06, y + 0.06, rf"${reg}$", fontsize=FS_XS, ha="center", va="bottom")
    A.plot([X0, XEND], [y, y], color=TEXT, lw=0.6, zorder=1)
    A.plot([X0 + 0.04, X0 + 0.08], [y - 0.045, y + 0.045], color=TEXT, lw=0.6, zorder=2)   # register slash
for xh in (XH1, XH2):
    for y in (YS, YA):
        box(A, xh, y, 0.17, r"$H$")
A.add_patch(Rectangle((XD0, YA - 0.12), XD1 - XD0, YS - YA + 0.24, fc="#f1f2f3", ec=TEXT, lw=0.7, zorder=3))
A.text((XD0 + XD1) / 2, (YS + YA) / 2, r"$\widetilde D_Z$", fontsize=FS, ha="center", va="center", zorder=4)
meter(A, XEND + 0.09, YS)
for dx, hh in ((0.0, 0.07), (0.032, 0.046), (0.064, 0.022)):                # discard: ancillas are traced out
    A.plot([XEND + dx] * 2, [YA - hh, YA + hh], color=TEXT, lw=0.8, zorder=4)
A.text(XEND + 0.12, YA, "traced out", fontsize=FS_XS, ha="left", va="center")
# D~_Z, block diagonal in the ancilla basis
MS, MX, MY = 0.68, 0.24, 0.20
bs = MS / 4
A.text(0.07, 1.17, r"$\widetilde D_Z$ in the ancilla basis", fontsize=FS_S, ha="left",
       va="center")
A.plot([0.10, WA - 0.10], [1.36, 1.36], color=FRAME, lw=0.5, ls=(0, (2, 2)))
for xb, d in ((MX - 0.02, 0.03), (MX + MS + 0.02, -0.03)):
    A.plot([xb + d, xb, xb, xb + d], [MY - 0.02, MY - 0.02, MY + MS + 0.02, MY + MS + 0.02], color=TEXT, lw=0.7)
for l in range(4):
    x, y = MX + l * bs, MY + MS - (l + 1) * bs
    A.add_patch(Rectangle((x + 0.006, y + 0.006), bs - 0.012, bs - 0.012, fc=BR[l], ec="none", alpha=0.18, zorder=2))
    A.add_patch(Rectangle((x + 0.006, y + 0.006), bs - 0.012, bs - 0.012, fc="none", ec=BR[l], lw=0.6, zorder=3))
    A.text(x + bs / 2, y + bs / 2, rf"$\theta^{{({l})}}$", fontsize=FS_XS, ha="center", va="center", color=BR[l],
           zorder=4)
    A.text(x + bs / 2, MY + MS + 0.06, f"{l:02b}", fontsize=FS_XS, ha="center", va="bottom", color=BR[l])
A.text(MX - 0.05, MY + MS + 0.06, r"$\ell$:", fontsize=FS_XS, ha="right", va="bottom")
A.text(MX + MS + 0.12, MY + MS / 2, r"$\theta^{(\ell)}_G={\sum}_S\,(-1)^{S\cdot\ell}\,\widetilde\theta_{G,S}$",
       fontsize=FS_S, ha="left", va="center")

# ==================================================== correlator geometry
AX_, AY_ = 0.80, 0.62


def ucurve(s, c, sc):
    s = np.asarray(s, float)
    return np.c_[c[0] + sc * AX_ * np.sin(s), c[1] - sc * AY_ * np.cos(s)]


def glyph(ax, x, y, peak, color, w=0.26, h=0.15):
    bins = np.arange(8)
    p = 0.10 + np.exp(-0.5 * ((bins - peak) / 0.9) ** 2)
    p = p / p.max() * h
    bw = w / 8
    for b, v in zip(bins, p):
        ax.add_patch(Rectangle((x + b * bw + 0.1 * bw, y), bw * 0.8, v, fc=color, ec="none", zorder=5))
    ax.plot([x, x + w], [y, y], color=TEXT, lw=0.4, zorder=5)


def draw_M(ax, c, sc, lw=6):
    P = ucurve(np.linspace(-2.05, 2.05, 200), c, sc)
    ax.plot(P[:, 0], P[:, 1], color=BAND, lw=lw, solid_capstyle="round", zorder=1)
    ax.plot(P[:, 0], P[:, 1], color="#b3b7bb", lw=0.5, zorder=1)


def branches(ax, s, c, sc, ms):
    P = ucurve(s, c, sc)
    m = P.mean(0)
    for p, col in zip(P, BR):
        ax.plot([m[0], p[0]], [m[1], p[1]], color=col, lw=0.7, zorder=3)
        ax.plot(*p, "o", ms=ms, color=col, mec="white", mew=0.4, zorder=6)
    ax.plot(*m, "o", ms=ms, color=TEXT, mec="white", mew=0.4, zorder=6)
    return P, m


def star(ax, xy, ms=7):
    ax.plot(*xy, "*", ms=ms, color=GRAY, mec="white", mew=0.4, zorder=6)


def pie(ax, xy, r, z=7):
    for l in range(4):
        ax.add_patch(Wedge(xy, r, 90 * l + 45, 90 * (l + 1) + 45, fc=BR[l], ec="white", lw=0.3, zorder=z))


def nearest(c, sc, q):
    ss = np.linspace(-2.05, 2.05, 2000)
    P = ucurve(ss, c, sc)
    return P[np.argmin(((P - np.asarray(q)) ** 2).sum(1))]


# ================================================ (b) the diversity floor
B = panel(M + WA + GAP, WB)
heading(B, "b", "Beating a single circuit needs diversity")
B.text(WB / 2, 1.92, r"$p_\Theta={\sum}_\ell\,\pi_\ell\,p_{\theta^{(\ell)}}$", fontsize=FS_S, ha="center",
       va="center")
cB, scB = (WB / 2, 1.13), 0.84
draw_M(B, cB, scB)
PB, mB = branches(B, [-1.65, -0.65, 0.65, 1.65], cB, scB, ms=4.8)
dB = (mB[0] + 0.0, mB[1] + 0.12)
star(B, dB, ms=8)
q = nearest(cB, scB, dB)
B.plot([dB[0], q[0]], [dB[1], q[1]], color=MUTED, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
B.text(q[0] + 0.05, q[1] + 0.09, r"$\Delta_{\text{anc-free}}$", fontsize=FS_XS, ha="left",
       va="center", color=MUTED)
for p, c, o, pk in zip(PB, BR, [(-0.36, -0.02), (-0.36, -0.12), (0.10, -0.12), (0.10, -0.02)], [0.6, 2.6, 4.4, 6.4]):
    glyph(B, p[0] + o[0], p[1] + o[1], pk, c)
B.text(mB[0] - 0.07, mB[1] - 0.03, r"$p_\Theta$", fontsize=FS_S, ha="right", va="top", zorder=7)
B.text(dB[0] + 0.08, dB[1] + 0.03, r"$p_{\mathrm{data}}$", fontsize=FS_S, ha="left", va="bottom", color=MUTED,
       zorder=7)
B.text(WB / 2, 0.40, r"single ancilla-free circuits", fontsize=FS_XS, color="#6d7175", ha="center", va="center")
B.text(WB / 2, 0.18, r"gain requires $D_{\mathrm{branch}}\geq\Delta_{\text{anc-free}}-\sqrt{\mathcal{L}}$",
       fontsize=FS_XS, ha="center", va="center")

# ================================================= (c) which branch differences are usable, three boxed rows
C = panel(M + WA + WB + 2 * GAP, WC)
heading(C, "c", "Which branch differences help")
PADX, TH_, TG = 0.07, 0.60, 0.05
XT = 0.95
ROWS = [
    ("coincident", [0.0] * 4, ["data-agnostic or global start:",
                               r"$D_{\mathrm{branch}}=0$, no separating gradient,",
                               r"GD escape time $\propto\ln(1/\delta)$"]),
    ("away", [0.95, 1.3, 1.65, 2.0], ["diverse, not aligned:",
                                       r"$D_{\mathrm{branch}}>0$, $\Lambda\approx1$,",
                                       "removing a branch helps"]),
    ("aligned", [-1.95, -1.35, 1.35, 1.95], ["aligned with the target:",
                                              r"$\Lambda\gg1$, each branch covers",
                                              "a part the others miss"]),
]
for k, (kind, ss, lines) in enumerate(ROWS):
    y0 = 2.0 - (k + 1) * TH_ - k * TG
    tile(C, PADX, y0, WC - 2 * PADX, TH_)
    c, sc = (0.48, y0 + 0.30), 0.36
    draw_M(C, c, sc, lw=3.6)
    dpt = (c[0], c[1] + 0.07)
    star(C, dpt, ms=5.5)
    if kind == "coincident":
        p0 = ucurve([-0.9], c, sc)[0]
        pie(C, p0, 0.04)
    else:
        branches(C, ss, c, sc, ms=3.2)
    for i, t in enumerate(lines):
        C.text(XT, y0 + TH_ - 0.15 - i * 0.15, t, fontsize=FS_XS, ha="left", va="center",
               color=TEXT if i else MUTED)

fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
print("wrote", OUT_PDF)
