"""Matplotlib defaults for the manuscript figures.

Both target styles ask for the same thing. The Physical Review style guide
suggests Times Roman or Arial lettering, with the smallest capitals at least
2 mm tall and curves at least 0.5 pt wide; Nature Portfolio (npj) asks for a
sans-serif face, preferably Helvetica or Arial, used identically across every
figure, with text between 5 and 7 pt and 8 pt bold panel letters. We therefore
set a Times-metric serif face matching the REVTeX body text, with STIX
mathtext, and sizes that clear the APS 2 mm requirement at the printed size: figures are included at their natural width,
so a label set at 8.5 pt here prints at 8.5 pt.

Usage, before creating any figure:

    from _paperstyle import apply
    apply()                 # or apply(base=8.0)
"""
from __future__ import annotations

import matplotlib


BASE = 8.5          # axis labels, titles, legend entries
# Physical Review asks for figure lettering in the text face: REVTeX sets the
# body in Times, so the figures use Times metrics (Nimbus Roman and TeX Gyre
# Termes are the URW and GUST clones) with the matching STIX mathtext.
SERIF = ["Nimbus Roman", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"]


def apply(base: float = BASE) -> None:
    small = base - 0.5
    matplotlib.rcParams.update({
        # Times-metric serif, matching the REVTeX body text
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        # sizes, in points of the printed figure
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": small,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        # rules and curves stay above the 0.5 pt floor
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })
