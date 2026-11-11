#!/usr/bin/env python3
"""Witness of Theorem 2 for every center: the coordinate with the largest bound.

check_theorem2_bound.py evaluates the exact curvature and the bound 8 w pi_l^2 (1 - t^2) at every one-body
coordinate alpha and every branch l (theorem2_bound_check.csv). The theorem is an existence statement, and the
witness used by curvature_decomposition.py, by theory_numerics.py and in the article is, for each
(dataset, center, L), the row of that table with the largest bound (the first one when several are equal).

Writes theorem2_witness.csv.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = ["dataset", "center", "L", "ancilla", "branch", "alpha", "pi", "t_alpha", "curv", "bound", "ratio"]

rows = list(csv.DictReader(open(HERE / "theorem2_bound_check.csv")))
groups = sorted({(r["dataset"], r["center"], int(r["L"])) for r in rows})
with open(HERE / "theorem2_witness.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for ds, center, L in groups:
        grp = [r for r in rows if (r["dataset"], r["center"], int(r["L"])) == (ds, center, L)]
        w.writerow(max(grp, key=lambda r: float(r["bound"])))        # max() keeps the first of equal maxima
print("wrote", HERE / "theorem2_witness.csv", f"({len(groups)} rows)")
