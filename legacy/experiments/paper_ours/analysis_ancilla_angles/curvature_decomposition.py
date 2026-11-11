#!/usr/bin/env python3
"""Split the center curvature into its two blocks (Theorem 2, Eq. bpr-d2).

    d^2 L / d theta_alpha^(l)^2
      = sum_{A ni alpha} 2 w_A pi_l^2 (g_A^{(alpha,l)})^2      [sensitivity]
      + 8 pi_l sum_{A ni alpha} w_A z_A(theta^(l)) (t_A - z_A(Theta))  [mismatch]

At the factorizing center (two-body angles zero) z_A(theta^(l)) = prod_{j in A}
t_j^(l) and g_A^{(alpha,l)} = -2 sin(2 theta_alpha^(l)) prod_{j in A\\alpha}
t_j^(l), so both blocks are evaluated exactly over all 2^n subsets.  The sum of
the two is checked against the exact second derivative measured by
check_theorem2_bound.py.

Writes curvature_decomposition.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ["IQPOPT_X64"] = "1"
os.environ["JAX_ENABLE_X64"] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ancilla_angle_grad_analysis as A  # noqa: E402
import exact_iqp_weighted as EW  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
CFG_SEED = 7
CENTERS = {"global": ("S1_visible", "s1_visible"),
           "cluster": ("S2_cluster", "s2_cluster")}


def center_of(ds, key, stem, a):
    built = A.build_run(ds, key, stem, a, CFG_SEED)
    tr = built["trainer"]
    from iqpopt.utils import initialize_from_data
    saved = tr.qcbm.get("noise_m_exp")
    tr.qcbm["noise_m_exp"] = 60.0
    art = tr._load_init_artifact(built["layout_name"])
    pg = pp = None
    if art is not None:
        pg = art[1]["gates"]
        pp = np.asarray(art[1]["params"], dtype=np.float64)
    c = np.asarray(tr._initial_params(
        initialize_from_data=initialize_from_data, gates=built["gates"],
        visible_bits=tr.visible_bits, previous_gates=pg, previous_params=pp,
        initialize_missing_visible=art is None), dtype=np.float64)
    tr.qcbm["noise_m_exp"] = saved
    return built, c


def subset_products(t, n, skip=None):
    """prod_{j in A} t_j for every subset A (bit j of the index = wire j),
    with wire `skip` forced to contribute 1."""
    out = np.ones(2 ** n, dtype=np.float64)
    idx = np.arange(2 ** n)
    for j in range(n):
        if j == skip:
            continue
        bit = (idx >> (n - 1 - j)) & 1
        out[bit == 1] *= t[j]
    return out


def wht(v):
    v = v.copy()
    n = int(np.log2(v.size))
    for j in range(n):
        v = v.reshape(2 ** j, 2, -1)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :], v[:, 1, :] = a + b, a - b
        v = v.reshape(-1)
    return v


def main():
    rows = []
    W = {(r["dataset"], r["center"], int(r["L"])): r
         for r in csv.DictReader(open(HERE / "theorem2_witness.csv"))}
    for ds in ("blobs", "ising"):
        for cen, (key, stem) in CENTERS.items():
            for a in ((0, 1, 2, 3) if cen == "global" else (1, 2, 3)):
                L = 2 ** a
                w = W.get((ds, cen, L))
                if w is None:
                    continue
                alpha, ell = int(w["alpha"]), int(w["branch"])
                built, center = center_of(ds, key, stem, a)
                tr = built["trainer"]
                n, gates = tr.visible_bits, built["gates"]
                if cen == "cluster":
                    lab = (EXP / f"{ds}_seed" /
                           f"dataset_s2_cluster_a{a}_usqm_seed{CFG_SEED}" / "y_train.csv")
                    pi = (EW.cluster_weights(lab, L) if lab.exists()
                          else np.full(L, 1 / L))
                else:
                    pi = np.full(L, 1.0 / L)
                B, gens = EW.branch_matrix(gates, n, a)
                th = np.einsum("lgm,m->lg", B, center)
                gi = {tuple(g): i for i, g in enumerate(gens)}
                tj = np.array([[np.cos(2 * th[l, gi[(j,)]]) for j in range(n)]
                               for l in range(L)])
                # branch and mixture correlators over all subsets
                zl = np.stack([subset_products(tj[l], n) for l in range(L)])
                zmix = np.einsum("l,li->i", pi, zl)
                # data correlators
                data = np.asarray(tr.train_bits, dtype=np.int64)
                hist = np.bincount((data * (2 ** np.arange(n - 1, -1, -1))).sum(axis=1),
                                   minlength=2 ** n).astype(np.float64) / len(data)
                tA = wht(hist)
                # sensitivity of the witness branch
                sin2 = np.sqrt(max(0.0, 1.0 - tj[ell, alpha] ** 2))
                gA = -2.0 * sin2 * subset_products(tj[ell], n, skip=alpha)
                has_alpha = (((np.arange(2 ** n) >> (n - 1 - alpha)) & 1) == 1)
                sens = mis = 0.0
                for sig in [float(s) for s in tr.qcbm["sigmas"]]:
                    p = (1.0 - np.exp(-1.0 / 2.0 / sig ** 2)) / 2.0
                    bits = ((np.arange(2 ** n)[:, None]
                             >> np.arange(n - 1, -1, -1)[None, :]) & 1)
                    logw = (bits * np.log(p) + (1 - bits) * np.log1p(-p)).sum(axis=1)
                    wA = np.exp(logw)
                    sens += float(np.sum(2 * wA[has_alpha] * pi[ell] ** 2
                                         * gA[has_alpha] ** 2))
                    mis += float(8 * pi[ell] * np.sum(
                        wA[has_alpha] * zl[ell][has_alpha]
                        * (tA[has_alpha] - zmix[has_alpha])))
                nsig = len(tr.qcbm["sigmas"])
                sens, mis = sens / nsig, mis / nsig
                meas = float(w["curv"])
                rows.append(dict(
                    dataset=ds, center=cen, ancilla=a, L=L, alpha=alpha,
                    branch=ell, pi=float(w["pi"]), t_alpha=float(w["t_alpha"]),
                    sensitivity=sens, mismatch=mis, sum=sens + mis,
                    curv_measured=meas,
                    closure_rel=abs(sens + mis - meas) / abs(meas),
                    bound=float(w["bound"]),
                    mismatch_over_sens=mis / sens,
                    mismatch_abs_over_bound=abs(mis) / float(w["bound"])))
                print(f"[decomp] {ds} {cen:7s} L={L}: sens={sens:.4e} "
                      f"mis={mis:+.3e} sum={sens+mis:.4e} meas={meas:.4e} "
                      f"closure={rows[-1]['closure_rel']:.1e} "
                      f"|mis|/sens={abs(mis)/sens:.4f}", flush=True)
    with open(HERE / "curvature_decomposition.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print("wrote", HERE / "curvature_decomposition.csv")


if __name__ == "__main__":
    main()
