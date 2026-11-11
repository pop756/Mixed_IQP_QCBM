#!/usr/bin/env python3
"""Theorem 2 numerically: c >= 8 w_{alpha} pi_l^2 (1 - (t_alpha^(l))^2) - O(1/n^3).

At the data-dependent center (cos 2 theta_j^(l) = t_j^(l), theta_jk^(l) = 0)
the theorem lower-bounds the center curvature at the witness coordinate alpha
of branch l*.  The statement is an existence claim about a witness, so we test
the stronger version: the bound is evaluated at EVERY one-body coordinate
alpha = 0..n-1 and EVERY branch l, and the minimum ratio over all of them is
reported.  The curvature is the exact second derivative along the compiled
direction that moves theta_alpha^(l) alone,

    v_{alpha,S} = (1/L) (-1)^{|l & S|},

t_alpha^(l) = cos(2 theta_alpha^(l)) is read off the center itself, and the
weights pi_l are the ones the scheme trains with (uniform for the global
center, cluster sizes for the cluster-initialized center).

Writes theorem2_bound_check.csv.
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
DATASETS = ["blobs", "ising"]
CENTERS = [("S1_visible", "s1_visible", "global", [0, 1, 2, 3]),
           ("S2_cluster", "s2_cluster", "cluster", [1, 2, 3])]
CFG_SEED = 7


def center_of(ds, key, stem, a):
    built = A.build_run(ds, key, stem, a, CFG_SEED)
    trainer = built["trainer"]
    from iqpopt.utils import initialize_from_data
    saved = trainer.qcbm.get("noise_m_exp")
    trainer.qcbm["noise_m_exp"] = 60.0
    art = trainer._load_init_artifact(built["layout_name"])
    pg = pp = None
    if art is not None:
        pg = art[1]["gates"]
        pp = np.asarray(art[1]["params"], dtype=np.float64)
    center = np.asarray(trainer._initial_params(
        initialize_from_data=initialize_from_data, gates=built["gates"],
        visible_bits=trainer.visible_bits, previous_gates=pg, previous_params=pp,
        initialize_missing_visible=art is None), dtype=np.float64)
    trainer.qcbm["noise_m_exp"] = saved
    return built, center


def main():
    import jax
    import jax.numpy as jnp
    rows = []
    for ds in DATASETS:
        for key, stem, label, ancillas in CENTERS:
            for a in ancillas:
                L = 2 ** a
                built, center = center_of(ds, key, stem, a)
                tr = built["trainer"]
                n, gates = tr.visible_bits, built["gates"]
                if label == "cluster":
                    lab = (EXP / f"{ds}_seed" /
                           f"dataset_s2_cluster_a{a}_usqm_seed{CFG_SEED}" / "y_train.csv")
                    pi = EW.cluster_weights(lab, L) if lab.exists() else np.full(L, 1 / L)
                else:
                    pi = np.full(L, 1.0 / L)
                _, loss_fn = EW.build_exact_loss_weighted(
                    gates, n, a, tr.train_bits, tr.qcbm["sigmas"], pi)
                B, vis_gens = EW.branch_matrix(gates, n, a)
                th_branch = np.einsum("lgm,m->lg", B, center)   # (L, n_gen)
                gen_of = {tuple(g): i for i, g in enumerate(vis_gens)}
                sig = [float(s) for s in tr.qcbm["sigmas"]]
                w1 = float(np.mean([(lambda p: p * (1 - p) ** (n - 1))(
                    (1 - np.exp(-1 / 2 / s ** 2)) / 2) for s in sig]))
                c0 = jnp.asarray(center)
                worst = (np.inf, None)
                for alpha in range(n):
                    gi = gen_of[(alpha,)]
                    v = np.zeros(len(gates))
                    for i, g in enumerate(gates):
                        if sorted({w for gen in g for w in gen if w < n}) != [alpha]:
                            continue
                        S = [w - n for gen in g for w in gen if w >= n]
                        v[i] = 1.0 / L
                        sgn_idx = i
                        del sgn_idx
                    for l in range(L):
                        vv = np.zeros(len(gates))
                        for i, g in enumerate(gates):
                            if sorted({w for gen in g for w in gen if w < n}) != [alpha]:
                                continue
                            S = [w - n for gen in g for w in gen if w >= n]
                            sign = -1.0 if sum((l >> s) & 1 for s in S) % 2 else 1.0
                            vv[i] = sign / L
                        vj = jnp.asarray(vv)

                        def dir_grad(t, vj=vj):
                            return jnp.vdot(jax.grad(loss_fn)(c0 + t * vj), vj)
                        curv = float(jax.grad(dir_grad)(0.0))
                        t_al = float(np.cos(2.0 * th_branch[l, gi]))
                        bound = 8.0 * w1 * pi[l] ** 2 * (1.0 - t_al ** 2)
                        ratio = curv / bound if bound > 0 else np.inf
                        rows.append(dict(dataset=ds, center=label, ancilla=a, L=L,
                                         alpha=alpha, branch=l, pi=float(pi[l]),
                                         t_alpha=t_al, w_one_body=w1,
                                         curv=curv, bound=bound, ratio=ratio,
                                         slack_1_over_n3=1.0 / n ** 3))
                        if ratio < worst[0]:
                            worst = (ratio, (alpha, l, curv, bound))
                print(f"[thm2] {ds} {label:8s} a={a} L={L}: min ratio={worst[0]:.4f} "
                      f"at alpha={worst[1][0]} branch={worst[1][1]} "
                      f"(c={worst[1][2]:.4e}, bound={worst[1][3]:.4e}); "
                      f"1/n^3={1/n**3:.3e}", flush=True)
    with open(HERE / "theorem2_bound_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", HERE / "theorem2_bound_check.csv")


if __name__ == "__main__":
    main()
