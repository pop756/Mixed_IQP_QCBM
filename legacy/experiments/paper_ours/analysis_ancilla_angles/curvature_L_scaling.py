#!/usr/bin/env python3
"""Eq. (rand-curv) / Theorem 1 numerically: the 1/L^2 center curvature.

At the data-agnostic center the theorem predicts, for a one-body generator
alpha and any single branch l,

    c = |d^2 L / d theta_alpha^(l)^2| = 8 w_{alpha} / L^2,      L = 2^a,

i.e. the mixture reduces the single-circuit curvature by 1/L^2.  A single
branch's angle is moved by the compiled-coordinate direction

    v_{alpha,S} = (1/L) (-1)^{|l & S|}        (inverse Walsh transform),

so this script walks the exact loss along v with an ABSOLUTE step (radians),
reports L(t), the directional gradient dL/dt and the center curvature, and
compares the curvature with 8 w_{alpha}/L^2 for a = 0..3.  It also sweeps the
same absolute uniform noise on every angle and records the full, visible and
ancilla gradient norms, so the L dependence can be read off at fixed absolute
perturbation.  Exact statevector throughout (x64), PennyLane-validated.

Writes curvature_L_scaling.csv and uniform_noise_absolute.csv.
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
import exact_iqp  # noqa: E402

DATASETS = ["blobs", "ising"]
ANCILLAS = [0, 1, 2, 3]
CENTERS = [("S4", "s4_unbiased", "data-agnostic"),
           ("S1_visible", "s1_visible", "data-dependent")]
STEPS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]      # absolute, radians
ABS_NOISE = [0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
K_DRAWS = 8
CFG_SEED = 7
ALPHA = 0          # the one-body generator used as the witness
BRANCH = 0         # branch l whose angle is moved


def center_of(ds, scheme_key, stem, a):
    built = A.build_run(ds, scheme_key, stem, a, CFG_SEED)
    trainer = built["trainer"]
    from iqpopt.utils import initialize_from_data
    p_saved = trainer.qcbm.get("noise_m_exp")
    trainer.qcbm["noise_m_exp"] = 60.0
    init_art = trainer._load_init_artifact(built["layout_name"])
    prev_g = prev_p = None
    if init_art is not None:
        prev_g = init_art[1]["gates"]
        prev_p = np.asarray(init_art[1]["params"], dtype=np.float64)
    center = np.asarray(trainer._initial_params(
        initialize_from_data=initialize_from_data, gates=built["gates"],
        visible_bits=trainer.visible_bits, previous_gates=prev_g,
        previous_params=prev_p, initialize_missing_visible=init_art is None,
    ), dtype=np.float64)
    trainer.qcbm["noise_m_exp"] = p_saved
    return built, center


def branch_direction(gates, n, a, alpha, branch):
    """Compiled-coordinate direction that moves only theta_alpha^(branch)."""
    L = 2 ** a
    v = np.zeros(len(gates), dtype=np.float64)
    for i, g in enumerate(gates):
        vis = sorted({w for gen in g for w in gen if w < n})
        if vis != [alpha]:
            continue
        S = [w - n for gen in g for w in gen if w >= n]
        sign = -1.0 if sum((branch >> s) & 1 for s in S) % 2 else 1.0
        v[i] = sign / L
    assert np.count_nonzero(v) == L, (np.count_nonzero(v), L)
    return v


def one_body_weight(n, sigmas):
    """w_{alpha} = mean_sigma P_sigma(A) for |A| = 1 in the exact loss."""
    ws = []
    for sig in [float(s) for s in sigmas]:
        p = (1.0 - np.exp(-1.0 / 2.0 / sig ** 2)) / 2.0
        ws.append(p * (1.0 - p) ** (n - 1))
    return float(np.mean(ws))


def main():
    import jax
    import jax.numpy as jnp
    rows_c, rows_u, validated = [], [], False
    for ds in DATASETS:
        for a in ANCILLAS:
            for scheme_key, stem, label in CENTERS:
                built, center = center_of(ds, scheme_key, stem, a)
                trainer = built["trainer"]
                n, gates, m = trainer.visible_bits, built["gates"], len(built["gates"])
                L = 2 ** a
                vg, loss_fn, corr = exact_iqp.build_exact_loss(
                    gates, n, a, trainer.train_bits, trainer.qcbm["sigmas"])
                if not validated:
                    exact_iqp.validate_probs(gates, n, a, trainer.qcbm, corr, center)
                    validated = True
                mask = A.sector_mask(gates, n)
                v = branch_direction(gates, n, a, ALPHA, BRANCH)
                vj, cj = jnp.asarray(v), jnp.asarray(center)

                # exact center curvature along v: d/dt (grad . v) at t = 0
                def dir_grad(t):
                    return jnp.vdot(jax.grad(loss_fn)(cj + t * vj), vj)
                curv = float(jax.grad(dir_grad)(0.0))
                w1 = one_body_weight(n, trainer.qcbm["sigmas"])
                pred = 8.0 * w1 / L ** 2
                print(f"[curv] {ds} a={a} {label:14s} L={L}: measured={curv:.6e} "
                      f"predicted 8w/L^2={pred:.6e} ratio={curv/pred:.4f}", flush=True)
                for t in STEPS:
                    val, g = vg(cj + t * vj)
                    rows_c.append(dict(dataset=ds, ancilla=a, L=L, center=label,
                                       n_params=m, step=t, w_one_body=w1,
                                       curv_measured=curv, curv_pred_8w_L2=pred,
                                       loss=float(val),
                                       dir_grad=float(np.dot(np.asarray(g), v)),
                                       gnorm=float(np.linalg.norm(np.asarray(g)))))
                # same ABSOLUTE uniform noise on every angle
                for d in ABS_NOISE:
                    Ls, G = [], []
                    for k in range(K_DRAWS if d > 0 else 1):
                        rng = np.random.default_rng(10_000 + k)
                        th = center + rng.uniform(-d, d, size=m)
                        val, g = vg(jnp.asarray(th))
                        Ls.append(float(val)); G.append(np.asarray(g, dtype=np.float64))
                    G = np.array(G)
                    rows_u.append(dict(
                        dataset=ds, ancilla=a, L=L, center=label, n_params=m,
                        noise_abs=d, draws=len(Ls),
                        loss_mean=float(np.mean(Ls)),
                        loss_shift=float(np.mean(Ls)) - float(Ls[0] if d == 0 else np.nan)
                        if d == 0 else float("nan"),
                        loss_var=float(np.var(Ls, ddof=1)) if len(Ls) > 1 else 0.0,
                        gnorm=float(np.mean(np.linalg.norm(G, axis=1))),
                        grms_perparam=float(np.mean(np.linalg.norm(G, axis=1)) / np.sqrt(m)),
                        gnorm_vis=float(np.mean(np.linalg.norm(G[:, ~mask], axis=1))),
                        gnorm_anc=float(np.mean(np.linalg.norm(G[:, mask], axis=1)))
                        if mask.any() else 0.0))
                    print(f"[unif] {ds} a={a} {label:14s} noise={d:<6g} "
                          f"L={rows_u[-1]['loss_mean']:.6f} |g|={rows_u[-1]['gnorm']:.4e} "
                          f"|g_anc|={rows_u[-1]['gnorm_anc']:.4e}", flush=True)
    for fname, rows in (("curvature_L_scaling.csv", rows_c),
                        ("uniform_noise_absolute.csv", rows_u)):
        with open(HERE / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("wrote", HERE / fname)


if __name__ == "__main__":
    main()
