#!/usr/bin/env python3
"""Step-0 gradients of the branch-coincident blobs runs (a=1) against the size
of the symmetry-breaking init noise.

The two coincident schemes (S1_visible = global, S4 = data-agnostic) start with
every ancilla-coupling angle at zero plus one Uniform(-r, r) draw, r =
(pi/2)/m**0.5 with m the parameter count. This script rebuilds that init through
the trainer's own code path, splits it into

    theta(c) = center + c * noise,     c = 0, ..., 3

(c = 1 reproduces the run exactly, c = 0 is exact branch coincidence), and
evaluates the loss gradient in both sectors:

  --mode exact : PennyLane/statevector-exact loss.  The visible marginal of the
                 (n+a)-qubit IQP state is computed with a Walsh-Hadamard
                 transform, its correlators z_A are read off a second transform,
                 and the loss is the exact op-average
                 L = mean_sigma sum_A P_sigma(A) (z_A - t_A)^2, i.e. the
                 expectation of the training estimator over its op draw and its
                 sampling.  Requires x64; validated against IqpSimulator.probs
                 (PennyLane, lightning.qubit) on the first configuration.

  --mode est   : the estimator the optimizer actually sees, iqpopt's
                 gen.mmd_loss_iqp with the run's n_ops / n_samples / sigmas,
                 evaluated on several PRNG keys so the spread of the estimate is
                 visible next to the exact value.

Writes noise_grad_blobs_<mode>.csv.  Read-only on all experiment artifacts.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

P = argparse.ArgumentParser()
P.add_argument("--mode", choices=["exact", "est"], required=True)
P.add_argument("--ds", default="blobs")
P.add_argument("--anc", type=int, default=1)
P.add_argument("--seeds", default="7,1234,2025")
P.add_argument("--schemes", default="S1_visible,S4")
P.add_argument("--scales", default="0,0.01,0.03,0.1,0.3,1,3")
P.add_argument("--keys", type=int, default=8, help="estimator PRNG keys (est mode)")
args = P.parse_args()

if args.mode == "exact":
    os.environ["IQPOPT_X64"] = "1"
    os.environ["JAX_ENABLE_X64"] = "1"

import ancilla_angle_grad_analysis as A  # noqa: E402

DS, ANC = args.ds, args.anc
STEM = {"S1_visible": "s1_visible", "S4": "s4_unbiased", "S2_cluster": "s2_cluster"}


def build(scheme_key, seed):
    """Rebuilt run plus the noise-free center of its initialization."""
    built = A.build_run(DS, scheme_key, STEM[scheme_key], ANC, seed)
    trainer = built["trainer"]
    theta_run = np.asarray(built["params_init"], dtype=np.float64)
    # same code path with the uniform radius r=(pi/2)/m**p driven to zero
    p_saved = trainer.qcbm.get("noise_m_exp")
    trainer.qcbm["noise_m_exp"] = 60.0   # r = (pi/2)/m**60 -> 0 in float64
    from iqpopt.utils import initialize_from_data
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
    built["center"] = center
    built["noise"] = theta_run - center
    built["mask_anc"] = A.sector_mask(built["gates"], trainer.visible_bits)
    return built


def branch_spread(theta, gates, n):
    """max_l max_G |theta^(l)_G - theta_bar_G| for the L=2**ANC branches."""
    keys, subsets = [], []
    for g in gates:
        vis = tuple(sorted({w for gen in g for w in gen if w < n}))
        anc = tuple(sorted({w - n for gen in g for w in gen if w >= n}))
        keys.append(vis)
        subsets.append(anc)
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    dev = np.zeros((2 ** ANC, len(uniq)), dtype=np.float64)
    for i, (k, S) in enumerate(zip(keys, subsets)):
        if not S:
            continue
        for l in range(2 ** ANC):
            sign = -1.0 if sum((l >> s) & 1 for s in S) % 2 else 1.0
            dev[l, uniq[k]] += sign * theta[i]
    return float(np.max(np.abs(dev)))


def sector_stats(g, m):
    gv, ga = g[~m], g[m]
    return dict(gnorm_vis=float(np.linalg.norm(gv)),
                gnorm_anc=float(np.linalg.norm(ga)),
                gmax_vis=float(np.max(np.abs(gv))),
                gmax_anc=float(np.max(np.abs(ga))),
                gmean_vis=float(np.mean(np.abs(gv))),
                gmean_anc=float(np.mean(np.abs(ga))),
                gratio_norm=float(np.linalg.norm(ga) / np.linalg.norm(gv)))


# ---------------------------------------------------------------- exact mode
def make_exact_loss(built, trainer):
    import jax
    import jax.numpy as jnp

    n = trainer.visible_bits
    N = n + ANC
    gates = built["gates"]
    # signs s_G(y) = (-1)^{sum_{j in G} y_j} for every basis state y, per gate
    idx = np.arange(2 ** N)
    bits = ((idx[:, None] >> np.arange(N - 1, -1, -1)[None, :]) & 1).astype(np.int8)
    S = np.empty((len(gates), 2 ** N), dtype=np.float64)
    for i, g in enumerate(gates):
        wires = sorted({w for gen in g for w in gen})
        S[i] = 1.0 - 2.0 * (bits[:, wires].sum(axis=1) % 2)
    S = jnp.asarray(S)

    def wht(t, axes):
        for ax in axes:
            t = jnp.moveaxis(t, ax, 0)
            a, b = t[0], t[1]
            t = jnp.moveaxis(jnp.stack([a + b, a - b], axis=0), 0, ax)
        return t

    shape_full = (2,) * N
    shape_vis = (2,) * n

    def correlators(theta):
        phase = theta @ S                       # sum_G theta_G s_G(y)
        f = jnp.exp(-1j * phase).reshape(shape_full)
        psi = wht(f, range(N)) / (2.0 ** N)
        p = jnp.abs(psi) ** 2
        p_vis = jnp.sum(p.reshape(shape_vis + (2 ** ANC,)), axis=-1)
        return wht(p_vis, range(n)).reshape(-1), p_vis.reshape(-1)

    # data correlators, exactly as the estimator's tr_train (empirical)
    data = np.asarray(trainer.train_bits, dtype=np.int64)
    hist = np.bincount((data * (2 ** np.arange(n - 1, -1, -1))).sum(axis=1),
                       minlength=2 ** n).astype(np.float64) / len(data)
    t_A = np.asarray(wht(jnp.asarray(hist).reshape(shape_vis), range(n))).reshape(-1)
    t_A = jnp.asarray(t_A)

    # op weights P_sigma(A) = prod_j p^{A_j} (1-p)^{1-A_j}
    A_bits = ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1)
    W = []
    for sig in [float(s) for s in trainer.qcbm["sigmas"]]:
        p_mmd = (1.0 - np.exp(-1.0 / 2.0 / sig ** 2)) / 2.0
        logw = (A_bits * np.log(p_mmd) + (1 - A_bits) * np.log1p(-p_mmd)).sum(axis=1)
        W.append(jnp.asarray(np.exp(logw)))

    def loss(theta):
        z, _ = correlators(theta)
        d2 = (z - t_A) ** 2
        return sum(jnp.sum(w * d2) for w in W) / len(W)

    return jax.jit(jax.value_and_grad(loss)), correlators


def validate_against_pennylane(built, trainer, correlators, theta):
    """IqpSimulator.probs (PennyLane, lightning.qubit) vs the transform above."""
    import jax.numpy as jnp
    from iqpopt import IqpSimulator
    q = trainer.qcbm
    circ = IqpSimulator(trainer.visible_bits + ANC, built["gates"],
                        device=str(q.get("device", "lightning.qubit")),
                        spin_sym=bool(q.get("spin_sym", False)),
                        sparse=bool(q.get("sparse", False)),
                        bitflip=bool(q.get("bitflip", False)))
    p_pl = np.asarray(circ.probs(jnp.asarray(theta)), dtype=np.float64)
    p_pl_vis = p_pl.reshape(-1, 2 ** ANC).sum(axis=1)
    _, p_mine = correlators(jnp.asarray(theta))
    p_mine = np.asarray(p_mine, dtype=np.float64)
    err = float(np.max(np.abs(p_pl_vis - p_mine)))
    print(f"[validate] max |p_pennylane - p_transform| = {err:.3e} "
          f"(sum {p_pl_vis.sum():.6f} / {p_mine.sum():.6f})", flush=True)
    return err


def run_exact(seeds, schemes, scales):
    import jax.numpy as jnp
    rows, validated = [], False
    for scheme in schemes:
        for seed in seeds:
            built = build(scheme, seed)
            trainer = built["trainer"]
            import exact_iqp
            vg, _loss_fn, correlators = exact_iqp.build_exact_loss(
                built["gates"], trainer.visible_bits, ANC, trainer.train_bits,
                trainer.qcbm["sigmas"])
            m = built["mask_anc"]
            if not validated:
                exact_iqp.validate_probs(
                    built["gates"], trainer.visible_bits, ANC, trainer.qcbm,
                    correlators, built["center"] + built["noise"])
                validated = True
            for c in scales:
                theta = built["center"] + c * built["noise"]
                val, g = vg(jnp.asarray(theta))
                g = np.asarray(g, dtype=np.float64)
                row = dict(mode="exact", dataset=DS, ancilla=ANC, scheme=scheme,
                           seed=seed, scale=c,
                           delta=branch_spread(theta, built["gates"],
                                               trainer.visible_bits),
                           delta_max_coupling=float(np.max(np.abs(theta[m]))),
                           loss=float(val), key_name="exact")
                row.update(sector_stats(g, m))
                rows.append(row)
                print(f"[exact] {scheme} seed={seed} c={c:g}: delta={row['delta']:.3e} "
                      f"|g_vis|={row['gnorm_vis']:.3e} |g_anc|={row['gnorm_anc']:.3e} "
                      f"ratio={row['gratio_norm']:.3e}", flush=True)
    return rows


# ------------------------------------------------------------ estimator mode
def run_est(seeds, schemes, scales, n_keys):
    import jax
    import jax.numpy as jnp
    import iqpopt.gen_qml as gen
    from iqpopt import IqpSimulator
    rows = []
    for scheme in schemes:
        for seed in seeds:
            built = build(scheme, seed)
            trainer = built["trainer"]
            q = trainer.qcbm
            m = built["mask_anc"]
            circ = IqpSimulator(trainer.visible_bits + ANC, built["gates"],
                                device=str(q.get("device", "lightning.qubit")),
                                spin_sym=bool(q.get("spin_sym", False)),
                                sparse=bool(q.get("sparse", False)),
                                bitflip=bool(q.get("bitflip", False)))
            kw = dict(iqp_circuit=circ, sigma=[float(s) for s in q["sigmas"]],
                      n_ops=int(q["n_ops"]), n_samples=int(q["n_samples"]),
                      wires=built["visible_wires"], jit=bool(q.get("loss_jit", True)),
                      sqrt_loss=False, max_batch_ops=q.get("max_batch_ops"),
                      max_batch_samples=q.get("max_batch_samples"),
                      ground_truth=trainer.train_bits)

            def f(p, k):
                return gen.mmd_loss_iqp(p, key=k, **kw)
            vg = jax.jit(jax.value_and_grad(f))

            rs = seed + ANC * 1000
            k0, _ = jax.random.split(jax.random.split(jax.random.PRNGKey(rs), 2)[0], 2)
            keys = [("step0", k0)] + [(f"alt{i}", jax.random.PRNGKey(900000 + i + rs))
                                      for i in range(1, n_keys)]
            for c in scales:
                th_np = built["center"] + c * built["noise"]
                theta = jnp.asarray(th_np, dtype=float)
                dlt = branch_spread(th_np, built["gates"], trainer.visible_bits)
                dmax = float(np.max(np.abs(th_np[m])))
                for kn, k in keys:
                    val, g = vg(theta, k)
                    g = np.asarray(g, dtype=np.float64)
                    row = dict(mode="est", dataset=DS, ancilla=ANC, scheme=scheme,
                               seed=seed, scale=c, delta=dlt,
                               delta_max_coupling=dmax, loss=float(val),
                               key_name=kn)
                    row.update(sector_stats(g, m))
                    rows.append(row)
                print(f"[est] {scheme} seed={seed} c={c:g}: delta={dlt:.3e} "
                      f"|g_anc| over {len(keys)} keys = "
                      f"{np.mean([r['gnorm_anc'] for r in rows[-len(keys):]]):.3e}"
                      f" +- {np.std([r['gnorm_anc'] for r in rows[-len(keys):]]):.1e}",
                      flush=True)
            jax.clear_caches()
    return rows


seeds = [int(s) for s in args.seeds.split(",")]
schemes = args.schemes.split(",")
scales = [float(c) for c in args.scales.split(",")]
rows = (run_exact(seeds, schemes, scales) if args.mode == "exact"
        else run_est(seeds, schemes, scales, args.keys))
out = HERE / f"noise_grad_{DS}_a{ANC}_{args.mode}.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
