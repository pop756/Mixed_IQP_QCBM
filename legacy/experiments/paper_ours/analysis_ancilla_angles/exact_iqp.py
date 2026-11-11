#!/usr/bin/env python3
"""Exact (statevector) IQP loss for the n=16 benchmarks, memory-safe in a.

phase(y) = sum_G theta_G (-1)^{G.y} is a Walsh polynomial, so it is the
Walsh-Hadamard transform of the coefficient array c[A] = sum_{G=A} theta_G.
That avoids the (n_gates x 2^N) sign matrix, which is 4.6 GB at a=3.

    psi = WHT(exp(-i phase)) / 2^N      (H^N e^{-i sum theta_G Z_G} H^N |0>)
    p   = |psi|^2  ->  trace out the a ancilla wires
    z_A = WHT(p_vis)                    (Pauli-Z correlators of the visible marginal)
    L   = mean_sigma sum_A P_sigma(A) (z_A - t_A)^2

L is the expectation of iqpopt's mmd_loss_iqp over its op draw and its
sampling, so its gradient is the exact version of the training gradient.
Validated against IqpSimulator.probs (PennyLane) by validate_probs().
"""
from __future__ import annotations

import numpy as np


def build_exact_loss(gates, n, a, train_bits, sigmas):
    import jax
    import jax.numpy as jnp

    N = n + a

    def wht(t, axes):
        for ax in axes:
            t = jnp.moveaxis(t, ax, 0)
            x, y = t[0], t[1]
            t = jnp.moveaxis(jnp.stack([x + y, x - y], axis=0), 0, ax)
        return t

    # gate -> index of its Walsh coefficient A_G (bit j = wire j, wire 0 = MSB)
    gate_index = np.array(
        [sum(1 << (N - 1 - w) for w in sorted({w for gen in g for w in gen}))
         for g in gates], dtype=np.int64)
    assert len(set(gate_index.tolist())) == len(gate_index), "duplicate gate support"
    gate_index_j = jnp.asarray(gate_index)

    shape_full, shape_vis = (2,) * N, (2,) * n

    def correlators(theta):
        c = jnp.zeros(2 ** N, dtype=theta.dtype).at[gate_index_j].set(theta)
        phase = wht(c.reshape(shape_full), range(N))
        psi = wht(jnp.exp(-1j * phase), range(N)) / (2.0 ** N)
        p = jnp.abs(psi) ** 2
        p_vis = jnp.sum(p.reshape(shape_vis + (2 ** a,)), axis=-1)
        return wht(p_vis, range(n)).reshape(-1), p_vis.reshape(-1)

    data = np.asarray(train_bits, dtype=np.int64)
    hist = np.bincount((data * (2 ** np.arange(n - 1, -1, -1))).sum(axis=1),
                       minlength=2 ** n).astype(np.float64) / len(data)
    t_A = jnp.asarray(np.asarray(
        wht(jnp.asarray(hist).reshape(shape_vis), range(n))).reshape(-1))

    A_bits = ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1)
    W = []
    for sig in [float(s) for s in sigmas]:
        p_mmd = (1.0 - np.exp(-1.0 / 2.0 / sig ** 2)) / 2.0
        logw = (A_bits * np.log(p_mmd) + (1 - A_bits) * np.log1p(-p_mmd)).sum(axis=1)
        W.append(jnp.asarray(np.exp(logw)))

    def loss(theta):
        z, _ = correlators(theta)
        d2 = (z - t_A) ** 2
        return sum(jnp.sum(w * d2) for w in W) / len(W)

    return jax.jit(jax.value_and_grad(loss)), jax.jit(loss), correlators


def validate_probs(gates, n, a, qcbm, correlators, theta):
    """IqpSimulator.probs (PennyLane, lightning.qubit) vs the transform above."""
    import jax.numpy as jnp
    from iqpopt import IqpSimulator
    circ = IqpSimulator(n + a, gates,
                        device=str(qcbm.get("device", "lightning.qubit")),
                        spin_sym=bool(qcbm.get("spin_sym", False)),
                        sparse=bool(qcbm.get("sparse", False)),
                        bitflip=bool(qcbm.get("bitflip", False)))
    p_pl = np.asarray(circ.probs(jnp.asarray(theta)), dtype=np.float64)
    p_pl_vis = p_pl.reshape(-1, 2 ** a).sum(axis=1)
    _, p_mine = correlators(jnp.asarray(theta))
    err = float(np.max(np.abs(p_pl_vis - np.asarray(p_mine, dtype=np.float64))))
    print(f"[validate] max |p_pennylane - p_transform| = {err:.3e}", flush=True)
    return err
