#!/usr/bin/env python3
"""Exact IQP loss for a branch mixture with arbitrary weights pi_l.

The mixture correlator is z_A = sum_l pi_l z_A(theta^(l)), so instead of
tracing the ancillas out of one (n+a)-qubit state we evaluate each branch as
its own n-qubit IQP circuit and average with the branch weights.  This is
exact for any pi (the cluster-initialized scheme trains with pi_l set by the
cluster sizes) and is cheaper than the (n+a)-qubit route.

Branch angles come from the compiled angles by the Walsh transform
theta^(l)_G = sum_S (-1)^{|l & S|} theta_{G,S}.
"""
from __future__ import annotations

import numpy as np


def branch_matrix(gates, n, a):
    """(L, m_vis, m) matrix B with theta^(l) = B[l] @ theta_compiled, plus the
    list of visible generators in the order used by the branch circuits."""
    L = 2 ** a
    vis_key, subsets = [], []
    for g in gates:
        vis = tuple(sorted({w for gen in g for w in gen if w < n}))
        anc = tuple(sorted({w - n for gen in g for w in gen if w >= n}))
        vis_key.append(vis)
        subsets.append(anc)
    uniq = sorted(set(vis_key))
    idx = {k: i for i, k in enumerate(uniq)}
    B = np.zeros((L, len(uniq), len(gates)), dtype=np.float64)
    for i, (k, S) in enumerate(zip(vis_key, subsets)):
        for l in range(L):
            sign = -1.0 if sum((l >> s) & 1 for s in S) % 2 else 1.0
            B[l, idx[k], i] = sign
    # visible generators as wire tuples, in the row order of B
    return B, [list(k) for k in uniq]


def build_exact_loss_weighted(gates, n, a, train_bits, sigmas, pi):
    import jax
    import jax.numpy as jnp

    L = 2 ** a
    B, vis_gens = branch_matrix(gates, n, a)
    pi = np.asarray(pi, dtype=np.float64)
    assert pi.shape == (L,) and abs(pi.sum() - 1) < 1e-12, (pi.shape, pi.sum())
    Bj, pij = jnp.asarray(B), jnp.asarray(pi)

    gen_index = jnp.asarray(np.array(
        [sum(1 << (n - 1 - w) for w in gen) for gen in vis_gens], dtype=np.int64))
    shape = (2,) * n

    def wht(t, axes):
        for ax in axes:
            t = jnp.moveaxis(t, ax, 0)
            x, y = t[0], t[1]
            t = jnp.moveaxis(jnp.stack([x + y, x - y], axis=0), 0, ax)
        return t

    def branch_corr(theta_branch):
        c = jnp.zeros(2 ** n, dtype=theta_branch.dtype).at[gen_index].set(theta_branch)
        phase = wht(c.reshape(shape), range(n))
        psi = wht(jnp.exp(-1j * phase), range(n)) / (2.0 ** n)
        p = jnp.abs(psi) ** 2
        return wht(p, range(n)).reshape(-1)

    data = np.asarray(train_bits, dtype=np.int64)
    hist = np.bincount((data * (2 ** np.arange(n - 1, -1, -1))).sum(axis=1),
                       minlength=2 ** n).astype(np.float64) / len(data)
    t_A = jnp.asarray(np.asarray(
        wht(jnp.asarray(hist).reshape(shape), range(n))).reshape(-1))

    A_bits = ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1)
    W = []
    for sig in [float(s) for s in sigmas]:
        p_mmd = (1.0 - np.exp(-1.0 / 2.0 / sig ** 2)) / 2.0
        logw = (A_bits * np.log(p_mmd) + (1 - A_bits) * np.log1p(-p_mmd)).sum(axis=1)
        W.append(jnp.asarray(np.exp(logw)))

    def loss(theta):
        z = jnp.einsum("l,li->i", pij,
                       jax.vmap(branch_corr)(jnp.einsum("lgm,m->lg", Bj, theta)))
        d2 = (z - t_A) ** 2
        return sum(jnp.sum(w * d2) for w in W) / len(W)

    return jax.jit(jax.value_and_grad(loss)), jax.jit(loss)


def cluster_weights(labels_path, L):
    """pi_l from the cluster label file (branch weight = group mass)."""
    y = np.loadtxt(labels_path, dtype=int, delimiter=",")
    y = np.atleast_1d(y).ravel()
    counts = np.bincount(y, minlength=L).astype(np.float64)
    assert counts.sum() == len(y)
    return counts / counts.sum()
