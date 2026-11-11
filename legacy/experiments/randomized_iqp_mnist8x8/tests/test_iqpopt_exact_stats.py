"""Exact statistics of the iqpopt <Z_A> estimator against a statevector reference.

The sample mean of op_expval(..., return_samples=True) and its parameter gradient
must agree with the exact value (numpy statevector, float64) within 4 standard
errors: for the visible marginal of the full circuit, and for branch-conditioned
samples, whose slots of branch b estimate the visible circuit that
reduce_to_visible_branch builds for b. Runs in whichever mode IQPOPT_EXACT selects.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iqpopt.iqp_optimizer as iqo
from iqpopt import IqpSimulator
from randomized_iqp.ancilla import reduce_to_visible_branch

N_VIS, ANC = 9, [9, 10]  # n = 11 qubits
N = N_VIS + len(ANC)
N_SAMPLES = 2 ** 18
TOL = 4.0  # standard errors


def _gates(rng):
    """Random generators of order 1-3 over all wires (ancilla subsets included), one
    two-generator gate, and a hidden-only gate."""
    gates = []
    for _ in range(28):
        k = rng.integers(1, 4)
        gates.append([sorted(int(w) for w in rng.choice(N, size=k, replace=False))])
    return gates + [[[0, 3], [5, 9]], [[9, 10]], [[2, 9, 10]]]


def _pair_gates(rng, n_wires, ancilla):
    """Generators with at most two visible wires plus any subset of the ancilla wires
    (the layouts _op_expval_pairs handles), and a hidden-only gate."""
    gates = []
    for _ in range(40):
        vis = rng.choice(n_wires, size=rng.integers(1, 3), replace=False)
        anc = [a for a in ancilla if rng.random() < 0.5]
        gates.append([sorted(int(w) for w in vis) + anc])
    return gates + ([[list(ancilla)]] if ancilla else [])


def _exact(gates, params, n, ops):
    """<Z_op> for every row of ops on n qubits: H^n exp(-i sum theta Z_g) H^n |0>."""
    X = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    phase = np.zeros(2 ** n)
    for th, gate in zip(params, gates):
        for gen in gate:
            phase += th * (1 - 2 * (X[:, gen].sum(1) % 2))
    a = (np.exp(-1j * phase) / 2 ** (n / 2)).reshape((2,) * n)
    for ax in range(n):
        a0, a1 = np.take(a, 0, ax), np.take(a, 1, ax)
        a = np.stack([a0 + a1, a0 - a1], axis=ax) / np.sqrt(2)
    p = np.abs(a.reshape(-1)) ** 2
    return np.array([(p * (1 - 2 * ((X @ op) % 2))).sum() for op in ops])


def _exact_grad(f, params, h=1e-6):
    out = []
    for j in range(len(params)):
        e = np.zeros(len(params))
        e[j] = h
        out.append((f(params + e) - f(params - e)) / (2 * h))
    return np.array(out)


def _check(name, vals, grads, exact, exact_grad):
    """vals (n_ops, m) and grads (m, P) are per-sample estimates; exact (n_ops,), exact_grad (P,)."""
    m = vals.shape[1]
    for est, ref in ((vals, exact), (grads.T, exact_grad)):
        mean = est.mean(1)
        se = est.std(1, ddof=1) / np.sqrt(m)
        z = np.abs(mean - ref) / (se + 1e-7)
        assert z.max() < TOL, f"{name}: max |mean - exact| = {z.max():.2f} standard errors at {z.argmax()}"


@pytest.mark.parametrize("case", ["plain", "dense", "branch_scan", "branch_loop", "pairs_branch", "pairs_no_hidden"])
def test_estimator_matches_statevector(case, monkeypatch) -> None:
    rng = np.random.default_rng(1)
    if case == "pairs_branch":
        gates = _pair_gates(rng, N_VIS, ANC)
    elif case == "pairs_no_hidden":
        gates = _pair_gates(rng, N, [])
    else:
        gates = _gates(rng)
    pairs = case.startswith("pairs")
    if pairs:  # small circuit: take op_expval's pairwise path anyway, and check that it does
        monkeypatch.setattr(iqo, "PAIRS_MIN_GAIN", 0)
        calls = []
        orig = IqpSimulator._op_expval_pairs
        monkeypatch.setattr(IqpSimulator, "_op_expval_pairs", lambda self, *a: calls.append(1) or orig(self, *a))
    params = rng.normal(0, 0.4, len(gates))
    ops = np.zeros((6, N), np.float32)
    ops[:, :N_VIS] = rng.random((6, N_VIS)) < 0.3
    ops[0, :N_VIS] = 0
    ops[0, 4] = 1
    ops[ops.sum(1) == 0, 1] = 1
    c = rng.normal(size=len(ops))  # gradient of sum_i c_i <Z_i>
    circuit = IqpSimulator(N, gates)

    kw = dict(ops_wires=None if case in ("dense", "pairs_no_hidden") else tuple(range(N_VIS)))
    branches = None
    if case.startswith("branch") or case == "pairs_branch":
        branches = ["00", "01", "10", "11"]
        masks = np.zeros((4, N), bool)
        masks[:, ANC] = True
        values = np.zeros((4, N), np.int32)
        values[:, ANC] = [[int(b) for b in bits] for bits in branches]
        counts = (2 ** 16,) * 4 if case == "branch_scan" else (50_000, 70_000, 60_000, 82_144)
        if not pairs:  # jnp: the grouped path (numpy lets op_expval read the branch layout)
            masks, values = jnp.asarray(masks), jnp.asarray(values)
        kw.update(condition_mask=masks, condition_values=values, condition_counts=counts,
                  max_batch_samples=2 ** 15 if case == "branch_scan" else 100_000, max_batch_ops=4)

    def f(p):
        return circuit.op_expval(p, jnp.asarray(ops), N_SAMPLES, jax.random.PRNGKey(11), return_samples=True, **kw)

    p32 = jnp.asarray(params, jnp.float32)
    vals = np.asarray(f(p32), np.float64)
    grads = np.asarray(jax.jacfwd(lambda p: jnp.asarray(c, jnp.float32) @ f(p))(p32), np.float64)
    assert vals.shape == (len(ops), N_SAMPLES)
    assert not pairs or bool(calls) != iqo.EXACT  # IQPOPT_EXACT=1: the grouped path

    if branches is None:
        full = lambda p: _exact(gates, p, N, ops.astype(int))
        _check(case, vals, grads, full(params), _exact_grad(lambda p: c @ full(p), params))
        return
    start = 0
    for bits, count in zip(branches, counts):
        def branch(p, bits=bits):
            vg, vp = reduce_to_visible_branch(gates, p, N_VIS, ANC, bits, zero_tol=0.0)
            return _exact(vg, vp, N_VIS, ops[:, :N_VIS].astype(int))
        sl = slice(start, start + count)
        _check(f"{case} branch {bits}", vals[:, sl], grads[sl], branch(params),
               _exact_grad(lambda p: c @ branch(p), params))
        start += count
