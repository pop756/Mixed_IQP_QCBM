"""The grouped expval path (IqpSimulator._phases) must match the dense
generator-matmul path it replaces, values and gradients."""
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

# the dense reference path rounds its matmuls on ROCm/TF32 GPUs (~1e-3 off)
jax.config.update("jax_default_matmul_precision", "highest")

N = 7
HIDDEN = [5, 6]
VISIBLE = [4, 0, 1, 2, 3]  # unsorted on purpose
# singleton, pair, visible+hidden, order-3 with hidden, hidden-only, duplicate wire
GATES = [[[0]], [[1, 2]], [[0, 5]], [[3, 4, 6]], [[5, 6]], [[2, 2, 1]], [[1, 5, 6]], [[1]]]


def _run(circuit, params, init_coefs, ops, **kw):
    def f(p):
        return circuit.op_expval(p, ops, 60, jax.random.PRNGKey(3), init_coefs=init_coefs,
                                 return_samples=True, ops_wires=VISIBLE, **kw)
    return np.asarray(f(params)), np.asarray(jax.grad(lambda p: jnp.sum(f(p) ** 2))(params))


@pytest.mark.parametrize("case", ["plain", "multigen_init_spin", "multi_condition_scan", "multi_condition_loop", "indep"])
def test_grouped_path_matches_dense_path(case, monkeypatch) -> None:
    rng = np.random.default_rng(0)
    gates = GATES + ([[[0], [2, 3]]] if case == "multigen_init_spin" else [])
    init_gates = [[[4]], [[0, 6]]] if case == "multigen_init_spin" else None
    circuit = IqpSimulator(N, gates, init_gates=init_gates, spin_sym=case == "multigen_init_spin")
    params = jnp.asarray(rng.normal(size=len(gates)), dtype=jnp.float32)
    init_coefs = jnp.asarray([0.3, -0.7]) if init_gates else None
    ops = np.zeros((9, N), dtype=np.float32)
    ops[:, VISIBLE] = rng.integers(0, 2, (9, len(VISIBLE)))
    kw = {}
    if case.startswith("multi_condition"):
        masks = np.zeros((4, N), bool)
        masks[:, HIDDEN] = True
        values = np.zeros((4, N), np.int32)
        values[:, HIDDEN] = [[0, 0], [0, 1], [1, 0], [1, 1]]
        kw = dict(condition_mask=jnp.asarray(masks), condition_values=jnp.asarray(values),
                  condition_counts=(10, 20, 0, 30),
                  max_batch_samples=20 if case.endswith("scan") else 25)
    if case == "indep":
        kw = dict(indep_estimates=True)

    fast = _run(circuit, params, init_coefs, jnp.asarray(ops), **kw)
    assert circuit._ops_split(VISIBLE) is not None
    monkeypatch.setattr(iqo, "MAX_HIDDEN_WIRES", -1)  # forces the dense path
    circuit._split_cache.clear()
    dense = _run(circuit, params, init_coefs, jnp.asarray(ops), **kw)
    for a, b in zip(fast, dense):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)  # float32 rounding


def test_sparse_and_bitflip_paths_still_run() -> None:
    params = jnp.asarray(np.linspace(-1, 1, len(GATES)), dtype=jnp.float32)
    ops = np.zeros((5, N), dtype=np.float32)
    ops[:, :3] = 1
    key = jax.random.PRNGKey(0)
    dense = IqpSimulator(N, GATES).op_expval_batch(params, jnp.asarray(ops), 40, key, return_samples=True)
    sparse = IqpSimulator(N, GATES, sparse=True).op_expval_batch(params, ops, 40, key, return_samples=True)
    np.testing.assert_allclose(np.asarray(sparse), np.asarray(dense), atol=2e-5)
    assert IqpSimulator(N, GATES, bitflip=True).op_expval(params, jnp.asarray(ops), 40, key)[0].shape == (5,)


# _op_expval_pairs: order-2 visible generators (all ancilla subsets), 24 visible wires
NV, ANC2 = 24, [24, 25]
PAIR_GATES = ([[[i]] for i in range(NV)] + [[[i, j]] for i in range(NV) for j in range(i + 1, NV)])
PAIR_GATES = [[sorted(g[0] + list(sub))] for g in PAIR_GATES for sub in ([], [24], [25], [24, 25])] + [[[24, 25]], [[24]]]


@pytest.mark.parametrize("case", ["branch_loop", "branch_scan", "single_condition", "no_hidden",
                                  "overflow", "entries_over_budget", "gpu_support", "gpu_overflow"])
def test_pairs_path_matches_grouped_path(case, monkeypatch) -> None:
    rng = np.random.default_rng(1)
    no_hidden = case == "no_hidden"
    gates = [g for g in PAIR_GATES if max(g[0]) < NV] if no_hidden else PAIR_GATES
    n = NV if no_hidden else NV + 2
    circuit = IqpSimulator(n, gates)
    params = jnp.asarray(rng.normal(0, 0.3, len(gates)), dtype=jnp.float32)
    ops = np.zeros((10, n), dtype=np.float32)
    for row, w in zip(ops, [1, 2, 2, 3, 3, 4, 5, 2, 1, 22 if "overflow" in case else 6]):
        row[rng.choice(NV, w, replace=False)] = 1
    kw = dict(ops_wires=None if no_hidden else tuple(range(NV)))
    if case.startswith("branch") or case in ("overflow", "entries_over_budget", "gpu_support", "gpu_overflow"):
        masks = np.zeros((4, n), bool)
        masks[:, ANC2] = True
        masks[2, 3] = True  # a conditioned visible wire too
        values = np.zeros((4, n), np.int32)
        values[:, ANC2] = [[0, 0], [0, 1], [1, 0], [1, 1]]
        values[2, 3] = 1
        kw.update(condition_mask=masks, condition_values=values, condition_counts=(10, 20, 0, 30),
                  max_batch_samples=20 if case == "branch_scan" else 25, max_batch_ops=4)
    if case == "single_condition":
        mask = np.zeros(n, bool)
        mask[ANC2] = True
        values = np.zeros(n, np.int32)
        values[ANC2] = [1, 0]
        kw.update(condition_mask=mask, condition_values=values)
    if case == "entries_over_budget":
        monkeypatch.setattr(iqo, "ENTRY_BUDGET", 1)  # CPU: support form in op chunks
    if case.startswith("gpu"):
        monkeypatch.setattr(iqo.jax, "default_backend", lambda: "gpu")

    def run():
        def f(p):
            return circuit.op_expval(p, jnp.asarray(ops), 60, jax.random.PRNGKey(3), return_samples=True, **kw)
        return np.asarray(f(params)), np.asarray(jax.grad(lambda p: jnp.sum(f(p) ** 2))(params))

    monkeypatch.setattr(iqo, "EXACT", False)  # tests the pairwise path in either mode
    monkeypatch.setattr(iqo, "PAIRS_MIN_GAIN", 0)  # small circuit: take the pairwise path anyway
    calls = []
    orig = IqpSimulator._op_expval_pairs
    monkeypatch.setattr(IqpSimulator, "_op_expval_pairs", lambda self, *a: calls.append(1) or orig(self, *a))
    pairs = run()
    assert calls
    monkeypatch.setattr(IqpSimulator, "_segments", staticmethod(lambda *a: None))  # the grouped path
    calls.clear()
    grouped = run()
    assert not calls
    for a, b in zip(pairs, grouped):  # float32 rounding of sums of ~100 angles (float64: 1e-12)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5 * np.abs(b).max())


def test_fast_mode_per_sample_within_1e2_of_exact_mode(monkeypatch) -> None:
    """Fast mode (default) against IQPOPT_EXACT=1 on the same samples, at a size where
    op_expval picks the pairwise path by itself: 64 visible wires, fully connected
    order 2, 2 ancillas with all subsets, sparse MMD-like ops, cluster branches."""
    from randomized_iqp.ancilla import augment_visible_gates_with_ancilla
    from randomized_iqp.gates import fully_connected_gates
    nv = 64
    gates, vis, anc = augment_visible_gates_with_ancilla(fully_connected_gates(nv, 2), nv, 2)
    circuit = IqpSimulator(nv + 2, gates)
    rng = np.random.default_rng(2)
    params = jnp.asarray(rng.normal(0, 0.3, len(gates)), dtype=jnp.float32)
    ops = np.zeros((200, nv + 2), np.float32)
    ops[:, :nv] = rng.random((200, nv)) < 0.05
    masks = np.zeros((4, nv + 2), bool)
    masks[:, anc] = True
    values = np.zeros((4, nv + 2), np.int32)
    values[:, anc] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    kw = dict(ops_wires=tuple(vis), condition_mask=masks, condition_values=values,
              condition_counts=(300, 250, 350, 300), max_batch_samples=500, max_batch_ops=128, return_samples=True)
    run = lambda: np.asarray(circuit.op_expval(params, jnp.asarray(ops), 1200, jax.random.PRNGKey(5), **kw))
    calls = []
    orig = IqpSimulator._op_expval_pairs
    monkeypatch.setattr(IqpSimulator, "_op_expval_pairs", lambda self, *a: calls.append(1) or orig(self, *a))
    monkeypatch.setattr(iqo, "EXACT", False)
    fast = run()
    assert calls
    monkeypatch.setattr(iqo, "EXACT", True)
    exact = run()
    assert len(calls) == 1
    assert np.abs(fast - exact).max() <= 1e-2
