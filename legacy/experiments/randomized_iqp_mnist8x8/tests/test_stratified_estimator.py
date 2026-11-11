"""The MMD^2 model term on stratified (branch-conditioned) samples.

stratified_mixture_moments must equal the old iid form on a single block, be
unbiased for (sum_l pi_l mu_l)^2 on several blocks (where the iid form is biased
low by Var_w(mu_l)/(N-1)), and the stratified global loss (all 2^a ancilla
patterns, uniform weights) must have the same expectation as the iid
gen.mmd_loss_iqp: the exact MMD^2 of the circuit's visible marginal.
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

import iqpopt.gen_qml as gen
import iqpopt.iqp_optimizer as iqo
from iqpopt import IqpSimulator
import randomized_iqp.trainer as trainer
from randomized_iqp.trainer import branch_weighted_mixture_mmd_loss, stratified_mixture_moments
from test_iqpopt_exact_stats import _exact


def _iid_form(x):
    """The model term the code used before: (tr^2 - sum x^2 / N^2) N / (N - 1)."""
    n = x.shape[-1]
    tr = x.mean(-1)
    return (tr * tr - (x * x).sum(-1) / n ** 2) * n / (n - 1)


def test_single_block_equals_iid_form() -> None:
    x = np.random.default_rng(0).uniform(-1, 1, (7, 50))
    mp, sq = stratified_mixture_moments(jnp.asarray(x), (50,))
    np.testing.assert_allclose(np.asarray(mp), x.mean(-1), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sq), _iid_form(x), atol=1e-6)
    # a one-sample block keeps the iid per-sample term and stays finite
    assert np.isfinite(np.asarray(stratified_mixture_moments(jnp.asarray(x), (1, 49))[1])).all()


def test_unbiased_on_stratified_samples_where_iid_form_is_not() -> None:
    rng = np.random.default_rng(1)
    mu, counts, pi = np.array([0.8, -0.5, 0.1]), (3, 5, 4), np.array([0.2, 0.5, 0.3])
    reps = 200_000
    x = np.concatenate([rng.normal(m, 0.6, (reps, c)) for m, c in zip(mu, counts)], axis=1)
    sq = np.asarray(stratified_mixture_moments(jnp.asarray(x), counts, pi)[1], dtype=np.float64)
    assert abs(sq.mean() - (pi @ mu) ** 2) < 4 * sq.std(ddof=1) / np.sqrt(reps)
    # the iid form with count weights w: expectation (w mu)^2 - Var_w(mu) / (N - 1)
    w, n = np.asarray(counts) / sum(counts), sum(counts)
    old = _iid_form(x)
    se = old.std(ddof=1) / np.sqrt(reps)
    bias = -(w @ mu ** 2 - (w @ mu) ** 2) / (n - 1)
    assert abs(old.mean() - ((w @ mu) ** 2 + bias)) < 4 * se
    assert abs(bias) > 10 * se


@pytest.mark.parametrize("path", ["chunked", "pairs"])
def test_stratified_global_loss_matches_iid_loss_and_exact(path, monkeypatch) -> None:
    rng = np.random.default_rng(4)
    n_vis, anc = 4, [4, 5]
    n = n_vis + len(anc)
    gates = []
    for _ in range(14):
        vis = rng.choice(n_vis, size=rng.integers(1, 3), replace=False)
        gates.append([sorted(int(w) for w in vis) + [a for a in anc if rng.random() < 0.5]])
    params = rng.normal(0, 1.0, len(gates))
    circuit = IqpSimulator(n, gates)
    data = rng.integers(0, 2, (200, n_vis))
    # few samples per branch and strongly differing branches: the iid form's bias is ~8 SE here
    sigma, n_ops, n_samples, keys = 1.0, 256, 8, 400
    wires = list(range(n_vis))

    # exact expectation of both estimators: sum_A w_A [z_A^2 - 2 z_A t_A + (t_A^2 m - 1)/(m - 1)]
    p = (1 - np.exp(-1 / 2 / sigma ** 2)) / 2
    subsets = (np.arange(2 ** n_vis)[:, None] >> np.arange(n_vis)) & 1
    w = p ** subsets.sum(1) * (1 - p) ** (n_vis - subsets.sum(1))
    ops = np.zeros((len(subsets), n))
    ops[:, :n_vis] = subsets
    z = _exact(gates, params, n, ops)
    t = (1 - 2 * ((data @ subsets.T) % 2)).mean(0)
    m = len(data)
    exact = float(w @ (z * z - 2 * z * t + (t * t * m - 1) / (m - 1)))

    masks = np.zeros((4, n), bool)
    masks[:, anc] = True
    values = np.zeros((4, n), np.int32)
    values[:, anc] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ground = jnp.asarray(data)
    calls = []
    if path == "pairs":  # the campaign's path: the estimator relies on its per-branch column blocks
        monkeypatch.setattr(iqo, "PAIRS_MIN_GAIN", 0)
        orig = IqpSimulator._op_expval_pairs
        monkeypatch.setattr(IqpSimulator, "_op_expval_pairs", lambda self, *a: calls.append(1) or orig(self, *a))
    # a fresh lambda per call: jax.jit caches on the function object
    strat = lambda: jax.jit(lambda k: branch_weighted_mixture_mmd_loss(
        params=jnp.asarray(params), key=k, iqp_circuit=circuit, ground_truth=ground,
        branch_condition_masks=tuple(masks), branch_condition_values=tuple(values),
        branch_sample_counts=(2, 2, 2, 2), sigma=sigma, n_ops=n_ops, n_samples=n_samples,
        wires=wires, branch_weights=(0.25, 0.25, 0.25, 0.25)))
    iid = jax.jit(lambda k: gen.mmd_loss_iqp(
        params=jnp.asarray(params), iqp_circuit=circuit, ground_truth=ground, sigma=sigma,
        n_ops=n_ops, n_samples=n_samples, key=k, wires=wires, jit=False))
    ks = jax.random.split(jax.random.PRNGKey(3), keys)

    def miss(f):
        vals = np.array([float(f(k)) for k in ks])
        return abs(vals.mean() - exact) / (vals.std(ddof=1) / np.sqrt(keys))

    assert miss(strat()) < 4, "stratified loss is biased"
    assert bool(calls) == (path == "pairs"), "stratified loss did not take the intended expval path"
    assert miss(iid) < 4, "iid loss is biased"
    # the same samples with the old iid form miss the exact value: the test sees the bias
    monkeypatch.setattr(trainer, "stratified_mixture_moments", lambda x, c, w=None: (x.mean(-1), _iid_form(x)))
    assert miss(strat()) > 4
