from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

import jax
import jax.numpy as jnp
import numpy as np

import pytest

from iqpopt import IqpSimulator


def test_op_expval_batch_conditioned_samples_change_estimator() -> None:
    circuit = IqpSimulator(2, gates=[[[0]], [[0, 1]]], sparse=False, device="default.qubit")
    params = jnp.asarray([0.2, 0.7], dtype=jnp.float64)
    ops = jnp.asarray([[1, 0]], dtype=jnp.float64)
    key = jax.random.PRNGKey(123)
    mask = jnp.asarray([False, True])
    values_zero = jnp.asarray([0, 0], dtype=jnp.int32)
    values_one = jnp.asarray([0, 1], dtype=jnp.int32)

    conditioned_zero = circuit.op_expval_batch(
        params,
        ops,
        n_samples=32,
        key=key,
        return_samples=True,
        condition_mask=mask,
        condition_values=values_zero,
    )
    conditioned_one = circuit.op_expval_batch(
        params,
        ops,
        n_samples=32,
        key=key,
        return_samples=True,
        condition_mask=mask,
        condition_values=values_one,
    )

    assert conditioned_zero.shape == conditioned_one.shape == (1, 32)
    assert not np.allclose(np.asarray(conditioned_zero), np.asarray(conditioned_one))


def test_op_expval_batch_multi_condition_accepts_batched_counts() -> None:
    circuit = IqpSimulator(2, gates=[[[0]], [[0, 1]]], sparse=False, device="default.qubit")
    params = jnp.asarray([0.2, 0.7], dtype=jnp.float64)
    ops = jnp.asarray([[1, 0]], dtype=jnp.float64)
    key = jax.random.PRNGKey(456)
    masks = jnp.array([[False, False], [True, False]], dtype=bool)
    values = jnp.array([[0, 0], [1, 0]], dtype=jnp.int32)
    counts = jnp.array([12, 18], dtype=jnp.int32)

    samples = circuit.op_expval_batch(
        params,
        ops,
        n_samples=30,
        key=key,
        return_samples=True,
        condition_mask=masks,
        condition_values=values,
        condition_counts=counts,
    )

    assert samples.shape == (1, 30)
    assert jnp.all(jnp.isfinite(samples))


def test_op_expval_batch_multi_condition_rejects_bad_counts() -> None:
    circuit = IqpSimulator(2, gates=[[[0]], [[0, 1]]], sparse=False, device="default.qubit")
    params = jnp.asarray([0.2, 0.7], dtype=jnp.float64)
    ops = jnp.asarray([[1, 0]], dtype=jnp.float64)
    key = jax.random.PRNGKey(789)
    masks = jnp.array([[False, False]], dtype=bool)
    values = jnp.array([[0, 0]], dtype=jnp.int32)
    counts = jnp.array([5], dtype=jnp.int32)

    with pytest.raises(ValueError, match=r"sum\(condition_counts\) must equal n_samples"):
        circuit.op_expval_batch(
            params,
            ops,
            n_samples=4,
            key=key,
            return_samples=True,
            condition_mask=masks,
            condition_values=values,
            condition_counts=counts,
        )
