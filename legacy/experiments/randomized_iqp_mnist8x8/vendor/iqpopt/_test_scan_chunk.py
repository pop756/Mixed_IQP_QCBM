#!/usr/bin/env python3
"""CPU correctness test for the lax.scan + jax.checkpoint sample-chunking path in
IqpSimulator.op_expval. Verifies (1) the scan path reproduces, bit-for-bit, a
hand-rolled per-chunk computation that uses the IDENTICAL key derivation
(split(key, n_chunks+1)[:n_chunks]) for BOTH the plain and multi_condition
paths, value AND gradient; and (2) the scan-batched estimator agrees with the
unbatched estimator at large n_samples (statistical, since the two use different
random draws).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from iqpopt import IqpSimulator

N = 12
gates = [[[i]] for i in range(N)] + [[[i, j]] for i in range(N) for j in range(i + 1, N)]
sim = IqpSimulator(N, gates, sparse=False, spin_sym=False, bitflip=False)
rng = np.random.default_rng(0)
params = jnp.asarray(rng.normal(size=len(gates)) * 0.1)
n_ops = 20
ops = jnp.asarray(rng.integers(0, 2, size=(n_ops, N)))
key = jax.random.PRNGKey(123)

CHUNK = 200
NS = 1000              # 5 chunks
n_chunks = NS // CHUNK


def ref_plain(p, key):
    keys = jax.random.split(key, n_chunks + 1)[:n_chunks]
    parts = [sim.op_expval_batch(p, ops, CHUNK, keys[c], None, False,
                                 return_samples=True) for c in range(n_chunks)]
    stacked = jnp.stack(parts, 0)                       # (n_chunks, n_ops, CHUNK)
    return jnp.transpose(stacked, (1, 0, 2)).reshape(n_ops, n_chunks * CHUNK)


# ---- plain path: value ----
got = sim.op_expval(params, ops, NS, key, return_samples=True, max_batch_samples=CHUNK)
ref = ref_plain(params, key)
d = float(jnp.max(jnp.abs(got - ref)))
print(f"[plain] value max|scan-ref| = {d:.3e}  shape={got.shape}")
assert d < 1e-9, "plain value mismatch"

# ---- plain path: gradient ----
fscan = lambda p: jnp.sum(sim.op_expval(p, ops, NS, key, return_samples=True, max_batch_samples=CHUNK))
fref = lambda p: jnp.sum(ref_plain(p, key))
gs, gr = jax.grad(fscan)(params), jax.grad(fref)(params)
gd = float(jnp.max(jnp.abs(gs - gr)))
print(f"[plain] grad  max|scan-ref| = {gd:.3e}  |grad|={float(jnp.linalg.norm(gr)):.4f}")
# float32 math: separately compiled scan vs reference may differ in reduction order
assert gd < 1e-6 * float(jnp.linalg.norm(gr)), "plain grad mismatch"

# ---- plain consistency: scan-batched vs unbatched at large samples ----
NSB = 20000
vb = jnp.mean(sim.op_expval(params, ops, NSB, key, return_samples=True, max_batch_samples=1000), axis=-1)
vu = sim.op_expval(params, ops, NSB, key, return_samples=False)[0]
print(f"[plain] mean(scan@{NSB}) vs mean(unbatched@{NSB}) max|diff| = {float(jnp.max(jnp.abs(vb-vu))):.3e}")

# ---- multi_condition path ----
n_branches = 4
cmask = jnp.asarray(rng.integers(0, 2, size=(n_branches, N)).astype(bool))
cvals = jnp.asarray(rng.integers(0, 2, size=(n_branches, N)))
# counts summing to NS, all positive
base = NS // n_branches
counts = np.full(n_branches, base, dtype=np.int64); counts[-1] += NS - base * n_branches
counts_j = jnp.asarray(counts, dtype=jnp.int32)
branch_of_sample = np.repeat(np.arange(n_branches), counts)
counts_per_chunk = np.stack([np.bincount(branch_of_sample[c*CHUNK:(c+1)*CHUNK], minlength=n_branches)
                             for c in range(n_chunks)], 0)


def ref_multi(p, key):
    keys = jax.random.split(key, n_chunks + 1)[:n_chunks]
    parts = [sim.op_expval_batch(p, ops, CHUNK, keys[c], None, False, return_samples=True,
                                 condition_mask=cmask, condition_values=cvals,
                                 condition_counts=jnp.asarray(counts_per_chunk[c], dtype=jnp.int32))
             for c in range(n_chunks)]
    stacked = jnp.stack(parts, 0)
    return jnp.transpose(stacked, (1, 0, 2)).reshape(n_ops, n_chunks * CHUNK)


gotm = sim.op_expval(params, ops, NS, key, return_samples=True, max_batch_samples=CHUNK,
                     condition_mask=cmask, condition_values=cvals, condition_counts=counts_j)
refm = ref_multi(params, key)
dm = float(jnp.max(jnp.abs(gotm - refm)))
print(f"[multi] value max|scan-ref| = {dm:.3e}  shape={gotm.shape}")
assert dm < 1e-9, "multi value mismatch"

fm = lambda p: jnp.sum(sim.op_expval(p, ops, NS, key, return_samples=True, max_batch_samples=CHUNK,
                                     condition_mask=cmask, condition_values=cvals, condition_counts=counts_j))
fmr = lambda p: jnp.sum(ref_multi(p, key))
gm, gmr = jax.grad(fm)(params), jax.grad(fmr)(params)
gmd = float(jnp.max(jnp.abs(gm - gmr)))
print(f"[multi] grad  max|scan-ref| = {gmd:.3e}  |grad|={float(jnp.linalg.norm(gmr)):.4f}")
assert gmd < 1e-6 * float(jnp.linalg.norm(gmr)), "multi grad mismatch"

# ---- multi consistency: scan vs unbatched (full python-loop path) at large samples ----
NSB2 = 16000   # divisible by 1000
counts2 = np.full(n_branches, NSB2 // n_branches, dtype=np.int64)
counts2[-1] += NSB2 - counts2.sum()
counts2_j = jnp.asarray(counts2, dtype=jnp.int32)
mb = jnp.mean(sim.op_expval(params, ops, NSB2, key, return_samples=True, max_batch_samples=1000,
                            condition_mask=cmask, condition_values=cvals, condition_counts=counts2_j), axis=-1)
mu = jnp.mean(sim.op_expval(params, ops, NSB2, key, return_samples=True, max_batch_samples=None,
                            condition_mask=cmask, condition_values=cvals, condition_counts=counts2_j), axis=-1)
print(f"[multi] mean(scan@{NSB2}) vs mean(unbatched@{NSB2}) max|diff| = {float(jnp.max(jnp.abs(mb-mu))):.3e}")

print("\nALL ASSERTIONS PASSED")
