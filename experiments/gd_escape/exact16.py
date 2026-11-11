"""Exact n = 16 mixed IQP-QCBM in JAX (float64): Born probabilities by a Walsh-Hadamard transform, the exact
training MMD^2 over all 2^n subsets (no subset sampling, no truncation), branch diversity, and the coincidence
Hessian block M of Proposition coinc-hessian. Shared by the escape experiment (escape.py) and the exact
likelihood metrics (../exact_metrics/).

Conventions are those of ../cosine_rerun/common.py: compiled angles c (L, G), branch angles theta = W c, circuit
U = exp(i sum_g theta_g X_g), one-body correlator cos(2 theta_j) at zero two-body angles; G = n + |E| with E all
pairs in iqp_fastcf.all_pairs order. Bit j of a basis-state index is qubit j.
"""
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cosine_rerun"))
from common import data, sigmas, born_probs16, fwht as fwht_np                       # noqa: E402
from iqp_fastcf import all_pairs, walsh                                              # noqa: E402

N16 = 16


def _hadamard(k):
    h = np.ones((1, 1))
    for _ in range(k):
        h = np.block([[h, h], [h, -h]])
    return h


_H = {}


def fwht(v):
    """Unnormalized Walsh-Hadamard transform along the last axis (length 2^n): out[x] = sum_y (-1)^{x.y} v[y], as
    H_hi V H_lo with V the (2^(n - n/2), 2^(n/2)) reshape (H_{2^n} = H_hi (x) H_lo). Two matmuls instead of n butterfly
    stages: XLA fuses a butterfly chain into an expression tree that grows as 2^n."""
    shp = v.shape
    n = int(np.log2(shp[-1]))
    lo = n // 2
    hi = n - lo
    for k in (lo, hi):
        if k not in _H:
            _H[k] = _hadamard(k)                         # numpy: a constant in every trace
    V = v.reshape(shp[:-1] + (2 ** hi, 2 ** lo))
    out = jnp.einsum("ab,...bc,cd->...ad", _H[hi].astype(v.dtype), V, _H[lo].astype(v.dtype))
    return out.reshape(shp)


class Exact16:
    """Everything exact for one n = 16 dataset with all-pairs edges."""

    def __init__(self, ds, a):
        x_tr, x_te = data(ds)
        n = x_tr.shape[1]
        assert n == N16
        self.ds, self.n, self.a, self.L = ds, n, a, 2 ** a
        self.edges = all_pairs(n)
        self.G = n + len(self.edges)
        idx = np.arange(2 ** n)
        bits = (idx[:, None] >> np.arange(n)) & 1
        s = 1.0 - 2.0 * bits
        # phase matrix: f(y) = S[y] @ theta with S = [s_j, s_j s_k]
        self.S = jnp.asarray(np.concatenate([s, s[:, self.edges[:, 0]] * s[:, self.edges[:, 1]]], axis=1))
        pc = bits.sum(1)
        # training weights: mean over the two training bandwidths (m-bar = 2, 6) of p^|A| (1 - p)^(n - |A|)
        w = np.zeros(2 ** n)
        for sg in sigmas(n):
            p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sg ** 2)))
            w += p ** pc * (1.0 - p) ** (n - pc)
        w /= len(sigmas(n))
        w[0] = 0.0                                     # A = {} carries z = t = 1
        self.w = jnp.asarray(w)
        self.W = jnp.asarray(walsh(a).astype(np.float64))
        enc = (x_tr.astype(np.int64) << np.arange(n)).sum(1)
        self.p_tr = np.bincount(enc, minlength=2 ** n) / len(enc)
        enc = (x_te.astype(np.int64) << np.arange(n)).sum(1)
        self.p_te = np.bincount(enc, minlength=2 ** n) / len(enc)
        self.t = jnp.asarray(fwht_np(self.p_tr))       # t_A = E (-1)^{A.y}
        self.x_tr, self.x_te = x_tr, x_te

    # --- model ------------------------------------------------------------------------------------------------
    def branch_probs(self, th):
        """(L, G) branch angles -> (L, 2^n) Born probabilities."""
        f = th @ self.S.T
        amp = fwht(jnp.exp(1j * f)) / 2 ** self.n
        return jnp.abs(amp) ** 2

    def branch_z(self, th):
        """(L, G) -> (L, 2^n) correlators z_A of every branch (index A as a bit mask)."""
        return fwht(self.branch_probs(th))

    def loss_c(self, c, pi):
        """Exact training MMD^2 (sum over all nonempty A) of compiled angles c (L, G), weights pi (L,)."""
        z = pi @ self.branch_z(self.W @ c)
        return jnp.sum(self.w * (z - self.t) ** 2)

    def diversity(self, c, pi):
        """Exact D_branch^2 at the training weights: sum_A w_A Var_{l~pi} z_A^(l)."""
        zl = self.branch_z(self.W @ c)
        zm = pi @ zl
        return jnp.sum(self.w * (pi @ (zl - zm) ** 2))

    # --- centres ----------------------------------------------------------------------------------------------
    def coincident_centre(self, scheme):
        """Branch angles of the data-agnostic or global centre (one-body only), as compiled angles (L, G)."""
        th = np.zeros(self.G)
        if scheme == "agnostic":
            th[:self.n] = np.pi / 4
        else:
            th[:self.n] = 0.5 * np.arccos(1.0 - 2.0 * self.x_tr.mean(0))
        c = np.zeros((self.L, self.G))
        c[0] = th                                       # S = {} row: every branch gets th
        return c

    def m_block(self, th_bar):
        """Coincidence Hessian block M_{GG'} = sum_A 2 w_A (z_A - t_A) d_G d_G' z_A at the shared angles th_bar (G,)."""
        r = fwht(self.branch_probs(th_bar[None]))[0] - self.t
        r = jax.lax.stop_gradient(r)

        def f(th):
            return jnp.sum(2.0 * self.w * r * fwht(self.branch_probs(th[None]))[0])
        return jax.hessian(f)(th_bar)

    def loss_single(self, th):
        """Exact training MMD^2 of one ancilla-free circuit with branch angles th (G,)."""
        z = fwht(self.branch_probs(th[None]))[0]
        return jnp.sum(self.w * (z - self.t) ** 2)

    def single_hessian(self, th):
        """Hessian of the single ancilla-free circuit's exact loss at th (G,)."""
        return jax.hessian(self.loss_single)(th)


def self_check():
    """The JAX Born probabilities match common.born_probs16, the loss matches the correlator sum."""
    E = Exact16("ising", 1)
    rng = np.random.default_rng(0)
    th = rng.uniform(-0.4, 0.4, E.G)
    p_j = np.asarray(E.branch_probs(jnp.asarray(th[None])))[0]
    p_n = born_probs16(th, E.n, E.edges)
    assert np.max(np.abs(p_j - p_n)) < 1e-14, np.max(np.abs(p_j - p_n))
    assert abs(p_j.sum() - 1) < 1e-12
    # one-body correlators against the closed form at zero two-body angles
    th1 = np.zeros(E.G); th1[:E.n] = rng.uniform(0, 1.5, E.n)
    z = np.asarray(E.branch_z(jnp.asarray(th1[None])))[0]
    assert np.max(np.abs(z[1 << np.arange(E.n)] - np.cos(2 * th1[:E.n]))) < 1e-12
    print("exact16 self-check ok")


if __name__ == "__main__":
    self_check()
