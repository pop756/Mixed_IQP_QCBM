"""Test MMD^2 of a model given only through samples, on the subsets of the paper's evaluator.

The paper's evaluator (iqp_fastcf.make_evaluator / iqp_hip_cf.MultiInitTrainer._ensure_ev) draws, for m-bar = 1..6,
keval = 10^4 subsets A ~ Bernoulli(m-bar/n)^n with the generator default_rng(1000 + m-bar), keeps 1 <= |A| <= 10, and
reports sum_A (z_A - t_A)^2 / keval with z_A exact and t_A the test moment; the value is the mean over m-bar. Here
z_A is replaced by the moment of N_s model samples, with the finite-sample bias removed:
E[(zhat - t)^2] = (z - t)^2 + (1 - z^2)/N_s, so (zhat - t)^2 - (1 - zhat^2)/(N_s - 1) is unbiased for (z - t)^2.
"""
import numpy as np

KEVAL, RMAX = 10_000, 10


def subsets(n, mb, keval=KEVAL, rmax=RMAX):
    mask = np.random.default_rng(1000 + mb).random((keval, n)) < mb / n
    lens = mask.sum(1)
    return [np.nonzero(row)[0] for row, ln in zip(mask, lens) if 1 <= ln <= rmax]


def _packed(x):
    """(N, n) 0/1 -> (n, W) uint64 bit columns (sample axis packed), and N."""
    N, n = x.shape
    pad = (-N) % 64
    xp = np.pad(x.T.astype(np.uint8), ((0, 0), (0, pad)))
    return np.ascontiguousarray(np.packbits(xp, axis=1, bitorder="little")).view(np.uint64), N


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def moments(x, As):
    """t_A = mean (-1)^{A.x} for every subset in As."""
    cols, N = _packed(x)
    out = np.empty(len(As))
    for i, A in enumerate(As):
        v = cols[A[0]].copy()
        for j in A[1:]:
            v ^= cols[j]
        ones = _POP[v.view(np.uint8)].sum()
        out[i] = (N - 2 * ones) / N
    return out


def moments_gpu(x, As, chunk=64):
    """moments() on the GPU (JAX): bit columns packed in uint32 along the sample axis, a zero column at index n
    pads every subset to RMAX, parities by XOR over the gathered columns, counts by population count."""
    import jax, jax.numpy as jnp
    N, n = x.shape
    pad = (-N) % 32
    xp = np.pad(x.T.astype(np.uint8), ((0, 1), (0, pad)))
    cols = jnp.asarray(np.ascontiguousarray(np.packbits(xp, axis=1, bitorder="little")).view(np.uint32))
    Ap = np.full((len(As), RMAX), n, dtype=np.int32)
    for i, A in enumerate(As):
        Ap[i, :len(A)] = A

    @jax.jit
    def f(a):
        v = jax.lax.reduce(cols[a], np.uint32(0), jax.lax.bitwise_xor, (1,))
        return jnp.sum(jax.lax.population_count(v).astype(jnp.int64), axis=1)
    ones = np.concatenate([np.asarray(f(jnp.asarray(Ap[i:i + chunk]))) for i in range(0, len(Ap) - chunk + 1, chunk)]
                          + ([np.asarray(f(jnp.asarray(np.pad(Ap[len(Ap) - len(Ap) % chunk:],
                                                               ((0, chunk - len(Ap) % chunk), (0, 0)),
                                                               constant_values=n))))[:len(Ap) % chunk]]
                             if len(Ap) % chunk else []))
    return (N - 2.0 * ones) / N


def test_mmd2(samples, x_test, mbars=range(1, 7), gpu=True):
    """(mean over m-bar, per m-bar list) of the bias-corrected sample-based test MMD^2."""
    n = x_test.shape[1]
    Ns = len(samples)
    mom = moments_gpu if gpu else moments
    per = []
    for mb in mbars:
        As = subsets(n, mb)
        zh, t = mom(samples, As), moments(x_test, As)
        per.append(float(np.sum((zh - t) ** 2 - (1 - zh ** 2) / (Ns - 1)) / KEVAL))
    return float(np.mean(per)), per
