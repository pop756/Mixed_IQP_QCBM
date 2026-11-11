"""Shared settings of the cosine closed-form rerun of the paper's training experiments (iqp_fastcf).

Datasets: the splits of the current manuscript. blobs and Ising (n = 16) from experiments/data/ (copies of the
old pipeline's paper_ours/data and ising/data, the rows the paper clustered); MNIST (50,000 / 10,000, n = 784) and
D-Wave (10,000 / 60,000, n = 484) from experiments/datasets/. Every dataset uses all qubit pairs as edges.
Training (all runs): stratified closed-form MMD^2 at m-bar = 2 and 6, Adam, lr 1e-3 with cosine decay to 0 over
15,000 steps, k_train = 2000, |A| <= 10. Test MMD^2: iqp_fastcf.make_evaluator (m-bar = 1..6, 10,000 subsets each).
"""
import itertools, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "packages" / "iqp_fastcf"))
from iqp_fastcf import walsh, sigma_for, all_pairs               # noqa: E402

STEPS, LR = 15000, 1e-3
EVAL_STEPS = (0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000)
SEEDS16 = [7, 11, 23, 42, 99, 314, 777, 1234, 2025, 2718]
SEEDS_BIG = [7, 11, 23, 1234, 2025]
INITS = ("global", "agnostic", "cluster")
D = REPO / "experiments"
DATASETS = {
    # name: (train csv, test csv, ancilla ladder, seeds, spectral subsample, (D2 sigma, D2 ancilla))
    "blobs": (D / "data/blobs/x_train.csv", D / "data/blobs/x_test.csv", (0, 1, 2, 3), SEEDS16, None, (1.93505, 3)),
    "ising": (D / "data/ising/x_train.csv", D / "data/ising/x_test.csv", (0, 1, 2, 3), SEEDS16, None, (1.03142, 2)),
    "mnist": (D / "datasets/MNIST_28x28_all/x_train.csv", D / "datasets/MNIST_28x28_all/x_test.csv", (0, 1, 2, 3, 4),
              SEEDS_BIG, 10000, (sigma_for(784, 2), 4)),
    "dwave": (D / "datasets/dwave/dwave_X_train.csv", D / "datasets/dwave/dwave_X_test.csv", (0, 1, 2, 3), SEEDS_BIG,
              10000, (sigma_for(484, 2), 3)),
}
RUNS = HERE / "runs"            # per-run summaries (committed)
PARAMS = HERE / "params"        # trained angles and full loss arrays (not committed)
CLUSTERS = HERE / "clusters"    # cluster labels per (dataset, a, seed) (committed)
RESULTS = HERE / "results"      # summary tables (committed)


def load_bits(path):
    raw = np.fromfile(path, dtype=np.uint8)
    nl = int(np.argmax(raw == ord("\n"))) + 1
    if nl > 1 and raw.size % nl == 0 and nl == 2 * np.count_nonzero(raw[:nl] == ord(",")) + 2:
        x = (raw.reshape(-1, nl)[:, 0:nl - 1:2] - ord("0")).astype(np.int8)      # fixed-width "b,b,...,b\n"
    else:
        x = np.loadtxt(path, delimiter=",", dtype=np.int8)
    assert set(np.unique(x)) <= {0, 1}, path
    return x


def data(ds):
    tr, te = DATASETS[ds][:2]
    return load_bits(tr), load_bits(te)


def sigmas(n):
    return [sigma_for(n, 2), sigma_for(n, 6)]


def theta(params, a):
    """Branch angles theta^(l) = W c (the kernel angle is 2 theta) from compiled angles c, (..., L, G)."""
    return np.einsum("lk,...kg->...lg", walsh(a), params)


def edges_of(n, onebody=False):
    return np.zeros((0, 2), dtype=np.int32) if onebody else all_pairs(n)


# --- exact quantities from closed forms -------------------------------------------------------------------------

def z1(th, n, edges):
    """One-body correlators <Z_j> = cos(2 th_j) prod_{k} cos(2 th_jk) of branches th (..., G)."""
    c = np.cos(2.0 * th)
    z = c[..., :n].copy()
    for e, (i, j) in enumerate(edges):
        z[..., i] *= c[..., n + e]
        z[..., j] *= c[..., n + e]
    return z


def born_probs16(th, n, edges):
    """Exact Born probabilities of one branch (n <= 20): p(x) = |2^-n sum_y (-1)^{x.y} e^{i f(y)}|^2,
    f(y) = sum_g theta_g prod_{j in g} s_j(y), s = 1 - 2y; index bit j = qubit j."""
    y = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    s = 1.0 - 2.0 * y
    f = s @ th[:n]
    if len(edges):
        f = f + (s[:, edges[:, 0]] * s[:, edges[:, 1]]) @ th[n:]
    amp = fwht(np.exp(1j * f)) / 2 ** n
    return np.abs(amp) ** 2


def fwht(v):
    """Unnormalized Walsh-Hadamard transform: out[x] = sum_y (-1)^{popcount(x & y)} v[y]."""
    v = np.array(v)
    N, h = len(v), 1
    while h < N:
        v = v.reshape(-1, 2, h)
        v = np.concatenate([v[:, :1] + v[:, 1:], v[:, :1] - v[:, 1:]], axis=1).reshape(N)
        h *= 2
    return v
