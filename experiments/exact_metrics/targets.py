"""Exact generating distributions of the two n = 16 benchmarks, reconstructed from their generators, and a check
that the paper's train/test splits (experiments/data/) are draws from them.

blobs: qml_benchmarks generate_8blobs: one of the eight peak patterns (../datasets/blobs/8_blobs_dataset/
       16_spins_8_blobs_peak_spins.csv, +-1 -> 1/0) uniformly, each bit flipped with p = 0.05 (manuscript Eq. blobs).
ising: ../datasets/ising/2d_random_lattice_dataset/generator.py: np.random.seed(666); J = A * rand(16, 16) * 2 with A
       the periodic 4 x 4 grid adjacency (row-major nodes, networkx order), energy E(s) = -s^T J s / 2
       (qml_benchmarks.data.ising.energy), p ~ exp(-E/T), T = 3, s = 2 z - 1.
Index of a bit string: sum_j z_j 2^j (bit j = qubit j), as in ../gd_escape/exact16.py.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DS = HERE.parent / "datasets"
sys.path.insert(0, str(HERE.parent / "cosine_rerun"))
from common import data                                                                # noqa: E402

N = 16
BITS = (np.arange(2 ** N)[:, None] >> np.arange(N)) & 1


def blobs_target(p_flip=0.05):
    peaks = (np.loadtxt(DS / "blobs/8_blobs_dataset/16_spins_8_blobs_peak_spins.csv", delimiter=",") > 0).astype(int)
    p = np.zeros(2 ** N)
    for b in peaks:
        d = (BITS != b).sum(1)
        p += p_flip ** d * (1 - p_flip) ** (N - d)
    return p / len(peaks)


def ising_couplings():
    import networkx as nx
    rs = np.random.RandomState(666)
    A = nx.adjacency_matrix(nx.grid_2d_graph(4, 4, periodic=True)).toarray()
    return A * rs.rand(*A.shape) * 2


def ising_target(T=3.0):
    J = ising_couplings()
    s = 2.0 * BITS - 1.0
    e = -np.einsum("xi,ij,xj->x", s, J, s) / 2.0
    w = np.exp(-(e - e.min()) / T)
    return w / w.sum()


TARGETS = {"blobs": blobs_target, "ising": ising_target}


def empirical(x):
    return np.bincount((x.astype(np.int64) << np.arange(N)).sum(1), minlength=2 ** N) / len(x)


def check(ds):
    """Two-sample agreement of the paper's splits with the reconstructed target: the log-likelihood of the test
    split under the target against its value expected for an exact sample, and the train-test TV against the
    target-test TV."""
    p = TARGETS[ds]()
    x_tr, x_te = data(ds)
    rng = np.random.default_rng(0)
    nll = -np.mean(np.log(p[(x_te.astype(np.int64) << np.arange(N)).sum(1)]))
    ent = -np.sum(p * np.log(np.where(p > 0, p, 1)))
    sd = np.sqrt((p * np.log(np.where(p > 0, p, 1)) ** 2).sum() - ent ** 2) / np.sqrt(len(x_te))
    pe = empirical(x_te)
    draw = empirical((rng.choice(2 ** N, size=len(x_te), p=p)[:, None] >> np.arange(N)) & 1)
    print(f"{ds}: test NLL under target {nll:.4f} nats vs entropy {ent:.4f} +- {sd:.4f};  "
          f"TV(test, target) {0.5 * np.abs(pe - p).sum():.4f} vs exact-sample TV {0.5 * np.abs(draw - p).sum():.4f}")


if __name__ == "__main__":
    for ds in ("blobs", "ising"):
        check(ds)
