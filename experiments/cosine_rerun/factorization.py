"""Finite-size residual diagnostics of the grouped data (fig_app_assumption, SM Sec. on Assumption 1).

The legacy script legacy/experiments/paper_ours/factorization_residuals.py with three changes only: the training sets
are the rerun's full splits (common.data), the groups are the rerun's spectral clusters of training seed 1234
(clusters/<ds>/a<a>_seed1234.npz), and the output goes to results/. Everything else is copied verbatim:
eps_k = group-weighted median over subsets |A| = k of |t_A - prod_j t_j|, its 95th percentile and maximum, the
surrogate floor from independent bits with each group's marginals and size, and the witness factors.

Run: python factorization.py   ->   results/factorization_residuals.csv
"""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "factorization_residuals.csv"
from common import CLUSTERS, data  # noqa: E402

# dataset, display name, ancilla counts whose clusterings exist
DATASETS = [
    ("blobs", "blobs", [1, 2, 3]),
    ("ising", "Ising", [1, 2, 3]),
    ("mnist", "MNIST", [1, 2, 3, 4]),
    ("dwave", "D-Wave", [1, 2, 3]),
]
BODIES = (2, 3, 4, 5, 6)
NSUB = 40000          # random subsets per body order when the pool is large
NULL_REPS = 5         # surrogate draws per group
RNG = np.random.default_rng(20260728)


def spins(path: Path) -> np.ndarray:
    """Training bits as +-1 spins, s_j = 1 - 2 x_j."""
    x = np.loadtxt(path, delimiter=",", dtype=np.int8)
    return (1 - 2 * x).astype(np.int8)


def subsets(pool: np.ndarray, k: int) -> np.ndarray:
    n = len(pool)
    if math.comb(n, k) <= NSUB:
        return pool[np.array(list(itertools.combinations(range(n), k)),
                             dtype=np.int32)]
    out = np.empty((NSUB, k), dtype=np.int32)
    for i in range(NSUB):
        out[i] = pool[RNG.choice(n, size=k, replace=False)]
    return out


def residuals(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """|t_A - prod_j t_j| for every subset in idx, from the sample block S."""
    N = S.shape[0]
    t1 = S.mean(0, dtype=np.float64)
    out = np.empty(len(idx), dtype=np.float64)
    step = max(1, int(4e8 // max(N, 1)))
    for s in range(0, len(idx), step):
        blk = idx[s:s + step]
        P = S[:, blk[:, 0]].astype(np.int8, copy=True)
        for j in range(1, blk.shape[1]):
            P *= S[:, blk[:, j]]
        tA = P.sum(0, dtype=np.int32) / N
        out[s:s + step] = np.abs(tA - np.prod(t1[blk], axis=1))
    return out


def surrogate(S: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Median and 95th percentile of the residual for independent bits with the
    same marginals and group size: the finite-sample floor of both statistics."""
    N, n = S.shape
    p = (1 - S.mean(0, dtype=np.float64)) / 2      # P(x_j = 1)
    med, q95 = [], []
    for _ in range(NULL_REPS):
        X = (RNG.random((N, n)) < p).astype(np.int8)
        r = residuals((1 - 2 * X).astype(np.int8), idx)
        med.append(float(np.median(r)))
        q95.append(float(np.quantile(r, 0.95)))
    return float(np.median(med)), float(np.median(q95))


def main() -> None:
    rows = []
    for ds, disp, ancillas in DATASETS:
        S = (1 - 2 * data(ds)[0]).astype(np.int8)
        N, n = S.shape
        pool = np.flatnonzero(np.abs(S.mean(0, dtype=np.float64)) < 1.0)
        idx = {k: subsets(pool.astype(np.int32), k) for k in BODIES}
        print(f"[{ds}] N={N} n={n} pool={len(pool)} subsets/order="
              + " ".join(f"{k}:{len(idx[k])}" for k in BODIES), flush=True)

        groupings = [(1, np.zeros(N, dtype=int))]
        for a in ancillas:
            y = np.load(CLUSTERS / ds / f"a{a}_seed1234.npz")["train_y"].astype(int)
            assert y.shape[0] == N
            groupings.append((1 << a, y))

        for L, y in groupings:
            masks = [y == lab for lab in range(L)]
            masks = [m for m in masks if m.any()]
            wts = np.array([m.sum() / N for m in masks])
            # Assumption 2: witness factor of the most massive group, and the
            # weakest group's best coordinate
            wit = [float(np.max(1.0 - S[m].mean(0, dtype=np.float64) ** 2))
                   for m in masks]
            heavy = int(np.argmax(wts))
            for k in BODIES:
                eps, nul, q95, nq95, mx = [], [], [], [], []
                for m in masks:
                    Sl = np.ascontiguousarray(S[m])
                    r = residuals(Sl, idx[k])
                    eps.append(float(np.median(r)))
                    q95.append(float(np.quantile(r, 0.95)))
                    mx.append(float(r.max()))
                    z_med, z_q95 = surrogate(Sl, idx[k])
                    nul.append(z_med)
                    nq95.append(z_q95)
                e = float(np.dot(wts, eps))
                rows.append(dict(
                    dataset=ds, display=disp, n=n, pool=len(pool), L=L, k=k,
                    n_groups=len(masks), eps=e,
                    eps_null=float(np.dot(wts, nul)),
                    eps_q95=float(np.dot(wts, q95)),
                    eps_null_q95=float(np.dot(wts, nq95)),
                    eps_max=float(np.max(mx)),
                    c_over_n=e ** (2.0 / k),
                    c_over_n_max=float(np.max(mx)) ** (2.0 / k),
                    pi_heavy=float(wts[heavy]),
                    witness_heavy=wit[heavy],
                    witness_worst=float(np.min(wit))))
                print(f"  L={L:2d} k={k}  eps={e:.4e}  null={rows[-1]['eps_null']:.4e}"
                      f"  q95={rows[-1]['eps_q95']:.4e}  max={rows[-1]['eps_max']:.4e}"
                      f"  C/n={rows[-1]['c_over_n']:.4f}"
                      f"  wit={wit[heavy]:.3f}/{min(wit):.3f}", flush=True)

    with OUT.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
