"""Stand-in for the dataset module of the earlier pipeline, for the 16-qubit classical baselines.

run_baselines.py and eval_baselines.py import `common` and `evaluate_sweep`, two shared modules of the earlier
training pipeline that are not part of this repository. This file and evaluate_sweep.py next to it define the few
names those two scripts use, with the values and function bodies of the earlier modules, for the two 16-qubit
datasets of the article (PO_DS=blobs or PO_DS=ising). The data are the splits of experiments/data/.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]                                   # legacy/experiments
SHARED = PAPER / "randomized_iqp_mnist8x8"
VENDOR = SHARED / "vendor"
for _p in (str(SHARED), str(VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DS = os.environ.get("PO_DS", "blobs")
if DS not in ("blobs", "ising"):
    raise ValueError(f"unknown PO_DS={DS!r}; this stand-in knows blobs and ising")
VISIBLE_BITS = 16
DATA_DIR = HERE.parents[3] / "experiments" / "data" / DS
PAULI_SWEEP = [1, 2, 3, 4, 5, 6]


def sigma_of_pauli(m, n=None):
    """Bandwidth whose kernel puts mean Pauli weight m on the sampled correlators.

    c = 1 - 2m/n is the kernel constant. For m < n/2 it is positive and this is
    the usual Gaussian-Hamming sigma. For m > n/2 it is NEGATIVE, which is still
    a positive-definite kernel (all Fourier weights stay positive while |c| < 1)
    and is the only way to reach the high-order half of the spectrum -- the plain
    family caps the mean weight at n/2. That branch is returned as a NEGATIVE
    sigma, which iqpopt's p_MMD reads through jnp.sign. m = n is the pure parity
    operator Z_1..Z_n (c = -1, sigma -> -inf); it is clamped to -1e3, which puts
    weight (1-5e-7)^n on the all-ones string.
    """
    n = VISIBLE_BITS if n is None else n
    c = 1.0 - 2.0 * m / n
    if abs(c) >= 1.0:                       # m = 0 or m = n: the degenerate ends
        return -1e3 if c < 0 else float("inf")
    if abs(c) < 1e-12:                      # m = n/2 exactly: the delta kernel
        return 1e-3
    s = float(np.sqrt(-1.0 / (2.0 * np.log(abs(c)))))
    return s if c > 0 else -s
