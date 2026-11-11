"""Stand-in for the evaluation module of the earlier pipeline: its exact 16-qubit helpers only (see common.py).

MMD^2 = sum_A w_A (z_A^model - z_A^data)^2 over all 2^16 subsets, with z the Walsh transform of a distribution and
w_A = alpha^(n - |A|) beta^|A|, alpha = (1 + b) / 2, beta = (1 - b) / 2, b = exp(-1 / (2 sigma^2)).
"""
import numpy as np
import torch

import common as C
from randomized_iqp.sampling import _fwht

N = C.VISIBLE_BITS
NS = 1 << N
_A = np.arange(NS, dtype=np.int64)
PC = np.zeros(NS, dtype=np.int64)
for _i in range(N):
    PC += (_A >> _i) & 1


def emp_z(bits):                                # bits: (M,N) int -> z_A of empirical dist
    idx = (bits.astype(np.int64) << np.arange(N)).sum(1)
    q = np.bincount(idx, minlength=NS).astype(np.float64); q /= q.sum()
    return _fwht(torch.as_tensor(q))


def mmd2_from_z(zp, zq, sigma):
    b = float(np.exp(-1.0 / (2.0 * sigma ** 2))); al = (1 + b) / 2; be = (1 - b) / 2
    w = torch.as_tensor(al ** (N - PC) * be ** PC, dtype=torch.float64)
    return float((w * (zp - zq) ** 2).sum())
