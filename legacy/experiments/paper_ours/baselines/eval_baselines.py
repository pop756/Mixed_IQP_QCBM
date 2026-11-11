"""Evaluation protocol for the classical baselines -- matches evaluate_sweep.py.

REUSES the paper's machinery (no kernel reimplementation):
  n <= 16 (blobs, ising): EXACT channel. Model gives an exact p(v) over 2^n
    states; z = _fwht(p); MMD^2 via evaluate_sweep.mmd2_from_z (same Walsh
    weights w_A = alpha^(n-|A|) beta^|A|). Central value = model vs FULL data;
    error band = data bootstrap with evaluate_sweep.SEEDS (5x), exactly like
    rows_exact().
  n > 16 (digits, dwave): SAMPLE channel. Model samples (block Gibbs, one
    sample per independent chain after long burn-in) vs data subsamples with
    randomized_iqp.metrics.mmd2_hamming_rbf, mirroring evaluate_sweep.true_rows:
    per seed, rng = np.random.default_rng(seed); both sides capped at 2000.
    NOTE (channel difference, documented): the IQP models in the paper are
    scored with the unbiased correlator estimator (gen.mmd_loss_iqp); a
    classical model has no correlator surrogate, so the honest analogue is the
    same sample-vs-sample protocol the paper's "True" floor uses.

CSV convention for rows that sit alongside the paper CSVs
(scheme,ancilla,L,pauli,sigma,mmd2_mean,mmd2_std,n_seeds):
  ancilla = -1 and L = 0 mark classical-baseline rows (pandas-loadable ints,
  never collide with real ancilla counts >= 0 / L = 2^a >= 1).

Import with PO_DS set (this module imports the dataset-parameterized common).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAPER_OURS = HERE.parent
if str(PAPER_OURS) not in sys.path:
    sys.path.insert(0, str(PAPER_OURS))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import common as C  # noqa: E402

SIGMAS = [C.sigma_of_pauli(m) for m in C.PAULI_SWEEP]
SIG2PAULI = {round(s, 5): m for s, m in zip(SIGMAS, C.PAULI_SWEEP)}
SEEDS = [1234, 2025, 7, 99, 314]          # = evaluate_sweep.SEEDS
EXACT = C.VISIBLE_BITS <= 16
N_EVAL = 2000                              # = evaluate_sweep true_rows cap

if EXACT:
    import evaluate_sweep as ES            # reuse emp_z / mmd2_from_z / _fwht path
    from randomized_iqp.sampling import _fwht
else:
    from randomized_iqp.metrics import mmd2_hamming_rbf


def load_split(name: str) -> np.ndarray:
    return np.loadtxt(C.DATA_DIR / f"x_{name}.csv", delimiter=",", dtype=np.int32)


# ---------------- exact channel (n <= 16) ----------------
def mmd2_exact_logp(logp: torch.Tensor, data_bits: np.ndarray,
                    bootstrap: bool = True) -> dict[float, tuple[float, float]]:
    """Exact MMD^2(model, data) per sigma. Returns {sigma: (central, boot_std)}.
    Central = model vs FULL data; std = data bootstrap (evaluate_sweep protocol)."""
    zp = _fwht(torch.exp(logp))
    zq = ES.emp_z(data_bits)
    if bootstrap:
        zb = [ES.emp_z(data_bits[np.random.default_rng(s).choice(
            len(data_bits), len(data_bits), True)]) for s in SEEDS]
    out = {}
    for s in SIGMAS:
        central = ES.mmd2_from_z(zp, zq, s)
        std = 0.0
        if bootstrap:
            boot = np.asarray([ES.mmd2_from_z(zp, z, s) for z in zb])
            std = float(boot.std(ddof=1))
        out[round(s, 5)] = (float(central), std)
    return out


# ---------------- sample channel (n > 16) ----------------
def mmd2_samples(model_bits: np.ndarray, data_bits: np.ndarray,
                 seed: int) -> dict[float, float]:
    """MMD^2(model samples, data subsample) per sigma; mirrors true_rows: the
    data side is subsampled to <= 2000 with np.random.default_rng(seed)."""
    r = np.random.default_rng(seed)
    ib = r.choice(len(data_bits), min(N_EVAL, len(data_bits)), replace=False)
    x = torch.as_tensor(np.asarray(model_bits, dtype=np.int64))
    y = torch.as_tensor(data_bits[ib].astype(np.int64))
    return {round(s, 5): float(mmd2_hamming_rbf(x, y, float(s))) for s in SIGMAS}


def mixing_diagnostics(rbm, burn_in: int, seeds=(11, 12)) -> dict:
    """Gibbs mixing check: two independent chain populations at burn_in and one
    at 2*burn_in; compare first/second moments (per-bit means, pairwise corr)."""
    a = rbm.sample(N_EVAL, burn_in, seed=seeds[0]).astype(np.float64)
    b = rbm.sample(N_EVAL, burn_in, seed=seeds[1]).astype(np.float64)
    c = rbm.sample(N_EVAL, 2 * burn_in, seed=seeds[0] + 100).astype(np.float64)

    def corr(x):
        return (x.T @ x) / len(x)

    def cmp(x, y):
        return {"mean_l1": float(np.abs(x.mean(0) - y.mean(0)).mean()),
                "mean_linf": float(np.abs(x.mean(0) - y.mean(0)).max()),
                "corr_l1": float(np.abs(corr(x) - corr(y)).mean())}

    return {"burn_in": int(burn_in),
            "seed_vs_seed@B": cmp(a, b),
            "B_vs_2B": cmp(a, c)}


def csv_rows(scheme: str, per_sigma: dict[float, tuple[float, float]],
             n_seeds: int) -> list[dict]:
    """Rows in the paper CSV schema; ancilla=-1, L=0 mark classical baselines."""
    return [{"scheme": scheme, "ancilla": -1, "L": 0,
             "pauli": SIG2PAULI[sk], "sigma": sk,
             "mmd2_mean": v[0], "mmd2_std": v[1], "n_seeds": n_seeds}
            for sk, v in sorted(per_sigma.items(), key=lambda kv: SIG2PAULI[kv[0]])]
