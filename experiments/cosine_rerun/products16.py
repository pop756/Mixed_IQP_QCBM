"""Product-mixture baselines at n = 16 (blobs, Ising), evaluated exactly.

The two baselines of products.py for the 16-qubit datasets, where every subset can be summed:
- center: the untrained cluster center, branch l = the product distribution with the one-body moments of group l of
  the spectral clustering (clusters/<ds>/a<a>_seed<s>.npz), pi_l = |C_l| / N; a = 0: the global product;
- em: the EM-fitted L-component Bernoulli mixture of products.py (products.em_bernoulli, EM seeds 0..9).
Test MMD^2 = sum_A w_A (z_A(model) - t_A)^2 over all 2^16 subsets, z the Walsh transform of the exact mixture
distribution, t_A the test moments, w_A = p^|A| (1 - p)^(n - |A|) with p = m-bar / n, mean over m-bar = 1..6.

The article quotes these values from the earlier pipeline
(legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/summary.csv, rows center and em).
`check` compares: em agrees to round-off, center to about 1e-6 (the earlier pipeline stored the center angles in a
start artifact).
usage: products16.py [build]  ->  results/product_baselines_<ds>.csv        products16.py check
"""
import csv, sys
from pathlib import Path
import numpy as np
from common import CLUSTERS, DATASETS, RESULTS, data, fwht
from products import em_bernoulli

N = 16
LEGACY = Path(__file__).resolve().parents[2] / "legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/summary.csv"
PC = np.array([bin(i).count("1") for i in range(1 << N)])
W = [(m / N) ** PC * (1 - m / N) ** (N - PC) for m in range(1, 7)]


def mixture_probs(q, pi):
    """Exact distribution of a product mixture with P(x_j = 1) = q[l, j]; bit j of the index is qubit j."""
    out = np.zeros(1 << N)
    for ql, pl in zip(q, pi):
        p = np.ones(1)
        for j in range(N):
            p = np.concatenate([p * (1 - ql[j]), p * ql[j]])
        out += pl * p
    return out


def build(names):
    for ds in names:
        xtr, xte = data(ds)
        X = xtr.astype(np.float64)
        t = fwht(np.bincount(xte.astype(np.int64) @ (1 << np.arange(N)), minlength=1 << N) / len(xte))
        seeds, rows = DATASETS[ds][3], []
        for a in DATASETS[ds][2]:
            L = 1 << a
            models = []
            for s in (seeds if a else seeds[:1]):
                y = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")["train_y"].astype(int) if a else np.zeros(len(X), int)
                models.append(("center", s if a else -1, np.stack([X[y == l].mean(0) for l in range(L)]),
                               np.bincount(y, minlength=L) / len(y)))
            for s in (range(len(seeds)) if a else [0]):
                q, pi, _, _ = em_bernoulli(X, L, s)
                models.append(("em", s if a else -1, q, pi))
            for meth, s, q, pi in models:
                z = fwht(mixture_probs(q, pi))
                per = [float(w @ (z - t) ** 2) for w in W]
                rows.append(dict(dataset=ds, method=meth, a=a, seed=s, test_mmd2=float(np.mean(per)),
                                 **{f"mbar{j + 1}": v for j, v in enumerate(per)}))
            for meth in ("center", "em"):
                v = [r["test_mmd2"] for r in rows if r["a"] == a and r["method"] == meth]
                print(f"[{ds}] a={a} {meth}: {np.mean(v) * 1e3:.4f}e-3 ({len(v)} models)", flush=True)
        with open(RESULTS / f"product_baselines_{ds}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def check(names, tol=1e-5):
    leg = {(r["dataset"], int(r["ancilla"]), r["column"]): float(r["mean"]) for r in csv.DictReader(open(LEGACY))
           if r["column"] in ("center", "em") and r["dataset"] in names}
    worst = 0.0
    for ds in names:
        rows = list(csv.DictReader(open(RESULTS / f"product_baselines_{ds}.csv")))
        for a in DATASETS[ds][2]:
            for meth in ("center", "em"):
                new = np.mean([float(r["test_mmd2"]) for r in rows if int(r["a"]) == a and r["method"] == meth])
                d = abs(new - leg[(ds, a, meth)]) / leg[(ds, a, meth)]
                worst = max(worst, d)
                print(f"{ds:6s} a={a} {meth:6s} {new:.9e}  earlier pipeline {leg[(ds, a, meth)]:.9e}  rel. diff. {d:.1e}")
    print(f"largest relative difference {worst:.1e} (allowed {tol:.0e})")
    return worst <= tol


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if a not in ("build", "check")] or ["blobs", "ising"]
    if "check" in sys.argv[1:]:
        sys.exit(0 if check(names) else 1)
    build(names)
