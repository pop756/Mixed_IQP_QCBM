"""Loss variance on the initialization patch against the system size n (Sec. III: local freedom from barren
plateaus means Var_Theta[L] = Omega(1/poly(n)) on an inverse-polynomial hypercube about the centre).

Model: uniform-graph mixed IQP-QCBM, all-pairs order-2 gates, a = 2 (L = 4). Loss: MMD^2 at the smaller training
bandwidth (m-bar = 2) between the mixture and the dataset, estimated with the paper's evaluator (iqp_fastcf
make_evaluator: 10^4 fixed subsets A ~ Bernoulli(m-bar/n)^n with |A| <= 10, exact correlators), so the same fixed
subsets enter every draw and the spread over Theta is the spread of the loss itself.
Starts (K draws each):
  agnostic, global, cluster : the centres of Sec. II F (cluster: k-means groups, weights = group masses; the paper's
                              spectral clustering is replaced by k-means for this sweep), plus independent uniform
                              noise of half-width r = pi / (2 sqrt(L G)) on every compiled angle (the campaign's patch);
  random                    : every branch angle uniform on [0, pi) (uniform weights): the barren-plateau reference.
Datasets: blobs(n): eight random n-bit patterns (fixed seed), each bit flipped with p = 0.05, 5000 strings;
          dwave(n): the first n spins of the 484-spin D-Wave training set (first 10^4 configurations).
usage: scaling.py <dataset> [K]  ->  results/scaling_<dataset>.json
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cosine_rerun"))
from common import data                                                             # noqa: E402
from iqp_fastcf import make_evaluator, all_pairs, walsh                            # noqa: E402
from iqp_fastcf.inits import kmeans                                                # noqa: E402

A_ANC, MBAR = 2, 2
NS = [12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 484]
SCHEMES = ("agnostic", "global", "cluster", "random")


def blobs(n, N=5000, seed=0):
    rng = np.random.default_rng(seed)
    pats = rng.integers(0, 2, (8, n))
    x = pats[rng.integers(0, 8, N)]
    return (x ^ (rng.random((N, n)) < 0.05)).astype(np.int8)


def dataset(name, n):
    if name == "blobs":
        return blobs(n)
    return data("dwave")[0][:, :n].copy()


def centre(x, scheme, L, G, n):
    """Branch angles (L, G) and weights (L,) of a centre (one-body angles only)."""
    th = np.zeros((L, G))
    if scheme == "agnostic":
        th[:, :n] = np.pi / 4
        return th, np.full(L, 1.0 / L)
    if scheme == "global":
        th[:, :n] = 0.5 * np.arccos(1.0 - 2.0 * x.mean(0))
        return th, np.full(L, 1.0 / L)
    lab = kmeans(x, L, seed=1234)
    pi = np.bincount(lab, minlength=L) / len(lab)
    for l in range(L):
        th[l, :n] = 0.5 * np.arccos(1.0 - 2.0 * x[lab == l].mean(0))
    return th, pi


def main():
    name = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    L = 2 ** A_ANC
    W = walsh(A_ANC)
    out = dict(dataset=name, a=A_ANC, mbar=MBAR, K=K, ns=[], schemes={s: [] for s in SCHEMES})
    for n in NS:
        if name == "dwave" and n > 484:
            continue
        x = dataset(name, n)
        edges = all_pairs(n)
        G = n + len(edges)
        r = (np.pi / 2) / math.sqrt(L * G)
        ev = make_evaluator(n, A_ANC, edges, x)
        rng = np.random.default_rng(n)
        t0 = time.perf_counter()
        for s in SCHEMES:
            if s == "random":
                th = rng.uniform(0, np.pi, (K, L, G))
                pi = np.full((K, L), 1.0 / L)
                c = np.einsum("lk,mkg->mlg", W, th) / L
            else:
                th0, pi0 = centre(x, s, L, G, n)
                c0 = W @ th0 / L
                c = c0[None] + rng.uniform(-r, r, (K, L, G))
                pi = np.repeat(pi0[None], K, 0)
            vals = []
            for b in range(0, K, 50):
                _, per = ev(c[b:b + 50], pi[b:b + 50])
                vals.append(per[:, MBAR - 1])
            v = np.concatenate(vals)
            out["schemes"][s].append(dict(n=n, mean=float(v.mean()), var=float(v.var(ddof=1)),
                                          var_se=float(v.var(ddof=1) * math.sqrt(2.0 / (K - 1)))))
        out["ns"].append(n)
        print(f"{name} n={n}: " + "  ".join(f"{s} var {out['schemes'][s][-1]['var']:.3e}" for s in SCHEMES)
              + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
        (HERE / "results").mkdir(exist_ok=True)
        json.dump(out, open(HERE / "results" / f"scaling_{name}.json", "w"))


if __name__ == "__main__":
    main()
