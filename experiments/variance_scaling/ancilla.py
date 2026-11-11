"""Loss variance on the initialization patch against the ancilla count a at fixed n (Sec. III: Theorems 1 and 2 hold
for L = 2^a = O(poly(n)) branches, so the variance must not decay exponentially in a).

Same model, loss, starts, datasets, and patch as scaling.py (all-pairs order-2 gates, training loss at m-bar = 2 on
10^4 fixed subsets with |A| <= 10 and exact correlators; uniform noise of half-width pi / (2 sqrt(L G)) on every
compiled angle about each centre; random reference: every branch angle uniform on [0, pi)), at fixed n = N_FIX and
a = 0..A_MAX. At a = 0 the cluster centre equals the global one.
usage: ancilla.py <dataset> [K]  ->  results/ancilla_<dataset>.json
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scaling import dataset, centre, MBAR, SCHEMES                                   # noqa: E402
from iqp_fastcf import make_evaluator, all_pairs, walsh                            # noqa: E402

N_FIX, A_MAX = 64, 5


def main():
    name = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n = N_FIX
    x = dataset(name, n)
    edges = all_pairs(n)
    G = n + len(edges)
    out = dict(dataset=name, n=n, mbar=MBAR, K=K, As=[], schemes={s: [] for s in SCHEMES})
    for a in range(A_MAX + 1):
        L = 2 ** a
        W = walsh(a)
        r = (np.pi / 2) / math.sqrt(L * G)
        ev = make_evaluator(n, a, edges, x)
        rng = np.random.default_rng(1000 + a)
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
            for b in range(0, K, 25):
                _, per = ev(c[b:b + 25], pi[b:b + 25])
                vals.append(per[:, MBAR - 1])
            v = np.concatenate(vals)
            out["schemes"][s].append(dict(a=a, mean=float(v.mean()), var=float(v.var(ddof=1)),
                                          var_se=float(v.var(ddof=1) * math.sqrt(2.0 / (K - 1)))))
        out["As"].append(a)
        print(f"{name} a={a}: " + "  ".join(f"{s} var {out['schemes'][s][-1]['var']:.3e}" for s in SCHEMES)
              + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
        (HERE / "results").mkdir(exist_ok=True)
        json.dump(out, open(HERE / "results" / f"ancilla_{name}.json", "w"))


if __name__ == "__main__":
    main()
