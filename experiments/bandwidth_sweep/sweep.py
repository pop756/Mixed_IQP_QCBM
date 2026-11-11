"""Branch diversity, mode-separation index and test MMD^2 of the Fig. 4 models at every evaluation bandwidth
m-bar = 1..6, and their Spearman correlations (Fig. 4 shows one bandwidth per dataset; this is the full sweep).

Models: ../cosine_rerun/params/<ds>_a<a>_main.npz at the sizes of Fig. 4 (blobs a = 3, Ising a = 2, D-Wave a = 3,
MNIST a = 4). For each m-bar the kernel is the Hamming Gaussian with p = m-bar/n (sigma = sigma_for(n, m-bar)):
  n = 16 : exact sums over all 2^16 subsets with w_A = p^|A| (1 - p)^(n - |A|),
  n > 16 : 10^4 subsets A ~ Bernoulli(p)^n (the uniforms of analyze.diversity, default_rng(20260728), thresholded
           at p, so the Fig. 4 bandwidth reproduces its values), 1 <= |A| <= 10 kept, weight
           1/10^4 each, z_A exact (analyze.branch_z), as analyze.diversity does at its one bandwidth.
D^2 = sum_A w_A Var_pi z_A, test MMD^2 = sum_A w_A (z_A(mix) - t_A)^2 with t_A from the test split, N_mode from the
exact one-body correlators (bandwidth-free). Spearman rho over all models, 95 % percentile bootstrap (10^4, seed
12345), as paper_numbers.py.
usage: sweep.py  ->  results/sweep_points.csv, results/sweep_spearman.csv, results/sweep_table.md
"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cosine_rerun"))
from common import data, theta, born_probs16, fwht, z1, PARAMS                      # noqa: E402
from analyze import branch_z, mode_index                                            # noqa: E402

SIZES = {"blobs": 3, "ising": 2, "dwave": 3, "mnist": 4}
FIG4_MBAR = {"blobs": 1, "ising": 3, "dwave": 2, "mnist": 2}
RES = HERE / "results"


def main():
    pts = []
    for ds, a in SIZES.items():
        z = np.load(PARAMS / f"{ds}_a{a}_main.npz")
        P, pi, seeds, names, edges = z["params"], z["pi"], z["seeds"], z["names"], z["edges"]
        xtr, xte = data(ds)
        n = xte.shape[1]
        TH = theta(P, a)
        nm = [mode_index(z1(TH[m], n, edges), pi[m], xte.astype(np.float64)) for m in range(len(P))]
        if n <= 16:
            t = fwht(np.bincount(xte.astype(np.int64) @ (1 << np.arange(n)), minlength=2 ** n) / len(xte))
            pc = np.array([bin(i).count("1") for i in range(2 ** n)])
            Zs = [np.stack([fwht(born_probs16(th, n, edges)) for th in TH[m]]) for m in range(len(P))]
        for mb in range(1, 7):
            p = mb / n
            if n <= 16:
                w = p ** pc * (1 - p) ** (n - pc)
                w[0] = 0.0
                tt, Zm = t, Zs
            else:
                ops = np.random.default_rng(20260728).random((10_000, n)) < p     # analyze.diversity's draw
                k = ops.sum(1)
                keep = (k >= 1) & (k <= 10)
                sub = np.full((keep.sum(), 10), n, dtype=np.int64)
                rr, cc = np.nonzero(ops[keep])
                sub[rr, np.concatenate([np.arange(c) for c in k[keep]])] = cc
                opsf = ops[keep].T.astype(np.float32)
                tt = sum((1 - 2 * ((xte[i:i + 5000].astype(np.float32) @ opsf) % 2)).sum(0).astype(np.float64)
                         for i in range(0, len(xte), 5000)) / len(xte)
                w = np.full(len(tt), 1.0 / len(ops))
                Zm = [branch_z(n, edges, TH[m], sub) for m in range(len(P))]
            for m in range(len(P)):
                Z = Zm[m]
                zbar, zsq = pi[m] @ Z, pi[m] @ (Z * Z)
                pts.append(dict(dataset=ds, a=a, mbar=mb, init=str(names[m]), seed=int(seeds[m]),
                                D2=float(w @ (zsq - zbar ** 2)), mmd2=float(w @ ((zbar - tt) ** 2)), N_mode=nm[m]))
            print(ds, "mbar", mb, flush=True)
    RES.mkdir(exist_ok=True)
    with open(RES / "sweep_points.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pts[0]))
        wr.writeheader()
        wr.writerows(pts)
    rows = []
    for ds in SIZES:
        for mb in list(range(1, 7)) + ["avg"]:
            if mb == "avg":
                sel = [r for r in pts if r["dataset"] == ds]
                keys = sorted({(r["init"], r["seed"]) for r in sel})
                xy = np.array([[np.mean([r["D2"] for r in sel if (r["init"], r["seed"]) == k]),
                                np.mean([r["mmd2"] for r in sel if (r["init"], r["seed"]) == k]),
                                [r["N_mode"] for r in sel if (r["init"], r["seed"]) == k][0]] for k in keys])
            else:
                sel = [r for r in pts if r["dataset"] == ds and r["mbar"] == mb]
                xy = np.array([[r["D2"], r["mmd2"], r["N_mode"]] for r in sel])
            rng = np.random.default_rng(12345)
            boots = [rng.integers(0, len(xy), len(xy)) for _ in range(10_000)]
            out = dict(dataset=ds, mbar=mb, models=len(xy))
            for name, col in (("rho_D", 0), ("rho_N", 2)):
                rho = spearmanr(xy[:, col], xy[:, 1]).statistic
                b = np.array([spearmanr(xy[i, col], xy[i, 1]).statistic for i in boots])
                lo, hi = np.nanpercentile(b, [2.5, 97.5])
                out[name], out[name + "_lo"], out[name + "_hi"] = float(rho), float(lo), float(hi)
            rows.append(out)
    with open(RES / "sweep_spearman.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines = ["| dataset | m-bar | rho_D (95 %) | rho_N (95 %) |", "|---|---|---|---|"]
    for r in rows:
        star = " (Fig. 4)" if r["mbar"] == FIG4_MBAR[r["dataset"]] else ""
        lines.append(f"| {r['dataset']} | {r['mbar']}{star} | {r['rho_D']:+.2f} ({r['rho_D_lo']:+.2f}, {r['rho_D_hi']:+.2f})"
                     f" | {r['rho_N']:+.2f} ({r['rho_N_lo']:+.2f}, {r['rho_N_hi']:+.2f}) |")
    (RES / "sweep_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
