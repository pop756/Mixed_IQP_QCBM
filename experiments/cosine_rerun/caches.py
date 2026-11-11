"""Caches of the mode-separation figure (fig_modesep, fig_modesep_full), computed exactly from the rerun's angles.
- modesep_blobs_a3.npz : blobs a = 3, per scheme and seed: D-bar = mean pairwise MMD^2 between branches at the
  m-bar = 2 bandwidth (exact, from the Walsh spectra of the branches' Born distributions), coverage = number of
  distinct blob peaks nearest to the binarized branch means; rep_means = exact one-body branch means of the
  scheme's lowest-D2 seed (results/branch_diversity.csv), the earlier figure's rule.
- modesep_ising_a2.npz : Ising a = 2, exact per-branch magnetization densities (M = sum_j (2 x_j - 1),
  bins of width 2, density) of each scheme at its lowest-D-bar seed, plus the training-data density
  (keys as the earlier cache).
- modesep_ext_cache.npz: Ising a = 3, the same at seed 7 with branches sorted by <M> (ising_* keys).
- dwave_overlap.npz    : D-Wave a = 3, seed 7: per-branch overlap mean E[q] = n^-1 sum_i z_i^2 and width from
  E[q^2] = n^-2 [n + sum_{i != j} z_ij^2] (exact closed-form z_i, z_ij), and the same for the training data inside
  each spectral group of that seed (cluster_qbar) and on the whole training set (full_qbar).
usage: caches.py  ->  results/modesep_*.npz, results/dwave_overlap.npz
"""
import csv
import numpy as np
from common import CLUSTERS, D, PARAMS, RESULTS, born_probs16, data, fwht, sigma_for, theta, z1
from analyze import branch_z

SCHEMES = ("global", "agnostic", "cluster")
MBINS = np.arange(-16.5, 18.5, 2.0)


def load(ds, a):
    z = np.load(PARAMS / f"{ds}_a{a}_main.npz")
    return theta(z["params"], a), z["pi"], z["seeds"], z["names"], z["edges"]


def walsh_weights(n, sigma):
    b = float(np.exp(-1.0 / (2.0 * sigma ** 2)))
    pc = np.array([bin(i).count("1") for i in range(2 ** n)])
    return ((1 + b) / 2) ** (n - pc) * ((1 - b) / 2) ** pc


def dbar(Z, w):
    L = len(Z)
    return float(np.mean([w @ (Z[i] - Z[j]) ** 2 for i in range(L) for j in range(L) if i != j]))


def lowest_d2(ds, init):
    rows = [r for r in csv.DictReader(open(RESULTS / "branch_diversity.csv")) if r["dataset"] == ds and r["init"] == init]
    return int(min(rows, key=lambda r: float(r["D2"]))["seed"])


def blobs():
    TH, pi, seeds, names, edges = load("blobs", 3)
    w = walsh_weights(16, sigma_for(16, 2))
    peaks = (np.loadtxt(D / "datasets/blobs/8_blobs_dataset/16_spins_8_blobs_peak_spins.csv", delimiter=",") > 0).astype(int)
    out = dict(seeds=np.array(sorted(set(seeds.tolist()))))
    db, cov, rep, rs = [], [], [], []
    for init in SCHEMES:
        d_, c_ = [], []
        for s in out["seeds"]:
            m = int(np.nonzero((seeds == s) & (names == init))[0][0])
            Z = np.stack([fwht(born_probs16(t, 16, edges)) for t in TH[m]])
            d_.append(dbar(Z, w))
            bm = ((1 - z1(TH[m], 16, edges)) / 2 > 0.5).astype(int)
            c_.append(len({int(np.abs(peaks - bm[i]).sum(1).argmin()) for i in range(len(bm))}))
        r = lowest_d2("blobs", init)
        m = int(np.nonzero((seeds == r) & (names == init))[0][0])
        db.append(d_); cov.append(c_); rs.append(r); rep.append((1 - z1(TH[m], 16, edges)) / 2)
    np.savez(RESULTS / "modesep_blobs_a3.npz", dbar=np.array(db), coverage=np.array(cov), rep_means=np.array(rep),
             rep_seed=np.array(rs), **out)
    print("blobs: D-bar mean", np.round(np.array(db).mean(1), 4), "coverage mean", np.array(cov).mean(1), "rep seeds", rs, flush=True)


def ising_hists(TH_m, edges, sort):
    y = (np.arange(2 ** 16)[:, None] >> np.arange(16)) & 1
    M_all = (2 * y - 1).sum(1)
    P = np.stack([born_probs16(t, 16, edges) for t in TH_m])
    h = np.array([np.histogram(M_all, bins=MBINS, weights=P[i], density=True)[0] for i in range(len(P))])
    mu = P @ M_all
    order = np.argsort(-mu) if sort else np.arange(len(P))
    return h[order], mu[order]


def ising():
    xtr = data("ising")[0]
    dM = np.histogram((2 * xtr.astype(int) - 1).sum(1), bins=MBINS, density=True)[0]
    TH, pi, seeds, names, edges = load("ising", 2)
    w = walsh_weights(16, sigma_for(16, 2))
    S = np.array(sorted(set(seeds.tolist())))
    dbars, hists, means, pick = [], [], [], []
    for init in SCHEMES:
        d_ = []
        for s in S:
            m = int(np.nonzero((seeds == s) & (names == init))[0][0])
            d_.append(dbar(np.stack([fwht(born_probs16(t, 16, edges)) for t in TH[m]]), w))
        s0 = int(S[int(np.argmin(d_))])
        m = int(np.nonzero((seeds == s0) & (names == init))[0][0])
        h, mu = ising_hists(TH[m], edges, sort=True)
        dbars.append(d_); hists.append(h); means.append(mu); pick.append(s0)
    np.savez(RESULTS / "modesep_ising_a2.npz", hists=np.array(hists), means=np.array(means), mbins=MBINS, data=dM,
             seed=np.array(pick), dbars=np.array(dbars), seeds=S)
    TH, pi, seeds, names, edges = load("ising", 3)
    hh, mm = [], []
    for init in SCHEMES:
        m = int(np.nonzero((seeds == 7) & (names == init))[0][0])
        h, mu = ising_hists(TH[m], edges, sort=True)
        hh.append(h); mm.append(mu)
    np.savez(RESULTS / "modesep_ext_cache.npz", ising_hists=np.array(hh), ising_means=np.array(mm), ising_mbins=MBINS,
             ising_data=dM)
    print("ising: a=2 lowest-D-bar seeds", pick, flush=True)


def dwave():
    TH, pi, seeds, names, edges = load("dwave", 3)
    xtr = data("dwave")[0]
    n = xtr.shape[1]
    sub = np.full((len(edges), 10), n, dtype=np.int64)
    sub[:, :2] = edges
    out = {}
    for init in SCHEMES:
        m = int(np.nonzero((seeds == 7) & (names == init))[0][0])
        zi = z1(TH[m], n, edges)                                   # (L, n)
        zij = branch_z(n, edges, TH[m], sub)                      # (L, E), each unordered pair once
        q = (zi ** 2).mean(1)
        q2 = (n + 2 * (zij ** 2).sum(1)) / n ** 2
        key = "cluster_initialized" if init == "cluster" else init
        out[f"q_{key}"], out[f"s_{key}"] = q, np.sqrt(np.maximum(q2 - q ** 2, 0))
    y = np.load(CLUSTERS / "dwave" / "a3_seed7.npz")["train_y"].astype(int)
    s = 1 - 2 * xtr.astype(np.float64)
    out["cluster_qbar"] = np.array([(s[y == l].mean(0) ** 2).mean() for l in range(8)])
    out["full_qbar"] = np.array((s.mean(0) ** 2).mean())
    np.savez(RESULTS / "dwave_overlap.npz", **out)
    print("dwave overlap:", {k: np.round(v, 3).tolist() for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    blobs()
    ising()
    dwave()
