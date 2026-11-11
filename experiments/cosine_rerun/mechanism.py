"""Mechanism analysis behind the D-Wave exception (contract condition 3), on the rerun's trained angles.
For global / agnostic / cluster at the largest a of each dataset, every training seed:
- mechanism_orders.csv: the evaluator's test MMD^2 (mean over m-bar = 1..6 of sum_A (zbar_A - t_A)^2 / 10,000 on the
  evaluator's subsets) split by the subset order |A| = 1..10; the contributions sum to `total`.
- mechanism_couplings.csv: per branch, the two-body angles theta_jk (mean |theta|, rms, number with |theta| > 0.1)
  and the rms over pairs of the branch's connected correlation c_jk = z_jk - z_j z_k (closed form, exact).
- mechanism_covariance.csv: the mixture's connected two-point correlation C = sum_l pi_l z^l_jk - zbar_j zbar_k
  splits exactly into a within-branch part W = sum_l pi_l c^l_jk and a between-branch part
  B = sum_l pi_l (z^l_j - zbar_j)(z^l_k - zbar_k). Reported: Frobenius norms over pairs j < k of W, B and C, the
  rms mismatch of C against the test data's connected correlations, and the same within / between split of the
  training data under the run's spectral groups (cluster labels of that seed and a).
- mechanism_alignment.csv (cluster models, whose branch l starts on group l): how far the within-branch part W and
  the between-branch part B are from the data's W_d and B_d (Frobenius norm of the difference, and the correlation
  over pairs), the mean correlation of each branch's c^l with its group's connected correlations, the mean ratio of
  their norms, and how similar the data's within-group correlation patterns are across groups (mean pairwise
  correlation over pairs of groups).
usage: mechanism.py  ->  results/mechanism_{orders,couplings,covariance,alignment}.csv
"""
import csv
import numpy as np
from common import CLUSTERS, DATASETS, PARAMS, RESULTS, data, theta, z1
from analyze import branch_z

SCHEMES = ("global", "agnostic", "cluster")


def eval_subsets(n):
    """The evaluator's subsets per m-bar (rng 1000 + m-bar, 10,000 draws, 1 <= |A| <= 10), padded with n."""
    out = []
    for mb in range(1, 7):
        M = np.random.default_rng(1000 + mb).random((10_000, n)) < mb / n
        k = M.sum(1)
        keep = (k >= 1) & (k <= 10)
        sub = np.full((keep.sum(), 10), n, dtype=np.int64)
        rr, cc = np.nonzero(M[keep])
        sub[rr, np.concatenate([np.arange(c) for c in k[keep]])] = cc
        out.append((sub, k[keep], M[keep]))
    return out


def moments(x, M):
    Mf = M.T.astype(np.float32)
    return sum((1 - 2 * ((x[i:i + 5000].astype(np.float32) @ Mf) % 2)).sum(0).astype(np.float64)
               for i in range(0, len(x), 5000)) / len(x)          # exact integer counts, float64 mean


def conn(s):
    """Connected two-point correlations of +-1 samples s (N, n): t_jk - t_j t_k."""
    m = s.mean(0)
    return (s.T @ s) / len(s) - np.outer(m, m)


def main():
    orders, coup, cov, align = [], [], [], []
    for ds in DATASETS:
        a = max(DATASETS[ds][2])
        z = np.load(PARAMS / f"{ds}_a{a}_main.npz")
        TH, pi, seeds, names, edges = theta(z["params"], a), z["pi"], z["seeds"], z["names"], z["edges"]
        xtr, xte = data(ds)
        n, L = xtr.shape[1], 2 ** a
        iu = np.triu_indices(n, 1)
        e_pos = {(int(i), int(j)): e for e, (i, j) in enumerate(edges)}
        perm = np.array([e_pos[(int(i), int(j))] for i, j in zip(*iu)])         # edge order -> triu order
        subs = eval_subsets(n)
        tt = [moments(xte, M) for _, _, M in subs]
        Ctest = conn(1 - 2 * xte.astype(np.float64))[iu]
        str_ = 1 - 2 * xtr.astype(np.float64)
        pair = np.full((len(edges), 10), n, dtype=np.int64)
        pair[:, :2] = edges
        for m in range(len(TH)):
            init, s = str(names[m]), int(seeds[m])
            if init not in SCHEMES:
                continue
            # order-resolved test MMD^2
            per_r = np.zeros(11)
            for (sub, k, _), t in zip(subs, tt):
                zbar = pi[m] @ branch_z(n, edges, TH[m], sub)
                np.add.at(per_r, k, (zbar - t) ** 2 / 10_000 / 6)
            tot = float(per_r.sum())
            orders += [dict(dataset=ds, init=init, seed=s, order=r, contribution=float(per_r[r]), total=tot) for r in range(1, 11)]
            # per-branch couplings and connected correlations (closed form)
            zi = z1(TH[m], n, edges)                                   # (L, n)
            zij = branch_z(n, edges, TH[m], pair)[:, perm]             # (L, pairs) in triu order
            c = zij - zi[:, iu[0]] * zi[:, iu[1]]
            th2 = TH[m][:, n:]
            for l in range(L):
                coup.append(dict(dataset=ds, init=init, seed=s, branch=l, pi=float(pi[m][l]),
                                 mean_abs_theta2=float(np.abs(th2[l]).mean()), rms_theta2=float(np.sqrt((th2[l] ** 2).mean())),
                                 n_strong_theta2=int((np.abs(th2[l]) > 0.1).sum()), rms_connected=float(np.sqrt((c[l] ** 2).mean()))))
            # within / between split of the mixture's connected correlations, and of the data's under the groups
            zbar = pi[m] @ zi
            W = pi[m] @ c
            B = pi[m] @ ((zi - zbar)[:, iu[0]] * (zi - zbar)[:, iu[1]])
            y = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")["train_y"].astype(int)
            w_ = np.bincount(y, minlength=L) / len(y)
            mu = np.stack([str_[y == g].mean(0) for g in range(L)])
            Wd = sum(w_[g] * conn(str_[y == g])[iu] for g in range(L))
            Bd = w_ @ ((mu - w_ @ mu)[:, iu[0]] * (mu - w_ @ mu)[:, iu[1]])
            nrm = lambda v: float(np.sqrt((v ** 2).sum()))
            if init == "cluster":
                Cg = [conn(str_[y == g])[iu] for g in range(L)]
                cr = lambda u, v: float(np.corrcoef(u, v)[0, 1])
                align.append(dict(dataset=ds, init=init, seed=s, within_err=nrm(W - Wd), within_corr=cr(W, Wd),
                                  between_err=nrm(B - Bd), between_corr=cr(B, Bd),
                                  branch_group_corr=float(np.mean([cr(c[l], Cg[l]) for l in range(L)])),
                                  branch_group_norm_ratio=float(np.mean([nrm(c[l]) / nrm(Cg[l]) for l in range(L)])),
                                  data_group_similarity=float(np.mean([cr(Cg[g], Cg[h]) for g in range(L) for h in range(g + 1, L)]))))
            cov.append(dict(dataset=ds, init=init, seed=s, within=nrm(W), between=nrm(B), total_norm=nrm(W + B),
                            within_frac=nrm(W) ** 2 / (nrm(W) ** 2 + nrm(B) ** 2),
                            mismatch_rms=float(np.sqrt(((W + B - Ctest) ** 2).mean())),
                            data_within=nrm(Wd), data_between=nrm(Bd), data_within_frac=nrm(Wd) ** 2 / (nrm(Wd) ** 2 + nrm(Bd) ** 2),
                            data_test_norm=nrm(Ctest)))
            print(f"{ds} a={a} {init:8s} seed {s:5d}: total {tot * 1e3:.4f}e-3, |A|=1 {per_r[1] * 1e3:.4f} |A|=2 {per_r[2] * 1e3:.4f}; "
                  f"within/between {nrm(W):.2f}/{nrm(B):.2f} (data {nrm(Wd):.2f}/{nrm(Bd):.2f}); mismatch {cov[-1]['mismatch_rms']:.4f}", flush=True)
    for name, rs in (("orders", orders), ("couplings", coup), ("covariance", cov), ("alignment", align)):
        with open(RESULTS / f"mechanism_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)


if __name__ == "__main__":
    main()
