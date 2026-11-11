"""Center-certificate quantities at the noise-free centers on the full MNIST / D-Wave training data, with the rerun's
cluster partitions, and the clustering settings of App. D C.
The formulas are those of experiments/paper_ours/fulldata/analysis.py:cert (= legacy center_certificate.py), copied
verbatim; only the center comes from the rerun's labels (clusters/<ds>/a<a>_seed<s>.npz) and the sigmas are the
rerun's training bandwidths (m-bar = 2, 6):
  L_train_V/U = mean over the two training kernels of the closed-form MMD^2 between the product center and the
                training set (V: full data-data term, U: unbiased data-data term),
  c_lead      = 8 wbar_1 pi_l^2 (1 - t_lj^2), maximised over (l, j); P = 2 t^2 / (1 - t^2) there; R_loss = P L_V.
usage: certificate.py -> results/certificate.csv, results/clustering_settings.csv
"""
import csv
import numpy as np
from common import CLUSTERS, DATASETS, RESULTS, data, sigmas


def write(rows, name):
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cert(name):
    X = data(name)[0].astype(np.float64)
    m, n = X.shape
    sig = sigmas(n)
    dd = {}
    for s in sig:
        lb, tot = -1.0 / (2.0 * s ** 2), 0.0
        for i in range(0, m, 2000):
            Xc = X[i:i + 2000]
            tot += float(np.exp((Xc @ (1 - X).T + (1 - Xc) @ X.T) * lb).sum())
        dd[s] = tot / m ** 2

    def mmd2_closed(q, pi, s):
        b = np.exp(-1.0 / (2.0 * s ** 2))
        al, be = (1 + b) / 2, (1 - b) / 2
        T, w = 1 - 2 * q, pi / pi.sum()
        MM = float(w @ np.exp(np.log(al + be * T[:, None, :] * T[None, :, :]).sum(-1)) @ w)
        MD = float((np.exp(X @ np.log(al - be * T).T + (1 - X) @ np.log(al + be * T).T) @ w).mean())
        return MM - 2 * MD + dd[s], MM - 2 * MD + (m * dd[s] - 1) / (m - 1)

    rows = []
    for a in DATASETS[name][2]:
        for s in (DATASETS[name][3] if a else [-1]):
            L = 2 ** a
            y = np.load(CLUSTERS / name / f"a{a}_seed{s}.npz")["train_y"].astype(int) if a else np.zeros(m, int)
            q = np.stack([X[y == l].mean(0) for l in range(L)])
            pi = np.bincount(y, minlength=L) / m
            vu = [mmd2_closed(q, pi, sg) for sg in sig]
            LV, LU = float(np.mean([v for v, _ in vu])), float(np.mean([u for _, u in vu]))
            p = np.array([(1 - np.exp(-1 / 2 / sg ** 2)) / 2 for sg in sig])
            w1 = float(np.mean(p * (1 - p) ** (n - 1)))
            t = 1 - 2 * q
            cl = 8 * w1 * (pi[:, None] ** 2) * (1 - t ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                P = np.where(np.abs(t) < 1, 2 * t ** 2 / (1 - t ** 2), np.inf)
            l, j = np.unravel_index(np.argmax(cl), cl.shape)
            rows.append(dict(dataset=name, ancilla=a, seed=s, L_train_V=LV, L_train_U=LU, wbar_1=w1,
                             lead_pi=float(pi[l]), lead_t=float(t[l, j]), lead_c_lead=float(cl[l, j]),
                             lead_P=float(P[l, j]), lead_R_loss=float(P[l, j] * LV),
                             max_abs_t_at_lead=float(abs(t[l, j])), R_over_c=float(P[l, j] * LV / cl[l, j])))
            print(rows[-1], flush=True)
    return rows


def clustering():
    rows = []
    for ds in DATASETS:
        for a in DATASETS[ds][2]:
            if not a:
                continue
            for s in DATASETS[ds][3]:
                z = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")
                w = np.bincount(z["train_y"].astype(int), minlength=2 ** a) / len(z["train_y"])
                rows.append(dict(dataset=ds, ancilla=a, seed=s, n_sub=int(z["n_sub"]), d_med=float(z["d_med"]),
                                 sigma_c=float(z["sigma"]), w_min=float(w.min()), w_max=float(w.max())))
    write(rows, "clustering_settings.csv")


if __name__ == "__main__":
    clustering()
    write(cert("dwave") + cert("mnist"), "certificate.csv")
