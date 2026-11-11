"""Initial angles and branch weights for a campaign (the arena judge's recipe, from cf_multi.py).

    c0, pi = make_inits(data, a, seeds)          # (3 * len(seeds), L, G), (3 * len(seeds), L), seed-major
    out = train(n, a, all_pairs(n), data, [sigma_for(n, 2), sigma_for(n, 6)], 1e-3, 15000, c0, pi, seeds)
"""
import itertools, math
import numpy as np

INITS = ("global", "agnostic", "cluster")


def all_pairs(n):
    return np.array(list(itertools.combinations(range(n), 2)), dtype=np.int32)


def sigma_for(n, mbar):                                       # Bernoulli(p)^n subsets with p = (1 - e^{-1/2s^2})/2 = mbar/n
    return 1.0 / math.sqrt(-2.0 * math.log(1.0 - 2.0 * mbar / n))


def walsh(a):
    if a == 0:
        return np.ones((1, 1))
    S = np.array(list(itertools.product((0, 1), repeat=a)), dtype=np.int64)
    return (-1.0) ** ((S @ S.T) % 2)


def kmeans(x, k, seed=1234, iters=100):                       # Lloyd with k-means++ seeding, on 0/1 rows
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        cc = np.array(c)
        d = np.maximum(np.min((x ** 2).sum(1)[:, None] - 2 * x @ cc.T + (cc ** 2).sum(1)[None], axis=1), 0)
        c.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin((x ** 2).sum(1)[:, None] - 2 * x @ c.T + (c ** 2).sum(1)[None], axis=1)
        new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab


def inits(data, a, n, G, seed, labels=None):
    """(name, c0 (L, G), pi (L,)) for the three initializations of one seed (cf_multi.py's recipe):
    global = coincident branches at the data's one-body moments, agnostic = coincident at pi/4, cluster = one branch per
    cluster with pi = cluster sizes; all + Unif(-r0, r0), r0 = (pi/2)/sqrt(L G), seeded. Clusters: `labels` (one per
    data row, in [0, L), e.g. the IQP paper's spectral clustering) or, by default, k-means on a 10k subsample."""
    L = 2 ** a
    mean_z = 1.0 - 2.0 * data.mean(0)
    if labels is not None:
        sub, lab = data, labels
    else:
        sub = data[np.random.default_rng(1234).permutation(len(data))[:10000]]
        lab = kmeans(sub, L) if L > 1 else np.zeros(len(sub), int)
    mu = {"global": np.tile(mean_z, (L, 1)), "agnostic": np.zeros((L, n)),
          "cluster": np.array([1.0 - 2.0 * sub[lab == l].mean(0) if np.any(lab == l) else mean_z for l in range(L)])}
    pis = {"global": np.ones(L) / L, "agnostic": np.ones(L) / L,
           "cluster": np.bincount(lab, minlength=L) / len(lab)}
    W, r0, out = walsh(a), (math.pi / 2) / math.sqrt(L * G), []
    for name in INITS:
        th0 = np.zeros((L, G))
        th0[:, :n] = 0.5 * np.arccos(np.clip(mu[name], -1, 1))
        c0 = (W.T @ th0) / L + np.random.default_rng(seed).uniform(-r0, r0, (L, G))
        out.append((name, c0, pis[name] / pis[name].sum()))
    return out


def make_inits(data, a, seeds, edges=None, labels=None):
    """c0 (M, L, G) and pi (M, L) for every seed x (global, agnostic, cluster), ordered seed-major as train expects.
    edges: the edges train will get (only their number is used; default all pairs). labels: cluster label per data row
    in [0, 2**a) for the cluster init (default: k-means)."""
    data = np.asarray(data, dtype=np.int8)
    n = data.shape[1]
    G = n + (n * (n - 1) // 2 if edges is None else len(np.asarray(edges).reshape(-1, 2)))
    if labels is not None:
        labels = np.asarray(labels).ravel()
        if labels.shape != (len(data),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"labels must be {len(data)} integers, one per data row")
        if labels.min() < 0 or labels.max() >= 2 ** a:
            raise ValueError(f"labels must lie in [0, {2 ** a})")
    c0, pi = [], []
    for sd in seeds:
        for _, c, p in inits(data, a, n, G, sd, labels):
            c0.append(c); pi.append(p)
    return np.array(c0), np.array(pi)
