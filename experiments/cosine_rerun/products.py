"""Product-mixture baselines on the full MNIST and D-Wave data (n = 16 values are carried from legacy):
- center: the untrained cluster center, branch l = the product distribution with the one-body moments of group l of
  the rerun's spectral clustering (clusters/<ds>/a<a>_seed<s>.npz), pi_l = |C_l| / N; a = 0: the global product;
- em: an EM-fitted L-component Bernoulli mixture, the n = 16 settings (k-means++ seeding with hard assignment,
  10 restarts kept by train log-likelihood, q clipped to [1e-3, 1 - 1e-3], tolerance 1e-10, at most 5000 iterations),
  EM seeds 0..4. MNIST runs the same EM with its E and M steps on the GPU (em_bernoulli_gpu).
Both become one-body IQP branches (theta = arcsin sqrt q, no two-body gates) scored by the rerun's evaluator.
The EM code is copied verbatim from legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/bernoulli_baseline.py.
usage: products.py [ds ...]  ->  results/product_baselines_<ds>.csv
"""
import csv, sys, time
import numpy as np
from common import CLUSTERS, DATASETS, RESULTS, data
from iqp_fastcf import make_evaluator, walsh

CLIP = 1e-3


def _logp_components(X, q, pi):
    return X @ np.log(q).T + (1.0 - X) @ np.log1p(-q).T + np.log(pi)


def _lse(a):
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def _kmeanspp_start(X, L, rng):
    idx = [int(rng.integers(len(X)))]
    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, L):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
        idx.append(int(rng.choice(len(X), p=p)))
        d2 = np.minimum(d2, ((X - X[idx[-1]]) ** 2).sum(1))
    cen = X[idx]
    lab = (X @ (1 - 2 * cen).T + cen.sum(1)).argmin(1)    # Hamming to each seed point
    q = np.stack([X[lab == l].mean(0) if (lab == l).any() else cen[l] for l in range(L)])
    pi = np.bincount(lab, minlength=L) + 1.0
    return np.clip(q, CLIP, 1 - CLIP), pi / pi.sum()


def em_bernoulli(X, L, seed, restarts=10, tol=1e-10, max_iter=5000):
    """Best-of-`restarts` EM for an L-component Bernoulli mixture (clipped M-step,
    which is the exact box-constrained maximizer, so EM stays monotone)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        q, pi = _kmeanspp_start(X, L, rng)
        prev = -np.inf
        for it in range(max_iter):
            lp = _logp_components(X, q, pi)
            lse = _lse(lp)
            ll = float(lse.mean())
            R = np.exp(lp - lse[:, None])
            Nk = R.sum(0) + 1e-12
            pi = Nk / Nk.sum()
            q = np.clip((R.T @ X) / Nk[:, None], CLIP, 1 - CLIP)
            if ll - prev < tol:
                break
            prev = ll
        ll = float(_lse(_logp_components(X, q, pi)).mean())
        if best is None or ll > best[2]:
            best = (q, pi, ll, it + 1)
    return best


def em_bernoulli_gpu(X, L, seed, restarts=10, tol=1e-10, max_iter=5000):
    """em_bernoulli with the E and M steps on the GPU (jax, float64): the same algorithm, starts and stopping rule;
    used for MNIST, where the CPU version takes hours."""
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    Xj = jnp.asarray(X)

    @jax.jit
    def step(q, pi):
        lp = Xj @ jnp.log(q).T + (1.0 - Xj) @ jnp.log1p(-q).T + jnp.log(pi)
        m = lp.max(1, keepdims=True)
        lse = (m + jnp.log(jnp.exp(lp - m).sum(1, keepdims=True)))[:, 0]
        R = jnp.exp(lp - lse[:, None])
        Nk = R.sum(0) + 1e-12
        return jnp.clip((R.T @ Xj) / Nk[:, None], CLIP, 1 - CLIP), Nk / Nk.sum(), lse.mean()

    @jax.jit
    def ll_of(q, pi):
        lp = Xj @ jnp.log(q).T + (1.0 - Xj) @ jnp.log1p(-q).T + jnp.log(pi)
        m = lp.max(1, keepdims=True)
        return (m + jnp.log(jnp.exp(lp - m).sum(1, keepdims=True)))[:, 0].mean()

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        q, pi = _kmeanspp_start(X, L, rng)
        q, pi, prev = jnp.asarray(q), jnp.asarray(pi), -np.inf
        for it in range(max_iter):
            q_new, pi_new, ll = step(q, pi)
            q, pi, ll = q_new, pi_new, float(ll)
            if ll - prev < tol:
                break
            prev = ll
        ll = float(ll_of(q, pi))
        if best is None or ll > best[2]:
            best = (np.asarray(q), np.asarray(pi), ll, it + 1)
    return best


def compiled(q, a):
    """Compiled angles of one-body branches with P(x_j = 1) = q[l, j]: theta = arcsin sqrt q, c = W^T theta / L."""
    th = np.arcsin(np.sqrt(np.clip(q, 0.0, 1.0)))
    return walsh(a).T @ th / (1 << a)


def main(names):
    for ds in names:
        xtr, xte = data(ds)
        X = xtr.astype(np.float64)
        n = X.shape[1]
        seeds = DATASETS[ds][3]
        rows = []
        for a in DATASETS[ds][2]:
            L = 1 << a
            ev = make_evaluator(n, a, np.zeros((0, 2), dtype=np.int32), xte)
            models = []
            for s in (seeds if a else seeds[:1]):
                y = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")["train_y"].astype(int) if a else np.zeros(len(X), int)
                q = np.stack([X[y == l].mean(0) for l in range(L)])
                models.append(("center", s if a else -1, q, np.bincount(y, minlength=L) / len(y), {}))
            for s in (range(len(seeds)) if a else [0]):
                t0 = time.time()
                q, pi, ll, its = (em_bernoulli_gpu if ds == "mnist" else em_bernoulli)(X, L, s)
                models.append(("em", s if a else -1, q, pi, dict(train_ll_per_bit=ll / n, em_iters=its,
                                                                  em_sec=round(time.time() - t0, 1), min_pi=float(pi.min()))))
                print(f"[{ds} em] a={a} seed={s} ll/bit={ll / n:.5f} iters={its} ({time.time() - t0:.0f} s)", flush=True)
            mean, per = ev(np.stack([compiled(q, a) for _, _, q, _, _ in models]), np.stack([p for _, _, _, p, _ in models]))
            # closed-form cross-check on the evaluator's own subsets: z_A = sum_l pi_l prod_{j in A} (1 - 2 q_lj)
            for (meth, s, q, pi, ex), m, pm in zip(models, mean, per):
                rows.append(dict(dataset=ds, method=meth, a=a, seed=s, test_mmd2=float(m),
                                 **{f"mbar{j + 1}": float(v) for j, v in enumerate(pm)}, **ex))
            print(f"[{ds}] a={a}: center {np.mean([r['test_mmd2'] for r in rows if r['a'] == a and r['method'] == 'center']) * 1e3:.4f}e-3"
                  f"  em {np.mean([r['test_mmd2'] for r in rows if r['a'] == a and r['method'] == 'em']) * 1e3:.4f}e-3", flush=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
        with open(RESULTS / f"product_baselines_{ds}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dwave", "mnist"])
