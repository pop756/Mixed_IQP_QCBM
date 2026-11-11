"""Pairwise classical baseline: the fully visible sigmoid belief network (FVSBN), an autoregressive model whose
conditional of bit j is logistic in the earlier bits, p(x_j = 1 | x_<j) = sigmoid(b_j + sum_{k<j} W_jk x_k).
It has n + n(n - 1)/2 parameters, the count of an order-2 all-pairs IQP circuit, and exact likelihood and sampling.

Training: maximum likelihood, full-batch Adam (lr 1e-2, 4000 steps) on the first 90 % of the training split, L2
penalty lam ||W||^2 / 2 per sample with lam from LAMS chosen by the validation NLL on the last 10 %, the step with the
best validation NLL kept; then refit is not done (the kept model is the early-stopped one).
Evaluation: the test MMD^2 of the paper (m-bar = 1..6, the evaluator's subsets) from N_S = 10^7 model samples
(sample_mmd.test_mmd2, bias-corrected; about 1 % sampling noise, checked against exact correlators on Ising) for
n > 16, and from exact correlators (all 2^16 probabilities) for n = 16, plus the exact KL to the generating
distribution (../exact_metrics/targets.py) for n = 16.
usage: fvsbn.py <ds>  ->  results/fvsbn_<ds>.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax

jax.config.update("jax_enable_x64", False)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "cosine_rerun"), str(HERE.parent / "exact_metrics")]
from common import data, fwht                                                          # noqa: E402
from sample_mmd import test_mmd2, subsets, moments, KEVAL                              # noqa: E402

LAMS = (0.0, 1e-5, 1e-4, 1e-3)
STEPS, LR, N_S = 4000, 1e-2, 10_000_000


def nll_fn(params, x, mask, lam):
    W, b = params
    logits = x @ (W * mask).T + b
    ll = x * jax.nn.log_sigmoid(logits) + (1 - x) * jax.nn.log_sigmoid(-logits)
    return -jnp.mean(jnp.sum(ll, axis=1)) + 0.5 * lam * jnp.sum((W * mask) ** 2)


def fit(x_fit, x_val, lam):
    n = x_fit.shape[1]
    mask = jnp.asarray(np.tril(np.ones((n, n), dtype=np.float32), -1))
    xf, xv = jnp.asarray(x_fit, jnp.float32), jnp.asarray(x_val, jnp.float32)
    p = np.clip(x_fit.mean(0), 1e-3, 1 - 1e-3)
    params = (jnp.zeros((n, n), jnp.float32), jnp.asarray(np.log(p / (1 - p)), jnp.float32))
    opt = optax.adam(LR)
    st = opt.init(params)

    @jax.jit
    def step(params, st):
        g = jax.grad(nll_fn)(params, xf, mask, lam)
        u, st = opt.update(g, st, params)
        return optax.apply_updates(params, u), st

    val = jax.jit(lambda prm: nll_fn(prm, xv, mask, 0.0))
    best = (float(val(params)), 0, params)
    for k in range(1, STEPS + 1):
        params, st = step(params, st)
        if k % 50 == 0:
            v = float(val(params))
            if v < best[0]:
                best = (v, k, params)
    return best, mask


def sample(params, mask, N, seed, batch=500_000):
    W, b = params
    Wm = W * mask
    n = W.shape[0]

    @jax.jit
    def one(key):
        u = jax.random.uniform(key, (batch, n))

        def body(j, s):
            logit = s @ Wm[j] + b[j]
            return s.at[:, j].set((u[:, j] < jax.nn.sigmoid(logit)).astype(jnp.float32))
        return jax.lax.fori_loop(0, n, body, jnp.zeros((batch, n), jnp.float32))
    keys = jax.random.split(jax.random.PRNGKey(seed), N // batch)
    return np.concatenate([np.asarray(one(k)).astype(np.int8) for k in keys])


def exact_probs(params, mask):
    """All 2^n probabilities (n = 16), index sum_j x_j 2^j."""
    W, b = (np.asarray(t, np.float64) for t in params)
    n = W.shape[0]
    x = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
    logits = x @ (W * np.asarray(mask)).T + b
    lp = np.sum(x * -np.logaddexp(0, -logits) + (1 - x) * -np.logaddexp(0, logits), axis=1)
    return np.exp(lp)


def main():
    ds = sys.argv[1]
    x_tr, x_te = data(ds)
    n = x_tr.shape[1]
    cut = int(0.9 * len(x_tr))
    t0 = time.perf_counter()
    fits = {}
    for lam in LAMS:
        (v, k, prm), mask = fit(x_tr[:cut], x_tr[cut:], lam)
        fits[lam] = (v, k, prm)
        print(f"{ds} lam {lam:g}: best val NLL {v:.4f} at step {k}", flush=True)
    lam = min(fits, key=lambda l: fits[l][0])
    v, k, prm = fits[lam]
    out = dict(dataset=ds, n=n, params=n + n * (n - 1) // 2, lam=lam, best_step=k, val_nll=v,
               val_nll_all={str(l): fits[l][0] for l in LAMS}, fit_s=round(time.perf_counter() - t0, 1))
    xt = jnp.asarray(x_te, jnp.float32)
    out["test_nll"] = float(nll_fn(prm, xt, mask, 0.0))
    if n == 16:
        from targets import TARGETS
        q = exact_probs(prm, mask)
        p = TARGETS[ds]()
        m = p > 0
        out["KL"] = float(np.sum(p[m] * np.log(p[m] / q[m])))
        out["TV"] = float(0.5 * np.abs(p - q).sum())
        zq = fwht(q)
        per = []
        for mb in range(1, 7):
            As = subsets(n, mb)
            idx = np.array([int(np.sum(1 << A)) for A in As])
            per.append(float(np.sum((zq[idx] - moments(x_te, As)) ** 2) / KEVAL))
        out["test_mmd2"], out["test_mmd2_per_mbar"], out["mmd_from"] = float(np.mean(per)), per, "exact"
    else:
        reps = []
        for seed in (0, 1):
            s = sample(prm, mask, N_S, seed)
            reps.append(test_mmd2(s, x_te))
        out["test_mmd2"] = float(np.mean([r[0] for r in reps]))
        out["test_mmd2_reps"] = [r[0] for r in reps]
        out["test_mmd2_per_mbar"] = list(np.mean([r[1] for r in reps], axis=0))
        out["mmd_from"] = f"2 x {N_S} samples"
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / f"fvsbn_{ds}.json", "w"), indent=1)
    print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ != "test_mmd2_per_mbar"}), flush=True)


if __name__ == "__main__":
    main()
