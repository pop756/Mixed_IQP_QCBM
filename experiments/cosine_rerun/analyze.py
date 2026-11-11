"""Summary tables of the cosine rerun (results/), from runs/*.json and params/*.npz.
- refs.csv: per dataset and m-bar, the test-noise floor, the train-vs-test reference and the independent-bit model
  on the evaluator's subsets (m-bar = 1..6, 10,000 Bernoulli(m-bar/n) subsets, rng 1000 + m-bar, 1 <= |A| <= 10).
- main_summary.csv, main_table.md, main_curves.csv, train_loss_curves.csv: the campaign (global / agnostic / cluster).
- eqweight_control.csv, onebody_summary.csv: the two training controls.
- zero2b.csv: cluster models at the largest a with every two-body angle set to zero, re-evaluated.
- branch_diversity.csv (D2 = sum_A w_A Var_pi[z_A], test MMD^2 at one bandwidth, as the paper's Eq. dbranch):
  exact over all 2^16 subsets for blobs (a = 3) and Ising (a = 2); for D-Wave (a = 3) and MNIST (a = 4) the mean over
  10,000 subsets A ~ Bernoulli(p_sigma)^n (rng 20260728) with exact closed-form z_A (|A| > 10 dropped and counted
  in dropped_large_A).
- diversity_refs.csv: the train-test sampling reference with the same weights / subsets as branch_diversity.csv.
- mode_index.csv: N_mode = exp[H(rho) - E_y H(gamma(y))] from the branches' one-body product distributions on the
  test set, same models as branch_diversity.csv.
- mnist_branch_marginals.npz: per-branch pixel means <x_j> = (1 - z_j)/2 of every MNIST a = 4 model.
"""
import json
import sys

import numpy as np
from common import (DATASETS, EVAL_STEPS, PARAMS, RESULTS, RUNS, STEPS, born_probs16, data, edges_of, fwht, theta, z1)
from iqp_fastcf import make_evaluator

RESULTS.mkdir(parents=True, exist_ok=True)
NAMES = list(DATASETS)


def load_run(ds, a, variant):
    p = RUNS / ds / f"a{a}_{variant}.json"
    return json.load(open(p)) if p.exists() else None


def final(m):
    return m["test"][str(STEPS)]


def stats(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


# ---- references --------------------------------------------------------------------------------------------------
def refs():
    rows = []
    for ds in NAMES:
        xtr, xte = data(ds)
        n = xtr.shape[1]
        m = (1 - 2 * xtr).mean(0)
        la, ng = np.log(np.maximum(np.abs(m), 1e-300)), (m < 0).astype(np.float64)
        for mb in range(1, 7):
            M = np.random.default_rng(1000 + mb).random((10_000, n)) < mb / n
            k = M.sum(1)
            M = M[(k >= 1) & (k <= 10)].astype(np.float32)
            par = lambda X: sum((1 - 2 * ((X[i:i + 5000].astype(np.float32) @ M.T) % 2)).sum(0).astype(np.float64)
                                for i in range(0, len(X), 5000)) / len(X)       # exact integer counts, float64 mean
            ttr, tte = par(xtr), par(xte)
            zp = np.exp(M @ la) * (1 - 2 * ((M @ ng) % 2))
            rows.append(dict(dataset=ds, mbar=mb, floor=float(((1 - ttr ** 2) / len(xte)).sum() / 1e4),
                             ref_train_vs_test=float(((ttr - tte) ** 2).sum() / 1e4), product=float(((zp - tte) ** 2).sum() / 1e4)))
        print("refs", ds, flush=True)
    return rows


def write_csv(name, rows):
    import csv
    if not rows:
        return
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---- campaign tables ---------------------------------------------------------------------------------------------
def campaign(R):
    summ, curves, losses, table = [], [], [], []
    for ds in NAMES:
        fl, rf = R[ds]["floor"], R[ds]["ref_train_vs_test"]
        for a in DATASETS[ds][2]:
            run = load_run(ds, a, "main")
            if run is None:
                continue
            cells = {}
            for init in ("global", "agnostic", "cluster"):
                ms = [m for m in run["models"] if m["init"] == init]
                if not ms:
                    continue
                mu, sd, k = stats([final(m) for m in ms])
                per = np.mean([m["test_per_mbar"] for m in ms], axis=0)
                summ.append(dict(dataset=ds, a=a, init=init, n_seeds=k, test_mmd2_mean=mu, test_mmd2_sd=sd,
                                 ratio_to_ref=(mu - fl) / (rf - fl), **{f"mbar{j + 1}": per[j] for j in range(6)}))
                cells[init] = (mu, sd)
                for e in EVAL_STEPS:
                    mu_e, sd_e, _ = stats([m["test"][str(e)] for m in ms])
                    curves.append(dict(dataset=ds, a=a, init=init, step=e, test_mmd2_mean=mu_e, test_mmd2_sd=sd_e))
                lb = np.mean([m["train_loss_50"] for m in ms], axis=0)
                losses += [dict(dataset=ds, a=a, init=init, step=50 * (i + 1), train_loss_mean=float(v)) for i, v in enumerate(lb)]
            best = min(cells, key=lambda k: cells[k][0])
            table.append(f"| {ds} | {a} | " + " | ".join(
                (f"**{cells[i][0] * 1e3:.4f} ± {cells[i][1] * 1e3:.4f}**" if i == best else f"{cells[i][0] * 1e3:.4f} ± {cells[i][1] * 1e3:.4f}")
                if i in cells else "–" for i in ("global", "agnostic", "cluster")) + f" | {rf * 1e3:.4f} | {fl * 1e3:.4f} |")
    write_csv("main_summary.csv", summ)
    write_csv("main_curves.csv", curves)
    write_csv("train_loss_curves.csv", losses)
    head = ("Test MMD^2 x 1e-3 at step 15,000 (mean over m-bar = 1..6; mean ± sd over training seeds; lowest in bold).\n"
            "ref = train split as the model; floor = test-noise floor.\n\n"
            "| dataset | a | global | agnostic | cluster | ref | floor |\n|---|---|---|---|---|---|---|\n")
    (RESULTS / "main_table.md").write_text(head + "\n".join(table) + "\n")


def controls():
    eq, fc1 = [], []
    for ds in NAMES:
        for a in DATASETS[ds][2]:
            run, main = load_run(ds, a, "eq"), load_run(ds, a, "main")
            if run and main:
                seeds = set(run["seeds"])
                row = dict(dataset=ds, a=a, seeds=" ".join(map(str, sorted(seeds))))
                for key, ms in (("global", [m for m in main["models"] if m["init"] == "global" and m["seed"] in seeds]),
                                ("cluster", [m for m in main["models"] if m["init"] == "cluster" and m["seed"] in seeds]),
                                ("cluster_eq", run["models"])):
                    mu, sd, k = stats([final(m) for m in ms])
                    row.update({f"{key}_mean": mu, f"{key}_sd": sd, f"{key}_n": k})
                eq.append(row)
            run = load_run(ds, a, "fc1")
            if run and main:
                ref_init = "cluster" if a else "global"
                mu, sd, k = stats([final(m) for m in run["models"]])
                mu2, sd2, k2 = stats([final(m) for m in main["models"] if m["init"] == ref_init])
                fc1.append(dict(dataset=ds, a=a, init=ref_init, onebody_mean=mu, onebody_sd=sd, onebody_n=k,
                                order2_mean=mu2, order2_sd=sd2, order2_n=k2))
    write_csv("eqweight_control.csv", eq)
    write_csv("onebody_summary.csv", fc1)


# ---- analyses on the trained angles ------------------------------------------------------------------------------
def params_of(ds, a, variant):
    z = np.load(PARAMS / f"{ds}_a{a}_{variant}.npz")
    return z["params"], z["pi"], z["seeds"], z["names"], z["edges"]


def zero2b():
    rows = []
    for ds, a in (("mnist", 4), ("dwave", 3)):
        if not (PARAMS / f"{ds}_a{a}_main.npz").exists():
            continue
        P, pi, seeds, names, edges = params_of(ds, a, "main")
        sel = names == "cluster"
        n = P.shape[2] - len(edges)
        ev = make_evaluator(n, a, edges, data(ds)[1])
        Z = P[sel].copy()
        Z[:, :, n:] = 0.0                    # theta = W c is linear in c: zero two-body c = zero two-body theta
        full, zero = ev(P[sel], pi[sel])[0], ev(Z, pi[sel])[0]
        rows += [dict(dataset=ds, a=a, seed=int(s), test_mmd2=float(f), test_mmd2_zero2b=float(z0))
                 for s, f, z0 in zip(seeds[sel], full, zero)]
        print("zero2b", ds, flush=True)
    write_csv("zero2b.csv", rows)


def branch_z(n, edges, th, sub):
    """Exact closed-form z_A of each branch th (B, G) on subsets sub (K, 10) padded with n (1 <= |A| <= 10)."""
    import jax.numpy as jnp
    from iqp_fastcf.iqp_hip_cf import _hip_eval_only, TILE_S
    lens = (sub < n).sum(1)
    tiles = [(b, int(r), s0, min(TILE_S, (1 << (int(r) - 1)) - s0), 0)
             for b, r in enumerate(lens) for s0 in range(0, 1 << (int(r) - 1), TILE_S)]
    th2 = 2.0 * th
    Th = np.zeros((len(th), 1, n, n))
    Th[:, 0, edges[:, 0], edges[:, 1]] = th2[:, n:]
    Th = Th + Th.transpose(0, 1, 3, 2)
    z = _hip_eval_only(jnp.asarray(th2[:, None, :n]), jnp.asarray(Th), jnp.ones((len(th), 1)),
                       jnp.asarray(sub.astype(np.int32)), jnp.asarray(np.array(tiles, dtype=np.int32)))
    return np.asarray(z)


def branch_necessity(Z, pi, t, w):
    """Leave-one-branch-out test MMD^2 (Sec. II D). Removing branch l and renormalising the weights changes the loss by
    Delta_l = w . ((zbar_{-l} - t)^2) - w . ((zbar - t)^2), zbar_{-l} = (zbar - pi_l z^(l)) / (1 - pi_l).
    Returns (Lambda, Delta (L,), loss): Lambda = exp(sum_l pi_l ln[(loss + Delta_l) / loss]), the pi-weighted geometric
    mean factor by which the loss grows when one branch is removed."""
    zbar = pi @ Z
    loss = float(w @ ((zbar - t) ** 2))
    zm = (zbar[None] - pi[:, None] * Z) / (1.0 - pi[:, None])
    delta = ((zm - t[None]) ** 2) @ w - loss
    return float(np.exp(pi @ np.log((loss + delta) / loss))), delta, loss


def mode_index(Zb, pi, x):
    """N_mode from one-body correlators Zb (L, n) of the branches, test strings x (M, n)."""
    Zc = np.clip(Zb, -1 + 1e-12, 1 - 1e-12)
    logq = np.log((1 + Zc) / 2) @ (1 - x).T + np.log((1 - Zc) / 2) @ x.T
    lg = np.log(np.clip(pi, 1e-300, None))[:, None] + logq
    g = np.exp(lg - lg.max(0))
    g /= g.sum(0)
    ent = lambda p, ax: -(np.clip(p, 1e-300, None) * np.log(np.clip(p, 1e-300, None))).sum(ax)
    return float(np.exp(ent(g.mean(1), 0) - ent(g, 0).mean()))


def diversity():
    rows, modes, marg, refs_div, need = [], [], {}, [], []
    for ds in NAMES:
        sigma, a = DATASETS[ds][5]
        if not (PARAMS / f"{ds}_a{a}_main.npz").exists():
            continue
        P, pi, seeds, names, edges = params_of(ds, a, "main")
        _, xte = data(ds)
        n = xte.shape[1]
        TH = theta(P, a)                                                  # (M, L, G)
        b = float(np.exp(-1.0 / (2.0 * sigma ** 2)))
        xtr = data(ds)[0]
        if n <= 16:
            idx = xte.astype(np.int64) @ (1 << np.arange(n))
            t = fwht(np.bincount(idx, minlength=2 ** n) / len(xte))       # test moments of every subset
            t_tr = fwht(np.bincount(xtr.astype(np.int64) @ (1 << np.arange(n)), minlength=2 ** n) / len(xtr))
            pc = np.array([bin(i).count("1") for i in range(2 ** n)])
            w = ((1 + b) / 2) ** (n - pc) * ((1 - b) / 2) ** pc
            Zall = lambda th_m: np.stack([fwht(born_probs16(th, n, edges)) for th in th_m])
            dropped = 0
        else:
            p = (1 - b) / 2
            ops = np.random.default_rng(20260728).random((10_000, n)) < p
            k = ops.sum(1)
            keep = (k >= 1) & (k <= 10)
            dropped = int((k > 10).sum())
            sub = np.full((keep.sum(), 10), n, dtype=np.int64)
            rr, cc = np.nonzero(ops[keep])                                   # row-major: columns ascending per row
            sub[rr, np.concatenate([np.arange(c) for c in k[keep]])] = cc
            opsf = ops[keep].T.astype(np.float32)
            t = sum((1 - 2 * ((xte[i:i + 5000].astype(np.float32) @ opsf) % 2)).sum(0).astype(np.float64)
                    for i in range(0, len(xte), 5000)) / len(xte)       # exact integer counts, float64 mean
            t_tr = sum((1 - 2 * ((xtr[i:i + 5000].astype(np.float32) @ opsf) % 2)).sum(0).astype(np.float64)
                       for i in range(0, len(xtr), 5000)) / len(xtr)
            w = np.full(len(t), 1.0 / len(ops))      # mean over all 10,000 draws; |A| = 0 (z = t = 1) adds nothing
            Zall = lambda th_m: branch_z(n, edges, th_m, sub)
        refs_div.append(dict(dataset=ds, a=a, sigma=sigma, ref_train_vs_test=float(w @ ((t_tr - t) ** 2))))
        for m in range(len(P)):
            Z = Zall(TH[m])
            zbar, zsq = pi[m] @ Z, pi[m] @ (Z * Z)
            D2, mmd2 = float(w @ (zsq - zbar ** 2)), float(w @ ((zbar - t) ** 2))
            gap = abs(float(pi[m] @ ((Z - t) ** 2 @ w)) - D2 - mmd2)
            rows.append(dict(dataset=ds, init=str(names[m]), seed=int(seeds[m]), a=a, sigma=sigma, D2=D2, mmd2=mmd2,
                             id_gap=gap, dropped_large_A=dropped))
            lam, delta, _ = branch_necessity(Z, pi[m], t, w)
            need.append(dict(dataset=ds, init=str(names[m]), seed=int(seeds[m]), a=a, D2=D2, mmd2=mmd2, Lambda=lam,
                             n_harmful=int((delta < 0).sum()), w_harmful=float(pi[m][delta < 0].sum()),
                             delta_rel=" ".join(f"{v:.4g}" for v in delta / mmd2)))
            Zb = z1(TH[m], n, edges)
            modes.append(dict(dataset=ds, init=str(names[m]), seed=int(seeds[m]), a=a, D2=D2, mmd2=mmd2,
                              N_mode=mode_index(Zb, pi[m], xte.astype(np.float64))))
            if ds == "mnist":
                marg[f"{names[m]}_seed{int(seeds[m])}"] = ((1 - Zb) / 2).astype(np.float32)
                marg[f"{names[m]}_seed{int(seeds[m])}_pi"] = pi[m]
        print("diversity", ds, flush=True)
    write_csv("branch_diversity.csv", rows)
    write_csv("mode_index.csv", modes)
    write_csv("branch_necessity.csv", need)
    write_csv("diversity_refs.csv", refs_div)
    if marg:
        np.savez_compressed(RESULTS / "mnist_branch_marginals.npz", **marg)


def selfcheck():
    """branch_z (HIP closed form) against exact Born probabilities on an n = 12 random order-2 branch."""
    from iqp_fastcf import all_pairs
    rng = np.random.default_rng(3)
    n = 12
    e = all_pairs(n)
    th = rng.uniform(-0.6, 0.6, (2, n + len(e)))
    Zx = np.stack([fwht(born_probs16(t_, n, e)) for t_ in th])
    A = [np.sort(rng.choice(n, rng.integers(1, 11), replace=False)) for _ in range(300)]
    sub = np.full((len(A), 10), n)
    for i, s in enumerate(A):
        sub[i, :len(s)] = s
    z = branch_z(n, e, th, sub)
    want = Zx[:, [int(sum(1 << j for j in s)) for s in A]]
    err = float(np.abs(z - want).max())
    assert err < 1e-10, err
    print(f"selfcheck: closed-form z_A vs exact Born probabilities, max |diff| {err:.1e}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["diversity"]:
        diversity()
        sys.exit(0)
    selfcheck()
    R = refs()
    write_csv("refs.csv", R)
    Rm = {ds: {k: np.mean([r[k] for r in R if r["dataset"] == ds]) for k in ("floor", "ref_train_vs_test", "product")}
          for ds in NAMES}
    campaign(Rm)
    controls()
    zero2b()
    diversity()
    print("ANALYZE DONE", flush=True)
