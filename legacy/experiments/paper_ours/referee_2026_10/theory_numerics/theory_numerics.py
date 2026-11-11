#!/usr/bin/env python3
"""Referee round 2026-10: exact center numerics for Theorem 2 and the compiled
common-mode / ancilla blocks (n=16 blobs + 2D Ising, all 2^16 subsets).

Centers are rebuilt through the training code path with zero
coincidence-breaking radius (analysis_ancilla_angles/check_theorem2_bound.center_of,
the same path that produced Table theory-check); cluster weights are the
y_train.csv cluster masses (cluster_reference.py / curvature_decomposition.py).
The loss is L = sum_A wbar_A (z_A - t_A)^2, wbar_A = mean over the two kernel
sigmas of p^|A| (1-p)^(n-|A|).

Stages (python theory_numerics.py [items|hessian|gd] ...):
  items   : Table theory-check reproduction (item 0), Theorem-2 remainder R_n
            (item 1), compiled common-mode curvature (item 2).
  hessian : single-circuit Hessian H_sys, mismatch block M, and the full
            (n+a)-qubit compiled Hessian block check at a=1,2 (item 3).
  gd      : plain GD from the coincident center + 1e-6 ancilla kick (item 3 opt).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["IQPOPT_X64"] = "1"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
ANA = HERE.parents[1] / "analysis_ancilla_angles"
sys.path.insert(0, str(ANA))

import check_theorem2_bound as C          # noqa: E402  center_of (radius 0)
import exact_iqp_weighted as EW            # noqa: E402  branch-weighted exact loss
from curvature_decomposition import subset_products, wht  # noqa: E402

import jax                                 # noqa: E402
import jax.numpy as jnp                    # noqa: E402

jax.config.update("jax_enable_x64", True)

LABELS = Path(__file__).resolve().parents[2]
KEYS = {"agnostic": ("S4", "s4_unbiased"), "global": ("S1_visible", "s1_visible"),
        "cluster": ("S2_cluster", "s2_cluster")}
DATASETS = ("blobs", "ising")


# ----------------------------------------------------------------------------
def bitrev(l, a):
    return int(format(l, f"0{a}b")[::-1], 2) if a else 0


def load(ds, cen, a, mode="exact"):
    """Center + exact data tensors for one configuration.

    Branch l of EW.branch_matrix carries ancilla bit s = (l >> s) & 1 on wire
    n+s, while the trainer maps cluster label k to the bitstring format(k, '0ab')
    on wires n, n+1, ... (MSB first): branch l holds group bitrev(l).
      mode="table": trainer center, pi_l = mass of LABEL l (the pairing used by
                    check_theorem2_bound.py / curvature_decomposition.py, i.e.
                    Table theory-check); mismatched for cluster L >= 4.
      mode="exact": pi_l = mass of the group branch l actually holds, one-body
                    angles = arccos(t_j^(l))/2 in float64, two-body angles 0.
    """
    key, stem = KEYS[cen]
    built, center = C.center_of(ds, key, stem, a)
    tr = built["trainer"]
    n, gates, L = tr.visible_bits, built["gates"], 2 ** a
    X = np.asarray(tr.train_bits, dtype=np.int64)
    Z = 1.0 - 2.0 * X
    perm = np.array([bitrev(l, a) for l in range(L)])
    if cen == "cluster":
        lab = LABELS / f"{ds}_seed" / f"dataset_s2_cluster_a{a}_usqm_seed7" / "y_train.csv"
        pi_lab = EW.cluster_weights(lab, L)
        y = np.loadtxt(lab, dtype=int, delimiter=",").ravel()
        t_formula = np.stack([Z[y == k].mean(0) for k in perm])     # group of branch l
        pi = pi_lab if mode == "table" else pi_lab[perm]
    elif cen == "global":
        pi = np.full(L, 1.0 / L)
        t_formula = np.tile(Z.mean(0), (L, 1))
    else:
        pi = np.full(L, 1.0 / L)
        t_formula = np.zeros((L, n))
    B, gens = EW.branch_matrix(gates, n, a)
    th = np.einsum("lgm,m->lg", B, center)            # branch angles (L, 136)
    one = np.array([gens.index([j]) for j in range(n)])
    two = np.array([i for i, g in enumerate(gens) if len(g) == 2])
    dev_center = float(np.max(np.abs(th[:, one] - 0.5 * np.arccos(t_formula))))
    max_two = float(np.max(np.abs(th[:, two])))
    th[:, two] = 0.0                                   # analyzed center: exactly zero
    if mode == "exact":
        th[:, one] = 0.5 * np.arccos(t_formula)
    # compiled center with the two-body angles exactly zero
    center0 = np.linalg.lstsq(B.reshape(L * len(gens), -1), th.reshape(-1), rcond=None)[0]
    tj = np.cos(2.0 * th[:, one])
    s2 = np.sin(2.0 * th[:, one])

    hist = np.bincount((X * (2 ** np.arange(n - 1, -1, -1))).sum(1),
                       minlength=2 ** n).astype(np.float64) / len(X)
    tA = wht(hist)
    idx = np.arange(2 ** n)
    size = np.zeros(2 ** n, dtype=np.int64)
    for j in range(n):
        size += (idx >> (n - 1 - j)) & 1
    sig = [float(s) for s in tr.qcbm["sigmas"]]
    ps = [(1.0 - np.exp(-0.5 / s ** 2)) / 2.0 for s in sig]
    wb = np.mean([p ** size * (1 - p) ** (n - size) for p in ps], axis=0)
    zl = np.stack([subset_products(tj[l], n) for l in range(L)])
    zbar = pi @ zl
    return dict(ds=ds, cen=cen, a=a, L=L, n=n, gates=gates, B=B, gens=gens,
                one=one, two=two, center=center, center0=center0, th=th, pi=pi,
                tj=tj, s2=s2, tA=tA, wb=wb, size=size, idx=idx, zl=zl,
                zbar=zbar, r=tA - zbar, X=X, sigmas=sig, mode=mode,
                dev_center=dev_center, max_two=max_two)


def jwht(v):
    N = int(np.log2(v.size))
    for k in range(N):
        v = v.reshape(2 ** k, 2, -1)
        v = jnp.stack([v[:, 0] + v[:, 1], v[:, 0] - v[:, 1]], axis=1)
    return v.reshape(-1)


def smooth_loss(d, gates, a):
    """(n+a)-qubit exact loss as exact_iqp.build_exact_loss but with
    p = Re^2 + Im^2.  exact_iqp uses jnp.abs(psi)**2, whose nested autodiff
    drops the 2|d psi|^2 term wherever psi = 0 -- and at branch coincidence
    every ancilla-flipped amplitude is exactly 0, so its compiled ancilla
    Hessian block comes out as M/2 instead of M."""
    n = d["n"]
    N = n + a
    gi = jnp.asarray(np.array([sum(1 << (N - 1 - w) for w in sorted({w for gen in g for w in gen}))
                               for g in gates], dtype=np.int64))
    tA, wb = jnp.asarray(d["tA"]), jnp.asarray(d["wb"])

    def corr(theta):
        c = jnp.zeros(2 ** N, dtype=theta.dtype).at[gi].set(theta)
        psi = jwht(jnp.exp(-1j * jwht(c))) / 2.0 ** N
        p = (psi.real ** 2 + psi.imag ** 2).reshape(2 ** n, 2 ** a).sum(1)
        return jwht(p)

    return jax.jit(lambda th: jnp.sum(wb * (corr(th) - tA) ** 2)), jax.jit(corr)


def has_j(d, j):
    return ((d["idx"] >> (d["n"] - 1 - j)) & 1) == 1


def ad_curv(d, v):
    """Exact second derivative of the weighted loss along compiled direction v."""
    _, loss = EW.build_exact_loss_weighted(d["gates"], d["n"], d["a"], d["X"],
                                           d["sigmas"], d["pi"])
    c0, vj = jnp.asarray(d["center0"]), jnp.asarray(v)
    dg = lambda t: jnp.vdot(jax.grad(loss)(c0 + t * vj), vj)  # noqa: E731
    return float(jax.grad(dg)(0.0)), float(loss(c0))


def compiled_dir(d, j, signs):
    """Compiled direction v with theta^(l)_j moved by signs[l]."""
    n, L = d["n"], d["L"]
    v = np.zeros(len(d["gates"]))
    for i, g in enumerate(d["gates"]):
        if sorted({w for gen in g for w in gen if w < n}) != [j]:
            continue
        S = [w - n for gen in g for w in gen if w >= n]
        v[i] = sum(signs[l] * (-1.0) ** sum((l >> s) & 1 for s in S)
                   for l in range(L)) / L
    return v


# ----------------------------------------------------------------------------
def branch_curv(d, l, j):
    """Closed-form c = d^2 L / d theta_j^(l)^2 at the factorizing center."""
    n, pi, wb = d["n"], d["pi"], d["wb"]
    h = has_j(d, j)
    q = subset_products(d["tj"][l] ** 2, n, skip=j)          # prod_{A\j} t_k^2
    sens = 8 * pi[l] ** 2 * d["s2"][l, j] ** 2 * np.sum(wb[h] * q[h])
    mis = 8 * pi[l] * np.sum(wb[h] * d["zl"][l][h] * d["r"][h])
    return sens + mis, sens, mis


def item01(d, lj=None):
    n, L, pi, wb, r, size = d["n"], d["L"], d["pi"], d["wb"], d["r"], d["size"]
    wj = wb[1 << (n - 1)]                                       # |A|=1 weight
    lead = 8 * wj * pi[:, None] ** 2 * (1 - d["tj"] ** 2)
    if lj is None:
        lj = np.unravel_index(np.argmax(lead), lead.shape)     # first max: l=0 on ties
    l, j = lj
    c, sens, mis = branch_curv(d, l, j)
    v = compiled_dir(d, j, [1.0 if k == l else 0.0 for k in range(L)])
    c_ad, loss = ad_curv(d, v)
    t = d["tj"][l, j]
    h = has_j(d, j)
    pref = 2 * t * t / (1 - t * t)
    terms = pref * wb * r ** 2
    by = {k: float(np.sum(terms[h & (size == k)])) for k in (2, 3, 4)}
    by[5] = float(np.sum(terms[h & (size >= 5)]))
    Rn = sum(by.values())
    Lc = float(np.sum(wb * r ** 2))
    return dict(dataset=d["ds"], center=d["cen"], mode=d["mode"], L=L, branch=int(l),
                group_label=bitrev(int(l), d["a"]) if d["cen"] == "cluster" else -1, j=int(j),
                pi=float(pi[l]), one_minus_t2=float(1 - t * t), t=float(t),
                c=c_ad, c_closed=c, c_sens=sens, c_mis=mis, mis_over_sens=mis / sens,
                c_lead=float(lead[l, j]), ratio_c_lead=c_ad / float(lead[l, j]),
                R_n=Rn, R_n_A2=by[2], R_n_A3=by[3], R_n_A4=by[4], R_n_A5p=by[5],
                lead_minus_Rn=float(lead[l, j]) - Rn, Rn_over_lead=Rn / float(lead[l, j]),
                nonvacuous=bool(lead[l, j] - Rn > 0), c_ge_bound=bool(c_ad >= lead[l, j] - Rn),
                loss_center=Lc, loss_center_ad=loss, R_loss=pref * Lc,
                Rn_le_Rloss=bool(Rn <= pref * Lc),
                max_abs_r_onebody=float(np.max(np.abs(r[size == 1]))),
                center_dev_vs_formula=d["dev_center"], max_abs_twobody_raw=d["max_two"])


def census(d):
    """c >= c_lead at every (branch, one-body) pair (closed form)."""
    wj = d["wb"][1 << (d["n"] - 1)]
    out = []
    for l in range(d["L"]):
        for j in range(d["n"]):
            lead = 8 * wj * d["pi"][l] ** 2 * (1 - d["tj"][l, j] ** 2)
            c = branch_curv(d, l, j)[0]
            out.append(dict(dataset=d["ds"], center=d["cen"], mode=d["mode"], L=d["L"],
                            branch=l, j=j, one_minus_t2=1 - d["tj"][l, j] ** 2,
                            c=c, c_lead=lead, ratio=c / lead if lead > 0 else np.inf))
    return out


def item2(d, j=None):
    """Common-mode curvature d^2 L / d theta~_{j,emptyset}^2 (all branches move)."""
    n, L, pi, wb, r, size = d["n"], d["L"], d["pi"], d["wb"], d["r"], d["size"]
    wj = wb[1 << (n - 1)]
    lead_all = 8 * wj * (pi @ np.abs(d["s2"])) ** 2
    if j is None:
        j = int(np.argmax(lead_all))
    h = has_j(d, j)
    gbar = sum(pi[l] * -2 * d["s2"][l, j] * subset_products(d["tj"][l], n, skip=j)
               for l in range(L))
    zb = d["zbar"]
    c_closed = float(np.sum(2 * wb[h] * (gbar[h] ** 2 + 4 * zb[h] * r[h])))
    hi = h & (size >= 2)
    Rcm = float(8 * np.sum(wb[hi] * np.abs(zb[hi]) * np.abs(r[hi])))
    c_ad, _ = ad_curv(d, compiled_dir(d, j, [1.0] * L))
    out = dict(dataset=d["ds"], center=d["cen"], L=L, j=j, c_cm=c_ad, c_cm_closed=c_closed,
               lead_cm=float(lead_all[j]), R_cm=Rcm, lead_minus_Rcm=float(lead_all[j]) - Rcm,
               c_ge_bound=bool(c_ad >= lead_all[j] - Rcm), eight_wj=8 * wj)
    if d["cen"] == "global":     # single-circuit analogue of item 1 (pi = 1)
        lead1 = 8 * wj * (1 - d["tj"][0] ** 2)
        j1 = int(np.argmax(lead1))
        t = d["tj"][0, j1]
        h1 = has_j(d, j1) & (size >= 2)
        R1 = 2 * t * t / (1 - t * t) * float(np.sum(wb[h1] * r[h1] ** 2))
        out.update(single_j=j1, single_lead=float(lead1[j1]), single_R=R1,
                   single_lead_minus_R=float(lead1[j1]) - R1)
    return out


def write(name, rows):
    with open(HERE / name, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader(); wr.writerows(rows)


def stage_items():
    rows0, rows1, rows2, cens = [], [], [], []
    witness = {(r["dataset"], r["center"], int(r["L"])): r
               for r in csv.DictReader(open(ANA / "theorem2_witness.csv"))}
    for ds in DATASETS:
        for cen, As in (("agnostic", (0, 1, 2, 3)), ("global", (0, 1, 2, 3)),
                        ("cluster", (1, 2, 3))):
            for a in As:
                L = 2 ** a
                # item 0: reproduce Table theory-check with the Table's pairing
                dt = load(ds, cen, a, "table")
                if cen == "agnostic":
                    c, _ = ad_curv(dt, compiled_dir(dt, 0, [1.0] + [0.0] * (L - 1)))
                    rows0.append(dict(dataset=ds, center=cen, L=L, c=c,
                                      c_ref=8 * dt["wb"][1 << (dt["n"] - 1)] / L ** 2))
                else:
                    w = witness[(ds, cen, L)]
                    r0 = item01(dt)
                    r0.update(paper_branch=int(w["branch"]), paper_j=int(w["alpha"]),
                              paper_c=float(w["curv"]), paper_c_lead=float(w["bound"]))
                    rows0.append(r0)
                    cens += census(dt)
                print(f"[item0] {rows0[-1]}", flush=True)
                # items 1-2 at the analyzed center (correct group/weight pairing)
                d = load(ds, cen, a, "exact")
                if cen != "agnostic":
                    cens += census(d)
                    r1 = item01(d)
                    rows1.append(r1)
                    lj = (int(witness[(ds, cen, L)]["branch"]), int(witness[(ds, cen, L)]["alpha"]))
                    if (r1["branch"], r1["j"]) != lj:     # also at the Table's coordinate
                        rows1.append(dict(item01(d, lj), mode="exact_at_table_witness"))
                    print(f"[item1] {rows1[-1]}", flush=True)
                r2 = item2(d)
                rows2.append(r2)
                print(f"[item2] {r2}", flush=True)
    write("item0_table_reproduction.csv", rows0)
    write("item1_theorem2_remainder.csv", rows1)
    write("item2_common_mode.csv", rows2)
    write("census_all_pairs.csv", cens)


# ----------------------------------------------------------------------------
def single_blocks(d, thbar):
    """H_sys and M (mismatch part) of the single circuit at angles thbar (136,)."""
    gates0 = [[list(g)] for g in d["gens"]]
    loss0, corr0 = smooth_loss(d, gates0, 0)
    x = jnp.asarray(thbar)
    H = np.asarray(jax.hessian(loss0)(x))
    g = np.asarray(jax.grad(loss0)(x))
    zb = np.asarray(corr0(x))
    res = jnp.asarray(2 * d["wb"] * (zb - d["tA"]))
    M = np.asarray(jax.hessian(lambda th: jnp.sum(res * corr0(th)))(x))
    return H, M, g, zb, float(loss0(x))


def full_hessian(d, x):
    gF = jax.grad(smooth_loss(d, d["gates"], d["a"])[0])
    hvp = jax.jit(jax.vmap(lambda v: jax.jvp(gF, (x,), (v,))[1]))
    m, E = len(d["gates"]), np.eye(len(d["gates"]))
    return np.concatenate([np.asarray(hvp(jnp.asarray(E[i:i + 16]))) for i in range(0, m, 16)])


def gate_GS(d):
    n = d["n"]
    G = [d["gens"].index(sorted({w for gen in g for w in gen if w < n})) for g in d["gates"]]
    S = [tuple(sorted({w - n for gen in g for w in gen if w >= n})) for g in d["gates"]]
    return np.array(G), S


def block_check(d, thbar, H, M):
    """Full compiled Hessian at the coincident compiled point (theta_{G,0}=thbar)."""
    Gi, S = gate_GS(d)
    x = np.zeros(len(d["gates"]))
    for i, s in enumerate(S):
        if s == ():
            x[i] = thbar[Gi[i]]
    t0 = time.time()
    HF = full_hessian(d, jnp.asarray(x))
    subsets = sorted(set(S), key=lambda s: (len(s), s))
    blk = {s: np.array(sorted([i for i in range(len(S)) if S[i] == s], key=lambda i: Gi[i]))
           for s in subsets}
    cross = max(np.max(np.abs(HF[np.ix_(blk[s], blk[u])]))
                for s in subsets for u in subsets if s != u)
    e_sys = np.max(np.abs(HF[np.ix_(blk[()], blk[()])] - H))
    e_M = max(np.max(np.abs(HF[np.ix_(blk[s], blk[s])] - M)) for s in subsets if s != ())
    return dict(L=d["L"], max_abs_H=float(np.max(np.abs(HF))), max_cross=float(cross),
                err_sys_block=float(e_sys), err_anc_blocks_vs_M=float(e_M),
                max_abs_M=float(np.max(np.abs(M))), secs=time.time() - t0)


def spec(d, H, M, g, zb):
    n, one, two, wb = d["n"], d["one"], d["two"], d["wb"]
    ev, U = np.linalg.eigh(M)
    u = U[:, 0]
    hev = np.linalg.eigvalsh(H)
    # Remark formula for diag M: -8 sum_{A: |G cap A| odd} w z (z - t)
    par = [np.bitwise_xor.reduce([(d["idx"] >> (n - 1 - w)) & 1 for w in G]) == 1
           for G in d["gens"]]
    diag_formula = np.array([-8 * np.sum(wb[p] * zb[p] * (zb[p] - d["tA"][p])) for p in par])
    out = dict(lam_min_M=float(ev[0]), lam_max_M=float(ev[-1]), tr_M=float(np.trace(M)),
               max_abs_diag_M=float(np.max(np.abs(np.diag(M)))),
               n_neg_M=int(np.sum(ev < -1e-12 * max(1.0, np.max(np.abs(ev))))),
               n_neg_M_strict=int(np.sum(ev < 0)),
               umin_weight_onebody=float(np.sum(u[one] ** 2)),
               umin_weight_twobody=float(np.sum(u[two] ** 2)),
               lam_min_Hsys=float(hev[0]), lam_max_Hsys=float(hev[-1]),
               n_neg_Hsys=int(np.sum(hev < 0)),
               grad_norm_sys=float(np.linalg.norm(g)),
               grad_norm_sys_onebody=float(np.linalg.norm(g[one])),
               err_diagM_vs_remark=float(np.max(np.abs(np.diag(M) - diag_formula))))
    w2 = wb[(1 << (n - 1)) | (1 << (n - 2))]
    tjk = np.array([d["tA"][(1 << (n - 1 - G[0])) | (1 << (n - 1 - G[1]))]
                    for G in d["gens"] if len(G) == 2])
    out.update(eight_w2_max_tjk=float(8 * w2 * np.max(np.abs(tjk))),
               w2bar=float(w2))
    # one-body/one-body entries vs -8 w_jk t_jk (exact only at the agnostic center)
    pred = np.zeros((n, n))
    for jk, G in enumerate([G for G in d["gens"] if len(G) == 2]):
        pred[G[0], G[1]] = pred[G[1], G[0]] = -8 * w2 * tjk[jk]
    out["err_Mjk_vs_-8wt"] = float(np.max(np.abs(M[np.ix_(one, one)] - pred)))
    # two-body block at the agnostic center: M[{jk},{kl}] = -8 w2 t_{jl}, else 0
    G2 = [set(G) for G in d["gens"] if len(G) == 2]
    pred22 = np.array([[-8 * w2 * d["tA"][sum(1 << (n - 1 - w) for w in (g ^ h))]
                        if len(g & h) == 1 else 0.0 for h in G2] for g in G2])
    out["err_M22_vs_formula"] = float(np.max(np.abs(M[np.ix_(two, two)] - pred22)))
    out["bound_holds"] = bool(ev[0] <= -out["eight_w2_max_tjk"] + 1e-15)
    # sector split: at a factorizing center (two-body angles 0) M = M_11 (+) M_22
    out.update(max_abs_M_one_two=float(np.max(np.abs(M[np.ix_(one, two)]))),
               lam_min_M_onebody=float(np.linalg.eigvalsh(M[np.ix_(one, one)])[0]),
               lam_min_M_twobody=float(np.linalg.eigvalsh(M[np.ix_(two, two)])[0]),
               n_diag_M_neg=int(np.sum(np.diag(M) < -1e-15)),
               max_abs_grad_twobody=float(np.max(np.abs(g[two]))),
               mean_tjk=float(np.mean(tjk)),
               # Rayleigh quotient of the uniform two-body vector (agnostic center):
               # 1^T M_22 1 / 120 = -16 (n-2) w2 mean_{j<l} t_jl
               rayleigh_uniform_twobody=float(np.mean(M[np.ix_(two, two)]) * len(two)),
               pred_rayleigh_agnostic=float(-16 * (n - 2) * w2 * np.mean(tjk)))
    return out, ev


def stage_hessian(do_full=True):
    rows, chks, eigs = [], [], {}
    for ds in DATASETS:
        for cen in ("agnostic", "global"):
            d = load(ds, cen, 1)
            thbar = d["th"][0]
            assert np.allclose(d["th"], thbar[None])          # coincident
            H, M, g, zb, Lc = single_blocks(d, thbar)
            r, ev = spec(d, H, M, g, zb)
            r.update(dataset=ds, center=cen, loss=Lc)
            rows.append(r)
            eigs[f"{ds}_{cen}"] = ev.tolist()
            print(f"[item3] {ds} {cen}: " + " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
                flush=True)
            if do_full:
                for a in (1, 2):
                    da = load(ds, cen, a)
                    c = block_check(da, thbar, H, M)
                    c.update(dataset=ds, point=cen)
                    chks.append(c)
                    print(f"[block] {c}", flush=True)
    if do_full:   # a generic coincident point (random angles, nonzero two-body)
        for a in (1, 2):
            da = load("blobs", "global", a)
            thbar = np.random.default_rng(0).uniform(-np.pi / 2, np.pi / 2, len(da["gens"]))
            H, M, g, zb, _ = single_blocks(da, thbar)
            c = block_check(da, thbar, H, M)
            c.update(dataset="blobs", point="random_coincident_seed0")
            chks.append(c)
            print(f"[block] {c}", flush=True)
    with open(HERE / "item3_mismatch_block.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader(); wr.writerows(rows)
    json.dump(eigs, open(HERE / "item3_M_eigenvalues.json", "w"))
    if chks:
        with open(HERE / "item3_block_check.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in chks for k in r)))
            wr.writeheader(); wr.writerows(chks)


# ----------------------------------------------------------------------------
def stage_gd(steps=3000, lr=0.05):
    rows, traj = [], {}
    for ds in DATASETS:
        for cen in ("agnostic", "global"):
            d = load(ds, cen, 1)
            vg, _ = EW.build_exact_loss_weighted(d["gates"], d["n"], 1, d["X"],
                                                 d["sigmas"], d["pi"])
            Gi, S = gate_GS(d)
            anc = np.array([s != () for s in S])
            thbar = d["th"][0]
            _, M, _, _, _ = single_blocks(d, thbar)
            ev, U = np.linalg.eigh(M)
            x0 = np.where(anc, 0.0, thbar[Gi])
            kicks = {"random": np.random.default_rng(1).standard_normal(len(S)) * anc,
                     "lam_min_eigvec": np.where(anc, U[Gi, 0], 0.0)}
            ancj = jnp.asarray(anc)

            @jax.jit
            def run(x):
                def body(x, _):
                    _, g = vg(x)
                    x = x - lr * g
                    return x, jnp.linalg.norm(jnp.where(ancj, x, 0.0))
                return jax.lax.scan(body, x, None, length=steps)

            for kname, k in kicks.items():
                x = jnp.asarray(x0 + 1e-6 * k / np.linalg.norm(k))
                _, nrm = run(x)
                nrm = np.concatenate([[1e-6], np.asarray(nrm)])
                lg = np.log(nrm)
                rate = lambda a, b: float((lg[b] - lg[a]) / (b - a))  # noqa: E731
                rows.append(dict(dataset=ds, center=cen, kick=kname,
                                 lr=lr, lam_min_M=float(ev[0]),
                                 pred_rate=float(np.log1p(lr * abs(ev[0]))) if ev[0] < 0 else
                                 float(np.log1p(-lr * ev[0])),
                                 rate_0_100=rate(0, 100), rate_0_1000=rate(0, 1000),
                                 rate_1000_3000=rate(1000, steps),
                                 norm_100=float(nrm[100]), norm_1000=float(nrm[1000]),
                                 norm_end=float(nrm[-1]), grows=bool(nrm[-1] > nrm[0])))
                traj[f"{ds}_{cen}_{kname}"] = nrm[::10].tolist()
                print(f"[gd] {rows[-1]}", flush=True)
    with open(HERE / "item3_gd_ancilla_growth.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader(); wr.writerows(rows)
    json.dump(traj, open(HERE / "item3_gd_ancilla_norm_every10.json", "w"))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "items"
    {"items": stage_items, "hessian": stage_hessian, "gd": stage_gd,
     "spectra": lambda: stage_hessian(False)}[st]()
