"""Every number the updated manuscript quotes from the rerun, with its source (findings-ledger provenance).
usage: paper_numbers.py  ->  results/paper_numbers.txt  (key | value | source)"""
import csv, json
from pathlib import Path
import numpy as np
from common import DATASETS, RESULTS, RUNS

LEG = Path(__file__).resolve().parents[2] / "legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/summary.csv"
HEAD = {"blobs": 3, "ising": 3, "mnist": 4, "dwave": 3}
out = []


def put(key, value, source):
    out.append(f"{key} | {value} | {source}")


def run(ds, a, v="main"):
    p = RUNS / ds / f"a{a}_{v}.json"
    return json.load(open(p)) if p.exists() else None


def fin(m):
    return m["test"]["15000"]


def models(ds, a, init, v="main"):
    r = run(ds, a, v)
    return {m["seed"]: m for m in r["models"] if m["init"] == init} if r else {}


def ms(vals, k=1e3):
    v = np.asarray(vals, float) * k
    return f"{v.mean():.4g} ± {v.std(ddof=1) if len(v) > 1 else 0:.2g} (x1e-3, n={len(v)})"


refs = {}
for r in csv.DictReader(open(RESULTS / "refs.csv")):
    refs.setdefault(r["dataset"], []).append(float(r["ref_train_vs_test"]))
for ds in DATASETS:
    put(f"ref {ds}", f"{np.mean(refs[ds]) * 1e3:.4g}e-3", "results/refs.csv ref_train_vs_test, mean over m-bar")
    for a in DATASETS[ds][2]:
        for init in ("global", "agnostic", "cluster"):
            mm = models(ds, a, init)
            if mm:
                put(f"test {ds} a{a} {init}", ms([fin(m) for m in mm.values()]), f"runs/{ds}/a{a}_main.json test[15000]")
        g, c = models(ds, a, "global"), models(ds, a, "cluster")
        if c:
            wins = sum(fin(c[s]) < fin(g[s]) for s in c)
            put(f"cluster<global seeds {ds} a{a}", f"{wins}/{len(c)}", f"runs/{ds}/a{a}_main.json paired by seed")
            put(f"ratio cluster/global {ds} a{a}", f"{np.mean([fin(m) for m in c.values()]) / np.mean([fin(m) for m in g.values()]):.3f}",
                "seed means")
    a = HEAD[ds]
    for step in ("250", "1000", "5000"):
        for init in ("global", "agnostic", "cluster"):
            mm = models(ds, a, init)
            put(f"test@{step} {ds} a{a} {init}", ms([m["test"][step] for m in mm.values()]), f"runs/{ds}/a{a}_main.json test[{step}]")
    # one-body (fc1) vs full, paired
    ob, fu = models(ds, a, "cluster", "fc1"), models(ds, a, "cluster")
    rr = [fin(fu[s]) and fin(ob[s]) / fin(fu[s]) for s in fu]
    put(f"one-body {ds} a{a}", ms([fin(m) for m in ob.values()]), f"runs/{ds}/a{a}_fc1.json")
    put(f"one-body/full ratio {ds} a{a}", f"{min(rr):.3g}-{max(rr):.3g} (mean {np.mean(rr):.3g}), paired over {len(rr)} seeds", "fc1 / main cluster")
    per_ob = np.mean([m["test_per_mbar"] for m in ob.values()], 0)
    per_fu = np.mean([m["test_per_mbar"] for m in fu.values()], 0)
    put(f"one-body/full per mbar {ds} a{a}", " ".join(f"{x:.3g}" for x in per_ob / per_fu), "seed means per m-bar = 1..6")
# product baselines at the headline size
leg = {(r["dataset"], int(r["ancilla"]), r["column"]): r for r in csv.DictReader(open(LEG))}
for ds in ("blobs", "ising"):
    for col in ("center", "em"):
        r = leg[(ds, HEAD[ds], col)]
        put(f"product {col} {ds} a{HEAD[ds]}", f"{float(r['mean']) * 1e3:.4g} ± {float(r['sd'] or 0) * 1e3:.2g} (x1e-3)",
            "legacy bernoulli_baseline/summary.csv (exact evaluation)")
    for a in DATASETS[ds][2]:
        r = leg.get((ds, a, "em"))
        if r:
            put(f"product em {ds} a{a}", f"{float(r['mean']) * 1e3:.4g}e-3", "legacy summary.csv")
for ds in ("mnist", "dwave"):
    rows = list(csv.DictReader(open(RESULTS / f"product_baselines_{ds}.csv")))
    for a in DATASETS[ds][2]:
        for col in ("center", "em"):
            v = [float(r["test_mmd2"]) for r in rows if r["method"] == col and int(r["a"]) == a]
            put(f"product {col} {ds} a{a}", ms(v), f"results/product_baselines_{ds}.csv")
# EM vs trained cluster at every a
for ds in DATASETS:
    for a in DATASETS[ds][2]:
        c = models(ds, a, "cluster")
        if not c:
            continue
        if ds in ("blobs", "ising"):
            em = float(leg[(ds, a, "em")]["mean"])
        else:
            em = np.mean([float(r["test_mmd2"]) for r in csv.DictReader(open(RESULTS / f"product_baselines_{ds}.csv"))
                          if r["method"] == "em" and int(r["a"]) == a])
        put(f"EM/trained-cluster {ds} a{a}", f"{em / np.mean([fin(m) for m in c.values()]):.3g}", "EM mean / cluster seed mean")
# two-body ablation
z = list(csv.DictReader(open(RESULTS / "zero2b.csv")))
for ds in ("mnist", "dwave"):
    f = [float(r["test_mmd2_zero2b"]) / float(r["test_mmd2"]) for r in z if r["dataset"] == ds]
    put(f"zero2b {ds}", ms([float(r["test_mmd2_zero2b"]) for r in z if r["dataset"] == ds]) + f"; factor {min(f):.3g}-{max(f):.3g}",
        "results/zero2b.csv")
# equal-weight control: peak over m-bar per seed, then mean / sd over seeds
for (ds, a) in (("blobs", 1), ("blobs", 2), ("ising", 2), ("ising", 3), ("mnist", 4), ("dwave", 3)):
    eq = run(ds, a, "eq")
    seeds = eq["seeds"]
    for name, mm in (("global", models(ds, a, "global")), ("cluster", models(ds, a, "cluster")),
                     ("cluster_eq", {m["seed"]: m for m in eq["models"]})):
        v = [max(mm[s]["test_per_mbar"]) for s in seeds]
        put(f"eq-peak {ds} a{a} {name}", ms(v), f"runs/{ds}/a{a}_{{main,eq}}.json max over m-bar")
        put(f"eq-mean {ds} a{a} {name}", ms([fin(mm[s]) for s in seeds]), "mean over m-bar")
# mode index and Spearman with bootstrap (as make_fig_diversity_scatter.py)
from scipy.stats import spearmanr
mode = list(csv.DictReader(open(RESULTS / "mode_index.csv")))
for ds in DATASETS:
    rows = [r for r in mode if r["dataset"] == ds]
    for init in ("global", "cluster", "agnostic"):
        v = [float(r["N_mode"]) for r in rows if r["init"] == init]
        put(f"N_mode {ds} {init}", f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}", "results/mode_index.csv")
    xy = np.array([[float(r["D2"]), float(r["mmd2"]), float(r["N_mode"])] for r in rows])
    rng = np.random.default_rng(12345)
    for col, name in ((0, "D2"), (2, "Nmode")):
        rho = spearmanr(xy[:, col], xy[:, 1]).statistic
        b = [spearmanr(xy[i, col], xy[i, 1]).statistic for i in (rng.integers(0, len(xy), len(xy)) for _ in range(10_000))]
        b = [x for x in b if not np.isnan(x)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        put(f"spearman {name}-mmd2 {ds}", f"{rho:+.2f} ({lo:+.2f}, {hi:+.2f})", "mode_index.csv, 10^4 bootstrap, seed 12345")
    put(f"spearman Nmode-D2 {ds}", f"{spearmanr(xy[:, 2], xy[:, 0]).statistic:+.2f}", "mode_index.csv")
    for init in ("global", "cluster", "agnostic"):
        v = [float(r["D2"]) for r in rows if r["init"] == init]
        put(f"D2 {ds} {init}", f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}", "results/branch_diversity.csv")
# MNIST data-agnostic
for a in (1, 2, 3, 4):
    v = [fin(m) for m in models("mnist", a, "agnostic").values()]
    put(f"mnist agnostic a{a}", f"{min(v) * 1e3:.3g}-{max(v) * 1e3:.3g} x1e-3; >2x a0 global: {sum(x > 2 * 1.019e-3 for x in v)}/{len(v)}",
        "runs/mnist/a*_main.json")
(RESULTS / "paper_numbers.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))

# ---- additions: SM diagnostics, equal-weight tests, data-agnostic MNIST branches ----
out2 = []
fr = list(csv.DictReader(open(RESULTS / "factorization_residuals.csv")))
for ds in DATASETS:
    Lmax = max(int(r["L"]) for r in fr if r["dataset"] == ds)
    rows = [r for r in fr if r["dataset"] == ds and int(r["L"]) == Lmax]
    q_med = max(float(r["eps"]) ** (2 / int(r["k"])) for r in rows)
    q_95 = max(float(r["eps_q95"]) ** (2 / int(r["k"])) for r in rows)
    out2.append(f"q_eff max k<=6 {ds} L={Lmax} | median {q_med:.2f}, q95 {q_95:.2f} | results/factorization_residuals.csv")
    out2.append(f"witness heavy {ds} L={Lmax} | {float(rows[0]['witness_heavy']):.2f} (worst {float(rows[0]['witness_worst']):.2f}) | factorization_residuals.csv")
from scipy.stats import wilcoxon
for (ds, a) in (("mnist", 4), ("dwave", 3)):
    eq = {m["seed"]: fin(m) for m in run(ds, a, "eq")["models"]}
    c = {s: fin(m) for s, m in models(ds, a, "cluster").items()}
    ss = sorted(eq)
    p = wilcoxon([eq[s] for s in ss], [c[s] for s in ss]).pvalue
    hi = sum(eq[s] > c[s] for s in ss)
    out2.append(f"eq vs cluster {ds} a{a} | Wilcoxon p={p:.2f}, eq higher on {hi}/{len(ss)} seeds | runs/{ds}/a{a}_{{eq,main}}.json")
z = np.load(RESULTS / "mnist_branch_marginals.npz")
from common import data
xte = data("mnist")[1].astype(np.float64)
for s in (7, 11, 23, 1234, 2025):
    Bm, pi = np.asarray(z[f"agnostic_seed{s}"], np.float64), np.asarray(z[f"agnostic_seed{s}_pi"])
    flat = (np.abs(Bm - 0.5) < 0.25).mean(1) > 0.5            # most pixels of the branch within 0.25 of 1/2
    Zc = np.clip(1 - 2 * Bm, -1 + 1e-12, 1 - 1e-12)
    logq = np.log((1 + Zc) / 2) @ (1 - xte).T + np.log((1 - Zc) / 2) @ xte.T + np.log(pi)[:, None]
    share = float(np.isin(logq.argmax(0), np.nonzero(flat)[0]).mean())
    out2.append(f"mnist agnostic a4 seed {s} | {int(flat.sum())} near-uniform branches (> half the pixels within 0.25 of 1/2), test share {share:.4f} | mnist_branch_marginals.npz")
(RESULTS / "paper_numbers.txt").write_text("\n".join(out + out2) + "\n")
print("\n".join(out2))

# ---- additions 2 ----
out3 = []
r = run("dwave", 3)
L_g = np.mean([m["train_loss_50"] for m in r["models"] if m["init"] == "global"], 0)
L_c = np.mean([m["train_loss_50"] for m in r["models"] if m["init"] == "cluster"], 0)
cross = int(np.argmax(L_g < L_c)) * 50 + 50
out3.append(f"dwave a3 training-loss crossing | global mean below cluster mean from step {cross} | runs/dwave/a3_main.json train_loss_50")
for s, init in ((314, "global"), (314, "agnostic"), (11, "cluster")):
    out3.append(f"ising a2 {init} seed {s} test | {fin(models('ising', 2, init)[s]) * 1e3:.4g}e-3 | runs/ising/a2_main.json")
for (ds, a) in (("blobs", 1), ("blobs", 2), ("ising", 2), ("ising", 3), ("mnist", 4), ("dwave", 3)):
    eq = {m["seed"]: max(m["test_per_mbar"]) for m in run(ds, a, "eq")["models"]}
    c = {s: max(m["test_per_mbar"]) for s, m in models(ds, a, "cluster").items()}
    ss = sorted(eq)
    p = wilcoxon([eq[s] for s in ss], [c[s] for s in ss]).pvalue if len(ss) > 3 else float("nan")
    out3.append(f"eq-peak vs cluster {ds} a{a} | Wilcoxon p={p:.2f}, eq higher on {sum(eq[s] > c[s] for s in ss)}/{len(ss)} | max over m-bar")
from scipy.stats import binom
for ds in DATASETS:
    n = 16 if ds in ("blobs", "ising") else (784 if ds == "mnist" else 484)
    drop = [binom.sf(10, n, mb / n) for mb in range(1, 7)]
    out3.append(f"kernel weight with |A| > 10 {ds} | max over m-bar {max(drop):.4f} (m-bar=6) | Binomial(n, m-bar/n) tail")
(RESULTS / "paper_numbers.txt").write_text("\n".join(out + out2 + out3) + "\n")
print("\n".join(out3))
