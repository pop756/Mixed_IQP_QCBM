#!/usr/bin/env python3
"""Per-cluster single circuits and the component upper bound (docs/2026-10-30-modesep-argument-contract.md).

usage: modesep_bound.py selftest | run <ds> | build | check
For a target split into components with weights pi, t = sum_k pi_k t^(k), the mixture loss obeys
sqrt(L) <= sum_k pi_k ||z(theta^(k)) - t^(k)||_w (triangle inequality in the correlator seminorm). Here the components
are the clusters of the cluster-initialized run (../cosine_rerun/clusters/<ds>/a<a>_seed<s>.npz, pi = cluster sizes).
run   : for every seed and cluster, one ancilla-free (a = 0) circuit trained on the cluster's training rows with the
        paper's protocol (iqp_fastcf.train, cosine 1e-3 -> 0 over 15,000 steps, k_train 2000, |A| <= 10, sigmas(n),
        global a = 0 initialization from the cluster's rows, stream seed = the run's seed); evaluated with
        make_evaluator (m-bar = 1..6): D_k^2 on the cluster's training rows, the bound B^2 = (sum_k pi_k D_k)^2, the
        assembled partition mixture on the full training and test sets, Delta_whole^2 (the cosine rerun's a = 0 global
        model) on the training set, and the stored test MMD^2 of the a = 0, cluster and global models
        -> runs/<ds>.json (committed), params/<ds>_seed<s>.npz (not committed).
build : results/compare.md and summary.json (seed means +- sd and per-seed values).
check : the four datasets complete for every seed, finite, and the bound holds for every seed.
Analysis sizes: blobs a=3, Ising a=2 (the size of the paper's Sec. V D analysis), D-Wave a=3, MNIST a=4.
Set TMPDIR to a scratch directory.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "cosine_rerun"))
from common import CLUSTERS, DATASETS, LR, PARAMS, RUNS, STEPS, data, edges_of, sigmas   # noqa: E402
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits, walsh               # noqa: E402

A_OF = {"blobs": 3, "ising": 2, "dwave": 3, "mnist": 4}
fails = 0


def ok(cond, msg):
    global fails
    fails += not cond
    print(("ok   " if cond else "FAIL ") + msg, flush=True)


def stored(ds, a):
    r = json.loads((RUNS / ds / f"a{a}_main.json").read_text())
    return {(d["seed"], d["init"]): d for d in r["models"]}


def labels(ds, s):
    a = A_OF[ds]
    lab = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")["train_y"].astype(np.int64)
    return lab, np.bincount(lab, minlength=1 << a) / len(lab)


def whole(ds, s):
    """The cosine rerun's a = 0 global model of seed s (compiled angles (1, G))."""
    z = np.load(PARAMS / f"{ds}_a0_main.npz")
    k = [m for m in range(len(z["names"])) if int(z["seeds"][m]) == s and str(z["names"][m]) == "global"]
    assert len(k) == 1
    return z["params"][k[0]]


def fit_cluster(n, edges, xk, s, steps=STEPS):
    c0, p0 = make_inits(xk, 0, [s], edges)                         # rows: global, agnostic
    out = train(n, 0, edges, xk, sigmas(n), cosine_lr(LR, steps), steps, c0[:1], p0[:1], [s])
    assert np.all(np.isfinite(out["params"]))
    return out["params"][0]                                        # (1, G)


def assemble(a, branch):
    """Compiled angles c (L, G) whose branch angles W c are the given (L, G) a = 0 angles."""
    return np.linalg.solve(walsh(a), branch)


def run(ds, steps=STEPS, seeds=None, out_dir=HERE):
    a = A_OF[ds]
    L = 1 << a
    x_tr, x_te = data(ds)
    n = x_tr.shape[1]
    edges = edges_of(n)
    seeds = list(seeds or DATASETS[ds][3])
    ev_tr, ev_te = make_evaluator(n, a, edges, x_tr), make_evaluator(n, a, edges, x_te)
    ev0_tr = make_evaluator(n, 0, edges, x_tr)
    st, st0 = stored(ds, a), stored(ds, 0)
    rows, t0 = [], time.perf_counter()
    for s in seeds:
        lab, pi = labels(ds, s)
        assert np.allclose(pi, st[(s, "cluster")]["pi"]) and np.all(pi > 0)
        branch, d2 = [], []
        for k in range(L):
            xk = x_tr[lab == k]
            th = fit_cluster(n, edges, xk, s, steps)
            d2.append(float(make_evaluator(n, 0, edges, xk)(th, np.ones(1))[0][0]))
            branch.append(th[0])
            print(f"{ds} seed {s} cluster {k}: {len(xk)} rows, pi {pi[k]:.4f}, D_k^2 x1e3 {d2[-1] * 1e3:.4f}  "
                  f"t {time.perf_counter() - t0:.0f}s", flush=True)
        c = assemble(a, np.stack(branch))
        (out_dir / "params").mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "params" / f"{ds}_seed{s}.npz", branch=np.stack(branch), compiled=c, pi=pi)
        b2 = float(np.dot(pi, np.sqrt(d2))) ** 2
        mix_tr, mix_te = float(ev_tr(c, pi)[0][0]), float(ev_te(c, pi)[0][0])
        whole_tr = float(ev0_tr(whole(ds, s), np.ones(1))[0][0])
        rows.append(dict(seed=s, pi=pi.tolist(), n_k=np.bincount(lab, minlength=L).tolist(), d2_k=d2, bound2=b2,
                         whole_train=whole_tr, partition_train=mix_tr, partition_test=mix_te,
                         whole_test=st0[(s, "global")]["test"]["15000"],
                         cluster_test=st[(s, "cluster")]["test"]["15000"],
                         global_test=st[(s, "global")]["test"]["15000"]))
        print(f"{ds} seed {s}: B^2 {b2 * 1e3:.4f}  Delta_whole^2 {whole_tr * 1e3:.4f}  partition train "
              f"{mix_tr * 1e3:.4f} test {mix_te * 1e3:.4f}  (x1e3)", flush=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs" / f"{ds}.json").write_text(json.dumps(dict(
        dataset=ds, a=a, steps=steps, lr=f"cosine {LR} -> 0", k_train=2000, seeds=seeds,
        clusters=f"experiments/cosine_rerun/clusters/{ds}/a{a}_seed<s>.npz", wall_s=round(time.perf_counter() - t0, 1),
        rows=rows), indent=1) + "\n")
    print(f"DONE {ds}: {len(seeds)} seeds x {L} clusters, {time.perf_counter() - t0:.0f} s", flush=True)


def runs():
    return {ds: json.loads((HERE / "runs" / f"{ds}.json").read_text())
            for ds in A_OF if (HERE / "runs" / f"{ds}.json").exists()}


COLS = [("whole_train", "Delta_whole^2 (a=0, train)"), ("bound2", "B^2 = (sum_k pi_k D_k)^2 (train)"),
        ("ratio", "B^2 / Delta_whole^2"), ("partition_train", "partition mixture, train"),
        ("whole_test", "a=0 model, test"), ("partition_test", "partition mixture, test"),
        ("cluster_test", "trained cluster-init mixture, test"), ("global_test", "trained global mixture, test")]


def build():
    rs = runs()
    L = ["# Per-cluster single circuits and the component upper bound (MMD^2 averaged over m̄ = 1..6)", "",
         "Per dataset at its analysis size: one ancilla-free circuit per cluster of the cluster-initialized run, trained "
         "on that cluster's training rows with the paper's protocol, assembled with pi = cluster sizes. Values x1e-3 "
         "except the ratio; seed mean +- sd.", "",
         "| dataset (a, seeds) | " + " | ".join(c for _, c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    summary = {}
    for ds, r in rs.items():
        v = {key: np.array([row[key] for row in r["rows"]], dtype=float) for key, _ in COLS if key != "ratio"}
        v["ratio"] = v["bound2"] / v["whole_train"]
        summary[ds] = {key: dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0)
                       for key, x in v.items()}
        L.append(f"| {ds} (a={r['a']}, {len(r['rows'])}) | " + " | ".join(
            (f"{v[key].mean():.3f} +- {v[key].std(ddof=1):.3f}" if key == "ratio" else
             f"{v[key].mean() * 1e3:.4f} +- {v[key].std(ddof=1) * 1e3:.4f}") for key, _ in COLS) + " |")
    L += ["", "## Per seed (x1e-3 except the ratio)", ""]
    for ds, r in rs.items():
        L += [f"**{ds}** (a={r['a']}, {r['wall_s']:.0f} s):", "", "| seed | " + " | ".join(c for _, c in COLS) + " |",
              "|---|" + "---|" * len(COLS)]
        for row in r["rows"]:
            row = dict(row, ratio=row["bound2"] / row["whole_train"])
            L.append(f"| {row['seed']} | " + " | ".join(
                f"{row[key]:.3f}" if key == "ratio" else f"{row[key] * 1e3:.4f}" for key, _ in COLS) + " |")
        L.append("")
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (HERE / "results" / "compare.md").write_text("\n".join(L) + "\n")
    print((HERE / "results" / "compare.md").read_text())


def check():
    rs = runs()
    for ds in A_OF:
        r = rs.get(ds)
        ok(r is not None and r["steps"] == STEPS and r["seeds"] == list(DATASETS[ds][3]), f"{ds}: run present, all seeds")
        if r is None:
            continue
        vals = [x for row in r["rows"] for x in row["d2_k"] + [row["bound2"], row["partition_train"], row["partition_test"]]]
        ok(all(np.isfinite(vals)) and all(len(row["d2_k"]) == 1 << r["a"] for row in r["rows"]), f"{ds}: finite, L clusters")
        ok(all(row["partition_train"] <= row["bound2"] * (1 + 1e-9) for row in r["rows"]),
           f"{ds}: partition mixture's training MMD^2 <= B^2 for every seed")
    cm = (HERE / "results" / "compare.md").read_text() if (HERE / "results" / "compare.md").exists() else ""
    ok(all(f"| {ds} (a=" in cm for ds in A_OF), "compare.md carries every dataset")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)


def selftest():
    # 1. labels and pi equal the cluster-initialized runs' for every dataset and seed; no empty cluster
    for ds, a in A_OF.items():
        st, good = stored(ds, a), 0
        for s in DATASETS[ds][3]:
            lab, pi = labels(ds, s)
            good += lab.min() >= 0 and lab.max() < 1 << a and np.all(pi > 0) and np.allclose(pi, st[(s, "cluster")]["pi"])
        ok(good == len(DATASETS[ds][3]), f"{ds} a={a}: labels and pi match the cluster-init runs ({good} seeds)")
    # 2. decomposition identity t = sum_k pi_k t^(k) on the evaluator's subsets, through two random a = 0 models:
    #    sum_k pi_k (|z1 - t_k|^2 - |z2 - t_k|^2) = |z1 - t|^2 - |z2 - t|^2 holds iff the moments decompose
    rng = np.random.default_rng(0)
    for ds in ("blobs", "dwave"):
        x_tr, _ = data(ds)
        n = x_tr.shape[1]
        edges = edges_of(n)
        lab, pi = labels(ds, 7)
        z = np.stack([whole(ds, 7), whole(ds, 11)])                 # trained a = 0 models: nonzero correlators
        full = make_evaluator(n, 0, edges, x_tr)(z, np.ones((2, 1)))[0]
        parts = [make_evaluator(n, 0, edges, x_tr[lab == k])(z, np.ones((2, 1)))[0] for k in range(len(pi))]
        lhs = sum(p * (q[0] - q[1]) for p, q in zip(pi, parts))
        ok(abs(full[0] - full[1]) > 1e-3 * full[0] and abs(lhs - (full[0] - full[1])) <= 1e-9 * abs(full[0] - full[1]),
           f"{ds}: decomposition identity {lhs:.12e} vs {full[0] - full[1]:.12e}")
    # 3. assembling branch angles: W c reproduces them
    th = rng.normal(size=(8, 50))
    ok(np.allclose(walsh(3) @ assemble(3, th), th, atol=1e-12), "assemble: W c = branch angles")
    # 4. the a = 0 protocol reproduces the cosine rerun's stored blobs seed-7 global model
    x_tr, x_te = data("blobs")
    n = x_tr.shape[1]
    edges = edges_of(n)
    th = fit_cluster(n, edges, x_tr, 7)
    te = float(make_evaluator(n, 0, edges, x_te)(th, np.ones(1))[0][0])
    ref = stored("blobs", 0)[(7, "global")]["test"]["15000"]
    ok(abs(te - ref) <= 1e-6 * ref and np.allclose(th, whole("blobs", 7), atol=1e-8),
       f"protocol: blobs seed-7 a=0 global retrained {te:.10e} vs stored {ref:.10e}, "
       f"max angle diff {np.abs(th - whole('blobs', 7)).max():.1e}")
    # 5. a short run (blobs seed 7, 300 steps): finite, and the bound holds on the training set
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    run("blobs", steps=300, seeds=[7], out_dir=tmp)
    row = json.loads((tmp / "runs" / "blobs.json").read_text())["rows"][0]
    ok(np.isfinite(row["partition_test"]) and row["partition_train"] <= row["bound2"] * (1 + 1e-9),
       f"short run: partition train {row['partition_train']:.6e} <= B^2 {row['bound2']:.6e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        selftest()
    elif a[:1] == ["run"] and len(a) == 2 and a[1] in A_OF:
        run(a[1])
    elif a == ["build"]:
        build()
    elif a == ["check"]:
        check()
    else:
        sys.exit(__doc__)
