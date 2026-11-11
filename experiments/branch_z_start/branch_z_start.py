#!/usr/bin/env python3
"""Start each branch from the trained global branch's own <Z> (docs/2026-11-02-branch-z-start-contract.md).

usage: branch_z_start.py selftest | starts | train <ds> | build | check [kill | train | review | all]
Branch-<Z> start, per dataset at its analysis size (blobs a=3, Ising a=2, D-Wave a=3, MNIST a=4) and seed: Z = z1 of the
trained global mixture's branches (the cosine rerun's model of that seed); branch l gets one-body angles
0.5 arccos(Z_j^(l)), two-body angles 0, compiled c0 = W^T th0 / L + Unif(-r0, r0) with r0 = (pi/2)/sqrt(L G) from
default_rng(seed) (iqp_fastcf.inits' recipe, so the perturbation equals the global start's), pi = 1/L.
starts    : builds every start, answers the kill-question (unperturbed start reproduces Z; perturbation equals the
            global start's; perturbed start's test MMD^2 finite) -> runs/starts.json.
train <ds>: the paper's protocol (iqp_fastcf.train, cosine 1e-3 -> 0 over 15,000 steps, k_train 2000, |A| <= 10,
            sigmas(n), stream seeds = the run seeds), all seeds in one call -> runs/<ds>_train.json, params/ (not committed).
build     : results/compare.md and summary.json with the verdicts of C1-C3 per dataset (criteria v1).
Set TMPDIR to a scratch directory.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CR, LPD = REPO / "experiments" / "cosine_rerun", REPO / "experiments" / "learned_partition"
sys.path[:0] = [str(CR), str(REPO / "experiments" / "modesep_bound"), str(LPD)]
from common import DATASETS, EVAL_STEPS, LR, STEPS, data, edges_of, sigmas, theta, z1   # noqa: E402
import learned_partition as LP                                                     # noqa: E402  (sources, rule)
import modesep_bound as MB                                                         # noqa: E402  (stored, A_OF)
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits, walsh         # noqa: E402

A_OF = MB.A_OF
REVIEW = REPO / "docs" / "2026-11-02-branch-z-start-review.md"
fails = 0


def ok(cond, msg):
    global fails
    fails += not cond
    print(("ok   " if cond else "FAIL ") + msg, flush=True)


def start(ds, s, Zb, n, G):
    """(c0 (L, G), pi (L,), unperturbed compiled angles, perturbation) of the branch-<Z> start."""
    a = A_OF[ds]
    L = 1 << a
    th0 = np.zeros((L, G))
    th0[:, :n] = 0.5 * np.arccos(np.clip(Zb, -1, 1))
    r0 = (np.pi / 2) / np.sqrt(L * G)
    pert = np.random.default_rng(s).uniform(-r0, r0, (L, G))
    center = (walsh(a).T @ th0) / L
    return center + pert, np.ones(L) / L, center, pert


def global_perturbation(ds, s, x_tr, edges):
    """The perturbation inside make_inits' global row of seed s (its row minus its center)."""
    a = A_OF[ds]
    L, n = 1 << a, x_tr.shape[1]
    G = n + len(edges)
    c_g = make_inits(x_tr, a, [s], edges)[0][0]
    th0 = np.zeros((L, G))
    th0[:, :n] = 0.5 * np.arccos(np.clip(1.0 - 2.0 * x_tr.mean(0), -1, 1))
    return c_g - (walsh(a).T @ th0) / L


def starts():
    out = {}
    for ds, a in A_OF.items():
        x_tr, x_te = data(ds)
        n = x_tr.shape[1]
        edges = edges_of(n)
        G = n + len(edges)
        ev = make_evaluator(n, a, edges, x_te)
        rows = []
        for s, (Zb, pi_src) in sorted(LP.sources(ds).items()):
            c0, pi, center, pert = start(ds, s, Zb, n, G)
            dz = float(np.abs(z1(theta(center, a), n, edges) - Zb).max())
            dp = float(np.abs(pert - global_perturbation(ds, s, x_tr, edges)).max())
            t0 = float(ev(c0, pi)[0][0])
            rows.append(dict(seed=s, max_abs_Z=float(np.abs(Zb).max()), z_dev=dz, pert_dev=dp, source_pi_uniform=bool(
                np.allclose(pi_src, 1.0 / len(pi_src))), start_test=t0))
            print(f"{ds} seed {s}: |z1(start) - Z| {dz:.1e}, perturbation vs global start {dp:.1e}, start test x1e3 "
                  f"{t0 * 1e3:.4f}", flush=True)
        out[ds] = dict(a=a, rows=rows)
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    (HERE / "runs" / "starts.json").write_text(json.dumps(out, indent=1) + "\n")


def train_ds(ds):
    a = A_OF[ds]
    x_tr, x_te = data(ds)
    n = x_tr.shape[1]
    edges = edges_of(n)
    G = n + len(edges)
    seeds = list(DATASETS[ds][3])
    src = LP.sources(ds)
    c0, pi = map(np.stack, zip(*[start(ds, s, src[s][0], n, G)[:2] for s in seeds]))
    evaluate = make_evaluator(n, a, edges, x_te)
    curves = {}

    def cb(step, params):
        curves[step] = evaluate(params, pi)[0].tolist()
        print(f"{ds} a={a} branch-<Z> start step {step:6d}  test MMD^2 x1e3 mean {np.mean(curves[step]) * 1e3:.4f}",
              flush=True)

    t0 = time.perf_counter()
    out = train(n, a, edges, x_tr, sigmas(n), cosine_lr(LR, STEPS), STEPS, c0, pi, seeds, eval_steps=EVAL_STEPS, callback=cb)
    wall = time.perf_counter() - t0
    finite = bool(np.all(np.isfinite(out["params"])) and np.all(np.isfinite(out["losses"])))
    (HERE / "params").mkdir(exist_ok=True)
    np.savez(HERE / "params" / f"{ds}_train.npz", params=out["params"], pi=pi, seeds=np.array(seeds))
    (HERE / "runs" / f"{ds}_train.json").write_text(json.dumps(dict(
        dataset=ds, a=a, steps=STEPS, lr=f"cosine {LR} -> 0", k_train=2000, seeds=seeds, plan=out["plan"],
        wall_s=round(wall, 1), finite=finite,
        models=[dict(seed=s, test={str(e): curves[e][m] for e in EVAL_STEPS}) for m, s in enumerate(seeds)]),
        indent=1) + "\n")
    print(f"DONE train {ds}: {len(seeds)} models, {wall:.0f} s, finite {finite}", flush=True)


def load(name):
    f = HERE / "runs" / name
    return json.loads(f.read_text()) if f.exists() else None


def kill():
    p = load("starts.json")
    if p is None:
        return None, []
    rows = [(ds, r) for ds, v in p.items() for r in v["rows"]]
    go = (len(rows) == sum(len(DATASETS[d][3]) for d in A_OF)
          and all(r["max_abs_Z"] <= 1 and r["z_dev"] <= 1e-12 and r["pert_dev"] <= 1e-15 and np.isfinite(r["start_test"])
                  for _, r in rows))
    return go, rows


def table(ds):
    a, tr, lp = A_OF[ds], load(f"{ds}_train.json"), json.loads((LPD / "runs" / f"{ds}_train.json").read_text())
    if tr is None:
        return None
    st, sd = MB.stored(ds, a), {r["seed"]: r for r in load("starts.json")[ds]["rows"]}
    trd, lpd = {m["seed"]: m for m in tr["models"]}, {m["seed"]: m for m in lp["models"]}
    return [dict(seed=s, bz=trd[s]["test"][str(STEPS)], cluster=st[(s, "cluster")]["test"][str(STEPS)],
                 glob=st[(s, "global")]["test"][str(STEPS)], learned=lpd[s]["test"][str(STEPS)],
                 bz0=sd[s]["start_test"], glob0=st[(s, "global")]["test"]["0"]) for s in DATASETS[ds][3]]


COLS = [("bz", "branch-<Z> start, trained"), ("cluster", "cluster start (spectral), trained"),
        ("glob", "global start, trained (source)"), ("learned", "learned-partition start, trained"),
        ("bz0", "branch-<Z> start, step 0"), ("glob0", "global start, step 0")]


def verdicts(t):
    col = lambda k: [r[k] for r in t]
    return dict(C1=LP.rule(col("bz"), col("cluster")), C2=LP.rule(col("bz"), col("glob")),
                C3=LP.rule(col("bz"), col("learned")))


def build():
    L = ["# Branch-<Z> start: each branch from the trained global branch's own <Z> (test MMD^2 x1e-3, m̄ = 1..6)", "",
         "Start: one-body angles 0.5 arccos(Z_j^(l)) of the trained global mixture's branch l (same seed), two-body 0, "
         "the global start's perturbation, pi = 1/L; trained with the paper's protocol. A diagnostic: the start uses a "
         "completed run.", ""]
    go, rows = kill()
    if rows:
        L += ["## Kill-question: is the start built correctly?", "",
              "| dataset | seeds | max abs Z | max abs z1(start) - Z | max abs perturbation - global start's | source pi "
              "uniform | start test, mean |", "|---|---|---|---|---|---|---|"]
        for ds in A_OF:
            r = [x for d, x in rows if d == ds]
            L.append(f"| {ds} (a={A_OF[ds]}) | {len(r)} | {max(x['max_abs_Z'] for x in r):.6f} | "
                     f"{max(x['z_dev'] for x in r):.1e} | {max(x['pert_dev'] for x in r):.1e} | "
                     f"{all(x['source_pi_uniform'] for x in r)} | {np.mean([x['start_test'] for x in r]) * 1e3:.4f} |")
        L += ["", f"Kill-question answer: {'YES' if go else 'NO'} (needs deviation <= 1e-12, the same perturbation to 1e-15 (rounding of the reconstruction) and a "
              "finite start for every dataset and seed).", ""]
    summary, vs_all, tabs = dict(kill=dict(answer=go)), {}, {ds: table(ds) for ds in A_OF}
    L += ["## Final test MMD^2 (seed mean +- sd)", "", "| dataset | " + " | ".join(c for _, c in COLS) + " |",
          "|---|" + "---|" * len(COLS)]
    for ds, t in tabs.items():
        if t is None:
            continue
        v = {k: np.array([r[k] for r in t]) for k, _ in COLS}
        L.append(f"| {ds} (a={A_OF[ds]}, {len(t)}) | " + " | ".join(
            f"{v[k].mean() * 1e3:.4f} +- {v[k].std(ddof=1) * 1e3:.4f}" for k, _ in COLS) + " |")
        summary[ds] = {k: dict(mean=float(v[k].mean()), sd=float(v[k].std(ddof=1))) for k, _ in COLS}
        vs_all[ds] = verdicts(t)
    if vs_all:
        L += ["", "## Verdicts (criteria v1, per dataset)", ""]
        for ds, vs in vs_all.items():
            for c, ref in (("C1", "cluster start"), ("C2", "global start"), ("C3", "learned-partition start")):
                L.append(f"- {ds} {c} (vs {ref}): {vs[c][0]}, lower on {vs[c][1]} of {len(tabs[ds])} seeds")
            L.append("")
        L += [f"Verdict {ds} {c}: {vs[c][0]}" for ds, vs in vs_all.items() for c in ("C1", "C2", "C3")] + [""]
        summary["verdicts"] = {ds: {c: dict(verdict=x[0], lower=x[1]) for c, x in vs.items()} for ds, vs in vs_all.items()}
    L += ["## Per seed (x1e-3)", ""]
    for ds, t in tabs.items():
        if t is None:
            continue
        tr = load(f"{ds}_train.json")
        L += [f"**{ds}** (training wall {tr['wall_s']:.0f} s):", "", "| seed | " + " | ".join(c for _, c in COLS) + " |",
              "|---|" + "---|" * len(COLS)]
        L += [f"| {r['seed']} | " + " | ".join(f"{r[k] * 1e3:.4f}" for k, _ in COLS) + " |" for r in t]
        L += ["", "| seed | " + " | ".join(str(e) for e in EVAL_STEPS) + " |", "|---|" + "---|" * len(EVAL_STEPS)]
        L += [f"| {m['seed']} | " + " | ".join(f"{m['test'][str(e)] * 1e3:.4f}" for e in EVAL_STEPS) + " |"
              for m in tr["models"]]
        L += ["| mean | " + " | ".join(f"{np.mean([m['test'][str(e)] for m in tr['models']]) * 1e3:.4f}"
                                       for e in EVAL_STEPS) + " |", ""]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (HERE / "results" / "compare.md").write_text("\n".join(L) + "\n")
    print((HERE / "results" / "compare.md").read_text())


def check(what):
    cm = (HERE / "results" / "compare.md").read_text() if (HERE / "results" / "compare.md").exists() else ""
    if what in ("kill", "all"):
        go, rows = kill()
        ok(bool(go), f"kill-question: start built correctly for all {len(rows)} (dataset, seed)")
    if what in ("train", "", "all"):
        for ds in A_OF:
            tr = load(f"{ds}_train.json")
            ok(tr is not None and tr["finite"] and tr["steps"] == STEPS and tr["seeds"] == list(DATASETS[ds][3]),
               f"{ds}: branch-<Z> start trained, all seeds, finite")
    if what in ("", "all"):
        ok("Kill-question answer:" in cm, "compare.md answers the kill-question")
        for ds in A_OF:
            t = table(ds)
            if t is not None:
                for c, (v, _) in verdicts(t).items():
                    ok(f"Verdict {ds} {c}: {v}" in cm, f"compare.md carries {ds} {c} recomputed from the runs ({v})")
    if what in ("review", "all"):
        r = REVIEW.read_text() if REVIEW.exists() else ""
        ok(bool(re.search(r"(?m)^Verdict:", r)), "review file with a Verdict line")
        sev = dict((i, s) for s, i in re.findall(r"(?m)^- \[([A-Z-]+)\] F(\d+)\b", r))
        for i, s in sev.items():
            mm = re.search(rf"(?m)^- F{i} disposition: (\w+)", r)
            ok(mm is not None, f"F{i} ({s}) has a disposition")
            if s == "MUST-FIX":
                ok(mm is not None and mm.group(1) == "fixed", f"F{i} MUST-FIX is fixed")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)


def selftest():
    for ds, a in A_OF.items():
        x_tr, x_te = data(ds)
        n = x_tr.shape[1]
        edges = edges_of(n)
        G = n + len(edges)
        Zb, pi_src = LP.sources(ds)[7]
        c0, pi, center, pert = start(ds, 7, Zb, n, G)
        dz = np.abs(z1(theta(center, a), n, edges) - Zb).max()
        two = np.abs(theta(center, a)[:, n:]).max()
        dp = np.abs(pert - global_perturbation(ds, 7, x_tr, edges)).max()
        ok(dz <= 1e-12 and two <= 1e-15 and dp <= 1e-15 and np.allclose(pi, 1.0 / (1 << a)) and np.allclose(pi_src, pi),
           f"{ds} seed 7: z1(start) = Z ({dz:.1e}), two-body 0 ({two:.1e}), perturbation = global start's ({dp:.1e}), "
           "pi = 1/L = source pi")
        if ds in ("blobs", "dwave"):
            out = train(n, a, edges, x_tr, sigmas(n), cosine_lr(LR, 2), 2, c0[None], pi[None], [7])
            t = make_evaluator(n, a, edges, x_te)(out["params"], pi[None])[0]
            ok(np.all(np.isfinite(out["params"])) and np.all(np.isfinite(t)), f"{ds}: 2-step smoke run finite "
               f"(test x1e3 {t[0] * 1e3:.4f})")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        selftest()
    elif a == ["starts"]:
        starts()
    elif a[:1] == ["train"] and len(a) == 2 and a[1] in A_OF:
        train_ds(a[1])
    elif a == ["build"]:
        build()
    elif a[:1] == ["check"] and len(a) <= 2:
        check(a[1] if len(a) == 2 else "")
    else:
        sys.exit(__doc__)
