#!/usr/bin/env python3
"""Start from the partition the trained global mixture learned (docs/2026-10-30-learned-partition-contract.md).

usage: learned_partition.py selftest | partition | train <ds> | parts <ds> | build | check [kill|train|runs|review|all]
partition : for every dataset at its analysis size (blobs a=3, Ising a=2, D-Wave a=3, MNIST a=4) and seed, the trained
            global mixture of the cosine rerun (source); its branches' one-body correlators (common.z1) define product
            distributions, and each training row goes to argmax_l gamma_l(y), the one-body responsibility of
            analyze.mode_index -> runs/partition.json (N_mode recomputed vs results/mode_index.csv, part sizes, adjusted
            Rand index against the spectral clusters) and runs/labels_<ds>.npz (committed).
train <ds>: the learned-partition start, make_inits' cluster row with the learned labels (one-body angles from each
            part's one-body <Z>, two-body 0, pi = part sizes, the usual perturbation), trained with the paper's protocol
            (iqp_fastcf.train, cosine 1e-3 -> 0 over 15,000 steps, k_train 2000, |A| <= 10, sigmas(n), stream seeds =
            the run seeds), all seeds in one call -> runs/<ds>_train.json, params/<ds>_train.npz (not committed).
parts <ds>: one a = 0 circuit per part on its training rows (modesep_bound.fit_cluster), assembled with pi = part
            sizes -> runs/<ds>_parts.json, params/<ds>_parts_seed<s>.npz (not committed).
build     : results/compare.md and summary.json with the kill-question and the verdicts of C1-C3 per dataset (v1).
Set TMPDIR to a scratch directory.
"""
import json, re, subprocess, sys, time
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CR, MSB = REPO / "experiments" / "cosine_rerun", REPO / "experiments" / "modesep_bound"
sys.path[:0] = [str(CR), str(MSB)]
from common import CLUSTERS, DATASETS, EVAL_STEPS, LR, RESULTS as CR_RESULTS, STEPS, data, edges_of, sigmas, theta, z1  # noqa
import analyze as AN                                                # noqa: E402  (mode_index, params_of)
import modesep_bound as MB                                          # noqa: E402  (fit_cluster, assemble, stored, A_OF)
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits  # noqa: E402

A_OF = MB.A_OF
REVIEW = REPO / "docs" / "2026-10-30-learned-partition-review.md"
fails = 0


def ok(cond, msg):
    global fails
    fails += not cond
    print(("ok   " if cond else "FAIL ") + msg, flush=True)


def responsibilities(Zb, pi, x):
    """gamma (L, M): the one-body responsibilities of analyze.mode_index for strings x (M, n)."""
    Zc = np.clip(Zb, -1 + 1e-12, 1 - 1e-12)
    lg = np.log(np.clip(pi, 1e-300, None))[:, None] + np.log((1 + Zc) / 2) @ (1 - x).T + np.log((1 - Zc) / 2) @ x.T
    g = np.exp(lg - lg.max(0))
    return g / g.sum(0)


def sources(ds):
    """seed -> (one-body correlators (L, n), pi (L,)) of the trained global mixtures."""
    a = A_OF[ds]
    P, pi, seeds, names, edges = AN.params_of(ds, a, "main")
    n = P.shape[2] - len(edges)
    return {int(s): (z1(theta(P[m], a), n, edges), pi[m]) for m, (s, nm) in enumerate(zip(seeds, names)) if nm == "global"}


def stored_nmode():
    rows = (CR_RESULTS / "mode_index.csv").read_text().splitlines()
    head = rows[0].split(",")
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r.split(",")))
        out[(d["dataset"], d["init"], int(d["seed"]))] = float(d["N_mode"])
    return out


def ari(u, v):
    """Adjusted Rand index of two labelings."""
    cont = np.zeros((u.max() + 1, v.max() + 1), dtype=np.int64)
    np.add.at(cont, (u, v), 1)
    s = sum(comb(int(x), 2) for x in cont.ravel())
    a_ = sum(comb(int(x), 2) for x in cont.sum(1))
    b_ = sum(comb(int(x), 2) for x in cont.sum(0))
    e = a_ * b_ / comb(len(u), 2)
    return (s - e) / (0.5 * (a_ + b_) - e)


def labels_of(ds):
    z = np.load(HERE / "runs" / f"labels_{ds}.npz")
    return {int(k[4:]): z[k].astype(np.int64) for k in z.files}


def partition():
    nm_ref, out = stored_nmode(), {}
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    for ds, a in A_OF.items():
        x_tr, x_te = data(ds)
        L = 1 << a
        xtr, xte = x_tr.astype(np.float64), x_te.astype(np.float64)
        rows, labs = [], {}
        for s, (Zb, pi) in sorted(sources(ds).items()):
            nm = AN.mode_index(Zb, pi, xte)
            lab = np.concatenate([responsibilities(Zb, pi, xtr[i:i + 10000]).argmax(0) for i in range(0, len(xtr), 10000)])
            spec = MB.labels(ds, s)[0]
            frac = np.bincount(lab, minlength=L) / len(lab)
            rows.append(dict(seed=s, N_mode=nm, N_mode_stored=nm_ref[(ds, "global", s)], part_sizes=frac.tolist(),
                             parts_ge_1pct=int((frac >= 0.01).sum()), empty_parts=int((frac == 0).sum()),
                             ari_vs_spectral=float(ari(lab, spec))))
            labs[f"seed{s}"] = lab.astype(np.int8)
            print(f"{ds} seed {s}: N_mode {nm:.6f} (stored {nm_ref[(ds, 'global', s)]:.6f}), parts >= 1 % "
                  f"{rows[-1]['parts_ge_1pct']}/{L}, ARI vs spectral {rows[-1]['ari_vs_spectral']:.3f}", flush=True)
        np.savez_compressed(HERE / "runs" / f"labels_{ds}.npz", **labs)
        out[ds] = dict(a=a, rows=rows)
    (HERE / "runs" / "partition.json").write_text(json.dumps(out, indent=1) + "\n")


def train_ds(ds):
    a = A_OF[ds]
    x_tr, x_te = data(ds)
    n = x_tr.shape[1]
    edges = edges_of(n)
    seeds = list(DATASETS[ds][3])
    labs = labels_of(ds)
    c0, pi = [], []
    for s in seeds:
        cs, ps = make_inits(x_tr, a, [s], edges, labels=labs[s])    # rows: global, agnostic, cluster
        c0.append(cs[2]); pi.append(ps[2])
    c0, pi = np.stack(c0), np.stack(pi)
    evaluate = make_evaluator(n, a, edges, x_te)
    curves = {}

    def cb(step, params):
        curves[step] = evaluate(params, pi)[0].tolist()
        print(f"{ds} a={a} learned-partition start step {step:6d}  test MMD^2 x1e3 mean {np.mean(curves[step]) * 1e3:.4f}",
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
        models=[dict(seed=s, pi=pi[m].tolist(), test={str(e): curves[e][m] for e in EVAL_STEPS})
                for m, s in enumerate(seeds)]), indent=1) + "\n")
    print(f"DONE train {ds}: {len(seeds)} models, {wall:.0f} s, finite {finite}", flush=True)


def parts_ds(ds):
    a = A_OF[ds]
    L = 1 << a
    x_tr, x_te = data(ds)
    n = x_tr.shape[1]
    edges = edges_of(n)
    seeds = list(DATASETS[ds][3])
    labs = labels_of(ds)
    ev_tr, ev_te = make_evaluator(n, a, edges, x_tr), make_evaluator(n, a, edges, x_te)
    rows, t0 = [], time.perf_counter()
    (HERE / "params").mkdir(exist_ok=True)
    for s in seeds:
        lab = labs[s]
        pi = np.bincount(lab, minlength=L) / len(lab)
        branch, d2 = [], []
        for k in range(L):
            xk = x_tr[lab == k]
            if len(xk) == 0:                                          # weight 0: any branch, the whole-data a = 0 model
                branch.append(MB.whole(ds, s)[0]); d2.append(0.0)
                continue
            th = MB.fit_cluster(n, edges, xk, s)
            d2.append(float(make_evaluator(n, 0, edges, xk)(th, np.ones(1))[0][0]))
            branch.append(th[0])
            print(f"{ds} seed {s} part {k}: {len(xk)} rows, D_k^2 x1e3 {d2[-1] * 1e3:.4f}  t {time.perf_counter() - t0:.0f}s",
                  flush=True)
        c = MB.assemble(a, np.stack(branch))
        np.savez(HERE / "params" / f"{ds}_parts_seed{s}.npz", branch=np.stack(branch), compiled=c, pi=pi)
        rows.append(dict(seed=s, pi=pi.tolist(), d2_k=d2, bound2=float(np.dot(pi, np.sqrt(d2))) ** 2,
                         partition_train=float(ev_tr(c, pi)[0][0]), partition_test=float(ev_te(c, pi)[0][0])))
        print(f"{ds} seed {s}: learned-partition mixture train {rows[-1]['partition_train'] * 1e3:.4f}  test "
              f"{rows[-1]['partition_test'] * 1e3:.4f}  B^2 {rows[-1]['bound2'] * 1e3:.4f}  (x1e3)", flush=True)
    (HERE / "runs" / f"{ds}_parts.json").write_text(json.dumps(dict(
        dataset=ds, a=a, steps=STEPS, seeds=seeds, wall_s=round(time.perf_counter() - t0, 1), rows=rows), indent=1) + "\n")
    print(f"DONE parts {ds}: {len(seeds)} seeds, {time.perf_counter() - t0:.0f} s", flush=True)


def load(name):
    f = HERE / "runs" / name
    return json.loads(f.read_text()) if f.exists() else None


def kill():
    p = load("partition.json")
    if p is None:
        return None, []
    rows = [(ds, r["seed"], abs(r["N_mode"] - r["N_mode_stored"]) / r["N_mode_stored"], r["parts_ge_1pct"])
            for ds, v in p.items() for r in v["rows"]]
    return all(d <= 1e-9 and k >= 2 for _, _, d, k in rows) and len(rows) == sum(len(DATASETS[d][3]) for d in A_OF), rows


def rule(x, y):
    """v1: promote if mean(x) < mean(y) and x lower on most seeds (>= 8 of 10, >= 4 of 5); reject if mean(x) > mean(y)."""
    lower, need = int(sum(p < q for p, q in zip(x, y))), (8 if len(x) == 10 else 4)
    if np.mean(x) < np.mean(y) and lower >= need:
        return "promote", lower
    return ("reject" if np.mean(x) > np.mean(y) else "hold"), lower


def table(ds):
    """Per-seed values, or None when a run is missing."""
    a, tr, pa = A_OF[ds], load(f"{ds}_train.json"), load(f"{ds}_parts.json")
    sp = json.loads((MSB / "runs" / f"{ds}.json").read_text())
    if tr is None or pa is None:
        return None
    st, st0 = MB.stored(ds, a), MB.stored(ds, 0)
    trd, pad, spd = ({r["seed"]: r for r in tr["models"]}, {r["seed"]: r for r in pa["rows"]},
                     {r["seed"]: r for r in sp["rows"]})
    return [dict(seed=s, learned_start=trd[s]["test"][str(STEPS)], cluster=st[(s, "cluster")]["test"][str(STEPS)],
                 glob=st[(s, "global")]["test"][str(STEPS)], learned_mix=pad[s]["partition_test"],
                 spectral_mix=spd[s]["partition_test"], a0=st0[(s, "global")]["test"][str(STEPS)],
                 learned_mix_train=pad[s]["partition_train"], learned_bound2=pad[s]["bound2"])
            for s in DATASETS[ds][3]]


COLS = [("learned_start", "learned-partition start, trained"), ("cluster", "cluster start (spectral), trained"),
        ("glob", "global start, trained (source)"), ("learned_mix", "learned-partition mixture (assembled)"),
        ("spectral_mix", "spectral partition mixture (assembled)"), ("a0", "a=0 circuit")]


def verdicts(t):
    col = lambda k: [r[k] for r in t]
    return dict(C1=rule(col("learned_start"), col("cluster")), C2=rule(col("learned_start"), col("glob")),
                C3=rule(col("learned_mix"), col("spectral_mix")))


def build():
    L = ["# Starting from the partition the trained global mixture learned (test MMD^2 x1e-3, m̄ = 1..6)", "",
         "Learned partition: each training row goes to argmax_l gamma_l(y), the one-body responsibility of the trained "
         "global mixture of the same seed (as in N_mode). A diagnostic of the initial partition: the partition comes "
         "from a completed run.", ""]
    go, krows = kill()
    p = load("partition.json")
    if p is not None:
        L += ["## Kill-question: is the partition read correctly?", "",
              "| dataset | seeds | max rel. deviation of N_mode from mode_index.csv | min parts >= 1 % of rows | "
              "empty parts | ARI vs spectral (mean) | N_mode (mean) |", "|---|---|---|---|---|---|---|"]
        for ds, v in p.items():
            r = v["rows"]
            L.append(f"| {ds} (a={v['a']}) | {len(r)} | {max(x[2] for x in krows if x[0] == ds):.1e} | "
                     f"{min(x['parts_ge_1pct'] for x in r)} of {1 << v['a']} | {sum(x['empty_parts'] for x in r)} | "
                     f"{np.mean([x['ari_vs_spectral'] for x in r]):.3f} | {np.mean([x['N_mode'] for x in r]):.3f} |")
        L += ["", f"Kill-question answer: {'YES' if go else 'NO'} (needs deviation <= 1e-9 and >= 2 parts with >= 1 % "
              "of the rows, for every dataset and seed).", ""]
    summary, vs_all = dict(kill=dict(answer=go)), {}
    L += ["## Final test MMD^2 (seed mean +- sd)", "", "| dataset | " + " | ".join(c for _, c in COLS) + " |",
          "|---|" + "---|" * len(COLS)]
    tabs = {ds: table(ds) for ds in A_OF}
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
            n_ = len(tabs[ds])
            for c, ref in (("C1", "cluster start"), ("C2", "global start"), ("C3", "spectral partition mixture")):
                L.append(f"- {ds} {c} (vs {ref}): {vs[c][0]}, lower on {vs[c][1]} of {n_} seeds")
            L.append("")
        L += [f"Verdict {ds} {c}: {vs[c][0]}" for ds, vs in vs_all.items() for c in ("C1", "C2", "C3")] + [""]
        summary["verdicts"] = {ds: {c: dict(verdict=x[0], lower=x[1]) for c, x in vs.items()} for ds, vs in vs_all.items()}
    L += ["## Per seed (x1e-3)", ""]
    for ds, t in tabs.items():
        if t is None:
            continue
        prow = {r["seed"]: r for r in p[ds]["rows"]}
        L += [f"**{ds}**:", "", "| seed | " + " | ".join(c for _, c in COLS) + " | learned mixture, train | its B^2 | "
              "N_mode (source) | ARI vs spectral | part sizes |", "|---|" + "---|" * (len(COLS) + 5)]
        L += [f"| {r['seed']} | " + " | ".join(f"{r[k] * 1e3:.4f}" for k, _ in COLS)
              + f" | {r['learned_mix_train'] * 1e3:.4f} | {r['learned_bound2'] * 1e3:.4f} | "
              f"{prow[r['seed']]['N_mode']:.3f} | {prow[r['seed']]['ari_vs_spectral']:.3f} | "
              + " ".join(f"{x:.3f}" for x in prow[r['seed']]['part_sizes']) + " |" for r in t]
        tr = load(f"{ds}_train.json")
        L += ["", f"Learned-partition start, test along the steps (wall {tr['wall_s']:.0f} s; part circuits "
              f"{load(f'{ds}_parts.json')['wall_s']:.0f} s):", "",
              "| seed | " + " | ".join(str(e) for e in EVAL_STEPS) + " |", "|---|" + "---|" * len(EVAL_STEPS)]
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
        ok(bool(go), f"kill-question: N_mode reproduced and >= 2 parts with >= 1 % for all {len(rows)} (dataset, seed)")
    if what in ("train", "runs", "", "all"):
        for ds in A_OF:
            tr = load(f"{ds}_train.json")
            ok(tr is not None and tr["finite"] and tr["steps"] == STEPS and tr["seeds"] == list(DATASETS[ds][3]),
               f"{ds}: learned-partition start trained, all seeds, finite")
    if what in ("runs", "", "all"):
        for ds in A_OF:
            pa = load(f"{ds}_parts.json")
            ok(pa is not None and [r["seed"] for r in pa["rows"]] == list(DATASETS[ds][3])
               and all(r["partition_train"] <= r["bound2"] * (1 + 1e-9) or r["bound2"] == 0 for r in pa["rows"]),
               f"{ds}: part circuits assembled for all seeds, bound holds")
    if what in ("", "all"):
        ok("Kill-question answer:" in cm, "compare.md answers the kill-question")
        for ds in A_OF:
            t = table(ds)
            if t is None:
                continue
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
    nm_ref = stored_nmode()
    for ds, a in A_OF.items():
        x_tr, x_te = data(ds)
        Zb, pi = sources(ds)[7]
        nm = AN.mode_index(Zb, pi, x_te.astype(np.float64))
        ok(abs(nm - nm_ref[(ds, "global", 7)]) <= 1e-9 * nm, f"{ds}: N_mode of the seed-7 global model {nm:.10f} "
           f"vs stored {nm_ref[(ds, 'global', 7)]:.10f}")
        g = responsibilities(Zb, pi, x_tr[:2000].astype(np.float64))
        lab = g.argmax(0)
        ok(np.abs(g.sum(0) - 1).max() < 1e-12 and lab.min() >= 0 and lab.max() < 1 << a,
           f"{ds}: responsibilities sum to 1 (max dev {np.abs(g.sum(0) - 1).max():.1e}), labels in range")
        # make_inits with these labels: pi = part sizes, one-body angles within the perturbation of the part moments
        n = x_tr.shape[1]
        edges = edges_of(n)
        xs = x_tr[:2000]
        cs, ps = make_inits(xs, a, [7], edges, labels=lab)
        frac = np.bincount(lab, minlength=1 << a) / len(lab)
        th = theta(cs[2], a)[:, :n]
        want = np.array([0.5 * np.arccos(np.clip(1 - 2 * xs[lab == l].mean(0), -1, 1)) if np.any(lab == l)
                         else 0.5 * np.arccos(np.clip(1 - 2 * xs.mean(0), -1, 1)) for l in range(1 << a)])
        r0 = (np.pi / 2) / np.sqrt((1 << a) * (n + len(edges)))
        ok(np.allclose(ps[2], frac) and np.abs(th - want).max() <= (1 << a) * r0 + 1e-12,
           f"{ds}: make_inits pi = part sizes, one-body start = part moments within {np.abs(th - want).max():.1e} "
           f"(<= L r0 = {(1 << a) * r0:.1e})")
    ok(ari(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == 1.0 and abs(ari(np.array([0, 1, 0, 1]),
       np.array([0, 0, 1, 1])) + 0.5) < 1e-12, "adjusted Rand index on two small cases")
    r = subprocess.run([sys.executable, str(MSB / "modesep_bound.py"), "selftest"], capture_output=True, text=True)
    ok(r.returncode == 0, f"modesep_bound selftest still passes ({r.stdout.strip().splitlines()[-1]})")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        selftest()
    elif a == ["partition"]:
        partition()
    elif a[:1] in (["train"], ["parts"]) and len(a) == 2 and a[1] in A_OF:
        (train_ds if a[0] == "train" else parts_ds)(a[1])
    elif a == ["build"]:
        build()
    elif a[:1] == ["check"] and len(a) <= 2:
        check(a[1] if len(a) == 2 else "")
    else:
        sys.exit(__doc__)
