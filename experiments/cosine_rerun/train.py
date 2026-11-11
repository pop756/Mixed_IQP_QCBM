"""Train one (dataset, a, variant) group of models with iqp_fastcf (cosine lr) and evaluate it during training.
variants:
  main : global / agnostic / cluster initialization (a = 0: global / agnostic) x every training seed
  eq   : equal-weight control: cluster initialization from the capacity-balanced groups, pi = 1/L (clusters.EQ_*)
  fc1  : one-body mixture (no two-body gates): cluster initialization (a = 0: global) x every training seed
usage: train.py <ds> <a> <variant>  ->  runs/<ds>/a<a>_<variant>.json (summary), params/<ds>_a<a>_<variant>.npz
"""
import json, sys, time
import numpy as np
from common import (CLUSTERS, DATASETS, EVAL_STEPS, LR, PARAMS, RUNS, STEPS, data, edges_of, sigmas)
from clusters import EQ_SEEDS
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits

ds, a, variant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert variant in ("main", "eq", "fc1") and a in DATASETS[ds][2]
x_tr, x_te = data(ds)
n, L = x_tr.shape[1], 2 ** a
edges = edges_of(n, onebody=(variant == "fc1"))
seeds = list(EQ_SEEDS[ds]) if variant == "eq" else DATASETS[ds][3]
names = {"main": ["global", "agnostic"] + (["cluster"] if a else []), "eq": ["cluster_eq"],
         "fc1": ["cluster" if a else "global"]}[variant]

c0, pi, labels = [], [], []
for s in seeds:
    lab = None
    if a:
        z = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")
        lab = z["eq_y" if variant == "eq" else "train_y"].astype(np.int64)
    cs, ps = make_inits(x_tr, a, [s], edges, labels=lab)          # rows: global, agnostic, cluster
    for nm in names:
        k = {"global": 0, "agnostic": 1, "cluster": 2, "cluster_eq": 2}[nm]
        c0.append(cs[k])
        pi.append(np.full(L, 1.0 / L) if nm == "cluster_eq" else ps[k])
        labels.append((s, nm))
c0, pi = np.stack(c0), np.stack(pi)
evaluate = make_evaluator(n, a, edges, x_te)
curves, per_mbar = {}, {}


def cb(step, params):
    mean, per = evaluate(params, pi)
    curves[step], per_mbar[step] = mean.tolist(), per.tolist()
    print(f"{ds} a={a} {variant} step {step:6d}  test MMD^2 x1e3: " +
          "  ".join(f"{nm} {np.mean([m for (s_, nm_), m in zip(labels, mean) if nm_ == nm]) * 1e3:.4f}" for nm in names),
          flush=True)


t0 = time.perf_counter()
out = train(n, a, edges, x_tr, sigmas(n), cosine_lr(LR, STEPS), STEPS, c0, pi, seeds, eval_steps=EVAL_STEPS, callback=cb)
wall = time.perf_counter() - t0
assert out["plan"] is not None, "the fast engine did not run"
assert np.all(np.isfinite(out["params"])) and np.all(np.isfinite(out["losses"]))
PARAMS.mkdir(parents=True, exist_ok=True)
np.savez(PARAMS / f"{ds}_a{a}_{variant}.npz", params=out["params"], losses=out["losses"].astype(np.float32), pi=pi,
         seeds=np.array([s for s, _ in labels]), names=np.array([nm for _, nm in labels]), edges=edges)
lb = out["losses"].reshape(STEPS // 50, 50, -1).mean(1)          # train loss, mean over blocks of 50 steps
models = [dict(seed=s, init=nm, pi=pi[m].tolist(), test={str(e): curves[e][m] for e in EVAL_STEPS},
               test_per_mbar=per_mbar[STEPS][m], train_loss_50=lb[:, m].tolist())
          for m, (s, nm) in enumerate(labels)]
(RUNS / ds).mkdir(parents=True, exist_ok=True)
json.dump(dict(dataset=ds, a=a, variant=variant, n=n, edges=len(edges), steps=STEPS, lr=f"cosine {LR} -> 0",
               seeds=seeds, plan=out["plan"], wall_s=round(wall, 1), models=models),
          open(RUNS / ds / f"a{a}_{variant}.json", "w"))
print(f"DONE {ds} a={a} {variant}: {len(labels)} models, plan {out['plan']}, {wall:.0f} s", flush=True)
