"""Trainable branch weights: the paper's main campaign (global / agnostic / cluster x every training seed, every
ancilla count a >= 1) rerun with the weights trained jointly with the angles, pi = softmax(beta), beta initialized to
log of the fixed weights of the campaign (1/L for global and agnostic, the cluster masses for cluster), Adam on beta
at the angles' cosine schedule (iqp_fastcf.train(train_pi=True)). Everything else (starts, noise, subset streams,
steps, evaluator) is the campaign of ../cosine_rerun, so every model pairs with its fixed-weight counterpart in
../cosine_rerun/runs/<ds>/a<a>_main.json by (seed, init).
usage: train_tw.py <ds> <a>  ->  runs/<ds>/a<a>_tw.json, params/<ds>_a<a>_tw.npz (not committed)
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cosine_rerun"))
from common import CLUSTERS, DATASETS, EVAL_STEPS, LR, STEPS, data, edges_of, sigmas   # noqa: E402
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits                   # noqa: E402

RUNS, PARAMS = HERE / "runs", HERE / "params"
ds, a = sys.argv[1], int(sys.argv[2])
assert a >= 1 and a in DATASETS[ds][2]
x_tr, x_te = data(ds)
n, L = x_tr.shape[1], 2 ** a
edges = edges_of(n)
seeds = DATASETS[ds][3]
names = ["global", "agnostic", "cluster"]

c0, pi0, labels = [], [], []
for s in seeds:
    lab = np.load(CLUSTERS / ds / f"a{a}_seed{s}.npz")["train_y"].astype(np.int64)
    cs, ps = make_inits(x_tr, a, [s], edges, labels=lab)          # rows: global, agnostic, cluster
    for k, nm in enumerate(names):
        c0.append(cs[k]); pi0.append(ps[k]); labels.append((s, nm))
c0, pi0 = np.stack(c0), np.stack(pi0)
evaluate = make_evaluator(n, a, edges, x_te)
curves, per_mbar, pis = {}, {}, {}


def cb(step, params, pi):
    mean, per = evaluate(params, pi)
    curves[step], per_mbar[step], pis[step] = mean.tolist(), per.tolist(), pi.tolist()
    print(f"{ds} a={a} tw step {step:6d}  test MMD^2 x1e3: " +
          "  ".join(f"{nm} {np.mean([m for (s_, nm_), m in zip(labels, mean) if nm_ == nm]) * 1e3:.4f}" for nm in names),
          flush=True)


t0 = time.perf_counter()
out = train(n, a, edges, x_tr, sigmas(n), cosine_lr(LR, STEPS), STEPS, c0, pi0, seeds, eval_steps=EVAL_STEPS,
            callback=cb, train_pi=True)
wall = time.perf_counter() - t0
assert out["plan"] is not None, "the fast engine did not run"
assert np.all(np.isfinite(out["params"])) and np.all(np.isfinite(out["losses"])) and np.all(np.isfinite(out["pi"]))
assert np.allclose(out["pi"].sum(1), 1.0)
PARAMS.mkdir(parents=True, exist_ok=True)
np.savez(PARAMS / f"{ds}_a{a}_tw.npz", params=out["params"], losses=out["losses"].astype(np.float32), pi=out["pi"],
         pi0=pi0, seeds=np.array([s for s, _ in labels]), names=np.array([nm for _, nm in labels]), edges=edges)
lb = out["losses"].reshape(STEPS // 50, 50, -1).mean(1)
models = [dict(seed=s, init=nm, pi0=pi0[m].tolist(), pi=out["pi"][m].tolist(),
               pi_curve={str(e): pis[e][m] for e in EVAL_STEPS}, test={str(e): curves[e][m] for e in EVAL_STEPS},
               test_per_mbar=per_mbar[STEPS][m], train_loss_50=lb[:, m].tolist())
          for m, (s, nm) in enumerate(labels)]
(RUNS / ds).mkdir(parents=True, exist_ok=True)
json.dump(dict(dataset=ds, a=a, variant="tw", n=n, edges=len(edges), steps=STEPS, lr=f"cosine {LR} -> 0",
               pi_lr="same schedule as the angles", seeds=seeds, plan=out["plan"], wall_s=round(wall, 1),
               models=models), open(RUNS / ds / f"a{a}_tw.json", "w"))
print(f"DONE {ds} a={a} tw: {len(labels)} models, plan {out['plan']}, {wall:.0f} s", flush=True)
