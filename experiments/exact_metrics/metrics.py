"""Exact likelihood-based metrics of the trained n = 16 models, independent of the MMD used for training.

For every trained model of the campaign (../cosine_rerun: main, fc1 = one-body mixture, eq = equal-weight control)
and of the trainable-weight campaign (../trainable_weights: tw), the exact model distribution
p = sum_l pi_l p_l (Born probabilities of every branch, common.born_probs16) is compared with the exact generating
distribution of targets.py:
  KL = KL(p_target || p_model) in nats (the expected log-loss excess on fresh target samples),
  TV = total variation distance,
  NLL = mean negative log-likelihood of the test split (nats; its value for the target itself is the reference).
Output: results/exact_metrics.json (one record per model) and results/exact_metrics.md (seed means and ranges).
usage: metrics.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CR, TW = HERE.parent / "cosine_rerun", HERE.parent / "trainable_weights"
sys.path.insert(0, str(CR))
from common import data, theta, born_probs16                                          # noqa: E402
from targets import TARGETS, N                                                        # noqa: E402

RES = HERE / "results"


def model_probs(params, pi, a, edges):
    th = theta(params, a)                                    # (L, G)
    return sum(w * born_probs16(t, N, edges) for w, t in zip(pi, th) if w > 0)


def metrics(p, q, idx_te):
    q = np.maximum(q, 1e-300)
    m = p > 0
    return dict(KL=float(np.sum(p[m] * np.log(p[m] / q[m]))), TV=float(0.5 * np.abs(p - q).sum()),
                NLL=float(-np.mean(np.log(q[idx_te]))))


def files():
    for ds in ("blobs", "ising"):
        for a in range(4):
            for var in ("main", "fc1", "eq"):
                f = CR / "params" / f"{ds}_a{a}_{var}.npz"
                if f.exists():
                    yield ds, a, var, f
            f = TW / "params" / f"{ds}_a{a}_tw.npz"
            if f.exists():
                yield ds, a, "tw", f


def main():
    recs, ref = [], {}
    for ds in ("blobs", "ising"):
        p = TARGETS[ds]()
        _, x_te = data(ds)
        idx = (x_te.astype(np.int64) << np.arange(N)).sum(1)
        ref[ds] = dict(NLL_target=float(-np.mean(np.log(p[idx]))),
                       entropy=float(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    cache = {ds: TARGETS[ds]() for ds in ("blobs", "ising")}
    idxs = {ds: (data(ds)[1].astype(np.int64) << np.arange(N)).sum(1) for ds in ("blobs", "ising")}
    for ds, a, var, f in files():
        z = np.load(f)
        edges = z["edges"]
        for m in range(z["params"].shape[0]):
            q = model_probs(z["params"][m], z["pi"][m], a, edges)
            r = dict(dataset=ds, a=a, variant=var, init=str(z["names"][m]), seed=int(z["seeds"][m]),
                     **metrics(cache[ds], q, idxs[ds]))
            recs.append(r)
        print(f"{ds} a={a} {var}: {z['params'].shape[0]} models", flush=True)
    RES.mkdir(exist_ok=True)
    json.dump(dict(reference=ref, models=recs), open(RES / "exact_metrics.json", "w"), indent=0)
    # table: seed mean (min-max) per (dataset, a, variant, init)
    lines = ["| dataset | a | variant | init | seeds | KL (nats) | TV | test NLL (nats) |", "|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["dataset"], r["a"], r["variant"], r["init"]) for r in recs})
    for k in keys:
        g = [r for r in recs if (r["dataset"], r["a"], r["variant"], r["init"]) == k]
        cell = lambda q: f"{np.mean([r[q] for r in g]):.4f} ({min(r[q] for r in g):.4f}-{max(r[q] for r in g):.4f})"
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(g)} | {cell('KL')} | {cell('TV')} | {cell('NLL')} |")
    for ds in ref:
        lines.append(f"\n{ds}: target entropy {ref[ds]['entropy']:.4f} nats, test NLL under the target "
                     f"{ref[ds]['NLL_target']:.4f} nats")
    (RES / "exact_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
