#!/usr/bin/env python3
"""Every number the mode-separation rewrite of the manuscript cites (docs/2026-11-02-modesep-manuscript-contract.md).

usage: numbers.py build | check
build : results/numbers.json from committed sources:
        - table: seed means (and sd) of experiments/modesep_bound/results/summary.json (a=0 circuit, assembled partition
          mixture, trained cluster-init and global mixtures; test and train MMD^2; the bound B^2 and its ratio);
        - diagnostics: seed means of experiments/learned_partition and experiments/branch_z_start summaries;
        - fields: one-body field magnitudes per dataset at its analysis size, over all seeds:
          spectral cluster means mu_l = 1 - 2 mean(x | cluster l) (experiments/cosine_rerun/clusters),
          trained branch one-body correlators Z_l = z1(theta(P, a)) of the cosine rerun's global and cluster-init
          mixtures, and the data's one-body means; mean |.| over branches and qubits, and the between-branch spread
          (rms over qubits of z_l - z_l', averaged over branch pairs).
check : recomputes everything and compares with results/numbers.json (exact).
"""
import sys
sys.path.pop(0)            # this file is named numbers.py: keep its folder from shadowing the stdlib module numpy imports
import json
from itertools import combinations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiments"
sys.path[:0] = [str(EXP / "cosine_rerun"), str(EXP / "modesep_bound")]
from common import CLUSTERS, DATASETS, data, theta, z1    # noqa: E402
import analyze as AN                                       # noqa: E402  (params_of)
from modesep_bound import A_OF                             # noqa: E402


def spread(V):
    """Mean over branch pairs of the rms (over qubits) difference, V (L, n)."""
    return float(np.mean([np.sqrt(np.mean((V[i] - V[j]) ** 2)) for i, j in combinations(range(len(V)), 2)]))


def fields(ds):
    a = A_OF[ds]
    x_tr, _ = data(ds)
    P, pi, seeds, names, edges = AN.params_of(ds, a, "main")
    n = x_tr.shape[1]
    mu, zg, zc = [], [], []
    for m, (s, nm) in enumerate(zip(seeds, names)):
        Z = z1(theta(P[m], a), n, edges)
        if nm == "global":
            zg.append(Z)
        elif nm == "cluster":
            zc.append(Z)
            lab = np.load(CLUSTERS / ds / f"a{a}_seed{int(s)}.npz")["train_y"].astype(np.int64)
            mu.append(np.stack([1.0 - 2.0 * x_tr[lab == l].mean(0) for l in range(1 << a)]))
    return dict(mu_abs_mean=float(np.mean(np.abs(mu))), z_global_abs_mean=float(np.mean(np.abs(zg))),
                z_cluster_abs_mean=float(np.mean(np.abs(zc))), data_abs_mean=float(np.mean(np.abs(1.0 - 2.0 * x_tr.mean(0)))),
                mu_spread=float(np.mean([spread(v) for v in mu])), z_global_spread=float(np.mean([spread(v) for v in zg])),
                z_cluster_spread=float(np.mean([spread(v) for v in zc])), seeds=len(mu))


def collect():
    mb = json.loads((EXP / "modesep_bound" / "results" / "summary.json").read_text())
    lp = json.loads((EXP / "learned_partition" / "results" / "summary.json").read_text())
    bz = json.loads((EXP / "branch_z_start" / "results" / "summary.json").read_text())
    out = dict(table={}, diagnostics={}, fields={})
    for ds in A_OF:
        out["table"][ds] = dict(a=A_OF[ds], seeds=len(DATASETS[ds][3]),
                                **{k: mb[ds][k] for k in ("whole_test", "partition_test", "cluster_test", "global_test",
                                                          "whole_train", "partition_train", "bound2", "ratio")})
        out["diagnostics"][ds] = dict(learned_start=lp[ds]["learned_start"], branch_z_start=bz[ds]["bz"],
                                      branch_z_step0=bz[ds]["bz0"], global_start=bz[ds]["glob"],
                                      cluster_start=bz[ds]["cluster"],
                                      verdicts_learned=lp["verdicts"][ds], verdicts_branch_z=bz["verdicts"][ds])
        out["fields"][ds] = fields(ds)
        t = out["table"][ds]
        out["table"][ds]["factor_single_over_partition_test"] = t["whole_test"]["mean"] / t["partition_test"]["mean"]
        rows = {r["seed"]: r for r in json.loads((EXP / "modesep_bound" / "runs" / f"{ds}.json").read_text())["rows"]}
        lps = {m["seed"]: m for m in json.loads((EXP / "learned_partition" / "runs" / f"{ds}_train.json").read_text())["models"]}
        bzs = {m["seed"]: m for m in json.loads((EXP / "branch_z_start" / "runs" / f"{ds}_train.json").read_text())["models"]}
        out.setdefault("per_seed", {})[ds] = [dict(
            seed=s, partition_test=rows[s]["partition_test"], partition_train=rows[s]["partition_train"],
            bound2=rows[s]["bound2"], whole_train=rows[s]["whole_train"], ratio=rows[s]["bound2"] / rows[s]["whole_train"],
            learned_start=lps[s]["test"]["15000"], branch_z_start=bzs[s]["test"]["15000"]) for s in DATASETS[ds][3]]
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    f = HERE / "results" / "numbers.json"
    if a == ["build"]:
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(collect(), indent=1) + "\n")
        print(f.read_text())
    elif a == ["check"]:
        same = f.exists() and json.loads(f.read_text()) == json.loads(json.dumps(collect()))
        print(("ok   " if same else "FAIL ") + "results/numbers.json equals a fresh recomputation from the committed sources")
        sys.exit(0 if same else 1)
    else:
        sys.exit(__doc__)
