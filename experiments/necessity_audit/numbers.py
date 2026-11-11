#!/usr/bin/env python3
"""What the branch necessity Lambda shows beyond its own loss (docs/2026-11-05-necessity-wording-contract.md).

usage: numbers.py build | check
build : results/numbers.json from experiments/cosine_rerun/results/branch_necessity.csv (untracked, made by another
        session; its sha256 is pinned here). Per dataset, over the models of the paper's diversity figure:
        - Spearman correlations with the test MMD^2 of Lambda, of D^2_branch and of the bound
          U = 1 + D^2 / ((L - 1)^2 MMD^2), and the rank agreement of Lambda with U. At uniform weights the pi-weighted
          mean of Delta_l is D^2 / (L - 1)^2, so Jensen gives Lambda <= U (checked for the uniform-weight schemes);
        - the same correlations without the failed MNIST data-agnostic models;
        - per scheme: test MMD^2, Lambda and U ranges, and the number and weight of harmful branches (Delta_l < 0).
check : recomputes everything and compares with results/numbers.json (exact), including the CSV hash.
"""
import sys
sys.path.pop(0)            # this file is named numbers.py: keep its folder from shadowing the stdlib module numpy imports
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[1] / "experiments" / "cosine_rerun" / "results" / "branch_necessity.csv"
UNIFORM = ("global", "agnostic")


def rho(x, y):
    return float(spearmanr(x, y).correlation)


def collect():
    raw = CSV.read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    out = dict(csv=dict(path=str(CSV.relative_to(HERE.parents[1])), sha256=hashlib.sha256(raw).hexdigest()))
    for ds in ("blobs", "ising", "dwave", "mnist"):
        R = [r for r in rows if r["dataset"] == ds]
        L = 2 ** int(R[0]["a"])
        mmd = np.array([float(r["mmd2"]) for r in R])
        lam = np.array([float(r["Lambda"]) for r in R])
        D2 = np.array([float(r["D2"]) for r in R])
        U = 1 + D2 / ((L - 1) ** 2 * mmd)
        uni = np.array([r["init"] in UNIFORM for r in R])
        keep = np.array([not (ds == "mnist" and r["init"] == "agnostic") for r in R])
        d = dict(models=len(R), L=L, rho_lambda=rho(lam, mmd), rho_D2=rho(D2, mmd), rho_U=rho(U, mmd),
                 rho_lambda_U=rho(lam, U), bound_uniform=f"{int((lam[uni] <= U[uni] * (1 + 1e-9)).sum())}/{int(uni.sum())}",
                 rho_lambda_kept=rho(lam[keep], mmd[keep]), rho_D2_kept=rho(D2[keep], mmd[keep]), schemes={})
        for ini in ("global", "agnostic", "cluster"):
            m = np.array([r["init"] == ini for r in R])
            nh = [int(r["n_harmful"]) for r, k in zip(R, m) if k]
            wh = [float(r["w_harmful"]) for r, k in zip(R, m) if k]
            d["schemes"][ini] = dict(seeds=int(m.sum()), mmd2_mean=float(mmd[m].mean()),
                                     lambda_min=float(lam[m].min()), lambda_max=float(lam[m].max()),
                                     U_min=float(U[m].min()), U_max=float(U[m].max()),
                                     harmful_min=min(nh), harmful_max=max(nh), w_harmful_min=min(wh), w_harmful_max=max(wh))
        out[ds] = d
    s = out["mnist"]["schemes"]
    out["mnist"]["failed_over_others"] = s["agnostic"]["mmd2_mean"] / np.mean([s["global"]["mmd2_mean"], s["cluster"]["mmd2_mean"]])
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
        print(("ok   " if same else "FAIL ") + "results/numbers.json equals a fresh recomputation from branch_necessity.csv "
              "(sha256 pinned)")
        sys.exit(0 if same else 1)
    else:
        sys.exit(__doc__)
