"""Trainable against fixed branch weights, paired by (seed, init): final test MMD^2 (mean over m-bar = 1..6) of
runs/<ds>/a<a>_tw.json against ../cosine_rerun/runs/<ds>/a<a>_main.json, the effective number of branches
exp(H(pi)) of the trained weights, their distance from the start, and (cluster) the drift from the cluster masses.
usage: compare.py  ->  results/compare.csv, results/compare.md
"""
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CR = HERE.parent / "cosine_rerun" / "runs"
RES = HERE / "results"
DS = {"blobs": (1, 2, 3), "ising": (1, 2, 3), "dwave": (1, 2, 3), "mnist": (1, 2, 3, 4)}


def neff(p):
    p = np.asarray(p)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    rows = []
    for ds, aa in DS.items():
        for a in aa:
            f = HERE / "runs" / ds / f"a{a}_tw.json"
            if not f.exists():
                continue
            tw = {(m["seed"], m["init"]): m for m in json.load(open(f))["models"]}
            fx = {(m["seed"], m["init"]): m for m in json.load(open(CR / ds / f"a{a}_main.json"))["models"]}
            for k, m in tw.items():
                g = fx[k]
                rows.append(dict(dataset=ds, a=a, seed=k[0], init=k[1], fixed=float(np.mean(g["test_per_mbar"])),
                                 trained=float(np.mean(m["test_per_mbar"])), neff0=neff(m["pi0"]), neff=neff(m["pi"]),
                                 pi_min=float(min(m["pi"])), pi_max=float(max(m["pi"])),
                                 tv_from_start=float(0.5 * np.abs(np.subtract(m["pi"], m["pi0"])).sum())))
    RES.mkdir(exist_ok=True)
    with open(RES / "compare.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines = ["| dataset | a | init | fixed x1e3 | trained x1e3 | ratio (min-max) | trained lower | N_eff start -> end | TV(pi, pi0) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds, aa in DS.items():
        for a in aa:
            for init in ("global", "agnostic", "cluster"):
                g = [r for r in rows if (r["dataset"], r["a"], r["init"]) == (ds, a, init)]
                if not g:
                    continue
                ra = np.array([r["trained"] / r["fixed"] for r in g])
                lines.append(f"| {ds} | {a} | {init} | {np.mean([r['fixed'] for r in g]) * 1e3:.4f} | "
                             f"{np.mean([r['trained'] for r in g]) * 1e3:.4f} | {ra.mean():.3f} ({ra.min():.3f}-{ra.max():.3f}) | "
                             f"{int((ra < 1).sum())}/{len(g)} | {np.mean([r['neff0'] for r in g]):.2f} -> "
                             f"{np.mean([r['neff'] for r in g]):.2f} | {np.mean([r['tv_from_start'] for r in g]):.3f} |")
    (RES / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
