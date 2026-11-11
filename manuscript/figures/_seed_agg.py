#!/usr/bin/env python3
r"""Seed aggregation of the paper's training results, read from the cosine rerun (experiments/cosine_rerun).

The rerun (2026-10-19) trains every model with one schedule (Adam, lr 1e-3 cosine to 0 over 15,000 steps,
closed-form MMD^2 at m-bar = 2, 6), all pairs as edges, the paper's seeds; see experiments/README.md. Each model's
test MMD^2 is evaluated at m-bar = 1..6 (runs/<ds>/a<a>_<variant>.json, `test_per_mbar`); the plotted value is the
mean over training seeds and the band the sample standard deviation over them (ddof = 1).

Scheme keys keep the earlier CSV convention: S1_visible = global, S4 = data-agnostic, S2_cluster = cluster
initialization, True = train-test sampling reference, product = the trained one-body mixture (variant fc1).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RR = REPO / "experiments" / "cosine_rerun"
RUNS, RESULTS = RR / "runs", RR / "results"
FULLDATA: dict = {}                 # the earlier full-data special cases no longer exist
FULLDATA_OUT = RESULTS

SEEDS = [7, 1234, 2025]
SEEDS_OF = {
    "blobs": [7, 11, 23, 42, 99, 314, 777, 1234, 2025, 2718],
    "ising": [7, 11, 23, 42, 99, 314, 777, 1234, 2025, 2718],
    "mnist": [7, 11, 23, 1234, 2025],
    "dwave": [7, 11, 23, 1234, 2025],
}
REPRESENTATIVE_SEED = 1234
N_OF = {"blobs": 16, "ising": 16, "mnist": 784, "dwave": 484}
LADDER = {"blobs": [0, 1, 2, 3], "ising": [0, 1, 2, 3], "mnist": [0, 1, 2, 3, 4], "dwave": [0, 1, 2, 3]}
SCHEME_OF = {"global": "S1_visible", "agnostic": "S4", "cluster": "S2_cluster"}
STEM_OF = {"global": "s1_visible", "agnostic": "s4_unbiased", "cluster": "s2_cluster"}
LOSS_STRIDE = 50                    # runs store the training loss averaged over blocks of 50 steps


def seeds_of(ds: str):
    return SEEDS_OF.get(ds, SEEDS)


def sigma_of(n: int, mbar: int) -> float:
    """Bandwidth whose Bernoulli subsets have mean size m-bar: p = (1 - e^{-1/2 s^2}) / 2 = m-bar / n."""
    return 1.0 / math.sqrt(-2.0 * math.log(1.0 - 2.0 * mbar / n))


def mbar_of(n: int, sigma: float) -> int:
    return int(round(n * (1.0 - math.exp(-1.0 / (2.0 * sigma ** 2))) / 2.0))


def run(ds: str, a: int, variant: str = "main"):
    p = RUNS / ds / f"a{a}_{variant}.json"
    return json.load(open(p)) if p.exists() else None


def refs(ds: str):
    """{m-bar: train-test sampling reference} from results/refs.csv."""
    return {int(r["mbar"]): float(r["ref_train_vs_test"]) for r in csv.DictReader(open(RESULTS / "refs.csv"))
            if r["dataset"] == ds}


def _models(ds: str):
    """(scheme key, a, seed, per-m-bar test values) of the main and one-body runs."""
    for a in LADDER[ds]:
        for variant in ("main", "fc1"):
            r = run(ds, a, variant)
            if r is None:
                continue
            for m in r["models"]:
                key = "product" if variant == "fc1" else SCHEME_OF[m["init"]]
                yield key, a, m["seed"], m["test_per_mbar"]


def aggregate_sweep(ds: str, rename: dict | None = None):
    """(sweep, L_of, sig_ticks) with sweep[scheme][ancilla] = (sigma_list, seed_mean_list, seed_std_list) over
    m-bar = 1..6; "True" holds the train-test reference at every ancilla (no seed spread)."""
    n = N_OF[ds]
    acc = defaultdict(list)
    for key, a, _s, per in _models(ds):
        acc[(key, a)].append(per)
    rename = rename or {}
    sweep = defaultdict(lambda: defaultdict(lambda: ([], [], [])))
    sig = [sigma_of(n, mb) for mb in range(1, 7)]
    for (key, a), v in acc.items():
        v = np.asarray(v, dtype=float)
        sweep[rename.get(key, key)][a] = (sig, v.mean(0).tolist(),
                                          (v.std(0, ddof=1) if len(v) > 1 else np.zeros(6)).tolist())
    ref = refs(ds)
    for a in LADDER[ds]:
        sweep["True"][a] = (sig, [ref[mb] for mb in range(1, 7)], [0.0] * 6)
    return sweep, {a: 2 ** a for a in LADDER[ds]}, sig


def final_means(ds: str):
    """{(scheme, a): (mean, sd)} over seeds of the final test MMD^2 averaged over m-bar = 1..6."""
    acc = defaultdict(list)
    for key, a, _s, per in _models(ds):
        acc[(key, a)].append(float(np.mean(per)))
    out = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in acc.items()}
    ref = float(np.mean(list(refs(ds).values())))
    for a in LADDER[ds]:
        out[("True", a)] = (ref, 0.0)
    return out


def aggregate_losses(ds: str, scheme_key: str, a: int):
    """Seed mean and seed std of the training loss (block means over LOSS_STRIDE steps), or None."""
    r = run(ds, a, "main")
    if r is None:
        return None
    init = {v: k for k, v in SCHEME_OF.items()}.get(scheme_key)
    M = np.array([m["train_loss_50"] for m in r["models"] if m["init"] == init], dtype=float)
    if not len(M):
        return None
    return M.mean(0), (M.std(0, ddof=1) if len(M) > 1 else np.zeros(M.shape[1]))


def em_curve(ds: str):
    """(ancillas, mean test MMD^2) of the EM-fitted product mixture: n = 16 from the legacy Bernoulli baseline
    (same data and labels; exact evaluation), MNIST / D-Wave from results/product_baselines_<ds>.csv."""
    if ds in ("mnist", "dwave"):
        rows = [r for r in csv.DictReader(open(RESULTS / f"product_baselines_{ds}.csv")) if r["method"] == "em"]
        a = sorted({int(r["a"]) for r in rows})
        return np.array(a), np.array([np.mean([float(r["test_mmd2"]) for r in rows if int(r["a"]) == x]) for x in a])
    p = REPO / "legacy" / "experiments" / "paper_ours" / "referee_2026_10" / "bernoulli_baseline" / "summary.csv"
    rows = sorted((r for r in csv.DictReader(open(p)) if r["dataset"] == ds and r["column"] == "em"),
                  key=lambda r: int(r["ancilla"]))
    return np.array([int(r["ancilla"]) for r in rows]), np.array([float(r["mean"]) for r in rows])


def scatter_rows(ds: str):
    """[(init, seed, D2, test MMD^2 at the D2 bandwidth, Lambda)] of the models of Fig. diversity (Lambda: branch
    necessity, results/branch_necessity.csv)."""
    mode = {(r["init"], r["seed"]): float(r["Lambda"]) for r in csv.DictReader(open(RESULTS / "branch_necessity.csv"))
            if r["dataset"] == ds}
    return [(r["init"], int(r["seed"]), float(r["D2"]), float(r["mmd2"]), mode[(r["init"], r["seed"])])
            for r in csv.DictReader(open(RESULTS / "branch_diversity.csv")) if r["dataset"] == ds]


def lowest_d2_seed(ds: str, init: str) -> int:
    return min((r for r in scatter_rows(ds) if r[0] == init), key=lambda r: r[2])[1]
