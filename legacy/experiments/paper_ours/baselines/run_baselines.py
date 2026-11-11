#!/usr/bin/env python3
"""Driver for the classical baselines (RBM all datasets; Deep EBM at n=16).

Usage (from baselines/ or anywhere):
    PO_DS=blobs  python run_baselines.py all          # tune+final RBM (+EBM if n<=16)
    PO_DS=digits python run_baselines.py tune_rbm
    PO_DS=digits python run_baselines.py final_rbm
    python run_baselines.py summary                   # all datasets -> summary.csv

CPU only (JAX_PLATFORMS=cpu, no GPU). Parallel over (config, seed) with a
process pool (<= 16 workers, 20-core box).

== Fairness / grids (sources) ==
Parameter matching to the quantum model at its best ancilla count
(per-branch gates x L): blobs/ising 136x8=1088 -> RBM h=64 (1104 params);
digits 2080x16=33280 -> h=512 (33344); dwave 3484x16=55744 -> h=115 (56259).
Each matched h is bracketed by h/2 and 2h.
RBM training follows Hinton's practical guide (TR 2010-003): PCD, chains =
batch, momentum 0.5->0.9, weight decay 1e-4, W~N(0,0.01), visible bias =
data log-odds. lr grids span the values selected by the recio2025 repo's own
RBM search (best_hyperparameters.yaml: blobs lr=1e-2, ising 1e-4, dwave 1e-5;
search space hyperparameter_settings.py: {1e-5..1e-2}); PCD-k in {1,10}
follows the recio2025 DeepEBM cdiv_steps grid.
EBM (n=16 only): tanh MLP energy (same family as recio2025 DeepEBM /
qml_benchmarks), widths {(64,64),(128,128)}, trained by EXACT full-batch NLL
(negative phase = enumeration of 2^16 states -- the EBM's ideal limit), Adam
lr {1e-3,3e-4,1e-4} (Adam default family, recio2025 EBM lr range 1e-5..1e-3).

== Protocol ==
tune:  hold out VAL_FRAC of train (10%; 20% for the small digits set), train
       the full grid x 5 seeds on the rest, score on validation with the
       paper's MMD^2 protocol (exact channel n<=16 / sample channel n>16,
       see eval_baselines.py). Model selection NEVER touches test.
select: per hidden size (RBM) / width (EBM), best (lr[,k]) by mean-over-sigma
       RANK of the seed-averaged validation MMD^2 (rank-based -> robust to the
       orders-of-magnitude scale differences across the sigma grid).
final: retrain the selected config per h on the FULL train split x 5 seeds,
       evaluate TEST MMD^2; mmd2_mean/std over the 5 training seeds (sampled
       channel: each seed gets its own eval seed from evaluate_sweep.SEEDS).
CSV convention: ancilla=-1, L=0 mark classical rows (see eval_baselines.py).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------- configuration ----------------
TRAIN_SEEDS = [101, 102, 103, 104, 105]
BATCH = 128                # spec; PCD chains = batch (Hinton)
WEIGHT_DECAY = 1e-4        # Hinton TR 2010-003
EBM_STEPS = 2000           # exact full-batch NLL steps (converged; 5000 gives no gain)
VAL_CHAINS = 1000          # sample channel, tuning
VAL_BURNIN = 2000
FINAL_CHAINS = 2000        # = evaluate_sweep true_rows sample cap
FINAL_BURNIN = 20000

# lr grids: calibrated on blobs (lr=1e-2 diverges with momentum 0.9; recio2025's
# sklearn best lr=1e-2 had NO momentum, i.e. ~10x smaller effective step) and
# spanning the recio2025 selected values per scale (ising 1e-4, dwave 1e-5).
# N_UPDATES: convergence-checked on blobs (100k @ lr 3e-4 k10 still improving
# vs 50k); scaled down at larger n for grid feasibility (documented).
RBM_GRID = {
    "blobs":  dict(h=[32, 64, 128],     match=64,  lr=[3e-3, 1e-3, 3e-4, 1e-4], k=[1, 10]),
    "ising":  dict(h=[32, 64, 128],     match=64,  lr=[3e-3, 1e-3, 3e-4, 1e-4], k=[1, 10]),
    "digits": dict(h=[256, 512, 1024],  match=512, lr=[3e-3, 1e-3, 3e-4, 1e-4], k=[1, 10]),
    "dwave":  dict(h=[57, 115, 230],    match=115, lr=[1e-3, 3e-4, 1e-4, 1e-5], k=[1, 10]),
    # mnist (n=784, rowcol_order2): mixture a=4 = 21,952 gates/branch x 16 = 351,232
    # params; RBM match 784*448+784+448 = 352,464 ~ matched at h=448.
    "mnist":  dict(h=[224, 448, 896],   match=448, lr=[1e-3, 3e-4, 1e-4, 1e-5], k=[1, 10]),
    # scalefree (n=100, FC order-2): mixture a=4 = 5,050 gates/branch x 16 = 80,800
    # params; RBM params 101h+100 -> matched at h=800 (80,900). The small h=100/200
    # widths probe the capacity range of the smaller mixtures (a=1-2).
    "scalefree": dict(h=[100, 200, 400, 800, 1600], match=800,
                      lr=[1e-3, 3e-4, 1e-4, 1e-5], k=[1, 10]),
    # archived leaky-MH scale-free data (same n=100 geometry -> same grid)
    "scalefree_leaky": dict(h=[100, 200, 400, 800, 1600], match=800,
                            lr=[1e-3, 3e-4, 1e-4, 1e-5], k=[1, 10]),
    # vq30 (n=30, FC order-2): mixture a=3 = 465 gates/branch x 8 = 3,720 params;
    # RBM params 31h+30 -> matched at h=120 (3,750), bracketed by h/2 and 2h.
    "vq30": dict(h=[60, 120, 240], match=120, lr=[3e-3, 1e-3, 3e-4, 1e-4], k=[1, 10]),
}
N_UPDATES_DS = {"blobs": 100000, "ising": 100000, "digits": 50000, "dwave": 30000, "mnist": 30000,
                "scalefree": 50000, "scalefree_leaky": 50000, "vq30": 100000}
EBM_GRID = dict(widths=[(64, 64), (128, 128)], lr=[1e-3, 3e-4, 1e-4])
QUANTUM_PARAMS = {"blobs": 1088, "ising": 1088, "digits": 33280, "dwave": 55744, "mnist": 351232,
                  "scalefree": 80800, "scalefree_leaky": 80800, "vq30": 3720}
VAL_FRAC = {"blobs": 0.1, "ising": 0.1, "digits": 0.2, "dwave": 0.1, "mnist": 0.1, "scalefree": 0.1,
            "scalefree_leaky": 0.1, "vq30": 0.1}

DATASETS = ["blobs", "ising", "digits", "dwave", "mnist", "scalefree"]
N_WORKERS = int(os.environ.get("BL_WORKERS", "16"))


def _worker_init(threads: int = 1):
    import torch
    torch.set_num_threads(threads)


# ---------------- task functions (run in pool workers) ----------------
def rbm_task(args):
    """Train one RBM (config, seed) and evaluate on the requested split."""
    (h, lr, k, seed, split) = args
    import torch
    torch.set_num_threads(int(os.environ.get("BL_TASK_THREADS", "1")))
    import eval_baselines as EB
    from rbm import RBM
    t0 = time.time()
    xtr_full = EB.load_split("train")
    if split == "val":
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(xtr_full))
        n_val = int(round(VAL_FRAC[EB.C.DS] * len(xtr_full)))
        x_fit, x_eval = xtr_full[perm[n_val:]], xtr_full[perm[:n_val]]
    else:
        x_fit, x_eval = xtr_full, EB.load_split("test")
    rbm = RBM(x_fit.shape[1], h, seed=seed)
    hist = rbm.fit(x_fit, n_updates=N_UPDATES_DS[EB.C.DS], lr=lr, k=k, batch_size=BATCH,
                   weight_decay=WEIGHT_DECAY,
                   val_data=(x_eval if split == "val" else None))
    if EB.EXACT:
        per = EB.mmd2_exact_logp(rbm.exact_logp(), x_eval,
                                 bootstrap=(split == "test"))
        rows = {sk: v for sk, v in per.items()}          # (central, std)
    else:
        if split == "val":
            bits = rbm.sample(VAL_CHAINS, VAL_BURNIN, seed=seed)
            es = seed
        else:
            es = EB.SEEDS[TRAIN_SEEDS.index(seed) % len(EB.SEEDS)]
            bits = rbm.sample(FINAL_CHAINS, FINAL_BURNIN, seed=es)
        per = EB.mmd2_samples(bits, x_eval, seed=es)
        rows = {sk: (v, 0.0) for sk, v in per.items()}
    out = dict(model="RBM", h=h, params=rbm.n_params, lr=lr, k=k, seed=seed,
               split=split, rows=rows, fe_gap=hist[-1].get("fe_gap"),
               recon=hist[-1]["recon_mse"], secs=round(time.time() - t0, 1))
    if split == "test":
        out["state"] = {kk: vv for kk, vv in rbm.state_dict().items()}
    return out


def ebm_task(args):
    (widths, lr, seed, split) = args
    import torch
    torch.set_num_threads(int(os.environ.get("BL_TASK_THREADS", "2")))
    import eval_baselines as EB
    import ebm16 as E
    t0 = time.time()
    xtr_full = EB.load_split("train")
    if split == "val":
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(xtr_full))
        n_val = int(round(VAL_FRAC[EB.C.DS] * len(xtr_full)))
        x_fit, x_eval = xtr_full[perm[n_val:]], xtr_full[perm[:n_val]]
    else:
        x_fit, x_eval = xtr_full, EB.load_split("test")
    model, hist = E.train_exact(x_fit, tuple(widths), lr=lr, n_steps=EBM_STEPS,
                                seed=seed, val_data=(x_eval if split == "val" else None))
    per = EB.mmd2_exact_logp(E.exact_logp(model, x_fit.shape[1]), x_eval,
                             bootstrap=(split == "test"))
    out = dict(model="EBM", h="x".join(map(str, widths)), params=model.n_params,
               lr=lr, k=0, seed=seed, split=split, rows=per,
               nll=hist[-1]["train_nll"], secs=round(time.time() - t0, 1))
    if split == "test":
        out["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        out["widths"] = list(widths)
    return out


# ---------------- orchestration ----------------
def _pool(n, threads=1):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(n, initializer=_worker_init, initargs=(threads,))


def _baseline_outdir(ds):
    """Per-dataset baseline output dir; BL_OUTDIR overrides (e.g. datasets whose
    tree lives outside paper_ours, like vq30)."""
    ov = os.environ.get("BL_OUTDIR")
    return Path(ov) if ov else HERE / ds


def _grid_csv_path(outdir):
    return outdir / "results_baselines.csv"


def _append_rows(path, results):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "h", "params", "lr", "k", "seed",
                                          "split", "pauli", "sigma", "mmd2", "mmd2_std"])
        if new:
            w.writeheader()
        import eval_baselines as EB
        for r in results:
            for sk, (val, std) in sorted(r["rows"].items(), key=lambda kv: EB.SIG2PAULI[kv[0]]):
                w.writerow({"model": r["model"], "h": r["h"], "params": r["params"],
                            "lr": r["lr"], "k": r["k"], "seed": r["seed"], "split": r["split"],
                            "pauli": EB.SIG2PAULI[sk], "sigma": sk,
                            "mmd2": val, "mmd2_std": std})


def select_best(results, keys=("lr", "k")):
    """Per h: best config by mean-over-sigma rank of seed-averaged val MMD^2."""
    import pandas as pd
    rows = []
    for r in results:
        for sk, (val, _) in r["rows"].items():
            rows.append(dict(h=str(r["h"]), lr=r["lr"], k=r["k"], sigma=sk, mmd2=val))
    df = pd.DataFrame(rows)
    agg = df.groupby(["h", "lr", "k", "sigma"], as_index=False)["mmd2"].mean()
    sel = {}
    for h, g in agg.groupby("h"):
        g = g.copy()
        g["rank"] = g.groupby("sigma")["mmd2"].rank()
        score = g.groupby(["lr", "k"])["rank"].mean().sort_values()
        (lr, k) = score.index[0]
        sel[h] = dict(lr=float(lr), k=int(k), rank_score=float(score.iloc[0]))
    return sel


def stage_tune(model_kind):
    import eval_baselines as EB
    ds = EB.C.DS
    outdir = _baseline_outdir(ds)
    outdir.mkdir(parents=True, exist_ok=True)
    if model_kind == "rbm":
        g = RBM_GRID[ds]
        tasks = [(h, lr, k, s, "val") for h in g["h"] for lr in g["lr"]
                 for k in g["k"] for s in TRAIN_SEEDS]
        fn = rbm_task
    else:
        tasks = [(w, lr, s, "val") for w in EBM_GRID["widths"]
                 for lr in EBM_GRID["lr"] for s in TRAIN_SEEDS]
        fn = ebm_task
    print(f"[tune-{model_kind}:{ds}] {len(tasks)} runs, {N_WORKERS} workers")
    t0 = time.time()
    results = []
    with _pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks), 1):
            r.pop("state", None)
            results.append(r)
            print(f"[tune-{model_kind}:{ds}] {i}/{len(tasks)} h={r['h']} lr={r['lr']} "
                  f"k={r['k']} seed={r['seed']} ({r['secs']}s)", flush=True)
    _append_rows(_grid_csv_path(outdir), results)
    sel = select_best(results)
    sp = outdir / f"selected_{model_kind}.json"
    sp.write_text(json.dumps(sel, indent=2))
    print(f"[tune-{model_kind}:{ds}] done in {time.time()-t0:.0f}s -> {sp}: {sel}")


def stage_final(model_kind):
    import torch
    import eval_baselines as EB
    ds = EB.C.DS
    outdir = _baseline_outdir(ds)
    sel = json.loads((outdir / f"selected_{model_kind}.json").read_text())
    if model_kind == "rbm":
        tasks = [(int(h), s["lr"], int(s["k"]), seed, "test")
                 for h, s in sel.items() for seed in TRAIN_SEEDS]
        fn = rbm_task
    else:
        tasks = [(tuple(int(x) for x in w.split("x")), s["lr"], seed, "test")
                 for w, s in sel.items() for seed in TRAIN_SEEDS]
        fn = ebm_task
    threads = int(os.environ.get("BL_TASK_THREADS", "1"))
    nw = min(N_WORKERS, len(tasks))
    print(f"[final-{model_kind}:{ds}] {len(tasks)} runs, {nw} workers x {threads} threads")
    t0 = time.time()
    results = []
    with _pool(nw, threads) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks), 1):
            results.append(r)
            print(f"[final-{model_kind}:{ds}] {i}/{len(tasks)} h={r['h']} seed={r['seed']} "
                  f"({r['secs']}s)", flush=True)
    # save models + grid rows
    fdir = outdir / "final"
    fdir.mkdir(exist_ok=True)
    for r in results:
        if "state" in r:
            torch.save({k: v for k, v in r.items() if k != "rows"},
                       fdir / f"{r['model'].lower()}_h{r['h']}_seed{r['seed']}.pt")
            r.pop("state")
    _append_rows(_grid_csv_path(outdir), results)
    # paper-schema CSV: aggregate over training seeds per scheme
    rows = []
    for h in sorted({str(r["h"]) for r in results}):
        rs = [r for r in results if str(r["h"]) == h]
        scheme = f"{rs[0]['model']}_h{h}"
        per = {}
        for sk in rs[0]["rows"]:
            vals = np.asarray([r["rows"][sk][0] for r in rs])
            per[sk] = (float(vals.mean()), float(vals.std(ddof=1)))
        rows += EB.csv_rows(scheme, per, n_seeds=len(rs))
    path = outdir / "mmd_sigma_sweep_baselines.csv"
    old = []
    if path.exists():
        import pandas as pd
        prev = pd.read_csv(path)
        keep = ~prev["scheme"].str.startswith(f"{results[0]['model']}_")
        old = prev[keep].to_dict("records")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "ancilla", "L", "pauli", "sigma",
                                          "mmd2_mean", "mmd2_std", "n_seeds"])
        w.writeheader()
        for r in old + rows:
            w.writerow(r)
    print(f"[final-{model_kind}:{ds}] wrote {path} ({len(rows)} new rows) "
          f"in {time.time()-t0:.0f}s")
    # mixing diagnostics for the sampled channel (matched-h model, first seed)
    if not EB.EXACT and model_kind == "rbm":
        from rbm import RBM
        match = RBM_GRID[ds]["match"]
        mp_ = fdir / f"rbm_h{match}_seed{TRAIN_SEEDS[0]}.pt"
        if mp_.exists():
            torch.set_num_threads(max(4, threads))
            sd = torch.load(mp_, map_location="cpu")["state"]
            rbm = RBM.from_state(sd)
            diag = EB.mixing_diagnostics(rbm, FINAL_BURNIN)
            xtr = EB.load_split("train").astype(np.float64)
            samp = rbm.sample(EB.N_EVAL, FINAL_BURNIN, seed=555).astype(np.float64)
            diag["model_vs_train"] = {
                "mean_l1": float(np.abs(samp.mean(0) - xtr.mean(0)).mean()),
                "corr_l1": float(np.abs((samp.T @ samp) / len(samp)
                                        - (xtr.T @ xtr) / len(xtr)).mean())}
            (outdir / "mixing_diagnostics.json").write_text(json.dumps(diag, indent=2))
            print(f"[final-{model_kind}:{ds}] mixing diagnostics: {diag}")


def stage_summary():
    import pandas as pd
    out = []
    for ds in DATASETS:
        dd = HERE / ds
        p = dd / "mmd_sigma_sweep_baselines.csv"
        if not p.exists():
            print(f"[summary] missing {p}, skipping {ds}")
            continue
        df = pd.read_csv(p)
        sels = {}
        for kind in ("rbm", "ebm"):
            sp = dd / f"selected_{kind}.json"
            if sp.exists():
                sels[kind.upper()] = json.loads(sp.read_text())
        for _, r in df.iterrows():
            model, h = r["scheme"].split("_h")
            sel = sels.get(model, {}).get(str(h), {})
            match = (str(RBM_GRID[ds]["match"]) == str(h)) if model == "RBM" else (h == "64x64")
            out.append(dict(dataset=ds, model=model, scheme=r["scheme"], h=h,
                            matched=int(match), quantum_params=QUANTUM_PARAMS[ds],
                            lr=sel.get("lr"), k=sel.get("k"),
                            pauli=r["pauli"], sigma=r["sigma"],
                            mmd2_mean=r["mmd2_mean"], mmd2_std=r["mmd2_std"],
                            n_seeds=r["n_seeds"]))
    pd.DataFrame(out).to_csv(HERE / "summary.csv", index=False)
    print(f"[summary] wrote {HERE/'summary.csv'} ({len(out)} rows)")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "summary":
        stage_summary()
        return
    import eval_baselines as EB
    ds = EB.C.DS
    stages = ([stage] if stage != "all" else
              ["tune_rbm", "final_rbm"] + (["tune_ebm", "final_ebm"] if EB.EXACT else []))
    for st in stages:
        kind = st.split("_")[1]
        if kind == "ebm" and not EB.EXACT:
            print(f"[skip] EBM only at n<=16 (ds={ds})")
            continue
        (stage_tune if st.startswith("tune") else stage_final)(kind)


if __name__ == "__main__":
    main()
