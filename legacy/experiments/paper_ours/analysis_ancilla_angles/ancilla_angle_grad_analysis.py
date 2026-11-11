#!/usr/bin/env python3
"""Ancilla-sector angle magnitudes + step-0 gradients for the Fig. 4 runs.

Analyzes the SAME runs fig_results_all.pdf plots: the 3-seed campaign under
legacy/experiments/paper_ours/<ds>_seed/ (schemes S1_visible=global,
S4=agnostic, S2_cluster=cluster; seeds 7/1234/2025; mnist S4 = st20k long run).

Phase "angles" (CPU):
  - loads each trained qcbm_artifact.pt, splits compiled gate angles into the
    visible sector (support on data qubits only) and ancilla sector (support
    touches >=1 ancilla wire), computes mean|theta| and RMS per sector;
  - reconstructs the run's exact init point through the trainer's own code path
    (RandomizedIqpTrainer._initial_params with the run's config yaml, incl. the
    seeded uniform noise draw) and reports the same stats at init and for the
    trained-minus-init change;
  - joins the per-seed test MMD^2 (mean over the mean-Pauli m-bar=1..6 sweep,
    identical convention to make_fig_results_all.seed_sweep_means).

Phase "grad" (GPU): rebuilds the training loss exactly as train_stage does
  (same loss_fn, payloads, batching, float32) and evaluates the value+gradient
  at the reconstructed init with the EXACT step-0 training PRNG key
  (random_state = seed + a*1000 + block_index0; key1=split(key)[0];
  step0 key = split(key1)[0]) plus 2 alternate estimator keys.
  Validation: the loss at the step-0 key must reproduce schedule[0].loss_first
  stored in the trained artifact (bit-identical init check).

Phase "aggregate": joins everything, writes tidy CSVs + report tables.

Read-only on all experiment artifacts; writes only into analysis_ancilla_angles/.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

EXP = Path(__file__).resolve().parents[1]
SHARED = EXP.parent / "randomized_iqp_mnist8x8"
VENDOR = SHARED / "vendor"
SWEEP_SCRIPTS = EXP.parent / "cluster_count_mmd_sweep" / "scripts"
for _p in (str(SWEEP_SCRIPTS), str(SHARED), str(VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 1234, 2025]
PANELS = {"blobs": [1, 2, 3], "ising": [1, 2, 3], "dwave": [1, 2, 3],
          "mnist": [1, 2, 3, 4]}
SCHEMES = [("S1_visible", "s1_visible"), ("S4", "s4_unbiased"),
           ("S2_cluster", "s2_cluster")]
LAYOUT = {"mnist": "rowcol_order2"}
SWEEP_FILES = ("mmd_sigma_sweep.csv", "mmd_sigma_sweep_s4.csv",
               "mmd_sigma_sweep_s4_st20k.csv", "mmd_sigma_sweep_s2_label.csv")


def layout_of(ds):
    return LAYOUT.get(ds, "fully_connected_order2")


def cfg_path(ds, stem, a, seed):
    infix = "_st20k" if (ds == "mnist" and stem == "s4_unbiased") else ""
    return EXP / f"{ds}_seed" / "configs" / f"{stem}_a{a}{infix}_usqm_seed{seed}.yaml"


def run_ckpt(ds, stem, a, seed):
    infix = "_st20k" if (ds == "mnist" and stem == "s4_unbiased") else ""
    return (EXP / f"{ds}_seed" / f"outputs_{stem}_a{a}{infix}_usqm_seed{seed}"
            / "checkpoints" / layout_of(ds) / f"a{a}")


def per_seed_mmd(ds):
    """{(scheme_key, a, seed): mean over the m-bar=1..6 sweep of mmd2_mean}.
    Mirrors make_fig_results_all.seed_sweep_means but keeps per-seed values,
    including the LONG_RUN fold (mnist S4_st20k -> S4)."""
    out = {}
    for N in SEEDS:
        rdir = EXP / f"{ds}_seed" / f"results_usqm_seed{N}"
        seen, acc = set(), {}
        for fname in SWEEP_FILES:
            fp = rdir / fname
            if not fp.exists():
                continue
            for r in csv.DictReader(open(fp)):
                key = (r["scheme"], int(r["ancilla"]), int(r["pauli"]))
                if key in seen:
                    continue
                seen.add(key)
                acc.setdefault((r["scheme"], int(r["ancilla"])), []).append(
                    float(r["mmd2_mean"]))
        if ds == "mnist":  # long run supersedes the shared budget (a>=1)
            for (scheme, a) in [k for k in acc if k[0] == "S4_st20k"]:
                acc[("S4", a)] = acc.pop(("S4_st20k", a))
        for (scheme, a), vals in acc.items():
            out[(scheme, a, N)] = float(np.mean(vals))
    return out


def sector_mask(gates, visible_bits):
    """True where the gate's support touches at least one ancilla wire."""
    return np.array([any(w >= visible_bits for gen in g for w in gen)
                     for g in gates], dtype=bool)


def sec_stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(v))), float(np.sqrt(np.mean(v * v)))


TOPQS = [1, 5, 10, 25]


def topq_of(v):
    """{q: mean of top q% of |v| (>=1 element), 'max': max|v|}."""
    out = {}
    if v.size == 0:
        return {q: float("nan") for q in TOPQS} | {"max": float("nan")}
    s = np.sort(np.abs(np.asarray(v, dtype=np.float64)))[::-1]
    for q in TOPQS:
        k = max(1, math.ceil(v.size * q / 100.0))
        out[q] = float(s[:k].mean())
    out["max"] = float(s[0])
    return out


def build_run(ds, scheme_key, stem, a, seed):
    """Rebuild (trainer, gates, wires, init params) through the trainer's own
    code path; returns everything needed by both phases."""
    from randomized_iqp import RandomizedIqpTrainer, load_experiment_config
    from randomized_iqp.gates import build_visible_gates
    from randomized_iqp.ancilla import augment_visible_gates_with_ancilla
    from iqpopt.utils import initialize_from_data

    cfg = cfg_path(ds, stem, a, seed)
    experiment = load_experiment_config(cfg)
    trainer = RandomizedIqpTrainer(experiment)
    layout_name = layout_of(ds)
    layout = trainer.config["layouts"][layout_name]
    visible_gates = build_visible_gates(layout_name, layout, trainer.visible_bits)
    gates, visible_wires, ancilla_wires = augment_visible_gates_with_ancilla(
        visible_gates, visible_bits=trainer.visible_bits, ancilla_bits=a,
        policy=str(trainer.qcbm.get("ancilla_gate_policy", "all_subsets")))

    previous_gates = previous_params = None
    init_art = trainer._load_init_artifact(layout_name)
    init_artifact_path = None
    if init_art is not None:
        init_artifact_path, payload = init_art
        previous_gates = payload["gates"]
        previous_params = np.asarray(payload["params"], dtype=np.float64)
    params_init = trainer._initial_params(
        initialize_from_data=initialize_from_data,
        gates=gates,
        visible_bits=trainer.visible_bits,
        previous_gates=previous_gates,
        previous_params=previous_params,
        initialize_missing_visible=init_artifact_path is None,
    )
    return dict(trainer=trainer, gates=gates, visible_wires=visible_wires,
                ancilla_wires=ancilla_wires, params_init=params_init,
                init_artifact_path=init_artifact_path, layout_name=layout_name)


def iter_runs(datasets):
    for ds in datasets:
        for scheme_key, stem in SCHEMES:
            for a in PANELS[ds]:
                for seed in SEEDS:
                    yield ds, scheme_key, stem, a, seed


# --------------------------------------------------------------------------
def phase_angles(datasets):
    import torch
    rows = []
    for ds in datasets:
        mmd = per_seed_mmd(ds)
        for ds_, scheme_key, stem, a, seed in iter_runs([ds]):
            ck = run_ckpt(ds, stem, a, seed)
            art = torch.load(ck / "qcbm_artifact.pt", map_location="cpu",
                             weights_only=False)
            built = build_run(ds, scheme_key, stem, a, seed)
            gates, vb = built["gates"], built["trainer"].visible_bits
            # sanity: rebuilt gates identical to the stored circuit
            assert list(art["gates"]) == list(gates), f"gate mismatch {ck}"
            assert list(art["ancilla_wires"]) == list(built["ancilla_wires"])
            theta = np.asarray(art["params"], dtype=np.float64)
            theta0 = np.asarray(built["params_init"], dtype=np.float64)
            m = sector_mask(gates, vb)
            assert m.sum() == len(gates) - len(art["visible_gates"])
            dtheta = theta - theta0
            ma_v, rms_v = sec_stats(theta[~m])
            ma_a, rms_a = sec_stats(theta[m])
            i_v, irms_v = sec_stats(theta0[~m])
            i_a, irms_a = sec_stats(theta0[m])
            d_v, drms_v = sec_stats(dtheta[~m])
            d_a, drms_a = sec_stats(dtheta[m])
            rows.append(dict(
                dataset=ds, scheme=scheme_key, ancilla=a, seed=seed,
                n_params=len(gates), n_vis=int((~m).sum()), n_anc=int(m.sum()),
                mmd2_test=mmd.get((scheme_key, a, seed), float("nan")),
                mean_abs_vis=ma_v, mean_abs_anc=ma_a,
                rms_vis=rms_v, rms_anc=rms_a,
                ratio_anc_vis=ma_a / ma_v if ma_v else float("nan"),
                init_mean_abs_vis=i_v, init_mean_abs_anc=i_a,
                init_rms_vis=irms_v, init_rms_anc=irms_a,
                d_mean_abs_vis=d_v, d_mean_abs_anc=d_a,
                d_rms_vis=drms_v, d_rms_anc=drms_a,
                d_ratio_anc_vis=d_a / d_v if d_v else float("nan"),
                selected_source=art.get("selected_source"),
                init_artifact="loaded" if built["init_artifact_path"] else "reconstructed",
            ))
            print(f"[angles] {ds} {scheme_key} a={a} seed={seed}: "
                  f"mean|th_vis|={ma_v:.4f} mean|th_anc|={ma_a:.4f} "
                  f"ratio={ma_a/ma_v:.3f} mmd2={rows[-1]['mmd2_test']:.3e}",
                  flush=True)
    fp = OUT / "angles_per_run.csv"
    write_csv(fp, rows)
    print("wrote", fp)


# --------------------------------------------------------------------------
def phase_grad(datasets):
    import jax
    import jax.numpy as jnp
    import torch
    import iqpopt.gen_qml as gen
    from iqpopt import IqpSimulator
    from randomized_iqp.trainer import branch_weighted_mixture_mmd_loss

    fp = OUT / "grad_per_run.csv"
    done = set()
    if fp.exists():
        for r in csv.DictReader(open(fp)):
            done.add((r["dataset"], r["scheme"], int(r["ancilla"]),
                      int(r["seed"]), r["key_name"]))
    rows = []
    for ds in datasets:
        for ds_, scheme_key, stem, a, seed in iter_runs([ds]):
            if all((ds, scheme_key, a, seed, kn) in done
                   for kn in ("step0", "alt1", "alt2")):
                continue
            built = build_run(ds, scheme_key, stem, a, seed)
            trainer = built["trainer"]
            qcbm = trainer.qcbm
            gates = built["gates"]
            m = sector_mask(gates, trainer.visible_bits)
            m_dev = jnp.asarray(m)
            circuit = IqpSimulator(
                trainer.visible_bits + a, gates,
                device=str(qcbm.get("device", "lightning.qubit")),
                spin_sym=bool(qcbm.get("spin_sym", False)),
                sparse=bool(qcbm.get("sparse", False)),
                bitflip=bool(qcbm.get("bitflip", False)))
            loss_fn = gen.mmd_loss_iqp
            kw = dict(
                iqp_circuit=circuit,
                sigma=[float(s) for s in qcbm["sigmas"]],
                n_ops=int(qcbm["n_ops"]),
                n_samples=int(qcbm["n_samples"]),
                wires=built["visible_wires"],
                jit=bool(qcbm.get("loss_jit", True)),
                sqrt_loss=False,
                max_batch_ops=qcbm.get("max_batch_ops"),
                max_batch_samples=qcbm.get("max_batch_samples"),
                ground_truth=trainer.train_bits,
            )
            mode = str(qcbm.get("training_mode", "")).strip().lower()
            if mode == "cluster_weighted_ancilla":
                payload = trainer._build_cluster_weight_payload(
                    ancilla_bits=a, n_qubits=trainer.visible_bits + a,
                    ancilla_wires=built["ancilla_wires"])
                loss_fn = branch_weighted_mixture_mmd_loss
                kw.update(branch_condition_masks=payload["condition_masks"],
                          branch_condition_values=payload["condition_values"],
                          branch_sample_counts=payload["sample_counts"])

            params = jnp.asarray(built["params_init"], dtype=float)

            # exact step-0 training key (single active schedule block => index 0)
            rs = seed + a * 1000
            key = jax.random.PRNGKey(rs)
            key1, _key2 = jax.random.split(key, 2)
            k_step0, _ = jax.random.split(key1, 2)
            keys = [("step0", k_step0),
                    ("alt1", jax.random.PRNGKey(900001 + rs)),
                    ("alt2", jax.random.PRNGKey(900002 + rs))]

            def f(p, k):
                return loss_fn(p, key=k, **kw)
            vg = jax.jit(jax.value_and_grad(f))

            art = torch.load(run_ckpt(ds, stem, a, seed) / "qcbm_artifact.pt",
                             map_location="cpu", weights_only=False)
            loss_first = float(art["schedule"][0]["loss_first"])

            for key_name, k in keys:
                if (ds, scheme_key, a, seed, key_name) in done:
                    continue
                val, g = vg(params, k)
                g = np.asarray(g, dtype=np.float64)
                val = float(val)
                gv, ga = g[~m], g[m]
                n_v, n_a = gv.size, ga.size
                nv2 = float(np.linalg.norm(gv))
                na2 = float(np.linalg.norm(ga))
                mv = float(np.mean(np.abs(gv)))
                mA = float(np.mean(np.abs(ga)))
                rel = abs(val - loss_first) / max(abs(loss_first), 1e-12)
                row = dict(
                    dataset=ds, scheme=scheme_key, ancilla=a, seed=seed,
                    key_name=key_name, loss_at_init=val,
                    loss_first_artifact=loss_first,
                    loss_match_rel=(rel if key_name == "step0" else float("nan")),
                    gnorm_vis=nv2, gnorm_anc=na2,
                    gmean_vis=mv, gmean_anc=mA,
                    gratio_norm=na2 / nv2 if nv2 else float("nan"),
                    gratio_mean=mA / mv if mv else float("nan"),
                    gratio_perparam=(na2 / math.sqrt(n_a)) / (nv2 / math.sqrt(n_v))
                    if nv2 else float("nan"),
                )
                tv, ta = topq_of(gv), topq_of(ga)
                for q in TOPQS:
                    row[f"top{q}_gmean_vis"] = tv[q]
                    row[f"top{q}_gmean_anc"] = ta[q]
                    row[f"top{q}_gratio"] = (ta[q] / tv[q] if tv[q]
                                             else float("nan"))
                row["gmax_vis"], row["gmax_anc"] = tv["max"], ta["max"]
                row["gmax_ratio"] = (ta["max"] / tv["max"] if tv["max"]
                                     else float("nan"))
                rows.append(row)
                tag = (f" MATCH loss_first rel={rel:.2e}"
                       if key_name == "step0" else "")
                print(f"[grad] {ds} {scheme_key} a={a} seed={seed} {key_name}: "
                      f"loss={val:.6f}{tag} |g_vis|={nv2:.3e} |g_anc|={na2:.3e} "
                      f"mean-ratio={mA/mv:.3e}", flush=True)
            # incremental save
            append_csv(fp, rows)
            rows = []
        jax.clear_caches()
    print("wrote", fp)


# --------------------------------------------------------------------------
def write_csv(fp, rows):
    if not rows:
        return
    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def append_csv(fp, rows):
    if not rows:
        return
    exists = fp.exists()
    with open(fp, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def phase_aggregate():
    import pandas as pd
    ang = pd.read_csv(OUT / "angles_per_run.csv")
    grad = pd.read_csv(OUT / "grad_per_run.csv")

    g_step0 = grad[grad.key_name == "step0"].drop(columns=["key_name"])
    g_all = (grad.groupby(["dataset", "scheme", "ancilla", "seed"])
             .agg(gratio_mean_avg=("gratio_mean", "mean"),
                  gratio_mean_spread=("gratio_mean", "std"),
                  gratio_norm_avg=("gratio_norm", "mean"),
                  gratio_perparam_avg=("gratio_perparam", "mean"),
                  gnorm_vis_avg=("gnorm_vis", "mean"),
                  gnorm_anc_avg=("gnorm_anc", "mean"),
                  gmean_vis_avg=("gmean_vis", "mean"),
                  gmean_anc_avg=("gmean_anc", "mean"))
             .reset_index())
    per = ang.merge(g_step0[["dataset", "scheme", "ancilla", "seed",
                             "loss_at_init", "loss_first_artifact",
                             "loss_match_rel"]],
                    on=["dataset", "scheme", "ancilla", "seed"], how="left")
    per = per.merge(g_all, on=["dataset", "scheme", "ancilla", "seed"],
                    how="left")
    per.to_csv(OUT / "ancilla_sector_per_run.csv", index=False)

    num_cols = [c for c in per.columns if c not in
                ("dataset", "scheme", "ancilla", "seed", "selected_source",
                 "init_artifact")]
    agg = (per.groupby(["dataset", "scheme", "ancilla"])[num_cols]
           .agg(["mean", "std"]))
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT / "ancilla_sector_aggregated.csv", index=False)
    print("wrote", OUT / "ancilla_sector_per_run.csv")
    print("wrote", OUT / "ancilla_sector_aggregated.csv")

    # printed table
    pd.set_option("display.width", 250)
    cols = ["dataset", "scheme", "ancilla", "mmd2_test_mean",
            "mean_abs_vis_mean", "mean_abs_anc_mean", "ratio_anc_vis_mean",
            "init_mean_abs_anc_mean", "d_mean_abs_anc_mean",
            "gratio_mean_avg_mean", "gratio_perparam_avg_mean",
            "loss_match_rel_mean"]
    print(agg[cols].to_string(index=False,
                              float_format=lambda v: f"{v:.4g}"))
    return per, agg


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "angles"
    dss = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PANELS)
    if what == "angles":
        phase_angles(dss)
    elif what == "grad":
        phase_grad(dss)
    elif what == "aggregate":
        phase_aggregate()
    else:
        raise SystemExit(f"unknown phase {what}")
