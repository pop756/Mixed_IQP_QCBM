#!/usr/bin/env python3
"""Rebuild the cluster-start artifacts of the earlier campaign (n = 16).

The cluster configurations of <ds>_seed/configs/ load their start from
<ds>_seed/outputs_s2_cluster_a<a>_init_usqm_seed<N>/checkpoints/fully_connected_order2/a<a>/qcbm_artifact.pt.
Those files were written by the init stage of the earlier pipeline and are not kept. This script writes them
again from the training split and the cluster labels, with the same calls as that stage (uniform-noise mode):
one ancilla-free circuit per cluster with one-body angles arcsin(sqrt(mean)) and two-body angles 0
(iqpopt initialize_from_data, zero_two_body), embedded into one ancilla circuit by
randomized_iqp.ancilla.embed_visible_branches_into_ancilla. No random number is drawn.

usage: rebuild_cluster_init.py [ds ...]        (default: blobs ising; every a = 1, 2, 3 and every seed with labels)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["IQPOPT_X64"] = "1"
os.environ["JAX_ENABLE_X64"] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = HERE.parent / "randomized_iqp_mnist8x8"
for p in (str(SHARED), str(SHARED / "vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402

jax.config.update("jax_enable_x64", True)

from iqpopt.utils import initialize_from_data                             # noqa: E402
from randomized_iqp.ancilla import branch_bitstrings, embed_visible_branches_into_ancilla   # noqa: E402
from randomized_iqp.gates import build_visible_gates                      # noqa: E402

LAYOUT_NAME, LAYOUT, N = "fully_connected_order2", {"type": "fully_connected", "max_order": 2}, 16
INIT_SCALE = 0.1                                   # inert here: the two-body angles are set to zero


def center(visible_gates, bits):
    p = np.asarray(initialize_from_data(visible_gates, jnp.asarray(bits, dtype=jnp.int32), scale=INIT_SCALE,
                                        param_noise=0.0, zero_two_body=True, noise_all=False), dtype=np.float64)
    return np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)


def build(ds, a, seed):
    root = HERE / f"{ds}_seed"
    train = np.loadtxt(REPO / "experiments" / "data" / ds / "x_train.csv", delimiter=",", dtype=np.int8)
    y = np.loadtxt(root / f"dataset_s2_cluster_a{a}_usqm_seed{seed}" / "y_train.csv", delimiter=",", dtype=np.int64).ravel()
    assert len(y) == len(train), (len(y), len(train))
    visible_gates = build_visible_gates(LAYOUT_NAME, LAYOUT, N)
    global_params = center(visible_gates, train)
    groups = sorted(int(v) for v in np.unique(y))
    by_branch = {format(i, f"0{a}b"): g for i, g in enumerate(groups)}
    circuits, rows = {}, []
    for branch in branch_bitstrings(a):
        g = by_branch.get(branch)
        if g is None:
            params, n, src = global_params.copy(), 0, "global_init_inactive_fill"
        else:
            sel = y == int(g)
            params, n, src = center(visible_gates, train[sel]), int(sel.sum()), "group_data_init"
        circuits[branch] = (visible_gates, params)
        rows.append({"branch": branch, "label": g, "n": n, "source": src, "param_l2": float(np.linalg.norm(params))})
    gates, params, meta = embed_visible_branches_into_ancilla(circuits, visible_bits=N, ancilla_bits=a, zero_tol=1e-12)
    out = root / f"outputs_s2_cluster_a{a}_init_usqm_seed{seed}" / "checkpoints" / LAYOUT_NAME / f"a{a}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "routed_init_v1", "layout_name": LAYOUT_NAME, "layout": LAYOUT, "n_qubits": N + a,
                "visible_bits": N, "ancilla_bits": a, "visible_wires": list(range(N)),
                "ancilla_wires": meta["ancilla_wires"], "visible_gates": visible_gates, "gates": gates,
                "params": torch.as_tensor(params, dtype=torch.float64),
                "losses": torch.as_tensor([], dtype=torch.float64), "training_mode": "routed_data_init_embedding",
                "branch_embedding": {"source": f"{ds}_seed_routed", "mapping": rows}}, out / "qcbm_artifact.pt")
    return out, len(gates)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["blobs", "ising"]:
        dirs = sorted((HERE / f"{ds}_seed").glob("dataset_s2_cluster_a[123]_usqm_seed*"))
        for d in dirs:
            a, seed = map(int, re.fullmatch(r"dataset_s2_cluster_a(\d)_usqm_seed(\d+)", d.name).groups())
            out, m = build(ds, a, seed)
            print(f"{ds} a={a} seed={seed}: {m} gates -> {out.relative_to(HERE)}", flush=True)
