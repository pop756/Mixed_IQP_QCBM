"""Check of the single-circuit reference points thstar_<ds>.npy of the escape experiment (n = 16, CPU).

For each dataset: the exact training MMD^2 and the gradient norm at the stored point, the same after an L-BFGS-B
polish (a stationary point stays where it is), and with --random K the loss reached by the same polish from K
random starts (angles uniform on [-pi/2, pi/2], numpy default_rng(0)).

The stored points are local minima of the exact single-circuit loss. They are not what a fresh run of diag.py
returns, and on the blobs a random start reaches a lower loss (start 2 of --random 12: 1.774e-2 against 1.833e-2).
usage: thstar_check.py [blobs] [ising] [--random K]
"""
import sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from exact16 import Exact16

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
K = int(args[args.index("--random") + 1]) if "--random" in args else 0
names = [a for a in args if a in ("blobs", "ising")] or ["blobs", "ising"]

for ds in names:
    E = Exact16(ds, 0)
    vg = jax.jit(jax.value_and_grad(E.loss_single))

    def fun(x):
        l, g = vg(jnp.asarray(x))
        return float(l), np.asarray(g, dtype=np.float64)

    def polish(x0):
        r = minimize(fun, np.asarray(x0, dtype=np.float64), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=50000, maxfun=200000, ftol=1e-16, gtol=1e-11, maxcor=30))
        return r.x, float(r.fun)

    stored = np.load(HERE / f"thstar_{ds}.npy")
    l0, g0 = fun(stored)
    x1, l1 = polish(stored)
    print(f"{ds:6s} stored point   loss {l0:.6e}  |grad| {np.linalg.norm(g0):.1e}  polished {l1:.6e}  "
          f"moved by {np.abs(x1 - stored).max():.1e} rad", flush=True)
    assert abs(l1 - l0) <= 1e-5 * l0, "the stored point is not a local minimum of the exact single-circuit loss"
    rng = np.random.default_rng(0)
    best = l1
    for k in range(K):
        _, l = polish(rng.uniform(-np.pi / 2, np.pi / 2, len(stored)))
        best = min(best, l)
        print(f"{ds:6s} random start {k:2d} polished {l:.6e}", flush=True)
    if K:
        print(f"{ds:6s} lowest single-circuit loss found {best:.6e}  (stored point {l0:.6e})", flush=True)
