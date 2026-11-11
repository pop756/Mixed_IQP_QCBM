"""Diagnostics for the escape experiment: the single-circuit optimum th* of the exact loss (Adam from the global
centre), lambda_max of its Hessian, and the spectrum of the coincidence block M at the centres and at th*."""
import sys
import numpy as np
import jax, jax.numpy as jnp, optax
from exact16 import Exact16

for ds in sys.argv[1:] or ["blobs", "ising"]:
    E = Exact16(ds, 0)
    th0 = jnp.asarray(E.coincident_centre("global")[0])
    f = jax.jit(lambda th: E.loss_c(th[None], jnp.ones(1)))
    opt = optax.adam(1e-2)

    @jax.jit
    def run(th, st):
        def body(carry, _):
            th, st = carry
            l, g = jax.value_and_grad(f)(th)
            u, st = opt.update(g, st, th)
            return (optax.apply_updates(th, u), st), l
        (th, st), ls = jax.lax.scan(body, (th, st), None, length=2000)
        return th, st, ls
    th, st = th0, opt.init(th0)
    for k in range(10):
        th, st, ls = run(th, st)
    print(ds, "floor (exact train MMD^2) %.6e" % float(ls[-1]))
    for nm, x in (("agnostic centre", jnp.asarray(E.coincident_centre("agnostic")[0])), ("global centre", th0),
                  ("th*", th)):
        Hs = np.asarray(E.single_hessian(x)); M = np.asarray(E.m_block(x))
        ev = np.linalg.eigvalsh(M); eh = np.linalg.eigvalsh(Hs)
        print(f"  {nm:16s} lam_max(H_sys) {eh[-1]:.4e}  lam_min(H_sys) {eh[0]:.3e}  lam_min(M) {ev[0]:.4e}  "
              f"#neg(M) {np.sum(ev < -1e-12)}  lam_max(M) {ev[-1]:.3e}")
    np.save(f"thstar_{ds}.npy", np.asarray(th))
