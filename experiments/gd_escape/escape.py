"""Escape from branch coincidence under plain gradient descent and Adam (n = 16, a = 1, exact loss).

Start: a coincident centre (global or data-agnostic) with the campaign's common-mode noise on the shared compiled
angles c_{} (half-width pi/(2 sqrt(m)), m = L G, one fixed draw), and a branch-separating perturbation c_{1} = delta u
(u ~ Unif[-1, 1]^G, one fixed draw), delta in DELTAS (delta = 0: exact coincidence). Uniform weights. Loss: the exact
training MMD^2 of exact16 (all 2^16 subsets, m-bar = 2, 6 averaged), so the only randomness is the start.

Optimizers: plain GD at a constant step ETA (eta lambda_max(H_sys) <= 0.16 at both centres and at th*, so GD is
stable and resolves the curvature time scales), and Adam at a constant lr 1e-3 (the campaign's initial lr;
beta = 0.9, 0.999, eps = 1e-8).

Recorded every REC steps: loss, separation s = ||c_{1}||_2, exact D_branch^2. Escape times:
  T_sep : first step with s >= S_C (the separating angles leave the linear regime),
  T_floor: first step with loss <= (1 - EPS_F) * floor, floor = the exact single-circuit optimum (diag.py); by
           Proposition diversity a coincident mixture cannot go below the floor.
Linear prediction (GD): along the delta = 0 trajectory th_bar(t), the separating angles obey
s_{t+1} = (I - eta M(th_bar_t)) s_t (Proposition coinc-hessian, L = 2); propagated once from u, it gives
T_sep^pred(delta) = first t with delta ||Phi_t u|| >= S_C, and the rate |lambda_min(M(th*))| sets the slope
dT/dln(1/delta) -> 1 / (eta |lambda_min(M(th*))|).
usage: escape.py <ds> <scheme>  ->  results/escape_<ds>_<scheme>.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from exact16 import Exact16

HERE = Path(__file__).resolve().parent
DELTAS = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
STEPS_GD, STEPS_ADAM, REC = 100000, 20000, 10
S_C, EPS_F = 0.05, 0.01
ETA = 1.0
SEED = 7

ds, scheme = sys.argv[1], sys.argv[2]
E = Exact16(ds, 1)
L, G = E.L, E.G
pi = jnp.full(L, 1.0 / L)
c_centre = E.coincident_centre(scheme)
rng = np.random.default_rng(SEED)
hw = (np.pi / 2) / np.sqrt(L * G)
common = rng.uniform(-hw, hw, G)
u = rng.uniform(-1.0, 1.0, G)
starts = []
for d in DELTAS:
    c = c_centre.copy()
    c[0] += common
    c[1] = d * u
    starts.append(c)
starts = jnp.asarray(np.stack(starts))
th_star = jnp.asarray(np.load(HERE / f"thstar_{ds}.npy"))
floor = float(E.loss_single(th_star))
lam_max = float(np.linalg.eigvalsh(np.asarray(E.single_hessian(jnp.asarray(c_centre[0] + common))))[-1])
eta = ETA
assert eta * lam_max < 0.5, lam_max
lam_M_star = float(np.linalg.eigvalsh(np.asarray(E.m_block(th_star)))[0])

loss_b = jax.vmap(lambda c: E.loss_c(c, pi))
grad_b = jax.vmap(jax.grad(lambda c: E.loss_c(c, pi)))
div_b = jax.vmap(lambda c: E.diversity(c, pi))


def record(c):
    return jnp.stack([loss_b(c), jnp.linalg.norm(c[:, 1], axis=-1), div_b(c)], axis=-1)


@jax.jit
def gd_chunk(c):
    def body(c, _):
        return c - eta * grad_b(c), None
    c, _ = jax.lax.scan(body, c, None, length=REC)
    return c, record(c)


def adam_init(c):
    return (jnp.zeros_like(c), jnp.zeros_like(c), 0)


@jax.jit
def adam_chunk(c, m, v, t0):
    def body(carry, k):
        c, m, v = carry
        g = grad_b(c)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        t = t0 + k + 1
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        return (c - 1e-3 * mh / (jnp.sqrt(vh) + 1e-8), m, v), None
    (c, m, v), _ = jax.lax.scan(body, (c, m, v), jnp.arange(REC))
    return c, m, v, record(c)


def run(kind, steps):
    c, rec = starts, [np.asarray(record(starts))]
    m = v = jnp.zeros_like(starts)
    t0 = time.perf_counter()
    for k in range(steps // REC):
        if kind == "gd":
            c, r = gd_chunk(c)
        else:
            c, m, v, r = adam_chunk(c, m, v, k * REC)
        rec.append(np.asarray(r))
    rec = np.stack(rec)                                   # (T, len(DELTAS), 3)
    print(f"{ds} {scheme} {kind}: {steps} steps in {time.perf_counter() - t0:.0f} s", flush=True)
    return rec


def first(mask):
    i = np.flatnonzero(mask)
    return int(i[0] * REC) if len(i) else None


# linear prediction of the separating angles along the delta = 0 GD trajectory
@jax.jit
def lin_chunk(cbar, s):
    def hvp(th, x):                                        # M(th) x, M the coincidence block at th
        f = lambda y: jnp.sum(2.0 * E.w * jax.lax.stop_gradient(E.branch_z(th[None])[0] - E.t)
                              * E.branch_z(y[None])[0])
        return jax.jvp(jax.grad(f), (th,), (x,))[1]

    def body(carry, _):
        cbar, s = carry
        th = cbar[0]                                       # branch angle at coincidence (L = 2: theta = c_{} )
        s_new = s - eta * hvp(th, s)
        cbar = cbar - eta * jax.grad(lambda c: E.loss_c(c, pi))(cbar)
        return (cbar, s_new), None
    (cbar, s), _ = jax.lax.scan(body, (cbar, s), None, length=REC)
    return cbar, s


out = dict(dataset=ds, scheme=scheme, a=1, seed=SEED, deltas=DELTAS, eta=eta, lam_max_centre=lam_max,
           lam_min_M_thstar=lam_M_star, floor=floor, S_C=S_C, EPS_F=EPS_F, REC=REC)
for kind, steps in (("gd", STEPS_GD), ("adam", STEPS_ADAM)):
    rec = run(kind, steps)
    out[kind] = dict(steps=steps, loss=rec[:, :, 0].T.tolist(), sep=rec[:, :, 1].T.tolist(),
                     div=rec[:, :, 2].T.tolist(),
                     T_sep=[first(rec[:, i, 1] >= S_C) for i in range(len(DELTAS))],
                     T_floor=[first(rec[:, i, 0] <= (1 - EPS_F) * floor) for i in range(len(DELTAS))],
                     final_loss=rec[-1, :, 0].tolist(), final_div=rec[-1, :, 2].tolist())
# linear propagation from the unit perturbation u (the scale delta multiplies it)
cbar = jnp.asarray(starts[0])
s = jnp.asarray(u)
lin = [float(jnp.linalg.norm(s))]
t0 = time.perf_counter()
for k in range(STEPS_GD // REC):
    cbar, s = lin_chunk(cbar, s)
    lin.append(float(jnp.linalg.norm(s)))
print(f"{ds} {scheme} linear propagation: {time.perf_counter() - t0:.0f} s", flush=True)
lin = np.array(lin)
out["gd"]["lin_norm_unit"] = lin.tolist()
out["gd"]["T_sep_pred"] = [first(d * lin >= S_C) if d > 0 else None for d in DELTAS]
(HERE / "results").mkdir(exist_ok=True)
json.dump(out, open(HERE / "results" / f"escape_{ds}_{scheme}.json", "w"))
print(json.dumps(dict(eta=eta, lam_min_M_thstar=lam_M_star, slope_pred=1 / (eta * abs(lam_M_star)),
                      T_sep_gd=out["gd"]["T_sep"], T_sep_pred=out["gd"]["T_sep_pred"],
                      T_floor_gd=out["gd"]["T_floor"], T_sep_adam=out["adam"]["T_sep"],
                      T_floor_adam=out["adam"]["T_floor"], final_gd=out["gd"]["final_loss"],
                      final_adam=out["adam"]["final_loss"])), flush=True)
