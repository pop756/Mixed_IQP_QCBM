"""Lockstep training of many mixed IQP-QCBM models on the closed-form MMD^2 loss.

    from iqp_fastcf import train, cosine_lr
    out = train(n, a, edges, data, sigmas, lr, steps, c0, pi, seeds, k_train=2000, rmax=10, gpu_budget=None,
                eval_steps=(), callback=None)

- data: (N, n) 0/1 int8 training set; edges: (E, 2) pairs of distinct qubits; G = n + E; L = 2**a; sigmas: training
  bandwidths.
- lr: a number (constant), a sequence of `steps` per-step values, or a function step -> lr (step 0-based), e.g.
  cosine_lr(1e-3, 15000) = optax.cosine_decay_schedule(1e-3, 15000).
- c0: (M, L, G) float64 initial compiled angles (S-major rows); pi: (M, L) branch weights (fixed, not trained).
- seeds: S subset-stream seeds; M must be a multiple of S and model m uses stream seeds[m // (M // S)].
- eval_steps, callback: callback(step, params) is called with a copy of the (M, L, G) parameters after `step` updates,
  for every step in eval_steps (0 <= step <= steps).
- Loss of a model per step: sum over b of w_b (sum_l pi_l z_{A_b}(theta^(l)) - t_{A_b})^2 with theta^(l) = 2 W c
  (W the Walsh matrix) and the stratified subsets A_b of the step (orders r = 1..rmax, k_r of each, weights w_b),
  drawn from the PCG64 stream of the model's seed exactly as iqp_hip_cf does; Adam (optax.adam(lr)) per model.
- Returns dict(params=(M, L, G) float64 numpy, losses=(steps, M) float64 numpy: the loss of every model at every step,
  evaluated before that step's update, plan=(offload level, models per chunk, branch parts) the fast engine ran, or None
  when the vendored engine ran).
This engine runs the whole campaign in one HIP library (fastcf.py): no JAX on the training path, models
processed in chunks sized by a device-memory budget (gpu_budget: None keeps all state on the device, the fastest;
bytes caps device memory and streams the angles and Adam state from pinned host memory, see _plan and fastcf.py).
Configurations outside the fast engine's templates (a > 4, rmax > 10) run on the vendored iqp_hip_cf engine.
make_evaluator gives the test MMD^2 of trained models (the IQP paper's evaluator; needs jax).
"""
import ctypes, math, os, warnings
import numpy as np
from .fastcf import load_lib

LOW_MEMORY = 182 << 20       # gpu_budget of the arena submission: about 0.2 GiB of device memory per run
CPU_FRAC = 0.65              # level 2: share of each chunk's Adam update done by host threads (relieves the host link)
MAX_CHUNK_ML = 32            # at most this many (model, branch) pairs per chunk


def cosine_lr(lr0, decay_steps, alpha=0.0):
    """optax.cosine_decay_schedule(lr0, decay_steps, alpha) as a function of the 0-based step."""
    return lambda s: lr0 * ((1.0 - alpha) * 0.5 * (1.0 + math.cos(math.pi * min(s, decay_steps) / decay_steps)) + alpha)


def _pack_u64(x_bits):
    n_rows, _ = x_bits.shape
    pad_bits = (-n_rows) % 64
    x_pad = np.pad(x_bits.T.astype(np.uint8), ((0, 1), (0, pad_bits)))
    return np.ascontiguousarray(np.packbits(x_pad, axis=1)).view(np.uint64), n_rows


def _check_bits(x, n, what):
    if x.ndim != 2 or x.shape[1] != n:
        raise ValueError(f"{what} must be (N, {n}), got shape {x.shape}")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError(f"{what} must hold 0/1 bits")


def _check_inputs(n, L, edges, data, c0, pi, S, steps):
    """Shapes and values the HIP engine trusts: a wrong shape there reads past a buffer or trains the wrong gates."""
    G = n + len(edges)
    _check_bits(data, n, "data")
    if len(edges):
        if edges.min() < 0 or edges.max() >= n or np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("edges must be pairs of distinct qubits in [0, n)")
        if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
            raise ValueError("edges contain a duplicate pair")
    if c0.ndim != 3 or c0.shape[1:] != (L, G):
        raise ValueError(f"c0 must be (M, {L}, {G}) for a = {L.bit_length() - 1} and {len(edges)} edges, got {c0.shape}")
    M = c0.shape[0]
    if pi.shape != (M, L):
        raise ValueError(f"pi must be ({M}, {L}) to match c0, got {pi.shape}")
    if M == 0 or M % S:
        raise ValueError(f"the number of models ({M}) must be a positive multiple of the number of seeds ({S})")
    if not (np.all(np.isfinite(c0)) and np.all(np.isfinite(pi))):
        raise ValueError("c0 and pi must be finite")
    if steps < 0:
        raise ValueError("steps must be >= 0")


def _lr_table(lr, steps):
    """(per-step lr array, constant?)."""
    if callable(lr):
        return np.array([float(lr(s)) for s in range(steps)], dtype=np.float64), False
    if np.ndim(lr) == 0:
        return np.full(steps, float(lr)), True
    t = np.asarray(lr, dtype=np.float64)
    if t.shape != (steps,):
        raise ValueError(f"an lr sequence must have one value per step ({steps}), got shape {t.shape}")
    return t, False


def _subset_plan(n, sigmas, k_train, rmax):
    w = np.zeros(rmax + 1, dtype=np.float64)
    for s_ in sigmas:
        p = 0.5 * (1.0 - math.exp(-1.0 / (2.0 * s_ ** 2)))
        w += [math.comb(n, r) * p ** r * (1.0 - p) ** (n - r) for r in range(rmax + 1)]
    wbar = w / len(sigmas)
    kr = {r: max(8, int(round(k_train * wbar[r] / wbar[1:].sum()))) for r in range(1, rmax + 1)}
    b_info, w_of_b, p_off = [], [], 0
    for r, k in kr.items():
        for _ in range(k):
            b_info.append((r, p_off))
            w_of_b.append(wbar[r] / k)
            p_off += 1 << (r - 1)
    return np.array(b_info, dtype=np.int32), np.array(w_of_b, dtype=np.float64), p_off


def _pcg_tables(seeds, n):
    mask_128 = (1 << 128) - 1
    mul_128 = (2549297995355413924 << 64) | 4865540595714422341
    mask_64 = (1 << 64) - 1
    pcg_inits, j_tabs = [], []
    for sd in seeds:
        st_dict = np.random.default_rng(sd + 1).bit_generator.state["state"]
        s0_128, inc_128 = int(st_dict["state"]), int(st_dict["inc"])
        pcg_inits.append([s0_128 & mask_64, s0_128 >> 64, inc_128 & mask_64, inc_128 >> 64])
        tab, cur_mul, cur_plus = [], 1, 0
        for _ in range(n):
            cur_mul = (cur_mul * mul_128) & mask_128
            cur_plus = (cur_plus * mul_128 + inc_128) & mask_128
            tab.append([cur_mul & mask_64, cur_mul >> 64, cur_plus & mask_64, cur_plus >> 64])
        j_tabs.append(tab)
    return np.ascontiguousarray(np.array(pcg_inits, dtype=np.uint64)), np.ascontiguousarray(np.array(j_tabs, dtype=np.uint64))


def _edge_index(n, edges):
    eidx = np.full((n, n), -1, dtype=np.int32)
    if len(edges):
        e = np.arange(len(edges), dtype=np.int32) + n
        eidx[edges[:, 0], edges[:, 1]] = e
        eidx[edges[:, 1], edges[:, 0]] = e
    d = np.arange(n)
    eidx[d, d] = d
    return eidx


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _plan(n, L, G, M, s_tot, B, W_u64, budget):
    """(offload level, models per chunk, branch parts, fits): the lowest level, then the fewest parts, whose device
    footprint fits the budget (bytes) with the largest chunk. Level 0: all state on the device; 1: Adam moments in pinned
    host memory; 2: compiled angles too. With parts > 1 (levels >= 1, one model per chunk) only L / parts dense
    branch-angle matrices are held at a time. fits = False: nothing fits, the smallest plan is returned."""
    fixed = n * n * 4 + (n + 1) * W_u64 * 8 + 64 * n + 64 * B + (8 << 20)
    def footprint(level, mc, h):
        per = mc * ((L // h) * 2 * n * n + L * (s_tot + 12 * B)) * 8
        per += (0, 1, 3)[level] * mc * L * G * 8
        return fixed + (3, 1, 0)[level] * M * L * G * 8 + per
    # level 1 only when level 2 cannot run (it needs >= 3 chunks, i.e. M >= 3): with the host share of the Adam update,
    # level 2 is as fast and always smaller
    off = 2 if M >= 3 else 1
    for level in (0, off):
        for h in [1 << i for i in range(L.bit_length())]:
            if h > 1 and level == 0:
                break
            best = 0
            for mc in range(1, M + 1):
                if mc * L > max(MAX_CHUNK_ML, L) or (level > 0 and 4 * mc > M and mc > 1) or (h > 1 and mc > 1):
                    break
                if footprint(level, mc, h) <= budget:
                    best = mc
            if best:
                nck = -(-M // best)
                return level, -(-M // nck), h, True      # same chunk count, evened-out chunk sizes
    return off, 1, L, False


def _train_reference(n, a, edges, data, sigmas, lrs, const, steps, c0, pi, seeds, k_train, rmax, eval_steps, callback):
    import jax, jax.numpy as jnp
    from .iqp_hip_cf import MultiInitTrainer, pack_u64_np
    if const:
        lr = float(lrs[0]) if steps else 1e-3
    else:
        tab = jnp.asarray(lrs)
        lr = lambda count: tab[jnp.minimum(count, steps - 1)]      # optax counts updates from 0
    x_u64, n_tr = pack_u64_np(np.asarray(data, dtype=np.int8))
    x_u64 = jnp.asarray(x_u64)
    tr = MultiInitTrainer(n=n, a=a, edges=edges, xtr_u64=x_u64, n_tr=n_tr, xte_u64=x_u64, n_te=n_tr,
                          sigmas=list(sigmas), lr=lr, seeds=list(seeds), k_train=k_train, rmax=rmax, chunk=50)
    cs = jnp.asarray(c0, dtype=jnp.float64)
    pis = jnp.asarray(pi, dtype=jnp.float64)
    states = jax.vmap(tr.opt.init)(cs)
    losses, s = [], 0
    for e in sorted(set(eval_steps) | {steps}):
        while s + 50 <= e:
            cs, states, ls = tr.run_chunk(cs, states, jnp.asarray(s, dtype=jnp.int32), pis)
            losses.append(np.asarray(ls)); s += 50
        while s < e:
            cs, states, ls = tr.update_step(cs, states, jnp.asarray(s, dtype=jnp.int32), pis)
            losses.append(np.asarray(ls)[None]); s += 1
        if e in eval_steps:
            callback(e, np.array(cs))
    out_l = np.concatenate(losses, axis=0) if losses else np.zeros((0, len(c0)))
    return dict(params=np.array(cs), losses=out_l, plan=None)


def _run_engine(lib, h, steps, lrs, const, losses, eval_steps, callback, shape, fired, pi_out=None):
    """Final params, or None if the engine failed. fired[0] is set once a callback has run. With trainable weights,
    pi_out (M, L) receives the current weights at every callback and at the end, and the callback gets them too."""
    params = np.zeros(shape, dtype=np.float64)
    s = 0
    for e in sorted(set(eval_steps) | {steps}):
        if const:
            if e > s and lib.eng_run(h, s, e - s, float(lrs[0]), 0.9, 0.999, 1e-8, _ptr(losses[s:e])) != 0:
                return None
        else:
            for t in range(s, e):     # one call per step: Adam's bias correction counts the global step
                if lib.eng_run(h, t, 1, float(lrs[t]), 0.9, 0.999, 1e-8, _ptr(losses[t])) != 0:
                    return None
        s = e
        if (e in eval_steps or e == steps) and lib.eng_get_params(h, _ptr(params)) != 0:
            return None
        if pi_out is not None and (e in eval_steps or e == steps) and lib.eng_get_pi(h, _ptr(pi_out)) != 0:
            return None
        if e in eval_steps:
            fired[0] = True
            if pi_out is None:
                callback(e, params.copy())
            else:
                callback(e, params.copy(), pi_out.copy())
    return params


def train(n, a, edges, data, sigmas, lr, steps, c0, pi, seeds, k_train=2000, rmax=10, gpu_budget=None,
          eval_steps=(), callback=None, train_pi=False, pi_lr_scale=1.0):
    """train_pi: also train the branch weights, pi = softmax(beta) with beta initialized to log pi (all pi > 0), by
    Adam at the angles' schedule times pi_lr_scale; the weights move once per step after every model's angle update
    used them. The result then carries pi=(M, L) final weights and callback(step, params, pi) gets the current ones.
    Fast engine only."""
    n, a, steps = int(n), int(a), int(steps)
    L = 1 << a
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    G = n + len(edges)
    seeds = [int(s) for s in seeds]
    data = np.asarray(data)
    c0 = np.asarray(c0, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    if c0.ndim == 2 and pi.ndim == 1:                     # a single model
        c0, pi = c0[None], pi[None]
    _check_inputs(n, L, edges, data, c0, pi, len(seeds), steps)
    data = data.astype(np.int8)
    c0, pi = np.ascontiguousarray(c0), np.ascontiguousarray(pi)
    eval_steps = {int(e) for e in eval_steps}
    if eval_steps and callback is None:
        raise ValueError("eval_steps needs a callback")
    if any(e < 0 or e > steps for e in eval_steps):
        raise ValueError(f"eval_steps must lie in [0, {steps}]")
    lrs, const = _lr_table(lr, steps)
    M, S = c0.shape[0], len(seeds)
    if train_pi and not np.all(pi > 0):
        raise ValueError("trainable weights start from log pi: every initial weight must be > 0")
    if train_pi and not (1 <= int(rmax) <= 10 and a <= 4):
        raise ValueError("trainable weights need the fast engine (a <= 4, rmax <= 10)")
    ref_args = (n, a, edges, data, sigmas, lrs, const, steps, c0, pi, seeds, int(k_train), int(rmax), eval_steps,
                callback)
    if not (1 <= int(rmax) <= 10 and a <= 4):           # outside the fast engine's templates: vendored engine
        warnings.warn(f"a = {a}, rmax = {rmax} is outside the fast kernels (a <= 4, rmax <= 10): running the slower "
                      "vendored iqp_hip_cf engine (needs jax)", RuntimeWarning, stacklevel=2)
        return _train_reference(*ref_args)
    x_u64, n_tr = _pack_u64(data)
    b_info, w_of_b, s_tot = _subset_plan(n, list(sigmas), int(k_train), int(rmax))
    pcg, jtab = _pcg_tables(seeds, n)
    eidx = _edge_index(n, edges)
    level, mc, parts, fits = _plan(n, L, G, M, s_tot, len(b_info), x_u64.shape[1],
                                   math.inf if gpu_budget is None else int(gpu_budget))
    if not fits:
        warnings.warn(f"gpu_budget = {gpu_budget} bytes cannot be met; using the smallest plan (level {level}, one model "
                      f"and {L // parts} branch(es) at a time)", RuntimeWarning, stacklevel=2)
    if "CFIQP_LEVEL" in os.environ:                      # manual override (experiments)
        level, mc, parts = map(int, os.environ["CFIQP_LEVEL"].split(","))
    # our kernels use no scratch (private) memory: skip the ROCm runtime's default 140 MiB per-queue scratch reservation
    os.environ.setdefault("HSA_SCRATCH_SINGLE_LIMIT", "0")
    lib = load_lib()
    try:
        ncpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        ncpu = os.cpu_count() or 4
    cpu_threads = max(1, min(16, ncpu - 4))
    cpu_frac = float(os.environ.get("CFIQP_CPU_FRAC", CPU_FRAC * cpu_threads / 16.0))
    # if the planned configuration cannot be set up (e.g. no pinned host memory), retry fully on the device, then
    # fall back to the vendored engine; each attempt starts from the inputs, so a retry is exact
    plans = [(level, mc, parts)] + ([(0, 1, 1)] if (level, mc, parts) != (0, 1, 1) else [])
    fired = [False]
    for lv, m_c, h_p in plans:
        h = ctypes.c_void_p()
        rc = lib.eng_create(ctypes.byref(h), n, a, M, S, len(b_info), s_tot, G, x_u64.shape[1], n_tr, m_c, lv, h_p,
                            cpu_frac, cpu_threads, _ptr(eidx),
                            _ptr(c0), _ptr(pi), _ptr(pcg), _ptr(jtab), _ptr(b_info), _ptr(w_of_b), _ptr(x_u64))
        if rc != 0:
            if h.value:
                lib.eng_destroy(h)
            warnings.warn(f"fast engine setup failed (level {lv}, {m_c} model(s) per chunk); see the HIP error above",
                          RuntimeWarning, stacklevel=2)
            continue
        losses = np.zeros((steps, M), dtype=np.float64)
        plan = np.zeros(3, dtype=np.int32)
        try:
            lib.eng_plan(h, _ptr(plan))
            pi_out = None
            if train_pi:
                pi_out = np.zeros((M, L), dtype=np.float64)
                if lib.eng_train_pi(h, float(pi_lr_scale)) != 0:
                    raise RuntimeError("eng_train_pi failed; see the HIP error above")
            params = _run_engine(lib, h, steps, lrs, const, losses, eval_steps, callback, (M, L, G), fired, pi_out)
        finally:
            lib.eng_destroy(h)
        if params is not None:
            out = dict(params=params, losses=losses, plan=tuple(int(v) for v in plan))
            if train_pi:
                out["pi"] = pi_out
            return out
        if fired[0]:
            raise RuntimeError("the fast engine failed after callbacks had run; see the HIP error above")
        warnings.warn(f"fast engine failed while training (level {lv}); see the HIP error above", RuntimeWarning,
                      stacklevel=2)
    warnings.warn("falling back to the vendored iqp_hip_cf engine (slower, needs jax)", RuntimeWarning, stacklevel=2)
    return _train_reference(*ref_args)


def make_evaluator(n, a, edges, x_test, keval=10_000, rmax=10):
    """evaluate(params, pi) -> (mean (M,), per_mbar (M, 6)): test MMD^2 of trained models, the IQP paper's evaluator
    (iqp_hip_cf.MultiInitTrainer.test_mmd2_batch; needs jax). For m-bar = 1..6, keval subsets A ~ Bernoulli(m-bar/n)^n
    (rng seed 1000 + m-bar), |A| <= rmax kept: sum_A (sum_l pi_l z_A(theta^(l)) - t_A)^2 / keval, t_A from x_test;
    mean = average over m-bar. params: compiled angles as train returns them, (M, L, G) or (L, G)."""
    import jax.numpy as jnp
    from .iqp_hip_cf import MultiInitTrainer, pack_u64_np
    n, L = int(n), 1 << int(a)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    G = n + len(edges)
    x_test = np.asarray(x_test)
    _check_bits(x_test, n, "x_test")
    x_u64, n_te = pack_u64_np(x_test.astype(np.int8))
    x_u64 = jnp.asarray(x_u64)
    tr = MultiInitTrainer(n=n, a=int(a), edges=edges, xtr_u64=x_u64, n_tr=n_te, xte_u64=x_u64, n_te=n_te,
                          sigmas=[1.0], keval=int(keval), rmax=int(rmax))

    def evaluate(params, pi):
        params, pi = np.asarray(params, dtype=np.float64), np.asarray(pi, dtype=np.float64)
        if params.ndim == 2 and pi.ndim == 1:
            params, pi = params[None], pi[None]
        if params.ndim != 3 or params.shape[1:] != (L, G) or pi.shape != (params.shape[0], L):
            raise ValueError(f"params must be (M, {L}, {G}) and pi (M, {L}), got {params.shape} and {pi.shape}")
        mean, per = tr.test_mmd2_batch(np.einsum("lk,mkg->mlg", tr.W, params), pi)
        return np.array(mean), np.array(per)
    return evaluate
