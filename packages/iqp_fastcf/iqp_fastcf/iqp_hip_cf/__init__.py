#!/usr/bin/env python3
r"""`iqp_hip_cf`: High-performance Closed-Form IQP-QCBM Correlator & Multi-Init Trainer Package.

Key innovations in `iqp_hip_cf`:
1. **100% GPU 128-bit PCG64 Jump-Ahead Subset Sampler + Fused Bit-Packed `popcll` Target Moments**:
   Advances NumPy's exact 128-bit PCG64 state per subset row on GPU via `O(log2(delta))` jump-ahead and
   precomputed 128-bit powers `J_TAB_J`, selects top-r indices via warp-level `uint64` key-index reductions,
   and computes exact empirical moments `t_A` via `uint64` XOR + hardware `__popcll` in ~0.10 ms/step.
2. **2-Level 16-Sign Trigonometric Angle-Addition (`sincos`) + Bank-Conflict-Free LDS**:
   Evaluates 16 sign combinations per 1 `fast_sincos` call using precomputed `sincos(x[r-2] +- x[r-1])` and
   `sincos(x[r-4] +- x[r-3])` per thread `j`, reducing transcendental evaluations by 16x for |A| >= 5.
3. **4-Way Montgomery Batch Inversion in Backward Pass**:
   Replaces 4 IEEE-754 float64 divisions per 4-sign group with 1 division (`1.0 / (c01 * c23)`) and 6 FMAs.
4. **Lockstep Multi-Init `jax.lax.scan` Training + Single-Pass Batched Evaluator**:
   Trains all initializations simultaneously in 50-step `jax.lax.scan` chunks with zero Python loop overhead.
"""
import ctypes, functools, itertools, math
import numpy as np, jax, jax.numpy as jnp, optax
jax.config.update("jax_enable_x64", True)
from .hip_ext import load_hip_ext, pack_u64_np

R_MAX = 12
TILE_S = 64

__all__ = [
    "R_MAX",
    "TILE_S",
    "compiled_gates",
    "ClosedFormLoss",
    "MultiInitTrainer",
    "load_hip_ext",
    "pack_u64_np",
]


def compiled_gates(n: int, a: int, edges):
    """Gate list matching S-major parameter layout: data wires 0..n-1, ancilla wires n..n+a-1."""
    base = [[j] for j in range(n)] + [list(e) for e in edges]
    return [[[*g, *[n + b for b in range(a) if (S >> (a - 1 - b)) & 1]]]
            for S in range(2 ** a) for g in base]


def _walsh_matrix(a: int) -> np.ndarray:
    if a == 0:
        return np.ones((1, 1), dtype=np.float64)
    ell = np.array(list(itertools.product((0, 1), repeat=a)), dtype=np.int32)
    return (-1.0) ** ((ell @ ell.T) % 2)


# ---------------------------------------------------------------------------
# Custom VJP wrappers for the ROCm HIP kernels
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _hip_mix_eval(th1, Th, pi, A_pad, tiles, b_info, s_tot):
    load_hip_ext()
    M, L_, _ = th1.shape
    B_tot = A_pad.shape[-2]
    z_type = jax.ShapeDtypeStruct((M, B_tot), jnp.float64)
    p_type = jax.ShapeDtypeStruct((M, L_, s_tot), jnp.float64)
    z, _ = jax.ffi.ffi_call("cf_mix_fwd", (z_type, p_type))(th1, Th, pi, A_pad, tiles)
    return z


def _hip_mix_fwd_rule(th1, Th, pi, A_pad, tiles, b_info, s_tot):
    load_hip_ext()
    M, L_, _ = th1.shape
    B_tot = A_pad.shape[-2]
    z_type = jax.ShapeDtypeStruct((M, B_tot), jnp.float64)
    p_type = jax.ShapeDtypeStruct((M, L_, s_tot), jnp.float64)
    z, p = jax.ffi.ffi_call("cf_mix_fwd", (z_type, p_type))(th1, Th, pi, A_pad, tiles)
    return z, (th1, Th, pi, A_pad, b_info, p)


def _hip_mix_bwd_rule(s_tot, res, dz):
    load_hip_ext()
    th1, Th, pi, A_pad, b_info, p = res
    d_th1_type = jax.ShapeDtypeStruct(th1.shape, jnp.float64)
    d_Th_type = jax.ShapeDtypeStruct(Th.shape, jnp.float64)
    d_th1, d_Th = jax.ffi.ffi_call("cf_mix_bwd", (d_th1_type, d_Th_type))(
        th1, Th, pi, A_pad, b_info, p, dz)
    return d_th1, d_Th, None, None, None, None


_hip_mix_eval.defvjp(_hip_mix_fwd_rule, _hip_mix_bwd_rule)


def _hip_eval_only(th1, Th, pi, A_pad, tiles):
    load_hip_ext()
    M = th1.shape[0]
    B_tot = A_pad.shape[-2]
    z_type = jax.ShapeDtypeStruct((M, B_tot), jnp.float64)
    return jax.ffi.ffi_call("cf_mix_eval_only", z_type)(th1, Th, pi, A_pad, tiles)


# ---------------------------------------------------------------------------
# Pure JAX Fallback for CPU / Small Circuits (n < 32)
# ---------------------------------------------------------------------------

_SIGN_TABLES = {}
for _r in range(1, R_MAX + 1):
    if _r == 1:
        _SIGN_TABLES[_r] = jnp.ones((1, 1), dtype=jnp.float64)
    else:
        _bits = np.array(list(itertools.product((1.0, -1.0), repeat=_r - 1)), dtype=np.float64)
        _SIGN_TABLES[_r] = jnp.asarray(np.concatenate([np.ones((1 << (_r - 1), 1)), _bits], axis=1))


class ClosedFormLoss:
    r"""Exact closed-form IQP-QCBM MMD^2 loss and gradient evaluator for |A| <= 12.

    Supports both:
    - Arbitrary `ops` lists (`value_and_grad(params, ops)`), using OpenMP bit-packed popcount
      moment evaluation + fixed-capacity HIP FFI kernels on ROCm GPU (or vectorized JAX on CPU).
    """

    def __init__(self, n: int, a: int, edges, data, sigma, pi=None):
        self.n = int(n)
        self.a = int(a)
        self.L = 1 << self.a
        edges_arr = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        self.edges = edges_arr
        self.G = self.n + len(edges_arr)
        self.p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * float(sigma) ** 2)))
        self.e0 = jnp.asarray(edges_arr[:, 0], dtype=jnp.int32)
        self.e1 = jnp.asarray(edges_arr[:, 1], dtype=jnp.int32)
        self.W = jnp.asarray(_walsh_matrix(self.a), dtype=jnp.float64)
        self.pi = jnp.ones((self.L,), dtype=jnp.float64) / self.L if pi is None else jnp.asarray(pi, dtype=jnp.float64)
        self.pi_batch = self.pi[None, :]

        data_np = np.asarray(data, dtype=np.uint8)
        self.data_u64_np, self.N_data = pack_u64_np(data_np)
        self.W_u64 = self.data_u64_np.shape[1]
        self.data_pm1 = jnp.asarray(1.0 - 2.0 * data_np.astype(np.float64))

        self._vg_cpu_cache = {}
        self.use_hip = (jax.default_backend() == "gpu") and (self.n >= 32)
        if self.use_hip:
            self._lib = load_hip_ext()
            self._B_cap = 1024
            self._T_cap = 4096
            self._S_cap = 262144
            self._alloc_host_buffers()
            self._jit_cache = {}

    def sample_ops(self, K, rng):
        ops = rng.random((K, self.n)) < self.p
        return [np.flatnonzero(o) for o in ops]

    def _alloc_host_buffers(self):
        self._A_pad_np = np.full((self._B_cap, 12), self.n, dtype=np.int32)
        self._b_info_np = np.zeros((self._B_cap, 2), dtype=np.int32)
        self._tiles_np = np.zeros((self._T_cap, 5), dtype=np.int32)
        self._t_np = np.zeros((self._B_cap,), dtype=np.float64)

    def _get_hip_vg(self, s_cap: int):
        fn = self._jit_cache.get(s_cap)
        if fn is None:
            n, L, G = self.n, self.L, self.G
            W = self.W
            e0, e1 = self.e0, self.e1
            pi_b = self.pi_batch

            def loss_fn(params, A_pad, tiles, b_info, t, inv_k):
                c = params.reshape(L, G)
                th = 2.0 * (W @ c)
                th1 = th[None, :, :n]
                Th = jnp.zeros((1, L, n, n), dtype=jnp.float64).at[:, :, e0, e1].set(th[None, :, n:])
                Th = Th + Th.transpose(0, 1, 3, 2)
                z = _hip_mix_eval(th1, Th, pi_b, A_pad, tiles, b_info, s_cap)[0]
                active = (b_info[:, 0] > 0).astype(jnp.float64)
                return jnp.sum(active * (z - t) ** 2) * inv_k

            fn = jax.jit(jax.value_and_grad(loss_fn))
            self._jit_cache[s_cap] = fn
        return fn

    def _pack_ops_hip(self, ops):
        K_total = len(ops)
        valid_arrays = []
        lengths = []
        offsets = []
        cur_off = 0
        for op in ops:
            r = len(op)
            if 0 < r <= R_MAX:
                arr = np.asarray(op, dtype=np.int32)
                valid_arrays.append(arr)
                lengths.append(r)
                offsets.append(cur_off)
                cur_off += r

        num_valid = len(lengths)
        if num_valid == 0:
            self._b_info_np[:, 0] = 0
            self._tiles_np[:, 0] = -1
            self._t_np[:] = 0.0
            return K_total, self._S_cap

        while num_valid > self._B_cap:
            self._B_cap *= 2
            self._alloc_host_buffers()

        flat_indices = np.concatenate(valid_arrays).astype(np.int32, copy=False)
        offsets_np = np.asarray(offsets, dtype=np.int32)
        lengths_np = np.asarray(lengths, dtype=np.int32)

        while True:
            p_off = self._lib.pack_ops_moments_omp(
                flat_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                offsets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                lengths_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int(num_valid),
                ctypes.c_int(self._B_cap),
                ctypes.c_int(self._T_cap),
                ctypes.c_int(self.n),
                ctypes.c_int(self.N_data),
                ctypes.c_int(self.W_u64),
                self.data_u64_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                self._A_pad_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                self._b_info_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                self._tiles_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                self._t_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if p_off >= 0:
                break
            self._T_cap *= 2
            self._alloc_host_buffers()

        while p_off > self._S_cap:
            self._S_cap *= 2

        return K_total, self._S_cap

    def value_and_grad(self, params, ops):
        params_j = jnp.asarray(params, dtype=jnp.float64)
        if len(ops) == 0:
            return jnp.asarray(0.0, dtype=jnp.float64), jnp.zeros_like(params_j)

        if self.use_hip:
            K_total, s_cap = self._pack_ops_hip(ops)
            vg_fn = self._get_hip_vg(s_cap)
            return vg_fn(
                params_j,
                jnp.asarray(self._A_pad_np),
                jnp.asarray(self._tiles_np),
                jnp.asarray(self._b_info_np),
                jnp.asarray(self._t_np),
                jnp.asarray(1.0 / K_total, dtype=jnp.float64),
            )

        return self._value_and_grad_jax(params_j, ops)

    @staticmethod
    @jax.jit
    def _group_sq(th1, Th, pi, A, t, keep, sig):
        c = jnp.cos(2.0 * jnp.einsum("sr,lbr->lbs", sig, th1[:, A]))
        ThA = Th[:, A, :] * keep[None, :, None, :]
        pr = jnp.prod(jnp.cos(2.0 * jnp.einsum("sr,lbrn->lbsn", sig, ThA)), axis=-1)
        z = (c * pr).mean(-1)
        return jnp.sum((pi @ z - t) ** 2)

    def _branch_angles(self, params):
        th = self.W @ params.reshape(self.L, self.G)
        th1 = th[:, :self.n]
        Th = jnp.zeros((self.L, self.n, self.n), dtype=jnp.float64)
        if len(self.edges):
            Th = Th.at[:, self.e0, self.e1].set(th[:, self.n:])
        return th1, Th + Th.transpose(0, 2, 1)

    def _loss_groups(self, params, groups, K):
        th1, Th = self._branch_angles(params)
        return sum(self._group_sq(th1, Th, self.pi, *g) for g in groups) / K

    def _value_and_grad_jax(self, params_j, ops):
        K_total = len(ops)
        by_r = {}
        for op in ops:
            r = len(op)
            if 0 < r <= R_MAX:
                by_r.setdefault(r, []).append(np.asarray(op, dtype=np.int32))
        if not by_r:
            return jnp.asarray(0.0, dtype=jnp.float64), jnp.zeros_like(params_j)

        groups = []
        data_pm1_np = np.asarray(self.data_pm1)
        for r, As in by_r.items():
            A = np.stack(As, axis=0)
            t = np.prod(data_pm1_np[:, A], axis=-1).mean(axis=0)
            mask = np.zeros((len(A), self.n), dtype=np.float64)
            np.put_along_axis(mask, A, 1.0, axis=1)
            groups.append((jnp.asarray(A), jnp.asarray(t), jnp.asarray(1.0 - mask), _SIGN_TABLES[r]))
        return jax.value_and_grad(self._loss_groups)(params_j, groups, K_total)


# ---------------------------------------------------------------------------
# Multi-Init Closed-Form Trainer & Evaluator (used by `cf_multi.py`)
# ---------------------------------------------------------------------------

def _moments_u64(x_u64, n_rows, A_pad):
    xored = jnp.bitwise_xor.reduce(x_u64[A_pad], axis=1)
    ones = jax.lax.population_count(xored).sum(axis=-1, dtype=jnp.int32)
    return (n_rows - 2 * ones).astype(jnp.float64) / n_rows


class MultiInitTrainer:
    r"""Lockstep GPU trainer and evaluator for S seeds x M_per_seed IQP-QCBM initializations."""

    def __init__(self, n: int, a: int, edges, xtr_u64, n_tr: int, xte_u64, n_te: int,
                 sigmas, lr: float = 1e-3, seed=1234, seeds=None, k_train: int = 2000,
                 rmax: int = 10, keval: int = 10_000, chunk: int = 50):
        load_hip_ext()
        self.n = int(n)
        self.a = int(a)
        self.L = 1 << self.a
        self.edges = np.asarray(edges, dtype=np.int32)
        self.G = self.n + len(self.edges)
        self.e0 = jnp.asarray(self.edges[:, 0], dtype=jnp.int32)
        self.e1 = jnp.asarray(self.edges[:, 1], dtype=jnp.int32)
        self.W = _walsh_matrix(self.a)
        self.W_j = jnp.asarray(self.W, dtype=jnp.float64)
        self.xtr_u64 = xtr_u64
        self.n_tr = int(n_tr)
        self.xte_u64 = xte_u64
        self.n_te = int(n_te)
        self.sigmas = list(sigmas)
        self.lr = lr if callable(lr) else float(lr)   # a callable is an optax schedule
        if seeds is not None:
            self.seeds = [int(s) for s in seeds]
        elif isinstance(seed, (list, tuple, np.ndarray)):
            self.seeds = [int(s) for s in seed]
        else:
            self.seeds = [int(seed)]
        self.seed = self.seeds[0]
        self.num_seeds = len(self.seeds)
        self.k_train = int(k_train)
        self.rmax = int(rmax)
        self.keval = int(keval)
        self.chunk = int(chunk)
        self.opt = optax.adam(self.lr)

        # Stratified order weights and counts
        w = np.zeros(self.rmax + 1, dtype=np.float64)
        for s_ in self.sigmas:
            p = 0.5 * (1.0 - math.exp(-1.0 / (2.0 * s_ ** 2)))
            w += [math.comb(self.n, r) * p ** r * (1.0 - p) ** (self.n - r) for r in range(self.rmax + 1)]
        self.wbar = w / len(self.sigmas)
        self.kr = {r: max(8, int(round(self.k_train * self.wbar[r] / self.wbar[1:].sum())))
                   for r in range(1, self.rmax + 1)}

        tiles_list, b_info_list, w_of_b_list = [], [], []
        b_idx, p_off = 0, 0
        for r, k in self.kr.items():
            S = 1 << (r - 1)
            for _ in range(k):
                b_info_list.append((r, p_off))
                w_of_b_list.append(self.wbar[r] / k)
                for s_start in range(0, S, TILE_S):
                    s_cnt = min(TILE_S, S - s_start)
                    tiles_list.append((b_idx, r, s_start, s_cnt, p_off))
                    p_off += s_cnt
                b_idx += 1

        self.b_tot = b_idx
        self.s_tot = p_off
        self.tiles_j = jnp.asarray(np.array(tiles_list, dtype=np.int32))
        self.b_info_j = jnp.asarray(np.array(b_info_list, dtype=np.int32))
        self.w_of_b_j = jnp.asarray(w_of_b_list, dtype=jnp.float64)

        # 128-bit PCG64 state + jump table per seed
        mask_128 = (1 << 128) - 1
        mul_128 = (2549297995355413924 << 64) | 4865540595714422341
        mask_64 = (1 << 64) - 1
        pcg_inits = []
        j_tabs = []
        for sd in self.seeds:
            rs_init = np.random.default_rng(sd + 1)
            st_dict = rs_init.bit_generator.state["state"]
            s0_128, inc_128 = int(st_dict["state"]), int(st_dict["inc"])
            pcg_inits.append([
                s0_128 & mask_64, s0_128 >> 64,
                inc_128 & mask_64, inc_128 >> 64
            ])
            j_tab_list = []
            cur_mul, cur_plus = 1, 0
            for _ in range(self.n):
                cur_mul = (cur_mul * mul_128) & mask_128
                cur_plus = (cur_plus * mul_128 + inc_128) & mask_128
                j_tab_list.append([cur_mul & mask_64, cur_mul >> 64, cur_plus & mask_64, cur_plus >> 64])
            j_tabs.append(j_tab_list)

        if self.num_seeds == 1:
            self.pcg_init_j = jnp.asarray(np.array(pcg_inits[0], dtype=np.uint64))
            self.j_tab_j = jnp.asarray(np.array(j_tabs[0], dtype=np.uint64))
        else:
            self.pcg_init_j = jnp.asarray(np.array(pcg_inits, dtype=np.uint64))
            self.j_tab_j = jnp.asarray(np.array(j_tabs, dtype=np.uint64))

        self._ev_cache = None
        self._build_Compiled_steps()

    def _build_Compiled_steps(self):
        n, L = self.n, self.L
        e0, e1 = self.e0, self.e1
        W_j = self.W_j
        tiles_j = self.tiles_j
        b_info_j = self.b_info_j
        w_of_b_j = self.w_of_b_j
        s_tot = self.s_tot
        b_tot = self.b_tot
        num_seeds = self.num_seeds
        pcg_init_j = self.pcg_init_j
        j_tab_j = self.j_tab_j
        xtr_u64 = self.xtr_u64
        n_tr = self.n_tr
        opt = self.opt
        chunk = self.chunk

        def gpu_sample_moments(step_idx):
            if num_seeds == 1:
                a_type = jax.ShapeDtypeStruct((b_tot, 10), jnp.int32)
                t_type = jax.ShapeDtypeStruct((b_tot,), jnp.float64)
            else:
                a_type = jax.ShapeDtypeStruct((num_seeds, b_tot, 10), jnp.int32)
                t_type = jax.ShapeDtypeStruct((num_seeds, b_tot), jnp.float64)
            return jax.ffi.ffi_call("cf_sample_moments", (a_type, t_type))(
                pcg_init_j, j_tab_j, jnp.asarray(step_idx, dtype=jnp.int32), b_info_j, xtr_u64, N_tr=int(n_tr))

        def loss_batch(cs, pijs, A_pad, t):
            M = cs.shape[0]
            th = 2.0 * jnp.einsum("lk,mkg->mlg", W_j, cs)
            th1 = th[:, :, :n]
            Th = jnp.zeros((M, L, n, n), dtype=jnp.float64).at[:, :, e0, e1].set(th[:, :, n:])
            Th = Th + Th.transpose(0, 1, 3, 2)
            z = _hip_mix_eval(th1, Th, pijs, A_pad, tiles_j, b_info_j, s_tot)
            if num_seeds == 1:
                losses = jnp.sum(w_of_b_j[None, :] * (z - t[None, :]) ** 2, axis=-1)
            else:
                m_per_seed = M // num_seeds
                diff = z.reshape(num_seeds, m_per_seed, b_tot) - t[:, None, :]
                losses = jnp.sum(w_of_b_j[None, None, :] * (diff ** 2), axis=-1).reshape(M)
            return jnp.sum(losses), losses

        vg_batch = jax.value_and_grad(loss_batch, has_aux=True)

        @jax.jit
        def update_step(cs, states, step_idx, pijs):
            A_pad, t = gpu_sample_moments(step_idx)
            (_, ls), gs = vg_batch(cs, pijs, A_pad, t)
            us, states = jax.vmap(opt.update)(gs, states, cs)
            return jax.vmap(optax.apply_updates)(cs, us), states, ls

        @jax.jit
        def run_chunk(cs, states, start_step, pijs):
            def body(carry, offset):
                c_cur, st_cur = carry
                c_next, st_next, ls = update_step(c_cur, st_cur, start_step + offset, pijs)
                return (c_next, st_next), ls
            (cs_out, st_out), ls_chunk = jax.lax.scan(body, (cs, states), jnp.arange(chunk, dtype=jnp.int32))
            return cs_out, st_out, ls_chunk

        @jax.jit
        def eval_batch_jit(ths, pis, ev_A_pad, ev_tiles, ev_t):
            M = ths.shape[0]
            th2 = 2.0 * ths
            th1 = th2[:, :, :n]
            Th = jnp.zeros((M, L, n, n), dtype=jnp.float64).at[:, :, e0, e1].set(th2[:, :, n:])
            Th = Th + Th.transpose(0, 1, 3, 2)
            z = _hip_eval_only(th1, Th, pis, ev_A_pad, ev_tiles)
            return (z - ev_t[None, :]) ** 2

        self.update_step = update_step
        self.run_chunk = run_chunk
        self._eval_batch_jit = eval_batch_jit

    def _ensure_ev(self):
        if self._ev_cache is None:
            bw = []
            for mb in range(1, 7):
                mask = np.random.default_rng(1000 + mb).random((self.keval, self.n)) < mb / self.n
                lens = mask.sum(axis=1)
                g = {}
                for r in range(1, self.rmax + 1):
                    sel = (lens == r)
                    if np.any(sel):
                        g[r] = np.nonzero(mask[sel])[1].astype(np.int32).reshape(-1, r)
                bw.append(g)

            a_pads, tiles, offsets = [], [], [0]
            b_idx = 0
            for g in bw:
                for r, A in g.items():
                    k = len(A)
                    a_pads.append(np.pad(A, ((0, 0), (0, 10 - r)), constant_values=self.n))
                    S = 1 << (r - 1)
                    for local_b in range(k):
                        for s_start in range(0, S, TILE_S):
                            s_cnt = min(TILE_S, S - s_start)
                            tiles.append((b_idx + local_b, r, s_start, s_cnt, 0))
                    b_idx += k
                offsets.append(b_idx)
            ev_A_pad = jnp.asarray(np.concatenate(a_pads, axis=0, dtype=np.int32))
            ev_tiles = jnp.asarray(np.array(tiles, dtype=np.int32))
            ev_t = jax.jit(lambda a_: _moments_u64(self.xte_u64, self.n_te, a_))(ev_A_pad)
            self._ev_cache = (ev_A_pad, ev_tiles, ev_t, offsets)
        return self._ev_cache

    def test_mmd2_batch(self, ths, pis):
        ev_A_pad, ev_tiles, ev_t, offsets = self._ensure_ev()
        sq = np.asarray(self._eval_batch_jit(jnp.asarray(ths), jnp.asarray(pis), ev_A_pad, ev_tiles, ev_t))
        per_all = [[float(sq[m, offsets[mb]:offsets[mb + 1]].sum()) / self.keval for mb in range(6)]
                   for m in range(len(ths))]
        return [float(np.mean(p)) for p in per_all], per_all

    def test_mmd2(self, th, pi):
        ms, ps = self.test_mmd2_batch(np.asarray(th)[None], np.asarray(pi)[None])
        return ms[0], ps[0]
