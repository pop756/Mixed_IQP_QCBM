"""Lean closed-form IQP trainer: one HIP library (no JAX/XLA on the training path), driven through ctypes.

Same computation as iqp_hip_cf.MultiInitTrainer (same PCG64 subset streams, same closed-form loss, same Adam), but
- models are processed in chunks: the dense symmetric (L, n, n) branch-angle matrices and their gradients only exist
  for the (model, branch) pairs of the current chunk (the one-body angles live on the diagonal); a large model can be
  split into branch parts (the Walsh map factorizes over branch bits, so gradient folds accumulate per part),
- forward: sign-pattern products accumulated in registers; its epilogue also prepares the backward's per-pattern
  coefficients (quad-transformed) and the one-body sums; backward: tan(b +- D) = (sin 2b +- sin 2D)/(cos 2b + cos 2D)
  shares one reciprocal per +-D pair, Gray-ordered top patterns use rotations instead of sincos, FP64 HW atomics,
- build / fold / update are LDS-tiled transposes over upper-triangle tile pairs with an in-register FWHT,
- offload levels (chosen by train._plan from a device-memory budget): 0 = all state on the device; 2 = compiled angles
  and Adam moments in pinned host memory, staged per chunk (two staging buffers) and updated by a PCIe-streaming Adam
  kernel on its own CUs plus host threads (share CPU_FRAC of each chunk), overlapped with the next chunk's kernels,
- the next step's subsets are drawn on a side stream; device memory is exactly what is hipMalloc'ed here (no allocator
  pools, no JAX runtime on the GPU, no unused runtime scratch reservation).
"""
import ctypes, hashlib, os, subprocess, tempfile
import numpy as np

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <omp.h>

#define DEVI __device__ __forceinline__
#define ASTR 10
#define FWD_NT 256
#define FWD_JC 128
#define BWD_NT 128

DEVI double fast_cos(double x) {
    constexpr double ONE_OVER_PI = 0.31830988618379067154;
    constexpr double PI_HI = 3.14159265358979311600;
    constexpr double PI_LO = 1.22464679914735320717e-16;
    double k = rint(x * ONE_OVER_PI);
    double r = fma(k, -PI_LO, fma(k, -PI_HI, x));
    double r2 = r * r;
    double c = fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2,
               -1.5619206968586226e-16, 4.7794773323873853e-14),
               -1.1470745597729725e-11), 2.0876756987868099e-09),
               -2.7557319223985893e-07), 2.4801587301587302e-05),
               -1.3888888888888889e-03), 4.1666666666666664e-02), -0.5), 1.0);
    return (((int)k) & 1) ? -c : c;
}

DEVI void fast_sincos(double x, double* s_out, double* c_out) {
    constexpr double TWO_OVER_PI = 0.63661977236758134308;
    constexpr double PIO2_HI = 1.57079632679489655800;
    constexpr double PIO2_LO = 6.12323399573676603587e-17;
    double k = rint(x * TWO_OVER_PI);
    double r = fma(k, -PIO2_LO, fma(k, -PIO2_HI, x));
    double r2 = r * r;
    double r3 = r * r2;
    double s = fma(r3, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2,
               -7.6471637318198164e-13, 1.6059043836821613e-10),
               -2.5052108385441720e-08), 2.7557319223985893e-06),
               -1.9841269841269841e-04), 8.3333333333333332e-03),
               -1.6666666666666666e-01), r);
    double c = fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2, fma(r2,
               -1.1470745597729725e-11, 2.0876756987868099e-09),
               -2.7557319223985893e-07), 2.4801587301587302e-05),
               -1.3888888888888889e-03), 4.1666666666666664e-02), -0.5), 1.0);
    int ki = (int)k;
    bool odd = (ki & 1) != 0;
    double cv = odd ? s : c;
    double sv = odd ? c : s;
    *c_out = ((ki + 1) & 2) ? -cv : cv;
    *s_out = (ki & 2) ? -sv : sv;
}

DEVI double fast_rcp(double x) {
    double r = __builtin_amdgcn_rcp(x);
    double e = fma(-x, r, 1.0);
    r = fma(r, e, r);
    e = fma(-x, r, 1.0);
    return fma(r, e, r);
}

DEVI double fast_tan(double x) {
    constexpr double TWO_OVER_PI = 0.63661977236758134308;
    constexpr double PIO2_HI = 1.57079632679489655800;
    constexpr double PIO2_LO = 6.12323399573676603587e-17;
    double k = rint(x * TWO_OVER_PI);
    double r = fma(k, -PIO2_LO, fma(k, -PIO2_HI, x));
    double r2 = r * r;
    double num = r * fma(r2, fma(r2, fma(r2, r2 - 990.0, 135135.0), -4729725.0), 34459425.0);
    double den = fma(r2, fma(r2, fma(r2, 45.0 * r2 - 13860.0, 945945.0), -16216200.0), 34459425.0);
    bool odd = (((int)k) & 1) != 0;
    double a = odd ? -den : num;
    double b = odd ? num : den;
    return a / b;
}

// ------------------------------------------------------------------------------------------------
// PCG64 jump-ahead subset sampler + bit-packed moments (identical semantics to iqp_hip_cf)
// ------------------------------------------------------------------------------------------------
DEVI __uint128_t pcg_advance_lcg_128(__uint128_t state, uint64_t delta, __uint128_t cur_mult, __uint128_t cur_plus) {
    __uint128_t acc_mult = 1;
    __uint128_t acc_plus = 0;
    while (delta > 0) {
        if (delta & 1ULL) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta >>= 1;
    }
    return acc_mult * state + acc_plus;
}

DEVI uint64_t pcg_xsl_rr(__uint128_t s) {
    uint64_t xorshifted = uint64_t(s >> 64) ^ uint64_t(s);
    uint32_t rot = uint32_t(s >> 122);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 63U));
}

// PCG64 state of every (seed, subset row) of a step: rowst[(s * B + b) * 2 + {0, 1}]
__global__ __launch_bounds__(256)
void rowstate_kernel(const uint64_t* __restrict__ pcg_init, int step_i, uint64_t* __restrict__ rowst, int B_tot, int S,
                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * B_tot) return;
    int s = idx / B_tot, b = idx - s * B_tot;
    const uint64_t* pcg_s = pcg_init + (size_t)s * 4;
    __uint128_t state0 = (__uint128_t(pcg_s[1]) << 64) | pcg_s[0];
    __uint128_t inc    = (__uint128_t(pcg_s[3]) << 64) | pcg_s[2];
    constexpr __uint128_t MUL = (__uint128_t(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
    uint64_t row_offset = ((uint64_t)step_i * (uint64_t)B_tot + (uint64_t)b) * (uint64_t)n;
    __uint128_t s_row = pcg_advance_lcg_128(state0, row_offset, MUL, inc);
    rowst[2 * (size_t)idx] = uint64_t(s_row);
    rowst[2 * (size_t)idx + 1] = uint64_t(s_row >> 64);
}

__global__ __launch_bounds__(512)
void sample_kernel(const uint64_t* __restrict__ rowst, const uint64_t* __restrict__ j_tab, int step_i,
                   const int* __restrict__ b_info, const uint64_t* __restrict__ xtr_u64,
                   int* __restrict__ A_pad, double* __restrict__ t_out, int B_tot, int n, int N_tr, int W_u64) {
    __shared__ uint64_t s_wave_min[8];
    __shared__ uint64_t s_global_min;
    __shared__ int s_A[12];
    __shared__ int s_pop[8];

    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int wave_id = tid >> 6;
    int lane = tid & 63;
    int r = b_info[b * 2];

    const uint64_t* j_tab_s = j_tab + (size_t)s * n * 4;
    int* A_pad_sb = A_pad + ((size_t)s * B_tot + b) * ASTR;
    const uint64_t* rs = rowst + 2 * ((size_t)s * B_tot + b);
    __uint128_t s_row = (__uint128_t(rs[1]) << 64) | rs[0];

    uint64_t my_val0 = ~0ULL;
    uint64_t my_val1 = ~0ULL;
    if (tid < n) {
        const uint64_t* jt = j_tab_s + (size_t)tid * 4;
        __uint128_t mul  = (__uint128_t(jt[1]) << 64) | jt[0];
        __uint128_t plus = (__uint128_t(jt[3]) << 64) | jt[2];
        __uint128_t st = s_row * mul + plus;
        my_val0 = (pcg_xsl_rr(st) & ~0x7FFULL) | (uint64_t)tid;
    }
    if (tid + 512 < n) {
        int j1 = tid + 512;
        const uint64_t* jt = j_tab_s + (size_t)j1 * 4;
        __uint128_t mul  = (__uint128_t(jt[1]) << 64) | jt[0];
        __uint128_t plus = (__uint128_t(jt[3]) << 64) | jt[2];
        __uint128_t st = s_row * mul + plus;
        my_val1 = (pcg_xsl_rr(st) & ~0x7FFULL) | (uint64_t)j1;
    }

    for (int k = 0; k < r; ++k) {
        uint64_t v = (my_val0 < my_val1) ? my_val0 : my_val1;
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            uint64_t other = __shfl_xor(v, offset, 64);
            v = (other < v) ? other : v;
        }
        if (lane == 0) {
            s_wave_min[wave_id] = v;
        }
        __syncthreads();
        if (tid < 8) {
            uint64_t wv = s_wave_min[tid];
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                uint64_t other = __shfl_xor(wv, offset, 64);
                wv = (other < wv) ? other : wv;
            }
            if (tid == 0) {
                s_global_min = wv;
                int best_j = (int)(wv & 0x7FFULL);
                s_A[k] = best_j;
                A_pad_sb[k] = best_j;
            }
        }
        __syncthreads();
        uint64_t gmin = s_global_min;
        if (my_val0 == gmin) my_val0 = ~0ULL;
        if (my_val1 == gmin) my_val1 = ~0ULL;
    }
    if (tid >= r && tid < ASTR) {
        s_A[tid] = n;
        A_pad_sb[tid] = n;
    }

    int local_ones = 0;
    for (int w = tid; w < W_u64; w += 512) {
        uint64_t xored = 0ULL;
        for (int k = 0; k < r; ++k) {
            xored ^= xtr_u64[(size_t)s_A[k] * W_u64 + w];
        }
        local_ones += __popcll(xored);
    }
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_ones += __shfl_xor(local_ones, offset, 64);
    }
    if (lane == 0) {
        s_pop[wave_id] = local_ones;
    }
    __syncthreads();
    if (tid == 0) {
        int total_ones = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) total_ones += s_pop[w];
        t_out[(size_t)s * B_tot + b] = (double)(N_tr - 2 * total_ones) / (double)N_tr;
    }
}

// one wave per (seed, subset row): the r smallest keys (pcg & ~0x7FF) | j over j < min(n, 1024), ascending (as
// iqp_hip_cf's sampler), then the bit-packed moment
__global__ __launch_bounds__(256)
void sample_wave_kernel(const uint64_t* __restrict__ rowst, const uint64_t* __restrict__ j_tab,
                        const int* __restrict__ b_info, const uint64_t* __restrict__ xtr_u64,
                        int* __restrict__ A_pad, double* __restrict__ t_out, int B_tot, int S, int n, int N_tr,
                        int W_u64) {
    int item = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (item >= S * B_tot) return;
    int s = item / B_tot, b = item - s * B_tot;
    int lane = threadIdx.x & 63;
    int r = b_info[b * 2];
    const uint64_t* j_tab_s = j_tab + (size_t)s * n * 4;
    int* A_pad_sb = A_pad + ((size_t)s * B_tot + b) * ASTR;
    const uint64_t* rs = rowst + 2 * ((size_t)s * B_tot + b);
    __uint128_t s_row = (__uint128_t(rs[1]) << 64) | rs[0];
    int nk = (n < 1024) ? n : 1024;
    uint64_t key[16];
    #pragma unroll
    for (int q = 0; q < 16; ++q) {
        int j = lane + 64 * q;
        key[q] = ~0ULL;
        if (j < nk) {
            const uint64_t* jt = j_tab_s + (size_t)j * 4;
            __uint128_t mul  = (__uint128_t(jt[1]) << 64) | jt[0];
            __uint128_t plus = (__uint128_t(jt[3]) << 64) | jt[2];
            key[q] = (pcg_xsl_rr(s_row * mul + plus) & ~0x7FFULL) | (uint64_t)j;
        }
    }
    int Ak[ASTR];
    #pragma unroll
    for (int k = 0; k < ASTR; ++k) Ak[k] = n;
    #pragma unroll
    for (int k = 0; k < ASTR; ++k) {
        if (k < r) {
            uint64_t v = key[0];
            #pragma unroll
            for (int q = 1; q < 16; ++q) v = (key[q] < v) ? key[q] : v;
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) {
                uint64_t o = __shfl_xor(v, off, 64);
                v = (o < v) ? o : v;
            }
            Ak[k] = (int)(v & 0x7FFULL);
            #pragma unroll
            for (int q = 0; q < 16; ++q) if (key[q] == v) key[q] = ~0ULL;
        }
    }
    if (lane < ASTR) {
        int a = Ak[0];
        #pragma unroll
        for (int k = 1; k < ASTR; ++k) a = (lane == k) ? Ak[k] : a;
        A_pad_sb[lane] = a;
    }
    int ones = 0;
    for (int w = lane; w < W_u64; w += 64) {
        uint64_t xored = 0ULL;
        #pragma unroll
        for (int k = 0; k < ASTR; ++k) if (k < r) xored ^= xtr_u64[(size_t)Ak[k] * W_u64 + w];
        ones += __popcll(xored);
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) ones += __shfl_xor(ones, off, 64);
    if (lane == 0) t_out[(size_t)s * B_tot + b] = (double)(N_tr - 2 * ones) / (double)N_tr;
}

// ------------------------------------------------------------------------------------------------
// Forward: z[ml][b] = (1/S) sum_s cos(a1(s)) P(s),  P(s) = prod_{j not in A} cos(sum_u sigma_u(s) Th[A_u][j])
// ------------------------------------------------------------------------------------------------
DEVI void combo16(double* acc, double st, double ct, double S0, double C0, double S1, double C1,
                  double SA0, double CA0, double SA1, double CA1) {
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        double ca = (sub == 0 || sub == 3) ? CA0 : CA1;
        double sa = (sub == 0) ? SA0 : ((sub == 1) ? SA1 : ((sub == 2) ? -SA1 : -SA0));
        double cb = fma(-st, sa, ct * ca);
        double sb = fma( ct, sa, st * ca);
        acc[4 * sub + 0] *= fma(-sb, S0, cb * C0);
        acc[4 * sub + 1] *= fma(-sb, S1, cb * C1);
        acc[4 * sub + 2] *= fma( sb, S1, cb * C1);
        acc[4 * sub + 3] *= fma( sb, S0, cb * C0);
    }
}

DEVI double block_sum_256(double v, double* red) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
    __syncthreads();
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = v;
    __syncthreads();
    double tot = 0.0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int w = 0; w < FWD_NT / 64; ++w) tot += red[w];
    }
    return tot;
}

template <int R>
DEVI void fwd_terms(const double* x, double* acc) {
    // multiply acc[s] by cos(sum_u sigma_u(s) x[u]) for all S = 2^(R-1) sign patterns (sigma_0 = +1)
    if constexpr (R == 1) {
        acc[0] *= fast_cos(x[0]);
    } else if constexpr (R == 2) {
        acc[0] *= fast_cos(x[0] + x[1]);
        acc[1] *= fast_cos(x[0] - x[1]);
    } else if constexpr (R == 3) {
        double S0, C0, S1, C1, sb, cb;
        fast_sincos(x[1] + x[2], &S0, &C0);
        fast_sincos(x[1] - x[2], &S1, &C1);
        fast_sincos(x[0], &sb, &cb);
        acc[0] *= fma(-sb, S0, cb * C0);
        acc[1] *= fma(-sb, S1, cb * C1);
        acc[2] *= fma( sb, S1, cb * C1);
        acc[3] *= fma( sb, S0, cb * C0);
    } else if constexpr (R == 4) {
        double S0, C0, S1, C1;
        fast_sincos(x[2] + x[3], &S0, &C0);
        fast_sincos(x[2] - x[3], &S1, &C1);
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            double sb, cb;
            fast_sincos(t ? x[0] - x[1] : x[0] + x[1], &sb, &cb);
            acc[4 * t + 0] *= fma(-sb, S0, cb * C0);
            acc[4 * t + 1] *= fma(-sb, S1, cb * C1);
            acc[4 * t + 2] *= fma( sb, S1, cb * C1);
            acc[4 * t + 3] *= fma( sb, S0, cb * C0);
        }
    } else {
        double S0, C0, S1, C1, SA0, CA0, SA1, CA1;
        fast_sincos(x[R - 2] + x[R - 1], &S0, &C0);
        fast_sincos(x[R - 2] - x[R - 1], &S1, &C1);
        fast_sincos(x[R - 4] + x[R - 3], &SA0, &CA0);
        fast_sincos(x[R - 4] - x[R - 3], &SA1, &CA1);
        #pragma unroll
        for (int t = 0; t < (1 << (R - 5)); ++t) {
            double base = x[0];
            #pragma unroll
            for (int u = 1; u < R - 4; ++u) base += ((t >> (R - 5 - u)) & 1) ? -x[u] : x[u];
            double st, ct;
            fast_sincos(base, &st, &ct);
            combo16(acc + 16 * t, st, ct, S0, C0, S1, C1, SA0, CA0, SA1, CA1);
        }
    }
}

// wave-level forward (R <= 5): one wave per (b, ml); lanes stride over the columns j
template <int R>
DEVI void fwd_wave_r(const double* __restrict__ Thm, const int* __restrict__ Ab, int n, int lane,
                     double* __restrict__ Pout, double* __restrict__ zout, double* __restrict__ Dout) {
    constexpr int S = 1 << (R - 1);
    int a_l = (lane < R) ? Ab[lane] : 0;
    int sA[R];
    #pragma unroll
    for (int u = 0; u < R; ++u) sA[u] = __shfl(a_l, u, 64);
    double acc[S];
    #pragma unroll
    for (int k = 0; k < S; ++k) acc[k] = 1.0;
    for (int j = lane; j < n; j += 64) {
        bool act = true;
        #pragma unroll
        for (int u = 0; u < R; ++u) act = act && (j != sA[u]);
        if (!act) continue;
        double x[R];
        #pragma unroll
        for (int u = 0; u < R; ++u) x[u] = Thm[(size_t)sA[u] * n + j];
        fwd_terms<R>(x, acc);
    }
    #pragma unroll
    for (int off = 1; off < 64; off <<= 1) {
        #pragma unroll
        for (int k = 0; k < S; ++k) acc[k] *= __shfl_xor(acc[k], off, 64);
    }
    double th_l = (lane < R) ? Thm[(size_t)a_l * n + a_l] : 0.0;
    double p = acc[0];
    #pragma unroll
    for (int k = 1; k < S; ++k) p = (lane == k) ? acc[k] : p;
    double a1 = __shfl(th_l, 0, 64);
    #pragma unroll
    for (int u = 1; u < R; ++u) {
        double tu = __shfl(th_l, u, 64);
        a1 += ((lane >> (R - 1 - u)) & 1) ? -tu : tu;
    }
    // z, and for the backward: q'(s) = -cos(a1) P(s) (quad-transformed for R >= 3, see bwd_cols) and the one-body
    // sums D_u = sum_s sigma_u(s) (-sin(a1) P(s))
    double zp = 0.0, qv = 0.0, dav = 0.0;
    if (lane < S) {
        double sa, ca;
        fast_sincos(a1, &sa, &ca);
        zp = ca * p;
        qv = -ca * p;
        dav = -sa * p;
    }
    if constexpr (R >= 3) {
        double pq = __shfl_xor(qv, 3, 64);
        int sm = lane & 3;
        if (lane < S) {
            if (sm == 0) Pout[lane] = qv + pq;
            else if (sm == 3) Pout[lane - 2] = pq - qv;
            else if (sm == 1) Pout[lane + 1] = qv + pq;
            else Pout[lane + 1] = pq - qv;
        }
    } else {
        if (lane < S) Pout[lane] = qv;
    }
    #pragma unroll
    for (int u = 0; u < R; ++u) {
        double v = (u > 0 && ((lane >> (R - 1 - u)) & 1)) ? -dav : dav;
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (lane == 0) Dout[u] = v;
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) zp += __shfl_xor(zp, off, 64);
    if (lane == 0) *zout = zp / (double)S;
}

// block-level forward (R >= 6): 256 threads = T top patterns x JG column groups, 16 sign patterns per thread;
// the per-column pair angles are computed once per column chunk and shared through LDS
template <int R>
DEVI void fwd_block_r(const double* __restrict__ Thm, const int* __restrict__ sA, const double* __restrict__ sth1,
                      int n, double* lds, double* __restrict__ Pout, double* __restrict__ zout,
                      double* __restrict__ Dout) {
    constexpr int S = 1 << (R - 1);
    constexpr int T = 1 << (R - 5);
    constexpr int JG = FWD_NT / T;
    constexpr int NX = R - 4;
    constexpr int JC = FWD_JC;
    double* xs = lds;
    double* pr = lds + NX * JC;
    int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    int tau = tid % T, jg = tid / T;
    double sg[NX];
    sg[0] = 1.0;
    #pragma unroll
    for (int u = 1; u < NX; ++u) sg[u] = ((tau >> (R - 5 - u)) & 1) ? -1.0 : 1.0;
    double acc[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 1.0;
    int half = tid / JC;           // wave-uniform: 0 -> pair (R-2, R-1) + even xs, 1 -> pair (R-4, R-3) + odd xs
    int jt = tid - half * JC;
    for (int j0 = 0; j0 < n; j0 += JC) {
        __syncthreads();
        {
            int j = j0 + jt;
            bool act = (j < n);
            #pragma unroll
            for (int u = 0; u < R; ++u) act = act && (j != sA[u]);
            int jj = (j < n) ? j : n - 1;
            if (half == 0) {
                double xa = Thm[(size_t)sA[R - 2] * n + jj];
                double xb = Thm[(size_t)sA[R - 1] * n + jj];
                xa = act ? xa : 0.0; xb = act ? xb : 0.0;
                double S0, C0, S1, C1;
                fast_sincos(xa + xb, &S0, &C0);
                fast_sincos(xa - xb, &S1, &C1);
                pr[0 * JC + jt] = S0; pr[1 * JC + jt] = C0; pr[2 * JC + jt] = S1; pr[3 * JC + jt] = C1;
                #pragma unroll
                for (int u = 0; u < NX; u += 2) {
                    double v = Thm[(size_t)sA[u] * n + jj];
                    xs[u * JC + jt] = act ? v : 0.0;
                }
            } else {
                double xa = Thm[(size_t)sA[R - 4] * n + jj];
                double xb = Thm[(size_t)sA[R - 3] * n + jj];
                xa = act ? xa : 0.0; xb = act ? xb : 0.0;
                double S0, C0, S1, C1;
                fast_sincos(xa + xb, &S0, &C0);
                fast_sincos(xa - xb, &S1, &C1);
                pr[4 * JC + jt] = S0; pr[5 * JC + jt] = C0; pr[6 * JC + jt] = S1; pr[7 * JC + jt] = C1;
                #pragma unroll
                for (int u = 1; u < NX; u += 2) {
                    double v = Thm[(size_t)sA[u] * n + jj];
                    xs[u * JC + jt] = act ? v : 0.0;
                }
            }
        }
        __syncthreads();
        int jn = min(JC, n - j0);
        for (int jl = jg; jl < jn; jl += JG) {
            double base = xs[jl];
            #pragma unroll
            for (int u = 1; u < NX; ++u) base = fma(sg[u], xs[u * JC + jl], base);
            double st, ct;
            fast_sincos(base, &st, &ct);
            combo16(acc, st, ct, pr[0 * JC + jl], pr[1 * JC + jl], pr[2 * JC + jl], pr[3 * JC + jl],
                    pr[4 * JC + jl], pr[5 * JC + jl], pr[6 * JC + jl], pr[7 * JC + jl]);
        }
    }
    // product over column groups: within the wave (lanes with equal tau), then across the 4 waves
    #pragma unroll
    for (int off = T; off < 64; off <<= 1) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] *= __shfl_xor(acc[k], off, 64);
    }
    __syncthreads();
    if (wave >= 2 && lane < T) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) lds[(wave - 2) * S + lane * 16 + k] = acc[k];
    }
    __syncthreads();
    if (wave < 2 && lane < T) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] *= lds[wave * S + lane * 16 + k];
    }
    __syncthreads();
    if (wave == 1 && lane < T) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) lds[lane * 16 + k] = acc[k];
    }
    __syncthreads();
    if (wave == 0) {
        double zp = 0.0;
        double D[R];
        #pragma unroll
        for (int u = 0; u < R; ++u) D[u] = 0.0;
        if (lane < T) {
            #pragma unroll
            for (int qd = 0; qd < 4; ++qd) {
                double qv[4];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int k = 4 * qd + e;
                    int s = lane * 16 + k;
                    double p = acc[k] * lds[s];
                    double a1 = sth1[0];
                    #pragma unroll
                    for (int u = 1; u < R; ++u) a1 += ((s >> (R - 1 - u)) & 1) ? -sth1[u] : sth1[u];
                    double sa, ca;
                    fast_sincos(a1, &sa, &ca);
                    zp += ca * p;
                    qv[e] = -ca * p;
                    double da = -sa * p;
                    D[0] += da;
                    #pragma unroll
                    for (int u = 1; u < R; ++u) D[u] += ((s >> (R - 1 - u)) & 1) ? -da : da;
                }
                int s0 = lane * 16 + 4 * qd;
                Pout[s0 + 0] = qv[0] + qv[3];
                Pout[s0 + 1] = qv[0] - qv[3];
                Pout[s0 + 2] = qv[1] + qv[2];
                Pout[s0 + 3] = qv[1] - qv[2];
            }
        }
        #pragma unroll
        for (int u = 0; u < R; ++u) {
            double v = D[u];
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0) Dout[u] = v;
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) zp += __shfl_xor(zp, off, 64);
        if (lane == 0) *zout = zp / (double)S;
    }
}

// blocks [0, nb_big): (ml, b) with r >= 6, b descending; blocks [nb_big, ...): 4 waves = 4 (b, ml) items, r <= 5
__global__ __launch_bounds__(FWD_NT) __attribute__((amdgpu_waves_per_eu(4, 8)))
void fwd_kernel(const double* __restrict__ Th, const int* __restrict__ A_pad, const int* __restrict__ b_info,
                double* __restrict__ z, double* __restrict__ P, int n, int L, int B, int S_tot, int Mps, int m0,
                int nml, int B6, int nb_big, int Lp, int l0, double* __restrict__ Dq) {
    // ml: (model, branch) pair of this launch (Lp branches per model starting at branch l0); gml: its index in P / z
    __shared__ double lds[1792];
    __shared__ int sA[ASTR];
    __shared__ double sth1[ASTR];
    size_t nn = (size_t)n * n;
    int blk = blockIdx.x;
    if (blk < nb_big) {
        int ml = blk % nml;
        int b = B - 1 - blk / nml;
        int r = b_info[2 * b];
        int p_off = b_info[2 * b + 1];
        int mc = ml / Lp;
        int gml = mc * L + l0 + ml % Lp;
        int m = m0 + mc;
        int seed = m / Mps;
        const double* Thm = Th + (size_t)ml * nn;
        if (threadIdx.x < r) {
            int a = A_pad[((size_t)seed * B + b) * ASTR + threadIdx.x];
            sA[threadIdx.x] = a;
            sth1[threadIdx.x] = Thm[(size_t)a * n + a];
        }
        __syncthreads();
        double* Pout = P + (size_t)gml * S_tot + p_off;
        double* zout = z + (size_t)gml * B + b;
        double* Dout = Dq + ((size_t)gml * B + b) * ASTR;
        switch (r) {
            case 6:  fwd_block_r<6>(Thm, sA, sth1, n, lds, Pout, zout, Dout); break;
            case 7:  fwd_block_r<7>(Thm, sA, sth1, n, lds, Pout, zout, Dout); break;
            case 8:  fwd_block_r<8>(Thm, sA, sth1, n, lds, Pout, zout, Dout); break;
            case 9:  fwd_block_r<9>(Thm, sA, sth1, n, lds, Pout, zout, Dout); break;
            default: fwd_block_r<10>(Thm, sA, sth1, n, lds, Pout, zout, Dout); break;
        }
    } else {
        int item = (blk - nb_big) * (FWD_NT / 64) + (threadIdx.x >> 6);
        if (item >= B6 * nml) return;
        int ml = item % nml;
        int b = B6 - 1 - item / nml;
        int r = b_info[2 * b];
        int p_off = b_info[2 * b + 1];
        int mc = ml / Lp;
        int gml = mc * L + l0 + ml % Lp;
        int m = m0 + mc;
        int seed = m / Mps;
        const double* Thm = Th + (size_t)ml * nn;
        const int* Ab = A_pad + ((size_t)seed * B + b) * ASTR;
        double* Pout = P + (size_t)gml * S_tot + p_off;
        double* zout = z + (size_t)gml * B + b;
        double* Dout = Dq + ((size_t)gml * B + b) * ASTR;
        int lane = threadIdx.x & 63;
        switch (r) {
            case 1:  fwd_wave_r<1>(Thm, Ab, n, lane, Pout, zout, Dout); break;
            case 2:  fwd_wave_r<2>(Thm, Ab, n, lane, Pout, zout, Dout); break;
            case 3:  fwd_wave_r<3>(Thm, Ab, n, lane, Pout, zout, Dout); break;
            case 4:  fwd_wave_r<4>(Thm, Ab, n, lane, Pout, zout, Dout); break;
            default: fwd_wave_r<5>(Thm, Ab, n, lane, Pout, zout, Dout); break;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Loss + dL/dz per model of the chunk (grid: models x subset blocks; loss row zeroed per step)
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(256)
void lossdz_kernel(const double* __restrict__ z, const double* __restrict__ pi, const double* __restrict__ t,
                   const double* __restrict__ w_of_b, double* __restrict__ dz, double* __restrict__ loss_out,
                   int L, int B, int Mps, int m0) {
    __shared__ double red[8];
    int mc = blockIdx.y;
    int m = m0 + mc;
    int seed = m / Mps;
    int b = blockIdx.x * 256 + threadIdx.x;
    double ls = 0.0;
    if (b < B) {
        double zm = 0.0;
        for (int l = 0; l < L; ++l) zm += pi[(size_t)m * L + l] * z[(size_t)(mc * L + l) * B + b];
        double diff = zm - t[(size_t)seed * B + b];
        double w = w_of_b[b];
        ls = w * (diff * diff);
        dz[(size_t)mc * B + b] = w * (2.0 * diff);
    }
    double tot = block_sum_256(ls, red);
    if (threadIdx.x == 0) unsafeAtomicAdd(&loss_out[m], tot);
}

// Trainable branch weights: dL/dpi_l = sum_b dz_b z_{l,b} per model of the chunk (grid: L x models of the chunk)
__global__ __launch_bounds__(256)
void pigrad_kernel(const double* __restrict__ z, const double* __restrict__ dz, double* __restrict__ gpi,
                   int L, int B, int m0) {
    __shared__ double red[8];
    int l = blockIdx.x, mc = blockIdx.y;
    double s = 0.0;
    for (int b = threadIdx.x; b < B; b += 256) s += dz[(size_t)mc * B + b] * z[(size_t)(mc * L + l) * B + b];
    double tot = block_sum_256(s, red);
    if (threadIdx.x == 0) gpi[(size_t)(m0 + mc) * L + l] = tot;
}

// pi = softmax(beta): Adam (optax) on the logits beta with dL/dbeta_l = pi_l (g_l - sum_k pi_k g_k), then pi from
// the updated beta (one thread per model)
__global__ __launch_bounds__(64)
void piadam_kernel(double* __restrict__ beta, double* __restrict__ mb, double* __restrict__ vb,
                   const double* __restrict__ gpi, double* __restrict__ pi, int M, int L, double neg_lr, double b1,
                   double b2, double omb1, double omb2, double eps, double bc1, double bc2) {
    int m = blockIdx.x * 64 + threadIdx.x;
    if (m >= M) return;
    const double* g = gpi + (size_t)m * L;
    double* p = pi + (size_t)m * L;
    double gbar = 0.0;
    for (int l = 0; l < L; ++l) gbar += p[l] * g[l];
    double bmax = -1e300;
    for (int l = 0; l < L; ++l) {
        size_t i = (size_t)m * L + l;
        double gk = p[l] * (g[l] - gbar);
        double mn = omb1 * gk + b1 * mb[i];
        double vn = omb2 * (gk * gk) + b2 * vb[i];
        mb[i] = mn;
        vb[i] = vn;
        beta[i] += neg_lr * ((mn / bc1) / (sqrt(vn / bc2) + eps));
        bmax = (beta[i] > bmax) ? beta[i] : bmax;
    }
    double zs = 0.0;
    for (int l = 0; l < L; ++l) zs += exp(beta[(size_t)m * L + l] - bmax);
    for (int l = 0; l < L; ++l) p[l] = exp(beta[(size_t)m * L + l] - bmax) / zs;
}

// ------------------------------------------------------------------------------------------------
// Backward: d Th[A_u][j] += sum_s sigma_u(s) q(s) tan(sum_v sigma_v Th[A_v][j]),  q(s) = -w cos(a1(s)) P(s)
//           d th1[A_u]  (diagonal of dTh) += sum_s sigma_u(s) (-w sin(a1(s)) P(s))
// ------------------------------------------------------------------------------------------------
// sq holds, per quad of sign patterns (s = 4i .. 4i+3), (q0 + q3, q0 - q3, q1 + q2, q1 - q2) for R >= 3 (plain q
// otherwise). Within a quad the angles are b + D0, b + D1, b - D1, b - D0, and
// tan(b +- D) = (sin 2b +- sin 2D) / (cos 2b + cos 2D), so each +-D pair shares one denominator.
// one block of 16 sign patterns (base index blk) given (cos, sin) of twice its top angle
template <int R>
DEVI void bwd_blk(int blk, double c2t, double s2t, const double* __restrict__ sq, double* __restrict__ dx,
                  double C2D0, double S2D0, double C2D1, double S2D1, double CA0, double SA0, double CA1, double SA1) {
    double blk_all = 0.0, blk_3 = 0.0;
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        int idx = blk + (sub << 2);
        double ca = (sub == 0 || sub == 3) ? CA0 : CA1;
        double sa = (sub == 0) ? SA0 : ((sub == 1) ? SA1 : ((sub == 2) ? -SA1 : -SA0));
        double c2b = fma(-s2t, sa, c2t * ca);
        double s2b = fma( c2t, sa, s2t * ca);
        double d0 = c2b + C2D0;
        double d1 = c2b + C2D1;
        // a shared denominator that cancels to (near) zero would give inf * 0: that +-D pair is dropped
        bool k0 = fabs(d0) > 1e-100, k1 = fabs(d1) > 1e-100;
        double inv = fast_rcp((k0 ? d0 : 1.0) * (k1 ? d1 : 1.0));
        double e0 = k0 ? (k1 ? d1 : 1.0) * inv : 0.0, e1 = k1 ? (k0 ? d0 : 1.0) * inv : 0.0;
        double qa0 = sq[idx], qb0 = sq[idx + 1], qa1 = sq[idx + 2], qb1 = sq[idx + 3];
        double p0p = fma(qa0, s2b, qb0 * S2D0) * e0;
        double p0m = fma(qb0, s2b, qa0 * S2D0) * e0;
        double p1p = fma(qa1, s2b, qb1 * S2D1) * e1;
        double p1m = fma(qb1, s2b, qa1 * S2D1) * e1;
        double g_sum = p0p + p1p;
        dx[R - 2] += p0m + p1m;
        dx[R - 1] += p0m - p1m;
        blk_all += g_sum;
        blk_3   += (sub & 1) ? -g_sum : g_sum;
        dx[R - 4] += (sub & 2) ? -g_sum : g_sum;
    }
    dx[R - 3] += blk_3;
    dx[0] += blk_all;
    #pragma unroll
    for (int u = 1; u < R - 4; ++u) dx[u] += ((blk >> (R - 1 - u)) & 1) ? -blk_all : blk_all;
}

// sq holds, per quad of sign patterns (s = 4i .. 4i+3), (q0 + q3, q0 - q3, q1 + q2, q1 - q2) for R >= 3 (plain q
// otherwise). Within a quad the angles are b + D0, b + D1, b - D1, b - D0, and
// tan(b +- D) = (sin 2b +- sin 2D) / (cos 2b + cos 2D), so each +-D pair shares one denominator.
template <int R>
DEVI void bwd_cols(const double* __restrict__ x, const double* __restrict__ sq, double* __restrict__ dx) {
    constexpr int S_ALL = 1 << (R - 1);
    if constexpr (R >= 7) {
        // groups of 4 blocks (top signs of x[R-6], x[R-5]) in Gray order: one sincos per group, three rotations
        double C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1, C5, S5, C6, S6;
        fast_sincos(2.0 * (x[R - 2] + x[R - 1]), &S2D0, &C2D0);
        fast_sincos(2.0 * (x[R - 2] - x[R - 1]), &S2D1, &C2D1);
        fast_sincos(2.0 * (x[R - 4] + x[R - 3]), &SA0, &CA0);
        fast_sincos(2.0 * (x[R - 4] - x[R - 3]), &SA1, &CA1);
        fast_sincos(4.0 * x[R - 5], &S5, &C5);
        fast_sincos(4.0 * x[R - 6], &S6, &C6);
        #pragma unroll 1
        for (int blk0 = 0; blk0 < S_ALL; blk0 += 64) {
            double base_top = x[0] + x[R - 6] + x[R - 5];
            #pragma unroll
            for (int u = 1; u < R - 6; ++u) base_top += ((blk0 >> (R - 1 - u)) & 1) ? -x[u] : x[u];
            double s2t, c2t;
            fast_sincos(2.0 * base_top, &s2t, &c2t);
            bwd_blk<R>(blk0, c2t, s2t, sq, dx, C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1);
            double c1 = fma(c2t, C5, s2t * S5), s1 = fma(s2t, C5, -(c2t * S5));      // sigma_{R-5}: + -> -
            bwd_blk<R>(blk0 + 16, c1, s1, sq, dx, C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1);
            double c3 = fma(c1, C6, s1 * S6), s3 = fma(s1, C6, -(c1 * S6));           // sigma_{R-6}: + -> -
            bwd_blk<R>(blk0 + 48, c3, s3, sq, dx, C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1);
            double c2 = fma(c3, C5, -(s3 * S5)), s2 = fma(s3, C5, c3 * S5);           // sigma_{R-5}: - -> +
            bwd_blk<R>(blk0 + 32, c2, s2, sq, dx, C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1);
        }
    } else if constexpr (R >= 5) {
        double C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1;
        fast_sincos(2.0 * (x[R - 2] + x[R - 1]), &S2D0, &C2D0);
        fast_sincos(2.0 * (x[R - 2] - x[R - 1]), &S2D1, &C2D1);
        fast_sincos(2.0 * (x[R - 4] + x[R - 3]), &SA0, &CA0);
        fast_sincos(2.0 * (x[R - 4] - x[R - 3]), &SA1, &CA1);
        #pragma unroll 1
        for (int blk = 0; blk < S_ALL; blk += 16) {
            double base_top = x[0];
            #pragma unroll
            for (int u = 1; u < R - 4; ++u) base_top += ((blk >> (R - 1 - u)) & 1) ? -x[u] : x[u];
            double s2t, c2t;
            fast_sincos(2.0 * base_top, &s2t, &c2t);
            bwd_blk<R>(blk, c2t, s2t, sq, dx, C2D0, S2D0, C2D1, S2D1, CA0, SA0, CA1, SA1);
        }
    } else if constexpr (R >= 3) {
        double C2D0, S2D0, C2D1, S2D1;
        fast_sincos(2.0 * (x[R - 2] + x[R - 1]), &S2D0, &C2D0);
        fast_sincos(2.0 * (x[R - 2] - x[R - 1]), &S2D1, &C2D1);
        #pragma unroll
        for (int idx = 0; idx < S_ALL; idx += 4) {
            double base = x[0];
            #pragma unroll
            for (int u = 1; u < R - 2; ++u) base += ((idx >> (R - 1 - u)) & 1) ? -x[u] : x[u];
            double s2b, c2b;
            fast_sincos(2.0 * base, &s2b, &c2b);
            double d0 = c2b + C2D0;
            double d1 = c2b + C2D1;
            // a shared denominator that cancels to (near) zero would give inf * 0: that +-D pair is dropped
            bool k0 = fabs(d0) > 1e-100, k1 = fabs(d1) > 1e-100;
            double inv = fast_rcp((k0 ? d0 : 1.0) * (k1 ? d1 : 1.0));
            double e0 = k0 ? (k1 ? d1 : 1.0) * inv : 0.0, e1 = k1 ? (k0 ? d0 : 1.0) * inv : 0.0;
            double qa0 = sq[idx], qb0 = sq[idx + 1], qa1 = sq[idx + 2], qb1 = sq[idx + 3];
            double p0p = fma(qa0, s2b, qb0 * S2D0) * e0;
            double p0m = fma(qb0, s2b, qa0 * S2D0) * e0;
            double p1p = fma(qa1, s2b, qb1 * S2D1) * e1;
            double p1m = fma(qb1, s2b, qa1 * S2D1) * e1;
            double g_sum = p0p + p1p;
            dx[0] += g_sum;
            #pragma unroll
            for (int u = 1; u < R - 2; ++u) dx[u] += ((idx >> (R - 1 - u)) & 1) ? -g_sum : g_sum;
            dx[R - 2] += p0m + p1m;
            dx[R - 1] += p0m - p1m;
        }
    } else {
        #pragma unroll
        for (int idx = 0; idx < S_ALL; ++idx) {
            double arg = x[0];
            #pragma unroll
            for (int u = 1; u < R; ++u) arg += ((idx >> (R - 1 - u)) & 1) ? -x[u] : x[u];
            double g = sq[idx] * fast_tan(arg);
            dx[0] += g;
            #pragma unroll
            for (int u = 1; u < R; ++u) dx[u] += ((idx >> (R - 1 - u)) & 1) ? -g : g;
        }
    }
}

// block-level backward (R >= 5): 128 threads = 128 columns of one (b, ml)
template <int R>
DEVI void bwd_block_r(const double* __restrict__ Thm, double* __restrict__ dThm, const double* __restrict__ Pm,
                      const double* __restrict__ Dm, const int* __restrict__ sA, double* __restrict__ sq,
                      double w, int n, int j, bool do_th1) {
    constexpr int S_ALL = 1 << (R - 1);
    int tid = threadIdx.x;
    for (int s = tid; s < S_ALL; s += BWD_NT) sq[s] = w * Pm[s];
    if (do_th1 && tid < R) unsafeAtomicAdd(&dThm[(size_t)sA[tid] * n + sA[tid]], w * Dm[tid]);
    __syncthreads();
    if (j >= n) return;
    #pragma unroll
    for (int u = 0; u < R; ++u) if (j == sA[u]) return;
    double x[R], dx[R];
    #pragma unroll
    for (int u = 0; u < R; ++u) {
        x[u] = Thm[(size_t)sA[u] * n + j];
        dx[u] = 0.0;
    }
    bwd_cols<R>(x, sq, dx);
    #pragma unroll
    for (int u = 0; u < R; ++u) unsafeAtomicAdd(&dThm[(size_t)sA[u] * n + j], dx[u]);
}

// wave-level backward (R <= 4): one wave per (b, ml); lanes stride over the columns
template <int R>
DEVI void bwd_wave_r(const double* __restrict__ Thm, double* __restrict__ dThm, const double* __restrict__ Pm,
                     const double* __restrict__ Dm, const int* __restrict__ Ab, double w, int n, int lane) {
    constexpr int S = 1 << (R - 1);
    int a_l = (lane < R) ? Ab[lane] : 0;
    int sA[R];
    #pragma unroll
    for (int u = 0; u < R; ++u) sA[u] = __shfl(a_l, u, 64);
    double q_l = (lane < S) ? w * Pm[lane] : 0.0;
    double sq[S];
    #pragma unroll
    for (int s = 0; s < S; ++s) sq[s] = __shfl(q_l, s, 64);
    if (lane < R) unsafeAtomicAdd(&dThm[(size_t)a_l * n + a_l], w * Dm[lane]);
    for (int j = lane; j < n; j += 64) {
        bool act = true;
        #pragma unroll
        for (int u = 0; u < R; ++u) act = act && (j != sA[u]);
        if (!act) continue;
        double x[R], dx[R];
        #pragma unroll
        for (int u = 0; u < R; ++u) {
            x[u] = Thm[(size_t)sA[u] * n + j];
            dx[u] = 0.0;
        }
        bwd_cols<R>(x, sq, dx);
        #pragma unroll
        for (int u = 0; u < R; ++u) unsafeAtomicAdd(&dThm[(size_t)sA[u] * n + j], dx[u]);
    }
}

// blocks [0, nb_big): (jc, ml, b) with r >= 5, b descending; blocks [nb_big, ...): 2 waves = 2 (b, ml) items, r <= 4
__global__ __launch_bounds__(BWD_NT) __attribute__((amdgpu_waves_per_eu(3, 8)))
void bwd_kernel(const double* __restrict__ Th, double* __restrict__ dTh, const double* __restrict__ P,
                const int* __restrict__ A_pad, const int* __restrict__ b_info, const double* __restrict__ dz,
                const double* __restrict__ pi, int n, int L, int B, int S_tot, int Mps, int m0,
                int nml, int njc, int B5, int nb_big, int Lp, int l0, const double* __restrict__ Dq) {
    __shared__ double sq[512];
    __shared__ int sA[ASTR];
    size_t nn = (size_t)n * n;
    int blk = blockIdx.x;
    if (blk < nb_big) {
        int jc = blk % njc;
        int ml = (blk / njc) % nml;
        int b = B - 1 - blk / (njc * nml);
        int r = b_info[2 * b];
        int p_off = b_info[2 * b + 1];
        int mc = ml / Lp;
        int l = l0 + ml % Lp;
        int m = m0 + mc;
        int seed = m / Mps;
        const double* Thm = Th + (size_t)ml * nn;
        double* dThm = dTh + (size_t)ml * nn;
        if (threadIdx.x < r) sA[threadIdx.x] = A_pad[((size_t)seed * B + b) * ASTR + threadIdx.x];
        __syncthreads();
        double w = dz[(size_t)mc * B + b] * (pi[(size_t)m * L + l] / (double)(1 << (r - 1)));
        const double* Pm = P + (size_t)(mc * L + l) * S_tot + p_off;
        const double* Dm = Dq + ((size_t)(mc * L + l) * B + b) * ASTR;
        int j = jc * BWD_NT + threadIdx.x;
        bool do_th1 = (jc == 0);
        switch (r) {
            case 5:  bwd_block_r<5>(Thm, dThm, Pm, Dm, sA, sq, w, n, j, do_th1); break;
            case 6:  bwd_block_r<6>(Thm, dThm, Pm, Dm, sA, sq, w, n, j, do_th1); break;
            case 7:  bwd_block_r<7>(Thm, dThm, Pm, Dm, sA, sq, w, n, j, do_th1); break;
            case 8:  bwd_block_r<8>(Thm, dThm, Pm, Dm, sA, sq, w, n, j, do_th1); break;
            case 9:  bwd_block_r<9>(Thm, dThm, Pm, Dm, sA, sq, w, n, j, do_th1); break;
            default: bwd_block_r<10>(Thm, dThm, Pm, Dm, sA, sq, w, n, j, do_th1); break;
        }
    } else {
        int item = (blk - nb_big) * (BWD_NT / 64) + (threadIdx.x >> 6);
        if (item >= B5 * nml) return;
        int ml = item % nml;
        int b = B5 - 1 - item / nml;
        int r = b_info[2 * b];
        int p_off = b_info[2 * b + 1];
        int mc = ml / Lp;
        int l = l0 + ml % Lp;
        int m = m0 + mc;
        int seed = m / Mps;
        const double* Thm = Th + (size_t)ml * nn;
        double* dThm = dTh + (size_t)ml * nn;
        const int* Ab = A_pad + ((size_t)seed * B + b) * ASTR;
        double w = dz[(size_t)mc * B + b] * (pi[(size_t)m * L + l] / (double)(1 << (r - 1)));
        const double* Pm = P + (size_t)(mc * L + l) * S_tot + p_off;
        const double* Dm = Dq + ((size_t)(mc * L + l) * B + b) * ASTR;
        int lane = threadIdx.x & 63;
        switch (r) {
            case 1:  bwd_wave_r<1>(Thm, dThm, Pm, Dm, Ab, w, n, lane); break;
            case 2:  bwd_wave_r<2>(Thm, dThm, Pm, Dm, Ab, w, n, lane); break;
            case 3:  bwd_wave_r<3>(Thm, dThm, Pm, Dm, Ab, w, n, lane); break;
            default: bwd_wave_r<4>(Thm, dThm, Pm, Dm, Ab, w, n, lane); break;
        }
    }
}

// Adam (optax) on a contiguous range; mu/nu (and c_out) may live in pinned host memory (streamed over PCIe)
__global__ __launch_bounds__(256)
void adam_kernel(const double* __restrict__ gbuf, const double* __restrict__ c_in, double* __restrict__ c_out,
                 double* __restrict__ mu, double* __restrict__ nu, size_t count, double neg_lr, double b1, double b2,
                 double omb1, double omb2, double eps, double bc1, double bc2) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        double gk = gbuf[i];
        double mn = omb1 * gk + b1 * mu[i];
        double vn = omb2 * (gk * gk) + b2 * nu[i];
        mu[i] = mn;
        nu[i] = vn;
        double mh = mn / bc1;
        double vh = vn / bc2;
        c_out[i] = c_in[i] + neg_lr * (mh / (sqrt(vh) + eps));
    }
}

// level 2: Adam on chunk q with c_old from the staging buffer and the result streamed to host memory; each staging
// element is then refilled with the host angles of chunk q + 2 (same buffer, read-before-write in the same thread)
__global__ __launch_bounds__(256)
void adam_stage_kernel(const double* __restrict__ gbuf, double* __restrict__ stg, double* __restrict__ c_out,
                       const double* __restrict__ c_next, double* __restrict__ mu, double* __restrict__ nu,
                       double* __restrict__ g_host, size_t count_gpu, size_t count, size_t count_next, double neg_lr,
                       double b1, double b2, double omb1, double omb2, double eps, double bc1, double bc2,
                       size_t i_begin, size_t i_end) {
    // elements [count_gpu, count) are updated by the host (their gradient goes to g_host); this launch covers
    // [i_begin, i_end)
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = i_begin + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < i_end; i += stride) {
        if (i >= count_gpu && i < count) g_host[i - count_gpu] = gbuf[i];
        if (i < count_gpu) {
            double gk = gbuf[i];
            double mn = omb1 * gk + b1 * mu[i];
            double vn = omb2 * (gk * gk) + b2 * nu[i];
            mu[i] = mn;
            nu[i] = vn;
            double mh = mn / bc1;
            double vh = vn / bc2;
            c_out[i] = stg[i] + neg_lr * (mh / (sqrt(vh) + eps));
        }
        if (i < count_next) stg[i] = c_next[i];
    }
}

__global__ __launch_bounds__(256)
void copy_kernel(const double* __restrict__ src, double* __restrict__ dst, size_t count) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) dst[i] = src[i];
}

// ------------------------------------------------------------------------------------------------
// Tiled (32 x 32) versions of build / fold / update over upper-triangle tile pairs (ti <= tj): the lower-triangle
// half goes through an LDS transpose so every global access is coalesced; the Walsh map is an in-register FWHT.
// ------------------------------------------------------------------------------------------------
#define TT 32
template <int L>
DEVI void fwht(double* v) {
    #pragma unroll
    for (int h = 1; h < L; h <<= 1) {
        #pragma unroll
        for (int i = 0; i < L; i += 2 * h) {
            #pragma unroll
            for (int k = i; k < i + h; ++k) {
                double a = v[k], b = v[k + h];
                v[k] = a + b;
                v[k + h] = a - b;
            }
        }
    }
}

DEVI void tile_pair(int p, int* ti, int* tj) {
    int t = (int)((sqrt(8.0 * (double)p + 1.0) - 1.0) * 0.5);
    while ((t + 1) * (t + 2) / 2 <= p) ++t;
    while (t * (t + 1) / 2 > p) --t;
    *tj = t;
    *ti = p - t * (t + 1) / 2;
}

template <int L>
DEVI double pick(const double* v, int l) {
    double r = v[0];
    #pragma unroll
    for (int k = 1; k < L; ++k) r = (k == l) ? v[k] : r;
    return r;
}

// Th[mc][l - l0][i][j] = 2 sum_k W[l][k] c[m][k][eidx[i][j]]  (l0 <= l < l0 + nl)
template <int L>
__global__ __launch_bounds__(512)
void build_tiled(const double* __restrict__ c, const int* __restrict__ eidx, double* __restrict__ Th,
                 int n, long G, int m0, int l0, int nl) {
    __shared__ double tile[TT][TT + 1];
    int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    int ti, tj;
    tile_pair(blockIdx.x, &ti, &tj);
    int mc = blockIdx.y;
    const double* cm = c + (size_t)(m0 + mc) * L * G;
    size_t nn = (size_t)n * n;
    double* Tm = Th + (size_t)mc * nl * nn;
    double v[2][L];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
        int g = (i < n && j < n) ? eidx[(size_t)i * n + j] : -1;
        #pragma unroll
        for (int k = 0; k < L; ++k) v[r][k] = (g >= 0) ? cm[(size_t)k * G + g] : 0.0;
        fwht<L>(v[r]);
    }
    if (ti == tj) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
            if (i < n && j < n) {
                #pragma unroll
                for (int l = 0; l < L; ++l) {
                    if (l >= l0 && l < l0 + nl) Tm[(size_t)(l - l0) * nn + (size_t)i * n + j] = 2.0 * v[r][l];
                }
            }
        }
        return;
    }
    #pragma unroll
    for (int l = 0; l < L; ++l) {
        if (l < l0 || l >= l0 + nl) continue;          // block-uniform
        int ll = l - l0;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
            double val = 2.0 * v[r][l];
            if (i < n && j < n) Tm[(size_t)ll * nn + (size_t)i * n + j] = val;
            tile[ty + 16 * r][tx] = val;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            int i2 = tj * TT + ty + 16 * r, j2 = ti * TT + tx;
            if (i2 < n && j2 < n) Tm[(size_t)ll * nn + (size_t)i2 * n + j2] = tile[tx][ty + 16 * r];
        }
        __syncthreads();
    }
}

// d theta of the upper-tile elements (i <= j): diagonal -> one-body, else dTh[i][j] + dTh[j][i], for the local
// branches 0 .. nl - 1 of the dTh buffer (global branch l0 + ll), accumulated into dv[r][l0 + ll]
template <int L>
DEVI void fold_gather(double* __restrict__ dT, double (*dv)[L], double (*tile)[TT + 1], int ti, int tj,
                      int tx, int ty, int n, int l0, int nl) {
    // every dTh element is read by exactly one thread here and reset to zero (ready for the next backward)
    size_t nn = (size_t)n * n;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int k = 0; k < L; ++k) dv[r][k] = 0.0;
    }
    for (int ll = 0; ll < nl; ++ll) {
        double* d = dT + (size_t)ll * nn;
        double add[2];
        if (ti == tj) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
                add[r] = 0.0;
                if (i < n && j < n && i <= j) {
                    if (i == j) {
                        add[r] = d[(size_t)i * n + i];
                        d[(size_t)i * n + i] = 0.0;
                    } else {
                        add[r] = d[(size_t)i * n + j] + d[(size_t)j * n + i];
                        d[(size_t)i * n + j] = 0.0;
                        d[(size_t)j * n + i] = 0.0;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                int i2 = tj * TT + ty + 16 * r, j2 = ti * TT + tx;
                double v = 0.0;
                if (i2 < n && j2 < n) {
                    v = d[(size_t)i2 * n + j2];
                    d[(size_t)i2 * n + j2] = 0.0;
                }
                tile[ty + 16 * r][tx] = v;
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
                add[r] = 0.0;
                if (i < n && j < n) {
                    add[r] = d[(size_t)i * n + j] + tile[tx][ty + 16 * r];
                    d[(size_t)i * n + j] = 0.0;
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            #pragma unroll
            for (int k = 0; k < L; ++k) dv[r][k] += (k == ll) ? add[r] : 0.0;
        }
    }
}

template <int L>
__global__ __launch_bounds__(512)
void fold_tiled(double* __restrict__ dTh, const int* __restrict__ eidx, double* __restrict__ gbuf,
                int n, long G, int l0, int nl, int accum) {
    __shared__ double tile[TT][TT + 1];
    int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    int ti, tj;
    tile_pair(blockIdx.x, &ti, &tj);
    int mc = blockIdx.y;
    size_t nn = (size_t)n * n;
    double dv[2][L];
    fold_gather<L>(dTh + (size_t)mc * nl * nn, dv, tile, ti, tj, tx, ty, n, l0, nl);
    double* gm = gbuf + (size_t)mc * L * G;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
        if (i >= n || j >= n || i > j) continue;
        int g = eidx[(size_t)i * n + j];
        if (g < 0) continue;
        fwht<L>(dv[r]);
        #pragma unroll
        for (int kk = 0; kk < L; ++kk) {
            if (kk < nl) gm[(size_t)(l0 + kk) * G + g] = 2.0 * dv[r][kk];
        }
    }
}

// split models: row h Lp + kk holds part h's partial transform F_h[kk] = 2 sum_ll (-1)^{popc(ll & kk)} d theta_{h,ll};
// since W[h Lp + ll][kh Lp + kk] = (-1)^{popc(h & kh)} (-1)^{popc(ll & kk)}, the gradient is
// g[kh Lp + kk] = sum_h (-1)^{popc(h & kh)} F_h[kk]: an H-point transform per (kk, column), done in place
template <int L>
__global__ __launch_bounds__(256)
void combine_parts(double* __restrict__ gbuf, long G, int Lp) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < (size_t)G; g += stride) {
        double v[L];
        #pragma unroll
        for (int k = 0; k < L; ++k) v[k] = gbuf[(size_t)k * G + g];
        #pragma unroll
        for (int h = 1; h < L; h <<= 1) {               // butterflies over the high (part) bits only
            if (h < Lp) continue;
            #pragma unroll
            for (int i = 0; i < L; i += 2 * h) {
                #pragma unroll
                for (int k = i; k < i + h; ++k) {
                    double a = v[k], b = v[k + h];
                    v[k] = a + b;
                    v[k + h] = a - b;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < L; ++k) gbuf[(size_t)k * G + g] = v[k];
    }
}

template <int L>
__global__ __launch_bounds__(512)
void update_tiled(double* __restrict__ dTh, const int* __restrict__ eidx, double* __restrict__ c,
                  double* __restrict__ mu, double* __restrict__ nu, int n, long G, int m0,
                  double neg_lr, double b1, double b2, double omb1, double omb2, double eps, double bc1, double bc2) {
    __shared__ double tile[TT][TT + 1];
    int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    int ti, tj;
    tile_pair(blockIdx.x, &ti, &tj);
    int mc = blockIdx.y;
    size_t nn = (size_t)n * n;
    double dv[2][L];
    fold_gather<L>(dTh + (size_t)mc * L * nn, dv, tile, ti, tj, tx, ty, n, 0, L);
    size_t moff = (size_t)(m0 + mc) * L * G;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        int i = ti * TT + ty + 16 * r, j = tj * TT + tx;
        if (i >= n || j >= n || i > j) continue;
        int g = eidx[(size_t)i * n + j];
        if (g < 0) continue;
        fwht<L>(dv[r]);
        #pragma unroll
        for (int k = 0; k < L; ++k) {
            double gk = 2.0 * dv[r][k];
            size_t idx = moff + (size_t)k * G + g;
            double mn = omb1 * gk + b1 * mu[idx];
            double vn = omb2 * (gk * gk) + b2 * nu[idx];
            mu[idx] = mn;
            nu[idx] = vn;
            double mh = mn / bc1;
            double vh = vn / bc2;
            c[idx] = c[idx] + neg_lr * (mh / (sqrt(vh) + eps));
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Host engine
// ------------------------------------------------------------------------------------------------
struct Engine {
    int n, a, L, M, S, Mps, B, S_tot, W_u64, N_tr, Mc, B5, B6, level, nchunk, H;
    long G;
    long long seq;
    hipStream_t st, st2, st3;
    double *c, *mu, *nu, *pi, *w_of_b, *t, *Th, *dTh, *z, *dz, *P, *Dq, *loss;
    double *beta, *mb, *vb, *gpi;   // trainable branch weights (eng_train_pi); null when the weights are fixed
    double pi_lr_scale;
    double* gbs[2];   // packed gradient: one buffer, two when a model is split (fold of q overlaps adam of q - 1)
    double* gb;
    double* stg[2];
    uint64_t *pcg, *jtab, *x;
    int *b_info, *A_pad, *eidx;
    int* Ab[2];
    double* tb[2];
    uint64_t* rowst;
    hipStream_t st4;
    hipEvent_t ev_samp[2], ev_used[2];
    int loss_cap;
    hipEvent_t ev_fold, ev_adam, ev_adamb[2], ev_stage[2], ev_grp[16];   // ev_grp[h]: rows of part h consumed
    hipEvent_t* ev_c;
    // host-side share of the Adam update (level 2): a worker thread updates elements [count_gpu, count) of each chunk
    double cpu_frac;
    int cpu_threads;
    double* g_host[2];
    size_t g_cap;
    hipEvent_t ev_gh[2];
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    struct Task { long long q; size_t off, cnt_gpu, cnt; double neg_lr, b1, b2, omb1, omb2, eps, bc1, bc2; int slot; };
    std::deque<Task> tasks;
    bool stop;
    std::atomic<long long> done;   // all host tasks with sequence <= done are finished
};

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); return -1; } } while (0)

template <typename T> static hipError_t dalloc(T** p, size_t count) { return hipMalloc((void**)p, count * sizeof(T) + 64); }

static void cpu_worker(Engine* E) {
    omp_set_num_threads(E->cpu_threads);
    for (;;) {
        Engine::Task tk;
        {
            std::unique_lock<std::mutex> lk(E->mtx);
            E->cv.wait(lk, [E] { return E->stop || !E->tasks.empty(); });
            if (E->tasks.empty()) return;
            tk = E->tasks.front();
            E->tasks.pop_front();
        }
        (void)hipEventSynchronize(E->ev_gh[tk.slot]);
        const double* g = E->g_host[tk.slot];
        double* c = E->c + tk.off;
        double* mu = E->mu + tk.off;
        double* nu = E->nu + tk.off;
        long long lo = (long long)tk.cnt_gpu, hi = (long long)tk.cnt;
        #pragma omp parallel for schedule(static)
        for (long long i = lo; i < hi; ++i) {
            double gk = g[i - lo];
            double mn = tk.omb1 * gk + tk.b1 * mu[i];
            double vn = tk.omb2 * (gk * gk) + tk.b2 * nu[i];
            mu[i] = mn;
            nu[i] = vn;
            double mh = mn / tk.bc1;
            double vh = vn / tk.bc2;
            c[i] = c[i] + tk.neg_lr * (mh / (std::sqrt(vh) + tk.eps));
        }
        {
            std::lock_guard<std::mutex> lk(E->mtx);
            E->done.store(tk.q);
        }
        E->cv.notify_all();
    }
}

static void wait_done(Engine* E, long long q) {
    if (q < 0) return;
    std::unique_lock<std::mutex> lk(E->mtx);
    E->cv.wait(lk, [E, q] { return E->done.load() >= q; });
}

// level 0: everything in device memory.
// level 1: the Adam moments live in pinned host memory; per chunk the gradient is folded into a packed buffer and a
//          PCIe-streaming Adam kernel runs on a second stream (own CUs), overlapped with the next chunk.
// level 2: the compiled angles live in pinned host memory too; a third stream (own CUs) stages the next chunks'
//          angles into two device buffers, from which the chunk is built and updated (the result streams back).
extern "C" int eng_create(void** out, int n, int a, int M, int S, int B, int S_tot, long G, int W_u64, int N_tr, int Mc,
                          int level, int H, double cpu_frac, int cpu_threads, const int* eidx, const double* c0, const double* pi, const uint64_t* pcg,
                          const uint64_t* jtab, const int* b_info, const double* w_of_b, const uint64_t* x) {
    Engine* E = new Engine();
    *out = (void*)E;                 // handed back even if setup fails below, so eng_destroy can release it
    E->n = n; E->a = a; E->L = 1 << a; E->M = M; E->S = S; E->Mps = M / S; E->B = B; E->S_tot = S_tot;
    E->G = G; E->W_u64 = W_u64; E->N_tr = N_tr; E->Mc = Mc; E->loss_cap = 0; E->loss = nullptr; E->seq = 0;
    E->nchunk = (M + Mc - 1) / Mc;
    E->level = (level == 2 && E->nchunk < 3) ? 1 : level;   // level 2 needs >= 3 chunks (staging refills q + 2)
    E->H = (E->level > 0 && Mc == 1 && H > 1 && (E->L % H) == 0) ? H : 1;   // a model's branches in H parts
    int L = E->L;
    size_t nn = (size_t)n * n;
    size_t cnt = (size_t)M * L * G;
    size_t ccnt = (size_t)Mc * L * G;
    if (E->level > 0) {
        int ncu = 0;
        CK(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, 0));
        int nw = (ncu + 31) / 32;
        int nup = (ncu >= 64) ? 16 : ncu / 4, nst = 0;
        uint32_t m_main[64] = {0}, m_up[64] = {0}, m_st[64] = {0};
        for (int i = 0; i < ncu && i < 64 * 32; ++i) {
            if (i < nup) m_up[i / 32] |= 1u << (i % 32);
            else if (i < nup + nst) m_st[i / 32] |= 1u << (i % 32);
            else m_main[i / 32] |= 1u << (i % 32);
        }
        if (hipExtStreamCreateWithCUMask(&E->st, nw, m_main) != hipSuccess ||
            hipExtStreamCreateWithCUMask(&E->st2, nw, m_up) != hipSuccess) {   // no CU masks: plain streams
            (void)hipGetLastError();
            CK(hipStreamCreateWithFlags(&E->st, hipStreamNonBlocking));
            CK(hipStreamCreateWithFlags(&E->st2, hipStreamNonBlocking));
        }
        CK(hipStreamCreateWithFlags(&E->st3, hipStreamNonBlocking));
    } else {
        CK(hipStreamCreateWithFlags(&E->st, hipStreamNonBlocking));
        CK(hipStreamCreateWithFlags(&E->st2, hipStreamNonBlocking));
        CK(hipStreamCreateWithFlags(&E->st3, hipStreamNonBlocking));
    }
    CK(hipEventCreateWithFlags(&E->ev_fold, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&E->ev_adam, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&E->ev_adamb[0], hipEventDisableTiming));
    for (int i = 0; i < 16; ++i) CK(hipEventCreateWithFlags(&E->ev_grp[i], hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&E->ev_adamb[1], hipEventDisableTiming));
    for (int i = 0; i < 2; ++i) CK(hipEventCreateWithFlags(&E->ev_stage[i], hipEventDisableTiming));
    E->ev_c = new hipEvent_t[E->nchunk]();
    E->cpu_frac = (E->level == 2) ? cpu_frac : 0.0;
    E->cpu_threads = cpu_threads > 0 ? cpu_threads : 1;
    E->g_host[0] = E->g_host[1] = nullptr;
    E->g_cap = 0;
    E->stop = false;
    E->done.store(-1);
    for (int i = 0; i < 2; ++i) CK(hipEventCreateWithFlags(&E->ev_gh[i], hipEventDisableTiming));
    if (E->cpu_frac > 0.0) {
        E->g_cap = (size_t)(E->cpu_frac * (double)ccnt) + 1024;
        for (int i = 0; i < 2; ++i) CK(hipHostMalloc((void**)&E->g_host[i], E->g_cap * 8, hipHostMallocDefault));
    }
    for (int k = 0; k < E->nchunk; ++k) CK(hipEventCreateWithFlags(&E->ev_c[k], hipEventDisableTiming));
    E->gb = E->stg[0] = E->stg[1] = E->gbs[0] = E->gbs[1] = nullptr;
    if (E->level == 2) {
        CK(hipHostMalloc((void**)&E->c, cnt * 8 + 64, hipHostMallocDefault));
        memcpy(E->c, c0, cnt * 8);
        for (int i = 0; i < 2; ++i) CK(dalloc(&E->stg[i], ccnt));
    } else {
        CK(dalloc(&E->c, cnt));
        CK(hipMemcpy(E->c, c0, cnt * 8, hipMemcpyHostToDevice));
    }
    if (E->level >= 1) {
        CK(hipHostMalloc((void**)&E->mu, cnt * 8 + 64, hipHostMallocDefault));
        CK(hipHostMalloc((void**)&E->nu, cnt * 8 + 64, hipHostMallocDefault));
        memset(E->mu, 0, cnt * 8); memset(E->nu, 0, cnt * 8);
        CK(dalloc(&E->gbs[0], ccnt));
        CK(hipMemset(E->gbs[0], 0, ccnt * 8));
        E->gbs[1] = E->gbs[0];
        E->gb = E->gbs[0];
    } else {
        CK(dalloc(&E->mu, cnt)); CK(dalloc(&E->nu, cnt));
        CK(hipMemset(E->mu, 0, cnt * 8)); CK(hipMemset(E->nu, 0, cnt * 8));
    }
    CK(dalloc(&E->pi, (size_t)M * L)); CK(dalloc(&E->w_of_b, B));
    for (int i = 0; i < 2; ++i) {
        CK(dalloc(&E->tb[i], (size_t)S * B)); CK(dalloc(&E->Ab[i], (size_t)S * B * ASTR));
        CK(hipMemset(E->Ab[i], 0, (size_t)S * B * ASTR * 4));
        CK(hipEventCreateWithFlags(&E->ev_samp[i], hipEventDisableTiming));
        CK(hipEventCreateWithFlags(&E->ev_used[i], hipEventDisableTiming));
    }
    E->t = E->tb[0];
    E->A_pad = E->Ab[0];
    CK(hipStreamCreateWithFlags(&E->st4, hipStreamNonBlocking));
    if (E->H > 1) {
        // one buffer of 2 L/H matrices: forward parts of 2 L/H branches, backward parts use halves as (dTh, Th)
        CK(dalloc(&E->Th, (size_t)2 * (L / E->H) * nn));
        E->dTh = nullptr;
    } else {
        CK(dalloc(&E->Th, (size_t)Mc * L * nn)); CK(dalloc(&E->dTh, (size_t)Mc * L * nn));
        CK(hipMemset(E->dTh, 0, (size_t)Mc * L * nn * 8));       // kept zero by the fold / update kernels
    }
    CK(dalloc(&E->z, (size_t)Mc * L * B)); CK(dalloc(&E->dz, (size_t)Mc * B));
    CK(dalloc(&E->P, (size_t)Mc * L * S_tot));
    CK(dalloc(&E->Dq, (size_t)Mc * L * B * ASTR));
    CK(dalloc(&E->pcg, (size_t)S * 4)); CK(dalloc(&E->jtab, (size_t)S * n * 4));
    CK(dalloc(&E->x, (size_t)(n + 1) * W_u64));
    CK(dalloc(&E->b_info, (size_t)B * 2)); CK(dalloc(&E->eidx, nn));
    CK(dalloc(&E->rowst, (size_t)S * B * 2));
    CK(hipMemcpy(E->pi, pi, (size_t)M * L * 8, hipMemcpyHostToDevice));
    CK(hipMemcpy(E->w_of_b, w_of_b, (size_t)B * 8, hipMemcpyHostToDevice));
    CK(hipMemcpy(E->pcg, pcg, (size_t)S * 4 * 8, hipMemcpyHostToDevice));
    CK(hipMemcpy(E->jtab, jtab, (size_t)S * n * 4 * 8, hipMemcpyHostToDevice));
    CK(hipMemcpy(E->x, x, (size_t)(n + 1) * W_u64 * 8, hipMemcpyHostToDevice));
    CK(hipMemcpy(E->b_info, b_info, (size_t)B * 2 * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(E->eidx, eidx, nn * 4, hipMemcpyHostToDevice));
    E->B5 = 0; E->B6 = 0;
    for (int b = 0; b < B; ++b) {
        if (b_info[2 * b] < 5) E->B5 = b + 1;
        if (b_info[2 * b] < 6) E->B6 = b + 1;
    }
    CK(hipDeviceSynchronize());
    if (E->cpu_frac > 0.0) E->worker = std::thread(cpu_worker, E);
    *out = (void*)E;
    return 0;
}

static void chunk_range(Engine* E, long long q, size_t* off, size_t* count) {
    int k = (int)(q % E->nchunk);
    int m0 = k * E->Mc;
    int mc = (E->M - m0 < E->Mc) ? (E->M - m0) : E->Mc;
    *off = (size_t)m0 * E->L * E->G;
    *count = (size_t)mc * E->L * E->G;
}

static void stage_chunk(Engine* E, long long q) {
    // prologue only: copy the host angles of chunk q into staging buffer q & 1 (later refills happen in adam_stage)
    size_t off, count;
    chunk_range(E, q, &off, &count);
    copy_kernel<<<128, 256, 0, E->st2>>>(E->c + off, E->stg[q & 1], count);
    (void)hipEventRecord(E->ev_stage[q & 1], E->st2);
}

template <int L>
static void launch_chunk(Engine* E, int k, int m0, int mc, double neg_lr, double b1, double b2, double omb1,
                         double omb2, double eps, double bc1, double bc2, double* loss_row) {
    int n = E->n, B = E->B;
    int H = E->H, Lp = L / H;
    int nml = mc * Lp;
    size_t nn = (size_t)n * n;
    long long q = E->seq++;
    int bi = (int)(q & 1);
    int ntl = (n + TT - 1) / TT;
    dim3 gt(ntl * (ntl + 1) / 2, mc);
    const double* csrc = E->c;
    int cm0 = m0;
    if (E->level == 2) {
        (void)hipStreamWaitEvent(E->st, E->ev_stage[bi], 0);
        csrc = E->stg[bi];
        cm0 = 0;
    } else if (E->level == 1) {
        (void)hipStreamWaitEvent(E->st, E->ev_c[k], 0);
    }
    // forward in parts of Lf = 2 Lp branches when split (no dTh needed yet), backward in parts of Lp
    int Hf = (H > 1) ? H / 2 : 1, Lf = L / Hf;
    int nmlf = mc * Lf;
    double* Thb = (H > 1) ? E->Th + (size_t)Lp * nn : E->Th;
    double* dThb = (H > 1) ? E->Th : E->dTh;
    int fnb_big = nmlf * (B - E->B6);
    int fnb = fnb_big + (E->B6 * nmlf + FWD_NT / 64 - 1) / (FWD_NT / 64);
    int njc = (n + BWD_NT - 1) / BWD_NT;
    int bnb_big = njc * nml * (B - E->B5);
    int bnb = bnb_big + (E->B5 * nml + BWD_NT / 64 - 1) / (BWD_NT / 64);
    for (int hf = 0; hf < Hf; ++hf) {
        build_tiled<L><<<gt, 512, 0, E->st>>>(csrc, E->eidx, E->Th, n, E->G, cm0, hf * Lf, Lf);
        fwd_kernel<<<fnb, FWD_NT, 0, E->st>>>(E->Th, E->A_pad, E->b_info, E->z, E->P, n, L, B, E->S_tot, E->Mps, m0,
                                             nmlf, E->B6, fnb_big, Lf, hf * Lf, E->Dq);
    }
    lossdz_kernel<<<dim3((B + 255) / 256, mc), 256, 0, E->st>>>(E->z, E->pi, E->t, E->w_of_b, E->dz, loss_row, L, B,
                                                               E->Mps, m0);
    if (E->beta) pigrad_kernel<<<dim3(L, mc), 256, 0, E->st>>>(E->z, E->dz, E->gpi, L, B, m0);
    for (int h = H - 1; h >= 0; --h) {          // part H - 1 is the upper half of the last forward part
        if (h != H - 1)
            build_tiled<L><<<gt, 512, 0, E->st>>>(csrc, E->eidx, Thb, n, E->G, cm0, h * Lp, Lp);
        if (H > 1 && h == H - 1) (void)hipMemsetAsync(dThb, 0, (size_t)nml * nn * 8, E->st);   // held forward angles
        bwd_kernel<<<bnb, BWD_NT, 0, E->st>>>(Thb, dThb, E->P, E->A_pad, E->b_info, E->dz, E->pi, n, L, B,
                                             E->S_tot, E->Mps, m0, nml, njc, E->B5, bnb_big, Lp, h * Lp, E->Dq);
        if (E->level == 0) {
            update_tiled<L><<<gt, 512, 0, E->st>>>(dThb, E->eidx, E->c, E->mu, E->nu, n, E->G, m0,
                                                   neg_lr, b1, b2, omb1, omb2, eps, bc1, bc2);
            return;
        }
        if (H > 1) (void)hipStreamWaitEvent(E->st, E->ev_grp[h], 0);   // adam(q - 1) consumed part h's rows
        else (void)hipStreamWaitEvent(E->st, E->ev_adam, 0);
        fold_tiled<L><<<gt, 512, 0, E->st>>>(dThb, E->eidx, E->gb, n, E->G, h * Lp, Lp, h != H - 1);
    }
    if (H > 1) {
        (void)hipStreamWaitEvent(E->st, E->ev_adam, 0);                 // all rows of adam(q - 1) consumed
        combine_parts<L><<<512, 256, 0, E->st>>>(E->gb, E->G, Lp);
    }
    (void)hipEventRecord(E->ev_fold, E->st);
    (void)hipStreamWaitEvent(E->st2, E->ev_fold, 0);
    size_t off = (size_t)m0 * L * E->G;
    if (E->level == 2) {
        // update chunk q from staging buffer bi and refill it with chunk q + 2 (nchunk >= 3: its last update is done)
        size_t off2, cnt2;
        chunk_range(E, q + 2, &off2, &cnt2);
        size_t cnt = (size_t)mc * L * E->G;
        size_t cnt_gpu = cnt;
        if (E->cpu_frac > 0.0) {
            size_t ccpu = (size_t)(E->cpu_frac * (double)cnt);
            if (ccpu > E->g_cap) ccpu = E->g_cap;
            cnt_gpu = cnt - ccpu;
            // g_host[bi] was read by host task q - 2; the refill below reads chunk q + 2, last updated by task
            // q + 2 - nchunk
            long long need = q - 2;
            if (q + 2 - E->nchunk > need) need = q + 2 - E->nchunk;
            wait_done(E, need);
        }
        size_t tot = (cnt > cnt2) ? cnt : cnt2;
        if (H > 1) {
            // part by part (rows h Lp .. h Lp + Lp - 1), in the order the next chunk's folds overwrite them
            size_t rows_part = (size_t)Lp * E->G;
            for (int hh = H - 1; hh >= 0; --hh) {
                size_t ib = (size_t)hh * rows_part, ie = (hh == H - 1) ? tot : ib + rows_part;
                adam_stage_kernel<<<256, 256, 0, E->st2>>>(E->gb, E->stg[bi], E->c + off, E->c + off2, E->mu + off,
                                                           E->nu + off, E->g_host[bi], cnt_gpu, cnt, cnt2, neg_lr,
                                                           b1, b2, omb1, omb2, eps, bc1, bc2, ib, ie);
                (void)hipEventRecord(E->ev_grp[hh], E->st2);
            }
        } else {
            adam_stage_kernel<<<256, 256, 0, E->st2>>>(E->gb, E->stg[bi], E->c + off, E->c + off2, E->mu + off,
                                                       E->nu + off, E->g_host[bi], cnt_gpu, cnt, cnt2, neg_lr, b1, b2,
                                                       omb1, omb2, eps, bc1, bc2, 0, tot);
        }
        (void)hipEventRecord(E->ev_adam, E->st2);
        (void)hipEventRecord(E->ev_adamb[bi], E->st2);
        (void)hipEventRecord(E->ev_stage[bi], E->st2);
        if (cnt_gpu < cnt) {
            (void)hipEventRecord(E->ev_gh[bi], E->st2);
            {
                std::lock_guard<std::mutex> lk(E->mtx);
                E->tasks.push_back(Engine::Task{q, off, cnt_gpu, cnt, neg_lr, b1, b2, omb1, omb2, eps, bc1, bc2, bi});
            }
            E->cv.notify_all();
        } else if (E->cpu_frac > 0.0) {
            std::lock_guard<std::mutex> lk(E->mtx);
            E->done.store(q);
        }
    } else {
        adam_kernel<<<256, 256, 0, E->st2>>>(E->gb, E->c + off, E->c + off, E->mu + off, E->nu + off,
                                             (size_t)mc * L * E->G, neg_lr, b1, b2, omb1, omb2, eps, bc1, bc2);
        for (int hh = 0; hh < H; ++hh) (void)hipEventRecord(E->ev_grp[hh], E->st2);
        (void)hipEventRecord(E->ev_adam, E->st2);
        (void)hipEventRecord(E->ev_adamb[bi], E->st2);
        (void)hipEventRecord(E->ev_c[k], E->st2);
    }
}

extern "C" int eng_run(void* h, int step0, int nsteps, double lr, double b1, double b2, double eps, double* losses_host) {
    Engine* E = (Engine*)h;
    // losses go through a device ring of `ring` steps, copied to the host in stream order as rows fill up
    int ring = (nsteps < 512) ? nsteps : 512;
    if (ring < 1) ring = 1;
    if (E->loss_cap < ring) {
        if (E->loss) CK(hipFree(E->loss));
        CK(dalloc(&E->loss, (size_t)ring * E->M));
        E->loss_cap = ring;
    }
    if (E->level == 2 && E->seq == 0) {
        stage_chunk(E, 0);
        stage_chunk(E, 1);
    }
    double omb1 = 1.0 - b1, omb2 = 1.0 - b2;
    // the subsets of step + 1 are drawn on their own stream while step runs (double-buffered A_pad / t)
    if (nsteps > 0) {
        int p0 = step0 & 1;
        rowstate_kernel<<<(E->S * E->B + 255) / 256, 256, 0, E->st4>>>(E->pcg, step0, E->rowst, E->B, E->S, E->n);
        sample_wave_kernel<<<(E->S * E->B + 3) / 4, 256, 0, E->st4>>>(E->rowst, E->jtab, E->b_info, E->x, E->Ab[p0],
                                                                     E->tb[p0], E->B, E->S, E->n, E->N_tr, E->W_u64);
        (void)hipEventRecord(E->ev_samp[p0], E->st4);
    }
    for (int it = 0; it < nsteps; ++it) {
        int step = step0 + it;
        int p = step & 1;
        if (it + 1 < nsteps) {
            (void)hipStreamWaitEvent(E->st4, E->ev_used[p ^ 1], 0);
            rowstate_kernel<<<(E->S * E->B + 255) / 256, 256, 0, E->st4>>>(E->pcg, step + 1, E->rowst, E->B, E->S,
                                                                          E->n);
            sample_wave_kernel<<<(E->S * E->B + 3) / 4, 256, 0, E->st4>>>(E->rowst, E->jtab, E->b_info, E->x,
                                                                         E->Ab[p ^ 1], E->tb[p ^ 1], E->B, E->S,
                                                                         E->n, E->N_tr, E->W_u64);
            (void)hipEventRecord(E->ev_samp[p ^ 1], E->st4);
        }
        (void)hipStreamWaitEvent(E->st, E->ev_samp[p], 0);
        E->A_pad = E->Ab[p];
        E->t = E->tb[p];
        double cnt = (double)(step + 1);
        double bc1 = 1.0 - pow(b1, cnt), bc2 = 1.0 - pow(b2, cnt);
        double* loss_row = E->loss + (size_t)(it % ring) * E->M;
        (void)hipMemsetAsync(loss_row, 0, (size_t)E->M * 8, E->st);
        for (int k = 0; k < E->nchunk; ++k) {
            int m0 = k * E->Mc;
            int mc = (E->M - m0 < E->Mc) ? (E->M - m0) : E->Mc;
            switch (E->L) {
                case 1:  launch_chunk<1>(E, k, m0, mc, -lr, b1, b2, omb1, omb2, eps, bc1, bc2, loss_row); break;
                case 2:  launch_chunk<2>(E, k, m0, mc, -lr, b1, b2, omb1, omb2, eps, bc1, bc2, loss_row); break;
                case 4:  launch_chunk<4>(E, k, m0, mc, -lr, b1, b2, omb1, omb2, eps, bc1, bc2, loss_row); break;
                case 8:  launch_chunk<8>(E, k, m0, mc, -lr, b1, b2, omb1, omb2, eps, bc1, bc2, loss_row); break;
                default: launch_chunk<16>(E, k, m0, mc, -lr, b1, b2, omb1, omb2, eps, bc1, bc2, loss_row); break;
            }
        }
        // the weights of every model move once per step, after all chunks used them (same stream, in order)
        if (E->beta)
            piadam_kernel<<<(E->M + 63) / 64, 64, 0, E->st>>>(E->beta, E->mb, E->vb, E->gpi, E->pi, E->M, E->L,
                                                              -lr * E->pi_lr_scale, b1, b2, omb1, omb2, eps, bc1,
                                                              bc2);
        (void)hipEventRecord(E->ev_used[p], E->st);
        if ((it + 1) % ring == 0 || it + 1 == nsteps) {
            int r0 = it - it % ring;
            (void)hipMemcpyAsync(losses_host + (size_t)r0 * E->M, E->loss, (size_t)(it - r0 + 1) * E->M * 8,
                                 hipMemcpyDeviceToHost, E->st);
        }
    }
    CK(hipGetLastError());
    CK(hipStreamSynchronize(E->st4));
    if (E->cpu_frac > 0.0) wait_done(E, E->seq - 1);
    CK(hipStreamSynchronize(E->st2));
    CK(hipStreamSynchronize(E->st3));
    CK(hipStreamSynchronize(E->st));
    return 0;
}

extern "C" int eng_get_params(void* h, double* out) {
    Engine* E = (Engine*)h;
    CK(hipDeviceSynchronize());
    size_t bytes = (size_t)E->M * E->L * E->G * 8;
    if (E->level == 2) memcpy(out, E->c, bytes);
    else CK(hipMemcpy(out, E->c, bytes, hipMemcpyDeviceToHost));
    return 0;
}

// switch on trainable branch weights: pi = softmax(beta), beta initialized to log pi (every pi must be > 0)
extern "C" int eng_train_pi(void* h, double lr_scale) {
    Engine* E = (Engine*)h;
    size_t ml = (size_t)E->M * E->L;
    double* hp = new double[ml];
    if (hipMemcpy(hp, E->pi, ml * 8, hipMemcpyDeviceToHost) != hipSuccess) { delete[] hp; return -1; }
    for (size_t i = 0; i < ml; ++i) {
        if (!(hp[i] > 0.0)) { delete[] hp; fprintf(stderr, "eng_train_pi: every weight must be > 0\n"); return -1; }
        hp[i] = log(hp[i]);
    }
    E->pi_lr_scale = lr_scale;
    CK(dalloc(&E->beta, ml)); CK(dalloc(&E->mb, ml)); CK(dalloc(&E->vb, ml)); CK(dalloc(&E->gpi, ml));
    CK(hipMemcpy(E->beta, hp, ml * 8, hipMemcpyHostToDevice));
    delete[] hp;
    CK(hipMemset(E->mb, 0, ml * 8)); CK(hipMemset(E->vb, 0, ml * 8)); CK(hipMemset(E->gpi, 0, ml * 8));
    return 0;
}

extern "C" int eng_get_pi(void* h, double* out) {
    Engine* E = (Engine*)h;
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out, E->pi, (size_t)E->M * E->L * 8, hipMemcpyDeviceToHost));
    return 0;
}

extern "C" int eng_plan(void* h, int* out) {   // the plan actually run: offload level, models per chunk, branch parts
    Engine* E = (Engine*)h;
    out[0] = E->level; out[1] = E->Mc; out[2] = E->H;
    return 0;
}

#define DESTROY_EV(e) do { if (e) (void)hipEventDestroy(e); } while (0)
#define DESTROY_ST(q) do { if (q) (void)hipStreamDestroy(q); } while (0)

extern "C" int eng_destroy(void* h) {
    Engine* E = (Engine*)h;
    if (!E) return 0;
    if (E->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(E->mtx);
            E->stop = true;
        }
        E->cv.notify_all();
        E->worker.join();
    }
    (void)hipDeviceSynchronize();
    for (int i = 0; i < 2; ++i) {
        if (E->g_host[i]) (void)hipHostFree(E->g_host[i]);
        DESTROY_EV(E->ev_gh[i]);
    }
    if (E->level == 2) { if (E->c) (void)hipHostFree(E->c); } else if (E->c) (void)hipFree(E->c);
    if (E->level >= 1) { if (E->mu) (void)hipHostFree(E->mu); if (E->nu) (void)hipHostFree(E->nu); }
    else { if (E->mu) (void)hipFree(E->mu); if (E->nu) (void)hipFree(E->nu); }
    if (E->gbs[1] && E->gbs[1] != E->gbs[0]) (void)hipFree(E->gbs[1]);
    double* bufs[] = {E->gbs[0], E->stg[0], E->stg[1], E->pi, E->w_of_b, E->tb[0], E->tb[1], E->Th, E->dTh, E->z, E->dz,
                      E->P, E->Dq, E->loss, E->beta, E->mb, E->vb, E->gpi};
    for (double* p : bufs) if (p) (void)hipFree(p);
    void* misc[] = {E->pcg, E->jtab, E->x, E->rowst, E->b_info, E->Ab[0], E->Ab[1], E->eidx};
    for (void* p : misc) if (p) (void)hipFree(p);
    for (int i = 0; i < 2; ++i) { DESTROY_EV(E->ev_samp[i]); DESTROY_EV(E->ev_used[i]); }
    DESTROY_ST(E->st4);
    if (E->ev_c) {
        for (int k = 0; k < E->nchunk; ++k) DESTROY_EV(E->ev_c[k]);
        delete[] E->ev_c;
    }
    DESTROY_EV(E->ev_fold);
    DESTROY_EV(E->ev_adam);
    DESTROY_EV(E->ev_adamb[0]); DESTROY_EV(E->ev_adamb[1]);
    for (int i = 0; i < 16; ++i) DESTROY_EV(E->ev_grp[i]);
    for (int i = 0; i < 2; ++i) DESTROY_EV(E->ev_stage[i]);
    DESTROY_ST(E->st3); DESTROY_ST(E->st2); DESTROY_ST(E->st);
    delete E;
    return 0;
}
"""

_LIB = None



_SEP = "// ------------------------------------------------------------------------------------------------\n"


def _split_sources(src):
    """Three translation units compiled in parallel: the forward kernel, the backward kernel, everything else (which
    launches the two through plain declarations)."""
    import re
    i_pcg = src.index(_SEP + "// PCG64")
    i_fwd = src.index(_SEP + "// Forward:")
    i_loss = src.index(_SEP + "// Loss + dL/dz")
    i_bwd = src.index(_SEP + "// Backward:")
    i_adam = src.index("// Adam (optax) on a contiguous range;")
    common, pcg, fwd, loss, bwd, rest = (src[:i_pcg], src[i_pcg:i_fwd], src[i_fwd:i_loss], src[i_loss:i_bwd],
                                         src[i_bwd:i_adam], src[i_adam:])
    bs_a = fwd.index("DEVI double block_sum_256(")
    bs_b = fwd.index("\n}\n", bs_a) + 3
    common += fwd[bs_a:bs_b]
    fwd = fwd[:bs_a] + fwd[bs_b:]

    def decl(part, name):
        m = re.search(r"__global__[^{;]*?void " + name + r"\(([^{]*?)\)\s*\{", part, re.S)
        return "__global__ void " + name + "(" + m.group(1) + ");\n"
    return [common + fwd, common + bwd, common + decl(fwd, "fwd_kernel") + decl(bwd, "bwd_kernel") + pcg + loss + rest]


ARCH = os.environ.get("IQP_FASTCF_ARCH", "gfx942")    # GPU target; developed and verified on MI300X (gfx942) only
HIPCC = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")


def _build(so_tmp, base):
    flags = ["-O3", "-fPIC", f"--offload-arch={ARCH}", "-munsafe-fp-atomics", "-fopenmp", "-Wno-unused-result"]
    hipcc = HIPCC
    try:
        parts = _split_sources(HIP_SRC)
        srcs, objs = [f"{base}_{i}.cpp" for i in range(3)], [f"{base}_{i}.o" for i in range(3)]
        for path, text in zip(srcs, parts):
            with open(path, "w") as f:
                f.write(text)
        procs = [subprocess.Popen([hipcc, *flags, "-c", s_, "-o", o_]) for s_, o_ in zip(srcs, objs)]
        if any(pr.wait() != 0 for pr in procs):
            raise RuntimeError("split build failed")
        subprocess.run([hipcc, "-shared", "-fPIC", f"--offload-arch={ARCH}", "-fopenmp", *objs, "-o", so_tmp],
                       check=True)
        tmp_files = srcs + objs
    except (ValueError, AttributeError, RuntimeError, subprocess.CalledProcessError, OSError):
        src = f"{base}.cpp"
        with open(src, "w") as f:
            f.write(HIP_SRC)
        subprocess.run([hipcc, *flags, "-shared", src, "-o", so_tmp], check=True)
        tmp_files = [src]
    for path in tmp_files:
        try:
            os.remove(path)
        except OSError:
            pass


def load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    h = hashlib.sha256((HIP_SRC + ARCH).encode()).hexdigest()[:16]
    tmp = tempfile.gettempdir()
    so_path = os.path.join(tmp, f"iqp_fastcf_{h}.so")
    if not os.path.exists(so_path):
        base = os.path.join(tmp, f"iqp_fastcf_{h}_{os.getpid()}")
        so_tmp = base + ".so"
        _build(so_tmp, base)
        os.replace(so_tmp, so_path)
    lib = ctypes.CDLL(so_path)
    P = ctypes.c_void_p
    lib.eng_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                               P, P, P, P, P, P, P, P]
    lib.eng_run.argtypes = [P, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, P]
    lib.eng_get_params.argtypes = [P, P]
    lib.eng_destroy.argtypes = [P]
    lib.eng_plan.argtypes = [P, P]
    lib.eng_train_pi.argtypes = [P, ctypes.c_double]
    lib.eng_get_pi.argtypes = [P, P]
    for f in (lib.eng_create, lib.eng_run, lib.eng_get_params, lib.eng_destroy, lib.eng_plan, lib.eng_train_pi,
              lib.eng_get_pi):
        f.restype = ctypes.c_int
    _LIB = lib
    return lib
