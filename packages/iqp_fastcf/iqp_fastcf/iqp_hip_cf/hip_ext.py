#!/usr/bin/env python3
r"""HIP/ROCm C++ extension and JAX FFI bindings for `iqp_hip_cf`.

Provides:
1. `cf_sample_moments`: 100% GPU 128-bit PCG64 jump-ahead subset sampler + bit-packed uint64 XOR/popcount target moments.
2. `cf_mix_fwd` / `cf_mix_eval_only`: Tiled (TILE_S = 64) forward mixed correlator z_A with 2-level 16-sign trigonometric
   angle-addition (`sincos`) and bank-conflict-free LDS reductions for |A| = 1..12.
3. `cf_mix_bwd`: Per-subset backward gradient kernel with 2-level 16-sign angle-addition and 4-way Montgomery batch
   inversion (`1.0 / (c01 * c23)`) for |A| = 1..12.
4. `pack_ops_and_moments`: OpenMP C++ helper to pack arbitrary Python subset lists `ops` into fixed-capacity padded buffers
   and compute exact empirical moments via hardware `__builtin_popcountll`.
"""
import ctypes, hashlib, os, subprocess, tempfile
import numpy as np, jax

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" int pack_ops_moments_omp(
    const int* __restrict__ flat_indices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    int num_valid,
    int B_cap,
    int T_cap,
    int n,
    int N_data,
    int W_u64,
    const uint64_t* __restrict__ data_u64,
    int* __restrict__ A_pad,
    int* __restrict__ b_info,
    int* __restrict__ tiles,
    double* __restrict__ t_out) {

    // Prefix sum for p_offset and tile_offset per valid subset
    int p_off = 0;
    int t_off = 0;
    for (int b = 0; b < num_valid; ++b) {
        int r = lengths[b];
        int S = 1 << (r - 1);
        int n_tiles = (S + 63) >> 6;
        if (t_off + n_tiles > T_cap) {
            return -1;
        }
        b_info[b * 2] = r;
        b_info[b * 2 + 1] = p_off;
        for (int ti = 0; ti < n_tiles; ++ti) {
            int s_start = ti << 6;
            int s_cnt = (S - s_start < 64) ? (S - s_start) : 64;
            int* tp = tiles + (t_off + ti) * 5;
            tp[0] = b;
            tp[1] = r;
            tp[2] = s_start;
            tp[3] = s_cnt;
            tp[4] = p_off + s_start;
        }
        p_off += S;
        t_off += n_tiles;
    }

    for (int b = num_valid; b < B_cap; ++b) {
        b_info[b * 2] = 0;
        b_info[b * 2 + 1] = 0;
        t_out[b] = 0.0;
    }
    for (int ti = t_off; ti < T_cap; ++ti) {
        int* tp = tiles + ti * 5;
        tp[0] = -1;
        tp[1] = 0;
        tp[2] = 0;
        tp[3] = 0;
        tp[4] = 0;
    }

    #pragma omp parallel for schedule(static) if(num_valid > 64)
    for (int b = 0; b < num_valid; ++b) {
        int r = lengths[b];
        const int* src = flat_indices + offsets[b];
        int* dst = A_pad + b * 12;
        for (int k = 0; k < r; ++k) dst[k] = src[k];
        for (int k = r; k < 12; ++k) dst[k] = n;

        int ones = 0;
        for (int w = 0; w < W_u64; ++w) {
            uint64_t xored = 0ULL;
            for (int k = 0; k < r; ++k) {
                xored ^= data_u64[(size_t)src[k] * W_u64 + w];
            }
            ones += __builtin_popcountll(xored);
        }
        t_out[b] = (double)(N_data - 2 * ones) / (double)N_data;
    }
    return p_off;
}

__device__ __forceinline__ __uint128_t pcg_advance_lcg_128(__uint128_t state, uint64_t delta, __uint128_t cur_mult, __uint128_t cur_plus) {
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

__device__ __forceinline__ uint64_t pcg_xsl_rr(__uint128_t s) {
    uint64_t xorshifted = uint64_t(s >> 64) ^ uint64_t(s);
    uint32_t rot = uint32_t(s >> 122);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 63U));
}

__global__ __launch_bounds__(512)
void sample_and_moments_kernel(const uint64_t* __restrict__ pcg_init,
                               const uint64_t* __restrict__ j_tab,
                               const int* __restrict__ step_ptr,
                               const int* __restrict__ b_info,
                               const uint64_t* __restrict__ xtr_u64,
                               int* __restrict__ A_pad,
                               double* __restrict__ t_out,
                               int B_tot, int n, int N_tr, int W_u64) {
    __shared__ uint64_t s_row_state[2];
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

    const uint64_t* pcg_s = pcg_init + (size_t)s * 4;
    const uint64_t* j_tab_s = j_tab + (size_t)s * n * 4;
    int* A_pad_sb = A_pad + ((size_t)s * B_tot + b) * 10;

    if (tid == 0) {
        __uint128_t state0 = (__uint128_t(pcg_s[1]) << 64) | pcg_s[0];
        __uint128_t inc    = (__uint128_t(pcg_s[3]) << 64) | pcg_s[2];
        constexpr __uint128_t MUL = (__uint128_t(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
        uint64_t step = (uint64_t)(*step_ptr);
        uint64_t row_offset = (step * (uint64_t)B_tot + (uint64_t)b) * (uint64_t)n;
        __uint128_t s_row = pcg_advance_lcg_128(state0, row_offset, MUL, inc);
        s_row_state[0] = uint64_t(s_row);
        s_row_state[1] = uint64_t(s_row >> 64);
    }
    __syncthreads();

    __uint128_t s_row = (__uint128_t(s_row_state[1]) << 64) | s_row_state[0];

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
    if (tid >= r && tid < 10) {
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

__device__ __forceinline__ double fast_cos(double x) {
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
    return (((long long)k) & 1) ? -c : c;
}

__device__ __forceinline__ void fast_sincos(double x, double* s_out, double* c_out) {
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
    long long ki = (long long)k;
    bool odd = (ki & 1) != 0;
    double cv = odd ? s : c;
    double sv = odd ? c : s;
    *c_out = ((ki + 1) & 2) ? -cv : cv;
    *s_out = (ki & 2) ? -sv : sv;
}

__device__ __forceinline__ double fast_tan(double x) {
    constexpr double TWO_OVER_PI = 0.63661977236758134308;
    constexpr double PIO2_HI = 1.57079632679489655800;
    constexpr double PIO2_LO = 6.12323399573676603587e-17;
    double k = rint(x * TWO_OVER_PI);
    double r = fma(k, -PIO2_LO, fma(k, -PIO2_HI, x));
    double r2 = r * r;
    double num = r * fma(r2, fma(r2, fma(r2, r2 - 990.0, 135135.0), -4729725.0), 34459425.0);
    double den = fma(r2, fma(r2, fma(r2, 45.0 * r2 - 13860.0, 945945.0), -16216200.0), 34459425.0);
    bool odd = (((long long)k) & 1) != 0;
    double a = odd ? -den : num;
    double b = odd ? num : den;
    return a / b;
}

template <int BLOCK, int R>
__device__ __forceinline__ void fwd_eval_r(
    const double* __restrict__ Th_ml_j,
    const int* __restrict__ s_A,
    const double* __restrict__ s_th1,
    double* __restrict__ s_c,
    double s_wave_prod[BLOCK / 64][64],
    int j, int n, int s_start) {

    constexpr int S_COUNT = ((1 << (R - 1)) < 64) ? (1 << (R - 1)) : 64;

    if (j < S_COUNT) {
        int s = s_start + j;
        double a1 = s_th1[0];
        #pragma unroll
        for (int u = 1; u < R; ++u) {
            a1 += ((s >> (R - 1 - u)) & 1) ? -s_th1[u] : s_th1[u];
        }
        s_c[j] = fast_cos(a1);
    }

    bool active = (j < n);
    double x[R];
    if (active) {
        #pragma unroll
        for (int u = 0; u < R; ++u) {
            if (j == s_A[u]) active = false;
        }
        if (active) {
            #pragma unroll
            for (int u = 0; u < R; ++u) {
                x[u] = Th_ml_j[(size_t)s_A[u] * n];
            }
        }
    }

    int wave_id = j >> 6;
    int lane = j & 63;

    if constexpr (R >= 5) {
        double C0 = 1.0, S0 = 0.0, C1 = 1.0, S1 = 0.0;
        double CA0 = 1.0, SA0 = 0.0, CA1 = 1.0, SA1 = 0.0;
        if (active) {
            fast_sincos(x[R - 2] + x[R - 1], &S0, &C0);
            fast_sincos(x[R - 2] - x[R - 1], &S1, &C1);
            fast_sincos(x[R - 4] + x[R - 3], &SA0, &CA0);
            fast_sincos(x[R - 4] - x[R - 3], &SA1, &CA1);
        }
        for (int blk = 0; blk < S_COUNT; blk += 16) {
            int s_blk = s_start + blk;
            double st = 0.0, ct = 1.0;
            if (active) {
                double base_top = x[0];
                #pragma unroll
                for (int u = 1; u < R - 4; ++u) {
                    base_top += ((s_blk >> (R - 1 - u)) & 1) ? -x[u] : x[u];
                }
                fast_sincos(base_top, &st, &ct);
            }
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int idx = blk + (sub << 2);
                double v0 = 1.0, v1 = 1.0, v2 = 1.0, v3 = 1.0;
                if (active) {
                    double ca = (sub == 0 || sub == 3) ? CA0 : CA1;
                    double sa = (sub == 0) ? SA0 : ((sub == 1) ? SA1 : ((sub == 2) ? -SA1 : -SA0));
                    double cb = fma(-st, sa, ct * ca);
                    double sb = fma( ct, sa, st * ca);
                    v0 = fma(-sb, S0, cb * C0);
                    v1 = fma(-sb, S1, cb * C1);
                    v2 = fma( sb, S1, cb * C1);
                    v3 = fma( sb, S0, cb * C0);
                }
                #pragma unroll
                for (int offset = 32; offset > 0; offset >>= 1) {
                    v0 *= __shfl_xor(v0, offset, 64);
                    v1 *= __shfl_xor(v1, offset, 64);
                    v2 *= __shfl_xor(v2, offset, 64);
                    v3 *= __shfl_xor(v3, offset, 64);
                }
                if (lane == 0) {
                    s_wave_prod[wave_id][idx]     = v0;
                    s_wave_prod[wave_id][idx + 1] = v1;
                    s_wave_prod[wave_id][idx + 2] = v2;
                    s_wave_prod[wave_id][idx + 3] = v3;
                }
            }
        }
    } else if constexpr (R >= 3) {
        double C0 = 1.0, S0 = 0.0, C1 = 1.0, S1 = 0.0;
        if (active) {
            fast_sincos(x[R - 2] + x[R - 1], &S0, &C0);
            fast_sincos(x[R - 2] - x[R - 1], &S1, &C1);
        }
        for (int idx = 0; idx < S_COUNT; idx += 4) {
            int s = s_start + idx;
            double v0 = 1.0, v1 = 1.0, v2 = 1.0, v3 = 1.0;
            if (active) {
                double base = x[0];
                #pragma unroll
                for (int u = 1; u < R - 2; ++u) {
                    base += ((s >> (R - 1 - u)) & 1) ? -x[u] : x[u];
                }
                double sb, cb;
                fast_sincos(base, &sb, &cb);
                v0 = fma(-sb, S0, cb * C0);
                v1 = fma(-sb, S1, cb * C1);
                v2 = fma( sb, S1, cb * C1);
                v3 = fma( sb, S0, cb * C0);
            }
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                v0 *= __shfl_xor(v0, offset, 64);
                v1 *= __shfl_xor(v1, offset, 64);
                v2 *= __shfl_xor(v2, offset, 64);
                v3 *= __shfl_xor(v3, offset, 64);
            }
            if (lane == 0) {
                s_wave_prod[wave_id][idx]     = v0;
                s_wave_prod[wave_id][idx + 1] = v1;
                s_wave_prod[wave_id][idx + 2] = v2;
                s_wave_prod[wave_id][idx + 3] = v3;
            }
        }
    } else {
        for (int idx = 0; idx < S_COUNT; ++idx) {
            int s = s_start + idx;
            double val = 1.0;
            if (active) {
                double arg = x[0];
                #pragma unroll
                for (int u = 1; u < R; ++u) {
                    arg += ((s >> (R - 1 - u)) & 1) ? -x[u] : x[u];
                }
                val = fast_cos(arg);
            }
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                val *= __shfl_xor(val, offset, 64);
            }
            if (lane == 0) {
                s_wave_prod[wave_id][idx] = val;
            }
        }
    }
}

template <int BLOCK, bool SAVE_P>
__global__ __launch_bounds__(BLOCK)
void mix_corr_fwd_kernel(const double* __restrict__ th1,
                         const double* __restrict__ Th,
                         const double* __restrict__ pi,
                         const int* __restrict__ A_pad,
                         const int* __restrict__ tiles,
                         double* __restrict__ z_out,
                         double* __restrict__ p_out,
                         int n, int L, int B_tot, int S_tot, int a_stride, int M_per_seed) {
    constexpr int NWAVES = BLOCK / 64;
    __shared__ int s_A[12];
    __shared__ double s_th1[12];
    __shared__ double s_c[64];
    __shared__ double s_wave_prod[NWAVES][64];

    int tile_id = (gridDim.x - 1) - blockIdx.x;
    const int* t_ptr = tiles + tile_id * 5;
    int b = t_ptr[0];
    if (b < 0) return;

    int r = t_ptr[1];
    int s_start = t_ptr[2];
    int s_count = t_ptr[3];
    int p_offset = t_ptr[4];
    int m = blockIdx.y;
    int l = blockIdx.z;
    int j = threadIdx.x;
    int ml = m * L + l;
    int s_seed = m / M_per_seed;

    if (j < r) {
        int a_u = A_pad[((size_t)s_seed * B_tot + b) * a_stride + j];
        s_A[j] = a_u;
        s_th1[j] = th1[(size_t)ml * n + a_u];
    }
    __syncthreads();

    const double* Th_ml_j = Th + ((size_t)ml * n) * n + j;
    switch (r) {
        case 1:  fwd_eval_r<BLOCK, 1>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 2:  fwd_eval_r<BLOCK, 2>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 3:  fwd_eval_r<BLOCK, 3>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 4:  fwd_eval_r<BLOCK, 4>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 5:  fwd_eval_r<BLOCK, 5>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 6:  fwd_eval_r<BLOCK, 6>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 7:  fwd_eval_r<BLOCK, 7>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 8:  fwd_eval_r<BLOCK, 8>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 9:  fwd_eval_r<BLOCK, 9>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 10: fwd_eval_r<BLOCK, 10>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        case 11: fwd_eval_r<BLOCK, 11>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
        default: fwd_eval_r<BLOCK, 12>(Th_ml_j, s_A, s_th1, s_c, s_wave_prod, j, n, s_start); break;
    }
    __syncthreads();

    if ((j >> 6) == 0) {
        double cp = 0.0;
        if (j < s_count) {
            double p = s_wave_prod[0][j];
            #pragma unroll
            for (int w = 1; w < NWAVES; ++w) {
                p *= s_wave_prod[w][j];
            }
            if constexpr (SAVE_P) {
                p_out[(size_t)ml * S_tot + p_offset + j] = p;
            }
            cp = s_c[j] * p;
        }
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            cp += __shfl_xor(cp, offset, 64);
        }
        if (j == 0) {
            double scale = pi[ml] / (double)(1 << (r - 1));
            atomicAdd(&z_out[m * B_tot + b], scale * cp);
        }
    }
}

template <int R>
__device__ __forceinline__ void bwd_eval_r(
    const double* __restrict__ Th_ml_j,
    double* __restrict__ d_Th_ml_j,
    double* __restrict__ d_th1_ml,
    const double* __restrict__ p_ml_base,
    const int* __restrict__ s_A,
    const double* __restrict__ s_th1,
    double* __restrict__ s_coeff,
    double* __restrict__ s_da1,
    double w_mlb, int j, int n) {

    constexpr int S_ALL = 1 << (R - 1);
    bool active = (j < n);
    double x[R], dx[R];
    double C0 = 1.0, S0 = 0.0, C1 = 1.0, S1 = 0.0;
    double CA0 = 1.0, SA0 = 0.0, CA1 = 1.0, SA1 = 0.0;
    if (active) {
        #pragma unroll
        for (int u = 0; u < R; ++u) {
            if (j == s_A[u]) active = false;
        }
        if (active) {
            #pragma unroll
            for (int u = 0; u < R; ++u) {
                x[u] = Th_ml_j[(size_t)s_A[u] * n];
                dx[u] = 0.0;
            }
            if constexpr (R >= 3) {
                fast_sincos(x[R - 2] + x[R - 1], &S0, &C0);
                fast_sincos(x[R - 2] - x[R - 1], &S1, &C1);
            }
            if constexpr (R >= 5) {
                fast_sincos(x[R - 4] + x[R - 3], &SA0, &CA0);
                fast_sincos(x[R - 4] - x[R - 3], &SA1, &CA1);
            }
        }
    }

    double d1 = 0.0;
    for (int s_start = 0; s_start < S_ALL; s_start += 64) {
        constexpr int s_count = (S_ALL < 64) ? S_ALL : 64;
        if (j < s_count) {
            int s = s_start + j;
            double a1 = s_th1[0];
            #pragma unroll
            for (int u = 1; u < R; ++u) {
                a1 += ((s >> (R - 1 - u)) & 1) ? -s_th1[u] : s_th1[u];
            }
            double p = p_ml_base[s_start + j];
            double sin_a1, cos_a1;
            fast_sincos(a1, &sin_a1, &cos_a1);
            s_coeff[j] = -w_mlb * cos_a1 * p;
            s_da1[j] = -w_mlb * sin_a1 * p;
        }
        __syncthreads();

        if (j < R) {
            for (int idx = 0; idx < s_count; ++idx) {
                int s = s_start + idx;
                bool neg = (j > 0) && (((s >> (R - 1 - j)) & 1) != 0);
                d1 += neg ? -s_da1[idx] : s_da1[idx];
            }
        }

        if (active) {
            if constexpr (R >= 5) {
                for (int blk = 0; blk < s_count; blk += 16) {
                    int s_blk = s_start + blk;
                    double base_top = x[0];
                    #pragma unroll
                    for (int u = 1; u < R - 4; ++u) {
                        base_top += ((s_blk >> (R - 1 - u)) & 1) ? -x[u] : x[u];
                    }
                    double st, ct;
                    fast_sincos(base_top, &st, &ct);
                    double blk_all = 0.0;
                    double blk_3 = 0.0;
                    #pragma unroll
                    for (int sub = 0; sub < 4; ++sub) {
                        int idx = blk + (sub << 2);
                        double ca = (sub == 0 || sub == 3) ? CA0 : CA1;
                        double sa = (sub == 0) ? SA0 : ((sub == 1) ? SA1 : ((sub == 2) ? -SA1 : -SA0));
                        double cb = fma(-st, sa, ct * ca);
                        double sb = fma( ct, sa, st * ca);

                        double c0 = fma(-sb, S0, cb * C0);
                        double s0 = fma( cb, S0, sb * C0);
                        double c1 = fma(-sb, S1, cb * C1);
                        double s1 = fma( cb, S1, sb * C1);
                        double c2 = fma( sb, S1, cb * C1);
                        double s2 = fma(-cb, S1, sb * C1);
                        double c3 = fma( sb, S0, cb * C0);
                        double s3 = fma(-cb, S0, sb * C0);

                        double c01 = c0 * c1;
                        double c23 = c2 * c3;
                        double inv = 1.0 / (c01 * c23);
                        double inv01 = c23 * inv;
                        double inv23 = c01 * inv;

                        double g0 = s_coeff[idx]     * (s0 * c1 * inv01);
                        double g1 = s_coeff[idx + 1] * (s1 * c0 * inv01);
                        double g2 = s_coeff[idx + 2] * (s2 * c3 * inv23);
                        double g3 = s_coeff[idx + 3] * (s3 * c2 * inv23);

                        double g01 = g0 + g1;
                        double g23 = g2 + g3;
                        double g_sum = g01 + g23;
                        dx[R - 2] += g01 - g23;
                        dx[R - 1] += (g0 - g1) + (g2 - g3);
                        blk_all += g_sum;
                        blk_3   += (sub & 1) ? -g_sum : g_sum;
                        dx[R - 4] += (sub & 2) ? -g_sum : g_sum;
                    }
                    dx[R - 3] += blk_3;
                    dx[0] += blk_all;
                    #pragma unroll
                    for (int u = 1; u < R - 4; ++u) {
                        dx[u] += ((s_blk >> (R - 1 - u)) & 1) ? -blk_all : blk_all;
                    }
                }
            } else if constexpr (R >= 3) {
                for (int idx = 0; idx < s_count; idx += 4) {
                    int s = s_start + idx;
                    double base = x[0];
                    #pragma unroll
                    for (int u = 1; u < R - 2; ++u) {
                        base += ((s >> (R - 1 - u)) & 1) ? -x[u] : x[u];
                    }
                    double sb, cb;
                    fast_sincos(base, &sb, &cb);
                    double c0 = fma(-sb, S0, cb * C0);
                    double s0 = fma( cb, S0, sb * C0);
                    double c1 = fma(-sb, S1, cb * C1);
                    double s1 = fma( cb, S1, sb * C1);
                    double c2 = fma( sb, S1, cb * C1);
                    double s2 = fma(-cb, S1, sb * C1);
                    double c3 = fma( sb, S0, cb * C0);
                    double s3 = fma(-cb, S0, sb * C0);

                    double c01 = c0 * c1;
                    double c23 = c2 * c3;
                    double inv = 1.0 / (c01 * c23);
                    double inv01 = c23 * inv;
                    double inv23 = c01 * inv;

                    double g0 = s_coeff[idx]     * (s0 * c1 * inv01);
                    double g1 = s_coeff[idx + 1] * (s1 * c0 * inv01);
                    double g2 = s_coeff[idx + 2] * (s2 * c3 * inv23);
                    double g3 = s_coeff[idx + 3] * (s3 * c2 * inv23);

                    double g01 = g0 + g1;
                    double g23 = g2 + g3;
                    double g_sum = g01 + g23;
                    dx[0] += g_sum;
                    #pragma unroll
                    for (int u = 1; u < R - 2; ++u) {
                        dx[u] += ((s >> (R - 1 - u)) & 1) ? -g_sum : g_sum;
                    }
                    dx[R - 2] += g01 - g23;
                    dx[R - 1] += (g0 - g1) + (g2 - g3);
                }
            } else {
                for (int idx = 0; idx < s_count; ++idx) {
                    int s = s_start + idx;
                    double arg = x[0];
                    #pragma unroll
                    for (int u = 1; u < R; ++u) {
                        arg += ((s >> (R - 1 - u)) & 1) ? -x[u] : x[u];
                    }
                    double g = s_coeff[idx] * fast_tan(arg);
                    dx[0] += g;
                    #pragma unroll
                    for (int u = 1; u < R; ++u) {
                        dx[u] += ((s >> (R - 1 - u)) & 1) ? -g : g;
                    }
                }
            }
        }
        if constexpr (S_ALL > 64) {
            __syncthreads();
        }
    }

    if (j < R) {
        atomicAdd(&d_th1_ml[s_A[j]], d1);
    }
    if (active) {
        #pragma unroll
        for (int u = 0; u < R; ++u) {
            atomicAdd(&d_Th_ml_j[(size_t)s_A[u] * n], dx[u]);
        }
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 3)
void mix_corr_bwd_kernel(const double* __restrict__ th1,
                         const double* __restrict__ Th,
                         const double* __restrict__ pi,
                         const int* __restrict__ A_pad,
                         const int* __restrict__ b_info,
                         const double* __restrict__ p_out,
                         const double* __restrict__ dz,
                         double* __restrict__ d_th1,
                         double* __restrict__ d_Th,
                         int n, int L, int B_tot, int S_tot, int a_stride, int M_per_seed) {
    __shared__ int s_A[12];
    __shared__ double s_th1[12];
    __shared__ double s_coeff[64];
    __shared__ double s_da1[64];

    int b = (gridDim.x - 1) - blockIdx.x;
    int r = b_info[b * 2];
    if (r <= 0) return;

    int p_offset = b_info[b * 2 + 1];
    int m = blockIdx.y;
    int l = blockIdx.z;
    int j = threadIdx.x;
    int ml = m * L + l;
    int s_seed = m / M_per_seed;

    if (j < r) {
        int a_u = A_pad[((size_t)s_seed * B_tot + b) * a_stride + j];
        s_A[j] = a_u;
        s_th1[j] = th1[(size_t)ml * n + a_u];
    }
    __syncthreads();

    double w_mlb = dz[m * B_tot + b] * (pi[ml] / (double)(1 << (r - 1)));
    const double* Th_ml_j = Th + ((size_t)ml * n) * n + j;
    double* d_Th_ml_j = d_Th + ((size_t)ml * n) * n + j;
    double* d_th1_ml = d_th1 + (size_t)ml * n;
    const double* p_ml = p_out + (size_t)ml * S_tot + p_offset;

    switch (r) {
        case 1:  bwd_eval_r<1>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 2:  bwd_eval_r<2>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 3:  bwd_eval_r<3>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 4:  bwd_eval_r<4>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 5:  bwd_eval_r<5>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 6:  bwd_eval_r<6>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 7:  bwd_eval_r<7>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 8:  bwd_eval_r<8>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 9:  bwd_eval_r<9>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 10: bwd_eval_r<10>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        case 11: bwd_eval_r<11>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
        default: bwd_eval_r<12>(Th_ml_j, d_Th_ml_j, d_th1_ml, p_ml, s_A, s_th1, s_coeff, s_da1, w_mlb, j, n); break;
    }
}

ffi::Error cf_sample_moments_impl(hipStream_t stream,
                                  ffi::Buffer<ffi::U64> pcg_init,
                                  ffi::Buffer<ffi::U64> j_tab,
                                  ffi::Buffer<ffi::S32> step_idx,
                                  ffi::Buffer<ffi::S32> b_info,
                                  ffi::Buffer<ffi::U64> xtr_u64,
                                  int64_t N_tr,
                                  ffi::ResultBuffer<ffi::S32> A_pad,
                                  ffi::ResultBuffer<ffi::F64> t_out) {
    int B_tot = b_info.dimensions()[0];
    int n = xtr_u64.dimensions()[0] - 1;
    int W_u64 = xtr_u64.dimensions()[1];
    int S = (pcg_init.dimensions().size() == 1) ? 1 : (int)pcg_init.dimensions()[0];
    dim3 grid(B_tot, S, 1);
    sample_and_moments_kernel<<<grid, 512, 0, stream>>>(
        pcg_init.typed_data(), j_tab.typed_data(), step_idx.typed_data(), b_info.typed_data(),
        xtr_u64.typed_data(), A_pad->typed_data(), t_out->typed_data(),
        B_tot, n, (int)N_tr, W_u64);
    return ffi::Error::Success();
}

ffi::Error cf_mix_fwd_impl(hipStream_t stream,
                           ffi::Buffer<ffi::F64> th1,
                           ffi::Buffer<ffi::F64> Th,
                           ffi::Buffer<ffi::F64> pi,
                           ffi::Buffer<ffi::S32> A_pad,
                           ffi::Buffer<ffi::S32> tiles,
                           ffi::ResultBuffer<ffi::F64> z_out,
                           ffi::ResultBuffer<ffi::F64> p_out) {
    auto dims = th1.dimensions();
    int M = dims[0], L = dims[1], n = dims[2];
    auto a_dims = A_pad.dimensions();
    int S_seeds = (a_dims.size() == 3) ? (int)a_dims[0] : 1;
    int B_tot   = (a_dims.size() == 3) ? (int)a_dims[1] : (int)a_dims[0];
    int a_stride = (a_dims.size() == 3) ? (int)a_dims[2] : (int)a_dims[1];
    int M_per_seed = M / S_seeds;
    int T = tiles.dimensions()[0];
    int S_tot = p_out->dimensions()[2];

    (void)hipMemsetAsync(z_out->typed_data(), 0, (size_t)M * B_tot * sizeof(double), stream);
    dim3 grid(T, M, L);
    if (n <= 512) {
        mix_corr_fwd_kernel<512, true><<<grid, 512, 0, stream>>>(
            th1.typed_data(), Th.typed_data(), pi.typed_data(),
            A_pad.typed_data(), tiles.typed_data(),
            z_out->typed_data(), p_out->typed_data(), n, L, B_tot, S_tot, a_stride, M_per_seed);
    } else {
        mix_corr_fwd_kernel<1024, true><<<grid, 1024, 0, stream>>>(
            th1.typed_data(), Th.typed_data(), pi.typed_data(),
            A_pad.typed_data(), tiles.typed_data(),
            z_out->typed_data(), p_out->typed_data(), n, L, B_tot, S_tot, a_stride, M_per_seed);
    }
    return ffi::Error::Success();
}

ffi::Error cf_mix_eval_only_impl(hipStream_t stream,
                                 ffi::Buffer<ffi::F64> th1,
                                 ffi::Buffer<ffi::F64> Th,
                                 ffi::Buffer<ffi::F64> pi,
                                 ffi::Buffer<ffi::S32> A_pad,
                                 ffi::Buffer<ffi::S32> tiles,
                                 ffi::ResultBuffer<ffi::F64> z_out) {
    auto dims = th1.dimensions();
    int M = dims[0], L = dims[1], n = dims[2];
    auto a_dims = A_pad.dimensions();
    int S_seeds = (a_dims.size() == 3) ? (int)a_dims[0] : 1;
    int B_tot   = (a_dims.size() == 3) ? (int)a_dims[1] : (int)a_dims[0];
    int a_stride = (a_dims.size() == 3) ? (int)a_dims[2] : (int)a_dims[1];
    int M_per_seed = M / S_seeds;
    int T = tiles.dimensions()[0];

    (void)hipMemsetAsync(z_out->typed_data(), 0, (size_t)M * B_tot * sizeof(double), stream);
    dim3 grid(T, M, L);
    if (n <= 512) {
        mix_corr_fwd_kernel<512, false><<<grid, 512, 0, stream>>>(
            th1.typed_data(), Th.typed_data(), pi.typed_data(),
            A_pad.typed_data(), tiles.typed_data(),
            z_out->typed_data(), nullptr, n, L, B_tot, 0, a_stride, M_per_seed);
    } else {
        mix_corr_fwd_kernel<1024, false><<<grid, 1024, 0, stream>>>(
            th1.typed_data(), Th.typed_data(), pi.typed_data(),
            A_pad.typed_data(), tiles.typed_data(),
            z_out->typed_data(), nullptr, n, L, B_tot, 0, a_stride, M_per_seed);
    }
    return ffi::Error::Success();
}

ffi::Error cf_mix_bwd_impl(hipStream_t stream,
                           ffi::Buffer<ffi::F64> th1,
                           ffi::Buffer<ffi::F64> Th,
                           ffi::Buffer<ffi::F64> pi,
                           ffi::Buffer<ffi::S32> A_pad,
                           ffi::Buffer<ffi::S32> b_info,
                           ffi::Buffer<ffi::F64> p_out,
                           ffi::Buffer<ffi::F64> dz,
                           ffi::ResultBuffer<ffi::F64> d_th1,
                           ffi::ResultBuffer<ffi::F64> d_Th) {
    auto dims = th1.dimensions();
    int M = dims[0], L = dims[1], n = dims[2];
    auto a_dims = A_pad.dimensions();
    int S_seeds = (a_dims.size() == 3) ? (int)a_dims[0] : 1;
    int B_tot   = (a_dims.size() == 3) ? (int)a_dims[1] : (int)a_dims[0];
    int a_stride = (a_dims.size() == 3) ? (int)a_dims[2] : (int)a_dims[1];
    int M_per_seed = M / S_seeds;
    int S_tot = p_out.dimensions()[2];

    (void)hipMemsetAsync(d_th1->typed_data(), 0, (size_t)M * L * n * sizeof(double), stream);
    (void)hipMemsetAsync(d_Th->typed_data(), 0, (size_t)M * L * n * n * sizeof(double), stream);
    dim3 grid(B_tot, M, L);
    if (n <= 512) {
        mix_corr_bwd_kernel<512><<<grid, 512, 0, stream>>>(
            th1.typed_data(), Th.typed_data(), pi.typed_data(),
            A_pad.typed_data(), b_info.typed_data(), p_out.typed_data(), dz.typed_data(),
            d_th1->typed_data(), d_Th->typed_data(), n, L, B_tot, S_tot, a_stride, M_per_seed);
    } else {
        mix_corr_bwd_kernel<1024><<<grid, 1024, 0, stream>>>(
            th1.typed_data(), Th.typed_data(), pi.typed_data(),
            A_pad.typed_data(), b_info.typed_data(), p_out.typed_data(), dz.typed_data(),
            d_th1->typed_data(), d_Th->typed_data(), n, L, B_tot, S_tot, a_stride, M_per_seed);
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cf_sample_moments, cf_sample_moments_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Attr<int64_t>("N_tr")
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cf_mix_fwd, cf_mix_fwd_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cf_mix_eval_only, cf_mix_eval_only_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cf_mix_bwd, cf_mix_bwd_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());
"""

_LIB = None


def load_hip_ext():
    global _LIB
    if _LIB is not None:
        return _LIB
    h = hashlib.sha256(HIP_SRC.encode()).hexdigest()[:12]
    so_path = os.path.join(tempfile.gettempdir(), f"iqp_hip_cf_{h}.so")
    if not os.path.exists(so_path):
        pid = os.getpid()
        src_tmp = os.path.join(tempfile.gettempdir(), f"iqp_hip_cf_{h}_{pid}.cpp")
        so_tmp = os.path.join(tempfile.gettempdir(), f"iqp_hip_cf_{h}_{pid}.so")
        with open(src_tmp, "w") as f:
            f.write(HIP_SRC)
        subprocess.run([
            "/opt/rocm/bin/hipcc", "-O3", "-shared", "-fPIC", "-fopenmp",
            "--offload-arch=gfx942", "-Wno-defaulted-function-deleted",
            f"-I{jax.ffi.include_dir()}", src_tmp, "-o", so_tmp
        ], check=True)
        os.replace(so_tmp, so_path)
        try:
            os.remove(src_tmp)
        except OSError:
            pass
    lib = ctypes.CDLL(so_path)
    lib.pack_ops_moments_omp.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.pack_ops_moments_omp.restype = ctypes.c_int

    jax.ffi.register_ffi_target("cf_sample_moments", jax.ffi.pycapsule(lib.cf_sample_moments), platform="ROCM")
    jax.ffi.register_ffi_target("cf_mix_fwd", jax.ffi.pycapsule(lib.cf_mix_fwd), platform="ROCM")
    jax.ffi.register_ffi_target("cf_mix_eval_only", jax.ffi.pycapsule(lib.cf_mix_eval_only), platform="ROCM")
    jax.ffi.register_ffi_target("cf_mix_bwd", jax.ffi.pycapsule(lib.cf_mix_bwd), platform="ROCM")
    _LIB = lib
    return _LIB


def pack_u64_np(x_bits):
    n_rows, _ = x_bits.shape
    pad_bits = (-n_rows) % 64
    x_pad = np.pad(x_bits.T.astype(np.uint8), ((0, 1), (0, pad_bits)))
    return np.ascontiguousarray(np.packbits(x_pad, axis=1)).view(np.uint64), n_rows
