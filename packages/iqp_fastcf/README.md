# iqp_fastcf

Fast, memory-lean closed-form training of mixed IQP-QCBMs on AMD GPUs (HIP, float64).

It trains many models in lockstep, for example 3 initializations × 5 seeds, with the stratified closed-form MMD² loss
and Adam. The results are the same as the `iqp_hip_cf` engine (`MultiInitTrainer`) within floating-point round-off:
the PCG64 subset streams, loss, gradients and Adam are identical. There is no JAX on the training path. One HIP library
is compiled with `hipcc` on first use and cached in `$TMPDIR`.

## Measured on one AMD Instinct MI300X

MNIST (n = 784, 50,000 samples), all 306,936 qubit pairs as edges, 15 models per run, 1,000 steps:

| a | iqp_hip_cf | iqp_fastcf (default) | iqp_fastcf (`gpu_budget=LOW_MEMORY`) |
|---|---|---|---|
| 0 | 6.4 s, 1.07 GiB | 2.0 s, 0.28 GiB | 2.0 s, 0.17 GiB |
| 1 | 13.9 s, 2.59 GiB | 3.6 s, 0.53 GiB | 4.1 s, 0.13 GiB |
| 2 | 24.5 s, 2.59 GiB | 6.8 s, 0.76 GiB | 7.5 s, 0.16 GiB |
| 3 | 45.5 s, 4.59 GiB | 13.2 s, 1.17 GiB | 13.9 s, 0.16 GiB |
| 4 | 86.8 s, 8.59 GiB | 26.3 s, 1.99 GiB | 30.3 s, 0.18 GiB |

How these were measured:
- Times are wall time of the whole `train` call, including setup and compilation.
- Memory is the peak device memory of the process.
- Source: the arena's `judge.py all --steps 1000`, run 2026-10-18 22:50-23:02 UTC.

The low-memory mode keeps about 0.2 GiB on the GPU. Angles and Adam state live in pinned host memory and are
streamed over PCIe each step, and host threads do part of the Adam update.

## Install

Requirements:
- Linux with ROCm (`hipcc` under `$ROCM_PATH`, default `/opt/rocm`).
- An AMD GPU with FP64 atomics. The code was developed and verified on gfx942 (MI300X) only. Another target can be
  set with `IQP_FASTCF_ARCH=gfx90a`, but it is untested.
- Python ≥ 3.9 and numpy.
- jax and optax, for the reference fallback and the tests. Install them with the `reference` extra.

Install it with pip from this directory. In a shared environment, use `--target <dir>` and add that directory to
`PYTHONPATH`. You can also put this directory itself on `PYTHONPATH`.

## Use

```python
import numpy as np
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits, sigma_for, all_pairs, INITS, LOW_MEMORY

data = np.load("bits.npy").astype(np.int8)        # (N, n) of 0/1
test = np.load("bits_test.npy").astype(np.int8)
n, a, seeds = data.shape[1], 4, [7, 11, 23, 1234, 2025]
edges = all_pairs(n)
c0, pi = make_inits(data, a, seeds, edges)        # global / agnostic / cluster x seeds, seed-major
evaluate = make_evaluator(n, a, edges, test)      # test MMD^2 (needs jax)
log = {}
out = train(n, a, edges, data, [sigma_for(n, 2), sigma_for(n, 6)], cosine_lr(1e-3, 15000), 15000, c0, pi, seeds,
            eval_steps=[0, 1000, 5000, 15000], callback=lambda step, p: log.update({step: evaluate(p, pi)[0]}))
out["params"]   # (M, 2**a, G) compiled angles, G = n + #edges
out["losses"]   # (steps, M) loss of every model before each update
out["plan"]     # (offload level, models per chunk, branch parts) the fast engine ran; None = the vendored engine ran
```

- `train(n, a, edges, data, sigmas, lr, steps, c0, pi, seeds, k_train=2000, rmax=10, gpu_budget=None, eval_steps=(), callback=None)`
  - `lr`: a number (constant), a sequence with one value per step, or a function of the 0-based step.
    `cosine_lr(lr0, T, alpha=0)` is `optax.cosine_decay_schedule(lr0, T, alpha)`, the IQP paper's schedule.
    A non-constant lr costs nothing measurable: the engine runs one step per call (D-Wave a = 3: 2.35 s vs 2.34 s
    for 300 steps).
  - `callback(step, params)` gets a copy of the (M, L, G) parameters after `step` updates, for every step in
    `eval_steps` (0 to `steps`).
  - Model `m` draws its subsets from the stream of `seeds[m // (M // len(seeds))]`.
  - `gpu_budget=None` keeps all state on the GPU, which is fastest. A number in bytes caps device memory;
    `LOW_MEMORY` is about 0.2 GiB. A budget that cannot be met gives a `RuntimeWarning` and the smallest plan.
    Runs with fewer than 3 models keep the angles on the device and only the Adam moments on the host.
  - Inputs are checked before anything reaches the GPU, and a mistake raises `ValueError`:
    - data must be (N, n) of 0/1;
    - edges must be distinct pairs of distinct qubits;
    - c0 must be (M, L, G) and pi must be (M, L);
    - M must be a multiple of the number of seeds;
    - the values must be finite.
  - Configurations outside the fast kernels' templates (a > 4 or rmax > 10) run on the vendored `iqp_hip_cf`
    engine, which needs jax. So does a failed fast-engine setup. Both give a `RuntimeWarning`.
- `make_evaluator(n, a, edges, x_test, keval=10000, rmax=10)` returns `evaluate(params, pi) -> (mean, per_mbar)`.
  It is the IQP paper's test MMD²:
  - for each m̄ = 1..6, 10,000 subsets A ~ Bernoulli(m̄/n)ⁿ (rng seed 1000 + m̄), keeping |A| ≤ 10;
  - the exact z_A against the test-set moments;
  - the mean is taken over m̄.
- `make_inits(data, a, seeds, edges=None, labels=None)` uses the recipe of the IQP paper's `cf_multi.py`:
  - global: branches start at the data's one-body moments;
  - agnostic: branches start at π/4;
  - cluster: one branch per cluster, with π set to the cluster sizes. Pass `labels` (one per data row, in [0, 2**a))
    to use your own clustering, such as the paper's spectral clustering. The default is k-means on a 10k subsample.
  - all three add a seeded `Unif(-r0, r0)` with `r0 = (π/2)/sqrt(L·G)`.
- `examples/campaign.py data.npy 4 15000 out.npz [--low-memory]` is the same campaign from the command line, with a
  constant lr.

## Check

`python tests/test_engine.py` (GPU and jax required; about 15 s) checks seven things:
- The fast engine matches the reference `iqp_hip_cf` trainer for a = 0, 2 and 4 on a small problem: losses to
  1e-8 relative, parameters to 1e-8 absolute.
- A cosine lr and the callback snapshots match the reference to the same tolerance. An lr given as a sequence gives
  the same result as the same lr given as a function.
- The low-memory mode matches the all-on-device mode to 1e-10.
- A 2-model run with a budget uses the host-offload level and matches.
- Bad inputs raise `ValueError`.
- The evaluator matches an exact numpy computation for product branches.
- `labels` set the cluster init.

In the arena, the engine also passed these checks against the frozen baseline:
- the statevector checks;
- the n = 784, a = 4 loss and gradient check, relative 1e-10;
- 300-step training on MNIST and D-Wave, relative 1e-8.

On the IQP paper's 15,000-step cosine campaigns (5 seeds × global / agnostic / cluster), against the stored
`iqp_hip_cf` results (2026-10-19):
- D-Wave (n = 484, a = 3, all pairs) and MNIST (n = 784, a = 4, window-6 edges), in both memory modes;
- 300-step losses agree to 7e-14 (D-Wave) and 1e-6 (MNIST);
- no NaN;
- final test MMD² equal to 0.04% per model, except MNIST agnostic, which is up to 1.1%;
- 2.7× (D-Wave) and 3.2× (MNIST) faster.

## Known limits

- Training is chaotic. Round-off-level differences grow over thousands of steps, so long runs agree with
  `iqp_hip_cf` statistically, not to 1e-6 at every step.
- Runs are not bit-reproducible, here or in `iqp_hip_cf`, because gradients are summed with FP64 atomics. Two runs
  of the same 300 steps differ in the parameters by up to:
  - 1e-12 on D-Wave and 2e-10 on MNIST here;
  - 1e-13 and 1e-11 in `iqp_hip_cf`.
- Only MI300X (gfx942) has been tested.
