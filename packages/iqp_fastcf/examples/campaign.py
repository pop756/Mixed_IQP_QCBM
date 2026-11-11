"""Train 3 inits x len(seeds) mixed IQP models on a 0/1 data set in one run and save the result.

    python examples/campaign.py data.npy 4 15000 out.npz [--low-memory]

data.npy: (N, n) array of 0/1. All qubit pairs are edges; bandwidths m-bar = 2 and 6; lr 1e-3; seeds 7, 11, 23, 1234, 2025.
"""
import sys, time
import numpy as np
from iqp_fastcf import train, make_inits, sigma_for, all_pairs, INITS, LOW_MEMORY

path, a, steps, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
data = np.load(path).astype(np.int8)
n, seeds = data.shape[1], [7, 11, 23, 1234, 2025]
c0, pi = make_inits(data, a, seeds)
t0 = time.perf_counter()
r = train(n, a, all_pairs(n), data, [sigma_for(n, 2), sigma_for(n, 6)], 1e-3, steps, c0, pi, seeds,
          gpu_budget=LOW_MEMORY if "--low-memory" in sys.argv else None)
print(f"n={n} a={a} {len(c0)} models x {steps} steps: {time.perf_counter() - t0:.1f} s; final loss per model:")
for i, l in enumerate(r["losses"][-1]):
    print(f"  seed {seeds[i // len(INITS)]:5d} {INITS[i % len(INITS)]:8s} {l:.6e}")
np.savez(out, params=r["params"], losses=r["losses"], pi=pi)
