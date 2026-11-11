"""Cluster labels for the cluster-initialized runs: the paper's spectral clustering (main.tex App. D C), rerun for
every training seed.
- n = 16: the whole training set. MNIST, D-Wave: a subsample of 10,000 training strings, drawn with
  default_rng(seed) (D-Wave: all 10,000). The remaining strings go to the nearest group centroid.
- Affinity exp(-d_H / d_med); leading 20 eigenvectors of D^-1/2 A D^-1/2; row-normalized embedding on the first L;
  k-means++ (20 restarts, 150 iterations, seed + L); groups relabelled by mean bit density.
- cluster_eq (equal-weight control, EQ_CONFIGS): capacity-balanced assignment to the same k-means centers.
  On MNIST the strings outside the subsample still go to the nearest centroid, so the control's weights are set to
  1/L in train.py.
The helpers are copied verbatim from legacy/experiments/cluster_count_mmd_sweep/scripts/prepare_cluster_sweep.py,
and group() follows legacy/experiments/paper_ours/prepare_routed.py:group_by_cluster.
usage: clusters.py [ds ...]  ->  clusters/<ds>/a<a>_seed<s>.npz (train_y, eq_y for EQ_CONFIGS, d_med, sizes)
"""
import json, sys, time
import numpy as np
from scipy.sparse.linalg import eigsh
from common import CLUSTERS, DATASETS, SEEDS_BIG, data

EQ_CONFIGS = {("blobs", 1), ("blobs", 2), ("ising", 2), ("ising", 3), ("mnist", 4), ("dwave", 3)}
EQ_SEEDS = {"blobs": (7, 1234, 2025), "ising": (7, 1234, 2025), "mnist": tuple(SEEDS_BIG), "dwave": tuple(SEEDS_BIG)}


# ---- verbatim from prepare_cluster_sweep.py ----------------------------------------------------------------------
def hamming_counts(a, b=None):
    a32 = np.asarray(a, dtype=np.int32)
    b32 = a32 if b is None else np.asarray(b, dtype=np.int32)
    a_sum = a32.sum(axis=1, dtype=np.int32)
    b_sum = b32.sum(axis=1, dtype=np.int32)
    dot = a32 @ b32.T
    return a_sum[:, None] + b_sum[None, :] - 2 * dot


def rbf_hamming_affinity(train, sigma):
    dist = hamming_counts(train).astype(np.float64)
    nonzero = dist[dist > 0]
    median_nonzero = float(np.median(nonzero)) if nonzero.size else 1.0
    sigma_value = float(np.sqrt(median_nonzero / 2.0)) if sigma is None else float(sigma)
    affinity = np.exp(-dist / (2.0 * sigma_value * sigma_value), dtype=np.float64)
    np.fill_diagonal(affinity, 1.0)
    return affinity, {"sigma": sigma_value, "median_nonzero_hamming": median_nonzero}


def spectral_basis(affinity, max_k, n_eigs, seed):
    degree = affinity.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = affinity * inv_sqrt[:, None] * inv_sqrt[None, :]
    k = min(max(int(max_k), int(n_eigs)), normalized.shape[0] - 2)
    vals, vecs = eigsh(normalized, k=k, which="LA", tol=1e-4, maxiter=3000)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def embedding_for_k(vecs, k):
    embed = np.asarray(vecs[:, :k], dtype=np.float64)
    row_norm = np.linalg.norm(embed, axis=1, keepdims=True)
    return embed / np.maximum(row_norm, 1e-12)


def kmeans_plus_plus_init(data, n_clusters, rng):
    n_samples = data.shape[0]
    centers = np.empty((n_clusters, data.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n_samples))
    centers[0] = data[first]
    closest = np.sum((data - centers[0]) ** 2, axis=1)
    for idx in range(1, n_clusters):
        total = float(closest.sum())
        if total <= 0:
            centers[idx] = data[int(rng.integers(0, n_samples))]
            continue
        choice = int(rng.choice(n_samples, p=closest / total))
        centers[idx] = data[choice]
        dist = np.sum((data - centers[idx]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    return centers


def run_kmeans(data, n_clusters, seed, n_init=20, max_iter=150):
    best_labels, best_centers, best_inertia = None, None, float("inf")
    base_rng = np.random.default_rng(seed)
    for _ in range(n_init):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        centers = kmeans_plus_plus_init(data, n_clusters, rng)
        labels = np.zeros(data.shape[0], dtype=np.int64)
        for iteration in range(max_iter):
            dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int64)
            if np.array_equal(new_labels, labels) and iteration > 0:
                break
            labels = new_labels
            for cluster in range(n_clusters):
                selected = data[labels == cluster]
                if len(selected):
                    centers[cluster] = selected.mean(axis=0)
                else:
                    centers[cluster] = data[int(rng.integers(0, data.shape[0]))]
        dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(dists, axis=1)))
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels.copy(), centers.copy()
    return best_labels, best_centers, best_inertia


def balanced_assign_flexible(data, centers, seed):
    n_samples, n_clusters = data.shape[0], centers.shape[0]
    base = n_samples // n_clusters
    remainder = n_samples % n_clusters
    capacities = np.asarray([base + (1 if c < remainder else 0) for c in range(n_clusters)], dtype=np.int64)
    dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    order = np.argsort(dists, axis=1)
    if n_clusters > 1:
        margin = dists[np.arange(n_samples), order[:, 1]] - dists[np.arange(n_samples), order[:, 0]]
    else:
        margin = np.zeros(n_samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_order = np.argsort(-(margin + rng.normal(0.0, 1e-12, size=n_samples)))
    labels = np.full(n_samples, -1, dtype=np.int64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for idx in sample_order:
        for cluster in order[idx]:
            if counts[cluster] < capacities[cluster]:
                labels[idx] = int(cluster)
                counts[cluster] += 1
                break
        if labels[idx] < 0:
            raise RuntimeError("balanced assignment failed")
    return labels


def relabel_by_mean_ones(labels, train):
    means = [(int(label), float(train[labels == label].mean())) for label in sorted(np.unique(labels))]
    mapping = {old: new for new, (old, _) in enumerate(sorted(means, key=lambda item: item[1]))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def assign_test_by_centroid(train, train_labels, test):
    labels = sorted(int(label) for label in np.unique(train_labels))
    centroids = np.stack([train[train_labels == label].mean(axis=0) for label in labels], axis=0)
    test64 = np.asarray(test, dtype=np.float64)
    out = []
    batch = 4096
    for start in range(0, len(test64), batch):
        chunk = test64[start:start + batch]
        dists = np.sum((chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        out.extend(labels[index] for index in np.argmin(dists, axis=1))
    return np.asarray(out, dtype=np.int64)
# ------------------------------------------------------------------------------------------------------------------


def group(vecs, sub, L, seed, balanced):
    """prepare_routed.group_by_cluster on a precomputed basis: labels of the (sub)sample."""
    embed = embedding_for_k(vecs, L)
    labels, centers, _ = run_kmeans(embed, L, seed + L, n_init=20, max_iter=150)
    if balanced:
        labels = balanced_assign_flexible(embed, centers, seed + L)
    return relabel_by_mean_ones(labels.astype(np.int64), sub)


def main(names):
    for ds in names:
        x, _ = data(ds)
        _, _, ladder, seeds, nsub, _ = DATASETS[ds]
        out = CLUSTERS / ds
        out.mkdir(parents=True, exist_ok=True)
        basis = None
        for seed in seeds:
            if nsub is not None and nsub < len(x):
                idx = np.sort(np.random.default_rng(seed).permutation(len(x))[:nsub])
            else:
                idx = np.arange(len(x))
            if basis is None or nsub is not None and nsub < len(x):      # the affinity depends on the subsample only
                t0 = time.time()
                aff, meta = rbf_hamming_affinity(x[idx], None)
                vals, vecs = spectral_basis(aff, max_k=16, n_eigs=20, seed=seed)
                del aff
                basis = (vecs, meta)
                print(f"[{ds} seed {seed}] spectral basis on {len(idx)} strings: d_med {meta['median_nonzero_hamming']:.1f}, "
                      f"sigma {meta['sigma']:.3f}, top eigenvalues {np.round(vals[:4], 4).tolist()} ({time.time() - t0:.0f} s)", flush=True)
            vecs, meta = basis
            rest = np.setdiff1d(np.arange(len(x)), idx)
            for a in ladder:
                if a == 0:
                    continue
                L = 2 ** a
                res = {}
                for kind, bal in (("train_y", False), ("eq_y", True)):
                    if bal and not ((ds, a) in EQ_CONFIGS and seed in EQ_SEEDS[ds]):
                        continue
                    y = np.empty(len(x), dtype=np.int64)
                    y[idx] = group(vecs, x[idx], L, seed, bal)
                    if len(rest):
                        y[rest] = assign_test_by_centroid(x[idx], y[idx], x[rest])
                    res[kind] = y.astype(np.int8)
                sizes = {k: np.bincount(v, minlength=L).tolist() for k, v in res.items()}
                np.savez_compressed(out / f"a{a}_seed{seed}.npz", **res, d_med=meta["median_nonzero_hamming"],
                                    sigma=meta["sigma"], n_sub=len(idx))
                print(f"[{ds} a={a} seed {seed}] masses {np.round(np.array(sizes['train_y']) / len(x), 3).tolist()}"
                      + (f"  eq {sizes['eq_y']}" if "eq_y" in sizes else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
