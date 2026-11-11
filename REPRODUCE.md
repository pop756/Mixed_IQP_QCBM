# Reproducing the figures and tables

Every figure and table of the article is listed below by its LaTeX label, with the script that makes it and the
files it reads. Labels that start with `sm:` belong to the Supplemental Material. All paths are relative to the
repository root.

**Status.** Each row ends with one of these.

- **runs**: runs from this repository alone, on CPU unless stated. This was checked in a fresh copy for this
  release.
- **needs GPU**: runs from this repository on the AMD GPU that `packages/iqp_fastcf` needs.
- **needs parameters**: reads trained angles (`params/*.npz`), which are not in the repository. Train first
  (section "Training").
- **regenerate first**: reads result files that are not in the repository. The command that makes them is given.
- **no data**: a schematic, or a table of definitions or settings. Nothing is computed.

The rows marked **needs GPU**, **needs parameters** and **regenerate first** were also run for this release, in a
fresh copy after retraining. The section "Result of the check for this release" gives the commands, the times
and how closely each result came back.

## Environment

`environment.yml` (conda env `iqpopt`). The article's runs used Python 3.12, numpy 2.4, jax 0.6.0, matplotlib
3.10 and one AMD Instinct MI300X (ROCm). The figure scripts and the reruns of the section "The earlier
pipeline" run on CPU.

The scripts put `packages/iqp_fastcf` on `sys.path` themselves. The library of the earlier pipeline imports the
modified IQPopt copy in `legacy/experiments/randomized_iqp_mnist8x8/vendor`.

## Figures

The figure scripts are in `manuscript/figures/` and write the PDF next to themselves. They share
`manuscript/figures/_seed_agg.py` (reads the campaign results), `manuscript/figures/_paths.py` and
`manuscript/figures/_paperstyle.py`. The column "reads" lists every data file a script opens. A folder stands
for the files in it.

| label | script | reads | status |
|---|---|---|---|
| `fig:overview` | `manuscript/figures/make_fig_1.py` | nothing | no data. The script runs |
| `fig:landscape` | `manuscript/figures/make_fig_landscape.py` | `experiments/variance_scaling/results/` for panels (a) to (c), and the four files `experiments/gd_escape/results/escape_<ds>_<scheme>.json` for panel (d) | regenerate first: the four escape files (below) |
| `fig:theory-check` | `manuscript/figures/make_fig_theory_check.py` | `legacy/experiments/paper_ours/analysis_ancilla_angles/curvature_L_scaling.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_blobs_a1_exact.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_ising_a1_exact.csv`, `legacy/experiments/paper_ours/referee_2026_10/theory_numerics/item1_theorem2_remainder.csv`, `legacy/experiments/paper_ours/referee_2026_10/theory_numerics/item2_common_mode.csv` | runs |
| `fig:train-results`, `fig:train-curves` | `manuscript/figures/make_fig_train_results.py` | `experiments/cosine_rerun/runs/`, `experiments/cosine_rerun/results/refs.csv` | runs |
| `fig:diversity` | `manuscript/figures/make_fig_diversity_scatter.py` | `experiments/cosine_rerun/results/branch_diversity.csv`, `experiments/cosine_rerun/results/branch_necessity.csv`, `experiments/cosine_rerun/results/diversity_refs.csv` | runs |
| `fig:modesep` | `manuscript/figures/make_fig_modesep.py` | `experiments/cosine_rerun/results/branch_diversity.csv`, `experiments/cosine_rerun/results/branch_necessity.csv`, `experiments/cosine_rerun/results/modesep_blobs_a3.npz`, `experiments/cosine_rerun/results/modesep_ising_a2.npz`, `experiments/cosine_rerun/results/mnist_branch_marginals.npz` | runs |
| `fig:app-modesep-full` | the same script with `MODESEP_FULL=1` | `experiments/cosine_rerun/results/branch_diversity.csv`, `experiments/cosine_rerun/results/branch_necessity.csv`, `experiments/cosine_rerun/results/modesep_blobs_a3.npz`, `experiments/cosine_rerun/results/modesep_ext_cache.npz`, `experiments/cosine_rerun/results/mnist_branch_marginals.npz` | runs |
| `sm:fig:app-assumption` | `manuscript/figures/make_fig_app_assumption.py` | `experiments/cosine_rerun/results/factorization_residuals.csv` | runs |
| `sm:fig:app-branch-spread` | `manuscript/figures/make_fig_app_branch_spread.py` | `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_blobs_a1_exact.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_blobs_a2_exact.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_blobs_a3_exact.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_ising_a1_exact.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_ising_a2_exact.csv`, `legacy/experiments/paper_ours/analysis_ancilla_angles/noise_grad_ising_a3_exact.csv` | runs |
| `sm:fig:app-sigma-sweeps` | `manuscript/figures/make_fig_app_sigma_sweeps.py` | `experiments/cosine_rerun/runs/`, `experiments/cosine_rerun/results/refs.csv`, `legacy/experiments/paper_ours/baselines/blobs/mmd_sigma_sweep_baselines.csv`, `legacy/experiments/paper_ours/baselines/ising/mmd_sigma_sweep_baselines.csv` | runs |

The inputs under `legacy/` are regenerated by the scripts of the section "The earlier pipeline".

**Where the inputs of `fig:landscape` come from.**

- Panels (a) to (c): `experiments/variance_scaling/scaling.py` and `experiments/variance_scaling/ancilla.py`
  (usage in their docstrings, needs GPU) write the four files of `experiments/variance_scaling/results/`.
- Panel (d): four files that are too large for the repository. Regenerate them with

  ```bash
  cd experiments/gd_escape
  for ds in blobs ising; do for sc in global agnostic; do python escape.py $ds $sc; done; done
  ```

  Each call took 2 to 4 minutes on the MI300X in the check for this release, and the four runs reproduced the
  article's escape times. `escape.py` takes its loss floor and the curvature at that point from the
  single-circuit reference points `experiments/gd_escape/thstar_blobs.npy` and
  `experiments/gd_escape/thstar_ising.npy`, which are included.

**Keep the two `thstar` files.** They are the single-circuit reference points of the escape experiment, and the
steps at which the article's separated mixtures pass below the single-circuit level are counted against them.

- `experiments/gd_escape/thstar_check.py` (CPU, under a minute) confirms that each array is a local minimum of
  the exact single-circuit training loss, with an MMD² of 1.833e-2 on the blobs and 5.14e-4 on Ising. The
  docstrings of `escape.py` and `diag.py` still call these points the exact single-circuit optimum. They were
  written before this check.
- `experiments/gd_escape/diag.py` computes such points (Adam from the global center) and overwrites the two
  files. A fresh run does not return the included arrays. In the check for this release it ended at 1.821e-2
  (CPU) and 1.829e-2 (GPU) on the blobs and at 5.65e-4 (CPU and GPU) on Ising.
- The included arrays are local minima, not proven global ones. `python thstar_check.py --random 12` polishes
  twelve random starts with L-BFGS, and `experiments/gd_escape/thstar_check.txt` is its output. On the blobs one
  start reaches 1.774e-2, below the included array. On Ising none goes below it (the lowest is 5.28e-4).
- The separated mixtures of the escape runs end between 1.07e-2 and 1.25e-2 on the blobs, below every
  single-circuit value above. Counted against 1.774e-2 instead of the included array, the steps at which they
  pass the single-circuit level grow by 10 to 60.

## Tables

| label | source | made by | status |
|---|---|---|---|
| `tab:init`, `tab:notation` | definitions in the text | nothing | no data |
| `tab:datasets` | `experiments/datasets/`, `experiments/data/` | section "Datasets" | no data. The dataset files are included |
| `tab:hyperparams` | the settings in `experiments/cosine_rerun/common.py` and `experiments/cosine_rerun/train.py` | nothing | no data |
| `tab:partition`, `tab:app-partition`, `tab:app-partition-seeds` | `experiments/modesep_bound/results/summary.json`, `experiments/modesep_bound/runs/`, `experiments/learned_partition/results/summary.json`, `experiments/branch_z_start/results/summary.json`, collected in `experiments/modesep_manuscript/results/numbers.json` | `experiments/modesep_bound/modesep_bound.py`, `experiments/learned_partition/learned_partition.py`, `experiments/branch_z_start/branch_z_start.py`, `experiments/modesep_manuscript/numbers.py` | runs for the `build` mode of the first three scripts, which rebuilds each summary from the committed runs. Their training modes need the GPU and the parameters of the campaign, and `numbers.py` needs parameters |
| `tab:eqweight-control` | `experiments/cosine_rerun/results/eqweight_control.csv` | `experiments/cosine_rerun/train.py` (variant `eq`), `experiments/cosine_rerun/analyze.py` | needs GPU. `analyze.py` rebuilds this table from the committed runs |
| `tab:product-baseline` | `experiments/cosine_rerun/results/product_baselines_mnist.csv`, `experiments/cosine_rerun/results/product_baselines_dwave.csv`, and for 16 qubits the rows `center` and `em` of blobs and ising in `legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/summary.csv` | `experiments/cosine_rerun/products.py`, and for 16 qubits `experiments/cosine_rerun/products16.py` | runs for `products16.py`, which recomputes the 16-qubit values exactly (about a minute). `products.py` needs the GPU (25 minutes) |
| `tab:bandwidth-sweep` | `experiments/bandwidth_sweep/results/sweep_table.md` | `experiments/bandwidth_sweep/sweep.py` | needs parameters |
| `tab:mode-index` | rows Λ and N_mode: `experiments/cosine_rerun/results/branch_necessity.csv` and `experiments/cosine_rerun/results/mode_index.csv`, summarized in `experiments/necessity_audit/results/numbers.json`. Correlations with their intervals: `experiments/cosine_rerun/results/diversity_trend_stats.txt` (see "Intervals of `tab:mode-index`" below) | `experiments/cosine_rerun/analyze.py`, `experiments/necessity_audit/numbers.py`, `manuscript/figures/make_fig_diversity_scatter.py` | needs parameters for these two tables of `analyze.py`. `numbers.py check` and the figure script run |
| `tab:likelihood` | `experiments/exact_metrics/results/exact_metrics.md` | `experiments/exact_metrics/metrics.py`, `experiments/exact_metrics/targets.py` | needs parameters. Do not run `metrics.py` without them (warning below) |
| `tab:weights` | `experiments/trainable_weights/results/compare.md`, `experiments/trainable_weights/runs/` | `experiments/trainable_weights/train_tw.py`, `experiments/trainable_weights/compare.py` | runs for `compare.py`, which rebuilds the table from the committed runs. `train_tw.py` needs the GPU |
| `sm:tab:rbm-settings` | `legacy/experiments/paper_ours/baselines/blobs/selected_rbm.json`, `legacy/experiments/paper_ours/baselines/ising/selected_rbm.json`, and the grids in `legacy/experiments/paper_ours/baselines/blobs/results_baselines.csv` and `legacy/experiments/paper_ours/baselines/ising/results_baselines.csv` | `legacy/experiments/paper_ours/baselines/run_baselines.py` | runs (section "The earlier pipeline") |
| `sm:tab:theory-check` | `legacy/experiments/paper_ours/referee_2026_10/theory_numerics/item0_table_reproduction.csv`, `legacy/experiments/paper_ours/referee_2026_10/theory_numerics/item1_theorem2_remainder.csv` | `legacy/experiments/paper_ours/referee_2026_10/theory_numerics/theory_numerics.py` | runs (section "The earlier pipeline") |

**Numbers quoted in the text.**

- `experiments/cosine_rerun/results/paper_numbers.txt` lists the numbers the text quotes from the campaign
  tables, each with its source file. `experiments/cosine_rerun/paper_numbers.py` rebuilds it from the committed
  results (runs).
- Three further scripts of `experiments/cosine_rerun/` make tables that the appendices quote and that are not
  in that list: `certificate.py` (`results/certificate.csv`, `results/clustering_settings.csv`, CPU, about a
  minute), `factorization.py` (`results/factorization_residuals.csv`, the input of `sm:fig:app-assumption`, CPU,
  about 45 minutes on one core) and `mechanism.py` (the `results/mechanism_*.csv` tables, needs parameters and the
  GPU).
- The other files of `legacy/experiments/paper_ours/referee_2026_10/theory_numerics/` hold the exact values
  behind the coincidence analysis (section "The earlier pipeline").
- The statement in the Discussion about autoregressive models rests on `experiments/classical_baseline/results/`
  (`experiments/classical_baseline/fvsbn.py`).

**Intervals of `tab:mode-index`.** `experiments/cosine_rerun/results/diversity_trend_stats.txt` holds the
correlations ρ_D and ρ_N of the table with their bootstrap intervals, as printed. It is the output of an earlier
version of `manuscript/figures/make_fig_diversity_scatter.py`, which printed these lines for N_mode where the
current version prints them for Λ. The current script prints ρ_D and ρ_Λ with their intervals (lines
`[trend D2]` and `[trend Lambda]`), and on the committed results it gives the printed values. When the third
column of its points is N_mode from `experiments/cosine_rerun/results/mode_index.csv` instead of Λ, the same
bootstrap writes the eight lines of the record byte for byte. `experiments/bandwidth_sweep/results/sweep_table.md`
and `experiments/cosine_rerun/results/paper_numbers.txt` also give ρ_N, from other bootstrap draws, with the
printed values and with bounds that differ from the printed ones by up to 0.03.

**Warning for `tab:likelihood`.** `experiments/exact_metrics/metrics.py` skips every model whose parameter file is
missing and then overwrites `results/exact_metrics.json` and `results/exact_metrics.md`. Without the parameters
it replaces the committed tables with empty ones.

## Datasets

- `experiments/datasets/blobs/8_blobs_dataset/` and `experiments/datasets/ising/2d_random_lattice_dataset/` are
  copied from the scaling-gqml repository (see `NOTICE`). Their `generator.py` needs the package qml-benchmarks,
  which `environment.yml` only mentions in a comment. The generators were not rerun for this release.
- `experiments/data/blobs/` and `experiments/data/ising/` are the 16-qubit splits the campaign reads. They hold
  the same strings as the raw files in the order `rng.choice(len(x), len(x), replace=False)` with
  `rng = numpy.random.default_rng(1234)`, training set first and test set second. This was checked for this
  release.
- `experiments/datasets/MNIST_28x28_all/` holds 50,000 training and 10,000 test images binarized to 784 bits,
  with their labels. `metadata.json` states the split and the threshold. The script that made the files is not
  included.
- `experiments/datasets/dwave/` holds 10,000 training and 60,000 test configurations of 484 spins.
  `download_data.py` downloads the source from Zenodo and writes the two files. It was not rerun for this
  release.

## Training

`experiments/cosine_rerun/` is the campaign. `train.py <ds> <a> <variant>` trains one group of models and
writes `runs/<ds>/a<a>_<variant>.json` (committed) and `params/<ds>_a<a>_<variant>.npz` (not committed). The
variants are `main` (global, data-agnostic and cluster initialization), `eq` (equal-weight control) and `fc1`
(one-body mixture).

`run_all.sh` runs every group and then `analyze.py`. **It skips a group whose summary in `runs/` exists, and
this repository ships all of them, so as it stands it trains nothing.** To retrain, move the summaries away
first:

```bash
cd experiments/cosine_rerun
mv runs runs_committed          # run_all.sh skips every group that has a summary
PY=python ./run_all.sh          # trains all groups, writes runs/ and params/, then runs analyze.py
```

To retrain one group, call `python train.py blobs 3 main` directly. It trains whether or not a summary exists
and overwrites it. The whole campaign takes under an hour on one MI300X (section "Result of the check for this
release").

- The cluster labels of every training seed are in `experiments/cosine_rerun/clusters/`. `clusters.py` rebuilds
  them, and `run_all.sh` calls it only when they are missing.
- Settings: all qubit pairs as two-body gates on every dataset, Adam with learning rate 1e-3 and cosine decay
  to zero over 15,000 steps, the stratified closed-form MMD² loss at m̄ = 2 and 6.
- Training seeds: 7, 11, 23, 42, 99, 314, 777, 1234, 2025, 2718 for 16 qubits, and 7, 11, 23, 1234, 2025 for
  MNIST and D-Wave.
- `analyze.py` turns the runs into the tables of `experiments/cosine_rerun/results/`. Without parameters it
  rebuilds the campaign and control tables and leaves the tables that need parameters as they are.
- `experiments/trainable_weights/run_all.sh` has the same skip rule for `runs/<ds>/a<a>_tw.json`.

**Scripts that need the parameters** (all read `experiments/cosine_rerun/params/`, some also the `params/` of
their own folder): the diversity, necessity, mode-index and zero-two-body tables of `analyze.py`, `caches.py`,
`mechanism.py`, `experiments/bandwidth_sweep/sweep.py`, `experiments/exact_metrics/metrics.py`,
`experiments/modesep_manuscript/numbers.py`, and the training modes of `experiments/modesep_bound/`,
`experiments/learned_partition/` and `experiments/branch_z_start/`. Each script states its usage and outputs
in its docstring.

## The earlier pipeline

The campaign above replaced an earlier two-stage pipeline built on IQPopt. Two kinds of results of the article
still come from it: the numerical tests of the theorems and the 16-qubit RBM baseline. The 16-qubit product
baselines were also first computed there. All three run again from this repository on CPU.

`legacy/experiments/randomized_iqp_mnist8x8/` is the mixed-IQP library of that pipeline with its tests and the
modified IQPopt copy.

**What was restored.** The analysis scripts reach the earlier campaign through one function, `build_run` of
`ancilla_angle_grad_analysis.py`. It reads a configuration file, the training split and, for the cluster
initialization, a start artifact.

- Configuration files: `legacy/experiments/paper_ours/blobs_seed/configs/` and
  `legacy/experiments/paper_ours/ising_seed/configs/` hold the 23 files per dataset that the scripts read (seed 7
  for all three initializations, seeds 1234 and 2025 for the global and data-agnostic ones). Their paths are
  relative to the dataset folder, and the training split is `experiments/data/`.
- Cluster labels of seed 7: `dataset_s2_cluster_a<a>_usqm_seed7/y_train.csv` in the two dataset folders.
- Cluster-start artifacts: not stored. `legacy/experiments/paper_ours/rebuild_cluster_init.py` writes them from
  the split and the labels with the calls of the earlier init stage. No random number is drawn.
- `ancilla_angle_grad_analysis.py` is used as a module. Its own command-line phases read trained checkpoints
  of the earlier campaign, which are not included.

**Numerical tests of the theorems.**

```bash
cd legacy/experiments/paper_ours
python rebuild_cluster_init.py                  # cluster-start artifacts
cd analysis_ancilla_angles
python check_theorem2_bound.py                  # theorem2_bound_check.csv
python select_witness.py                        # theorem2_witness.csv
python curvature_decomposition.py               # curvature_decomposition.csv
python curvature_L_scaling.py                   # curvature_L_scaling.csv, uniform_noise_absolute.csv
for ds in blobs ising; do for a in 1 2 3; do
  python noise_grad_blobs.py --mode exact --ds $ds --anc $a --scales 0,0.003,0.01,0.03,0.1,0.3,1,3
done; done                                      # noise_grad_<ds>_a<a>_exact.csv
cd ../referee_2026_10/theory_numerics
python theory_numerics.py items                 # item0, item1, item2, census
python theory_numerics.py hessian               # item3_mismatch_block, item3_M_eigenvalues, item3_block_check
python theory_numerics.py gd                    # item3_gd_ancilla_growth, item3_gd_ancilla_norm_every10
```

`exact_iqp.py` and `exact_iqp_weighted.py` are the exact-loss modules these scripts import.

**RBM baseline.**

```bash
cd legacy/experiments/paper_ours/baselines
for ds in blobs ising; do
  PO_DS=$ds BL_OUTDIR=rerun_$ds python run_baselines.py tune_rbm     # grid on a validation split, selected_rbm.json
  PO_DS=$ds BL_OUTDIR=rerun_$ds python run_baselines.py final_rbm    # selected settings on the full split, test MMD²
done
```

- `run_baselines.py` trains with `rbm.py` and scores with `eval_baselines.py`, in up to 16 processes
  (`BL_WORKERS`).
- `BL_OUTDIR` keeps the rerun apart. Without it the script appends its rows to the committed tables of
  `baselines/<ds>/`.
- `common.py` and `evaluate_sweep.py` in this folder stand in for two shared modules of the earlier pipeline.
  They define only the names the two scripts use, for these two datasets.
- Use the two stages. The mode `all` also trains an energy-based model whose module is not included. The
  committed tables still hold the rows of that model, which the article does not use.

**16-qubit product baselines.**

```bash
cd experiments/cosine_rerun
python products16.py            # results/product_baselines_blobs.csv, results/product_baselines_ising.csv
python products16.py check      # compares with the values quoted from the earlier pipeline
```

The EM values agree with the quoted ones to round-off and the cluster-center values to 2e-6. The earlier
pipeline read the center angles from a start artifact, and `products16.py` uses the cluster means directly.

`legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/results_blobs.csv` and
`legacy/experiments/paper_ours/referee_2026_10/bernoulli_baseline/results_ising.csv` hold the per-seed values of
the earlier pipeline behind the rows `center` and `em` of `summary.csv`. The script that wrote them,
`bernoulli_baseline.py`, is not included. `products.py` carries a copy of its EM routine, and `products16.py`
uses that copy. The other rows of `summary.csv` belong to the earlier campaign, and the article does not use
them.

## Result of the check for this release

Everything in this section was run on 2026-11-10 in fresh copies of this repository, with the commands given
above, on a machine shared with other jobs (20 CPU cores, one AMD Instinct MI300X). "Committed" means the result
files in this repository, which are those of the article.

**Earlier pipeline and 16-qubit product baselines (CPU).**

- The 22 tables of the numerical tests of the theorems and of `products16.py` were deleted, written again and
  compared with the committed ones cell by cell. Ten are identical. In the other twelve the largest relative
  difference is 2.5e-11, in a variance of order 1e-13 (`loss_var` of `uniform_noise_absolute.csv`), and 7e-14
  in every other cell. Two columns are left out of this comparison: the timing column of
  `item3_block_check.csv`, and `closure_rel` of `curvature_decomposition.csv`, a residual that is zero in exact
  arithmetic and stays at 1e-15.
- The RBM protocol was run in full for both datasets, 120 tuning runs and 15 final runs each. The RBM rows of
  `results_baselines.csv` and the file `selected_rbm.json` came out identical to the committed ones. The RBM
  rows of `mmd_sigma_sweep_baselines.csv`, which the committed table prints with 13 digits, agree to 8e-13.
- Time: 80 minutes in all. `check_theorem2_bound.py` took 5 minutes and the other theorem tests 13 minutes side
  by side. The RBM protocol took 35 minutes for the blobs and 28 minutes for Ising with 16 processes.

**Campaign (GPU).** `run_all.sh`, after `mv runs runs_committed`, trained all 40 groups (492 models) in 43
minutes and ran `analyze.py` in 8 minutes, with two jobs of another project on the same GPU.

- Final test MMD² of each model against the committed summaries, as a relative difference: at most 1.4e-9 for
  the 312 models at 16 qubits, at most 2.9e-4 for the 80 D-Wave models (median 1e-11) and at most 2.5e-4 for the
  100 MNIST models (median 4e-6).
- Means over seeds, which are what the article's figures and tables show: at most 5.8e-5 over all 70
  combinations of dataset, ancilla count, variant and initialization.
- The test loss of single models at intermediate steps differs by up to 0.8 % (MNIST) and 0.14 % (D-Wave).
- Tables of `results/` after `analyze.py`:
  - `refs.csv` and `diversity_refs.csv` are identical.
  - Final values: the branch diversity and the branch necessity change by at most 6e-4, the per-branch entries
    `delta_rel` of `branch_necessity.csv` by at most 0.5 %, the mode index by at most 0.5 %, and the loss with the
    two-body angles set to zero by at most 0.3 %.
  - Standard deviations over the five MNIST seeds are more sensitive, because that spread is itself below 1 % of
    the mean. At the final step they change by up to 2 % (`main_summary.csv`, `eqweight_control.csv`,
    `onebody_summary.csv`).
  - Curves (`main_curves.csv`, `train_loss_curves.csv`): seed means at intermediate steps change by up to 0.4 %.
    The standard deviation over seeds at intermediate steps changes by up to 19 % (MNIST, a = 4, global, step
    4000: 8.9e-6 against 7.5e-6 committed).
  - In `main_table.md` three printed numbers change: one mean in its fifth digit (15.4264 against 15.4270) and
    two standard deviations in their last printed digit.
- Entries of `mnist_branch_marginals.npz` (per-branch pixel means of the MNIST models) differ by less than
  1.4e-3 on average in every model. Single pixels differ by up to 0.02 in the global and cluster models and by up
  to 0.27 in one data-agnostic model.

Training therefore reproduces the 16-qubit results to nine digits and the losses at 484 and 784 qubits to about
four digits.

**Everything else (GPU unless stated).** In the same copy, on the retrained parameters, every other script
behind the article was then run in this order, with `TMPDIR` set to a scratch directory. Not in this chain:
`clusters.py` (below), `experiments/gd_escape/diag.py` and `experiments/gd_escape/thstar_check.py` (section "Keep
the two `thstar` files"), and the dataset scripts (section "Datasets").

```bash
cd experiments/cosine_rerun
python caches.py && python mechanism.py && python products.py && python certificate.py && python factorization.py
cd ../trainable_weights
mv runs runs_committed && PY=python ./run_all.sh && python compare.py
cd ../bandwidth_sweep && python sweep.py
cd ../exact_metrics && python targets.py && python metrics.py
cd ../modesep_bound
for ds in blobs ising dwave mnist; do python modesep_bound.py run $ds; done
python modesep_bound.py build && python modesep_bound.py check
cd ../learned_partition
python learned_partition.py partition
for ds in blobs ising dwave mnist; do python learned_partition.py train $ds; done
for ds in blobs ising dwave mnist; do python learned_partition.py parts $ds; done
python learned_partition.py build && python learned_partition.py check runs
cd ../branch_z_start
python branch_z_start.py starts
for ds in blobs ising dwave mnist; do python branch_z_start.py train $ds; done
python branch_z_start.py build && python branch_z_start.py check train
cd ../modesep_manuscript && python numbers.py build && python numbers.py check
cd ../necessity_audit && python numbers.py build && python numbers.py check
cd ../variance_scaling
for ds in blobs dwave; do python scaling.py $ds; python ancilla.py $ds; done
cd ../gd_escape
for ds in blobs ising; do for sc in global agnostic; do python escape.py $ds $sc; done; done
cd ../classical_baseline
for ds in blobs ising dwave mnist; do python fvsbn.py $ds; done
cd ../cosine_rerun && python paper_numbers.py
cd ../../manuscript/figures
for f in make_fig_*.py; do python $f; done
MODESEP_FULL=1 python make_fig_modesep.py
```

Every step ended without error. The GPU steps took 2 hours 45 minutes in all, partly with other jobs on the same
GPU, and `factorization.py` ran next to them on one CPU core. The table gives the time of each step and the
largest relative difference between its output and the committed files. Timing columns are left out.

| step | time | difference from the committed files |
|---|---|---|
| `caches.py` | 2 min | 2e-10. One integer differs: the count `coverage` of one blobs model in `modesep_blobs_a3.npz` is 7 against 6, because one branch mean equals 1/2 to round-off and the count binarizes at 1/2. The figures do not show this count. The script also writes `dwave_overlap.npz`, which is not in the repository |
| `mechanism.py` | 4 min | 16 qubits: 1e-6. D-Wave and MNIST: the terms of the test MMD² by subset order 0.8 %, the covariance and alignment tables 0.13 %, the per-branch statistics of `mechanism_couplings.csv` 2.3 %, and its counts of angles above 0.1 (`n_strong_theta2`) by up to 23 where they are below 1,000 (151 against 128 in one MNIST branch) and by up to 3.3 % above. The aggregates that the article prints from these tables do not move |
| `products.py` | 25 min | D-Wave identical, MNIST 2e-15 |
| `certificate.py` (CPU) | 1 min | identical |
| `factorization.py` (CPU, one core) | 43 min | identical |
| `trainable_weights`: `run_all.sh`, `compare.py` | 21 min | final test MMD² of each model: 1.2e-9 (blobs), 2e-12 (Ising), 2.3e-3 (D-Wave, median 3e-6), 1.2e-3 (MNIST, median 3e-5). Trained weights: at most 1.4e-3 in absolute value |
| `sweep.py` | 5 min | single points: 1e-7 (blobs), 9e-11 (Ising), 0.5 % (D-Wave), 0.4 % (MNIST) |
| `targets.py`, `metrics.py` | 5 min | Ising: 2e-11. Blobs: total variation 5e-9, KL divergence and test NLL below 1e-4 for 238 of 246 models and up to 1.1 % and 0.4 % for the others. The KL divergence is sensitive to the small probability a model leaves on a blob pattern it misses |
| `modesep_bound.py` | 20 min | 9e-10 (blobs), 7e-13 (Ising), 2.9e-4 (D-Wave), 2.5e-4 (MNIST) |
| `learned_partition.py` | 21 min | blobs and Ising: the same partition, results to 4e-10. D-Wave and MNIST: see below |
| `branch_z_start.py` | 5 min | blobs 5e-7, Ising 1e-10. D-Wave and MNIST: see below |
| the two `numbers.py` | 1 min | both checks pass. The values follow the rows above |
| `scaling.py`, `ancilla.py` | 5 min | identical |
| `escape.py` | 9 min | every escape time identical. Final losses: 1e-13 (gradient descent), 3e-5 (Adam) |
| `fvsbn.py` | 41 min | blobs, Ising and D-Wave: the same test MMD² (test NLL of the blobs to 3e-6). MNIST: see below |
| `paper_numbers.py` | 20 s | 19 of 275 lines differ, all of them for MNIST or D-Wave |
| figure scripts | 1 min | drawn from the rerun results, the figures show no visible difference from the committed PDFs. Rendered at 110 dpi, six of them are pixel-identical. In the other five, pixels differ by at most a quarter of full scale, nearly all by less than 1/25, and at most 11 per figure by more than a tenth |

**Cluster labels.** `experiments/cosine_rerun/clusters.py`, which writes the cluster labels in
`experiments/cosine_rerun/clusters/`, was run separately on CPU in another fresh copy, after the committed labels
were moved away. It took 5 minutes, and all 95 label files came back byte for byte.

**Where the rerun moves a printed value of the article.** Every table of the article that the rerun touches was
computed from the committed and from the rerun results at the precision the article prints: `tab:partition`,
`tab:app-partition`, `tab:app-partition-seeds`, `tab:mode-index`, `tab:bandwidth-sweep`, `tab:likelihood`,
`tab:eqweight-control`, `tab:weights` and `tab:product-baseline`, together with the numbers that the text quotes
from `paper_numbers.txt`, the two `numbers.json` and the mechanism tables. Nothing moves in `tab:partition`,
`tab:eqweight-control`, `tab:product-baseline`, the rows Λ of `tab:mode-index` and the aggregates of the mechanism
appendix. The 16-qubit values came back in every printed digit, apart from two entries of `tab:likelihood`. The
values below move. The signs, the order of the schemes and the intervals that include zero are the same in the
rerun and in the article. Percentages are relative to the committed value.

- **Rank correlations over the MNIST models** (`tab:mode-index`, `tab:bandwidth-sweep` and the text that quotes
  them). They are computed from the ranks of 15 models, or of 10 without the failed data-agnostic ones, and after
  retraining a few pairs of models with nearly tied values swap ranks. At the figure bandwidth these are ranks 9
  and 10 in the branch diversity and ranks 6 and 7 in the branch necessity, whose values change by less than 6e-4.
  - `tab:mode-index`, as `make_fig_diversity_scatter.py` prints it from the rerun results: ρ_D +0.21 (−0.53, +0.71)
    against +0.17 (−0.55, +0.69), and ρ_Λ −0.94 (−1.00, −0.77) against −0.95 (−1.00, −0.80). ρ_N, computed as in
    "Intervals of `tab:mode-index`", is −0.48 (−0.78, +0.14) against −0.49 (−0.79, +0.14).
  - `tab:bandwidth-sweep`: ρ_D +0.64 against +0.69 at m̄ = 1, +0.21 against +0.17 at m̄ = 2 and −0.92 against
    −0.94 at m̄ = 6. ρ_N moves by 0.01 or 0.02 at five of the six bandwidths.
  - Quoted in the text as well: the necessity without the failed models, −0.84 against −0.88, and the necessity
    against the diversity per unit loss, 0.975 against 0.979. The article prints the latter as 0.98, and the rerun
    value lies on the rounding boundary.
  - The rows of the blobs, Ising and D-Wave are unchanged.
- **The two diagnostic starts built from the trained global mixtures** (`tab:app-partition`,
  `tab:app-partition-seeds`, and the two values quoted for D-Wave in the main text). Their starts depend on the
  trained parameters, so the rerun does not begin where the committed runs began.
  - The learned partition assigns 72 of the 50,000 D-Wave strings (five seeds of 10,000) and 1,711 of the 250,000
    MNIST strings to another group. The branch-correlator starts differ by up to 3e-4 in their test MMD².
  - D-Wave, seed means of the final test MMD² (×10⁻³): 0.775 against 0.788 for the learned-partition start and
    0.494 against 0.488 for the branch-correlator start. All ten D-Wave entries of these starts in
    `tab:app-partition-seeds` move, by up to 5.6 % and 4.9 %.
  - MNIST: the seed means print as before (0.355 and 0.353). Four of the ten per-seed entries move, by 0.001 to
    0.003 (×10⁻³, at most 0.6 %).
  - A second training of the two starts from the same retrained parameters, and so from identical starts, agreed
    with the first to 3e-4 for every seed on D-Wave and MNIST. The differences above therefore come from the
    starts. On D-Wave a change of the start of this size moves the final value of a seed by a few percent.
  - Run instead on the original parameters of the campaign, which are not in this repository, the same commands
    rebuild the committed starts exactly (every string in the same group, the same start losses), and the trained
    results agree with the committed ones to 4e-4 on D-Wave and MNIST. The scripts reproduce the committed tables.
    What moves is the input, the retrained global mixtures from which the starts are built.
  - The files `results/compare.md` of these two folders also print verdict lines of the authors' internal
    protocol, which compare nearly tied means. One of them is not stable: for the branch-correlator start on
    MNIST the committed file says "lower on 4 of 5 seeds" than the cluster start and the rerun says 2 of 5. The
    article does not use these lines.
- **`tab:app-partition-seeds`, partition mixture.** One entry moves by one unit in its last digit: D-Wave seed
  2025, 0.926 against 0.925. The two values are 0.925502 and 0.925498 (×10⁻³), on either side of the rounding
  boundary.
- **`tab:likelihood`.** Two seed means of the KL divergence on the blobs: 3.671 against 3.674 (a = 1, cluster
  initialization) and 2.255 against 2.253 (a = 2, data-agnostic).
- **`tab:mode-index`, rows N_mode.** 15.04 against 15.05 and 13.32 against 13.34 on MNIST, 5.55 ± 0.67 against
  5.54 ± 0.68 on D-Wave.
- **`tab:weights`.** Five entries move by one unit in their last digit. On MNIST: two ratios of the trained to the
  fixed-weight test MMD² (0.983 against 0.984 and 0.997 against 0.998), the trained value at a = 1 with the global
  start (0.753 against 0.752, from 0.752501 and 0.752496) and the fixed-weight value at a = 4 with the
  data-agnostic start (15.426 against 15.427). On D-Wave: the upper end of one range (1.253 against 1.252).
- **Autoregressive baseline on MNIST** (not printed in the article). Early stopping picks step 3,200 against
  3,750, on a validation loss that differs by 1e-5. The test NLL agrees to 1e-6 and the test MMD² is 0.872e-3
  against 0.853e-3.

The committed results, which are those of the article, were kept.

**Tests.**

- Library of the earlier pipeline: 36 tests pass on CPU.
- `packages/iqp_fastcf`: all eight tests passed in every run on the idle GPU. In two runs with other jobs on the
  machine one test failed. In the run where it was recorded, it was `test_trainable_weights`: two runs that should
  give the same losses agreed to 1.5e-12, and the test asks for 1e-12.
