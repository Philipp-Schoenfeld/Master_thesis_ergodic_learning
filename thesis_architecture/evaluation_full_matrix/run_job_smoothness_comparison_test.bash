#!/bin/bash
#SBATCH -J smooth_cmp_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Test run for the big smoothness comparison (Philipp's request, 2026-10-08):
# 5 variants (cfm_only, cfm_svgd, linear_svgd_bspline, linear_svgd_raw,
# linear_svgd_raw_smooth), each a distribution of 30 trajectories, on ALL 25
# holdout shapes ('val' split of ergodic_dataset_775.db -- no unknown-region
# shapes in there, matching "without unknown regions"), every SVGD state of
# every trajectory logged to results/<out_tag>/smoothness_comparison.db.
#
# This job does NOT run the full 25-shape matrix -- it only
#   1. runs the self-test (test_smoothness_comparison.py; needs the GPU +
#      the real CFM checkpoint for its last part),
#   2. runs the REAL per-cell workload (n_init=30, n_iters=600, all 5
#      variants) on a SMALL subset of shapes (--n_test_shapes below),
#   3. prints disk usage and the per-(shape,variant) wall-clock time
#      (run_smoothness_comparison.py already logs "done in Xs" per cell),
# so that the full 25-shape run's time/disk budget can be read off directly
# instead of guessed. The full run is NOT started automatically -- ask before
# running it (see run_job_smoothness_comparison.bash, written after this
# test's numbers are in).
#
# New code exercised here for the first time on a GPU (only lightly checked
# locally, no local JAX/GPU available): `exploration/common/sun_refine.py`'s
# two opt-in extensions (`log_space='raw'`: genuine per-step raw-waypoint
# logging instead of always going through a B-spline fit; `smoothness_weight`:
# an explicit smoothness force in the FM-Stein score, the "sun" backend had
# none before). Both default to the old behaviour and are covered by
# test_smoothness_comparison.py's parts 1-4 (CPU, numpy/toy phi) -- this job
# is the first time they run inside the real batched GPU path
# (svgd_batched.BatchedSunTorch) against a real holdout shape and the real
# CFM checkpoint.
#
# `--smoothness_weight` (default in run_smoothness_comparison.py: 200.0) is
# explicitly PROVISIONAL -- read this test's linear_svgd_raw vs.
# linear_svgd_raw_smooth smoothness numbers (results/<out_tag>/summary.csv)
# before trusting it for the full run; it may need an order-of-magnitude
# adjustment either way (see run_smoothness_comparison.py's module docstring).

OUT_TAG=smoothness_comparison_test_20261008
N_TEST_SHAPES=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[test] $(date) self-test"
srun --unbuffered python -u test_smoothness_comparison.py || { echo "[test] self-test FAILED"; exit 1; }

echo "[test] $(date) mini-run: ${N_TEST_SHAPES} shapes, all 5 variants, n_init=30, n_iters=600 (real scale)"
T0=$(date +%s)
srun --unbuffered python -u run_smoothness_comparison.py \
    --out_tag ${OUT_TAG} \
    --n_shapes ${N_TEST_SHAPES} \
    --n_init 30 \
    --n_iters 600
rc=$?
echo "[test] $(date) mini-run exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/smoothness_comparison.db

echo "[test] $(date) plots/summary"
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}

echo "[test] $(date) done -- read the per-cell 'done in Xs' lines above (or"
echo "  %x-%j.out) to extrapolate the full 25-shape run's time/disk budget."
cat results/${OUT_TAG}/summary.csv
