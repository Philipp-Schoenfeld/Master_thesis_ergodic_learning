#!/bin/bash
#SBATCH -J smooth_cmp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Full smoothness comparison (Philipp's request, 2026-10-08): all 25 holdout
# shapes ('val' split, ergodic_dataset_775.db -- no unknown-region shapes in
# there, i.e. already "without unknown regions"), 5 variants, each a
# distribution of 30 trajectories, n_iters=600, every SVGD state logged
# (results/<out_tag>/smoothness_comparison.db). See
# run_smoothness_comparison.py's module docstring for the variant definitions
# and run_job_smoothness_comparison_test.bash for the self-test.
#
# Time/resource budget, measured from the real test run (job 164887,
# 2026-10-09, 2 shapes at the exact same n_init=30/n_iters=600 scale):
#   - core computation (run_smoothness_comparison.py): 103 s / 2 shapes
#     -> ~52 s/shape -> ~22 min for all 25
#   - plots (plot_smoothness_comparison.py): ~125 s / 2 shapes
#     -> ~26 min for all 25 (dominated by the per-trajectory arclength
#     resampling in the smoothness metric / distribution plots, CPU-bound)
#   - total estimate: well under 1 h; the 2 h limit below is a safety margin,
#     not a tight budget -- no --time_budget_h / resume logic needed.
#   - disk: ~22 MB / 2 shapes -> well under 1 GB for all 25.
# Note: this cluster's JAX falls back to CPU (no CUDA-enabled jaxlib in the
# `thesis` env -- "An NVIDIA GPU may be present ... falling back to cpu" in
# the test job's .err) for the "sun" FM-Stein solver; the above timings
# already reflect that, not an idealised GPU number. The CFM forward pass
# (torch) does use the GPU. Worth fixing the env at some point, but it did
# not block this run or make it slow enough to matter.
#
# --smoothness_weight (default 200, see run_smoothness_comparison.py) is
# PROVISIONAL but validated on real data: at 600 iterations it produced a
# real ~30% reduction in smoothness energy vs. the no-force raw-waypoint
# variant (not negligible, not runaway) -- see results/
# smoothness_comparison_test_20261008/plots/per_shape/A.png. Left unchanged
# for this run; revisit after seeing the full 25-shape distribution.

OUT_TAG=smoothness_comparison_20261009

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) full run -> results/${OUT_TAG}"
T0=$(date +%s)
srun --unbuffered python -u run_smoothness_comparison.py \
    --out_tag ${OUT_TAG} \
    --n_init 30 \
    --n_iters 600
rc=$?
echo "[job] $(date) run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/smoothness_comparison.db

echo "[job] $(date) plots/summary"
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}
echo "[job] $(date) done"
