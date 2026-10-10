#!/bin/bash
#SBATCH -J smooth_v2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Round 2 of the full smoothness comparison (Philipp's request, 2026-10-09):
# all 25 holdout shapes, 7 variants (sun_refine.py's dynamics-based solver,
# point-mass and new jerk-penalised, vs. tsvec_svgd.py's new direct
# preconditioned-SVGD reimplementation of the TSVEC paper, B-spline and raw
# waypoints), --n_iters 2000, 30 particles as ONE joint SVGD swarm, every
# state logged. See run_smoothness_comparison_v2.py's module docstring for
# the variant definitions and run_job_smoothness_comparison_v2_test.bash for
# the self-test.
#
# Time/resource budget, measured from the real test run (job 164983,
# 2026-10-09, 2 shapes at this exact n_init=30/n_iters=2000 scale, AFTER
# fixing the w_smooth freeze -- see below):
#   - core computation (run_smoothness_comparison_v2.py): 10.5 min / 2 shapes
#     -> ~5.25 min/shape -> ~131 min (2.2 h) for all 25. The new tsvec_svgd
#     solver actually uses the GPU (unlike JAX/sun_refine, which still falls
#     back to CPU on this cluster -- "An NVIDIA GPU may be present ...
#     falling back to cpu" in the .err, same as round 1): bspline/raw
#     tsvec_svgd cells took ~25-45 s each at 2000 iterations, vs. ~72-78 s for
#     the CPU-bound sun_refine cells.
#   - plots (plot_smoothness_comparison.py): 154 s / 2 shapes -> ~32 min for
#     all 25 (dominated by decoding/rendering ~2001-state logs for the
#     ergodic-error panel, CPU-bound, same mechanism as round 1).
#   - total estimate: ~2.7-3 h; the 4 h limit below is a safety margin given
#     this is the first 25-shape run of brand-new solver code, not a tight
#     budget.
#   - disk: 53 MB / 2 shapes -> ~660 MB for all 25.
#
# w_smooth for the "+smoothness" tsvec_svgd variants (cfm_bspline_svgd,
# linear_bspline_svgd, linear_waypoint_svgd_smooth) is NOT the TSVEC paper's
# literal value (5.0) -- that FROZE the solver at a straight-line init (first
# test attempt, job 164977: the smoothness residual's Jacobian is constant,
# so even a zero residual contributes a large, fixed Gauss-Newton curvature,
# ~6700x the ergodic+boundary terms' combined curvature at w_smooth=5.0,
# collapsing the preconditioned step to near-zero from iteration 0). Fixed to
# W_SMOOTH_ACTIVE=1e-4 in run_smoothness_comparison_v2.py (empirically swept,
# then confirmed non-degenerate at this job's real 2 shapes/2000-iteration
# scale: smooth_final_mean 0.29/3.47 for the B-spline/raw+smoothness
# variants, vs. a near-zero 5.5e-5 at w_smooth=5.0 -- see that file's long
# comment for the full diagnosis). Still provisional/not exhaustively swept,
# same status as round 1's smoothness_weight.

OUT_TAG=smoothness_comparison_v2_20261009

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) full run -> results/${OUT_TAG}"
T0=$(date +%s)
srun --unbuffered python -u run_smoothness_comparison_v2.py \
    --out_tag ${OUT_TAG} \
    --n_init 30 \
    --n_iters 2000
rc=$?
echo "[job] $(date) run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/smoothness_comparison.db

echo "[job] $(date) plots/summary"
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}
echo "[job] $(date) done"
