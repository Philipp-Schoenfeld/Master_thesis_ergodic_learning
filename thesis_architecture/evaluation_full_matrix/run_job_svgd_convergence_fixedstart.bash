#!/bin/bash
#SBATCH -J svgd_conv_fs
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@120

# Re-run of the 2026-10-01 SVGD convergence benchmark (run_svgd_convergence.py
# -> results/svgd_convergence_20261001/plots/overview_E_total.png and
# overview_J_3000iters.png), per Philipp's request 2026-10-08:
#   - only the UCB strategy (not lse/eid),
#   - every init (cfm / random_walk / linear) AND the SVGD refinement itself
#     pinned to a fixed start point at the bottom-left corner of the
#     workspace, via the new --start_pos flag (reuses the start-conditioning
#     the CFM checkpoint, `random_walk_path` and `SvgdRefiner.refine` already
#     support -- see run_svgd_convergence.py's --start_pos help).
# Everything else matches the original run exactly: conditions
# (ground_truth, half_known, ten_samples -- none_known was not part of that
# run), methods (cfm, random_walk, linear), n_init=30, n_iters=1000,
# representation=particles, --refiner tsvec (the default before the
# 2026-10-06 switch to 'sun', matching the un-suffixed original output dir).
# Own --out_tag -> own DB, original results untouched.
#
# plot_svgd_convergence.py then writes overview_E_total.png (same as the
# original) and overview_J.png (same metric as overview_J_3000iters.png, but
# naturally capped at 1000 iterations since this DB holds no 3000-iteration
# extension -- exactly what was asked for).
#
# Time budget: the first attempt (job 164081, 100 min hard limit) only
# finished 4/12 shapes -- the fixed-start pin (`WaypointPins`, a torch
# autograd pass every SVGD iteration to get the pin penalty/gradient) turned
# out to cost ~4x an unpinned run (~16 min/shape instead of the ~4 min/shape
# the original un-pinned run implied), not the ~45 min/90 min originally
# estimated from the 2026-10-01 log. This run resumes it (same --out_tag,
# finished (shape, cond, strategy, method, init_idx) rows are skipped) with
# a generous 4h hard limit / --time_budget_h 3.5 (210 min submission budget,
# ~30 min buffer for the pool to drain and for the two plot calls) --
# estimated remaining work (~8 shapes) is ~130 min.

OUT_TAG=svgd_convergence_fixedstart_ucb_20261008

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) full run -> results/${OUT_TAG}"
srun --unbuffered python -u run_svgd_convergence.py \
    --out_tag ${OUT_TAG} \
    --conditions ground_truth,half_known,ten_samples \
    --strategies ucb \
    --start_pos 0.04,0.04 \
    --refiner tsvec \
    --workers 6 \
    --time_budget_h 3.5
rc=$?
echo "[job] $(date) run finished with exit code ${rc}"

echo "[job] $(date) plots/tables"
srun --unbuffered python -u plot_svgd_convergence.py --out_tag ${OUT_TAG}
srun --unbuffered python -u plot_svgd_convergence.py --out_tag ${OUT_TAG} --metric J
