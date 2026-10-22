#!/bin/bash
#SBATCH -J smooth_v3
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Round 3 of the full smoothness comparison (Philipp's request, 2026-10-11):
# all 25 holdout shapes, 11 variants, --n_iters 2000, every state logged.
#   cfm_only, cfm_bspline_svgd, linear_bspline_svgd, linear_waypoint_svgd,
#   linear_waypoint_svgd_smooth, linear_sun_pointmass, linear_sun_jerk,
#   cfm_adam_bspline, linear_adam_bspline, linear_adam_waypoint,
#   linear_adam_waypoint_smooth.
# Re-use: the round-2 DB (job 164986) is still VALID for cfm_only and the two sun
# variants (same settings, solver code unchanged) -> they are copied over and only
# loaded. Its four TSVEC variants were computed with the broken solver (see
# tsvec_svgd.py, "Energy scale and update-rule fixes") and are dropped, as are
# all runs with n_iters != 2000. TSVEC (fixed solver, linear_waypoint_svgd with
# LM damping 10) and the four new Adam-SVGD variants are computed fresh.
# The runner resumes per (shape, variant, candidate), so re-submitting this
# script after a time-out continues where it stopped.
#
# Estimate: sun cells (~1 h) are skipped; TSVEC ~25-45 s/cell (GPU) x 4 x 25
# ~1.5 h; Adam-SVGD x 4 x 25 cells; plots ~1 h. The 12 h limit is a safety
# margin; --time_budget_h 9 stops submitting new cells before the plots.

OUT_TAG=smoothness_comparison_v3_20261011
SRC_TAG=smoothness_comparison_v2_20261009

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) seeding results/${OUT_TAG} from ${SRC_TAG}"
python -u seed_smoothness_db.py \
    --src results/${SRC_TAG}/smoothness_comparison.db \
    --dst results/${OUT_TAG}/smoothness_comparison.db \
    --keep cfm_only,linear_sun_pointmass,linear_sun_jerk \
    --n_iters 2000

echo "[job] $(date) full run -> results/${OUT_TAG}"
T0=$(date +%s)
srun --unbuffered python -u run_smoothness_comparison_v2.py \
    --out_tag ${OUT_TAG} \
    --n_init 30 \
    --n_iters 2000 \
    --time_budget_h 9
rc=$?
echo "[job] $(date) run finished with exit code ${rc} after $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/smoothness_comparison.db

echo "[job] $(date) plots/summary"
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}
echo "[job] $(date) done"
