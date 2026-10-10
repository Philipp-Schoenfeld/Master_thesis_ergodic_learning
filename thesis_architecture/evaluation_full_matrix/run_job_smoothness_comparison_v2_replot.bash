#!/bin/bash
#SBATCH -J smooth_v2_replot
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH -p stud
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Re-plot only (Philipp's request, 2026-10-09): symlog y-axis on the
# smoothness-energy panel so the near-zero variants (B-spline, smoothness
# force) stay visible alongside the unconstrained raw-waypoint variant's
# growth past 20 -- a linear axis flattened them into noise (see
# plot_smoothness_comparison.py's `_metrics`/`_apply_yscale`). No SVGD
# re-run needed, reuses the already-computed round-2 DB (job 164986,
# results/smoothness_comparison_v2_20261009) -- no GPU requested, pure
# CPU/sqlite/matplotlib, same as round 1's replot job (164948).

OUT_TAG=smoothness_comparison_v2_20261009

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) re-plotting results/${OUT_TAG}"
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${OUT_TAG}
rc=$?
echo "[job] $(date) done, exit code ${rc}"
