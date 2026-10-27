#!/bin/bash
#SBATCH -J smooth_proj
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Adds "optimise waypoints, then project onto a B-spline" variants to the v6 smoothness comparison
# (Philipp's request, 2026-10-26). NO solver is re-run: the six waypoint variants of v6
# (TSVEC raw / +smoothness, Adam raw / +smoothness, Sun point-mass, Sun jerk) already store every
# iterate, so `project_waypoint_variants.py` least-squares-fits each logged 128-point curve with the
# NXI = 25 control-point B-spline (cp 0 fixed to the start) and stores it as `<variant>_proj`.
# CPU only (no GPU): projection, smoothness/path-length series and the ergodic-error checkpoints are
# cheap; the job is resumable per (shape, variant, candidate), so re-submitting after a time-out
# continues where it stopped.
#
# Steps: project -> convergence + summary on the new DB -> combine with v6's summary/convergence
# -> overview, overview_convergence, convergence_linear of the combined result.

SRC_TAG=smoothness_comparison_v6_20261012
PROJ_TAG=smoothness_comparison_v6_wpproj_20261026
COMB_TAG=smoothness_comparison_v6_with_proj_20261026

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

for f in results/${SRC_TAG}/smoothness_comparison.db results/${SRC_TAG}/summary.csv results/${SRC_TAG}/convergence.npz; do
    [ -f "$f" ] || { echo "[job] missing $f"; exit 1; }
done

echo "[job] $(date) projecting waypoint variants of ${SRC_TAG}"
srun --unbuffered python -u project_waypoint_variants.py --src_tag ${SRC_TAG} --out_tag ${PROJ_TAG} || exit 1

echo "[job] $(date) convergence + summary for ${PROJ_TAG}"
srun --unbuffered python -u compute_convergence.py --out_tag ${PROJ_TAG} || exit 1
srun --unbuffered python -u plot_smoothness_comparison.py --out_tag ${PROJ_TAG} || exit 1

echo "[job] $(date) combining with ${SRC_TAG} -> ${COMB_TAG}"
srun --unbuffered python -u combine_projection_results.py --base_tag ${SRC_TAG} --proj_tag ${PROJ_TAG} \
    --out_tag ${COMB_TAG} || exit 1
srun --unbuffered python -u plot_smoothness_overview.py --out_tag ${COMB_TAG}
srun --unbuffered python -u plot_convergence_linear.py --out_tag ${COMB_TAG}

du -sh results/${PROJ_TAG}
echo "[job] $(date) done"
