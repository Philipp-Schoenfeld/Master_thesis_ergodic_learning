#!/bin/bash
#SBATCH -J mission_budget_fx_replot
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:15:00
#SBATCH -p stud
#SBATCH --mem=6G
#SBATCH -c 4

# Re-runs ONLY the plot_mission_continuous.py --svgd_conv step for the completed
# full-execution budget-matched run (job 166924, finished 2026-10-12, exit 0,
# mission data already correctly stored). The first attempt crashed on a
# consistency-check false positive (worst_seg=1.0 from a 1-point resample-length
# off-by-one on 14/4944 rounds, caused by 16-bit state-blob quantisation noise on
# the now-longer full_execution candidates -- see plot_mission_continuous.py's
# comment at the shard_curves shape-mismatch branch); fixed to tolerate a <=2
# point difference by comparing the shared prefix. No GPU needed (CPU-only,
# reads the stored shard DBs).

OUT_TAG=mission_eval_budget_matched_fullexec_20261011
PLOT_TAG=${OUT_TAG}_sun

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) replot -> results/${PLOT_TAG}"
srun --unbuffered python -u plot_mission_continuous.py --out_tag ${PLOT_TAG} \
    --max_units 28 --workers 4 --svgd_conv --conv_window 100 --conv_tol 0.05 --n_iters 600
rc=$?
echo "[job] $(date) replot finished with exit code ${rc}"
