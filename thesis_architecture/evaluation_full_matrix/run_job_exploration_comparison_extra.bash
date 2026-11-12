#!/bin/bash
#SBATCH -J expl_cmp_extra
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Exploration comparison, the MISSING methods only (run_exploration_comparison.py): self-supervised
# + TSVEC, best U-Net (old500), ergodic search without prior (lintsvec), line to a GP-sampled point
# (gpgoal), best straight ray (linray). The first five methods (unet630, dit630, rw, lawn, linadam)
# are already in plots/exploration_comparison/exploration_comparison.db and are NOT recomputed.
#
# The cluster job writes its OWN DB (only the new methods); fold it into the main DB locally:
#   python merge_exploration_dbs.py --src plots/exploration_comparison_extra/exploration_comparison_extra.db
#
# 25 shapes x 7 variants x 2 kernels x 5 methods = 1750 missions (n_rounds 5, 8 candidates, 500 iterations,
# same seeds as the first run). Estimate from the stored mission times of the first run (local RTX 2070):
# ~4-5 h; on a cluster GPU less. --max_minutes stops cleanly, re-submitting continues (resumable).
#
# Prerequisites on the cluster: checkpoints/selfsup_..._final.pt (selfsup_planner.DEFAULT_SELFSUP_CKPT),
# transfer/netz2d_startpunkt.pt (old500).
#
#   sbatch run_job_exploration_comparison_extra.bash
#   METHODS=selfsup,old500 sbatch run_job_exploration_comparison_extra.bash     # subset

METHODS=${METHODS:-selfsup,old500,lintsvec,gpgoal,linray}
MAX_MIN=${MAX_MIN:-540}
TAG=${TAG:-}      # optional suffix so parallel jobs (e.g. one per method) write separate DBs
OUT_DIR=plots/exploration_comparison_extra${TAG:+_$TAG}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) methods=${METHODS}"
srun --unbuffered python -u run_exploration_comparison.py \
    --methods ${METHODS} \
    --db ${OUT_DIR}/exploration_comparison_extra.db \
    --out_dir ${OUT_DIR} \
    --max_minutes ${MAX_MIN} || exit 1
echo "[job] $(date) finished"
