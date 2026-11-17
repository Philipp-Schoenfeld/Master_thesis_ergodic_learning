#!/bin/bash
#SBATCH -J expl_cmp_obst
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Active-sensing benchmark WITH OBSTACLES (run_exploration_comparison.py --obstacles dice6): the same
# missions as the obstacle-free comparison (25 heldout shapes x 7 variants x 2 kernels, 5 rounds,
# 8 candidates, 500 SVGD iterations, kappa 1 -> 0.05, same seeds), six circular obstacles of radius
# 0.08 arranged like the pips of a die, obstacle avoidance in every method (workspace_obstacles.py).
#
# -C rtx: the V100 of the dgx-station ran this pipeline ~7x slower (pre-flight 211242: 170 s vs. 23 s
# per mission on an RTX 3090).
#
# One job per method group, each with its OWN DB (TAG), merged afterwards:
#   python merge_exploration_dbs.py --src plots/exploration_comparison_obstacles_dice6_r0.08_<TAG>/exploration_comparison.db \
#       --dst plots/exploration_comparison_obstacles_dice6_r0.08/exploration_comparison_obstacles_dice6_r0.08.db
# Resumable: finished (shape, kernel, method, variant) rows are skipped, so after a time limit or a
# scavenger preemption the same command simply continues.
#
#   TAG=a METHODS=unet630,old500 sbatch run_job_exploration_comparison_obstacles.bash

METHODS=${METHODS:?set METHODS=...}
TAG=${TAG:?set TAG=...}
MAX_MIN=${MAX_MIN:-930}
NAME=exploration_comparison_obstacles_dice6_r0.08
OUT_DIR=plots/${NAME}_${TAG}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) methods=${METHODS} tag=${TAG}"
srun --unbuffered python -u run_exploration_comparison.py \
    --obstacles dice6 \
    --methods ${METHODS} \
    --db ${OUT_DIR}/exploration_comparison.db \
    --out_dir ${OUT_DIR} \
    --max_minutes ${MAX_MIN} || exit 1
echo "[job] $(date) finished"
