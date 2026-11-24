#!/bin/bash
#SBATCH -J expl_cmp_3d
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# 3D exploration comparison (run_exploration_comparison_3d.py): the 2D active-sensing benchmark
# (25 heldout shapes x 7 variants x {RBF, AK}, 5 rounds, 8 candidates, 500 SVGD iterations,
# kappa 1 -> 0.05, same seeds) with the densities on a plane in the unit cube and everything planned /
# measured / scored in 3D coordinates with the 3D networks.
#
# -C rtx: the V100 of the dgx-station ran the 2D pipeline ~7x slower.
#
# One job per method group, each with its OWN DB (TAG), merged afterwards (add-only):
#   python run_exploration_comparison_3d.py --plane flat --merge_from \
#       plots/exploration_comparison_3d_flat_<TAG>/exploration_comparison.db
# Resumable: finished (shape, kernel, method, variant) rows are skipped, so after the time limit or a
# scavenger preemption the same command simply continues.
#
#   TAG=a METHODS=unet630,dit630 sbatch run_job_exploration_comparison_3d.bash
#   TAG=b METHODS=dits_old,nooffset200 sbatch run_job_exploration_comparison_3d.bash
#   TAG=c METHODS=selfsup,lintsvec,linadam sbatch run_job_exploration_comparison_3d.bash
#   TAG=d METHODS=rw,lawn,gpgoal,linray,tsp sbatch run_job_exploration_comparison_3d.bash
# optional: PLANE=tilted (default flat), EXTRA="--walk_len 6.1 --tsp_points 12"

METHODS=${METHODS:?set METHODS=...}
TAG=${TAG:?set TAG=...}
PLANE=${PLANE:-flat}
MAX_MIN=${MAX_MIN:-1380}
OUT_DIR=plots/exploration_comparison_3d_${PLANE}_${TAG}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) methods=${METHODS} tag=${TAG} plane=${PLANE}"
srun --unbuffered python -u run_exploration_comparison_3d.py \
    --plane ${PLANE} \
    --methods ${METHODS} \
    --out_dir ${OUT_DIR} \
    --no_plots \
    --max_minutes ${MAX_MIN} ${EXTRA} || exit 1
echo "[job] $(date) finished"
