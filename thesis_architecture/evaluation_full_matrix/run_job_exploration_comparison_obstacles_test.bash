#!/bin/bash
#SBATCH -J expl_obst_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=16G
#SBATCH -c 4

# Pre-flight of the obstacle benchmark at the REAL budget (8 candidates, 500 SVGD iterations,
# 5 rounds) on 2 heldout shapes x 1 variant x both kernels for all 13 methods: GPU timing per method
# and collision check of every driven path (check_obstacle_runs.py). Own folder, does not touch the
# benchmark DBs.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

# RUN=<n>: separate folder per pre-flight run (the runner resumes, it would skip finished rows)
OUT_DIR=plots/exploration_comparison_obstacles_dice6_r0.08_preflight${RUN:+_${RUN}}
echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
# 1) unit tests of the obstacle building blocks (CPU, ~3 min)
srun --unbuffered python -u test_workspace_obstacles.py || exit 1
# 2) all 13 methods at the real budget on 2 shapes, then the independent collision check
srun --unbuffered python -u run_exploration_comparison.py \
    --obstacles dice6 --n_shapes ${N_SHAPES:-2} --n_variants ${N_VARIANTS:-1} \
    --methods unet630,dit630,rw,lawn,linadam,selfsup,old500,unet0,unet1890,lm1300,lintsvec,gpgoal,linray \
    --db ${OUT_DIR}/exploration_comparison.db --out_dir ${OUT_DIR} || exit 1
srun --unbuffered python -u check_obstacle_runs.py --db ${OUT_DIR}/exploration_comparison.db
echo "[job] $(date) finished"
