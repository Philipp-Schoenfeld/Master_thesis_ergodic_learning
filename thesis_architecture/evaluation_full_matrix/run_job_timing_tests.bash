#!/bin/bash
#SBATCH -J timing_tests
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx3090
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Compute-time tests of the active-sensing pipeline (run_timing_tests.py): network forward vs. n_init,
# network-free candidate sources, SVGD refinement vs. n_init / iterations (TSVEC, Adam), belief update
# of rbf / ak / np vs. stored measurements, and one full mission per method x kernel on 3 shapes.
# Times are only comparable on one machine: the job pins ONE GPU type (rtx3090; change the -C line to
# time another type) and writes hardware.json with the results.
# Do not run other jobs on the same node at the same time if avoidable (timing noise).
#
# Estimate: components ~30-40 min, missions 13 methods x 2 kernels x 3 shapes x 2 reps ~ 1-1.5 h.
#
#   sbatch run_job_timing_tests.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
# OBST=dice6: the pipeline of the obstacle benchmark (results/timing_tests_cluster_rtx3090_obstacles_dice6_r0.08/).
srun --unbuffered python -u run_timing_tests.py --tag cluster_rtx3090 --refiners tsvec_svgd,adam_svgd \
    ${OBST:+--obstacles ${OBST}} || exit 1
echo "[job] $(date) finished"
