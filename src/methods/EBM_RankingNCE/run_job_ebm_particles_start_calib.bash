#!/bin/bash
#SBATCH -J ebm_rankingnce_calib
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=debug
##SBATCH -C 'vram24gb|vram32gb|vram48gb'
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=SIGTERM@120

# Timing calibration, NOT a real training run: a handful of epochs at the
# real (default) architecture/batch settings, so the wall-clock-per-epoch
# number extrapolates honestly to a full run before committing cluster time
# to it. Same idea as thesis_architecture/run_job_ergodic_dfl_test.bash.
# No -p needed (`main` is the default partition since `stud` closed on
# 2026-10-16); --qos=debug starts ahead of the regular queue and fits (this
# job needs well under its 2 GPU / 4h cap), which is the intended use for a
# short timing/correctness check like this one -- see CLAUDE.md.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/src/methods/EBM_RankingNCE
mkdir -p checkpoints

srun --unbuffered python3 -u train_ebm_particles_start.py \
    --epochs 3 --save_every 3 --tag calib
