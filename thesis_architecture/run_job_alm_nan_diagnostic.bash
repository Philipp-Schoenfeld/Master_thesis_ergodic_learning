#!/bin/bash
#SBATCH -J alm_nan_diag
#SBATCH -t 00:10:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud

# Pure diagnostic, CPU-only, no GPU needed: finds exactly where the DFL loss
# first turns NaN at production scale (real bsplinax basis, real DB data,
# S=256). Not a training run.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
srun --unbuffered python alm_nan_diagnostic.py
