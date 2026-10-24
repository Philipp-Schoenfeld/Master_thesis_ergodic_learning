#!/bin/bash
#SBATCH -J dfl_batch_diag
#SBATCH -t 00:20:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'

# Diagnostic only (10 batches, mini_batch=64, copies_per_char=5 -> tiny
# dataset so this loads fast): finds whether cfm_loss or the DFL term is
# the one that turns NaN first in real training, and inspects the model's
# own lambda0 output at that point.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
export COPIES_PER_CHAR=75
export N_BATCHES=900
srun --unbuffered python dfl_batch_diagnostic.py
