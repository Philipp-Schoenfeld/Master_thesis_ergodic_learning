#!/bin/bash
#SBATCH -J ergodic_dfl_test_b
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGTERM@120

# DFL-only timing (separate job, runs in parallel with the baseline-only
# job 167989): baseline epoch 1 measured at 607s/ep at these settings, so
# this gets its own generous time budget instead of sharing a 40min window
# with the baseline phase.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
mkdir -p checkpoints

echo "=== DFL ON (dfl_weight=0.1, default n_outer=5/inner_steps=8) ==="
T0=$(date +%s)
srun --unbuffered python flow_matching_runner_ergodic.py \
    --S 256 --nxi 25 --D 384 --p_flip 0.0 --copies_per_char 75 --lr 2e-4 \
    --predict_lambda --viz_every 0 \
    --dfl_weight 0.1 --epochs 3 --save_every 3 \
    --save_model checkpoints/dfl_test_on.pt
T1=$(date +%s)
echo "=== DFL-ON wall time: $((T1 - T0))s for 3 epochs, $(( (T1 - T0) / 3 ))s/epoch ==="
