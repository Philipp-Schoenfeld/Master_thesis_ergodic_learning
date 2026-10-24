#!/bin/bash
#SBATCH -J ergodic_dfl_test
#SBATCH -t 00:40:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGTERM@120

# Short timing/correctness smoke test for the new Decision-Focused-Learning
# (DFL) lambda0 loss, NOT a real training run: a handful of epochs with and
# without --dfl_weight, at the same architecture/batch settings as the real
# run_job_ergodic.bash, so the wall-clock-per-epoch numbers extrapolate
# honestly to a full run.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
mkdir -p checkpoints

COMMON="--S 256 --nxi 25 --D 384 --p_flip 0.0 --copies_per_char 75 --lr 2e-4 --predict_lambda --viz_every 0"
N_EPOCHS=8

echo "=== [1/2] BASELINE (dfl_weight=0.0, old code path) ==="
T0=$(date +%s)
srun --unbuffered python flow_matching_runner_ergodic.py $COMMON \
    --dfl_weight 0.0 --epochs $N_EPOCHS --save_every $N_EPOCHS \
    --save_model checkpoints/dfl_test_baseline.pt
T1=$(date +%s)
echo "=== BASELINE wall time: $((T1 - T0))s for $N_EPOCHS epochs ==="

echo "=== [2/2] DFL ON (dfl_weight=0.1, default n_outer/inner_steps) ==="
T2=$(date +%s)
srun --unbuffered python flow_matching_runner_ergodic.py $COMMON \
    --dfl_weight 0.1 --epochs $N_EPOCHS --save_every $N_EPOCHS \
    --save_model checkpoints/dfl_test_on.pt
T3=$(date +%s)
echo "=== DFL-ON wall time: $((T3 - T2))s for $N_EPOCHS epochs ==="

echo "=== SUMMARY ==="
echo "baseline: $((T1 - T0))s total, $(( (T1 - T0) / N_EPOCHS ))s/epoch"
echo "dfl_on:   $((T3 - T2))s total, $(( (T3 - T2) / N_EPOCHS ))s/epoch"
