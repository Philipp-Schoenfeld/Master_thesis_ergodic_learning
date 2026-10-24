#!/bin/bash
#SBATCH -J ergodic_dfl_test_c
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGTERM@120

# job 168020 (mini_batch=512, same as baseline) OOM'd immediately: baseline
# forward/backward alone already uses ~11.57/11.63 GiB on this GPU, leaving
# no headroom for the DFL unrolled second-order graph. Retry at a much
# smaller mini_batch to get an actual DFL epoch time, and additionally with
# the default mini_batch but a cheaper DFL config (fewer outer/inner steps)
# to see whether that alone would have fit.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
mkdir -p checkpoints

COMMON="--S 256 --nxi 25 --D 384 --p_flip 0.0 --copies_per_char 75 --lr 2e-4 --predict_lambda --viz_every 0 --dfl_weight 0.1"

echo "=== [A] DFL ON, mini_batch=64, default n_outer=5/inner_steps=8 ==="
T0=$(date +%s)
srun --unbuffered python flow_matching_runner_ergodic.py $COMMON \
    --mini_batch 64 --epochs 3 --save_every 3 \
    --save_model checkpoints/dfl_test_c_mb64.pt
RC_A=$?
T1=$(date +%s)
echo "=== [A] exit=$RC_A wall time: $((T1 - T0))s for 3 epochs, $(( (T1 - T0) / 3 ))s/epoch ==="

echo "=== [B] DFL ON, mini_batch=512 (production), n_outer=2/inner_steps=4 ==="
T2=$(date +%s)
srun --unbuffered python flow_matching_runner_ergodic.py $COMMON \
    --mini_batch 512 --dfl_n_outer 2 --dfl_inner_steps 4 \
    --epochs 3 --save_every 3 \
    --save_model checkpoints/dfl_test_c_mb512_small.pt
RC_B=$?
T3=$(date +%s)
echo "=== [B] exit=$RC_B wall time: $((T3 - T2))s for 3 epochs, $(( (T3 - T2) / 3 ))s/epoch ==="
