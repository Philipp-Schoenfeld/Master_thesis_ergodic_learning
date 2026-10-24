#!/bin/bash
#SBATCH -J belief_np_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:45:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 2

# GPU smoke test before the real NP runs:
# 1) 3000 NP pretraining steps at the real batch size: step time, val trend
#    against the AK reference on identical measurements.
# 2) 3 epochs of the real w300 flow config with that NP belief: pool build
#    time, epoch time, holdout plot.
# Checkpoints go to checkpoints/test/, which the real job scripts' globs never
# match, so nothing from this test can be resumed by accident.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture
mkdir -p checkpoints/test

echo "[$(date +%T)] NP pretraining test start"
srun --unbuffered python -u train_belief_np.py \
    --k_min 32 --k_max 128 --noise 0.01 \
    --steps 3000 --batch 64 --val_every 500 --save_every 0 \
    --ak_reference \
    --save_model checkpoints/test/belief_np.pt || exit 1
echo "[$(date +%T)] NP pretraining test done"

NP_CKPT=$(ls -t checkpoints/test/belief_np_date_*_final.pt | head -1)
echo "Using NP belief: $NP_CKPT"

echo "[$(date +%T)] Flow training test start"
srun --unbuffered python -u flow_matching_runner_particles.py \
    --D 384 --n_particles 256 --copies_per_char 15 --p_flip 0.0 \
    --epochs 3 --mini_batch 256 --save_every 0 --viz_every 0 \
    --lambda_erg 300 --erg_K 8 --erg_pts 128 --erg_t_power 2.0 \
    --belief_mode np --belief_np_ckpt "$NP_CKPT" \
    --belief_k_min 32 --belief_k_max 128 --belief_noise 0.01 --belief_pool 8 \
    --save_model checkpoints/test/cond_particles_crossattn.pt
echo "[$(date +%T)] Flow training test done"
