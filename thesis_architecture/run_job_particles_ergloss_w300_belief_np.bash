#!/bin/bash
#SBATCH -J flow_erg_w300_np
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# CFM+ErgLoss (w=300) conditioned on a neural-process belief from 32-128 noisy
# point measurements per shape. Identical to
# run_job_particles_ergloss_w300_belief_ak.bash except --belief_mode np, so the
# two runs compare AK vs. NP directly. Uses the newest final NP checkpoint from
# run_job_belief_np_pretrain.bash; submit with --dependency=afterok:<pretrain id>.
# '-C rtx' = Ampere cards only; on the V100 (no native bf16) an epoch took
# ~160 s instead of ~31 s (test job 180557).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

NP_CKPT=$(ls -t checkpoints/belief_np_date_*_D128_K32-128_n0.01_final.pt 2>/dev/null | head -1)
if [ ! -f "$NP_CKPT" ]; then
    echo "No final NP checkpoint found in checkpoints/, aborting."
    exit 1
fi
echo "Using NP belief: $NP_CKPT"

WEIGHT=300
ARGS=(
    --D 384
    --n_particles 256
    --copies_per_char 15
    --p_flip 0.0
    --epochs 500
    --mini_batch 256
    --save_every 20
    --viz_every 20
    --lambda_erg $WEIGHT
    --erg_K 8
    --erg_pts 128
    --erg_t_power 2.0
    --belief_mode np
    --belief_np_ckpt "$NP_CKPT"
    --belief_k_min 32
    --belief_k_max 128
    --belief_noise 0.01
    --belief_pool 8
    --use_wandb
)

LATEST_CKPT=$(ls -t checkpoints/*flow_matching_particle_ergodic_date_*_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w${WEIGHT}-K8-tp2_BELIEF-NP-K32-128-n0.01-P8_ep*.pt 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming training from: $LATEST_CKPT"
    srun --unbuffered python -u flow_matching_runner_particles.py --resume "$LATEST_CKPT" "${ARGS[@]}"
else
    echo "Starting new training..."
    srun --unbuffered python -u flow_matching_runner_particles.py "${ARGS[@]}"
fi
