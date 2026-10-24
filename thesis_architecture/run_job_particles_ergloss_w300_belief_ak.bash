#!/bin/bash
#SBATCH -J flow_erg_w300_ak
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# CFM+ErgLoss (w=300) conditioned on an Attentive-Kernel GP belief built from
# 32-128 noisy point measurements per shape, instead of the true density.
# All other hyperparameters match run_job_particles_ergloss_w300.bash, so the
# two runs compare directly. The ergodic loss still targets the true density.
#
# Resources: the w300 reference run (job 132007) took 4h08 on an RTX 3080 Ti
# with 3.6 GB peak RSS; on an RTX 2080 Ti the same setup took ~18h, and on the
# V100 (no native bf16, training uses bf16 autocast) an epoch took ~160 s
# instead of ~31 s. The 'rtx' feature covers exactly the Ampere cards
# (3080 Ti, 3090, A5000, gn11); the vram* tags are cumulative and would
# also match the V100. The belief pool (750 shapes x 8) adds a few minutes at startup
# and is rebuilt identically on resume (fixed --belief_seed).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

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
    --belief_mode ak
    --belief_k_min 32
    --belief_k_max 128
    --belief_noise 0.01
    --belief_pool 8
    --use_wandb
)

LATEST_CKPT=$(ls -t checkpoints/*flow_matching_particle_ergodic_date_*_nxi25_D384_N256_C15_flip0.0_ERGLOSS-w${WEIGHT}-K8-tp2_BELIEF-AK-K32-128-n0.01-P8_ep*.pt 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming training from: $LATEST_CKPT"
    srun --unbuffered python -u flow_matching_runner_particles.py --resume "$LATEST_CKPT" "${ARGS[@]}"
else
    echo "Starting new training..."
    srun --unbuffered python -u flow_matching_runner_particles.py "${ARGS[@]}"
fi
