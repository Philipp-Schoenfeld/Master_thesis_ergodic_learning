#!/bin/bash
#SBATCH -J belief_np_pretrain
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Meta-trains the attentive neural process belief (train_belief_np.py) on the
# train split with the same measurement model as the AK belief run
# (K = 32-128 measurements, noise std 0.01). --ak_reference fits the AK-GP once
# on the same val measurements, so W&B shows NP vs. AK reconstruction quality.
# Followed by run_job_particles_ergloss_w300_belief_np.bash (afterok).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

ARGS=(
    --k_min 32
    --k_max 128
    --noise 0.01
    --steps 50000
    --batch 64
    --ak_reference
    --use_wandb
)

LATEST_CKPT=$(ls -t checkpoints/belief_np_date_*_D128_K32-128_n0.01_ep*.pt 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Resuming NP pretraining from: $LATEST_CKPT"
    srun --unbuffered python -u train_belief_np.py --resume "$LATEST_CKPT" "${ARGS[@]}"
else
    echo "Starting NP pretraining..."
    srun --unbuffered python -u train_belief_np.py "${ARGS[@]}"
fi
