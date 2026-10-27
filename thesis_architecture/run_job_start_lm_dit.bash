#!/bin/bash
#SBATCH -J start_lm_dit
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120
#SBATCH --requeue
#SBATCH --open-mode=append

# DiT backbone (--backbone dit: Diffusion Transformer with adaLN-Zero + particle cross-attention,
# flow_matching_cond_particles_dit.py, DiT-S sizes D384 / 12 blocks / 6 heads, ~40M params)
# against the U-Net (87.5M) on the same data and the SAME schedule as the 1-day lambda ablation
# (run_job_start_lm_lambda.bash: 180 epochs x 25 copies = 120k steps, cosine lr 1e-4, batch 256,
# Sinkhorn ErgLoss blur 0.05 t^2, --target_cps --start_inpaint, LM teacher DB). Only the
# backbone differs, so e.g. DiT at lambda 630 is compared to U-Net job lam630 (ERG50).
#
#   sbatch -J dit630 run_job_start_lm_dit.bash 630 ERG50
#   sbatch -J dit630 --qos=scavenger run_job_start_lm_dit.bash 630 ERG50   # beyond the 3-GPU limit
#
# Scavenger: on preemption Slurm sends SIGTERM (120 s grace), the runner writes an emergency
# _ep####.pt, --requeue puts the job back and this script resumes from it (same job ID, log
# appended). Regular checkpoints every 10 epochs. A finished run (_final.pt) makes a restart a no-op.
#
# Time (test job 183523, RTX 3090, this config): DiT 0.126 s/step = 84 s/epoch with ErgLoss
# (U-Net on the same card: 0.505 s/step = 336 s/epoch, 4x slower); 180 epochs = ~4.2 h + 4 holdout
# viz + ~9 min density grids at every (re)start -> ~5 h; -t 12 h leaves room for slower rtx cards.

LAMBDA=${1:?lambda_erg}
TAG=${2:?tag, e.g. ERG50}
EPOCHS=180
COPIES=25

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi 25 --D 384 --backbone dit --dit_depth 12 --dit_heads 6 --dit_mlp_ratio 4 \
      --n_particles 256 --copies_per_char ${COPIES} --p_flip 0.0 \
      --epochs ${EPOCHS} --mini_batch 256 --save_every 10 --viz_every 50 \
      --lambda_erg ${LAMBDA} --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --target_cps --start_inpaint --tag LAM${TAG} --keep_checkpoints 1 \
      --db ergodic_dataset_lm_nxi25.db --use_wandb"
PATTERN="checkpoints/*_START_*_CPS_INPAINT_DIT-d12-h6_LAM${TAG}_*"

FINAL_DONE=$(ls ${PATTERN}_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t ${PATTERN}_ep*.pt 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  lambda_erg=${LAMBDA} tag=${TAG}  restart=${SLURM_RESTART_COUNT:-0}  start: $(date)"
if [ -f "$FINAL_DONE" ]; then
    echo "Run already finished ($FINAL_DONE) -- nothing to do."
elif [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starting from scratch: DiT backbone, CFM+ErgLoss, start conditioning, LM teacher DB"
    srun --unbuffered python flow_matching_runner_start.py $ARGS
fi
echo "end: $(date)"
