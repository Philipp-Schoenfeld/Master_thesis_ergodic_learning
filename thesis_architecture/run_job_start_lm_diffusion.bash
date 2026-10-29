#!/bin/bash
#SBATCH -J start_lm_diff
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1-06:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120
#SBATCH --requeue
#SBATCH --open-mode=append

# Diffusion instead of flow matching on the SAME U-Net (--objective diffusion,
# diffusion_cond_particles_start.py: cosine-VP schedule in flow time, v-prediction by default,
# DDPM sampler eta=1 for the holdout plots). Everything else is the schedule of the 1-day lambda
# ablation (run_job_start_lm_lambda.bash) and the DiT run (run_job_start_lm_dit.bash):
# 180 epochs x 25 copies = 120k steps, cosine lr 1e-4, batch 256, D384, 256 particles,
# Sinkhorn ErgLoss blur 0.05 t^2, --target_cps --start_inpaint, LM teacher DB, from scratch.
# Only the training objective differs, so e.g.
#   diffusion lambda 630  <->  U-Net CFM job lam630 (ERG50)  <->  DiT CFM job dit630
#
#   sbatch -J diff630 run_job_start_lm_diffusion.bash 630 ERG50          # v-prediction
#   sbatch -J diff0   run_job_start_lm_diffusion.bash 0 ERG0             # pure diffusion, no ErgLoss (MPD-like)
#   sbatch -J diff630eps run_job_start_lm_diffusion.bash 630 ERG50 eps   # eps-prediction
#   add --qos=scavenger to sbatch beyond the 3-GPU limit (preemption -> requeue -> resume)
#
# lambda_erg is the same absolute weight on the same quantity (ErgLoss of the predicted endpoint);
# the denoising loss has a different scale than the CFM loss, so the realised ErgLoss share differs
# from ERG50 -- it is recomputed from the logged diff / erg per epoch, as in the lambda ablation.
#
# Time: same network and step count as lam630 (job 183458: 17.6 h incl. viz); the diffusion loss
# costs the same as the CFM loss -> ~18-22 h on rtx cards, -t 30 h leaves room.
# Restart logic: a finished run (_final.pt) makes a restart a no-op; an interrupted one resumes.

LAMBDA=${1:?lambda_erg}
TAG=${2:?tag, e.g. ERG50}
PRED=${3:-v}
EPOCHS=180
COPIES=25

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi 25 --D 384 --n_particles 256 --copies_per_char ${COPIES} --p_flip 0.0 \
      --epochs ${EPOCHS} --mini_batch 256 --save_every 10 --viz_every 50 \
      --lambda_erg ${LAMBDA} --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --objective diffusion --diff_pred ${PRED} --diff_eta 1.0 \
      --target_cps --start_inpaint --tag LAM${TAG} --keep_checkpoints 1 \
      --db ergodic_dataset_lm_nxi25.db --use_wandb"
PATTERN="checkpoints/*_START_*_CPS_INPAINT_DIFF-${PRED}_LAM${TAG}_*"

FINAL_DONE=$(ls ${PATTERN}_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t ${PATTERN}_ep*.pt 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  lambda_erg=${LAMBDA} tag=${TAG} pred=${PRED}  restart=${SLURM_RESTART_COUNT:-0}  start: $(date)"
if [ -f "$FINAL_DONE" ]; then
    echo "Run already finished ($FINAL_DONE) -- nothing to do."
elif [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starting from scratch: diffusion (${PRED}-pred) + ErgLoss, U-Net, start conditioning, LM teacher DB"
    srun --unbuffered python flow_matching_runner_start.py $ARGS
fi
echo "end: $(date)"
