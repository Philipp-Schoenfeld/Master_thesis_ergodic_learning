#!/bin/bash
#SBATCH -J start_lm_lam
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# 1-day ablation of the ErgLoss multiplier on the teacher-study DB: same model, data and
# options as run_job_start_lm.bash (start-conditioned CFM + Sinkhorn ErgLoss, --target_cps
# --start_inpaint, from scratch), only --lambda_erg and the length differ.
#
# Multipliers from the 1300 run (job 181274): CFM ~0.44, raw ErgLoss ~7.0e-4 from epoch ~100
# on, i.e. CFM/Erg ~ 630 and the weighted ErgLoss share s = lam / (630 + lam) (1300 -> 67%).
# lam = 630 * s / (1 - s):  share 25% -> 210,  50% -> 630,  75% -> 1890.
# (The raw terms shift a little with lam, so the realised shares are approximate; the log
# prints cfm / erg per epoch -- the shares are recomputed from it afterwards.)
#
# Length: 180 epochs x 25 copies (666 steps/epoch) = 120k steps. The 1300 run needed 340-440
# s/epoch (incl. a holdout viz every 25 epochs; here every 50) -> 180 epochs = 17-22 h, inside
# the 24 h limit. A 600-epoch 1300 run is NOT comparable (cosine schedule 3.3x longer);
# a 1300 run with the same 180 epochs is the reference.
#
#   sbatch -J lam210 run_job_start_lm_lambda.bash 210 ERG25
#   sbatch -J lam630 run_job_start_lm_lambda.bash 630 ERG50
#   sbatch -J lam1890 run_job_start_lm_lambda.bash 1890 ERG75
#   (reference:  sbatch -J lam1300 run_job_start_lm_lambda.bash 1300 ERG67)

LAMBDA=${1:?lambda_erg}
TAG=${2:?tag, e.g. ERG25}
EPOCHS=180
COPIES=25

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi 25 --D 384 --n_particles 256 --copies_per_char ${COPIES} --p_flip 0.0 \
      --epochs ${EPOCHS} --mini_batch 256 --save_every 10 --viz_every 50 \
      --lambda_erg ${LAMBDA} --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --target_cps --start_inpaint --tag LAM${TAG} --keep_checkpoints 1 \
      --db ergodic_dataset_lm_nxi25.db --use_wandb"

FINAL_DONE=$(ls checkpoints/*_START_*_CPS_INPAINT_LAM${TAG}_*_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t checkpoints/*_START_*_CPS_INPAINT_LAM${TAG}_*_ep*.pt 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  lambda_erg=${LAMBDA} tag=${TAG}  start: $(date)"
if [ -f "$FINAL_DONE" ]; then
    echo "Run already finished ($FINAL_DONE) -- nothing to do."
elif [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    srun --unbuffered python flow_matching_runner_start.py $ARGS
fi
echo "end: $(date)"
