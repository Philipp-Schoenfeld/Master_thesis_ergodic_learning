#!/bin/bash
#SBATCH -J start_adam
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Start-conditioned CFM + ErgLoss (the L500 configuration of
# run_job_start_lang.bash: D384, 256 particles, batch 256, Sinkhorn ErgLoss
# w1300 blur 0.05 t^2), trained FROM SCRATCH (no --finetune_from) on the
# Adam-SVGD DB (ergodic_dataset_generator/run_job_dataset_adam_svgd.bash:
# 1172 base densities + 1172 #unk0 large-area variants, 3 start points each,
# linear init). COPIES 25 x 3 starts keeps the epoch size of the old 75 copies
# per density.
#
# Two opt-in changes for this data (run name gets _CPS_INPAINT):
#   --target_cps     targets are the solver's exact B-spline control points
#   --start_inpaint  control point 0 is held at the start along the whole flow
#
# NXI / EPOCHS / COPIES from the test jobs 181119/181120 (RTX A6000):
#   nxi 40 = waypoint solve projected onto 40 control points (see
#   run_job_dataset_adam_svgd.bash).
#   0.53 s/step at nxi 25 (140 s per 267-step epoch; L500 on a 3080 Ti: 0.54),
#   nxi 40 costs ~1.1x per step (local fwd+bwd ratio incl. Sinkhorn: 32 -> 1.07,
#   50 -> 1.15) -> ~0.59 s/step; the budget below assumes 0.61. 6816 train rows x 25 copies / 256 = 666 steps/epoch
#   -> ~410 s/epoch incl. a holdout viz (~70 s) every 25 epochs.
#   550 epochs = ~63 h + ~10 min startup (density grids): ~3 days with ~8 h
#   margin for a slower card. 550 x 666 = 366k steps (L500: 174k).
#   If the limit is hit anyway, the SIGTERM handler saves a checkpoint and the
#   pre-queued afterany follow-up job resumes it (see the logic below).
#
# Restart logic as in run_job_start_finetune_improved.bash: a finished run
# (only _final.pt left) makes a re-submitted job do nothing; an interrupted one
# (SIGTERM handler wrote its _ep####.pt) is resumed.
#
#   sbatch --dependency=afterok:<DS_JOBID> run_job_start_adam_svgd.bash

NXI=40
EPOCHS=550
COPIES=25

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi ${NXI} --D 384 --n_particles 256 --copies_per_char ${COPIES} --p_flip 0.0 \
      --epochs ${EPOCHS} --mini_batch 256 --save_every 10 --viz_every 25 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --target_cps --start_inpaint --tag ADAMSVGD --keep_checkpoints 1 \
      --db ergodic_dataset_adam_svgd_nxi${NXI}.db --use_wandb"

FINAL_DONE=$(ls checkpoints/*_START_*_CPS_INPAINT_ADAMSVGD_*_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t checkpoints/*_START_*_CPS_INPAINT_ADAMSVGD_*_ep*.pt 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
if [ -f "$FINAL_DONE" ]; then
    echo "Run already finished ($FINAL_DONE) -- nothing to do."
elif [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starting from scratch: CFM+ErgLoss, start conditioning, Adam-SVGD DB nxi=${NXI}"
    srun --unbuffered python flow_matching_runner_start.py $ARGS
fi
echo "end: $(date)"
