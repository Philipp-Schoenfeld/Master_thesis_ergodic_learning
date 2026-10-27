#!/bin/bash
#SBATCH -J start_lm
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
# teacher-study DB (ergodic_dataset_generator/run_job_dataset_lm*.bash:
# 1172 base densities + 1172 #unk0 large-area variants, 3 start points each,
# LM teacher lm_K16_cont_o2_w15 projected onto 25 control points).
#
# Two opt-in changes for this data (run name gets _CPS_INPAINT):
#   --target_cps     targets are the DB's exact B-spline control points
#   --start_inpaint  control point 0 is held at the start along the whole flow
#
# Budget (timing job 181120, RTX A6000, exactly this config at nxi 25 with the
# same 6816 training rows): 0.53 s/step (L500 on a 3080 Ti: 0.54).
# 6816 x 25 copies / 256 = 666 steps/epoch -> ~355 s/epoch incl. a holdout viz
# (~70 s) every 25 epochs. 600 epochs = ~59 h (~67 h at 0.6 s/step on a slower
# card) + ~10 min startup -> inside 3 days. 600 x 666 = 400k steps (L500: 174k).
# If the limit is hit anyway, the SIGTERM handler saves a checkpoint and the
# pre-queued afterany follow-up job resumes it.
#
# Restart logic: a finished run (only _final.pt left) makes a re-submitted job
# do nothing; an interrupted one (_ep####.pt) is resumed.
#
#   sbatch --dependency=afterok:<MERGE_JOBID> run_job_start_lm.bash

NXI=25
EPOCHS=600
COPIES=25

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi ${NXI} --D 384 --n_particles 256 --copies_per_char ${COPIES} --p_flip 0.0 \
      --epochs ${EPOCHS} --mini_batch 256 --save_every 10 --viz_every 25 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --target_cps --start_inpaint --tag LMTEACHER --keep_checkpoints 1 \
      --db ergodic_dataset_lm_nxi${NXI}.db --use_wandb"

FINAL_DONE=$(ls checkpoints/*_START_*_CPS_INPAINT_LMTEACHER_*_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t checkpoints/*_START_*_CPS_INPAINT_LMTEACHER_*_ep*.pt 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
if [ -f "$FINAL_DONE" ]; then
    echo "Run already finished ($FINAL_DONE) -- nothing to do."
elif [ -f "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starting from scratch: CFM+ErgLoss, start conditioning, LM teacher DB nxi=${NXI}"
    srun --unbuffered python flow_matching_runner_start.py $ARGS
fi
echo "end: $(date)"
