#!/bin/bash
#SBATCH -J start_lm_dit_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Timing + preemption test of the DiT backbone (--backbone dit, flow_matching_cond_particles_dit.py)
# in exactly the configuration of run_job_start_lm_dit.bash (DiT-S: D384, 12 blocks, 6 heads;
# 256 particles, batch 256, 25 copies, Sinkhorn ErgLoss, --target_cps --start_inpaint, LM DB).
#
#   0. time_backbones.py: s/step of U-Net and DiT, with and without ErgLoss, same GPU
#   1. training starts; 2 min after its first tqdm line the training step gets SIGTERM via
#      `scancel --signal=TERM <step>` -- what Slurm sends on a scavenger preemption (a TERM to
#      srun itself is not the same: srun then kills the step) -> the runner must write an
#      emergency _ep####.pt and exit
#   2. a second call resumes from that checkpoint (the logic of run_job_start_lm_dit.bash)
#      and finishes epoch 2 -> _final.pt with backbone=dit, the _ep file is rotated away
#
# The checkpoint and plots are deleted at the end (test run, tag DITTEST).
#
#   sbatch run_job_start_lm_dit_test.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi 25 --D 384 --backbone dit --dit_depth 12 --dit_heads 6 \
      --n_particles 256 --copies_per_char 25 --p_flip 0.0 \
      --epochs 2 --mini_batch 256 --save_every 1 --viz_every 1 \
      --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --target_cps --start_inpaint --tag DITTEST --keep_checkpoints 1 \
      --db ergodic_dataset_lm_nxi25.db"
PATTERN="checkpoints/*_START_*_CPS_INPAINT_DIT-d12-h6_DITTEST_*"
ERRLOG="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
rm -f $PATTERN

echo "=== 0: s/step U-Net vs DiT ==="
python -u time_backbones.py --steps 30

echo "=== 1: train, SIGTERM to the step 2 min after training started ==="
srun --unbuffered python flow_matching_runner_start.py $ARGS &
PID=$!
for i in $(seq 1 120); do grep -q "Training:" "$ERRLOG" 2>/dev/null && break; sleep 10; done
echo "training started: $(date)"
sleep 120
STEP=$(squeue -h -s -j "$SLURM_JOB_ID" -o %i | grep -v -e batch -e extern | tail -1)
echo "sending SIGTERM to step $STEP: $(date)"
scancel --signal=TERM "$STEP"
wait $PID
echo "exit code $?  $(date)"
ls -la $PATTERN

LATEST=$(ls -t ${PATTERN}_ep*.pt 2>/dev/null | head -1)
if [ ! -f "$LATEST" ]; then echo "FAIL: no emergency checkpoint"; exit 1; fi

echo "=== 2: resume from $LATEST ==="
srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
echo "exit code $?  $(date)"
ls -la $PATTERN

FINAL=$(ls ${PATTERN}_final.pt 2>/dev/null | head -1)
if [ ! -f "$FINAL" ]; then echo "FAIL: no final checkpoint"; exit 1; fi
python - "$FINAL" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print({k: ck.get(k) for k in ['backbone', 'dit_depth', 'dit_heads', 'start_cond', 'start_inpaint', 'epoch']})
print(f"params: {sum(v.numel() for v in ck['model_state_dict'].values()) / 1e6:.1f}M")
EOF
rm -f $PATTERN Trajectory_data_generator/*_DIT-d12-h6_DITTEST_*
echo "PASS  end: $(date)"
