#!/bin/bash
#SBATCH -J start_lm_diff_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Smoke + timing test of --objective diffusion in the configuration of
# run_job_start_lm_diffusion.bash (U-Net D384, 256 particles, batch 256, Sinkhorn ErgLoss,
# --target_cps --start_inpaint, LM DB), but 3 epochs on 2 copies (~55 steps/epoch):
#   1. training starts; once epoch 1 is checkpointed the step gets SIGTERM (what Slurm sends on a
#      scavenger preemption) -> emergency _ep####.pt; a second call resumes it (checks the
#      objective guard on resume) and finishes -> _final.pt
#   2. the final checkpoint carries objective/diff_pred, loads through model_zoo.load_model
#      (default constructor, flag adopted on load) and samples finite curves with DDPM and DDIM
# s/epoch from the tqdm line x (666 / 55) gives the per-epoch time of the real run.
# Checkpoints and plots are deleted at the end (tag DIFFTEST).
#
#   sbatch run_job_start_lm_diffusion_test.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture

ARGS="--nxi 25 --D 384 --n_particles 256 --copies_per_char 2 --p_flip 0.0 \
      --epochs 3 --mini_batch 256 --save_every 1 --viz_every 0 \
      --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --objective diffusion --diff_pred v --diff_eta 1.0 \
      --target_cps --start_inpaint --tag DIFFTEST --keep_checkpoints 1 \
      --db ergodic_dataset_lm_nxi25.db"
PATTERN="checkpoints/*_START_*_CPS_INPAINT_DIFF-v_DIFFTEST_*"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
rm -f $PATTERN

echo "=== 1: train, SIGTERM to the step once epoch 1 is checkpointed ==="
srun --unbuffered python flow_matching_runner_start.py $ARGS &
PID=$!
for i in $(seq 1 180); do ls ${PATTERN}_ep0001.pt >/dev/null 2>&1 && break; sleep 10; done
STEP=$(squeue -h -s -j "$SLURM_JOB_ID" -o %i | grep -v -e batch -e extern | tail -1)
echo "sending SIGTERM to step $STEP: $(date)"
scancel --signal=TERM "$STEP"
wait $PID
echo "exit code $?  $(date)"
ls -la $PATTERN
LATEST=$(ls -t ${PATTERN}_ep*.pt 2>/dev/null | head -1)
if [ ! -f "$LATEST" ]; then echo "FAIL: no emergency checkpoint"; exit 1; fi

echo "=== 1b: resume from $LATEST ==="
srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
ls -la $PATTERN
FINAL=$(ls ${PATTERN}_final.pt 2>/dev/null | head -1)
if [ ! -f "$FINAL" ]; then echo "FAIL: no final checkpoint"; exit 1; fi

echo "=== 2: checkpoint metadata, model_zoo load, sampling ==="
python - "$FINAL" <<'EOF'
import sys, torch
from model_zoo import load_model
from flow_matching_cond_particles_start import generate_particle_trajectories
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print({k: ck.get(k) for k in ['objective', 'diff_pred', 'diff_schedule', 'diff_eta',
                              'start_cond', 'start_inpaint', 'lambda_erg', 'epoch']})
assert ck['objective'] == 'diffusion' and ck['diff_pred'] == 'v'
dev = 'cuda'
model, kind, meta = load_model(sys.argv[1], dev)
assert model.diffusion_pred == 'v' and model.start_inpaint, (model.diffusion_pred, model.start_inpaint)
parts = torch.rand(256, 3, device=dev)
start = torch.tensor([0.2, 0.3], device=dev)
for eta in (1.0, 0.0):
    cps, _ = generate_particle_trajectories(model, parts, num_samples=8, nxi=25, nd=2,
                                            steps=100, device=dev, start=start, eta=eta)
    assert torch.isfinite(cps).all() and torch.allclose(cps[:, 0], start)
    print(f"eta={eta}: cps range [{cps.min():.3f}, {cps.max():.3f}]  std {cps.std():.3f}")
print("model_zoo load + sampling OK")
EOF
RC=$?
rm -f $PATTERN Trajectory_data_generator/*_DIFF-v_DIFFTEST_*
if [ $RC -ne 0 ]; then echo "FAIL"; exit 1; fi
echo "PASS  end: $(date)"
