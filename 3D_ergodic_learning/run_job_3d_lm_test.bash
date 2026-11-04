#!/bin/bash
#SBATCH -J 3d_lm_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Test run for the 3D lambda-630 trainings on the LM DB (run_job_3d_lm.bash):
#
#   0. test_3d_port.py + test_3d_backbones.py (CPU, regression + new parts)
#   1. bench_backbones_3d.py: s/step and peak memory of the candidate sizes at
#      batch 256 with the full loss (CFM + Sinkhorn footprint ErgLoss + orientation)
#      -> sets the epoch budget of the full runs
#   2. U-Net, full DB, exactly the arguments of run_job_3d_lm.bash: training
#      starts, 3 min later the step gets SIGTERM (what Slurm sends at the time
#      limit) -> emergency _ep####.pt; then the chain logic resumes from it and
#      finishes epoch 2 on a reduced set (--n_train_shapes 5, a resume check, not
#      a training)
#   3. DiT, full DB, one full epoch -> step time, memory, holdout plot, checkpoint
#
# Checkpoints are deleted at the end (tags test3dunet / test3ddit), plots stay in viz/.
#
#   sbatch run_job_3d_lm_test.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

ERRLOG="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  start: $(date)"

COMMON="--db3d ergodic_dataset_3d_lm.db \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0 --start_inpaint --target_cps \
    --erg_on footprint --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 \
    --erg_t_power 2.0 --erg_pts 128 \
    --lambda_ori 0.012 --w_cfm_rot 0.5 --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --n_particles 512 --grid_res 64 --copies_per_char 1 --p_flip 0.0 \
    --lr 1e-4 --mini_batch 256 --save_every 1 --viz_every 1 --n_holdout_viz 10 \
    --keep_checkpoints 1 --save_model checkpoints/lm3d.pt"
UNET="--D 384"
DIT="--backbone dit --D 512 --dit_depth 16 --dit_heads 8"

echo "=== 0: test suites ==="
for T in test_3d_port.py test_3d_backbones.py; do
    python -u $T > test_out.$SLURM_JOB_ID.txt 2>&1
    RC=$?
    tail -3 test_out.$SLURM_JOB_ID.txt
    if [ $RC -ne 0 ]; then cat test_out.$SLURM_JOB_ID.txt; echo "FAIL: $T"; exit 1; fi
done
rm -f test_out.$SLURM_JOB_ID.txt

echo "=== 1: s/step and memory of the candidate sizes (batch 256) ==="
python -u bench_backbones_3d.py --db3d ergodic_dataset_3d_lm.db --mini_batch 256 --steps 30

echo "=== 2a: U-Net, full DB, SIGTERM 3 min after training started ==="
PAT_U="checkpoints/lm3d_flow3d_particle_ergodic_test3dunet_*"
rm -f $PAT_U
srun --unbuffered python flow_matching_runner_particles.py $COMMON $UNET \
    --epochs 2 --run_tag test3dunet &
PID=$!
for i in $(seq 1 120); do grep -q "Training:" "$ERRLOG" 2>/dev/null && break; sleep 10; done
echo "training started: $(date)"
sleep 180
STEP=$(squeue -h -s -j "$SLURM_JOB_ID" -o %i | grep -v -e batch -e extern | tail -1)
echo "sending SIGTERM to step $STEP: $(date)"
scancel --signal=TERM "$STEP"
wait $PID
echo "exit code $?  $(date)"
ls -la $PAT_U
LATEST=$(ls -t ${PAT_U}_ep*.pt 2>/dev/null | head -1)
if [ ! -f "$LATEST" ]; then echo "FAIL: no emergency checkpoint"; exit 1; fi

echo "=== 2b: resume from $LATEST (reduced set) ==="
srun --unbuffered python flow_matching_runner_particles.py $COMMON $UNET \
    --epochs 2 --run_tag test3dunet --n_train_shapes 5 --resume "$LATEST"
echo "exit code $?  $(date)"
ls -la $PAT_U
FINAL=$(ls -t ${PAT_U}_ep0002.pt 2>/dev/null | head -1)
if [ ! -f "$FINAL" ]; then echo "FAIL: no epoch-2 checkpoint after resume"; exit 1; fi

echo "=== 3: DiT, full DB, one epoch ==="
PAT_D="checkpoints/lm3d_flow3d_particle_ergodic_test3ddit_*"
rm -f $PAT_D
srun --unbuffered python flow_matching_runner_particles.py $COMMON $DIT \
    --epochs 1 --run_tag test3ddit
echo "exit code $?  $(date)"
ls -la $PAT_D

python - $FINAL $(ls -t ${PAT_D}_ep*.pt | head -1) <<'EOF'
import sys, torch
for f in sys.argv[1:]:
    ck = torch.load(f, map_location='cpu', weights_only=True)
    print(f.split('/')[-1])
    print('  ', {k: ck.get(k) for k in ['backbone', 'dit_depth', 'dit_heads', 'D', 'start_cond',
                                         'start_inpaint', 'target_cps', 'erg_metric',
                                         'lambda_erg', 'epoch', 'loss']})
    print(f"   params: {sum(v.numel() for v in ck['model_state_dict'].values()) / 1e6:.1f}M"
          f"  start_inpaint_flag in state dict: {'start_inpaint_flag' in ck['model_state_dict']}")
EOF
rm -f $PAT_U $PAT_D
echo "PASS  end: $(date)"
