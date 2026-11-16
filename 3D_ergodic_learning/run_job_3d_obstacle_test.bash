#!/bin/bash
#SBATCH -J 3d_obs_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Test of the obstacle-conditioned 3D training (run_job_3d_obstacle.bash):
#   0. test_3d_port.py, test_3d_backbones.py, test_obstacle_cond.py (CPU)
#   1. U-Net and DiT, full DB, the exact loss/flag set of the main job but a
#      smaller net (U-Net D256, DiT D384/8/6) and EPOCHS epochs each: step time,
#      memory, the obstacle terms' magnitude next to cfm/erg/ori, the holdout
#      plot with spheres, and the final compliance check (hit rate with the
#      sphere vs. with the null token on the same sphere).
#   2. SIGTERM/resume of the chain logic is the one of run_job_3d_lm_test.bash
#      and unchanged, so it is not repeated here.
# Checkpoints are deleted at the end, plots stay in viz/.
#
#   sbatch run_job_3d_obstacle_test.bash            # EPOCHS=8
#   EPOCHS=3 sbatch run_job_3d_obstacle_test.bash

EPOCHS=${EPOCHS:-8}
DB3D=${DB3D:-ergodic_dataset_3d_lm.db}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  start: $(date)"

echo "=== 0: test suites ==="
for T in test_3d_port.py test_3d_backbones.py test_obstacle_cond.py; do
    python -u $T > test_out.$SLURM_JOB_ID.txt 2>&1
    RC=$?
    tail -2 test_out.$SLURM_JOB_ID.txt
    if [ $RC -ne 0 ]; then cat test_out.$SLURM_JOB_ID.txt; echo "FAIL: $T"; exit 1; fi
done
rm -f test_out.$SLURM_JOB_ID.txt

COMMON="--db3d $DB3D \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0 --start_inpaint --target_cps \
    --erg_on footprint --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 \
    --erg_t_power 2.0 --erg_pts 128 \
    --lambda_ori 0.012 --w_cfm_rot 0.5 --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --obstacle_cond --p_obstacle 0.5 --lambda_obs 10000 --lambda_surf 10000 \
    --surf_clearance 0.07 --obs_r_min 0.04 --obs_r_max 0.10 --obs_hit_frac 0.7 \
    --n_particles 512 --grid_res 64 --copies_per_char 1 --p_flip 0.0 --parts_on_gpu \
    --lr 1e-4 --mini_batch 256 --epochs $EPOCHS --save_every $EPOCHS \
    --viz_every $EPOCHS --n_holdout_viz 10 --keep_checkpoints 1 \
    --save_model checkpoints/obstest.pt --obs_eval_items 24"

echo "=== 1a: U-Net D256, $EPOCHS epochs ==="
srun --unbuffered python flow_matching_runner_particles.py $COMMON --D 256 --run_tag obstestunet
echo "exit code $?  $(date)"

echo "=== 1b: DiT D384/8/6, $EPOCHS epochs ==="
srun --unbuffered python flow_matching_runner_particles.py $COMMON \
    --backbone dit --D 384 --dit_depth 8 --dit_heads 6 --run_tag obstestdit
echo "exit code $?  $(date)"

ls -la checkpoints/obstest_*
python - $(ls -t checkpoints/obstest_*_ep*.pt) <<'PYEOF'
import sys, torch
for f in sys.argv[1:]:
    ck = torch.load(f, map_location='cpu', weights_only=True)
    sd = ck['model_state_dict']
    print(f.split('/')[-1][:60], {k: ck.get(k) for k in ('backbone', 'D', 'obstacle_cond', 'p_obstacle',
                                                          'lambda_obs', 'lambda_surf', 'epoch', 'loss')},
          'obstacle keys:', sum(k.startswith(('obstacle_emb', 'null_obstacle')) for k in sd))
PYEOF
rm -f checkpoints/obstest_*
echo "PASS  end: $(date)"
