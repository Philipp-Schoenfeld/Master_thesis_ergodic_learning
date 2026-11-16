#!/bin/bash
#SBATCH -J 3d_obs630
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# 3D obstacle-conditioned training: run_job_3d_lm.bash plus
#   --obstacle_cond  one sphere (centre, radius) or none per training sample,
#                    FiLM-conditioned (zero-init), "none" = learned null token
#   obstacle loss    on the endpoint estimate x1_hat, the same curve the ErgLoss
#                    scores: sphere penetration (only where a sphere exists) and
#                    surface clearance (every sample, the surface is an obstacle)
#   teacher repair   colliding stored paths are pushed out of the sphere so the
#                    CFM target agrees with the obstacle loss
# Everything else (DB, losses, schedule, 750 epochs) is the lambda-630 LM run.
#
#   sbatch -J 3d_obs630_unet run_job_3d_obstacle.bash unet
#   sbatch -J 3d_obs630_dit  run_job_3d_obstacle.bash dit
#   next link: sbatch -J ... --dependency=afterany:<JOBID> run_job_3d_obstacle.bash <same backbone>
#
# A link resumes from the latest _ep####.pt of its run; a link that finds the run
# finished exits before starting Python. Timing/memory as in run_job_3d_lm.bash
# (U-Net D512 ~367 s/epoch, DiT-B ~186 s/epoch; the obstacle terms add a few %).
#
# Fine-tune from an existing checkpoint instead of training from scratch:
#   INIT=checkpoints/lm3d_flow3d_particle_ergodic_lm630dit_..._ep0750.pt \
#       sbatch -J 3d_obs_ft_dit run_job_3d_obstacle.bash dit
# Only the weights are loaded (--init_model, fresh optimizer); the new obstacle
# modules start at zero, so epoch 0 reproduces the old net exactly. Schedule:
# FT_EPOCHS (default 150) epochs, 10 warm-up epochs to FT_LR (default 2e-5),
# cosine down to 1e-6. Chain links resume from the fine-tune's own checkpoint.
#
# Optional DB override for a different teacher DB (e.g. the refined one):
#   DB3D=ergodic_dataset_3d_lm_ref.db sbatch ... run_job_3d_obstacle.bash dit

BACKBONE=${1:?backbone: unet | dit}
DB3D=${DB3D:-ergodic_dataset_3d_lm.db}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

case "$BACKBONE" in
    unet) MODEL="--D 512";                                           EPOCHS=750 ;;
    dit)  MODEL="--backbone dit --D 768 --dit_depth 12 --dit_heads 12"; EPOCHS=750 ;;
    *) echo "unknown backbone $BACKBONE"; exit 1 ;;
esac
INIT=${INIT:-}
if [ -n "$INIT" ]; then
    [ -f "$INIT" ] || { echo "INIT checkpoint $INIT not found"; exit 1; }
    EPOCHS=${FT_EPOCHS:-150}
    FT_ARGS="--lr ${FT_LR:-2e-5} --warmup_epochs 10 --lr_min 1e-6"
    FT_TAG="ft"
else
    FT_ARGS="--lr 1e-4"
    FT_TAG=""
fi
SAVE_EVERY=$(( EPOCHS / 40 > 0 ? EPOCHS / 40 : 1 ))
VIZ_EVERY=$(( EPOCHS / 10 > 0 ? EPOCHS / 10 : 1 ))
RUN_TAG="obs630${BACKBONE}${FT_TAG}"

PAT="checkpoints/obs3d_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "backbone=$BACKBONE  db=$DB3D  epochs=$EPOCHS  save_every=$SAVE_EVERY  viz_every=$VIZ_EVERY  start: $(date)"

if [ -f "$LATEST" ]; then
    DONE=$(echo "$LATEST" | sed -E 's/.*_ep0*([0-9]+)\.pt$/\1/')
    if [ "$DONE" -ge "$EPOCHS" ]; then
        echo "Run already finished ($LATEST) -- nothing to do."
        exit 0
    fi
    echo "Resuming from: $LATEST"
    START_ARG="--resume $LATEST"
else
    echo "=== Cold start: test suites ==="
    python -u test_3d_backbones.py > /dev/null 2>&1 \
        || { echo "test_3d_backbones.py failed -- abort before training."; exit 1; }
    python -u test_obstacle_cond.py > /dev/null 2>&1 \
        || { echo "test_obstacle_cond.py failed -- abort before training."; exit 1; }
    echo "Starting from scratch"
    START_ARG=""
    [ -n "$INIT" ] && START_ARG="--init_model $INIT"
fi

srun --unbuffered python flow_matching_runner_particles.py $START_ARG $MODEL \
    --db3d "$DB3D" \
    --run_tag "$RUN_TAG" --save_model checkpoints/obs3d.pt \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0 --start_inpaint --target_cps \
    --erg_on footprint --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 \
    --erg_t_power 2.0 --erg_pts 128 \
    --lambda_ori 0.012 --w_cfm_rot 0.5 --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --obstacle_cond --p_obstacle 0.5 --lambda_obs 10000 --lambda_surf 10000 \
    --surf_clearance 0.07 --obs_r_min 0.04 --obs_r_max 0.10 --obs_hit_frac 0.7 \
    --n_particles 512 --grid_res 64 --copies_per_char 1 --p_flip 0.0 --parts_on_gpu \
    $FT_ARGS --mini_batch 256 --epochs "$EPOCHS" \
    --save_every "$SAVE_EVERY" --viz_every "$VIZ_EVERY" --n_holdout_viz 10 \
    --keep_checkpoints 1 --use_wandb --wandb_project flow3d-surfaces
echo "end: $(date)"
