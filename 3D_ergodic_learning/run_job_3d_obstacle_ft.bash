#!/bin/bash
#SBATCH -J 3d_obs_ft
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Fine-tune the lambda-630 LM checkpoints (run_job_3d_lm.bash) for obstacle
# conditioning. Each backbone is its own job and is sized to finish in ~22 h.
#
#   sbatch -J 3d_obs_ft_unet run_job_3d_obstacle_ft.bash unet
#   sbatch -J 3d_obs_ft_dit  run_job_3d_obstacle_ft.bash dit
#
# Start point: the newest lm3d_flow3d_particle_ergodic_lm630<backbone>_*_ep*.pt in
# checkpoints/ (override with INIT=...). Only the weights are loaded
# (--init_model); the obstacle modules start at zero, so epoch 0 reproduces the
# old net exactly. Same DB, losses and flags as the source run, plus
# --obstacle_cond, the obstacle loss (sphere + surface clearance) and the
# teacher repair (see run_job_3d_obstacle.bash).
#
# Epoch budget (s/epoch measured for the source run on an A5000, batch 256,
# 328 steps/epoch; +~5 % for sampling/repair/obstacle loss; the 24 GB card set
# in -C contains only cards at least as fast as the A5000):
#   unet D512  ~370 s -> ~390 s/epoch  x 200 epochs = 21.7 h
#   dit  D768  ~187 s -> ~196 s/epoch  x 400 epochs = 21.8 h
# plus DB load, ~8 checkpoints and the final compliance check: ~22 h in total.
# The 1-day limit is only a safety margin: at the limit SIGTERM -> emergency
# checkpoint, and a chain link (sbatch --dependency=afterany:<JOBID> ... same
# backbone) resumes it. A link that finds the run finished exits at once.
#
# Schedule: 10 warm-up epochs to FT_LR, cosine to 1e-6, fresh optimizer.
# Overridable: DB3D, INIT, FT_LR (5e-5), FT_EPOCHS, LAMBDA_OBS (5e4),
# LAMBDA_SURF (1e4), SURF_CLEARANCE (0.05).

BACKBONE=${1:?backbone: unet | dit}
DB3D=${DB3D:-ergodic_dataset_3d_lm.db}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

case "$BACKBONE" in
    unet) MODEL="--D 512";                                              EPOCHS=${FT_EPOCHS:-200} ;;
    dit)  MODEL="--backbone dit --D 768 --dit_depth 12 --dit_heads 12"; EPOCHS=${FT_EPOCHS:-400} ;;
    *) echo "unknown backbone $BACKBONE"; exit 1 ;;
esac
SAVE_EVERY=$(( EPOCHS / 40 > 0 ? EPOCHS / 40 : 1 ))
VIZ_EVERY=$(( EPOCHS / 10 > 0 ? EPOCHS / 10 : 1 ))
RUN_TAG="obs630${BACKBONE}ft"

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
    INIT=${INIT:-$(ls -t checkpoints/lm3d_flow3d_particle_ergodic_lm630${BACKBONE}_*_ep*.pt 2>/dev/null | head -1)}
    if [ ! -f "$INIT" ]; then
        echo "No source checkpoint (checkpoints/lm3d_..._lm630${BACKBONE}_*_ep*.pt) -- set INIT=..."
        exit 1
    fi
    echo "Fine-tuning from: $INIT"
    echo "=== Cold start: test suites ==="
    python -u test_3d_backbones.py > /dev/null 2>&1 \
        || { echo "test_3d_backbones.py failed -- abort before training."; exit 1; }
    python -u test_obstacle_cond.py > /dev/null 2>&1 \
        || { echo "test_obstacle_cond.py failed -- abort before training."; exit 1; }
    START_ARG="--init_model $INIT"
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
    --obstacle_cond --p_obstacle 0.5 --lambda_obs "${LAMBDA_OBS:-50000}" \
    --lambda_surf "${LAMBDA_SURF:-10000}" --surf_clearance "${SURF_CLEARANCE:-0.05}" \
    --obs_r_min 0.04 --obs_r_max 0.10 --obs_hit_frac 0.7 \
    --n_particles 512 --grid_res 64 --copies_per_char 1 --p_flip 0.0 --parts_on_gpu \
    --lr "${FT_LR:-5e-5}" --warmup_epochs 10 --lr_min 1e-6 \
    --mini_batch 256 --epochs "$EPOCHS" \
    --save_every "$SAVE_EVERY" --viz_every "$VIZ_EVERY" --n_holdout_viz 10 \
    --keep_checkpoints 1 --use_wandb --wandb_project flow3d-surfaces
echo "end: $(date)"
