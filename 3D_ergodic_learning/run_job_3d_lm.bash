#!/bin/bash
#SBATCH -J 3d_lm630
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# 3D SE(3) CFM + Sinkhorn ErgLoss at lambda 630 on the LM teacher DB projected to
# 200 surfaces (ergodic_dataset_3d_lm.db, 93 219 entries: 83 789 train,
# val_form / val_flaeche / val_beides), trained FROM SCRATCH, one backbone per job:
#
#   unet  1D U-Net, enlarged for 3D from D384 (87.7M, the 2D lambda-630 U-Net,
#         job lam630) to D512 (155.8M): 3D particle tokenizer, rotation head
#         (pos 3 + rot6 6)
#   dit   DiT with adaLN-Zero + particle cross-attention, enlarged for 3D from
#         DiT-S (D384/12 blocks/6 heads, 40M) to DiT-B (D768/12/12, 159.4M),
#         parameter parity with the U-Net
#
# Why larger than in 2D: the 3D state has 9 channels instead of 2, the
# condition is a 512-particle cloud on 200 different surfaces, and the DB has
# 12x the training entries of the 2D LM DB (83 789 vs 6 816).
#
# What is taken over from the 2D lambda runs (run_job_start_lm_lambda.bash,
# run_job_start_lm_dit.bash): Sinkhorn ErgLoss (blur 0.05, t^2) at lambda 630,
# --target_cps (control points least-squares fitted to the stored path instead
# of picked from it), --start_inpaint (control point 0 held at the start), batch
# 256, lr 1e-4 cosine to 1e-5, CFG p_drop 0.1, no flips.
# What is taken over from the last 3D run (run_job_3d_no_offset_200.bash):
# --rot_full, orientation head with lambda_ori 0.012 (w_point 0.1, w_standoff
# 300, standoff 0.12 +- 0.03, w_angsmooth 2), w_cfm_rot 0.5, coverage scored at
# the beam footprint (--erg_on footprint), 512 particles, uniform draw over the
# surface groups. The start is never dropped (--p_drop_start 0): with
# inpainting it is part of the state, as in 2D.
#
#   sbatch -J 3d_lm630_unet run_job_3d_lm.bash unet
#   sbatch -J 3d_lm630_dit  run_job_3d_lm.bash dit
#   next link: sbatch -J ... --dependency=afterany:<JOBID> run_job_3d_lm.bash <same backbone>
#
# A link resumes from the latest _ep####.pt of its run; a link that finds the run
# finished (epoch == EPOCHS) exits before starting Python.
#
# Epochs and timing (test job 186004, RTX A5000, batch 256, full loss):
#   bench s/step: U-Net D384 0.748, D512 1.017; DiT D384/12 0.255, D512/16 0.402,
#   D768/12 0.515. A full DiT-D512/16 epoch incl. checkpoint + holdout plot took
#   182 s against 132 s of pure steps. 83 789 / 256 = 328 steps/epoch.
#   Same schedule for both backbones (as in the 2D comparison): 750 epochs = 246k
#   steps (2x the 120k of the 2D lambda runs). U-Net D512 ~367 s/epoch -> ~77 h
#   (Mon afternoon/evening on an A5000; +25 % overhead still before Tue 02:00),
#   DiT-B ~186 s/epoch -> ~39 h. The U-Net exceeds the 3-day limit: at the limit
#   SIGTERM (--signal) -> emergency _ep####.pt -> the chained link resumes.
#   Peak memory: U-Net D512 11.3 GiB, DiT-B 14.0 GiB (+0.7 GB particle stack with
#   --parts_on_gpu) -> 24 GB cards only; peak host RAM 4.7 GB -> --mem=8G.

BACKBONE=${1:?backbone: unet | dit}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

case "$BACKBONE" in
    unet) MODEL="--D 512";                                           EPOCHS=750 ;;
    dit)  MODEL="--backbone dit --D 768 --dit_depth 12 --dit_heads 12"; EPOCHS=750 ;;
    *) echo "unknown backbone $BACKBONE"; exit 1 ;;
esac
SAVE_EVERY=$(( EPOCHS / 40 > 0 ? EPOCHS / 40 : 1 ))
VIZ_EVERY=$(( EPOCHS / 10 > 0 ? EPOCHS / 10 : 1 ))
RUN_TAG="lm630${BACKBONE}"

PAT="checkpoints/lm3d_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "backbone=$BACKBONE  epochs=$EPOCHS  save_every=$SAVE_EVERY  viz_every=$VIZ_EVERY  start: $(date)"

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
    echo "Starting from scratch"
    START_ARG=""
fi

srun --unbuffered python flow_matching_runner_particles.py $START_ARG $MODEL \
    --db3d ergodic_dataset_3d_lm.db \
    --run_tag "$RUN_TAG" --save_model checkpoints/lm3d.pt \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0 --start_inpaint --target_cps \
    --erg_on footprint --lambda_erg 630 --erg_metric sinkhorn --sinkhorn_blur 0.05 \
    --erg_t_power 2.0 --erg_pts 128 \
    --lambda_ori 0.012 --w_cfm_rot 0.5 --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --n_particles 512 --grid_res 64 --copies_per_char 1 --p_flip 0.0 --parts_on_gpu \
    --lr 1e-4 --mini_batch 256 --epochs "$EPOCHS" \
    --save_every "$SAVE_EVERY" --viz_every "$VIZ_EVERY" --n_holdout_viz 10 \
    --keep_checkpoints 1 --use_wandb --wandb_project flow3d-surfaces
echo "end: $(date)"
