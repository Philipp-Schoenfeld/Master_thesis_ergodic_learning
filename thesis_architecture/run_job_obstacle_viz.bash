#!/bin/bash
#SBATCH -J obstacle_viz
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:45:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -c 2

# Inference-time obstacle guidance: verify the repulsion, then regenerate the
# holdout visualizations with and without guidance for each checkpoint.
#
# This is inference only - no training, no dataset build - so it asks for far
# less than the training jobs: 8G instead of 32G, 2 cores instead of 4, and
# 45 min instead of 24 h. The heaviest single item is the 3360x3777 figure that
# matplotlib renders per checkpoint.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

CHECKPOINTS=(
    "checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_10_23h52min_nxi25_D384_N128_C15_flip0.0_ep0400.pt"
    "checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_10_23h38min_nxi25_D384_N256_C75_flip0.0_ep0080.pt"
)

N_GEN=3
STEPS=100
OBSTACLE_WEIGHT=20.0
OBSTACLE_T_START=0.3

echo "=============================================================="
echo " Obstacle guidance - holdout visualization"
echo " Job $SLURM_JOB_ID on $(hostname)   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ── Guard: the repulsion must be correct before spending GPU time on it ───────
echo ""
echo "--- Verifying repulsion (gradients, chain rule, polish) ---"
if ! srun --unbuffered python -u test_obstacle_grad.py; then
    echo "[ABORT] Repulsion checks failed - not running the visualizations."
    exit 1
fi

# ── One visualization per checkpoint ─────────────────────────────────────────
FAILED=0
for CKPT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=============================================================="
    echo " Checkpoint: $(basename "$CKPT")"
    echo "=============================================================="

    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] Checkpoint not found: $CKPT"
        FAILED=1
        continue
    fi

    if srun --unbuffered python -u visualize_checkpoint.py \
            --checkpoint "$CKPT" \
            --obstacle \
            --n_gen "$N_GEN" \
            --steps "$STEPS" \
            --obstacle_weight "$OBSTACLE_WEIGHT" \
            --obstacle_t_start "$OBSTACLE_T_START"; then
        echo "[OK] $(basename "$CKPT")"
    else
        echo "[FAIL] $(basename "$CKPT")"
        FAILED=1
    fi
done

echo ""
echo "=============================================================="
echo " Done $(date '+%Y-%m-%d %H:%M:%S'). Images in:"
echo "   Trajectory_data_generator/viz_rerun/"
ls -lh Trajectory_data_generator/viz_rerun/*_obstacle_*.png 2>/dev/null | tail -5
echo "=============================================================="

exit $FAILED
