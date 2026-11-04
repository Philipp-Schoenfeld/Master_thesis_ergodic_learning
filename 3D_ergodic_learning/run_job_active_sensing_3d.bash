#!/bin/bash
#SBATCH -J as3d
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# 3D active-sensing missions (active_sensing_3d.py), port of the 2D replanning
# missions: RBF-GP belief over the surface particles, UCB target (kappa 1.0 ->
# 0.05 over 5 rounds), 30 candidates per round refined for 150 Adam iterations
# with the training objective, best candidate driven completely, sensor reads
# the particles near the beam footprint. 27 holdout missions (9 per split of the
# LM DB), conditions none_known and ten_samples, five initialisers: four
# start-conditioned networks and linear rays. A tiny smoke pass runs first.
# Resumable: finished (method, condition) cells are skipped on re-submission.
#
#   sbatch run_job_active_sensing_3d.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning
export MPLBACKEND=Agg

REF=$(ls -t checkpoints/dbcmp_*dbcmpref_*_ep0150.pt | head -1)
OLD=$(ls -t checkpoints/dbcmp_*dbcmpold_*_ep0150.pt | head -1)
BIG=$(ls checkpoints/lm3d_*lm630dit_*_ep0750.pt | head -1)
UNET=$(ls checkpoints/lm3d_*lm630unet_*_ep0750.pt | head -1)
for f in "$REF" "$OLD" "$BIG" "$UNET"; do [ -f "$f" ] || { echo "checkpoint missing: $f"; exit 1; }; done
MODELS=("DiT-S, refined DB=$REF" "DiT-S, old DB=$OLD" "DiT-B 750 ep, old DB=$BIG" "U-Net D512 750 ep, old DB=$UNET")
METHODS=("DiT-S, refined DB" "DiT-S, old DB" "DiT-B 750 ep, old DB" "U-Net D512 750 ep, old DB" linear)

python -u test_active_sensing_3d.py > /dev/null 2>&1 || { echo "unit tests failed"; exit 1; }

echo "=== smoke pass $(date) ==="
rm -rf results/active_sensing_3d_smoke
srun --unbuffered python -u active_sensing_3d.py --models "${MODELS[@]}" --methods "${METHODS[@]}" \
    --conditions none_known --per_split 1 --rounds 2 --n_cand 4 --steps 10 --refine_iters 5 \
    --out_dir results/active_sensing_3d_smoke || exit 1

echo "=== full pass $(date) ==="
srun --unbuffered python -u active_sensing_3d.py --models "${MODELS[@]}" --methods "${METHODS[@]}" \
    --per_split 9 --rounds 5 --n_cand 30 --steps 100 --refine_iters 150 \
    --out_dir results/active_sensing_3d || exit 1
srun --unbuffered python -u plot_active_sensing_3d.py --out_dir results/active_sensing_3d
echo "end: $(date)"
