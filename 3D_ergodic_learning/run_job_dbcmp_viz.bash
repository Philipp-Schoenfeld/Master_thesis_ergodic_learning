#!/bin/bash
#SBATCH -J dbcmp_viz
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx&vram24gb'
#SBATCH --mem=16G
#SBATCH -c 2

# Re-evaluation of the DB-refinement comparison with trajectories dumped for the
# interactive HTML viewer (build_dbcmp_html.py). Four learned models:
# DiT-S refined DB, DiT-S old DB, DiT-B 750 ep, U-Net D512 750 ep.
#   pass A: holdout of the OLD DB    -> results/dbcmp_viz_old   (teacher = projected)
#   pass B: holdout of the REFINED DB -> results/dbcmp_viz_ref  (teacher = refined)
#
#   sbatch run_job_dbcmp_viz.bash

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

for PASS in "old ergodic_dataset_3d_lm.db" "ref ergodic_dataset_3d_lm_ref.db"; do
    set -- $PASS
    echo "=== pass $1 ($2)  $(date) ==="
    srun --unbuffered python eval_lm3d_compare.py --db3d "$2" --models "${MODELS[@]}" \
        --per_split 100 --n_gen 4 --steps 100 --n_examples 3 --dump_npz \
        --out_dir "results/dbcmp_viz_$1" || exit 1
done
echo "end: $(date)"
