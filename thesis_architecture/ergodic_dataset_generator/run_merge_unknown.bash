#!/bin/bash
#SBATCH -J unk_merge
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:30:00
#SBATCH -p stud
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=4G
#SBATCH -c 1

# Fuehrt die acht Teil-DBs des Explorations-Array-Jobs (run_data_gen_unknown.bash,
# Job 161234) mit ergodic_dataset_improved.db zu ergodic_dataset_improved_final.db
# zusammen. Reiner I/O-Job, kein JAX/Torch noetig, ein Kern genuegt.
#
#   sbatch --dependency=afterok:161234 run_merge_unknown.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

echo "Start: $(date)"
python -u merge_unknown_db.py \
    --base ergodic_dataset_improved.db \
    --out ergodic_dataset_improved_final.db \
    ergodic_dataset_improved_unknown_part*.db
echo "Ende: $(date)"
