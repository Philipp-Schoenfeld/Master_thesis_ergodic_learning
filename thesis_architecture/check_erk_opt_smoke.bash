#!/bin/bash
#SBATCH -J check_erk_opt
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:20:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=6G
#SBATCH -c 2

# Nur-Diagnose: der eingebaute Selbsttest fuer exploration_optimierung
# (Kaltstart-Tests + kleine Mission mit echtem Checkpoint, n_shapes=2/3,
# SVGD 30 Iterationen). Prueft, ob der Import-Fehler (trim_to_length fehlte
# in common/metrics.py, weil exploration/ auf dem Cluster veraltet war) durch
# das Nachziehen von exploration/, SE3_SVGD/ und src/ tatsaechlich behoben
# ist — bevor die grosse erk_opt-Kette (voll, 23h) wieder eingereiht wird.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

srun --unbuffered python -m exploration_optimierung.test_smoke
