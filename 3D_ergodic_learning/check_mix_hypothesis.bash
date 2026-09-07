#!/bin/bash
#SBATCH -J check_mix
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:15:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=6G
#SBATCH -c 2

# Nur-Diagnose (nicht Teil der eigentlichen Job-Kette): prueft, ob die
# grenzwertige Abweichung der Warmstart-Pruefung (9.5-12% bei ~5-12%
# Toleranzband) daran liegt, dass pruefe_warmstart.py gleichverteilt
# zieht, waehrend flaechen200 mit --mix ebene=0.25 trainiert wurde.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

srun --unbuffered python pruefe_warmstart_mix.py
