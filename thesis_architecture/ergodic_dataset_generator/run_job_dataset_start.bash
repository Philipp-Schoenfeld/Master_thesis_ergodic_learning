#!/bin/bash
#SBATCH -J ds_start
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:30:00
#SBATCH -p stud
#SBATCH --mem=16G
#SBATCH -c 8

# Neue Datenbank mit zufaelligen Startpunkten auf der ganzen Flaeche.
#
# **Keine GPU angefordert.** Der SVGD-Loeser laeuft ueber JAX, und auf diesem
# Cluster ist kein CUDA-faehiges jaxlib installiert — JAX faellt ohnehin auf die
# CPU zurueck. Eine GPU zu belegen haette den Lauf keine Sekunde beschleunigt
# und haette einen der knappen Beschleuniger blockiert, auf die die beiden
# Trainingslaeufe warten.
#
# Zeitrechnung: lokal gemessen 4 bis 6 Sekunden je Form bei 700 Iterationen.
# 775 Formen ergeben rund 65 Minuten; reserviert sind 150.
#
# Gegenueber der bestehenden Datenbank aendert sich zweierlei:
#   --x0_mode ueberall   Startpunkt gleichverteilt statt immer unten links
#   --num_iters 700      hundert Iterationen mehr als bisher (600)
#
#   sbatch run_job_dataset_start.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

echo "Knoten:     $(hostname)"
echo "Kerne:      $SLURM_CPUS_PER_TASK"
echo "Startzeit:  $(date)"

srun --unbuffered python -u generate_dataset.py \
    --mode full \
    --x0_mode ueberall \
    --x0_margin 0.03 \
    --num_iters 700 \
    --tsteps 200 \
    --dt 0.05 \
    --seed 20260824 \
    --db ergodic_dataset_start.db

echo "Endzeit:    $(date)"
ls -la ergodic_dataset_start.db
