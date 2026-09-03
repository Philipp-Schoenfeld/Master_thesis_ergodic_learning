#!/bin/bash
#SBATCH -J erk_opt
#SBATCH -p stud
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=SIGTERM@120
#SBATCH -o logs/erk_opt-%j.out
#SBATCH -e logs/erk_opt-%j.err
##SBATCH -C 'rtx3090'      # auf der stud-Partition verboten, bleibt auskommentiert

# Suche nach der besten Betriebseinstellung der Laengeneinheit-Mission.
#
# Der Lauf ist fortsetzbar: jede fertige Auswertung liegt als JSON unter
# exploration_optimierung/results/cache/. Dieses Skript rechnet 23 Stunden und
# beendet sich dann selbst sauber (--max_stunden 23), also *vor* dem harten
# 24-h-Limit, damit alle Tabellen noch geschrieben werden. Der naechste Job der
# Kette nimmt die Arbeitsliste dort wieder auf, wo dieser aufgehoert hat.
#
# 16 CPU-Kerne sind kein Luxus: die SVGD-Verfeinerung laeuft in Arbeitsprozessen
# und ist bei den hohen Iterationszahlen der teuerste Teil. Mit den ueblichen
# -c 4 waere der Cluster hier langsamer als ein Arbeitsplatzrechner mit zehn
# freien Kernen.

set -e
cd ~/Master_thesis/thesis_architecture/
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

echo "Job $SLURM_JOB_ID auf $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --unbuffered python -m exploration_optimierung.optimize \
    --mode beide \
    --search voll \
    --param_points 16 \
    --seeds 2 \
    --workers 14 \
    --max_stunden 23

echo "Ende $(date)"
