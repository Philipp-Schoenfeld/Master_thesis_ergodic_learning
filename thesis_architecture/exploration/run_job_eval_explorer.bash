#!/bin/bash
#SBATCH -J explorer_eval
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Auswertung der drei trainierten Explorer in den Varianten A-E.
#
# Wird mit einer Abhaengigkeit auf alle drei Trainings eingereiht und startet
# damit automatisch, sobald das letzte durch ist:
#   sbatch --dependency=afterok:<ID1>:<ID2>:<ID3> run_job_eval_explorer.bash
#
# `afterok` und nicht `afterany`: laeuft eines der Trainings in einen Fehler,
# soll die Auswertung nicht auf halbfertigen Checkpoints laufen und Zahlen
# erzeugen, die spaeter niemand mehr zuordnen kann. Bei einem Zeitlimit-Abbruch
# ist dagegen `--dependency=afterany` plus ein Folge-Training richtig.
#
# --auto nimmt die jeweils neuesten Checkpoints je Ziel aus checkpoints/.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

export MPLBACKEND=Agg

echo "=== Vorhandene Explorer-Checkpoints ==="
ls -1t checkpoints/explorer_*.pt 2>/dev/null | head -12 || echo "keine gefunden"
echo

# Kontrolle: das auf WAHREN Dichten trainierte Netz. Uebertraegt es sich ohne
# Weiteres auf Glaubensdichten, waere das ganze Training hier ueberfluessig —
# eine Aussage, die gemessen und nicht angenommen werden sollte.
CONTROL=$(ls -t ../checkpoints/*ERGLOSS-SINKHORN-w1300*_final.pt \
             ../checkpoints/*ERGLOSS-SINKHORN-w1300*_ep0500.pt 2>/dev/null | head -1)
CTRL_ARG=""
if [ -f "$CONTROL" ]; then
    echo "Kontrolle: $CONTROL"
    CTRL_ARG="--control_ckpt $CONTROL"
fi

srun --unbuffered python evaluate_trained.py \
    --auto $CTRL_ARG \
    --shapes 25 \
    --grid_res 48 \
    --n_prior 12 \
    --n_particles 192 \
    --segments 3 --rounds 3 \
    --target_length 4.0 \
    --budgets 0.4 0.7 1.0 \
    --anytime_points 12 \
    --grad_steps 200 \
    --viz_shapes 6 \
    --out_dir results/eval

echo
echo "=== Ergebnisse ==="
ls -la results/eval/
