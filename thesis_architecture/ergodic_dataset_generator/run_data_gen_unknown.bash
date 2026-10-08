#!/bin/bash
#SBATCH -J unk_ds
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -p stud
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=3G
#SBATCH -c 2
#SBATCH --array=0-7
#SBATCH --signal=SIGTERM@120

# Explorations-Datensatz: je Zieldichte aus ergodic_dataset_improved.db zwei
# zusaetzliche Varianten mit einem zufaelligen "unbekannten" Flaechenbereich
# (siehe unknown_region.py), danach dieselbe Loeser-Pipeline (heuristische
# Initialisierung + SVGD, hier mit ergodizitaetsbasiertem statt
# laengenbasiertem Konvergenzabbruch).
#
# Ressourcen uebernommen von run_data_gen.bash (dortige sacct-Messung: der
# Loeser rechnet ueber JAX einzeln auf einem Kern, egal wie viele angefordert
# werden -- mehr Kerne/RAM wuerden nur andere Nutzer und die eigenen
# GPU-Trainings blockieren, die auf das Kontingent cpu=50/gres_gpu=3/mem=150G
# warten). BEWUSST OHNE GPU.
#
# Aufteilung nach Form-INDEX, nicht nach Form-Variante: jede der acht
# Aufgaben bearbeitet 1/8 der 1172 Basis-Zieldichten und erzeugt dafuer BEIDE
# Varianten -- so bleiben pro Aufgabe alle Varianten einer Form beisammen,
# falls das je fuer Debugging relevant wird.
#
# Jede Aufgabe schreibt in eine EIGENE Datenbank (gleicher Grund wie bei
# run_data_gen.bash: gleichzeitige Schreibzugriffe auf dieselbe SQLite-Datei
# blockieren sich). Zusammengefuehrt wird mit merge_unknown_db.py, das an eine
# KOPIE von ergodic_dataset_improved.db anhaengt statt das Original zu
# veraendern.
#
#   sbatch run_data_gen_unknown.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

N_TASKS=8
TOTAL=1172
VON=$(( SLURM_ARRAY_TASK_ID * TOTAL / N_TASKS ))
BIS=$(( (SLURM_ARRAY_TASK_ID + 1) * TOTAL / N_TASKS ))

echo "Aufgabe $SLURM_ARRAY_TASK_ID: Zieldichten $VON bis $BIS"
echo "Knoten: $(hostname)   Start: $(date)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

python -u generate_dataset_unknown.py \
    --src_db ergodic_dataset_improved.db \
    --out "ergodic_dataset_improved_unknown_part${SLURM_ARRAY_TASK_ID}.db" \
    --shapes_from "$VON" --shapes_to "$BIS"

echo "Ende: $(date)"
