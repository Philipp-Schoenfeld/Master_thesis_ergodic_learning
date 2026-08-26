#!/bin/bash
#SBATCH -J laengen_ds
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#SBATCH -t 12:00:00
#SBATCH -p stud
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=3G
#SBATCH -c 2
#SBATCH --array=0-7
#SBATCH --signal=SIGTERM@120

# Ressourcen sind gemessen, nicht geschaetzt. Aus `sacct` ueber die Laeufe
# vom 24. bis 26.08.:
#
#     Job                CPU-Eff   RAM-Eff   RAM genutzt
#     surf_lang           25.0 %     4.8 %      1.54 GB
#     start_kurz          25.2 %    10.1 %      3.25 GB
#     len_test            26.4 %     8.1 %      2.59 GB
#     Datensatz-Array     18.5 %    21.7 %      1.74 GB
#
# Jeder Job nutzt effektiv EINEN Kern — die Arbeit liegt auf der GPU
# beziehungsweise ist in JAX sequentiell ueber die Zeitschritte. Vier Kerne
# und 32 G anzufordern blockiert andere Nutzer und die eigenen Jobs, die sonst
# am Kontingent (cpu=50, gres/gpu=3, mem=150G) warten.
# Laengen-Datensatz: je Zielverteilung bis zu fuenfzehn Varianten aus einem
# einzigen SVGD-Lauf, an festen Iterations-Checkpoints mitgeschrieben.
#
# BEWUSST OHNE GPU. Der Loeser rechnet ueber JAX auf der CPU; eine Karte
# braeuchte er nicht, und ohne --gres laeuft der Job auch dann, wenn die drei
# GPUs des Kontingents von Trainings belegt sind.
#
# Aufteilung: acht Array-Aufgaben zu je zwei Kernen sind 16 CPUs und lassen
# damit 34 des Kontingents fuer die GPU-Trainings frei. Mit den vorherigen
# sechs Kernen je Aufgabe waren es 48 — die eigenen Trainings warteten dann
# auf `AssocGrpCpuLimit`, obwohl eine GPU frei war.
#
# Jede Aufgabe schreibt in eine EIGENE Datenbank. Acht Prozesse, die
# gleichzeitig in dieselbe SQLite-Datei schreiben, blockieren sich
# gegenseitig; zusammengefuehrt wird am Ende mit merge_length_db.py.
#
#   sbatch run_data_gen.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/ergodic_dataset_generator

N_TASKS=8
TOTAL=1187
VON=$(( SLURM_ARRAY_TASK_ID * TOTAL / N_TASKS ))
BIS=$(( (SLURM_ARRAY_TASK_ID + 1) * TOTAL / N_TASKS ))

echo "Aufgabe $SLURM_ARRAY_TASK_ID: Formen $VON bis $BIS"
echo "Knoten: $(hostname)   Start: $(date)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

python -u generate_dataset_length.py \
    --db "ergodic_dataset_length_part${SLURM_ARRAY_TASK_ID}.db" \
    --tsteps 400 \
    --konvergenz 0.01 \
    --shapes_from "$VON" --shapes_to "$BIS"

echo "Ende: $(date)"
