#!/bin/bash
#SBATCH -J start_ft_imp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Finetuning des Partikel+Start-Checkpoints (run_job_start_lang.bash, L500,
# CFM+ErgLoss) auf ergodic_dataset_improved_final.db: 1172 Zieldichten je nur
# noch eine (die ergodisch beste) Trajektorie, plus zwei Explorations-
# Varianten mit zufaelligem "unbekanntem" Flaechenbereich je Zieldichte
# (siehe ergodic_dataset_generator/unknown_region.py) -- 3516 Zeilen statt
# 1187.
#
# WICHTIG, erst nach Pruefung einzureichen:
#   1. ergodic_dataset_improved_final.db muss im Generator-Verzeichnis liegen
#      (per rsync hochgeladen, nach merge_unknown_db.py).
#   2. Der Quell-Checkpoint unten muss der REPARIERTE Stand sein (start_cond
#      korrekt gesetzt) -- vor dem Hochladen mit model_zoo.load_model lokal
#      geprueft, siehe Session-Notizen.
#
# Erster Start: --finetune_from uebernimmt nur die Gewichte, Optimizer/
# Scheduler/Epochenzaehler sind frisch ueber die vollen neuen 500 Epochen
# (anders als --resume, das einen bereits laufenden/unterbrochenen Lauf an
# genau diesem Punkt fortsetzt und daher dessen Optimizer-/Scheduler-State
# braucht -- ein `_final.pt` hat den nicht mehr, siehe `nach_endstand`).
# Eigener --tag (FTIMPROVED), damit dieser Lauf nicht mit dem Quell-Lauf
# (Tag L500) oder miteinander laufenden anderen Experimenten kollidiert.
#
# Folge-Jobs nach Zeitlimit-Abbruch werden ueber eine vorab eingereihte
# `--dependency=afterany:<JOBID>`-Kette automatisch nachgefuehrt (siehe
# Session-Notizen zur Zeitschaetzung). Das Skript erkennt seinen eigenen
# Zwischenstand (Tag FTIMPROVED) automatisch und wechselt dann selbst auf
# --resume statt --finetune_from -- und wenn der Lauf schon vollstaendig
# abgeschlossen ist (nur noch `_final.pt` vorhanden, siehe `nach_endstand`),
# tut ein ueberzaehliger Kettenglied-Job GAR NICHTS, statt die kompletten 500
# Epochen versehentlich neu zu starten.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

FINETUNE_FROM="checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"

ARGS="--D 384 --n_particles 256 --copies_per_char 75 --p_flip 0.0 \
      --epochs 500 --mini_batch 256 --save_every 10 --viz_every 10 \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --tag FTIMPROVED --keep_checkpoints 1 \
      --db ergodic_dataset_improved_final.db --use_wandb"

FINAL_DONE=$(ls checkpoints/*_START_FLAT*_FTIMPROVED_*_final.pt 2>/dev/null | head -1)
LATEST=$(ls -t checkpoints/*_START_FLAT*_FTIMPROVED_*_ep*.pt 2>/dev/null | head -1)

if [ -f "$FINAL_DONE" ]; then
    echo "Lauf bereits abgeschlossen ($FINAL_DONE) -- ueberzaehliger Kettenglied-Job, nichts zu tun."
elif [ -f "$LATEST" ]; then
    echo "Setze Finetune-Lauf fort von: $LATEST"
    srun --unbuffered python flow_matching_runner_start.py --resume "$LATEST" $ARGS
else
    echo "Starte Finetuning von: $FINETUNE_FROM"
    srun --unbuffered python flow_matching_runner_start.py --finetune_from "$FINETUNE_FROM" $ARGS
fi
