#!/bin/bash
#SBATCH -J len_warm
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=10G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Ressourcen, gemessen statt geschaetzt — und einmal falsch gemacht:
#
#   sacct MaxRSS (.batch-Schritt) der Laeufe vom 26./27.08.
#     136800  len_test      7.11 G   lief durch
#     137029  len_hp (8 G)  6.91 G   lief durch (86 % der Anforderung)
#     137409  len_warm(6 G) 6.05 G   OUT_OF_MEMORY nach 1:10 h
#     137412  len_hp (6 G)  5.97 G   OUT_OF_MEMORY nach 1:10 h
#     137413  len_hp (6 G)  5.97 G   OUT_OF_MEMORY nach 1:10 h
#     137414  len_hp (6 G)  5.76 G   OUT_OF_MEMORY nach 4:05 h
#
# Eine fruehere Fassung dieses Skripts forderte 6 G an, gestuetzt auf eine
# Effizienzangabe von 2,59 GB. Diese Zahl war falsch — der tatsaechliche
# Spitzenwert desselben Laufs betrug 7,11 G. Vier Jobs starben daran.
#
# 10 G liegen ueber dem gemessenen Spitzenwert und lassen rund 40 % Luft; das
# sind etwa 71 % Auslastung, weit oberhalb dessen, was der Cluster bemaengelt
# hatte. Seit der Entdopplung der Dichtekarten (1.187 statt 20.593 Gitter,
# rund 2,6 GB weniger) duerfte der Bedarf spuerbar darunter liegen — das ist
# nach dem ersten Lauf per `sacct MaxRSS` zu pruefen und DANN zu senken,
# nicht vorher.

# Ketten-Skripts waeren 32 % gewesen.
#
# Zwei Kerne, weil die Arbeit auf der GPU liegt und der Prozess effektiv
# einen Kern nutzt — der zweite deckt den Vorlauf ab, in dem die Dichtekarten
# ueber JAX auf der CPU gerechnet werden. Ob `-c 1` reicht, ist die naechste
# Sache, die nach diesem Lauf per `sacct` zu pruefen ist.

# Laengenkonditionierung mit WARMSTART aus dem startpunktkonditionierten Netz.
#
# Der Gegenpol zu run_job_length_hp.bash, das bei null anfaengt. Beide Laeufe
# benutzen dieselbe Architektur, denselben Datensatz, dieselben
# Hyperparameter und dieselbe (korrigierte) Laengenkodierung — der einzige
# Unterschied ist der Ausgangszustand. Damit ist der Vergleich sauber.
#
# Warum das gehen kann: die laengenkonditionierte Architektur unterscheidet
# sich von der startpunktkonditionierten um genau fuenf Tensoren, naemlich den
# Laengenkopf (154.752 von 87.663.938 Parametern, 0,177 %). Alles andere wird
# uebernommen; `pos_emb` wird von n_xi=25 auf 64 entlang der
# Kontrollpunktachse interpoliert. Der Laengenkopf wird mit Nullausgang
# initialisiert, sodass das erweiterte Netz im ersten Schritt funktional
# bitgleich mit dem Ausgangsmodell ist.
#
# Ressourcen wie bei der Kette: gemessen 25 % CPU-Effizienz und rund 3 GB RAM,
# die Arbeit liegt auf der GPU.
#
#   sbatch run_job_length_warm.bash
#   SEED_MODELL=... EPOCHS=400 sbatch run_job_length_warm.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

LR=${LR:-1e-4}
CFG=${CFG:-2.0}
PDROPLEN=${PDROPLEN:-0.1}
TAG=${TAG:-WARM}
EPOCHS=${EPOCHS:-200}
LENFREQ=${LENFREQ:-linear}
SEED_MODELL=${SEED_MODELL:-checkpoints/netz2d_startpunkt.pt}

echo "Warmstart-Lauf: TAG=$TAG  LR=$LR  CFG=$CFG  PDROPLEN=$PDROPLEN  LENFREQ=$LENFREQ"
echo "Knoten: $(hostname)   Start: $(date)"

ARGS="--D 384 --n_particles 256 --nxi 64 --copies_per_char 2 --p_flip 0.0 \
      --epochs $EPOCHS --mini_batch 128 --lr $LR \
      --save_every 10 --viz_every 20 --keep_checkpoints 1 \
      --p_drop_length $PDROPLEN --cfg_weight $CFG \
      --length_freqs $LENFREQ \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --db ergodic_dataset_length.db --tag $TAG --use_wandb"

MARKE=$(echo "$LENFREQ" | tr '[:lower:]' '[:upper:]')FREQ
MUSTER="checkpoints/*_LEN-*_${MARKE}_FROMSTART_${TAG}_*"

# Wie im Ketten-Skript: nach dem Endstand gibt es keine `_ep`-Staende mehr,
# ein Folgeglied wuerde sonst erneut beim Ausgangsmodell beginnen.
FERTIG=$(ls -t ${MUSTER}_final.pt 2>/dev/null | head -1)
if [ -n "$FERTIG" ]; then
    echo "Warmstart-Lauf ist bereits abgeschlossen: $FERTIG"
    exit 0
fi

EIGEN=$(ls -t ${MUSTER}_ep*.pt 2>/dev/null | head -1)

if [ -n "$EIGEN" ] && [ -f "$EIGEN" ]; then
    # Ab dem zweiten Glied normal fortsetzen. --resume gewinnt im Runner
    # ohnehin gegen --init_from; der Warmstart wird also nicht wiederholt.
    echo "Setze fort von: $EIGEN"
    python -u flow_matching_runner_length.py --resume "$EIGEN" $ARGS
elif [ -f "$SEED_MODELL" ]; then
    echo "Erstes Glied: Warmstart aus $SEED_MODELL"
    python -u flow_matching_runner_length.py --init_from "$SEED_MODELL" $ARGS
else
    echo "FEHLER: Ausgangsmodell $SEED_MODELL nicht gefunden." >&2
    echo "Ohne es waere dies nur ein zweiter Lauf von Grund auf — und damit" >&2
    echo "kein Vergleich. Abbruch, statt stillschweigend das Falsche zu tun." >&2
    exit 1
fi

echo "Ende: $(date)"
