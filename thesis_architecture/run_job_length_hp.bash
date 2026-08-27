#!/bin/bash
#SBATCH -J len_hp
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

# Konfiguration und brauchte 2,59 GB Spitze. 8 G waren 32 %
# Effizienz, 6 G sind rund 43 % bei weiterhin doppeltem Spielraum.
#
# Jeder Job nutzt effektiv EINEN Kern — die Arbeit liegt auf der GPU
# beziehungsweise ist in JAX sequentiell ueber die Zeitschritte. Vier Kerne
# und 32 G anzufordern blockiert andere Nutzer und die eigenen Jobs, die sonst
# am Kontingent (cpu=50, gres/gpu=3, mem=150G) warten.
# Ein Glied der Hyperparameter-Kette. Die Einstellung kommt ueber Umgebungs-
# variablen, damit dasselbe Skript fuer alle Punkte der Suche dient und die
# Verkettung per --dependency ohne Skript-Kopien auskommt.
#
#   LR, CFG, LOGSCALE, PDROPLEN, TAG, LENFREQ  --  von submit_length_chain.sh
#
# LENFREQ waehlt die Frequenzen der Laengenkodierung. Die Vorgabe bleibt
# `oktaven`, damit dieses Skript ohne gesetzte Variable dasselbe tut wie
# bisher; die Kette setzt `linear`. Hintergrund: unter `oktaven` ist der
# ganze Merkmalsvektor periodisch in der normierten Laenge mit der Periode 2,
# und der Datensatzbereich umfasst 3,34 Perioden — L = 4,00 / 10,22 / 24,17
# sind bitgleiche Eingaben, das Netz kann sie nicht unterscheiden.
#
# 24 Stunden sind die Obergrenze auf stud. Wird ein Lauf abgeschnitten,
# nimmt das naechste Glied denselben TAG und setzt am Checkpoint fort; die
# Rotation haelt je Lauf genau einen Stand.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

LR=${LR:-1e-4}
CFG=${CFG:-2.0}
LOGSCALE=${LOGSCALE:-}
PDROPLEN=${PDROPLEN:-0.1}
TAG=${TAG:-HP}
EPOCHS=${EPOCHS:-200}
LENFREQ=${LENFREQ:-oktaven}

echo "Punkt der Suche: TAG=$TAG  LR=$LR  CFG=$CFG  LOGSCALE=${LOGSCALE:-auto}  PDROPLEN=$PDROPLEN  LENFREQ=$LENFREQ"

SKALA=""
[ -n "$LOGSCALE" ] && SKALA="--log_scale $LOGSCALE"

ARGS="--D 384 --n_particles 256 --nxi 64 --copies_per_char 2 --p_flip 0.0 \
      --epochs $EPOCHS --mini_batch 128 --lr $LR \
      --save_every 10 --viz_every 20 --keep_checkpoints 1 \
      --p_drop_length $PDROPLEN --cfg_weight $CFG $SKALA \
      --length_freqs $LENFREQ \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --db ergodic_dataset_length.db --tag $TAG --use_wandb"

if [ "$LENFREQ" = "oktaven" ]; then
    MARKE=""
    MUSTER="checkpoints/*_LEN-*_${TAG}_*"
else
    MARKE=$(echo "$LENFREQ" | tr '[:lower:]' '[:upper:]')FREQ
    MUSTER="checkpoints/*_LEN-*_${MARKE}_${TAG}_*"
fi

# Ist dieser Punkt der Suche schon fertig? Dann nichts tun.
#
# Das ist kein Luxus, sondern noetig, seit je Punkt MEHRERE Glieder eingereiht
# werden: 200 Epochen brauchen bei gemessenen 583 s/Epoche rund 32 h und
# passen nicht in das 24-Stunden-Limit. Nach dem Endstand loescht die Rotation
# alle `_ep`-Staende — ein zweites Glied wuerde ohne diese Pruefung keinen
# Zwischenstand mehr finden und stillschweigend WIEDER BEI NULL anfangen.
FERTIG=$(ls -t ${MUSTER}_final.pt 2>/dev/null | head -1)
if [ -n "$FERTIG" ]; then
    echo "Punkt $TAG ist bereits abgeschlossen: $FERTIG"
    echo "Nichts zu tun."
    exit 0
fi

# Fortsetzungsstand: nur ein Stand DERSELBEN Frequenzwahl. Ein Stand der
# anderen waere technisch brauchbar (der Runner setzt den Laengenkopf per
# --reset_length_emb auto selbst zurueck), aber dann waere der Lauf kein
# Training "von null auf" mehr, sondern eines auf einem Ruecken, der bereits
# auf diesem Datensatz vortrainiert ist. Genau dieser Unterschied ist das,
# was der Warmstart-Lauf messen soll — er darf hier nicht durch die
# Hintertuer einfliessen. Wer es ausdruecklich will, setzt SEED=<pfad>.
LATEST=${SEED:-$(ls -t ${MUSTER}_ep*.pt 2>/dev/null | head -1)}

if [ -n "$LATEST" ] && [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    python -u flow_matching_runner_length.py --resume "$LATEST" $ARGS
else
    echo "Kein Stand zu $TAG/$LENFREQ gefunden — Training von null auf."
    python -u flow_matching_runner_length.py $ARGS
fi
