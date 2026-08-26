#!/bin/bash
#SBATCH -J len_hp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 2
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
# Ein Glied der Hyperparameter-Kette. Die Einstellung kommt ueber Umgebungs-
# variablen, damit dasselbe Skript fuer alle Punkte der Suche dient und die
# Verkettung per --dependency ohne Skript-Kopien auskommt.
#
#   LR, CFG, LOGSCALE, PDROPLEN, TAG  --  gesetzt von submit_length_chain.sh
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

echo "Punkt der Suche: TAG=$TAG  LR=$LR  CFG=$CFG  LOGSCALE=${LOGSCALE:-auto}  PDROPLEN=$PDROPLEN"

SKALA=""
[ -n "$LOGSCALE" ] && SKALA="--log_scale $LOGSCALE"

ARGS="--D 384 --n_particles 256 --nxi 64 --copies_per_char 2 --p_flip 0.0 \
      --epochs $EPOCHS --mini_batch 128 --lr $LR \
      --save_every 10 --viz_every 20 --keep_checkpoints 1 \
      --p_drop_length $PDROPLEN --cfg_weight $CFG $SKALA \
      --lambda_erg 1300 --erg_metric sinkhorn --sinkhorn_blur 0.05 --erg_t_power 2 \
      --db ergodic_dataset_length.db --tag $TAG --use_wandb"

LATEST=$(ls -t checkpoints/*_LEN-*_${TAG}_*_ep*.pt 2>/dev/null | head -1)
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    python -u flow_matching_runner_length.py --resume "$LATEST" $ARGS
else
    python -u flow_matching_runner_length.py $ARGS
fi
