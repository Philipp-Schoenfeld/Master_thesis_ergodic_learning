#!/bin/bash
#SBATCH -J surf_10h
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 10:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Zehn-Stunden-Lauf auf derselben Flaechenauswahl wie surfA_kurz.
#
# Warum es diesen Lauf gibt: surfA_kurz (Job 135085) war mit 350 Epochen nach
# 1:29 h durch — nicht weil er auskonvergiert waere, sondern weil sein
# Kosinus-Plan ausgelaufen war. Die letzte Zeile lautete
#
#     350/350 [1:29:11, 13.93s/ep] cfm=0.12159 erg=1.66e-03 loss=0.73496 lr=1.00e-05
#
# Die Lernrate stand also am Boden, der Verlust aber noch bei 0,735. Die
# Epochenzahl war eine Annahme ueber die Hardware, keine Aussage ueber
# Konvergenz: geschaetzt waren 45 s je Epoche auf einer RTX 2080, tatsaechlich
# waren es 13,93 s auf der zugeteilten Karte — Faktor 3,2 daneben.
#
# Dieser Lauf setzt daher nicht fort, sondern faengt neu an. Ein Resume wuerde
# den Lernratenplan an seinem Boden weiterfuehren und damit kaum noch etwas
# bewegen; ein frischer Lauf faehrt den Kosinus ueber die volle neue Laenge.
#
# Die Epoche war bisher winzig. Mit --copies_per_char 1 und mini_batch 64
# sind 5250 Trainingseintraege genau 82 Schritte je Epoche; 350 Epochen waren
# damit 28.700 Schritte — gegen 113.500 im 2D-Referenzlauf. Der Runner-Standard
# ist 15, die 1 stammte aus dem Timing-Test.
#
# Dieser Lauf setzt --copies_per_char 30: 157.500 Stichproben, 2.461 Schritte je
# Epoche. 80 Epochen sind 196.875 Schritte, also das 6,9-fache von surfA_kurz
# und das 1,6-fache des 2D-Referenzlaufs. Bei den gemessenen 0,170 s je Schritt
# sind das 8:43 h — gut eine Stunde Reserve im 10-Stunden-Fenster, damit der
# Kosinus-Plan sicher auslaeuft statt vom Zeitlimit abgeschnitten zu werden.
#
# Wichtig zur Einordnung: Kopien und Epochen sind hier austauschbar. Das
# Produkt bestimmt die Schrittzahl, und ab Zeile 279 des Runners wird in JEDER
# Epoche neu gemischt (torch.randperm), es entsteht durch mehr Kopien also auch
# kein besseres Durchmischen. 30x80 und 4x600 waeren dasselbe Training. Gewaehlt
# ist 30, weil es der 2D-Konvention entspricht und die Epoche als Einheit wieder
# etwas bedeutet.
#
# Was sich dadurch tatsaechlich aendert, ist die Kadenz: --save_every und
# --viz_every zaehlen in Epochen. Bei nur 80 Epochen waeren die alten Werte 20
# und 100 unbrauchbar gewesen — letzterer haette KEINE einzige Visualisierung
# erzeugt. Gesetzt sind daher 5 und 10.
#
# Vorbehalt: die Rechnung gilt fuer die RTX 3080 Ti, auf der surfA_kurz lief
# (Knoten cn14). Auf der Karte von surf_lang dauert eine Epoche das 5,2-fache.
# Landet der Job dort, schneidet das Zeitlimit ihn ab und --signal=SIGTERM@120
# schreibt einen letzten Stand — der Lernratenplan waere dann nicht ausgelaufen.
# Eine GPU laesst sich auf stud nicht auswaehlen, das ist nicht abzustellen.
#
# surfA_kurz wird dabei NICHT ueberschrieben: eigener Marker surfA_10h, und die
# Checkpoint-Rotation loescht nur, was auf denselben run_str passt.
#
#   sbatch run_job_3d_10h.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/3D_ergodic_learning

TAG="surfA_10h"
CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Falls ein frueherer Versuch abgebrochen ist: den letzten Stand aufnehmen.
RES=$(ls -t "$CKPT_DIR"/*_${TAG}_*_ep*.pt 2>/dev/null | head -1)
RES_ARG=""
if [ -n "$RES" ]; then
    echo "Fortsetzung ab: $(basename "$RES")"
    RES_ARG="--resume $RES"
fi

srun --unbuffered python -u flow_matching_runner_particles.py \
    --db3d ergodic_dataset_3d.db \
    --run_tag "$TAG" \
    $RES_ARG \
    --orientation --rot_full \
    --lambda_erg 100 --erg_K 6 --erg_on position --erg_t_power 2.0 \
    --lambda_ori 0.0 --w_cfm_rot 1.0 \
    --D 384 --n_particles 512 --nxi 25 \
    --copies_per_char 30 --mini_batch 64 --lr 1e-4 \
    --save_every 5 --keep_checkpoints 1 \
    --viz_every 10 \
    --save_model "$CKPT_DIR/surf.pt" \
    --use_wandb \
    --epochs 75

echo "Endzeit:    $(date)"
