#!/bin/bash
#SBATCH -J flow3d_ori_probe
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 04:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# VORLAUF zu run_job_particles_3d_ori.bash — 40 statt 500 Epochen.
#
# Zweck: Epochenzeit, Speicherbedarf und vor allem das Verhaeltnis von
# train/ori_weighted zu train/erg_weighted in W&B ablesen, bevor 20 h darauf
# verwendet werden. lambda_ori=0.012 ist aus Stufe-0-Loesungen kalibriert, wo
# die Bahn noch in der Zielebene liegt und der Standoff-Term entsprechend gross
# ist. Sobald sich die Bahn abhebt, faellt er stark (im selbstueberwachten Lauf
# von 75 auf 0.14) — das Verhaeltnis verschiebt sich also waehrend des Trainings
# und die Kalibrierung ist danach ehrlicher zu waehlen.
#
# Modellgroesse bleibt identisch zum vollen Lauf, sonst sagt der Vorlauf nichts
# ueber Speicher und Epochenzeit aus. Bleibt lambda_ori unveraendert, setzt der
# volle Lauf ueber denselben Run-String auf diesen Checkpoints auf.
#
# CFM in 3D mit ergodischem Term UND Orientierung als Ziel (Stufe 3).
#
# Unterschied zum bisherigen Orientierungspfad: die Rotation wird nicht mehr
# gegen die Stufe-0-Look-at-Frames imitiert, sondern gegen ihre eigene
# Zielfunktion optimiert — Zeigen, Standoff, Winkelglattheit. Die Frames sind
# keine Messdaten (die Datenbank enthaelt ueberhaupt keine Orientierung),
# sondern eine geometrische Konstruktion; ein Netz, das sie perfekt trifft, hat
# nur eine Formel nachgebaut. Dieselbe Deckel-Logik, die der ergodische Term in
# 2D fuer die Position durchbrochen hat.
#
# --erg_on footprint ist hier keine Option, sondern Voraussetzung. Die Daten
# sind angehobene 2D-Bahnen, die Kurve liegt also *in* der Zielebene und ihr
# Abstand zur Flaeche ist ~0. Ein Standoff von 0.12 verlangt, die Ebene zu
# verlassen; ein auf der Position gemessener ergodischer Term verlangt, darin zu
# bleiben. Am Fussabdruck gemessen wollen beide dasselbe: der Roboter haelt
# Abstand, der Strahl landet wieder auf der Flaeche. Der Runner bricht ohne
# --erg_on footprint mit einer Erklaerung ab.
#
# Gewichtung: der Orientierungsterm ist auf Stufe-0-Loesungen im Gradienten rund
# 8000x staerker als der ergodische (gemessen ueber test_3d_port.py und die
# Kalibrierung darin). lambda_ori=0.012 stellt ihn neben lambda_erg=100 auf
# vergleichbare Staerke. Ungewichtet wuerde er beides erschlagen.
#
# w_cfm_rot=0.5 laesst die Stufe-0-Imitation als Warm-Start stehen, statt sie
# ganz zu streichen. Fuer den reinen Zielbetrieb w_cfm_rot=0 setzen; fuer die
# alte reine Imitation lambda_ori=0.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

echo "=== Phase 0: Testsuite ==="
srun --unbuffered python test_3d_port.py \
    || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

LAMBDA_ERG=${LAMBDA_ERG:-100}
LAMBDA_ORI=${LAMBDA_ORI:-0.012}
W_CFM_ROT=${W_CFM_ROT:-0.5}

PAT="checkpoints/cond_particles_3d_flow3d_*_SE3-*_ERGLOSS-w${LAMBDA_ERG}-*"
PAT="${PAT}_ORILOSS-w${LAMBDA_ORI}-*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)
RESUME=""
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    RESUME="--resume $LATEST"
fi

echo
echo "=== Training: ErgLoss + Orientierungsziel ==="
srun --unbuffered python flow_matching_runner_particles.py \
    $RESUME \
    --orientation --frame_mode lookat \
    --erg_on footprint \
    --lambda_erg $LAMBDA_ERG --erg_K 6 --erg_pts 128 --erg_t_power 2.0 \
    --lambda_ori $LAMBDA_ORI --w_cfm_rot $W_CFM_ROT \
    --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --D 384 --n_particles 512 --grid_res 64 \
    --copies_per_char 15 --p_flip 0.0 \
    --epochs 40 --mini_batch 128 \
    --save_every 10 --viz_every 10 \
    --use_wandb
