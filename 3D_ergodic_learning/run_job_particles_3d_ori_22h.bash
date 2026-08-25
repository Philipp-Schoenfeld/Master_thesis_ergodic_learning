#!/bin/bash
#SBATCH -J flow3d_ori22
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 22:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# CFM in 3D mit ergodischem Term UND Orientierung als Ziel (Stufe 3), 22 h.
#
# ── Warum 22 h und warum das trotzdem nicht sicher reicht ──────────────────
# profile_step.py hat den Schritt aufgeschluesselt: das CFM-Netz (87.5 M
# Parameter) ist 94.4 % davon, die Fussabdruck-Kopplung und der
# Orientierungsterm zusammen 1.2 %, das Partikel-Sampling 4.2 %. An den
# Zusatztermen ist also nichts zu holen — die Laufzeit ist die des Netzes.
#
# Auf einer RTX 2080 SUPER sind das ~15 ms/Sample, also ~176 s/Epoche bei
# mini_batch 128 und rund 24 h fuer 500 Epochen. Der erste Lauf landete auf
# dgx-station und kam dort auf ~75 ms/Sample, das Fuenffache, was 878 s/Epoche
# ergab. Welchen Knoten man bekommt, entscheidet also ueber Faktor 5 — und
# GPU-Cherry-Picking ist auf stud verboten (die -C-Zeile bleibt auskommentiert).
#
# Deshalb: 22 h statt 20 (das Limit der Partition ist 24 h), --save_every 10
# statt 20, damit ein Abbruch hoechstens zehn Epochen kostet, und eine
# Job-Kette. Die Epochenzahl bleibt bei 500, weil das die Vergleichslaenge der
# 2D-Reihe ist; sie wird nicht an die Wanduhr angepasst.
#
# Folgejob anhaengen (so oft wie noetig):
#   sbatch --dependency=afterany:<JOBID> run_job_particles_3d_ori_22h.bash
# Der Lauf nimmt ueber den Run-String automatisch seinen letzten Checkpoint auf.
#
# ── Warum --erg_on footprint Pflicht ist ───────────────────────────────────
# Die Daten sind angehobene 2D-Bahnen: die Kurve liegt in der Zielebene, ihr
# Abstand zur Flaeche ist ~0. Ein Standoff von 0.12 verlangt, die Ebene zu
# verlassen; ein auf der Position gemessener ergodischer Term verlangt, darin zu
# bleiben. Am Fussabdruck gemessen wollen beide dasselbe. Der Runner bricht ohne
# --erg_on footprint mit einer Erklaerung ab.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

LAMBDA_ERG=${LAMBDA_ERG:-100}
LAMBDA_ORI=${LAMBDA_ORI:-0.012}
W_CFM_ROT=${W_CFM_ROT:-0.5}

PAT="checkpoints/cond_particles_3d_flow3d_*_SE3-*_ERGLOSS-w${LAMBDA_ERG}-*"
PAT="${PAT}_ORILOSS-w${LAMBDA_ORI}-*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

# Die Testsuite nur beim Kaltstart. In einer Job-Kette hat sie beim ersten Glied
# bereits bestanden, und der Code aendert sich zwischen den Gliedern nicht.
if [ ! -f "$LATEST" ]; then
    echo "=== Phase 0: Testsuite (Kaltstart) ==="
    srun --unbuffered python test_3d_port.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }
fi

RESUME=""
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    RESUME="--resume $LATEST"
else
    echo "Kaltstart (kein passender Checkpoint gefunden)."
fi

echo
echo "=== Training: ErgLoss(footprint) + Orientierungsziel, 22 h ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

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
    --epochs 500 --mini_batch 128 \
    --save_every 10 --viz_every 20 \
    --use_wandb
