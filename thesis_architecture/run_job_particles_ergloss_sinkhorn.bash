#!/bin/bash
#SBATCH -J flow_erg_sink
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 20:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Sinkhorn statt Fourier im ergodischen Loss.
#
# Warum: die Reihe w=2..400 hat E_ergodic von 4.25 auf 2.14 gedrueckt, waehrend
# die basisfreie Abdeckung bei 0.049-0.052 stehenblieb und den Solver (0.0476)
# nie geschlagen hat. Das ist die Signatur einer abgeschnittenen Metrik: die
# K=8-Basis (64 Moden) sieht die Feinstruktur nicht, die die Abdeckung misst.
# Die Sinkhorn-Divergenz hat keine Abschneidung. Vgl. Sun, Pinosky & Murphey
# (2025, arXiv:2504.17872), die dieselbe Ersetzung vornehmen.
#
# Gewichtswahl: nicht 300 uebernehmen. Auf Solver-Trajektorien ist der
# Sinkhorn-Gradient 4.25x schwaecher als der Fourier-Gradient (gemessen in
# test_sinkhorn_metric.py), also ist w=1300 das Gegenstueck zu w=300 aus der
# Fourier-Reihe. Bei gleichem w waere der Term schlicht 4x schwaecher und der
# Vergleich wertlos.
#
# Kosten: sinkhorn_scaling=0.5 statt der geomloss-Vorgabe 0.9 macht den Term
# 6x billiger, bei 0.16 % Abweichung im Gradienten (4 % im Wert, aber den nutzt
# das Training nicht). Auf CPU bleibt Sinkhorn damit rund 5x teurer als
# Fourier; auf der GPU sollte der Abstand kleiner sein, weil die 128x256-Matrix
# gut parallelisiert. Die erste Epochenzeit im W&B-Log pruefen: reicht sie
# nicht fuer 500 Epochen in 20 h, faengt der SIGTERM-Handler den Lauf ab und
# eine Folgejob-Kette per --dependency=afterany setzt ihn fort.
#
# WICHTIG: braucht geomloss in der thesis-Umgebung (pip install geomloss).
#
# Ausgewertet wird spaeter wie alle anderen Laeufe mit der Fourier-Metrik und
# der Abdeckungsdistanz. Das ist Absicht: die Frage ist, ob ein auf Sinkhorn
# trainiertes Modell auf der *anderen* Metrik mithaelt und bei der Abdeckung
# gewinnt.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

python -c "import geomloss" 2>/dev/null || {
    echo "FEHLER: geomloss fehlt in der thesis-Umgebung."
    echo "Einmalig nachinstallieren:  pip install geomloss"
    exit 1
}

WEIGHT=${WEIGHT:-1300}
BLUR=${BLUR:-0.05}

COMMON=(
    --D 384
    --n_particles 256
    --copies_per_char 15
    --p_flip 0.0
    --epochs 500
    --mini_batch 256
    --save_every 20
    --viz_every 20
    --lambda_erg $WEIGHT
    --erg_metric sinkhorn
    --sinkhorn_blur $BLUR
    --sinkhorn_scaling 0.5
    --erg_pts 128
    --erg_t_power 2.0
    --use_wandb
)

PAT="checkpoints/*flow_matching_particle_ergodic_date_*_nxi25_D384_N256_C15_flip0.0"
PAT="${PAT}_ERGLOSS-SINKHORN-w${WEIGHT}-blur${BLUR}-tp2_ep*.pt"
LATEST_CKPT=$(ls -t $PAT 2>/dev/null | head -1)

if [ -f "$LATEST_CKPT" ]; then
    echo "Setze Training fort von: $LATEST_CKPT"
    srun --unbuffered python flow_matching_runner_particles.py \
        --resume "$LATEST_CKPT" "${COMMON[@]}"
else
    echo "Starte neues Training (w=$WEIGHT, blur=$BLUR)..."
    srun --unbuffered python flow_matching_runner_particles.py "${COMMON[@]}"
fi
