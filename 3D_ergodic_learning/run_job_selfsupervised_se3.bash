#!/bin/bash
#SBATCH -J selfsup_se3
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 20:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Selbstueberwachtes 3D-Training MIT Orientierung (Stufe 1+2).
# Das Modell lernt Position und Blickrichtung gemeinsam gegen die
# Solver-Energie plus Zeige-, Standoff- und Winkelglattheitsterm.
# Die Abdeckung wird auf dem Sensor-Fussabdruck gemessen, nicht auf der
# Roboterposition — nur so haengt die Abdeckung ueberhaupt von der
# Orientierung ab.
#
# Gegenstueck ohne Orientierung: run_job_selfsupervised_3d.bash
# Beide Laeufe sind direkt vergleichbar, die Run-Strings unterscheiden
# sich durch den _SE3-...-Marker.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

# Rechenknoten haben kein Display. matplotlib waehlt zwar meist selbst Agg,
# aber der Testlauf rendert echte Figures, und ein Backend-Fehler dort wuerde
# den Job abbrechen, bevor das Training ueberhaupt beginnt.
export MPLBACKEND=Agg

# Phase 0: Testlauf. Kostet Sekunden und faengt genau die Fehlerklasse ab, die
# sonst erst der Machbarkeits-Gate nach 40 Epochen bemerkt — und dann ohne die
# Ursache zu nennen. Konkreter Anlass: `geodesic_angle` hatte einen NaN im
# Rueckwaertspass bei identischen aufeinanderfolgenden Rotationen, also exakt im
# null-initialisierten Startzustand jedes --orientation-Laufs. Der Vorwaertspass
# meldete dabei lauter gesunde Werte, das Netz war nach einem Schritt tot.
echo "=== Phase 0: Testsuite ==="
srun --unbuffered python test_3d_port.py \
    || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

# Beim Fortsetzen nach dem Zeitlimit ist der Machbarkeitstest bereits bestanden.
# Ihn erneut zu fahren kostet in jeder Kette der Folgejobs Rechenzeit, ohne
# etwas Neues zu pruefen.
LATEST=$(ls -t checkpoints/selfsup_3d_selfsup3d_*_SE3-*_ep*.pt 2>/dev/null | head -1)
if [ -f "$LATEST" ]; then
    echo
    echo "=== Phase 1 uebersprungen (laufendes Training gefunden) ==="
else

echo
echo "=== Phase 1: Machbarkeitstest mit Orientierung ==="
srun --unbuffered python flow_matching_runner_particles_selfsupervised.py \
    --orientation --ergodic_on footprint \
    --shapes A,organic_0,digit_5 \
    --n_candidates 1 --diversity_weight 0.0 \
    --epochs 60 --gate_epochs 40 --assert_energy_drops --min_drop_ratio 0.5 \
    --D 128 --n_particles 128 --grid_res 48 --erg_K 6 \
    --mini_batch 3 --save_every 0 --viz_every 60 \
    || { echo "Machbarkeitstest fehlgeschlagen — Abbruch."; exit 1; }

fi

echo
echo "=== Phase 2: volles Training ==="
# Neu ermitteln: Phase 1 schreibt zwar keine Checkpoints (--save_every 0), aber
# so steht die Suche unmittelbar vor ihrer Verwendung.
LATEST=$(ls -t checkpoints/selfsup_3d_selfsup3d_*_SE3-*_ep*.pt 2>/dev/null | head -1)
RESUME=""
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    RESUME="--resume $LATEST"
fi

srun --unbuffered python flow_matching_runner_particles_selfsupervised.py \
    $RESUME \
    --orientation --ergodic_on footprint \
    --standoff_target 0.12 --standoff_band 0.03 \
    --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --n_train_shapes 750 \
    --n_candidates 8 --diversity_weight 10 \
    --epochs 500 \
    --D 384 --n_particles 512 --grid_res 64 --erg_K 8 \
    --mini_batch 32 \
    --save_every 20 --viz_every 20 \
    --use_wandb
