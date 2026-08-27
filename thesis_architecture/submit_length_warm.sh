#!/bin/bash
# Der Warmstart-Arm: Laengenkonditionierung auf dem auskonvergierten
# startpunktkonditionierten Netz.
#
#   ./submit_length_warm.sh [JOBID-auf-die-gewartet-wird]
#
# Zwei Glieder, weil 200 Epochen bei gemessenen 583 s/Epoche rund 32 h
# brauchen und das Limit auf stud bei 24 h liegt. Das zweite Glied setzt am
# Zwischenstand fort; ist der Lauf schon fertig, beendet es sich sofort
# wieder (Pruefung auf `_final` im Jobskript).
set -u
WARTE=${1:-}
DEP=""
[ -n "$WARTE" ] && DEP="--dependency=afterany:$WARTE"

GLIEDER=${GLIEDER:-2}
echo "Warmstart-Arm ($GLIEDER Glieder):"
for i in $(seq 1 "$GLIEDER"); do
    id=$(sbatch --parsable $DEP --export=ALL,TAG=WARM,LR=1e-4,CFG=2.0,PDROPLEN=0.1,EPOCHS=200,LENFREQ=linear run_job_length_warm.bash)
    echo "  Glied $i  ->  Job $id  ${DEP:+(nach ${DEP#*:})}"
    DEP="--dependency=afterany:$id"
done
echo
echo "Letztes Glied: $id"
