#!/bin/bash
# Die Hyperparameter-Suche als Kette einreihen.
#
# Verkettet statt parallel, aus einem harten Grund: das Kontingent erlaubt drei
# GPUs gleichzeitig, und die sind von den laufenden Trainings belegt. Eine
# Kette laeuft auch dann durch, wenn nur ein Platz frei wird, und jedes Glied
# beginnt erst, wenn das vorige beendet ist — mit afterany auch dann, wenn es
# ins Zeitlimit gelaufen ist.
#
#   ./submit_length_chain.sh [JOBID-auf-die-gewartet-wird]
#
# Gesucht wird ueber drei Achsen, je Achse ein Glied:
#   1  Lernrate            1e-4 (Bezug) gegen 3e-4
#   2  CFG-Gewicht         2.0 gegen 4.0
#   3  Log-Normierung      automatisch gegen fest 1.0
#   4  Laengen-Dropout     0.1 gegen 0.25
set -u
WARTE=${1:-}
DEP=""
[ -n "$WARTE" ] && DEP="--dependency=afterany:$WARTE"

reihe () {   # $1=TAG  $2=Variablen
    local id
    id=$(sbatch --parsable $DEP --export=ALL,$2 run_job_length_hp.bash)
    echo "  $1  ->  Job $id  ${DEP:+(nach ${DEP#*:})}"
    DEP="--dependency=afterany:$id"
    echo "$id"
}

echo "Hyperparameter-Kette:"
A=$(reihe "lr1e-4  (Bezug)" "TAG=LR1E4,LR=1e-4,CFG=2.0,PDROPLEN=0.1,EPOCHS=200" | tail -1)
B=$(reihe "lr3e-4"          "TAG=LR3E4,LR=3e-4,CFG=2.0,PDROPLEN=0.1,EPOCHS=200" | tail -1)
C=$(reihe "cfg4.0"          "TAG=CFG40,LR=1e-4,CFG=4.0,PDROPLEN=0.1,EPOCHS=200" | tail -1)
D=$(reihe "logscale1.0"     "TAG=LOGS10,LR=1e-4,CFG=2.0,LOGSCALE=1.0,PDROPLEN=0.1,EPOCHS=200" | tail -1)
E=$(reihe "pdroplen0.25"    "TAG=PD025,LR=1e-4,CFG=2.0,PDROPLEN=0.25,EPOCHS=200" | tail -1)
echo
echo "Letztes Glied: $E"
