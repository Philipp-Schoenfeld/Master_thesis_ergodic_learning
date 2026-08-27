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
#
# Alle Glieder laufen mit LENFREQ=linear, also der korrigierten
# Laengenkodierung. Unter der bisherigen Wahl `oktaven` waren verschiedene
# Laengen bitgleich kodiert (3,34 Perioden ueber dem Datensatzbereich), und
# kein Punkt der Suche haette eine Aussage ueber die Laengenkonditionierung
# erlaubt. Liegt zu einem TAG bereits ein alter Stand vor, setzt das Glied
# darauf auf und der Runner initialisiert den Laengenkopf neu.
set -u
WARTE=${1:-}
DEP=""
[ -n "$WARTE" ] && DEP="--dependency=afterany:$WARTE"

# Je Punkt der Suche werden MEHRERE Glieder eingereiht. Grund: 200 Epochen
# brauchen bei gemessenen 583 s/Epoche rund 32 h, das Limit auf stud liegt bei
# 24 h. Mit nur einem Glied je Punkt wuerde kein einziger Punkt jemals fertig —
# das naechste Glied begaenne einen anderen TAG, und der vorige bliebe fuer
# immer bei ~150 Epochen stehen. Ist ein Punkt schon fertig, beendet sich sein
# Folgeglied sofort (Pruefung auf `_final` im Jobskript), es kostet also nichts.
GLIEDER=${GLIEDER:-2}

reihe () {   # $1=TAG  $2=Beschriftung  $3=Variablen
    local id=""
    if ! gewuenscht "$1"; then
        echo "  $2  uebersprungen" >&2
        echo ""
        return
    fi
    for _g in $(seq 1 "$GLIEDER"); do
        id=$(sbatch --parsable $DEP --export=ALL,$3 run_job_length_hp.bash)
        echo "  $2  Glied $_g  ->  Job $id  ${DEP:+(nach ${DEP#*:})}" >&2
        DEP="--dependency=afterany:$id"
    done
    echo "$id"
}

# PUNKTE waehlt aus, welche Punkte eingereiht werden — noetig, wenn nur
# einzelne nachgeholt werden muessen (etwa nach einem Abbruch) und die
# uebrigen bereits laufen. Leer bedeutet: alle.
#   PUNKTE="LR1E4"           nur der Bezugspunkt
#   PUNKTE="LR3E4 CFG40"     zwei davon
PUNKTE=${PUNKTE:-}
gewuenscht () {
    [ -z "$PUNKTE" ] && return 0
    for p in $PUNKTE; do [ "$p" = "$1" ] && return 0; done
    return 1
}

echo "Hyperparameter-Kette${PUNKTE:+ (nur: $PUNKTE)}:"
A=$(reihe "LR1E4" "lr1e-4  (Bezug)" "TAG=LR1E4,LR=1e-4,CFG=2.0,PDROPLEN=0.1,EPOCHS=200,LENFREQ=linear")
B=$(reihe "LR3E4" "lr3e-4"          "TAG=LR3E4,LR=3e-4,CFG=2.0,PDROPLEN=0.1,EPOCHS=200,LENFREQ=linear")
C=$(reihe "CFG40" "cfg4.0"          "TAG=CFG40,LR=1e-4,CFG=4.0,PDROPLEN=0.1,EPOCHS=200,LENFREQ=linear")
D=$(reihe "LOGS10" "logscale1.0"     "TAG=LOGS10,LR=1e-4,CFG=2.0,LOGSCALE=1.0,PDROPLEN=0.1,EPOCHS=200,LENFREQ=linear")
E=$(reihe "PD025" "pdroplen0.25"    "TAG=PD025,LR=1e-4,CFG=2.0,PDROPLEN=0.25,EPOCHS=200,LENFREQ=linear")
echo
echo "Letztes Glied: $E"
