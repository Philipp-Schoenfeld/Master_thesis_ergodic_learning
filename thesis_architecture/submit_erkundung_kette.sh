#!/bin/bash
# Job-Kette fuer die Erkundungs-Optimierung.
#
# Der volle Suchraum (16 Parameterpunkte, 2 Startwerte, 768 Auswertungen)
# braucht mehr als die 24 h, die ein einzelner Job auf der stud-Partition
# laufen darf. Deshalb werden mehrere Jobs mit `afterany` verkettet: jeder
# rechnet 23 h, beendet sich sauber, und der naechste setzt ueber den
# Zwischenspeicher fort. `afterany` statt `afterok`, damit die Kette auch dann
# weiterlaeuft, wenn ein Job am Zeitlimit haengt statt sauber zu enden.
#
#   ./submit_erkundung_kette.sh 3      # drei Glieder einreihen
set -e
N=${1:-3}
JOB=$(sbatch --parsable run_job_erkundung_opt.bash)
echo "Glied 1: $JOB"
for i in $(seq 2 "$N"); do
    JOB=$(sbatch --parsable --dependency=afterany:"$JOB" run_job_erkundung_opt.bash)
    echo "Glied $i: $JOB"
done
echo
echo "Stand ansehen (auf dem Cluster, ohne Job):"
echo "  python -m exploration_optimierung.optimize --mode beide --search voll \\"
echo "      --param_points 16 --seeds 2 --status"
