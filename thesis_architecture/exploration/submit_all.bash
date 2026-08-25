#!/bin/bash
# Reiht die drei Trainings ein und haengt die Auswertung mit einer
# Abhaengigkeit daran. Kein srun, kein direktes Ausfuehren der Job-Skripte —
# ausschliesslich sbatch.
#
#   bash submit_all.bash
#
# Die Auswertung startet automatisch, sobald ALLE drei Trainings erfolgreich
# beendet sind (afterok). Bricht eines am Zeitlimit ab, muss stattdessen ein
# Folge-Training mit --dependency=afterany:<ID> eingereiht werden und die
# Auswertung an dieses gehaengt werden.
set -e
cd ~/Master_thesis/thesis_architecture/exploration

ID_B=$(sbatch --parsable --job-name=expl_belief \
        --export=ALL,OBJECTIVE=belief    run_job_explorer.bash)
ID_S=$(sbatch --parsable --job-name=expl_segment \
        --export=ALL,OBJECTIVE=segment   run_job_explorer.bash)
ID_L=$(sbatch --parsable --job-name=expl_lookahead \
        --export=ALL,OBJECTIVE=lookahead run_job_explorer.bash)

echo "Trainings eingereiht:  belief=$ID_B  segment=$ID_S  lookahead=$ID_L"

ID_E=$(sbatch --parsable --dependency=afterok:${ID_B}:${ID_S}:${ID_L} \
        run_job_eval_explorer.bash)
echo "Auswertung eingereiht: $ID_E  (startet nach allen drei Trainings)"
echo
squeue -u "$USER" -o '%.8i %.16j %.9T %.10M %.11l %R'
