#!/bin/bash
#SBATCH -J mission_eval
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Replanning-Missions-Auswertung (Cluster-Pendant zu run_mission_eval_full.cmd):
# 25 Holdout-Formen x 3 Wissensstufen (half_known, ten_samples, none_known)
# x 3 Zieldichte-Strategien (lse, ucb, eid) x 3 Init-Methoden (cfm,
# random_walk, linear) = 27 Missions-Sets; pro Runde 30 Kandidaten x 1500
# SVGD-Schritte, beste Bahn eine Laengeneinheit fahren, bis 99 % der
# Wahrheitsmasse ueberstrichen sind (max. 40 Runden).
#
# Ablauf: erst der Selbsttest (bricht bei Fehler ab), dann der volle Lauf,
# dann die Plots/Tabellen. Resumable je (Form, Runde): jede Runde wird in
# einer SQLite-Transaktion gespeichert, ein erneutes Einreichen desselben
# Skripts setzt dort fort. --time_budget_h 22 oeffnet nach 22 h keine neuen
# Runden mehr, damit der Job sauber vor dem 24-h-Limit endet; ein Folge-Job
# per `sbatch --dependency=afterany:<JOBID> run_job_mission_eval.bash` macht
# den Rest.
#
# Ressourcen gemessen im Testjob 162387 (3 Sets, V100): ~1,2 Kerne im Mittel,
# MaxRSS 2,9 GB -> Hauptprozess + 2 Pool-Worker = -c 3, 8G (Puffer fuer 4
# offene Sets). Engpass ist die CFM-Planung (~6,7 s je Form und Runde).
# Platzbedarf: ~18 MB je Set und Runde (Testjob), d. h. bis ~20 GB unter
# results/mission_eval_20261006 (jeder
# SVGD-Zustand jedes Kandidaten wird gespeichert).

OUT_TAG=mission_eval_20261006

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) Selbsttest"
srun --unbuffered python -u test_mission_eval.py || { echo "[job] Selbsttest FEHLGESCHLAGEN - kein voller Lauf"; exit 1; }

echo "[job] $(date) voller Lauf -> results/${OUT_TAG}"
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --workers 2 \
    --parallel_sets 4 \
    --time_budget_h 22
rc=$?
echo "[job] $(date) Lauf beendet mit Exit-Code ${rc}"

echo "[job] $(date) Plots/Tabellen"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${OUT_TAG}
