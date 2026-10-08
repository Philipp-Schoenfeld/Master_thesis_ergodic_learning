#!/bin/bash
#SBATCH -J mission_eval_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Testlauf der Replanning-Missions-Auswertung auf der Cluster-GPU, Grundlage
# fuer die Zeitschaetzung des vollen Laufs (run_job_mission_eval.bash):
#   1. Selbsttest inkl. Teil 7 (echter CFM-Checkpoint, braucht die GPU),
#   2. Mini-Lauf mit der ECHTEN Batch-Groesse (alle 25 Formen x 30 Kandidaten
#      x 1500 SVGD-Schritte), aber nur 1 Wissensstufe x 1 Strategie x alle
#      3 Methoden (3 von 27 Sets) und hoechstens 3 Runden -- liefert die
#      Sekunden pro Runde je Methode (Log-Zeilen "SVGD batch ... s") und
#      den Plattenbedarf pro Runde (du am Ende).
# Ergebnis landet im Ordner des vollen Laufs (results/mission_eval_20261006):
# die Testrunden werden dort beim vollen Lauf per Resume weiterverwendet.

OUT_TAG=mission_eval_20261006

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[test] $(date) Selbsttest"
srun --unbuffered python -u test_mission_eval.py || { echo "[test] Selbsttest FEHLGESCHLAGEN"; exit 1; }

echo "[test] $(date) Mini-Lauf"
T0=$(date +%s)
srun --unbuffered python -u run_mission_eval.py \
    --out_tag ${OUT_TAG} \
    --conditions half_known \
    --strategies lse \
    --max_rounds 3 \
    --workers 3 \
    --parallel_sets 4
echo "[test] $(date) Mini-Lauf Exit-Code $? nach $(( $(date +%s) - T0 )) s"

du -sh results/${OUT_TAG}
du -sh results/${OUT_TAG}/shards/*

echo "[test] $(date) Plots"
srun --unbuffered python -u plot_mission_eval.py --out_tag ${OUT_TAG}
echo "[test] $(date) fertig"
