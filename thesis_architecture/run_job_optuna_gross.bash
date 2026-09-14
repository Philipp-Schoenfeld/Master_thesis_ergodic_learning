#!/bin/bash
#SBATCH -J optuna_gross
#SBATCH -p stud
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=SIGTERM@120
#SBATCH -o logs/optuna_gross-%j.out
#SBATCH -e logs/optuna_gross-%j.err
##SBATCH -C 'rtx3090'      # auf der stud-Partition verboten, bleibt auskommentiert

# Ergaenzende Optuna-Suche zur lokal laufenden Studie `ideal_v2` (Raum
# `ideal`, 12 Dimensionen). Dieser Lauf sucht im groesseren Raum `gross`
# (16 Dimensionen: zusaetzlich gp_variance, flow_steps, gp_res, max_obs,
# visit_bandwidth) mit deutlich breiterem Zufalls-Start (150 statt 50) --
# die lokale Suche war nach 126 Versuchen schon auf `eid`+`quantile`
# eingerastet (80/126), bevor die anderen sechs Phi-Modelle ausreichend
# gesehen wurden. Details und der Zusammenfuehr-Schritt nach Jobende stehen
# in exploration_optimierung/CLUSTER_RUN_ideal_v2_gross.md.
#
# Fortsetzbar wie die lokale Studie: `study.db` (SQLite) und `cache/`
# (JSON je Konfiguration+Seed, Inhalts-Hash-benannt) liegen unter
# exploration_optimierung/results/optuna/. Nach dem harten 24h-Limit mit
# `sbatch --dependency=afterany:<JOBID> run_job_optuna_gross.bash`
# weiterfuehren -- derselbe --study-Name laedt automatisch weiter.
#
# 16 CPU-Kerne wie in run_job_erkundung_opt.bash: die SVGD-Verfeinerung
# laeuft in Arbeitsprozessen und ist bei hohen Iterationszahlen der
# teuerste Teil.

set -e
cd ~/Master_thesis/thesis_architecture/
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

echo "Job $SLURM_JOB_ID auf $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --unbuffered python -u -m exploration_optimierung.optuna_search \
    --study ideal_v2_gross_cluster --space gross --seeds 3 \
    --trials 5000 --timeout_h 23 \
    --min_resource 4 --random_startup 150 --workers 14

echo "Ende $(date)"
