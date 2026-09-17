#!/bin/bash
#SBATCH -J best_of_n_full
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Voller Best-of-30-Umfang, "30x komplett unabhaengig" (--selection post_svgd):
# 9 Strategien (die im bestehenden full_run_with_optuna_ideal_v2_20260916
# verwendeten, ohne das nicht-repraesentative mi_optuna_gross) x 2
# Repraesentationen (particles, spectral -- spectral lief dort noch nie mit)
# x 2 Replan-Schemata x 30 Kandidaten x 4 Wissensstufen x 25 Formen, plus die
# Heuristik-/Linear-Waypoint-Varianten (Phase 2).
#
# Idempotent/resumable je einzelner Zeile (run_best_of_n_matrix.py::row_done
# prueft vor jeder teuren Generierung, ob raw/**/metrics.json schon existiert)
# -- derselbe Befehl kann beliebig oft erneut abgeschickt werden, jeder Lauf
# setzt automatisch dort fort, wo der vorige durch das 24h-Zeitlimit (oder
# das SIGTERM@120 davor, das die Tabellen/Plots aus allem bisher Gecachten
# neu baut und sauber beendet) unterbrochen wurde. Fuer die von Philipp
# gewuenschte Kette aus drei Jobs: dieses Skript dreimal per
# `sbatch --dependency=afterany:<JOBID>` einreichen, siehe CLAUDE.md.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

srun --unbuffered python run_best_of_n_matrix.py \
    --representations particles,spectral \
    --strategies niveau_svgd0,niveau_svgd25,ucb_tuned_svgd0,ucb_tuned_svgd25,mass_tuned_svgd0,mass_tuned_svgd25,eid_tuned_svgd0,eid_tuned_svgd25,eid_optuna_ideal_v2 \
    --replan_schemes no_replan,replan_1_6 \
    --n_candidates 30 \
    --selection post_svgd \
    --heuristic_families lse,ucb,mass,eid \
    --out_tag best_of_30_full_20260918
