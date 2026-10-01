#!/bin/bash
#SBATCH -J best_of_n_10h
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Reduzierter Umfang der presvgd-Version (siehe run_job_best_of_n_full_presvgd.bash),
# um von den kalibrierten ~33-34h (voller Umfang, 9 Strategien) auf ~10h zu
# kommen: nur 3 Strategien statt 9 (ein Vertreter je Familie: niveau, ucb,
# eid_optuna_ideal_v2 -- mass-Familie ausgelassen), alles andere unveraendert
# (beide Repraesentationen, beide Replan-Schemata, alle 4 Wissensstufen, alle
# 25 Formen, n_candidates=30, --selection pre_svgd). Rechnerische Schaetzung:
# ~33-34h * 3/9 ~= 11h -- Zeitlimit auf 12h gesetzt (statt 24h), Puffer von
# ~1h fuer die Schaetzungenauigkeit. Eigener --out_tag, kein Ueberlapp mit den
# anderen best_of_30_*-Ordnern.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

srun --unbuffered python run_best_of_n_matrix.py \
    --representations particles,spectral \
    --strategies niveau_svgd25,ucb_tuned_svgd25,eid_optuna_ideal_v2 \
    --replan_schemes no_replan,replan_1_6 \
    --n_candidates 30 \
    --selection pre_svgd \
    --heuristic_families lse,ucb,eid \
    --spectral_ckpt ~/Master_thesis/thesis_architecture/checkpoints/cond_spectral_crossattn_ergodic_S256_nxi25_D384_flip0.0_final.pt \
    --out_tag best_of_30_10h_20260918
