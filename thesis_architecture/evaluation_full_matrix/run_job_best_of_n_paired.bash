#!/bin/bash
#SBATCH -J best_of_n_paired
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Voller Umfang mit --paired_svgd (neu, 2026-09-18, siehe run_best_of_n_matrix.py
# --paired_svgd Hilfetext und variant_runner.PAIRED_SVGD_FAMILIES): alle 4
# Paar-Familien (niveau, ucb, mass, eid_tuned) teilen sich je (rep, cond) EINEN
# 30-Kandidaten-Pool fuer no_replan, werten den Sieger aber sowohl roh als
# auch mit dem *_svgd25-Budget verfeinert aus -- statt zwei unabhaengige Pools
# zu ziehen. eid_optuna_ideal_v2 bleibt eigenstaendig (kein Partner zum
# Teilen). replan_1_6 bleibt fuer alle 9 Strategien unabhaengig (siehe
# Kommentar bei PAIRED_SVGD_FAMILIES: Glaubens-Divergenz nach Runde 1 macht
# Teilen dort wissenschaftlich unsauber). random_walk: 30 Kandidaten, bester
# wird roh (0 SVGD) UND mit 500 SVGD-Iterationen verfeinert ausgewertet.
#
# Smoke-getestet (Job 155582, 1 Form, n_candidates=3): korrekte Ordnerstruktur,
# keine Duplikate/Fehler, svgd0 vs. svgd25 zeigen die erwartete Verbesserung
# (25.25 -> 19.29 E_ergodic_total) an genau demselben Basis-Kandidaten.
#
# Zeitschaetzung: ~28-32h (kalibrierte ~33-34h fuer ungepaarte 9-Strategien-
# pre_svgd-Vollmatrix minus ~5-15% durch das Teilen im no_replan-Anteil).
# Als Kette aus zwei 24h-Jobs eingereicht (sbatch --dependency=afterany):
# 2x24h=48h deckt die Schaetzung mit Puffer. Idempotent/resumable je Zeile
# wie alle anderen Skripte hier -- derselbe Befehl kann beliebig oft erneut
# abgeschickt werden.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

srun --unbuffered python run_best_of_n_matrix.py \
    --representations particles,spectral \
    --strategies niveau_svgd0,niveau_svgd25,ucb_tuned_svgd0,ucb_tuned_svgd25,mass_tuned_svgd0,mass_tuned_svgd25,eid_tuned_svgd0,eid_tuned_svgd25,eid_optuna_ideal_v2 \
    --replan_schemes no_replan,replan_1_6 \
    --n_candidates 30 \
    --selection pre_svgd \
    --paired_svgd \
    --heuristic_families lse,ucb,mass,eid \
    --spectral_ckpt ~/Master_thesis/thesis_architecture/checkpoints/cond_spectral_crossattn_ergodic_S256_nxi25_D384_flip0.0_final.pt \
    --out_tag best_of_30_paired_20260918
