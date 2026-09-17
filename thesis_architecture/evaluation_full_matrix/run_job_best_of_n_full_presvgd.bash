#!/bin/bash
#SBATCH -J best_of_n_full_presvgd
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# "Kurze" Version von run_job_best_of_n_full.bash: exakt derselbe Umfang (9
# Strategien x 2 Repraesentationen x 2 Replan-Schemata x 30 Kandidaten x 4
# Wissensstufen x 25 Formen, plus Heuristik/Linear), aber --selection
# pre_svgd statt post_svgd: pro Entscheidungspunkt werden 30 Netz-Kandidaten
# in einem Batch erzeugt, VOR SVGD anhand der Wahrheit bewertet, und nur der
# beste wird SVGD-verfeinert (statt alle 30 komplett unabhaengig zu
# verfeinern und danach zu vergleichen). Gemessen ~2.25x schneller als
# run_job_best_of_n_full.bash (siehe variant_runner.py, Kommentar ueber
# REPRESENTATIONS_BEST_OF_N) -- reale kalibrierte Schaetzung bei GLEICHEM
# Umfang: ~33-34h, NICHT 10h (zwei direkte lokale Messungen an diesem
# Code stimmen auf ~75-77h fuer die post_svgd-Version ueberein, /2.25 ergibt
# die ~33-34h). Falls eine einzelne <24h-Job-Laufzeit gewuenscht ist, dafuer
# den Umfang reduzieren (z.B. --representations particles, ohne spectral:
# ~16-17h) -- vor dem Start im Skript unten anpassen.
#
# Trade-off der Vorab-Auswahl (siehe variant_runner.py-Kommentar im Detail):
# waehlt anhand der *unverfeinerten* Netz-Ausgabe, nicht des fertigen
# Ergebnisses -- nur eine gute Naeherung, solange SVGD die Rangfolge nicht
# stark veraendert (sollte bei den hier verwendeten kleinen svgd_iters von
# 0/25/50 gelten, aber nicht gegen die post_svgd-Version verifiziert).
# Ausserdem: da nur EINE finale Bahn pro Testfall entsteht, zeigt der
# Mittelwert/Std-Marker im Diagramm bei dieser Version die Streuung der 30
# *unverfeinerten* Rohkandidaten (coverage_vs_truth), nicht der 30 fertigen
# Bahnen wie bei der post_svgd-Version.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

srun --unbuffered python run_best_of_n_matrix.py \
    --representations particles,spectral \
    --strategies niveau_svgd0,niveau_svgd25,ucb_tuned_svgd0,ucb_tuned_svgd25,mass_tuned_svgd0,mass_tuned_svgd25,eid_tuned_svgd0,eid_tuned_svgd25,eid_optuna_ideal_v2 \
    --replan_schemes no_replan,replan_1_6 \
    --n_candidates 30 \
    --selection pre_svgd \
    --heuristic_families lse,ucb,mass,eid \
    --out_tag best_of_30_full_presvgd_20260918
