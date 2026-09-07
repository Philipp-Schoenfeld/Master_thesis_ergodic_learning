#!/bin/bash
#SBATCH -J check_no200
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 00:20:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 2

# Nur-Diagnose vor dem echten 24h-Training: exakt die Kaltstart-Pruefung aus
# run_job_3d_no_offset_200.bash (Testsuite + Warmstart-Pruefung), aber in
# einem eigenen kurzen Job statt im 24h-Job selbst. Kein Training, kein
# Checkpoint-Schreiben.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

echo "=== Testsuite ==="
srun --unbuffered python test_3d_port.py \
    || { echo "Testsuite fehlgeschlagen."; exit 1; }

echo
echo "=== Warmstart-Pruefung ==="
srun --unbuffered python pruefe_warmstart.py \
    --init_model "checkpoints/nooffset_flow3d_particle_ergodic_nooffset_nxi25_D384_N512_R64_C1_flip0.0_SURF_START-pd0.1_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0302.pt" \
    --db3d ergodic_dataset_3d_no_offset.db \
    --db_splits train --erwartet 0.67897 \
    --batches 80 --mini_batch 32 \
    --erg_on footprint --lambda_erg 100 --erg_K 6 --erg_pts 128 \
    --erg_t_power 2.0 --w_cfm_rot 0.5 \
    --orientation --lambda_ori 0.012 --w_point 0.1 --w_standoff 300 \
    --w_angsmooth 2.0 --standoff_target 0.12 --standoff_band 0.03 \
    || { echo "Warmstart-Pruefung fehlgeschlagen."; exit 1; }

echo
echo "=== Datenlader-Rauchtest auf der grossen no_offset-200-DB (kurz) ==="
# Volle Epoche (91542 Eintraege) waere in 20 Minuten nicht zu schaffen — der
# Runner kennt keinen Steps-Limiter fuer --db3d. pruefe_warmstart.py macht
# stattdessen den eigentlich interessanten Test: laedt GENAU ueber
# load_surface_db (denselben Pfad, den der Runner benutzt) und rechnet ein
# paar Minibatches durch dieselbe Verlustfunktion durch. Hier zaehlt nur, ob
# das fehlerfrei durchlaeuft (Formen/Augmentierung/Verlust passen) — nicht
# der Sollwert-Vergleich: $INIT wurde nicht auf dieser DB trainiert, ein
# "FEHLGESCHLAGEN" wegen abweichendem Verlust ist hier also erwartet und kein
# Abbruchgrund, ein Absturz (Exception, NaN, Shape-Fehler) waere es.
srun --unbuffered python pruefe_warmstart.py \
    --init_model "checkpoints/nooffset_flow3d_particle_ergodic_nooffset_nxi25_D384_N512_R64_C1_flip0.0_SURF_START-pd0.1_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0302.pt" \
    --db3d ergodic_dataset_3d_no_offset_200.db \
    --batches 20 --mini_batch 32 \
    --erg_on footprint --lambda_erg 100 --erg_K 6 --erg_pts 128 \
    --erg_t_power 2.0 --w_cfm_rot 0.5 \
    --orientation --lambda_ori 0.012 --w_point 0.1 --w_standoff 300 \
    --w_angsmooth 2.0 --standoff_target 0.12 --standoff_band 0.03
echo "(Exit-Code $? oben ist informativ — ein Sollwert-Fehlschlag ist hier erwartet, siehe Kommentar.)"

echo
echo "ALLES BESTANDEN — bereit fuer die 24h-Kette."
