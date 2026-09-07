#!/bin/bash
#SBATCH -J flaechen200
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Feinabgleich, zweiter Sprung: von 79 auf 200 Zielflaechen. Warmstart aus dem
# ep0250-Endstand des ersten Sprungs (7 -> 79 Flaechen), auf der neuen, um vier
# selbstgemachte Kategorien erweiterten Datenbank (siehe LIZENZEN.md):
# Organismen (Metaball/Marching-Cubes), Baugruppen (boolesche Verknuepfung),
# Superquadriken, Woerter. 91.950 Eintraege statt bisher 34.693 (2,65x).
#
# ── Ressourcen: hochgerechnet, nicht (noch) gemessen ───────────────────────
# --mem=16G: Der Vorlaeufer-Job (79 Flaechen, 34.693 Eintraege) brauchte
#            gemessen ~4 GB im Betrieb bei angefordertem --mem=12G (1,21 GB
#            Partikelstapel+Zieltensor, ~2,5 GB Torch/CUDA-Kontext). Linear auf
#            2,65x mehr Eintraege hochgerechnet: ~3,5-4,5 GB Datenlader-Anteil,
#            macht zusammen mit den Fixkosten ~6-7 GB. 16G ist Sicherheitsmarge
#            fuer eine Hochrechnung statt einer Messung — nach diesem ersten
#            Segment per `sacct -j <ID> -o JobID,Elapsed,MaxRSS,ReqMem` pruefen
#            und fuer die Folgeglieder ggf. senken.
# GPU-Speicher aendert sich durch die groessere Datenbank NICHT: trainiert wird
# mini-batchweise, die GPU sieht pro Schritt nur einen Batch. Der VRAM-Bedarf
# haengt an Modellgroesse/--D/--n_particles/--grid_res/--mini_batch — die sind
# alle unveraendert gegenueber dem Vorlaeufer-Job.
# -c 2: wie beim Vorlaeufer, die Arbeit liegt auf der GPU.
#
# ── Zeitbudget: bis Mittwochvormittag ───────────────────────────────────────
# Epochendauer beim Vorlaeufer (34.693 Eintraege): ~500 s/Epoche. Linear auf
# 91.950 Eintraege hochgerechnet: ~1.325 s/Epoche (~22 min) — eine Epoche ist
# ein voller gewichteter Durchlauf durch die Daten, die Eintragszahl bestimmt
# die Dauer direkt. Drei Kettenglieder à 22 h (66 h) reichen von Sonntag bis
# Mittwochvormittag. --epochs 300 ist bewusst hoeher gesetzt als die in drei
# Segmenten realistisch erreichbaren ~175-180 Epochen: die Kette soll nicht
# vorzeitig auf "_final" laufen und stehenbleiben, sondern bis Mittwoch
# durchlaufen. Der am Mittwoch vorgefundene Zwischenstand (nicht "_final") ist
# der Endstand dieser Runde — kein Fehlschlag, sondern das erwartete Ergebnis
# eines zeitbudgetierten statt konvergenzbudgetierten Laufs.
#
# ── Warmstart: --init_model, nicht --resume ─────────────────────────────────
# Wie beim Vorlaeufer-Sprung (7 -> 79 Flaechen): neuer Lauf, nur die Gewichte
# wandern mit, frisches AdamW und frischer Lernplan. Ab dem zweiten
# Kettenglied normal per --resume fortsetzen (siehe Logik unten).
#
# Folgejob anhaengen:
#   sbatch --dependency=afterany:<JOBID> run_job_3d_flaechen200.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

RUN_TAG=${RUN_TAG:-flaechen200}
DB3D=${DB3D:-ergodic_dataset_3d.db}
INIT=${INIT:-checkpoints/startfl_flow3d_particle_ergodic_startfl22_nxi25_D384_N512_R64_C1_flip0.0_SURF_START-pd0.1_MIX-ebene=0.25_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0250.pt}

PAT="checkpoints/flaechen200_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

if [ ! -f "$LATEST" ]; then
    echo "=== Kaltstart: Testsuite ==="
    srun --unbuffered python test_3d_port.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

    echo
    echo "=== Kaltstart: Warmstart-Pruefung ==="
    # Muss auf der Datenbank pruefen, auf der $INIT tatsaechlich trainiert
    # wurde (79 Flaechen, 38.645 Eintraege) — NICHT auf ergodic_dataset_3d.db
    # (jetzt die neue 200-Flaechen-Version) und NICHT auf
    # ergodic_dataset_3d_alt.db (das ist die Referenz fuer einen frueheren,
    # anderen Sprung). Andernfalls schlaegt der Vergleich zwingend fehl: der
    # gespeicherte Endverlust des Checkpoints stammt von einer anderen
    # Datenverteilung als die, gegen die hier gemessen wird. Genau das ist am
    # 30.08. passiert (Abweichung 129,81 % statt der erwarteten <10 %) und hat
    # die ganze Kette in unter drei Minuten durchfallen lassen, bevor es
    # aufgefallen ist.
    # Muss dieselbe Zielfunktion bilden wie das Training unten (erg_on,
    # w_cfm_rot, UND den vollen Orientierungsterm) — pruefe_warmstart.py
    # verglich frueher ohne Orientierungsterm (die Konfiguration von
    # ep1750), waehrend $INIT (ep0250) bereits MIT Orientierungsterm
    # trainiert wurde. Diese Diskrepanz allein hob den Verlust von 0,93 auf
    # ueber 2,2 und liess die Pruefung am 30.08. zweimal falsch durchfallen,
    # bevor pruefe_warmstart.py um --orientation/--lambda_ori/... erweitert
    # wurde.
    srun --unbuffered python pruefe_warmstart.py \
        --init_model "$INIT" --db3d ergodic_dataset_3d_79flaechen.db \
        --batches 40 --mini_batch 32 \
        --erg_on footprint --lambda_erg 100 --erg_K 6 --erg_pts 128 \
        --erg_t_power 2.0 --w_cfm_rot 0.5 \
        --orientation --lambda_ori 0.012 --w_point 0.1 --w_standoff 300 \
        --w_angsmooth 2.0 --standoff_target 0.12 --standoff_band 0.03 \
        || { echo "Warmstart-Pruefung fehlgeschlagen — Abbruch."; exit 1; }
fi

if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    START_ARG="--resume $LATEST"
else
    echo "Kaltstart, Warmstart aus: $INIT"
    START_ARG="--init_model $INIT"
fi

echo
echo "=== Training: 200 Flaechen + Startpunkt, 22 h ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

srun --unbuffered python flow_matching_runner_particles.py \
    $START_ARG \
    --db3d "$DB3D" \
    --run_tag "$RUN_TAG" \
    --save_model checkpoints/flaechen200.pt \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0.1 \
    --mix ebene=0.25 --ebene_flach_anteil 0.5 \
    --erg_on footprint \
    --lambda_erg 100 --erg_K 6 --erg_pts 128 --erg_t_power 2.0 \
    --lambda_ori 0.012 --w_cfm_rot 0.5 \
    --w_point 0.1 --w_standoff 300 --w_angsmooth 2.0 \
    --standoff_target 0.12 --standoff_band 0.03 \
    --D 384 --n_particles 512 --grid_res 64 \
    --copies_per_char 1 --p_flip 0.0 \
    --lr 2e-5 --warmup_epochs 50 --lr_min 1e-6 \
    --epochs 300 --mini_batch 128 \
    --save_every 10 --viz_every 20 \
    --n_holdout_viz 10 \
    --use_wandb --wandb_project flow3d-surfaces
