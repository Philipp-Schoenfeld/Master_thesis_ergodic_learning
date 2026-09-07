#!/bin/bash
#SBATCH -J 3d_no_offset
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Eigenstaendiges Training auf der neuen, finalen 3D-Datenbank ohne Offset
# (ergodic_dataset_3d_no_offset.db, 7750 Eintraege ueber 10 Grundformen:
# bunny, ebene_diagonal/flach/gekippt, ei, kegel, kugel, prisma, torus,
# wuerfel). Warmstart aus dem 200-Flaechen-Zwischenstand (flaechen200,
# ep0111), aber NICHT dessen Fortsetzung: eigener Optimierer, eigener
# Lernplan, --init_model statt --resume beim ersten Kettenglied.
#
# Die neue Datenbank hat ein anderes Schema als ergodic_dataset_3d.db: keine
# `gruppe`- und keine `view_x/y/z/view_id`-Spalten (nur EIN Blickwinkel je
# Form, keine Gruppen-Mischung), aber `start_pos` ist vorhanden — die
# Startpunkt-Konditionierung traegt also mit. --mix entfaellt deshalb (jede
# Zeile faellt sonst auf die Gruppe "unbekannt", eine Gewichtung waere
# wirkungslos) und --db_splits muss auf train/val umgestellt werden (die
# 4-Wege-Splits val_form/val_flaeche/val_beides existieren hier nicht).
#
# ── Ressourcen: hochgerechnet aus der gemessenen Groessenordnung ───────────
# 7750 Eintraege sind rund 9% der 82415 des flaechen200-Laufs (dort gemessen
# ~4 GB bei 16G Anforderung, Sicherheitsmarge fuer eine Hochrechnung). Linear
# herunterskaliert bleiben die Fixkosten (Torch/CUDA-Kontext ~2,5 GB), der
# datenabhaengige Anteil schrumpft auf einen Bruchteil. 8G ist ausreichend
# Puffer, ohne das Kontingent unnoetig zu blockieren — nach dem ersten
# Kettenglied mit `sacct -j <ID> -o JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem`
# pruefen und bei Bedarf fuer die Folgeglieder nachziehen.
# -c 2: wie bei allen 3D-Laeufen liegt die Arbeit auf der GPU.
#
# Folgejob anhaengen:
#   sbatch --dependency=afterany:<JOBID> run_job_3d_no_offset.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

RUN_TAG=${RUN_TAG:-nooffset}
DB3D=${DB3D:-ergodic_dataset_3d_no_offset.db}
INIT=${INIT:-checkpoints/flaechen200_flow3d_particle_ergodic_flaechen200_nxi25_D384_N512_R64_C1_flip0.0_SURF_START-pd0.1_MIX-ebene=0.25_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0111.pt}

PAT="checkpoints/nooffset_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

if [ ! -f "$LATEST" ]; then
    echo "=== Kaltstart: Testsuite ==="
    srun --unbuffered python test_3d_port.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

    echo
    echo "=== Kaltstart: Warmstart-Pruefung ==="
    # $INIT (flaechen200, ep0111) wurde auf ergodic_dataset_3d.db trainiert,
    # mit erg_on footprint + vollem Orientierungsterm UND --mix ebene=0.25
    # --ebene_flach_anteil 0.5 — die Pruefung muss auf GENAU dieser
    # Datenbank, GENAU dieser Zielfunktion UND GENAU dieser Ziehverteilung
    # laufen, nicht auf ergodic_dataset_3d_no_offset.db (der neuen
    # Zieldatenbank dieses Laufs) und nicht gleichverteilt. Am 04.09. fiel
    # die Pruefung ohne --mix bei 12-13% Abweichung durch (Toleranzband
    # ~5-12%), obwohl --init_model alle Gewichte bitgenau lud; unter der
    # Ziehverteilung des Trainings (ebene mit 25% statt ihres natuerlichen
    # Anteils von 5,8%) liegt derselbe Checkpoint nur noch 1,1% neben seinem
    # aufgezeichneten Verlust. Ohne --mix vergleicht die Pruefung eine
    # andere Stichprobe als die, aus der der aufgezeichnete Verlust stammt,
    # und faellt durch einen Sprung durch, der nichts mit einem defekten
    # Checkpoint zu tun hat (siehe auch Kommentar in
    # run_job_3d_flaechen200.bash zum 30.08.-Vorfall — derselbe Fehlertyp:
    # Pruefung und Training vergleichen zwei verschiedene Groessen).
    srun --unbuffered python pruefe_warmstart.py \
        --init_model "$INIT" --db3d ergodic_dataset_3d.db \
        --batches 80 --mini_batch 32 \
        --mix ebene=0.25 --ebene_flach_anteil 0.5 \
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
echo "=== Training: neue Datenbank ohne Offset, eigenstaendig, volle Epochen ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

srun --unbuffered python flow_matching_runner_particles.py \
    $START_ARG \
    --db3d "$DB3D" \
    --db_splits train val \
    --run_tag "$RUN_TAG" \
    --save_model checkpoints/nooffset.pt \
    --orientation --frame_mode lookat --rot_full \
    --start_cond --p_drop_start 0.1 \
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
