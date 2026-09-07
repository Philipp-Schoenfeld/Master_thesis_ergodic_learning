#!/bin/bash
#SBATCH -J 3d_no_offset200
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 24:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Dasselbe Training wie run_job_3d_no_offset.bash (siehe dort fuer die
# ausfuehrliche Begruendung von LR/Warmup/Zielfunktion), jetzt auf der
# grossen, finalen 200-Flaechen-Datenbank OHNE OFFSET
# (ergodic_dataset_3d_no_offset_200.db, 91542 Eintraege, 200 Traegerflaechen,
# 1307 Formen — dieselbe Grossenordnung wie die alte ergodic_dataset_3d.db,
# aber mit dem Offset-Fix aus der kleinen Datenbank). Anders als die kleine
# no_offset-DB hat diese wieder das volle Schema (gruppe, view_x/y/z/view_id),
# --db_splits bleibt deshalb auf der Standardeinstellung des Runners
# (train/val_form/val_flaeche/val_beides).
#
# Warmstart: NICHT vom alten flaechen200-Checkpoint (der ist mit-Offset-
# Vorlaeufer), sondern vom Checkpoint, der dieses Wochenende auf der KLEINEN
# no_offset-Datenbank fertig trainiert wurde (10 Grundformen, 300 Epochen,
# Verlust 0.852 -> 0.679). Eigenstaendiges Training: --init_model, frischer
# Optimierer und Lernplan, nicht --resume.
#
# ── Ressourcen ──────────────────────────────────────────────────────────────
# 91542 Eintraege sind in derselben Groessenordnung wie die 82415 des
# flaechen200-Laufs (dort gemessen ~4-7 GB bei 16G Anforderung) — deshalb
# --mem=16G statt der 8G der kleinen no_offset-DB. Nach dem ersten
# Kettenglied mit
#   sacct -j <ID> -o JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem
# pruefen und bei Bedarf fuer die Folgeglieder nachziehen.
#
# ── Zeitbudget ──────────────────────────────────────────────────────────────
# flaechen200 (82415 Eintraege, aehnliche Groessenordnung) schaffte auf einem
# langsamen Knoten (dgx-station, V100) nur ~111 von 300 Epochen in 24h. Diese
# Kette reiht deshalb vorsorglich mehrere 24h-Glieder; ein Glied, das nichts
# mehr zu tun findet (--resume auf einen bereits "_final"-Stand), beendet
# sich in unter zwei Minuten und kostet praktisch nichts.
#
# Folgejob anhaengen:
#   sbatch --dependency=afterany:<JOBID> run_job_3d_no_offset_200.bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/3D_ergodic_learning

export MPLBACKEND=Agg

RUN_TAG=${RUN_TAG:-nooffset200}
DB3D=${DB3D:-ergodic_dataset_3d_no_offset_200.db}
INIT=${INIT:-checkpoints/nooffset_flow3d_particle_ergodic_nooffset_nxi25_D384_N512_R64_C1_flip0.0_SURF_START-pd0.1_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0302.pt}

PAT="checkpoints/nooffset200_flow3d_particle_ergodic_${RUN_TAG}_*_ep*.pt"
LATEST=$(ls -t $PAT 2>/dev/null | head -1)

if [ ! -f "$LATEST" ]; then
    echo "=== Kaltstart: Testsuite ==="
    srun --unbuffered python test_3d_port.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }

    echo
    echo "=== Kaltstart: Warmstart-Pruefung ==="
    # $INIT wurde auf der KLEINEN no_offset-Datenbank trainiert, gleichverteilt
    # (kein --mix — die kleine DB hat gar keine gruppe-Spalte). Die Pruefung
    # muss GENAU diese Datenbank und GENAU diese Zielfunktion nachbilden, NICHT
    # ergodic_dataset_3d_no_offset_200.db (der neuen Zieldatenbank dieses
    # Laufs). Siehe run_job_3d_no_offset.bash zum 04.09.-Vorfall, warum eine
    # falsche Ziehverteilung hier zu einem Fehlschlag fuehrt, der nichts mit
    # einem defekten Checkpoint zu tun hat.
    #
    # --erwartet explizit gesetzt: der gespeicherte 'loss' in $INIT ist faelsch-
    # licherweise 0.0 (ein Speicherfehler des Folgejobs 149544/149545, der nach
    # Epoche 300 noch eine Resume-Runde antrat und dabei mit einer nicht neu
    # gesetzten Verlustvariable speicherte — die Gewichte selbst sind davon
    # nicht betroffen). Der tatsaechliche Endverlust steht im Rohlog von Job
    # 149543 bei Epoche 300/300: loss=0.67897.
    srun --unbuffered python pruefe_warmstart.py \
        --init_model "$INIT" --db3d ergodic_dataset_3d_no_offset.db \
        --db_splits train --erwartet 0.67897 \
        --batches 80 --mini_batch 32 \
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
echo "=== Training: 200 Flaechen ohne Offset, eigenstaendig, volle Epochen ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

srun --unbuffered python flow_matching_runner_particles.py \
    $START_ARG \
    --db3d "$DB3D" \
    --run_tag "$RUN_TAG" \
    --save_model checkpoints/nooffset200.pt \
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
