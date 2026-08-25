#!/bin/bash
#SBATCH -J explorer
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 22:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Amortisierte Planung ueber Glaubenszustaenden — ein Skript fuer alle drei
# Trainingsziele, gewaehlt ueber die Umgebungsvariable OBJECTIVE.
#
#   OBJECTIVE=belief     volle Trajektorie aus UCB-Glaubensdichte   -> A, D, E
#   OBJECTIVE=segment    kurzer Abschnitt mit festem Startpunkt     -> C
#   OBJECTIVE=lookahead  belief + differenzierbare Vorausschau      -> B
#
# Warum drei und nicht fuenf: A, C, D und E unterscheiden sich nicht im Modell,
# sondern in der Missionsschleife zur Laufzeit. Fuenf Trainings waeren drei
# identische Kopien und drei mal 22 h GPU-Zeit ohne Erkenntnisgewinn.
#
# Einreihen (nicht dieses Skript direkt starten, nur ueber sbatch):
#   OBJECTIVE=belief    sbatch --export=ALL,OBJECTIVE=belief    run_job_explorer.bash
#   OBJECTIVE=segment   sbatch --export=ALL,OBJECTIVE=segment   run_job_explorer.bash
#   OBJECTIVE=lookahead sbatch --export=ALL,OBJECTIVE=lookahead run_job_explorer.bash
#
# Danach die Auswertung mit Abhaengigkeit auf alle drei, siehe submit_all.bash.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

export MPLBACKEND=Agg

OBJECTIVE=${OBJECTIVE:-belief}
EPOCHS=${EPOCHS:-400}
N_STATES=${N_STATES:-6000}
N_SHAPES=${N_SHAPES:-500}

echo "=== Ziel: $OBJECTIVE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Der Glaubens-Cache haengt nicht vom Modell ab und wird zwischen Laeufen
# wiederverwendet. Beim ersten Lauf kostet er einige Minuten GP-Inversionen.
LATEST=$(ls -t checkpoints/explorer_${OBJECTIVE}_*_ep*.pt 2>/dev/null | head -1)
RESUME=""
if [ -f "$LATEST" ]; then
    echo "Setze fort von: $LATEST"
    RESUME="--resume $LATEST"
else
    echo "Kaltstart — zuerst die Testsuite."
    srun --unbuffered python test_exploration.py \
        || { echo "Testsuite fehlgeschlagen — Abbruch vor dem Training."; exit 1; }
fi

# lookahead haelt den GP live und braucht deshalb kleinere Batches:
# pro Sample eine Cholesky-Zerlegung im Vorwaertspass.
EXTRA=""
if [ "$OBJECTIVE" = "lookahead" ]; then
    EXTRA="--mini_batch 32 --n_probe 32 --w_unc 0.01"
fi

srun --unbuffered python training/train_explorer.py \
    $RESUME \
    --objective $OBJECTIVE \
    --n_shapes $N_SHAPES \
    --n_states $N_STATES \
    --grid_res 32 \
    --kappa 0.0 6.0 \
    --nxi 25 --seg_nxi 10 \
    --D 384 --n_particles 192 \
    --erg_K 8 --erg_pts 128 --metric fourier \
    --w_erg 100.0 \
    --target_length 4.0 \
    --epochs $EPOCHS \
    --mini_batch 64 \
    --save_every 20 \
    --use_wandb \
    $EXTRA
