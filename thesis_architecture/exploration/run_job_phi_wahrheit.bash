#!/bin/bash
#SBATCH -J phi_wahrheit
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# Phi-Kreuz mit ECHTEM Vorwissen, startpunkt-konditioniertem Netz
# und Variante D zusaetzlich zu den uebrigen Missionen.
#
# Drei Aenderungen gegenueber run_job_phi_muster.bash:
#
# 1. --prior_mode wahrheit. Im bekannten Gebiet liegt die Grundwahrheit exakt
#    vor (sigma = 0), ausserhalb gar kein Wissen (mu = 0, sigma = 1). Bisher
#    wurden dort nur 60 Punktmessungen gezogen und der GP interpolierte.
#
# 2. Das Netz ist der startpunkt-konditionierte Lauf start_lang. Damit kann
#    Variante D dort weiterplanen, wo der Roboter gerade steht — der bisherige
#    `nearest`-Einstieg samt Anfahrt war nur ein Behelf fuer ein Netz ohne
#    Start-Eingang.
#
# 3. glaube-D laeuft zusaetzlich zu den sechs bisherigen Missionen.
#    30 Runden zu je einem Zehntel der geplanten Bahn ergeben rund drei
#    Bahnlaengen und sind damit mit glaube-R (3 volle Runden) vergleichbar.
#
# 3 Muster x 7 Zieldichten x 12 Formen x 7 Missionen.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

CKPT=$(ls -t ../checkpoints/*START_FLAT400_L500*_ep*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "Kein start_lang-Checkpoint gefunden."; exit 1; fi
echo "Checkpoint: $CKPT"

for MUSTER in haelfte quadranten loch; do
  for PHI in ucb stretch mass ei lse mi eid; do
    OUT="results/phi_wahrheit/${MUSTER}_${PHI}"
    if [ -f "$OUT/metriken.csv" ]; then echo "[skip] $OUT"; continue; fi
    echo "=== $MUSTER / $PHI ==="
    srun --unbuffered python -u apply_cfm_belief.py \
        --ckpt "$CKPT" \
        --shapes 12 --rounds 3 \
        --missions orakel glaube-1 glaube-R zweistufig glaube-D B-warm maeher \
        --phi_model "$PHI" \
        --prior_pattern "$MUSTER" \
        --prior_mode wahrheit \
        --gp_noise 0.05 \
        --d_rounds 30 --d_execute_frac 0.10 --d_join netz \
        --save_paths \
        --out_dir "$OUT"
  done
done
echo "Fertig: $(date)"
