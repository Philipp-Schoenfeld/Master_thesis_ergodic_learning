#!/bin/bash
#SBATCH -J cfm_belief
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4

# Das trainierte CFM+ErgLoss-Netz auf glaubenskonditionierte Zieldichten
# anwenden. Kein Training, reine Auswertung — das Netz wird nicht angefasst,
# nur seine Konditionierung gewechselt.
#
# Deckt in einem Lauf ab:
#   Variante A   Phi = mu + kappa*sigma, eine Planung  (Mission glaube-1)
#                dieselbe Dichte ueber mehrere Runden  (Mission glaube-R)
#   Variante E   erst Phi = sigma, dann Phi = mu       (Mission zweistufig)
#   Variante B   gelernter Warmstart der differenzierbaren Verfeinerung,
#                dieselbe Verfeinerung aus Rauschen als Kontrolle, und
#                Kandidatenauswahl per Vorausschau ohne jede Optimierung
#
# Dazu drei Bezugsgroessen: das Netz auf der wahren Dichte (Obergrenze),
# Gradientenabstieg auf derselben Glaubensdichte bei gleicher Weglaenge, und
# eine Maeherbahn.
#
# Einreichen:
#   sbatch run_job_cfm_belief.bash
#   sbatch --export=ALL,CKPT=/pfad/zum/checkpoint.pt run_job_cfm_belief.bash
#
# Ohne CKPT wird der neueste Treffer auf checkpoints/*ERGLOSS*.pt genommen;
# die letzten fuenf Kandidaten stehen im Log, damit die Wahl nachvollziehbar
# bleibt.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture/exploration

# Fest gewaehlter Checkpoint statt der Automatik: die haette bei 310 Treffern
# wieder den neuesten nach Zeitstempel genommen, und welcher das ist, haengt
# von der Reihenfolge vergangener Laeufe ab statt von einer Entscheidung.
DEFAULT_CKPT="$HOME/Master_thesis/thesis_architecture/checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_18_16h45min_nxi25_D384_N256_C15_flip0.0_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt"
CKPT="${CKPT:-$DEFAULT_CKPT}"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint fehlt: $CKPT" >&2
    exit 1
fi
echo "Checkpoint:  $(basename "$CKPT")"
CKPT_ARG="--ckpt $CKPT" 

echo "Knoten:     $(hostname)"
echo "Startzeit:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --unbuffered python -u apply_cfm_belief.py \
    $CKPT_ARG \
    --shapes 12 \
    --rounds 3 \
    --kappa0 3.0 \
    --kappa1 0.3 \
    --n_prior 12 \
    --n_particles 256 \
    --truth_res 128 \
    --gp_res 64 \
    --gp_noise 0.05 \
    --sensor_radius 0.06 \
    --phi_mode uniform \
    --grad_steps 200 \
    --variant_b \
    --save_paths \
    --refine_steps 100 \
    --refine_lr 0.03 \
    --n_probe 40 \
    --lambda_unc 1.0 \
    --lambda_cov 20000 \
    --select_k 8 \
    --anytime_points 12 \
    --viz_shapes 4 \
    --out_dir results/cfm_belief

echo "Endzeit:    $(date)"
