#!/bin/bash
#SBATCH -J ergodic_dfl_finish
#SBATCH -t 18:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p interactive
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGTERM@120

# One-shot finish of the DFL run from run_job_ergodic_dfl.bash: resumes from
# its latest checkpoint (epoch 433/500, see slurm-168249.out) and runs the
# remaining ~67 epochs to completion, instead of requesting another full 24h
# allocation. Time budget sized for the WORST observed per-epoch rate on
# this cluster (node cn05: ~820s/ep, vs. cn14's ~245s/ep -- see earlier
# node-variance finding): 67 * 820s =~ 15.3h, plus ~1-1.5h buffer for the
# viz_every=100 holdout pass at epoch 500 and the final train+holdout
# visualisations run() always does after "Training complete!".

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd ~/Master_thesis/thesis_architecture

LATEST_CKPT=$(ls -t checkpoints/cond_spectral_crossattn_ergodic_dfl_flow_matching_ergodic_date_*_S256_nxi25_D384_C75_flip0.0_ep*.pt 2>/dev/null | head -1)

COMMON="--S 256 --nxi 25 --D 384 --p_flip 0.0 --copies_per_char 75 --lr 2e-4 \
  --mini_batch 64 --predict_lambda --dfl_weight 0.1 \
  --epochs 500 --save_every 50 --viz_every 100 \
  --save_model checkpoints/cond_spectral_crossattn_ergodic_dfl.pt \
  --use_wandb --wandb_project flow-matching --run_name ergodic_dfl_v1"

if [ -z "$LATEST_CKPT" ]; then
    echo "!!! Kein Checkpoint gefunden -- erwartet wurde eine Fortsetzung von ep0433. Breche ab."
    exit 1
fi

echo "Setze DFL-Training fort von: $LATEST_CKPT"
srun --unbuffered python flow_matching_runner_ergodic.py $COMMON --resume "$LATEST_CKPT"
