#!/bin/bash
#SBATCH -J ergodic_dfl
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=4G
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGTERM@120

# Production training run with the Decision-Focused-Learning lambda0 loss
# (see alm_solver_torch.py / compute_spectral_cfm_loss_dfl). mini_batch=64
# because mini_batch=512 (the baseline's own default) OOMs on an 11.63 GiB
# GPU once the DFL unrolled graph is added -- confirmed on this cluster;
# --n_lambda is left unset so it auto-computes to
# alm_solver_torch.n_lambda_for_alm(nxi). --save_model uses a distinct stem
# so this run's checkpoints never collide with the plain-baseline run
# (run_job_ergodic.bash) or any of the throwaway dfl_test_*/alm_*_diagnostic
# checkpoints already sitting in checkpoints/ from earlier manual testing.
#
# Same job-chaining pattern as run_job_ergodic.bash: resubmit this exact
# script with `sbatch --dependency=afterany:<JOBID>` and its own
# checkpoint-detection below picks up where the previous submission's
# 24h time limit cut it off.

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
    echo "Kein Checkpoint gefunden. Starte komplett neues DFL-Training..."
    srun --unbuffered python flow_matching_runner_ergodic.py $COMMON
else
    echo "Setze DFL-Training fort von: $LATEST_CKPT"
    srun --unbuffered python flow_matching_runner_ergodic.py $COMMON --resume "$LATEST_CKPT"
fi
