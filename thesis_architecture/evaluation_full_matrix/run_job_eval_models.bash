#!/bin/bash
#SBATCH -J eval_models
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --signal=SIGTERM@120

# Incremental model comparison (eval_start_cfm_models.py): every model is evaluated on the SAME
# protocol (all 144 holdout cases x 30 samples, warm start from 90 curves on the 48 stored-start densities, start
# grid) with the same particles and flow noise, and all raw results are stored under
# results/${OUT_TAG}/models/<label>/. Finished parts are skipped, so the job can simply be
# re-submitted after a timeout, and models trained later are added with another job:
#
#   sbatch run_job_eval_models.bash 'unet-l210=*_INPAINT_LAMERG25_ERGLOSS-*_final.pt' ...
#   sbatch run_job_eval_models.bash 'unet-l1890=*_INPAINT_LAMERG75_ERGLOSS-*_final.pt' \
#                                   'dit-l630=*_INPAINT_DIT-d12-h6_LAMERG50_ERGLOSS-*_final.pt'
#
# Each argument is label=<glob in checkpoints/>, the glob must match exactly one file. The report
# (tables + plots over ALL stored models) is rebuilt at the end. ~30 min per model (estimate, scaled up from ~10 min at the old sizes) + a few min of
# model-independent baselines on the first run (timing from eval_lm_model, RTX class).

OUT_TAG=cfm_models_20261028
REF=unet-l1300-600ep
DB=$HOME/Master_thesis/thesis_architecture/ergodic_dataset_generator/ergodic_dataset_lm_nxi25.db
CKPT_DIR=$HOME/Master_thesis/thesis_architecture/checkpoints

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)  start: $(date)"
for spec in "$@"; do
    LABEL=${spec%%=*}
    MATCH=( $(ls ${CKPT_DIR}/${spec#*=} 2>/dev/null) )
    if [ ${#MATCH[@]} -ne 1 ]; then
        echo "[$LABEL] pattern '${spec#*=}' matches ${#MATCH[@]} checkpoints -- skipped"
        continue
    fi
    echo "=== $LABEL <- $(basename ${MATCH[0]})  $(date +%H:%M)"
    srun --unbuffered python -u eval_start_cfm_models.py run --label "$LABEL" --ckpt "${MATCH[0]}" \
        --db "$DB" --out_tag "$OUT_TAG"
done
srun --unbuffered python -u eval_start_cfm_models.py report --out_tag "$OUT_TAG" --ref "$REF"
echo "exit code $?  end: $(date)"
