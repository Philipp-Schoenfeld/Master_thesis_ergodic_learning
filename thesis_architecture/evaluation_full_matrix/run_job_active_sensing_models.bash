#!/bin/bash
#SBATCH -J as_models
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 03:00:00
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH -C rtx
#SBATCH --mem=8G
#SBATCH -c 3
#SBATCH --signal=SIGTERM@120

# Active sensing with different start-conditioned CFM planners (2-3 h test). Builds on the full
# belief comparison (run_job_active_sensing_full.bash: replanning missions of run_mission_eval.py,
# UCB target with kappa 1.0 -> 0.05 over <= 5 rounds, full execution, TSVEC-SVGD refinement), and
# keeps what that run found: the belief model matters, the initialiser hardly does at 1500
# iterations. Here the question is the thesis' own: does a better generative prior help when the
# solver budget is small?
#
#   belief       ak (Attentive-Kernel GP, the best calibrated belief of the full run)
#   conditions   none_known, ten_samples          (25 holdout shapes each)
#   refinement   100 TSVEC-SVGD iterations per round (instead of 1500), 30 candidates
#   planners     cfm with 4 checkpoints, one run each (out_tag as_models_20261028_<label>):
#                  dit630   DiT, lambda 630, 180 epochs        (flow_matching_cond_particles_dit.py)
#                  unet630  U-Net, lambda 630, 180 epochs      (same data and budget as dit630)
#                  old500   transfer/netz2d_startpunkt.pt      (the planner of the full run)
#                  lm1300   U-Net, lambda 1300, 600 epochs     (LM teacher DB)
#                  unet1890 / unet0 / diff630   U-Net lambda 1890, U-Net without ErgLoss, diffusion (v-pred, DDPM sampler)
#                and linear rays (no network), computed in the first run only
#
# Time estimate (RTX 3090 / A5000; plan() of 30 candidates: U-Net ~6.4 s, DiT ~2.4 s; one round plans
# all 25 shapes in lockstep): U-Net planner ~25 min per run (2 conditions, <= 5 rounds), DiT ~10 min,
# SVGD at 100 iterations and the AK belief update (~0.8 s per shape-round) add ~10 min per run
# -> ~1.7-2.3 h in total. TOTAL_H stops opening new rounds in time; the log prints planning and
# SVGD minutes per mission set. Resumable: re-submitting continues (finished shape-rounds are
# never recomputed), so a cut run is completed by running the same script again.
#
#   sbatch run_job_active_sensing_models.bash
#   TOTAL_H=1.7 ONLY="unet1890 unet0 diff630" sbatch --export=ALL -t 02:00:00 run_job_active_sensing_models.bash   # add models later

TOTAL_H=${TOTAL_H:-2.6}
ONLY=${ONLY:-}      # space-separated labels to run (default: all); e.g. ONLY="unet1890 unet0 diff630"
N_ITERS=100
TAG=as_models_20261028
CKPT_DIR=$HOME/Master_thesis/thesis_architecture/checkpoints
# label=glob in checkpoints/ ('default' = the runner's default checkpoint, transfer/netz2d_startpunkt.pt)
MODELS=(
  "dit630=*_INPAINT_DIT-d12-h6_LAMERG50_ERGLOSS-*_final.pt"
  "unet630=*_INPAINT_LAMERG50_ERGLOSS-*_final.pt"
  "old500=default"
  "lm1300=*_INPAINT_LMTEACHER_ERGLOSS-SINKHORN-w1300-*_final.pt"
  "unet1890=*_INPAINT_LAMERG75_ERGLOSS-*_final.pt"
  "unet0=*_INPAINT_LAMERG0_final.pt"
  "diff630=*_INPAINT_DIFF-v_LAMERG50_ERGLOSS-*_final.pt"
)
# Models whose run directory already holds a finished mission set are re-run cheaply: finished
# shape-rounds are never recomputed, so only the models added later cost time.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd ~/Master_thesis/thesis_architecture/evaluation_full_matrix

echo "[job] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) TOTAL_H=${TOTAL_H} n_iters=${N_ITERS}"
T0=$(date +%s)
for spec in "${MODELS[@]}"; do
    LABEL=${spec%%=*}
    PAT=${spec#*=}
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $LABEL "* ]]; then continue; fi
    if [ "$PAT" = "default" ]; then
        CKPT_ARG=""
    else
        MATCH=( $(ls ${CKPT_DIR}/${PAT} 2>/dev/null) )
        if [ ${#MATCH[@]} -ne 1 ]; then
            echo "[job] $LABEL: pattern '${PAT}' matches ${#MATCH[@]} checkpoints -- skipped"
            continue
        fi
        CKPT_ARG="--ckpt ${MATCH[0]}"
    fi
    REMAIN=$(python -c "print(round(${TOTAL_H} - ($(date +%s) - ${T0}) / 3600.0, 3))")
    if python -c "import sys; sys.exit(0 if ${REMAIN} > 0.15 else 1)"; then :; else
        echo "[job] $LABEL: only ${REMAIN} h left of the budget -- skipped (re-submit to continue)"
        continue
    fi
    METHODS=cfm
    if [ "$LABEL" = "dit630" ]; then METHODS=cfm,linear; fi   # rays are computed once, in the dit630 run
    echo "[job] $(date) === $LABEL ($METHODS)  ${CKPT_ARG:-default checkpoint}  budget ${REMAIN} h"
    srun --unbuffered python -u run_mission_eval.py \
        --out_tag ${TAG}_${LABEL} \
        --refiner tsvec_svgd \
        --conditions none_known,ten_samples \
        --strategies ucb \
        --methods ${METHODS} \
        --beliefs ak \
        --max_rounds 5 \
        --n_iters ${N_ITERS} \
        --kappa0 1.0 --kappa1 0.05 \
        --full_execution \
        --workers 2 \
        --parallel_sets 4 \
        --time_budget_h ${REMAIN} \
        ${CKPT_ARG} \
        || echo "[job] $LABEL: run_mission_eval.py failed (continuing with the next model)"
    srun --unbuffered python -u plot_active_sensing_eval.py --out_tag ${TAG}_${LABEL}_tsvec_svgd \
        || echo "[job] $LABEL: plot_active_sensing_eval.py failed"
done
echo "[job] $(date) finished after $(( $(date +%s) - T0 )) s"
