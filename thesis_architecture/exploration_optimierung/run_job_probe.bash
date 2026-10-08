#!/bin/bash
#SBATCH -J policy_probe
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:15:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@60

# ===========================================================================
# Disposable timing probe for the oracle stage's real per-round cost with
# actual parallelism (--workers 7), after job 154992 stalled at 150 shapes
# under --workers 0 (all SVGD refinement serialized onto one of 8 allocated
# CPUs). Writes to its own throwaway files (policy_datensatz_probe.csv /
# policy_orakel_probe.json), never the real dataset the full run reads/writes.
#
# Stage 2 (new): a few PPO iterations at the corrected --episoden_pro_iter 6
# (run_job_policy.bash's PPO_EPISODEN, raised from ppo.py's old default of 2
# to fix the noisy-advantage problem diagnosed in policy_b_training.json).
# `run_job_policy.bash`'s PPO_SEK_ITER estimate currently just scales the one
# measurement on record (65 s/iter at episoden_pro_iter=2) linearly by 3x --
# an assumption, not a measurement. This stage measures the real per-iteration
# wall time under the actual new setting on real cluster hardware, so
# GESAMT_MIN/PPO_ITER in the full run can be calibrated instead of guessed.
# Also writes to throwaway files (policy_b_probe.pt / _probe.json), and
# bc_epochen=0 skips behavior-cloning pretraining -- irrelevant to iteration
# timing and would just burn probe budget.
#
# Queue only via sbatch, never srun directly:
#
#     sbatch run_job_probe.bash
# ===========================================================================

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate thesis
cd ~/Master_thesis/thesis_architecture || exit 1

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

if [ -n "$SLURM_JOB_ID" ] && command -v srun >/dev/null 2>&1; then
  RUN="srun --unbuffered"
else
  RUN=""
fi

echo "=========================================================="
echo "Zeit-Sondierung: 10 Formen, 2 Runden, 1 Seed, --workers 7"
echo "  Job         : ${SLURM_JOB_ID:-lokal}   Knoten: $(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

$RUN python -m exploration_optimierung.policy.oracle \
    --n_shapes 10 --n_max 2 --seeds 1 --split train --workers 7 \
    --zufallsmaske \
    --out exploration_optimierung/results/policy_datensatz_probe.csv \
    --bericht exploration_optimierung/results/policy_orakel_probe.json \
    --max_minuten 30

echo
echo "=========================================================="
echo "Zeit-Sondierung: PPO, episoden_pro_iter=6 (10 Formen, 8 Runden), --workers 7"
echo "=========================================================="
$RUN python -m exploration_optimierung.policy.ppo \
    --iterationen 4 --n_envs 8 --episoden_pro_iter 6 --n_max 8 \
    --n_shapes 10 --split train --workers 7 --bc_epochen 0 \
    --out exploration_optimierung/policy/ablage/policy_b_probe.pt \
    --bericht exploration_optimierung/results/policy_b_probe.json \
    --max_minuten 25

echo "Fertig."
