#!/bin/bash
#SBATCH -J dagger
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 06:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --signal=SIGTERM@120

# ===========================================================================
# DAgger-style data aggregation for Option A (exploration_optimierung/policy).
#
# Prerequisite: `results/policy_datensatz.csv` must already exist, i.e.
# `run_job_policy.bash` (or `policy.oracle` directly) has run at least once
# with --split train. This job never touches that file -- it copies it into
# `results/policy_datensatz_dagger.csv` and grows that copy instead, so a
# failed or exploratory DAgger run can never corrupt the base dataset that
# `run_job_policy.bash` treats as "already done, skip stage 1".
#
# Why this exists: `evaluate.py` shows Option A beats the fixed setting on a
# clean per-decision cross-validation, but loses to it on the full mission
# rollout. That gap is covariate shift, not a state-representation problem --
# see `policy/dagger.py`'s module docstring for the full argument. This job
# closes it by letting Option A drive its own rollouts and labeling the
# states it actually visits with dense oracle supervision, then retraining.
#
# Queue only via sbatch, never srun directly:
#
#     sbatch run_job_dagger.bash
# ===========================================================================

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate thesis
# Absolute path, same reasoning as in run_job_policy.bash: under sbatch, $0
# points at a cached copy of this script, not the submission directory.
cd ~/Master_thesis/thesis_architecture || exit 1

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

GESAMT_MIN=${GESAMT_MIN:-330}                  # ~5,5 h von 6 h
RUNDEN=${RUNDEN:-4}
MISSIONEN_PRO_RUNDE=${MISSIONEN_PRO_RUNDE:-15}
N_MAX=${N_MAX:-10}
SPLIT=${SPLIT:-train}                          # wie oracle.py/ppo.py: 'val' bleibt Test
FOLDS=${FOLDS:-5}

ERG=exploration_optimierung/results
ABLAGE=exploration_optimierung/policy/ablage

if [ -n "$SLURM_JOB_ID" ] && command -v srun >/dev/null 2>&1; then
  RUN="srun --unbuffered"
else
  RUN=""
fi

echo "=========================================================="
echo "DAgger — Option A auf selbst gefahrenen Zustaenden nachtrainieren"
echo "  Budget      : ${GESAMT_MIN} min"
echo "  Runden      : $RUNDEN x $MISSIONEN_PRO_RUNDE Missionen, n_max $N_MAX (Split '$SPLIT')"
echo "  Job         : ${SLURM_JOB_ID:-lokal}   Knoten: $(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ ! -s "$ERG/policy_datensatz.csv" ]; then
  echo "Kein $ERG/policy_datensatz.csv -- erst run_job_policy.bash (Stufe 1) laufen lassen."
  exit 1
fi

$RUN python -m exploration_optimierung.policy.dagger \
    --runden "$RUNDEN" --missionen_pro_runde "$MISSIONEN_PRO_RUNDE" \
    --n_max "$N_MAX" --split "$SPLIT" --folds "$FOLDS" --workers 0 \
    --max_minuten "$GESAMT_MIN" \
    || echo "DAgger mit Fehler beendet — der aggregierte Datensatz bis dahin bleibt gueltig."

echo
echo "=========================================================="
echo "Fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$ERG"/policy_datensatz_dagger.csv "$ERG"/policy_a_dagger.json \
         "$ABLAGE"/policy_a_dagger.pt; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "Zum Vergleich gegen 'fest'/'gelernt_a'/'rl_b' auswerten:"
echo "  python -m exploration_optimierung.policy.evaluate --split val \\"
echo "      --modell_a $ABLAGE/policy_a_dagger.pt --tag policy_dagger"
echo "=========================================================="
