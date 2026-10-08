#!/bin/bash
#SBATCH -J ppo_cont
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 08:30:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@120

# ===========================================================================
# PPO-only continuation of the just-trained `policy_b.pt` (job 161997) -- no
# Orakel, no Option A, no DAgger. Reason: that run's PPO stage ran its full
# 200 iterations, but `q(n)` stayed flat at ~0.10-0.12 the whole time while
# entropy fell from 2.3 to ~1.3 -- the signature of a policy that already
# exploited/converged, not one still improving. Just running more iterations
# of the SAME config would almost certainly repeat that plateau.
#
# This job instead resumes from those exact weights (`--init`) and reopens
# exploration before continuing:
#   --bc_epochen 0        skip behavior cloning -- `policy_b.pt` already
#                          encodes it, re-cloning would just waste budget.
#   --seed 1               different rollout stream than the original run
#                          (seed 0), so the first iterations do not just
#                          replay what the old run already ended on.
#   --log_std_boost 1.0     directly widens the learned continuous-action
#                          std (x e^1 ~ x2.7) instead of waiting for a
#                          raised entropy coefficient to slowly climb it
#                          back up over many iterations.
#   --c_entropie 0.03       3x the previous hardcoded 0.01, so exploration
#                          decays slower this time instead of collapsing
#                          again within ~200 iterations.
# Writes to its own `policy_b_cont.pt`/`_training.json` -- never overwrites
# `policy_b.pt` -- so the two can be compared before deciding which to keep.
#
# Also fixes the pre-existing display bug where `ertrag`/`J~` in ppo.py's
# console output and report JSON summed rewards across ALL episodes of an
# iteration instead of per episode (ppo.py's `puffer['rew']` stacks every
# episode of `--episoden_pro_iter` on the same time axis) -- this is why job
# 161997's log showed "J~-3.05" instead of the ~0.28 that the independent
# `evaluate.py` run actually measured. Purely a reporting fix; it never
# affected the PPO update itself, which always operated per-timestep.
#
# Final evaluation (new, within budget): scores `policy_b_cont.pt` against
# 'fest' and the existing 'gelernt_a' on the real 25 validation shapes, tag
# 'policy_cont' -- so the continuation's effect is visible immediately
# instead of requiring a separate manual step.
#
# Queue only via sbatch, never srun directly:
#
#     sbatch run_job_ppo_cont.bash
# ===========================================================================

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate thesis
cd ~/Master_thesis/thesis_architecture || exit 1

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

GESAMT_MIN=${GESAMT_MIN:-490}          # ~8,2 h von 8,5 h
ITER=${ITER:-100}                      # Obergrenze; Budget bremst frueher
SEED=${SEED:-1}
LOG_STD_BOOST=${LOG_STD_BOOST:-1.0}
C_ENTROPIE=${C_ENTROPIE:-0.03}
N_SHAPES_TRAIN=${N_SHAPES_TRAIN:-300}   # Rollout-Pool fuer PPOs eigene Episoden
                                       # -- entkoppelt von der (teuren)
                                       # Orakel-CSV, die hier dank
                                       # --bc_epochen 0 gar nicht gebraucht
                                       # wird. 300 von ~750 verfuegbaren
                                       # Trainingsformen, gegen den
                                       # Wiederholungsdruck der bisherigen
                                       # 35 (288 Ziehungen aus nur 35 Formen
                                       # ueber die geplanten Iterationen).
SPLIT=${SPLIT:-train}
PPO_ENVS=${PPO_ENVS:-8}
PPO_EPISODEN=${PPO_EPISODEN:-6}
PPO_NMAX=${PPO_NMAX:-8}
WORKERS=${WORKERS:-7}
EVAL_NMAX=${EVAL_NMAX:-8}
EVAL_SEEDS=${EVAL_SEEDS:-2}
MIN_EVAL=${MIN_EVAL:-25}

ERG=exploration_optimierung/results
ABLAGE=exploration_optimierung/policy/ablage
POLICY_B_INIT="$ABLAGE/policy_b.pt"
POLICY_B_OUT="$ABLAGE/policy_b_cont.pt"

ENDE=$(( $(date +%s) + GESAMT_MIN * 60 ))
rest_min() { echo $(( ( ENDE - $(date +%s) ) / 60 )) ; }

if [ -n "$SLURM_JOB_ID" ] && command -v srun >/dev/null 2>&1; then
  RUN="srun --unbuffered"
else
  RUN=""
fi

echo "=========================================================="
echo "PPO-Fortsetzung mit frischer Exploration (nur Stufe 3, kein Orakel/A/DAgger)"
echo "  Start       : $POLICY_B_INIT"
echo "  log_std_boost: +${LOG_STD_BOOST}  c_entropie: ${C_ENTROPIE}  Seed: ${SEED}"
echo "  Budget      : ${GESAMT_MIN} min"
echo "  Job         : ${SLURM_JOB_ID:-lokal}   Knoten: $(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ ! -s "$POLICY_B_INIT" ]; then
  echo "Kein $POLICY_B_INIT -- erst run_job_policy.bash laufen lassen."
  exit 1
fi

PPO_BUDGET=$(( $(rest_min) - MIN_EVAL ))
$RUN python -m exploration_optimierung.policy.ppo \
    --init "$POLICY_B_INIT" --bc_epochen 0 --seed "$SEED" \
    --log_std_boost "$LOG_STD_BOOST" --c_entropie "$C_ENTROPIE" \
    --iterationen "$ITER" --n_envs "$PPO_ENVS" \
    --episoden_pro_iter "$PPO_EPISODEN" --n_max "$PPO_NMAX" \
    --n_shapes "$N_SHAPES_TRAIN" --split "$SPLIT" --workers "$WORKERS" \
    --out "$POLICY_B_OUT" --bericht "$ERG/policy_b_cont_training.json" \
    --max_minuten "$PPO_BUDGET" \
    || echo "PPO-Fortsetzung mit Fehler beendet — weiter mit dem, was vorliegt"

echo
echo "=========================================================="
echo "PPO-Fortsetzung fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$POLICY_B_OUT" "$ERG/policy_b_cont_training.json"; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "=========================================================="

# ── Auswertung ──────────────────────────────────────────────────────────────
EVAL_BUDGET=$(( $(rest_min) - 5 ))
if [ ! -s "$POLICY_B_OUT" ]; then
  echo
  echo "Auswertung uebersprungen — kein $POLICY_B_OUT entstanden."
elif [ "$EVAL_BUDGET" -lt 10 ]; then
  echo
  echo "Auswertung uebersprungen — zu wenig Restzeit (${EVAL_BUDGET} min). Nachholen mit:"
  echo "  python -m exploration_optimierung.policy.evaluate --split val \\"
  echo "      --modell_b $POLICY_B_OUT --tag policy_cont"
else
  echo
  echo "Auswertung — volle Holdout-Menge, Budget ${EVAL_BUDGET} min  ($(date '+%H:%M'))"
  $RUN python -m exploration_optimierung.policy.evaluate \
      --n_max "$EVAL_NMAX" --seeds "$EVAL_SEEDS" --split val \
      --modell_a "$ABLAGE/policy_a.pt" --modell_b "$POLICY_B_OUT" \
      --tag policy_cont --workers "$WORKERS" --max_minuten "$EVAL_BUDGET" \
      || echo "      Auswertung mit Fehler beendet"
fi

echo
echo "=========================================================="
echo "Alles fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$POLICY_B_OUT" "$ERG/policy_b_cont_training.json" \
         "$ERG/policy_cont_vergleich.json" "$ERG/policy_cont_metriken.csv" \
         "$ERG/policy_cont_panel.png" "$ERG/policy_cont_kurven.png"; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "=========================================================="
