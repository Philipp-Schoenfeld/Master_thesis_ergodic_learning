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
#
# WORKERS (fixed -- previously hardcoded to 0)
# ---------------------------------------------
# This job requested `-c 4` but ran dagger.py with `--workers 0`, i.e. fully
# serial SVGD refinement on one core, leaving 3 of the 4 allocated CPUs idle
# for the entire run -- the exact bug `run_job_policy.bash` already
# postmortemed for the oracle stage (job 154992: `mission.refine_batch` runs
# SVGD serially in NumPy without a process pool unless told otherwise). Fixed
# to use a process pool sized to the allocation, one core left for the main
# process.
#
# Final evaluation stage (new)
# -----------------------------
# Previously this job only printed the `evaluate.py` command needed to score
# `policy_a_dagger.pt`, leaving it as a manual follow-up step that was easy to
# forget. It now runs automatically at the end, within what remains of
# GESAMT_MIN, tagged 'policy_dagger' so it writes its own
# results/policy_dagger_*.{csv,json,png} instead of overwriting
# run_job_policy.bash's 'policy' tag. EVAL_NMAX/EVAL_SEEDS match that
# script's stage 4 defaults so the J numbers are comparable across tags.
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
WORKERS=${WORKERS:-3}                          # von -c 4: ein Kern fuer den Hauptprozess frei
EVAL_NMAX=${EVAL_NMAX:-8}                      # wie run_job_policy.bash Stufe 4
EVAL_SEEDS=${EVAL_SEEDS:-2}
MIN_EVAL=${MIN_EVAL:-30}                       # Reserve, damit die Auswertung noch stattfindet

ERG=exploration_optimierung/results
ABLAGE=exploration_optimierung/policy/ablage

ENDE=$(( $(date +%s) + GESAMT_MIN * 60 ))
rest_min() { echo $(( ( ENDE - $(date +%s) ) / 60 )) ; }

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

DAGGER_BUDGET=$(( $(rest_min) - MIN_EVAL ))
$RUN python -m exploration_optimierung.policy.dagger \
    --runden "$RUNDEN" --missionen_pro_runde "$MISSIONEN_PRO_RUNDE" \
    --n_max "$N_MAX" --split "$SPLIT" --folds "$FOLDS" --workers "$WORKERS" \
    --max_minuten "$DAGGER_BUDGET" \
    || echo "DAgger mit Fehler beendet — der aggregierte Datensatz bis dahin bleibt gueltig."

echo
echo "=========================================================="
echo "DAgger fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$ERG"/policy_datensatz_dagger.csv "$ERG"/policy_a_dagger.json \
         "$ABLAGE"/policy_a_dagger.pt; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "=========================================================="

# ── Auswertung ──────────────────────────────────────────────────────────────
# Auf der vollen Holdout-Menge ('val', nie beim Training gesehen), gegen
# 'fest'/'gelernt_a'/'rl_b' aus run_job_policy.bash -- dessen Modelle werden
# hier wiederverwendet (--modell_a/--modell_b greifen auf die Pfade in
# $ABLAGE zurueck, sofern sie existieren; sonst laesst evaluate.py die Spalte
# aus), damit der Vergleich alle vier Regler nebeneinander zeigt.
EVAL_BUDGET=$(( $(rest_min) - 5 ))
if [ ! -s "$ABLAGE/policy_a_dagger.pt" ]; then
  echo
  echo "Auswertung uebersprungen — kein $ABLAGE/policy_a_dagger.pt entstanden."
elif [ "$EVAL_BUDGET" -lt 10 ]; then
  echo
  echo "Auswertung uebersprungen — zu wenig Restzeit (${EVAL_BUDGET} min). Nachholen mit:"
  echo "  python -m exploration_optimierung.policy.evaluate --split val \\"
  echo "      --modell_a $ABLAGE/policy_a_dagger.pt \\"
  echo "      --modell_b $ABLAGE/policy_b.pt --tag policy_dagger"
else
  echo
  echo "Auswertung — volle Holdout-Menge, Budget ${EVAL_BUDGET} min  ($(date '+%H:%M'))"
  $RUN python -m exploration_optimierung.policy.evaluate \
      --n_max "$EVAL_NMAX" --seeds "$EVAL_SEEDS" --split val \
      --modell_a "$ABLAGE/policy_a_dagger.pt" --modell_b "$ABLAGE/policy_b.pt" \
      --tag policy_dagger --workers "$WORKERS" --max_minuten "$EVAL_BUDGET" \
      || echo "      Auswertung mit Fehler beendet"
fi

echo
echo "=========================================================="
echo "Alles fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$ERG"/policy_datensatz_dagger.csv "$ERG"/policy_a_dagger.json \
         "$ABLAGE"/policy_a_dagger.pt \
         "$ERG"/policy_dagger_vergleich.json "$ERG"/policy_dagger_metriken.csv \
         "$ERG"/policy_dagger_panel.png "$ERG"/policy_dagger_kurven.png; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "=========================================================="
