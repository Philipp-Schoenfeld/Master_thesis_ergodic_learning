#!/bin/bash
#SBATCH -J policy
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 23:00:00
#SBATCH -p stud
#SBATCH --gres=gpu:1
##SBATCH -C 'rtx3080|rtx3090|a5000'
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --signal=SIGTERM@120

# ===========================================================================
# Gelernte Regler fuer die Laengeneinheit-Mission — die ganze Kette in einem
# Lauf von zwoelf Stunden (exploration_optimierung/policy).
#
#   1. Orakel        jede Runde alle Kandidaten ausprobieren
#                    -> Obergrenze + Trainingsdatensatz
#   2. Option A      Wertmodell, formweise kreuzvalidiert
#   3. Option B      Verhaltensklonen aus 1, danach PPO im Missionsloop
#   4. Auswertung    alle Regler ueber die volle Holdout-Menge,
#                    Metriken *und* Abbildungen
#
# Einreihen — nur ueber sbatch, dieses Skript nie direkt mit srun starten:
#
#     sbatch run_job_policy.bash
#
# Lokal (ohne SLURM) laeuft dieselbe Datei, sofern das Repo auch dort unter
# ~/Master_thesis/thesis_architecture liegt (der feste cd-Pfad unten ist
# absichtlich absolut, siehe Begruendung dort):
#
#     bash run_job_policy.bash
#
# Zeitverwaltung
# --------------
# Die vier Stufen teilen sich ein *gemeinsames* Budget (GESAMT_MIN, per
# Voreinstellung 11,5 h innerhalb des 12-h-Limits). Nach jeder Stufe wird neu
# gerechnet, wie viel bleibt: wird das Orakel frueher fertig, bekommt PPO die
# Zeit, und umgekehrt. Jede Stufe bekommt ihr Budget als `--max_minuten` und
# hoert damit **von selbst geordnet auf** — sie schreibt, was sie hat, statt
# vom Zeitlimit abgeschnitten zu werden. `--signal=SIGTERM@120` wirkt
# zusaetzlich: auch dann wird noch geschrieben.
#
# Fortsetzen nach dem Zeitlimit
# -----------------------------
# Stufen, deren Ergebnis schon vorliegt, werden uebersprungen (FORCE=1 erzwingt
# sie). Ein Folgejob setzt die Kette also dort fort, wo sie stand:
#
#     JID=$(sbatch --parsable run_job_policy.bash)
#     sbatch --dependency=afterany:$JID run_job_policy.bash
#
# Train/eval split fix (previously a leak)
# -----------------------------------------
# All earlier runs of this script used split='val' for every stage, i.e. the
# oracle dataset, Option A, and PPO were all trained on the exact 25 shapes
# `evaluate.py` then scored them on. `mission.load_holdout` and
# `policy/README.md` document `--split train` precisely to avoid this (train
# on the planner's ~750 training shapes, keep the 25 validation shapes
# exclusively for evaluate.py) -- that path just was not used. SPLIT now
# defaults to 'train' for stages 1 and 3, and stage 4 (evaluate.py) hardcodes
# '--split val' regardless of $SPLIT, so the two can no longer accidentally
# collapse into each other again. split='train' has ~750 candidate shapes
# available (see `shape_library.train_shape_names`), vastly more than the
# 25-shape default this script used to run with -- one of the causes
# `features.py` itself lists for keeping the state a 17-scalar vector instead
# of a richer encoding (small dataset -> risk of memorizing shapes).
#
# N_SHAPES_TRAIN calibration -- corrected after job 154992 (see postmortem
# below) and a measured probe, not the original 0.38s/cloud reference
# ------------------------------------------------------------------------
# The original estimate (0.38s/cloud, "RTX 2070 SUPER") only ever modeled
# GPU planning cost. A real timing probe (10 shapes, 2 rounds, `--workers 7`,
# on a dgx-station V100) measured ~1.5s/cloud end to end -- planning AND
# SVGD refinement together, which the old formula never accounted for at
# all. At that rate, 150 shapes (once seriously attempted in job 154992)
# would need ~60h for the oracle stage alone -- far past the 24h hard limit
# no matter how -t/GESAMT_MIN are tuned. 35 shapes is what actually fits
# (~14h oracle stage, ~20h job total, comfortable margin under 24h). Getting
# to more shapes than fit in one job would need genuine multi-job chaining
# over *disjoint* shape subsets, which needs an `--offset`/`--shape_start`
# argument in oracle.py first (not implemented) -- `sbatch
# --dependency=afterany` alone does NOT do this: this script skips stage 1
# entirely once `policy_datensatz.csv` is non-empty, regardless of whether
# all N_SHAPES_TRAIN/SEEDS actually finished, so a chained follow-up job
# would silently continue training on an incomplete dataset rather than
# fetch further shapes.
#
# Job 154992 postmortem (150 shapes, --workers 0)
# -------------------------------------------------
# Looked hung (0% GPU for 5.5h, `sacct` TotalCPU stuck at 00:00:00) but was
# most likely just very slow, not deadlocked: `mission.refine_batch` runs
# SVGD *serially, single-core, in NumPy* without a process pool, and this
# job had `--workers 0` despite requesting 8 CPUs (`-c 8`) -- 7 of them sat
# idle for the entire SVGD-heavy portion of every round. That alone
# explains most of the ~6.5x gap between the observed and estimated
# per-cloud cost. A second, independent bug compounded it: the outer
# seed-loop's `budget.abgelaufen()` check (oracle.py) had no lookahead term
# (unlike the inner round-loop's, which passes the last round's duration),
# so after seed 0 was cut short by its own budget, the *remaining* time
# still looked "not yet expired" and the loop started a second, five-hour-
# per-round seed it could never finish. Both are fixed: `--workers` below
# defaults to 7 (of the 8 allocated, one left for the main process), and
# oracle.py's seed-loop now passes the previous seed's duration as a
# lookahead, exactly like the round-loop already did.
#
# Random unknown-region masking (ZUFALLSMASKE)
# ---------------------------------------------
# Off by default, old behaviour (fully-blind start) unchanged. When set to 1,
# every shape starts with a random, spatially-coherent region already fully
# known (ground truth revealed) and the rest fully unknown, instead of the
# usual blind start -- see `common.belief.MaskiertesWissen`/`zufalls_maske`
# and oracle.py/ppo.py/evaluate.py's `--zufallsmaske` flag. It is applied to
# BOTH learners: the oracle dataset that trains Option A (the MLP) AND
# behavior-clones Option B's PPO warm start, and PPO's own on-policy env
# rollouts get the same flag independently (`ppo.py --zufallsmaske`) since
# those are separate mission rollouts, not read from the CSV. evaluate.py
# also gets it, so train and eval distributions match -- evaluating a policy
# trained under masking against a fully-blind eval would just reintroduce
# the same covariate-shift problem `dagger.py` exists to fix. All output
# filenames get a `_maske` suffix (via SUFFIX below) so a masked run never
# overwrites the unmasked baseline's results -- run with
# `sbatch -J policy_maske run_job_policy.bash` (job name only; SLURM reads
# `-J` from the sbatch command line before any `#SBATCH` pragma, so
# overriding it there needs no edit here) to also keep the two apart in
# `squeue`/log filenames.
# ===========================================================================

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate thesis
# NICHT ueber "$(dirname "$0")": unter sbatch fuehrt SLURM eine zwischen-
# gespeicherte Kopie des Skripts aus, deren Pfad nichts mehr mit dem
# Einreihungsort zu tun hat -- $0 zeigt dann ins Leere und der spaetere
# "python -m exploration_optimierung..." scheitert mit ModuleNotFoundError
# (beobachtet bei Job 151898: alle vier Stufen scheiterten in 21s). Wie jedes
# andere Job-Skript in diesem Projekt deshalb ein fester, absoluter Pfad.
cd ~/Master_thesis/thesis_architecture || exit 1

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# ── Stellschrauben (alle per Umgebungsvariable ueberschreibbar) ────────────
GESAMT_MIN=${GESAMT_MIN:-1350}         # 22,5 h von 23 h — Rest ist Sicherheit
N_SHAPES_TRAIN=${N_SHAPES_TRAIN:-35}   # Formen fuer Orakel + PPO (Split 'train')
N_SHAPES_EVAL=${N_SHAPES_EVAL:-25}   # die volle Validierungsmenge (VALIDATION_SHAPES)
N_MAX=${N_MAX:-10}                  # Runden je Orakel-Mission
SEEDS=${SEEDS:-2}
PARAM_PUNKTE=${PARAM_PUNKTE:-4}
SVGD_BUCKETS=${SVGD_BUCKETS:-"0 25 100"}
SPLIT=${SPLIT:-train}               # Orakel + PPO; Stufe 4 nutzt immer 'val'
PPO_ITER=${PPO_ITER:-200}           # Obergrenze; das Zeitbudget bremst frueher
PPO_ENVS=${PPO_ENVS:-8}
PPO_NMAX=${PPO_NMAX:-8}
EVAL_NMAX=${EVAL_NMAX:-8}
FOLDS=${FOLDS:-5}
FORCE=${FORCE:-0}
ZUFALLSMASKE=${ZUFALLSMASKE:-0}      # 1 = zufaellige Unbekannt-Region statt blindem Start
UNBEKANNT_MIN=${UNBEKANNT_MIN:-0.5}
UNBEKANNT_MAX=${UNBEKANNT_MAX:-0.9}
# SVGD-Verfeinerung ist die einzige CPU-gebundene, je Form unabhaengige
# Stelle der Schleife (`mission.refine_batch`) -- mit `--workers 0` lief sie
# bisher seriell auf einem einzigen Kern, obwohl der Job 8 anfordert
# (`-c 8` oben). Bei 25 Formen fiel das kaum auf; bei 150 Formen wurde genau
# das zum dominanten, im Zeitbudget unten (nur GPU-Planungskosten) gar nicht
# erfassten Kostenfaktor -- siehe Postmortem zu Job 154992. 7 statt 8: ein
# Kern bleibt dem Hauptprozess (Planung, Python-Overhead) frei, sonst werden
# alle angeforderten CPUs tatsaechlich benutzt statt teilweise leerzulaufen.
WORKERS=${WORKERS:-7}
# Aus, weil bei der neu gemessenen Rate (1,5 statt 0,38 s/Wolke) allein die
# Orakel-Vergleichsspalte der Auswertung (25 Formen x K Kandidaten x
# EVAL_NMAX Runden x SEEDS) auf ~8h statt den urspruenglich angenommenen
# ~2h anwaechst -- ohne echten Zusatznutzen: `evaluate.py` laedt den
# schaerferen `orakel_rollout`-Wert aus `policy_orakel.json` ohnehin bereits
# unabhaengig von diesem Flag. 1 setzt sie zusaetzlich wieder an.
MIT_ORAKEL=${MIT_ORAKEL:-0}

# Reserven, damit die spaeteren Stufen ueberhaupt stattfinden
MIN_A=${MIN_A:-20}                  # Option A braucht nur Minuten
MIN_B=${MIN_B:-150}                 # PPO unter 2,5 h lohnt kaum
MIN_EVAL=${MIN_EVAL:-30}            # ohne Orakelspalte reicht das

ERG=exploration_optimierung/results
ABLAGE=exploration_optimierung/policy/ablage

# Getrennte Dateinamen bei aktiver Maskierung, damit ein maskierter Lauf die
# unmaskierte Baseline nie ueberschreibt (siehe Kommentarblock oben).
if [ "$ZUFALLSMASKE" = "1" ]; then
  MASKE_FLAGS="--zufallsmaske --unbekannt_min $UNBEKANNT_MIN --unbekannt_max $UNBEKANNT_MAX"
  SUFFIX="_maske"
else
  MASKE_FLAGS=""
  SUFFIX=""
fi
DATENSATZ="$ERG/policy_datensatz${SUFFIX}.csv"
ORAKEL_JSON="$ERG/policy_orakel${SUFFIX}.json"
POLICY_A_PT="$ABLAGE/policy_a${SUFFIX}.pt"
POLICY_A_JSON="$ERG/policy_a_training${SUFFIX}.json"
POLICY_B_PT="$ABLAGE/policy_b${SUFFIX}.pt"
POLICY_B_JSON="$ERG/policy_b_training${SUFFIX}.json"
TAG="policy${SUFFIX}"

ENDE=$(( $(date +%s) + GESAMT_MIN * 60 ))
rest_min() { echo $(( ( ENDE - $(date +%s) ) / 60 )) ; }

if [ -n "$SLURM_JOB_ID" ] && command -v srun >/dev/null 2>&1; then
  RUN="srun --unbuffered"            # sonst haengen tqdm/print in der .err-Datei
else
  RUN=""
fi

echo "=========================================================="
echo "Gelernte Regler — ganze Kette"
echo "  Budget      : ${GESAMT_MIN} min (Ende $(date -d "@$ENDE" '+%H:%M' 2>/dev/null || echo "+${GESAMT_MIN}min"))"
echo "  Formen      : $N_SHAPES_TRAIN Training (Split '$SPLIT'), $N_SHAPES_EVAL Auswertung (immer 'val'), n_max $N_MAX, Seeds $SEEDS"
if [ "$ZUFALLSMASKE" = "1" ]; then
  echo "  Maskierung  : an -- unbekannter Anteil ~Uniform($UNBEKANNT_MIN, $UNBEKANNT_MAX), Dateien mit Suffix '$SUFFIX'"
else
  echo "  Maskierung  : aus (blinder Start, bisheriges Verhalten)"
fi
echo "  Kandidaten  : 4 Modelle x $PARAM_PUNKTE Parameter x [$SVGD_BUCKETS]"
echo "  Job         : ${SLURM_JOB_ID:-lokal}   Knoten: $(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# ── Grobe Erwartung, bevor irgendetwas rechnet ─────────────────────────────
# Bezugswert: 1,5 s je Entscheidung-Kandidat, Planung UND SVGD-Verfeinerung
# zusammen, gemessen mit --workers 7 auf einer dgx-station V100 (Job 155259,
# 10 Formen). Der fruehere Wert (0,38 s) hatte nur die GPU-Planung erfasst
# und die CPU-gebundene SVGD-Verfeinerung komplett ausgelassen -- das war
# einer der beiden Gruende, warum Job 154992 bei 150 Formen kollabierte
# (siehe Postmortem oben). SEK_JE_WOLKE bleibt ueberschreibbar, falls ein
# spaeterer Lauf auf anderer Hardware/anderem --workers-Wert eine neue
# Sondierung liefert.
SEK_JE_WOLKE=${SEK_JE_WOLKE:-150}     # Hundertstelsekunden, also 1,50 s
N_BUCKETS=$(echo $SVGD_BUCKETS | wc -w)
K=$(( 4 * PARAM_PUNKTE * N_BUCKETS ))
WOLKEN_ORAKEL=$(( N_SHAPES_TRAIN * K * N_MAX * SEEDS ))
WOLKEN_EVAL=$(( N_SHAPES_EVAL * K * EVAL_NMAX * SEEDS ))
MIN_ORAKEL=$(( WOLKEN_ORAKEL * SEK_JE_WOLKE / 100 / 60 ))
MIN_EVAL_ORAKEL=$(( WOLKEN_EVAL * SEK_JE_WOLKE / 100 / 60 ))
if [ "$MIT_ORAKEL" = "1" ]; then EVAL_EXTRA=$MIN_EVAL_ORAKEL; else EVAL_EXTRA=0; fi
echo "  Kandidaten je Entscheidung: $K"
echo "  Erwartung (Massstab dgx-station V100 mit --workers 7, gemessen in Job 155259):"
echo "    Stufe 1 Orakel        ~${MIN_ORAKEL} min  (${WOLKEN_ORAKEL} geplante Wolken)"
echo "    Stufe 2 Option A      ~10 min"
echo "    Stufe 3 PPO           bis $(( PPO_ITER * 65 / 60 )) min (${PPO_ITER} Iterationen; Budget bremst frueher)"
if [ "$MIT_ORAKEL" = "1" ]; then
  echo "    Stufe 4 Auswertung    ~10 min + ~${MIN_EVAL_ORAKEL} min fuer die Orakelspalte (MIT_ORAKEL=1)"
else
  echo "    Stufe 4 Auswertung    ~10 min (MIT_ORAKEL=0 -- orakel_rollout-Obergrenze kommt trotzdem aus Stufe 1)"
fi
echo "    Summe                 ~$(( MIN_ORAKEL + 10 + PPO_ITER * 65 / 60 + 10 + EVAL_EXTRA )) min von ${GESAMT_MIN} min"

# ── 1. Orakel ──────────────────────────────────────────────────────────────
if [ "$FORCE" != "1" ] && [ -s "$DATENSATZ" ]; then
  echo; echo "[1/4] Orakel uebersprungen — $DATENSATZ liegt vor (FORCE=1 erzwingt neu)."
else
  BUDGET=$(( $(rest_min) - MIN_A - MIN_B - MIN_EVAL ))
  echo; echo "[1/4] Orakel — Budget ${BUDGET} min  ($(date '+%H:%M'))"
  if [ "$BUDGET" -lt 30 ]; then
    echo "      zu wenig Zeit, uebersprungen"
  else
    $RUN python -m exploration_optimierung.policy.oracle \
        --n_shapes "$N_SHAPES_TRAIN" --n_max "$N_MAX" --seeds "$SEEDS" \
        --param_punkte "$PARAM_PUNKTE" --svgd_buckets $SVGD_BUCKETS \
        --split "$SPLIT" --plan_batch 128 --workers "$WORKERS" \
        --out "$DATENSATZ" --bericht "$ORAKEL_JSON" \
        $MASKE_FLAGS \
        --max_minuten "$BUDGET" || echo "      Orakel mit Fehler beendet — weiter mit dem, was vorliegt"
  fi
fi

# ── 2. Option A ────────────────────────────────────────────────────────────
if [ ! -s "$DATENSATZ" ]; then
  echo; echo "[2/4] Option A uebersprungen — kein Datensatz vorhanden."
elif [ "$FORCE" != "1" ] && [ -s "$POLICY_A_PT" ]; then
  echo; echo "[2/4] Option A uebersprungen — $POLICY_A_PT liegt vor."
else
  echo; echo "[2/4] Option A — Wertmodell, ${FOLDS} Falten  ($(date '+%H:%M'))"
  # Kein --zufallsmaske hier: Option A liest nur die CSV, die schon (nicht)
  # maskiert wurde -- die Maskierung wirkt beim Erzeugen der Missionsdaten
  # (Stufe 1), nicht beim reinen Regressionstraining auf der Tabelle.
  $RUN python -m exploration_optimierung.policy.train --folds "$FOLDS" \
      --datensatz "$DATENSATZ" --out "$POLICY_A_PT" --bericht "$POLICY_A_JSON" \
      || echo "      Option A mit Fehler beendet — weiter"
fi

# ── 3. Option B ────────────────────────────────────────────────────────────
if [ "$FORCE" != "1" ] && [ -s "$POLICY_B_PT" ]; then
  echo; echo "[3/4] Option B uebersprungen — $POLICY_B_PT liegt vor."
else
  BUDGET=$(( $(rest_min) - MIN_EVAL ))
  echo; echo "[3/4] Option B — Verhaltensklonen + PPO, Budget ${BUDGET} min  ($(date '+%H:%M'))"
  if [ "$BUDGET" -lt 20 ]; then
    echo "      zu wenig Zeit, uebersprungen"
  else
    # --zufallsmaske hier zusaetzlich zu Stufe 1 noetig: PPOs eigene
    # On-Policy-Rollouts in MissionUmgebung sind eigene Missionen, keine
    # Zeilen aus der CSV -- ohne diese Wiederholung saehe nur das
    # Verhaltensklonen (aus der CSV) die Maskierung, PPOs eigenes Training
    # aber nicht.
    $RUN python -m exploration_optimierung.policy.ppo \
        --iterationen "$PPO_ITER" --n_envs "$PPO_ENVS" --n_max "$PPO_NMAX" \
        --n_shapes "$N_SHAPES_TRAIN" --split "$SPLIT" --workers "$WORKERS" \
        --datensatz "$DATENSATZ" --out "$POLICY_B_PT" --bericht "$POLICY_B_JSON" \
        $MASKE_FLAGS \
        --max_minuten "$BUDGET" || echo "      Option B mit Fehler beendet — weiter"
  fi
fi

# ── 4. Auswertung ──────────────────────────────────────────────────────────
BUDGET=$(( $(rest_min) - 5 ))
echo; echo "[4/4] Auswertung — volle Holdout-Menge, Budget ${BUDGET} min  ($(date '+%H:%M'))"
if [ "$BUDGET" -lt 10 ]; then
  echo "      zu wenig Zeit — bitte in einem Folgejob nachholen:"
  echo "      sbatch --export=ALL,FORCE=0 run_job_policy.bash"
else
  # MIT_ORAKEL=1 faehrt das Orakel zusaetzlich live als Regler mit (teuer:
  # ~${MIN_EVAL_ORAKEL} min bei der aktuellen Formenzahl/-groesse, siehe
  # Vorwarnung oben) -- Voreinstellung 0, weil die Obergrenze schon kostenlos
  # aus Stufe 1s `policy_orakel.json` (orakel_rollout, sogar die schaerfere
  # der beiden Zahlen) in den Vergleich einfliesst.
  # '--split val' ist hier bewusst fest verdrahtet statt an $SPLIT gekoppelt:
  # die Auswertung muss immer auf der echten Holdout-Menge laufen, egal womit
  # Orakel/PPO trainiert wurden -- sonst waere die Zahl wieder die verzerrte
  # von vorher. --zufallsmaske dagegen wird hier bewusst mitgegeben: die
  # Auswertung muss unter derselben Bedingung laufen, unter der trainiert
  # wurde, sonst waere das genau das Trainings-/Einsatz-Missverhaeltnis, das
  # dagger.py fuer die Rundenwahl selbst schon behebt.
  ORAKEL_FLAG=""
  [ "$MIT_ORAKEL" = "1" ] && ORAKEL_FLAG="--mit_orakel"
  $RUN python -m exploration_optimierung.policy.evaluate \
      --n_shapes "$N_SHAPES_EVAL" --n_max "$EVAL_NMAX" --seeds "$SEEDS" \
      --split val $ORAKEL_FLAG \
      --param_punkte "$PARAM_PUNKTE" --svgd_buckets $SVGD_BUCKETS \
      --modell_a "$POLICY_A_PT" --modell_b "$POLICY_B_PT" --tag "$TAG" \
      --orakel_bericht "$ORAKEL_JSON" \
      $MASKE_FLAGS \
      --workers "$WORKERS" --max_minuten "$BUDGET" \
      || echo "      Auswertung mit Fehler beendet"
fi

echo
echo "=========================================================="
echo "Fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$DATENSATZ" "$ORAKEL_JSON" \
         "$POLICY_A_JSON" "$POLICY_B_JSON" \
         "$ERG"/${TAG}_metriken.csv "$ERG"/${TAG}_vergleich.json \
         "$ERG"/${TAG}_panel.png "$ERG"/${TAG}_kurven.png \
         "$ERG"/${TAG}_aktionen.png "$POLICY_A_PT" "$POLICY_B_PT"; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "=========================================================="
