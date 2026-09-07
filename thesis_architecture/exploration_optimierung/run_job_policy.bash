#!/bin/bash
#SBATCH -J policy
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 12:00:00
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
# Lokal (ohne SLURM) laeuft dieselbe Datei unveraendert:
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
# ===========================================================================

set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate thesis
cd "$(dirname "$0")/.." || exit 1          # -> thesis_architecture/

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# ── Stellschrauben (alle per Umgebungsvariable ueberschreibbar) ────────────
GESAMT_MIN=${GESAMT_MIN:-690}       # 11,5 h von 12 h — Rest ist Sicherheit
N_SHAPES=${N_SHAPES:-25}
N_MAX=${N_MAX:-10}                  # Runden je Orakel-Mission
SEEDS=${SEEDS:-2}
PARAM_PUNKTE=${PARAM_PUNKTE:-4}
SVGD_BUCKETS=${SVGD_BUCKETS:-"0 25 100"}
SPLIT=${SPLIT:-val}
PPO_ITER=${PPO_ITER:-200}           # Obergrenze; das Zeitbudget bremst frueher
PPO_ENVS=${PPO_ENVS:-8}
PPO_NMAX=${PPO_NMAX:-8}
EVAL_NMAX=${EVAL_NMAX:-8}
FOLDS=${FOLDS:-5}
FORCE=${FORCE:-0}

# Reserven, damit die spaeteren Stufen ueberhaupt stattfinden
MIN_A=${MIN_A:-20}                  # Option A braucht nur Minuten
MIN_B=${MIN_B:-150}                 # PPO unter 2,5 h lohnt kaum
MIN_EVAL=${MIN_EVAL:-110}           # Auswertung inkl. Orakelspalte

ERG=exploration_optimierung/results
ABLAGE=exploration_optimierung/policy/ablage

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
echo "  Formen      : $N_SHAPES (Split '$SPLIT'), n_max $N_MAX, Seeds $SEEDS"
echo "  Kandidaten  : 4 Modelle x $PARAM_PUNKTE Parameter x [$SVGD_BUCKETS]"
echo "  Job         : ${SLURM_JOB_ID:-lokal}   Knoten: $(hostname)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# ── Grobe Erwartung, bevor irgendetwas rechnet ─────────────────────────────
# Bezugswert: 0,38 s je geplanter Partikelwolke bei flow_steps=100, gemessen
# auf einer RTX 2070 SUPER. Die Planung ist rechengebunden und skaliert ab
# Stapelgroesse 16 linear, deshalb genuegt die Zahl der Wolken als Mass. Eine
# schnellere Karte verschiebt alles proportional nach unten; die Stufen
# richten sich ohnehin nach dem Budget, das hier ist nur die Vorwarnung.
N_BUCKETS=$(echo $SVGD_BUCKETS | wc -w)
K=$(( 4 * PARAM_PUNKTE * N_BUCKETS ))
WOLKEN_ORAKEL=$(( N_SHAPES * K * N_MAX * SEEDS ))
WOLKEN_EVAL=$(( N_SHAPES * K * EVAL_NMAX * SEEDS ))
MIN_ORAKEL=$(( WOLKEN_ORAKEL * 38 / 100 / 60 ))
MIN_EVAL_ORAKEL=$(( WOLKEN_EVAL * 38 / 100 / 60 ))
echo "  Kandidaten je Entscheidung: $K"
echo "  Erwartung (Massstab RTX 2070 SUPER, schnellere Karte entsprechend weniger):"
echo "    Stufe 1 Orakel        ~${MIN_ORAKEL} min  (${WOLKEN_ORAKEL} geplante Wolken)"
echo "    Stufe 2 Option A      ~10 min"
echo "    Stufe 3 PPO           bis $(( PPO_ITER * 65 / 60 )) min (${PPO_ITER} Iterationen; Budget bremst frueher)"
echo "    Stufe 4 Auswertung    ~10 min + ~${MIN_EVAL_ORAKEL} min fuer die Orakelspalte"
echo "    Summe                 ~$(( MIN_ORAKEL + 10 + PPO_ITER * 65 / 60 + 10 + MIN_EVAL_ORAKEL )) min von ${GESAMT_MIN} min"

# ── 1. Orakel ──────────────────────────────────────────────────────────────
if [ "$FORCE" != "1" ] && [ -s "$ERG/policy_datensatz.csv" ]; then
  echo; echo "[1/4] Orakel uebersprungen — $ERG/policy_datensatz.csv liegt vor (FORCE=1 erzwingt neu)."
else
  BUDGET=$(( $(rest_min) - MIN_A - MIN_B - MIN_EVAL ))
  echo; echo "[1/4] Orakel — Budget ${BUDGET} min  ($(date '+%H:%M'))"
  if [ "$BUDGET" -lt 30 ]; then
    echo "      zu wenig Zeit, uebersprungen"
  else
    $RUN python -m exploration_optimierung.policy.oracle \
        --n_shapes "$N_SHAPES" --n_max "$N_MAX" --seeds "$SEEDS" \
        --param_punkte "$PARAM_PUNKTE" --svgd_buckets $SVGD_BUCKETS \
        --split "$SPLIT" --plan_batch 128 --workers 0 \
        --max_minuten "$BUDGET" || echo "      Orakel mit Fehler beendet — weiter mit dem, was vorliegt"
  fi
fi

# ── 2. Option A ────────────────────────────────────────────────────────────
if [ ! -s "$ERG/policy_datensatz.csv" ]; then
  echo; echo "[2/4] Option A uebersprungen — kein Datensatz vorhanden."
elif [ "$FORCE" != "1" ] && [ -s "$ABLAGE/policy_a.pt" ]; then
  echo; echo "[2/4] Option A uebersprungen — $ABLAGE/policy_a.pt liegt vor."
else
  echo; echo "[2/4] Option A — Wertmodell, ${FOLDS} Falten  ($(date '+%H:%M'))"
  $RUN python -m exploration_optimierung.policy.train --folds "$FOLDS" \
      || echo "      Option A mit Fehler beendet — weiter"
fi

# ── 3. Option B ────────────────────────────────────────────────────────────
if [ "$FORCE" != "1" ] && [ -s "$ABLAGE/policy_b.pt" ]; then
  echo; echo "[3/4] Option B uebersprungen — $ABLAGE/policy_b.pt liegt vor."
else
  BUDGET=$(( $(rest_min) - MIN_EVAL ))
  echo; echo "[3/4] Option B — Verhaltensklonen + PPO, Budget ${BUDGET} min  ($(date '+%H:%M'))"
  if [ "$BUDGET" -lt 20 ]; then
    echo "      zu wenig Zeit, uebersprungen"
  else
    $RUN python -m exploration_optimierung.policy.ppo \
        --iterationen "$PPO_ITER" --n_envs "$PPO_ENVS" --n_max "$PPO_NMAX" \
        --n_shapes "$N_SHAPES" --split "$SPLIT" --workers 0 \
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
  # Das Orakel als Regler mitzufahren kostet ein Vielfaches; es bleibt drin,
  # weil ohne diese Spalte nicht einzuordnen ist, wie viel vom Erreichbaren
  # die gelernten Regler holen. Reicht die Zeit nicht, laesst `--max_minuten`
  # genau diese Spalte aus und schreibt den Rest.
  $RUN python -m exploration_optimierung.policy.evaluate \
      --n_shapes "$N_SHAPES" --n_max "$EVAL_NMAX" --seeds "$SEEDS" \
      --split "$SPLIT" --mit_orakel \
      --param_punkte "$PARAM_PUNKTE" --svgd_buckets $SVGD_BUCKETS \
      --workers 0 --max_minuten "$BUDGET" \
      || echo "      Auswertung mit Fehler beendet"
fi

echo
echo "=========================================================="
echo "Fertig um $(date '+%H:%M')  —  entstanden:"
for f in "$ERG"/policy_datensatz.csv "$ERG"/policy_orakel.json \
         "$ERG"/policy_a_training.json "$ERG"/policy_b_training.json \
         "$ERG"/policy_metriken.csv "$ERG"/policy_vergleich.json \
         "$ERG"/policy_panel.png "$ERG"/policy_kurven.png \
         "$ERG"/policy_aktionen.png "$ABLAGE"/policy_a.pt "$ABLAGE"/policy_b.pt; do
  if [ -e "$f" ]; then
    echo "  [ja  ] $f  ($(du -h "$f" 2>/dev/null | cut -f1))"
  else
    echo "  [nein] $f"
  fi
done
echo "=========================================================="
