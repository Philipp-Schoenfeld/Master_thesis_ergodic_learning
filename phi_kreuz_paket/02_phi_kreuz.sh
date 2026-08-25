#!/bin/bash
# Das Phi-Kreuz: 3 Vorwissensmuster x 7 Zieldichten x 12 Formen x 7 Missionen.
#
#   ./02_phi_kreuz.sh            alle 12 Formen  (GPU: ~30 min)
#   ./02_phi_kreuz.sh 1          nur Form "A"    (zum schnellen Ausprobieren)
#   ./02_phi_kreuz.sh 12 cpu     erzwingt CPU    (~3 h, nur als Notnagel)
#
# Fertige Zellen werden uebersprungen — ein Abbruch ist unkritisch.
set -u
cd "$(dirname "$0")/../thesis_architecture/exploration" || exit 1

FORMEN=${1:-12}
GERAET=${2:-cuda}
CK=$(ls -t ../checkpoints/netz2d_startpunkt.pt ../checkpoints/*START_FLAT*_ep*.pt 2>/dev/null | head -1)
if [ -z "$CK" ]; then
  echo "Kein Startpunkt-Checkpoint gefunden. Er kommt NICHT ueber Git —"
  echo "hole netz2d_startpunkt.pt aus Google Drive nach thesis_architecture/checkpoints/."
  exit 1
fi
SUFFIX=""; [ "$FORMEN" != "12" ] && SUFFIX="_A"

echo "Checkpoint: $(basename "$CK")"
echo "Formen: $FORMEN   Geraet: $GERAET"
echo

for MUSTER in haelfte quadranten loch; do
  for PHI in ucb stretch mass ei lse mi eid; do
    OUT="results/phi_wahrheit${SUFFIX}/${MUSTER}_${PHI}"
    if [ -f "$OUT/metriken.csv" ]; then echo "[fertig] $MUSTER/$PHI"; continue; fi
    echo "[$(date +%H:%M)] $MUSTER / $PHI"
    python -u apply_cfm_belief.py --ckpt "$CK" \
      --shapes "$FORMEN" --rounds 3 \
      --missions orakel glaube-1 glaube-R zweistufig glaube-D B-warm maeher \
      --phi_model "$PHI" --prior_pattern "$MUSTER" --prior_mode wahrheit \
      --gp_noise 0.05 \
      --d_rounds 30 --d_execute_frac 0.10 --d_join netz \
      --save_paths --out_dir "$OUT" --device "$GERAET" \
      || echo "  FEHLER in $MUSTER/$PHI"
  done
done
echo
echo "Fertige Zellen: $(ls results/phi_wahrheit${SUFFIX}/*/metriken.csv 2>/dev/null | wc -l) von 21"
