#!/bin/bash
set -e
export PYTHONUTF8=1
PATTERNS="haelfte quadranten loch"
MODELS="ucb stretch mass ei lse mi eid"
LOG="../../visualisierungen/logs/06_phi_kreuz_laenge_12shapes.txt"
: > "$LOG"
for pat in $PATTERNS; do
  for mdl in $MODELS; do
    echo "[$(date +%H:%M)] $pat / $mdl" | tee -a "$LOG"
    python apply_cfm_belief.py \
      --ckpt ../checkpoints/netz2d_laenge.pt \
      --shapes 12 \
      --prior_pattern "$pat" --phi_model "$mdl" \
      --prior_mode wahrheit \
      --missions orakel glaube-1 glaube-R zweistufig glaube-D maeher \
      --d_rounds 30 --d_join netz \
      --save_paths \
      --out_dir "../../visualisierungen/01_phi_kreuz_laenge/trajektorien/${pat}_${mdl}" \
      >> "$LOG" 2>&1
    sleep 2
  done
done
echo "PHIKREUZ_LAENGE_12SHAPES_DONE" | tee -a "$LOG"
