#!/bin/bash
# Prueft, ob alles da ist, um das Phi-Kreuz zu fahren. Aendert nichts.
cd "$(dirname "$0")/.." || exit 1
W=$(pwd)
FEHLT=0
ok ()   { printf "  \033[32mok\033[0m    %s\n" "$1"; }
weg ()  { printf "  \033[31mFEHLT\033[0m %s\n" "$1"; FEHLT=1; }

echo "Projektwurzel: $W"
echo
echo "── Code ──"
for f in thesis_architecture/exploration/apply_cfm_belief.py \
         thesis_architecture/exploration/common/belief.py \
         thesis_architecture/exploration/common/acquisition.py \
         thesis_architecture/exploration/common/planner.py \
         thesis_architecture/exploration/common/data.py \
         thesis_architecture/exploration/common/metrics.py \
         thesis_architecture/exploration/common/observation.py \
         thesis_architecture/exploration/common/baselines.py \
         thesis_architecture/flow_matching_cond_particles_start.py \
         thesis_architecture/flow_matching_cond_particles_crossattn.py \
         thesis_architecture/ergodic_metric.py \
         thesis_architecture/obstacles.py \
         bsplinax-main/bsplinax/bspline.py ; do
  [ -f "$f" ] && ok "$f" || weg "$f"
done

echo
echo "── Neue Bausteine im Code ──"
grep -q "class MaskiertesWissen" thesis_architecture/exploration/common/belief.py 2>/dev/null \
  && ok "MaskiertesWissen (Grundwahrheit / kein Wissen)" || weg "MaskiertesWissen"
grep -q "prior_mode" thesis_architecture/exploration/apply_cfm_belief.py 2>/dev/null \
  && ok "--prior_mode wahrheit" || weg "--prior_mode"
grep -q "'netz'" thesis_architecture/exploration/apply_cfm_belief.py 2>/dev/null \
  && ok "--d_join netz (Variante D mit Startpunkt)" || weg "--d_join netz"
grep -q "PHI_MODELLE" thesis_architecture/exploration/common/acquisition.py 2>/dev/null \
  && ok "7 Phi-Modelle" || weg "PHI_MODELLE"
grep -q "start_cond" thesis_architecture/exploration/apply_cfm_belief.py 2>/dev/null \
  && ok "CfmPlanner erkennt start_cond" || weg "start_cond im CfmPlanner"

echo
echo "── Daten ──"
[ -f thesis_architecture/ergodic_dataset_generator/ergodic_dataset_775.db ] \
  && ok "ergodic_dataset_775.db (Holdout-Formen fuers Kreuz)" \
  || weg "ergodic_dataset_775.db  <- per Drive holen, NICHT ueber Git"
CK=$(ls -t thesis_architecture/checkpoints/*START_FLAT*_ep*.pt \
        thesis_architecture/checkpoints/netz2d_startpunkt.pt 2>/dev/null | head -1)
[ -n "$CK" ] && ok "Checkpoint: $(basename "$CK")" \
  || weg "Startpunkt-Checkpoint  <- per Drive holen, NICHT ueber Git"

echo
echo "── Pakete ──"
for m in torch numpy matplotlib scipy geomloss; do
  python -c "import $m" 2>/dev/null && ok "$m" || weg "$m  (pip install $m)"
done
python - <<'PY' 2>/dev/null && ok "CUDA verfuegbar" || weg "CUDA nicht verfuegbar — es liefe auf der CPU (40x langsamer)"
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY

echo
if [ -n "$CK" ]; then
  echo "── Checkpoint-Inhalt ──"
  python - "$CK" <<'PY'
import torch, sys
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
for k in ('epoch','start_cond','n_flat','D','nxi','n_particles'):
    print('  %-12s %s' % (k, c.get(k)))
if not c.get('start_cond'):
    print('  ACHTUNG: kein start_cond -> --d_join netz waere wirkungslos')
PY
fi

echo
[ "$FEHLT" -eq 0 ] && echo "Alles vorhanden. Weiter mit 02_phi_kreuz.sh" \
                   || echo "Es fehlt etwas — siehe oben."
exit $FEHLT
