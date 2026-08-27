#!/bin/bash
# Prueft die Checkpoint-Auswahl beider Jobskripte gegen realistische Dateinamen.
set -u
T=$(mktemp -d); cd "$T"; mkdir checkpoints
P="checkpoints/cond_particles_crossattn_flow_matching_particle_ergodic_date_08_27_09h00min_nxi64_D384_N256_C2_flip0.0_START_FLAT400"
E="ERGLOSS-SINKHORN-w1300-blur0.05-tp2"
fehler=0

waehle_hp () {  # $1=TAG $2=LENFREQ  -> gibt "FERTIG|<datei>" oder "<datei>" oder "NEU"
    local TAG=$1 LENFREQ=$2 MARKE MUSTER FERTIG LATEST
    if [ "$LENFREQ" = "oktaven" ]; then MUSTER="checkpoints/*_LEN-*_${TAG}_*"
    else MARKE=$(echo "$LENFREQ" | tr '[:lower:]' '[:upper:]')FREQ
         MUSTER="checkpoints/*_LEN-*_${MARKE}_${TAG}_*"; fi
    FERTIG=$(ls -t ${MUSTER}_final.pt 2>/dev/null | head -1)
    [ -n "$FERTIG" ] && { echo "FERTIG"; return; }
    LATEST=$(ls -t ${MUSTER}_ep*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && echo "$(basename "$LATEST")" || echo "NEU"
}
waehle_warm () {  # $1=TAG $2=LENFREQ
    local TAG=$1 LENFREQ=$2 MARKE MUSTER FERTIG EIGEN
    MARKE=$(echo "$LENFREQ" | tr '[:lower:]' '[:upper:]')FREQ
    MUSTER="checkpoints/*_LEN-*_${MARKE}_FROMSTART_${TAG}_*"
    FERTIG=$(ls -t ${MUSTER}_final.pt 2>/dev/null | head -1)
    [ -n "$FERTIG" ] && { echo "FERTIG"; return; }
    EIGEN=$(ls -t ${MUSTER}_ep*.pt 2>/dev/null | head -1)
    [ -n "$EIGEN" ] && echo "$(basename "$EIGEN")" || echo "WARMSTART"
}
pruefe () { # $1=was $2=erwartet $3=ist
    if [ "$2" = "$3" ]; then echo "[ok] $1"; else echo "[!!] $1: erwartet '$2', ist '$3'"; fehler=$((fehler+1)); fi
}

echo "--- 1  leeres Verzeichnis ---"
pruefe "von null auf"        "NEU"        "$(waehle_hp LR1E4 linear)"
pruefe "Warmstart"           "WARMSTART"  "$(waehle_warm WARM linear)"

echo "--- 2  nur ein ALTER Oktaven-Stand liegt da ---"
touch "${P}_LEN-pd0.1_LR1E4_${E}_ep0148.pt"
pruefe "linear ignoriert ihn"     "NEU"    "$(waehle_hp LR1E4 linear)"
pruefe "oktaven wuerde ihn nehmen" "$(basename ${P}_LEN-pd0.1_LR1E4_${E}_ep0148.pt)" "$(waehle_hp LR1E4 oktaven)"

echo "--- 3  eigener LINEARFREQ-Stand kommt dazu ---"
command sleep 1; touch "${P}_LEN-pd0.1_LINEARFREQ_LR1E4_${E}_ep0060.pt"
pruefe "setzt am eigenen fort" "$(basename ${P}_LEN-pd0.1_LINEARFREQ_LR1E4_${E}_ep0060.pt)" "$(waehle_hp LR1E4 linear)"

echo "--- 4  Warmstart-Stand darf den Kettenpunkt nicht stoeren ---"
command sleep 1; touch "${P}_LEN-pd0.1_LINEARFREQ_FROMSTART_WARM_${E}_ep0070.pt"
pruefe "LR1E4 bleibt bei seinem"  "$(basename ${P}_LEN-pd0.1_LINEARFREQ_LR1E4_${E}_ep0060.pt)" "$(waehle_hp LR1E4 linear)"
pruefe "WARM findet seinen"       "$(basename ${P}_LEN-pd0.1_LINEARFREQ_FROMSTART_WARM_${E}_ep0070.pt)" "$(waehle_warm WARM linear)"

echo "--- 5  Endstand: Folgeglied darf NICHT neu beginnen ---"
touch "${P}_LEN-pd0.1_LINEARFREQ_LR1E4_${E}_final.pt"
rm -f "${P}_LEN-pd0.1_LINEARFREQ_LR1E4_${E}_ep0060.pt"      # so raeumt die Rotation auf
pruefe "Kettenpunkt meldet fertig" "FERTIG" "$(waehle_hp LR1E4 linear)"
touch "${P}_LEN-pd0.1_LINEARFREQ_FROMSTART_WARM_${E}_final.pt"
rm -f "${P}_LEN-pd0.1_LINEARFREQ_FROMSTART_WARM_${E}_ep0070.pt"
pruefe "Warmstart meldet fertig"   "FERTIG" "$(waehle_warm WARM linear)"

echo "--- 6  anderer Kettenpunkt bleibt unberuehrt ---"
pruefe "LR3E4 faengt an"  "NEU"  "$(waehle_hp LR3E4 linear)"

cd /; rm -rf "$T"
echo
[ "$fehler" -eq 0 ] && echo "Alle Pruefungen bestanden." || { echo "$fehler Fehler"; exit 1; }
