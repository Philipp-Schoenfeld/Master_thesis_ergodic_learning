r"""
evalkit
=======
Unterbau der metrischen Auswertung in `auswertung/`.

Dieses Paket implementiert **nur das Neue**: die Wissensszenarien, die
Exploration/Exploitation-Metriken, die Rekonstruktionsmetrik nach Länge, die
Abbildungen und die Fortschrittsanzeige. Alles andere wird aus dem bestehenden
Repo importiert und nicht noch einmal geschrieben:

    constraints/common.py            geführte Generierung (`guided_generate`),
                                     `ErgodicMetrics`, Kurven-Helfer, Plotstil,
                                     `write_metrics`, `summarise`
    constraints/04_path_length/      `TargetLength` (Bogenlängen-Constraint)
    exploration/common/belief.py     `GPBelief`, `MaskiertesWissen`, `muster_maske`
    exploration/common/observation.py`measure`, `thin`
    exploration/common/acquisition.py`ucb_density`, `particles_from_density`
    exploration/common/metrics.py    `coverage_vs_truth`, `information_gain`,
                                     `belief_rmse`
    model_zoo.py                     Checkpoint -> Modell + Meta
    ergodic_energy_torch.py          `coverage_distance`, `ErgodicEnergy`

Der Import richtet zugleich `sys.path` ein — an genau einer Stelle, wie in
`exploration/common/__init__.py`.

**Eine Namensfalle, die hier bewusst umschifft wird.** Es gibt zwei Dinge
namens `common`: das flache Modul `constraints/common.py` und das Paket
`exploration/common/`. Auf `sys.path` liegt deshalb `constraints/` (damit
`import common` das Constraint-Modul trifft), aber **nicht** `exploration/` —
dessen Unterbau wird als `exploration.common.…` über das Archivverzeichnis
importiert. Wer `exploration/` zusätzlich auf den Pfad legt, überschreibt je
nach Importreihenfolge das eine mit dem anderen.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
AUSW_DIR = os.path.dirname(_here)                        # auswertung/
ARCH_DIR = os.path.dirname(AUSW_DIR)                     # thesis_architecture/
ROOT_DIR = os.path.dirname(ARCH_DIR)                     # Projektwurzel
TRANSFER_DIR = os.path.join(ROOT_DIR, 'transfer')
RESULTS_DIR = os.path.join(AUSW_DIR, 'results')

for _p in (ARCH_DIR,
           os.path.join(ARCH_DIR, 'ergodic_dataset_generator'),
           os.path.join(ARCH_DIR, 'constraints'),
           os.path.join(ARCH_DIR, 'constraints', '04_path_length'),
           os.path.join(ROOT_DIR, 'bsplinax-main'),
           os.path.join(ROOT_DIR, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Titel und Metrikbeschriftungen tragen griechische Buchstaben und Pfeile; die
# Windows-Konsole faellt sonst auf cp1252 zurueck und bricht einen ganzen Lauf
# fuer ein `print` ab.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                                  # pragma: no cover
            pass

__all__ = ['AUSW_DIR', 'ARCH_DIR', 'ROOT_DIR', 'TRANSFER_DIR', 'RESULTS_DIR']
