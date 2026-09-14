r"""
evaluation_full_matrix
=======================
Grosse Holdout-Auswertung: fuer jede der 24 Validierungsformen, fuer vier
Wissensstufen und ueber alle Initialisierungs-/Replanning-/SVGD-Varianten
werden Trajektorien erzeugt und gegen die wahre Dichte bewertet.

Baut ausschliesslich auf dem bestehenden `exploration/`-Unterbau auf
(`common.belief`, `common.acquisition`, `common.svgd_refine`,
`common.baselines`, `common.metrics`, `apply_cfm_belief.CfmPlanner`) statt
ihn zu duplizieren — siehe `exploration/README.md` fuer die Varianten A-E,
von denen hier A (`no_replan`) und D (`replan_1_6`, ueber `LENGTH_UNIT`)
wiederverwendet werden.

Der Import richtet `sys.path` ein, dieselbe Loesung wie
`exploration/__init__.py` bzw. `exploration_optimierung/__init__.py`.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)                        # thesis_architecture/
_root = os.path.dirname(_arch)                         # Projektwurzel
_expl = os.path.join(_arch, 'exploration')

for _p in (_expl, _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'),
           os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

ARCH_DIR = _arch
ROOT_DIR = _root
EXPL_DIR = _expl
DEFAULT_CKPT = os.path.join(_root, 'transfer', 'netz2d_startpunkt.pt')
RESULTS_DIR = os.path.join(_here, 'results')
