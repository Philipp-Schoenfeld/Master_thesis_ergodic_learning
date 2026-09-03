r"""
exploration_optimierung
=======================
Suche nach der besten Betriebseinstellung der Laengeneinheit-Mission.

Der Import richtet zugleich `sys.path` ein, damit die Module dieses Ordners
`exploration/common`, `apply_cfm_belief`, den Formen-Generator und den
SVGD-Solver in `SE3_SVGD/` finden. Das an einer Stelle zu erledigen ist
weniger fehleranfaellig, als es in jedem Runner zu wiederholen — dieselbe
Loesung wie in `exploration/common/__init__.py`.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)                       # thesis_architecture/
_root = os.path.dirname(_arch)                       # Projektwurzel
_expl = os.path.join(_arch, 'exploration')

for _p in (_expl, _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'),
           os.path.join(_root, 'bsplinax-main'),
           os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

ARCH_DIR = _arch
ROOT_DIR = _root
EXPL_DIR = _expl
DEFAULT_CKPT = os.path.join(_root, 'transfer', 'netz2d_startpunkt.pt')
RESULTS_DIR = os.path.join(_here, 'results')
