r"""
Gemeinsamer Unterbau der Explorationsvarianten A-E.

Der Import richtet zugleich `sys.path` ein. Die Varianten liegen zwei Ebenen
unter `thesis_architecture` und brauchen von dort `ergodic_metric`, `obstacles`
und den Formen-Generator, ausserdem `bsplinax` aus dem Projektwurzelverzeichnis.
Das hier an einer Stelle zu erledigen ist weniger fehleranfaellig, als es in
jedem der fuenf Runner zu wiederholen.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.normpath(os.path.join(_here, '..', '..'))
_root = os.path.normpath(os.path.join(_arch, '..'))

for _p in (_arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'bsplinax-main'),
           os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
