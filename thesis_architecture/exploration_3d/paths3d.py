"""
paths3d.py
==========
Locations and import setup for the 3D active-sensing simulation.

The 3D project (``3D_ergodic_learning/``) has modules whose names collide with
the 2D ones (``obstacles``, ``ergodic_energy_torch``, ``model_zoo`` ...), so
this folder must never import anything from the 2D ``thesis_architecture``
modules.  ``setup()`` puts the 3D folder first on ``sys.path``.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)
ROOT = os.path.dirname(ARCH)
D3 = os.path.join(ROOT, '3D_ergodic_learning')

#: Projected 3D dataset (200 surfaces); holds the held-out entries and the 2D
#: density definitions of every shape.
DB = os.path.join(D3, 'ergodic_dataset_3d_no_offset_200.db')

#: SE(3) flow-matching network (position + 6D orientation, start-conditioned)
#: trained on that dataset.
DEFAULT_CKPT = os.path.join(
    ROOT, 'transfer',
    'nooffset200_flow3d_particle_ergodic_nooffset200_nxi25_D384_N512_R64_C1_flip0.0_'
    'SURF_START-pd0.1_WU50-lr2e-05_SO3_SE3-lookat_ERGLOSS-w100-K6-tp2-onfootprint_'
    'ORILOSS-w0.012-pt0.1-so0.12-cfmrot0.5_ep0300.pt')

#: Held-out splits of the dataset: new forms on new surfaces, new surfaces with
#: known forms, new forms on known surfaces.
HELDOUT_SPLITS = ('val_beides', 'val_flaeche', 'val_form')


def setup():
    """Put the 3D folder (first) and this folder on ``sys.path``."""
    for p in (HERE, D3):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
