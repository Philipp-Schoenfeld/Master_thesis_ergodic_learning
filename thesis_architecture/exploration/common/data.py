r"""
data.py
=======
Wahre Dichtefelder aus der bestehenden Trainingsdatenbank, plus die Konstruktion
eines Ausgangsglaubens.

Die Datenbank kennt keine partielle Beobachtbarkeit — sie enthaelt die fertige
Dichte. Der Ausgangsglaube wird deshalb hier erzeugt, und die Wahl, *wie*, ist
eine inhaltliche: `n_prior` zufaellige Vorabmessungen entsprechen einem Roboter,
der ein paar Stichproben genommen hat, aber nichts ueber die Struktur weiss.
Mit `n_prior=0` startet er vollstaendig blind, was den Explorationsdruck
maximiert und den Unterschied zwischen den Varianten am deutlichsten zeigt.
"""

import json
import os
import sqlite3
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.normpath(os.path.join(_here, '..', '..'))
_gen = os.path.join(_arch, 'ergodic_dataset_generator')
for _p in (_arch, _gen):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_DB = os.path.join(_gen, 'ergodic_dataset_775.db')


def load_truth(labels=None, n=6, split='val', resolution=48, db_path=DEFAULT_DB,
               device='cpu'):
    """Wahre Dichtefelder. -> (Namen, Tensor (n, R, R) auf [0,1] normiert)."""
    from shape_library import pdf_on_grid

    conn = sqlite3.connect(db_path)
    q = ("SELECT shape_name, density_params FROM ergodic_pairs "
         "WHERE split=? ORDER BY id ASC")
    rows = conn.execute(q, (split,)).fetchall()
    conn.close()

    picked, seen = [], set()
    for name, params in rows:
        if labels is not None and name not in labels:
            continue
        if name in seen:
            continue
        seen.add(name)
        picked.append((name, params))
        if labels is None and len(picked) >= n:
            break

    names, grids = [], []
    for name, params in picked:
        d, _, _ = pdf_on_grid(json.loads(params), resolution=resolution)
        d = np.asarray(d, dtype=np.float32)
        grids.append(d / max(d.max(), 1e-12))
        names.append(name)
    return names, torch.tensor(np.stack(grids), device=device)


def initial_belief(truth, n_prior=0, grid_res=48, lengthscale=0.08,
                   noise=1e-2, seed=0, device='cpu'):
    """Ausgangsglaube, optional mit einigen zufaelligen Vorabmessungen."""
    from .belief import GPBelief
    from .observation import measure

    b = GPBelief(grid_res=grid_res, lengthscale=lengthscale, noise=noise,
                 device=device)
    if n_prior > 0:
        g = torch.Generator(device='cpu').manual_seed(seed)
        pts = torch.rand(n_prior, 2, generator=g).to(device)
        p, v = measure(pts, truth, noise_std=noise, generator=None)
        b.observe(p, v)
    return b
