r"""
generate.py
===========
Aus einer Zieldichte eine Bahn machen — duenne Huelle um bestehenden Code.

Erzeugt wird ausschliesslich mit `constraints.common.guided_generate`. Diese
Funktion integriert die Flow-ODE mit klassifikatorfreier Fuehrung, kennt die
Rampe fuer eine Inferenz-Kraft und den anschliessenden Polish-Abstieg, und sie
nimmt seit dieser Auswertung zusaetzlich `length`/`length_cfg_weight` entgegen,
damit **derselbe** Sampler alle vier Arme fahren kann:

    frei         keine Laengenvorgabe, keine Kraft   (Referenz)
    kond         gelernter FiLM-Laengenkanal des Checkpoints
    kraft        Inferenz-Kraft `TargetLength` aus constraints/04_path_length
    kond+kraft   beides zusammen

Dass alle vier durch dieselbe Funktion laufen, ist der ganze Punkt: gleicher
Seed, gleiche Konditionierungspartikel, gleiches `cfg_weight`, gleiche
ODE-Schritte. Der einzige Unterschied ist der jeweils zugeschaltete
Mechanismus — sonst vergliche man Sampler statt Steuermechanismen.
"""

import torch

from common import (basis_torch, curve_of, density_and_particles,   # constraints/common.py
                    energy_force, guided_generate)
from length import TargetLength                                     # constraints/04_path_length
from exploration.common.acquisition import particles_from_density

ARME = ['frei', 'kond', 'kraft', 'kond+kraft']

# Voreinstellungen der Laengenkraft. Uebernommen aus
# `constraints/04_path_length/run_compare_conditioning.py`, damit die Zahlen
# dieser Auswertung an die dort gemessenen anschliessen.
KRAFT_STD = dict(force_weight=30.0, force_t_start=0.3, max_force=0.5,
                 polish_steps=400, polish_lr=0.05)


def basis(meta, pts=256, deg=5, device='cpu'):
    """B-Spline-Basismatrix (pts, nxi) fuer das Rendern der Kontrollpunkte."""
    return basis_torch(meta['nxi'], pts, deg, device=device)


def partikel_aus_dichte(phi, n_particles, device, seed=0, mode='uniform'):
    """Konditionierungswolke (N,3) aus einer max-normierten Zieldichte.

    Dieselbe Konvention wie `sample_particles` im Trainingsrunner: Orte
    gleichverteilt ueber dem Traeger, Dichtewert als drittes Merkmal. Nur so
    passt die Wolke ohne Anpassung in das trainierte Netz.
    """
    g = torch.Generator(device=str(device)).manual_seed(seed)
    return particles_from_density(phi, n_particles, device=device, mode=mode,
                                  generator=g)


def partikel_aus_form(shape_name, meta, device, grid_res=64, seed=0):
    """(d_map, particles) einer Holdout-Form — der Trainingspfad unveraendert."""
    return density_and_particles(shape_name, meta, device,
                                 grid_res=grid_res, seed=seed)


def erzeuge(modell, particles, B, *, arm='frei', ziel_laenge=None,
            length_cfg=0.0, steps=100, seed=0, device='cpu', kraft=None):
    """Eine Bahn fuer einen Arm. -> (cps (1,nxi,nd), curve (T,nd))

    Args:
        arm: einer aus `ARME`. Fuer alles ausser 'frei' wird `ziel_laenge`
            gebraucht.
        kraft: optionale Ueberschreibung der Kraftparameter (`KRAFT_STD`).
    """
    if arm not in ARME:
        raise KeyError(f"unbekannter Arm {arm!r}; bekannt: {ARME}")
    if arm != 'frei' and ziel_laenge is None:
        raise ValueError(f"Arm {arm!r} braucht eine ziel_laenge")

    will_kond = arm in ('kond', 'kond+kraft')
    will_kraft = arm in ('kraft', 'kond+kraft')

    if will_kond and not modell.length_cond:
        raise ValueError(
            f"Modell {modell.name!r} hat keine Laengenkonditionierung "
            f"(length_cond=False) — Arm {arm!r} ist dafuer nicht definiert.")

    force = None
    kw = {}
    if will_kraft:
        force = energy_force(TargetLength(target=ziel_laenge, mode='exact').energy, B)
        kw.update(KRAFT_STD)
        kw.update(kraft or {})

    cps = guided_generate(
        modell.model, modell.meta, particles, force=force, num_samples=1,
        steps=steps, device=device, seed=seed,
        length=(float(ziel_laenge) if will_kond else None),
        length_cfg_weight=(length_cfg if will_kond else 0.0), **kw)
    return cps, curve_of(cps, B)[0]
