#!/usr/bin/env python3
r"""
Variante A — eine kombinierte Dichte, einmal planen
===================================================
Der einfachste Ansatz, der die Fragestellung ueberhaupt beantwortet, und der
Referenzpunkt fuer alle anderen.

Bekanntes und Unbekanntes werden *vor* dem ergodischen Verlust zu einer Dichte
verrechnet, Phi = mu + kappa*sigma, und diese eine Dichte wird abgedeckt. Damit
bleibt es ein wohlgestelltes Problem — eine Bahn, eine Zielverteilung.

Der naheliegende Alternativentwurf, zwei getrennte ergodische Verluste zu
addieren, ist genau das nicht: Ergodizitaet verlangt Uebereinstimmung mit *einer*
Verteilung, und zwei gleichzeitig sind im Allgemeinen unerfuellbar. Die Bahn
findet dann einen Kompromiss, der keine von beiden trifft, und das Verhaeltnis
der Gewichte entscheidet willkuerlich, welche staerker verfehlt wird.

Grenze dieser Variante: kappa ist fest, und die Unsicherheitskarte ist zum
Planungszeitpunkt eingefroren. Beim Abfahren faellt die Unsicherheit aber dort,
wo die Bahn gerade war — der Planer zaehlt also doppelt und deckt Gebiete
gruendlich ab, die eine einzige Durchquerung bereits aufgeloest haette. Genau
das behebt Variante C.

    python run_a.py --shapes 4 --kappa 2.0
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..')))

from common.acquisition import (ucb_density, particles_from_density,  # noqa: E402
                                is_degenerate)
from common.data import load_truth, initial_belief                   # noqa: E402
from common.observation import measure, thin                         # noqa: E402
from common.planner import GradientPlanner, ModelPlanner             # noqa: E402
from common import metrics                                           # noqa: E402


def run_mission(truth, planner, kappa=2.0, n_particles=192, n_prior=0,
                grid_res=48, noise=0.02, sensor_radius=0.03, seed=0,
                label='A'):
    """Eine Mission: einmal planen, abfahren, messen, bewerten."""
    b0 = initial_belief(truth, n_prior=n_prior, grid_res=grid_res,
                        noise=noise, seed=seed)
    mu, sd = b0.posterior_grid()
    phi = ucb_density(mu, sd, kappa=kappa)
    degenerate = is_degenerate(phi)
    parts = particles_from_density(phi, n_particles)

    cps = planner.plan(parts, n_candidates=1)
    curve = planner.render(cps)[0]

    b1 = b0.clone()
    pts, vals = measure(curve, truth, noise_std=noise,
                        sensor_radius=sensor_radius)
    b1.observe(*thin(pts, vals))

    return curve, metrics.summarise(
        label, curve, truth, b0, b1,
        extra={'plan_s': round(planner.last_wallclock, 3), 'kappa': kappa,
               'flat': int(degenerate)})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=4)
    p.add_argument('--kappa', type=float, nargs='+', default=[0.0, 2.0, 5.0],
                   help='0 = reine Ausbeutung, gross = reine Erkundung.')
    p.add_argument('--n_prior', type=int, default=12,
                   help='Vormessungen. Bei 0 ist der Glaube flach und Phi '
                        'unabhaengig von kappa gleichverteilt — dann ist der '
                        'kappa-Vergleich bedeutungslos.')
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--checkpoint', default=None,
                   help='Trainiertes Netz statt Gradientenplaner.')
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print(f"\n=== Variante A: kombinierte Dichte Phi = mu + kappa*sigma ===")
    print(f"  Formen: {', '.join(names)}\n")

    rows = []
    for k in a.kappa:
        agg = []
        for i, name in enumerate(names):
            planner = (ModelPlanner(a.checkpoint) if a.checkpoint
                       else GradientPlanner(metric=a.metric, steps=a.steps,
                                            seed=i))
            _, r = run_mission(T[i], planner, kappa=k, n_prior=a.n_prior,
                               grid_res=a.grid_res, n_particles=a.n_particles,
                               seed=i, label=f"A kappa={k:g}")
            agg.append(r)
        mean = {kk: (sum(x[kk] for x in agg) / len(agg)
                     if isinstance(agg[0][kk], (int, float)) else agg[0][kk])
                for kk in agg[0]}
        mean['variant'] = f"A  kappa={k:g}"
        rows.append(mean)

    metrics.print_table(rows, f"Mittel ueber {len(names)} Formen")
    print("\n  coverage: kleiner ist besser · info_gain: groesser ist besser")
    if any(r.get('flat', 0) for r in rows):
        print("  [!] flat=1: Zieldichte war gleichverteilt — kappa ohne Wirkung.")
        print("      Mit --n_prior > 0 starten, damit der Glaube Struktur hat.")
    print()


if __name__ == '__main__':
    main()
