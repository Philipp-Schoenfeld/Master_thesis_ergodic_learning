#!/usr/bin/env python3
r"""
Variante E — erst erkunden, dann abdecken (Baseline)
====================================================
Die naheliegende Zerlegung in zwei Phasen, und deshalb die Baseline, die eine
Verteidigung verlangt: erst eine Bahn, die *nur* Unsicherheit abbaut, dann eine
zweite, die *nur* die inzwischen aufgedeckte Dichte abdeckt.

Zwei getrennte Verluste nacheinander sind wohlgestellt — anders als zwei
gleichzeitig, siehe Variante A. Der Preis ist woanders:

* **Doppeltes Budget.** E faehrt zwei volle Trajektorien. Es muss A also
  deutlich schlagen, nicht nur knapp, um den Aufwand zu rechtfertigen. `path_len`
  in der Ergebnistabelle macht das sichtbar.
* **Die Erkundungsphase ist blind fuer die Aufgabe.** Sie baut Unsicherheit auch
  dort ab, wo am Ende nichts abzudecken ist. Genau diese Verschwendung soll eine
  kombinierte Dichte vermeiden.

Wenn E gegen A gewinnt, ist die kombinierte Dichte die falsche Idee. Ohne diesen
Vergleich laesst sich das nicht sagen.

    python run_e.py --shapes 4
"""

import argparse
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..')))

from common.acquisition import ucb_density, particles_from_density   # noqa: E402
from common.data import load_truth, initial_belief                   # noqa: E402
from common.observation import measure, thin                         # noqa: E402
from common.planner import GradientPlanner, ModelPlanner             # noqa: E402
from common import metrics                                           # noqa: E402


def _connect(curve1, curve2, n=16):
    """Gerades Verbindungsstueck zwischen den beiden Phasen.

    Der `GradientPlanner` kennt `start=` und kann Phase 2 dort beginnen lassen,
    wo Phase 1 endete. Das trainierte CFM-Netz kann das nicht — es hat keinen
    Start-Eingang. Diese Luecke einfach zu ueberspringen wuerde E eine Strecke
    schenken, die ein Roboter fahren muesste; sie wird deshalb eingefuegt und
    mitgemessen.
    """
    a = torch.linspace(0, 1, n, device=curve1.device).unsqueeze(-1)
    return curve1[-1].unsqueeze(0) * (1 - a) + curve2[0].unsqueeze(0) * a


def run_mission(truth, make_planner, n_particles=192, n_prior=12, grid_res=48,
                noise=0.02, sensor_radius=0.03, seed=0, label='E',
                gp_noise=None):
    b0 = initial_belief(truth, n_prior=n_prior, grid_res=grid_res,
                        noise=gp_noise if gp_noise is not None else noise,
                        seed=seed)

    # ── Phase 1: reine Erkundung (kappa -> unendlich, also nur sigma) ──────
    mu, sd = b0.posterior_grid()
    # norm='max' statt der Voreinstellung 'sum': der dritte Partikelkanal muss
    # in derselben Skala liegen wie im Training des Netzes (dort `max = 1`).
    # Fuer den Gradientenplaner ist es gleichgueltig — dessen Zielkoeffizienten
    # sind gewichtsnormiert und damit skaleninvariant.
    phi_explore = ucb_density(torch.zeros_like(mu), sd, kappa=1.0, norm='max')
    p1 = make_planner(seed)
    cps1 = p1.plan(particles_from_density(phi_explore, n_particles), 1)
    curve1 = p1.render(cps1)[0].detach().cpu()

    b_mid = b0.clone()
    pts, vals = measure(curve1, truth, noise_std=noise,
                        sensor_radius=sensor_radius)
    b_mid.observe(*thin(pts, vals))

    # ── Phase 2: reine Ausbeutung der aufgedeckten Dichte (nur mu) ────────
    mu2, sd2 = b_mid.posterior_grid()
    phi_cover = ucb_density(mu2, torch.zeros_like(sd2), kappa=0.0, norm='max')
    p2 = make_planner(seed + 1000)
    parts2 = particles_from_density(phi_cover, n_particles)
    # `start=` nur uebergeben, wenn der Planer es auch verarbeitet. Es
    # stillschweigend zu uebergeben und verschlucken zu lassen war der Fehler,
    # an dem Variante C gescheitert ist.
    takes_start = 'start' in inspect.signature(p2.plan).parameters
    cps2 = (p2.plan(parts2, 1, start=curve1[-1]) if takes_start
            else p2.plan(parts2, 1))
    curve2 = p2.render(cps2)[0].detach().cpu()

    link = (torch.zeros((0, 2), device=curve1.device) if takes_start
            else _connect(curve1, curve2))

    b1 = b_mid.clone()
    seg = torch.cat([link, curve2], dim=0)
    pts, vals = measure(seg, truth, noise_std=noise,
                        sensor_radius=sensor_radius)
    b1.observe(*thin(pts, vals))

    full = torch.cat([curve1, link, curve2], dim=0)
    return full, metrics.summarise(
        label, full, truth, b0, b1,
        extra={'plan_s': round(p1.last_wallclock + p2.last_wallclock, 3),
               'cov_ph2': float(metrics.coverage_vs_truth(curve2, truth)),
               'link': round(metrics.path_length(link) if len(link) else 0.0, 3)})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=4)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--checkpoint', default=None,
                   help='Trainiertes CFM+ErgLoss-Netz statt Gradientenabstieg. '
                        'Beide Phasen nutzen dasselbe, unveraenderte Netz — nur '
                        'die Konditionierung wechselt von sigma auf mu.')
    p.add_argument('--gp_noise', type=float, default=0.05,
                   help='Rauschterm des GP; bewusst groesser als das Messrauschen, '
                        'weil Messpunkte auf einer Bahn dicht beieinanderliegen.')
    p.add_argument('--device', default=None)
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print("\n=== Variante E: zweistufig, erst sigma dann mu (Baseline) ===")
    print(f"  Formen: {', '.join(names)}\n")

    dev = a.device or ('cuda' if torch.cuda.is_available() and a.checkpoint
                       else 'cpu')

    def mk(seed):
        return (ModelPlanner(a.checkpoint, device=dev) if a.checkpoint
                else GradientPlanner(metric=a.metric, steps=a.steps, seed=seed))

    rows = [run_mission(T[i], mk, n_prior=a.n_prior, grid_res=a.grid_res,
                        n_particles=a.n_particles, seed=i, gp_noise=a.gp_noise,
                        label=f"E {names[i][:14]}")[1]
            for i in range(len(names))]
    mean = {k: (sum(r[k] for r in rows) / len(rows)
                if isinstance(rows[0][k], (int, float)) else rows[0][k])
            for k in rows[0]}
    mean['variant'] = 'E  Mittel'
    metrics.print_table(rows + [mean])
    print("\n  path_len ist hier rund doppelt so gross wie bei A —")
    print("  E muss A deshalb deutlich schlagen, nicht nur knapp.\n")


if __name__ == '__main__':
    main()
