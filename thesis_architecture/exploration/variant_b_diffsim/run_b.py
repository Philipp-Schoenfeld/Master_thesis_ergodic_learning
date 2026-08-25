#!/usr/bin/env python3
r"""
Variante B — differenzierbare Simulation statt Belohnung
========================================================
Die einzige Variante, in der die Bahn dafuer optimiert wird, was sie *lernen*
wird, statt dafuer, eine feste Zieldichte zu treffen.

Der naheliegende Entwurf hierfuer waere Reinforcement Learning: Trajektorie
ausrollen, Belohnung aus aufgedeckter Unsicherheit und Abdeckungsguete bilden,
zurueckrechnen. Das ist unnoetig und schaedlich. Beide Belohnungsgroessen sind
terminale Skalare ueber eine ganze Trajektorie, und sie auf 25 Kontrollpunkte
zurueckzurechnen ist ein Kreditzuweisungsproblem mit sehr hoher Varianz.

Es geht exakt, und der Grund ist eine Eigenschaft des Gauss-Prozesses:

    **Die Posterior-Varianz haengt nur von den Messorten ab, nicht von den
    gemessenen Werten.**

Man kann also ausrechnen, wie viel Unsicherheit eine geplante Bahn aufloesen
wird, *bevor* man sie abfaehrt — ohne die Messung zu simulieren und ohne eine
Annahme ueber ihren Ausgang. `GPBelief.uncertainty_after` tut genau das, ist
differenzierbar in den Messorten, und die Messorte sind die Bahn.

Damit wird aus einem verrauschten RL-Problem ein deterministisches:

    min_cps   U(cps)  +  lambda_cov * E_erg(cps, Phi)

U ist die erwartete Restunsicherheit nach dem Abfahren, E_erg der gewohnte
ergodische Fehler gegen die aktuelle Glaubensdichte. Der erste Term zieht die
Bahn dorthin, wo sie etwas lernt; der zweite dorthin, wo vermutlich etwas ist.
Beide Gradienten sind exakt.

    python run_b.py --shapes 3 --lambda_cov 50
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..')))

from common.acquisition import ucb_density, particles_from_density   # noqa: E402
from common.data import load_truth, initial_belief                   # noqa: E402
from common.observation import measure, thin                         # noqa: E402
from common.planner import GradientPlanner                           # noqa: E402
from common import metrics                                           # noqa: E402


def _probe_points(curve, n_probe, sensor_radius=0.0, n_ring=4):
    """Die Orte, an denen die Vorausschau rechnet.

    Muss denselben Messprozess abbilden wie `observation.measure`, sonst
    optimiert die Variante gegen ein Versprechen, das nicht haelt. Ohne den
    Sensorring war die Vorausschau in einem ersten Lauf um rund 40 % zu
    pessimistisch — sie sah nur die Bahnpunkte, gemessen wurde aber eine
    Umgebung um jeden davon.
    """
    idx = torch.linspace(0, curve.shape[0] - 1, n_probe).long()
    pts = curve[idx]
    if sensor_radius > 0 and n_ring > 0:
        ang = torch.arange(n_ring, device=curve.device, dtype=curve.dtype)
        ang = ang * (2 * torch.pi / n_ring)
        off = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        pts = torch.cat([pts, (pts.unsqueeze(1) + sensor_radius *
                               off.unsqueeze(0)).reshape(-1, 2)], dim=0)
    return pts.clamp(0.0, 1.0)


def plan_differentiable(belief, particles, planner, n_probe=40,
                        lambda_cov=50.0, lambda_unc=1.0, steps=150, lr=0.05,
                        seed=0, init=None, sensor_radius=0.0):
    """Optimiert Kontrollpunkte gegen erwartete Restunsicherheit + Abdeckung.

    `n_probe` bestimmt, an wie vielen Stellen der Bahn die Unsicherheits-
    vorausschau ausgewertet wird. Der GP kostet O(n^3) in der Zahl der
    Messpunkte, und die Vorausschau steht in der inneren Schleife — 40 Punkte
    sind der Kompromiss zwischen einer treuen Abbildung der Bahn und einer
    Optimierung, die in Sekunden statt Minuten laeuft.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    cps = (init.clone() if init is not None
           else (0.3 * torch.randn(1, planner.nxi, 2, generator=g) + 0.5))
    cps = cps.clamp(0.02, 0.98).requires_grad_(True)
    parts = particles.unsqueeze(0)

    opt = torch.optim.Adam([cps], lr=lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        curve = torch.einsum('pi,kid->kpd', planner.B, cps)[0]

        probe = _probe_points(curve, n_probe, sensor_radius)
        unc = belief.uncertainty_after(probe)
        cov = planner.erg.coverage_error(cps, parts).sum()
        reg = planner._regularisers(cps).sum()

        (lambda_unc * unc + lambda_cov * cov + reg).backward()
        opt.step()

    return cps.detach().clamp(0.0, 1.0), time.perf_counter() - t0


def run_mission(truth, planner, lambda_cov=50.0, lambda_unc=1.0, kappa=1.0,
                n_particles=192, n_prior=12, grid_res=48, noise=0.02,
                sensor_radius=0.03, steps=150, n_probe=40, seed=0, label='B'):
    b0 = initial_belief(truth, n_prior=n_prior, grid_res=grid_res,
                        noise=noise, seed=seed)
    mu, sd = b0.posterior_grid()
    parts = particles_from_density(ucb_density(mu, sd, kappa=kappa), n_particles)

    cps, wall = plan_differentiable(b0, parts, planner, n_probe=n_probe,
                                    lambda_cov=lambda_cov,
                                    lambda_unc=lambda_unc, steps=steps,
                                    seed=seed, sensor_radius=sensor_radius)
    curve = planner.render(cps)[0]

    b1 = b0.clone()
    pts, vals = measure(curve, truth, noise_std=noise,
                        sensor_radius=sensor_radius)
    b1.observe(*thin(pts, vals))

    # Was die Vorausschau versprochen hat, gegen das, was eingetreten ist.
    with torch.no_grad():
        predicted = float(b0.uncertainty_after(
            _probe_points(curve, n_probe, sensor_radius)))
    actual = float(b1.total_uncertainty())

    return curve, metrics.summarise(
        label, curve, truth, b0, b1,
        extra={'plan_s': round(wall, 3), 'unc_pred': predicted,
               'unc_real': actual, 'lam_cov': lambda_cov})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=3)
    p.add_argument('--lambda_cov', type=float, nargs='+', default=[0.0, 50.0],
                   help='0 = reine Erkundung durch Unsicherheitsabbau.')
    p.add_argument('--lambda_unc', type=float, default=1.0)
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--grid_res', type=int, default=32,
                   help='Kleiner als bei A-E: das Gitter steht in der inneren '
                        'Schleife der Vorausschau.')
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--n_probe', type=int, default=40)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print("\n=== Variante B: differenzierbare Vorausschau statt Belohnung ===")
    print(f"  Formen: {', '.join(names)}\n")

    rows = []
    for lc in a.lambda_cov:
        per = []
        for i in range(len(names)):
            pl = GradientPlanner(metric=a.metric, steps=1, seed=i)
            _, r = run_mission(T[i], pl, lambda_cov=lc, lambda_unc=a.lambda_unc,
                               kappa=a.kappa, n_prior=a.n_prior,
                               grid_res=a.grid_res, n_particles=a.n_particles,
                               steps=a.steps, n_probe=a.n_probe, seed=i,
                               label=f"B lam={lc:g}")
            per.append(r)
        mean = {k: (sum(x[k] for x in per) / len(per)
                    if isinstance(per[0][k], (int, float)) else per[0][k])
                for k in per[0]}
        mean['variant'] = f"B  lambda_cov={lc:g}"
        rows.append(mean)

    metrics.print_table(rows, f"Mittel ueber {len(names)} Formen")
    print("\n  unc_pred ist die Vorausschau vor dem Abfahren, unc_real das")
    print("  Ergebnis danach. Beide muessen nahe beieinander liegen — sonst")
    print("  optimiert die Variante gegen ein Versprechen, das nicht haelt.")
    gaps = [abs(r['unc_pred'] - r['unc_real']) / max(r['unc_real'], 1e-9)
            for r in rows]
    print(f"  Groesste Abweichung Vorausschau/Wirklichkeit: {max(gaps):.1%}")
    print("\n  Zur Gewichtung: der Unsicherheitsterm liegt in der Groessen-")
    print("  ordnung mehrerer hundert, der ergodische Fehler bei 1e-2. Ohne")
    print("  ein lambda_cov in der Groessenordnung 1e4 ist die Abdeckung im")
    print("  Gesamtziel praktisch nicht vertreten.\n")


if __name__ == '__main__':
    main()
