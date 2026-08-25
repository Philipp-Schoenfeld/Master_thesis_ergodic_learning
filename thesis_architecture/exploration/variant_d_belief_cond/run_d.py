#!/usr/bin/env python3
r"""
Variante D — auf den Glauben konditionieren, lang planen, kurz fahren
=====================================================================
Klassisches MPC-Muster, auf die Explorationsfrage angewandt, und die Variante,
die dem bestehenden Netz am naechsten liegt.

In jeder Runde wird eine *vollstaendige* Trajektorie fuer die aktuelle
Zieldichte erzeugt, aber nur ihr erster Bruchteil abgefahren. Danach wird der
Glaube fortgeschrieben und komplett neu generiert.

Unterschied zu Variante C, der leicht untergeht: C plant eine *Fortsetzung* der
bisherigen Bahn und braucht dafuer die Historie in der Abdeckungsrechnung. D
plant jedes Mal von vorn und verzichtet auf dieses Gedaechtnis — die Information
ueber das bereits Gesehene steckt stattdessen vollstaendig im Glauben, weil
besuchte Gebiete dort keine Unsicherheit mehr tragen und aus der Zieldichte
verschwinden. Das ist die einfachere Buchfuehrung, um den Preis, dass Abdeckung
von bereits *bekannter* Dichte nicht erinnert wird.

Warum das die architektonisch interessanteste Variante ist: der Eingang des
Netzes aendert sich **gar nicht**. Es konditioniert ohnehin auf Partikelwolken,
und ob die aus einer bekannten Dichte oder aus einem UCB-Glauben gezogen wurden,
sieht es nicht. Amortisierte Inferenz ueber Glaubenszustaende ist damit ein
Wechsel der Eingabedaten, keine Architekturaenderung.

    python run_d.py --shapes 3 --rounds 4 --execute_frac 0.35
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..')))

from common.acquisition import (ucb_density, particles_from_density,  # noqa: E402
                                kappa_schedule)
from common.data import load_truth, initial_belief                    # noqa: E402
from common.observation import measure, thin                          # noqa: E402
from common.planner import GradientPlanner, ModelPlanner              # noqa: E402
from common import metrics                                            # noqa: E402


def run_mission(truth, make_planner, rounds=4, execute_frac=0.35,
                n_particles=192, n_prior=12, grid_res=48, noise=0.02,
                sensor_radius=0.03, kappa0=4.0, kappa1=0.5, seed=0, label='D'):
    b = initial_belief(truth, n_prior=n_prior, grid_res=grid_res,
                       noise=noise, seed=seed)
    b0 = b.clone()

    driven, start, total_plan_s = [], None, 0.0

    for r in range(rounds):
        mu, sd = b.posterior_grid()
        k = kappa_schedule(r, rounds, kappa0, kappa1)
        parts = particles_from_density(ucb_density(mu, sd, kappa=k), n_particles)

        planner = make_planner(seed * 100 + r)
        cps = planner.plan(parts, 1, start=start)
        full = planner.render(cps)[0]
        total_plan_s += planner.last_wallclock

        # Nur den vorderen Teil abfahren: der Rest wird ohnehin verworfen,
        # sobald der Glaube sich geaendert hat.
        n_exec = max(2, int(round(execute_frac * full.shape[0])))
        seg = full[:n_exec]

        pts, vals = measure(seg, truth, noise_std=noise,
                            sensor_radius=sensor_radius)
        b.observe(*thin(pts, vals, max_points=48))

        driven.append(seg)
        start = seg[-1].detach()

    path = torch.cat(driven, dim=0)
    return path, metrics.summarise(
        label, path, truth, b0, b,
        extra={'plan_s': round(total_plan_s, 3), 'rounds': rounds,
               'exec_frac': execute_frac})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=3)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--execute_frac', type=float, default=0.35)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--kappa0', type=float, default=4.0)
    p.add_argument('--kappa1', type=float, default=0.5)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--checkpoint', default=None)
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print("\n=== Variante D: glaubenskonditioniert, lang planen kurz fahren ===")
    print(f"  Formen: {', '.join(names)}  |  {a.rounds} Runden, "
          f"{a.execute_frac:.0%} je Runde abgefahren\n")

    def mk(seed):
        return (ModelPlanner(a.checkpoint) if a.checkpoint
                else GradientPlanner(metric=a.metric, steps=a.steps, seed=seed))

    rows = [run_mission(T[i], mk, rounds=a.rounds, execute_frac=a.execute_frac,
                        n_prior=a.n_prior, grid_res=a.grid_res,
                        n_particles=a.n_particles, kappa0=a.kappa0,
                        kappa1=a.kappa1, seed=i, label=f"D {names[i][:12]}")[1]
            for i in range(len(names))]
    mean = {k: (sum(r[k] for r in rows) / len(rows)
                if isinstance(rows[0][k], (int, float)) else rows[0][k])
            for k in rows[0]}
    mean['variant'] = 'D  Mittel'
    metrics.print_table(rows + [mean])
    print("\n  Der Eingang des Netzes bleibt unveraendert — nur die Wolke,")
    print("  aus der konditioniert wird, kommt jetzt aus dem Glauben.\n")


if __name__ == '__main__':
    main()
