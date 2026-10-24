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

Generalized, multi-round mode (`--n_rounds`)
---------------------------------------------
The two-phase mission above is the extreme case of a more general pattern:
drive one complete trajectory, fold its measurements into the belief, plan
another. `--n_rounds N` (N > 2) switches to exactly that: `N` full
trajectories, each planned against the same additive UCB density
`Phi = mu + kappa * sigma` as Variant C, with `kappa` annealed from
`--kappa0` (exploration-heavy) down to `--kappa1` (exploitation-heavy) across
the rounds via `kappa_schedule`. The difference to Variant C is that each
round here drives a COMPLETE trajectory rather than a short receding-horizon
segment. Without `--n_rounds`, behaviour is unchanged from the original
two-phase mission above.

    python run_e.py --shapes 4 --n_rounds 4 --kappa0 4.0 --kappa1 0.5
"""

import argparse
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..')))

from common.acquisition import (ucb_density, particles_from_density,  # noqa: E402
                                kappa_schedule)
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


def run_mission_annealed(truth, make_planner, n_rounds=4, n_particles=192,
                         n_prior=12, grid_res=48, noise=0.02,
                         sensor_radius=0.03, kappa0=4.0, kappa1=0.5,
                         seed=0, label='E-R', gp_noise=None):
    r"""Generalized Variant E: `n_rounds` complete trajectories, kappa annealed.

    Unlike the two-phase `run_mission` above (which hard-zeroes mu or sigma
    for a pure-exploration / pure-exploitation split), this runs `n_rounds`
    rounds of the same additive UCB density `Phi = mu + kappa * sigma` that
    Variant C uses, with `kappa` falling from `kappa0` to `kappa1` across the
    rounds (`kappa_schedule`) -- early rounds favour exploration, late rounds
    favour exploitation. Each round plans and drives one COMPLETE trajectory
    (not a short receding-horizon segment like Variant C), then folds the
    measurements from that whole trajectory into the belief before the next
    round is planned.

    `n_rounds=2` does not reproduce `run_mission` bit-for-bit: `run_mission`
    sets mu/sigma to exactly zero in each phase, whereas here both terms stay
    present throughout and only their mixing ratio shifts. The two-phase
    baseline is therefore kept as its own function instead of being
    special-cased here, so that difference stays visible rather than hidden
    behind a shared code path.
    """
    b = initial_belief(truth, n_prior=n_prior, grid_res=grid_res,
                       noise=gp_noise if gp_noise is not None else noise,
                       seed=seed)
    b0 = b.clone()

    segments = []
    phis = []
    total_plan_s = 0.0

    for r in range(n_rounds):
        mu, sd = b.posterior_grid()
        kap = kappa_schedule(r, n_rounds, kappa0, kappa1)
        phi = ucb_density(mu, sd, kappa=kap, norm='max')
        phis.append(phi.detach().cpu())
        parts = particles_from_density(phi, n_particles)

        planner = make_planner(seed * 100 + r)
        # Same distinction as `_connect` above: the trained net has no start
        # input and needs an explicit link segment; the gradient planner
        # takes `start=` directly and already begins the curve there.
        takes_start = 'start' in inspect.signature(planner.plan).parameters
        prev_end = segments[-1][-1] if segments else None

        if prev_end is not None and takes_start:
            cps = planner.plan(parts, 1, start=prev_end)
        else:
            cps = planner.plan(parts, 1)
        curve = planner.render(cps)[0].detach().cpu()
        total_plan_s += planner.last_wallclock

        if prev_end is not None and not takes_start:
            curve = torch.cat([_connect(segments[-1], curve), curve], dim=0)

        pts, vals = measure(curve, truth, noise_std=noise,
                            sensor_radius=sensor_radius)
        b.observe(*thin(pts, vals))
        segments.append(curve)

    full = torch.cat(segments, dim=0)
    row = metrics.summarise(
        label, full, truth, b0, b,
        extra={'plan_s': round(total_plan_s, 3), 'rounds': n_rounds,
               'kappa0': kappa0, 'kappa1': kappa1})
    # `segments` (per-round curves, in driving order) and `phis` (the belief-
    # based target density that governed each round's planning, on
    # `common.belief.GPBelief`'s grid) are returned alongside the usual
    # (curve, row) pair, so callers that want to tell rounds apart -- e.g. a
    # plot of each round against the density it was actually planned
    # against -- do not need to re-run the mission. Existing callers that
    # index `[1]` for the metrics row are unaffected.
    return full, row, segments, phis


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
    p.add_argument('--n_rounds', type=int, default=None,
                   help='Switch to the generalized multi-round mode: N complete '
                        'trajectories with kappa annealed from --kappa0 to '
                        '--kappa1 instead of the fixed two-phase mission. '
                        'Omit to keep the original two-phase behaviour.')
    p.add_argument('--kappa0', type=float, default=4.0,
                   help='Starting kappa (round 0) in --n_rounds mode -- high '
                        'means exploration-heavy.')
    p.add_argument('--kappa1', type=float, default=0.5,
                   help='Final kappa (last round) in --n_rounds mode -- low '
                        'means exploitation-heavy.')
    p.add_argument('--device', default=None)
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)

    dev = a.device or ('cuda' if torch.cuda.is_available() and a.checkpoint
                       else 'cpu')

    def mk(seed):
        return (ModelPlanner(a.checkpoint, device=dev) if a.checkpoint
                else GradientPlanner(metric=a.metric, steps=a.steps, seed=seed))

    if a.n_rounds:
        print(f"\n=== Variante E-R: {a.n_rounds} full trajectories, "
              f"kappa {a.kappa0} -> {a.kappa1} ===")
        print(f"  Formen: {', '.join(names)}\n")
        rows = [run_mission_annealed(
                    T[i], mk, n_rounds=a.n_rounds, n_prior=a.n_prior,
                    grid_res=a.grid_res, n_particles=a.n_particles,
                    kappa0=a.kappa0, kappa1=a.kappa1, seed=i,
                    gp_noise=a.gp_noise,
                    label=f"E-R{a.n_rounds} {names[i][:10]}")[1]
                for i in range(len(names))]
        mean_label = f'E-R{a.n_rounds}  Mittel'
    else:
        print("\n=== Variante E: zweistufig, erst sigma dann mu (Baseline) ===")
        print(f"  Formen: {', '.join(names)}\n")
        rows = [run_mission(T[i], mk, n_prior=a.n_prior, grid_res=a.grid_res,
                            n_particles=a.n_particles, seed=i,
                            gp_noise=a.gp_noise,
                            label=f"E {names[i][:14]}")[1]
                for i in range(len(names))]
        mean_label = 'E  Mittel'

    mean = {k: (sum(r[k] for r in rows) / len(rows)
                if isinstance(rows[0][k], (int, float)) else rows[0][k])
            for k in rows[0]}
    mean['variant'] = mean_label
    metrics.print_table(rows + [mean])
    print("\n  path_len ist hier rund doppelt so gross wie bei A —")
    print("  E muss A deshalb deutlich schlagen, nicht nur knapp.\n")


if __name__ == '__main__':
    main()
