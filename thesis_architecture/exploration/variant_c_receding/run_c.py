#!/usr/bin/env python3
r"""
Variante C — Receding Horizon mit Abdeckungsgedaechtnis
=======================================================
Die methodisch treueste Variante, und die, in der sich Amortisierung auszahlt.

Die Mission wird in Abschnitte geteilt. Vor jedem Abschnitt wird der Glaube
fortgeschrieben, die Zieldichte neu berechnet und nur ein kurzes Stueck geplant.
Damit verschwindet der Konstruktionsfehler von Variante A: die Unsicherheit ist
nicht mehr zum Planungszeitpunkt eingefroren, sondern faellt dort, wo der
Roboter gerade war, bevor der naechste Abschnitt geplant wird.

Zwei Details entscheiden, ob das funktioniert:

1. **Die bereits abgefahrene Bahn geht in die Abdeckung ein.**
   `GradientPlanner.plan(history=...)` setzt die Historie vor den neuen
   Abschnitt, damit spaetere Abschnitte Luecken schliessen statt Bekanntes zu
   wiederholen.

   Die Ablation (`--ablate_history`) zeigt das allerdings **nicht** so einfach,
   wie man erwarten wuerde: ohne Gedaechtnis faellt die Abdeckung besser aus,
   bei rund 40 % laengerem Pfad. Der Grund ist naheliegend, sobald man ihn
   sieht — ohne Historie versucht *jeder* Abschnitt die ganze Verteilung zu
   treffen und faehrt entsprechend weit aus. Der Vergleich ist damit durch die
   freie Pfadlaenge verfaelscht und sagt ueber das Gedaechtnis fuer sich
   genommen nichts.

   Wer das sauber messen will, muss die Pfadlaenge festhalten, etwa ueber eine
   Laengenstrafe oder eine Geschwindigkeitsgrenze pro Abschnitt. Solange das
   nicht geschieht, ist `hist=0` gegen `hist=1` keine Aussage, sondern eine
   offene Frage — die Spalte `path_len` steht in der Tabelle, damit das
   sichtbar bleibt.
2. **kappa faellt ueber die Mission.** Frueh erkunden, spaet abdecken — ein
   fester Wert kann das nicht ausdruecken. Erst mehrere Planungsschritte machen
   den Zeitplan ueberhaupt nutzbar.

Kostenseite: `n_segments` Planungen pro Mission statt einer. Mit dem
Gradientenplaner ist das `n_segments` mal die volle Optimierung — genau die
Rechnung, die ein amortisierter Generator um Groessenordnungen unterbietet.
Die Spalte `plan_s` ist deshalb hier die interessanteste der Tabelle.

    python run_c.py --shapes 3 --segments 4
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


def run_mission(truth, make_planner, n_segments=4, seg_nxi=10, n_particles=192,
                n_prior=12, grid_res=48, noise=0.02, sensor_radius=0.03,
                kappa0=4.0, kappa1=0.5, seed=0, label='C', use_history=True):
    b = initial_belief(truth, n_prior=n_prior, grid_res=grid_res,
                       noise=noise, seed=seed)
    b0 = b.clone()

    history = torch.zeros(0, 2)
    start = None
    total_plan_s = 0.0

    for s in range(n_segments):
        mu, sd = b.posterior_grid()
        k = kappa_schedule(s, n_segments, kappa0, kappa1)
        parts = particles_from_density(ucb_density(mu, sd, kappa=k), n_particles)

        planner = make_planner(seed * 100 + s, seg_nxi)
        cps = planner.plan(parts, 1,
                           history=history if use_history else None,
                           start=start)
        seg = planner.render(cps)[0]
        total_plan_s += planner.last_wallclock

        pts, vals = measure(seg, truth, noise_std=noise,
                            sensor_radius=sensor_radius)
        b.observe(*thin(pts, vals, max_points=48))

        history = torch.cat([history, seg], dim=0)
        start = seg[-1].detach()

    return history, metrics.summarise(
        label, history, truth, b0, b,
        extra={'plan_s': round(total_plan_s, 3), 'segs': n_segments,
               'hist': int(use_history)})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=3)
    p.add_argument('--segments', type=int, default=4)
    p.add_argument('--seg_nxi', type=int, default=10,
                   help='Kontrollpunkte pro Abschnitt.')
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--kappa0', type=float, default=4.0)
    p.add_argument('--kappa1', type=float, default=0.5)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--ablate_history', action='store_true',
                   help='Ohne Abdeckungsgedaechtnis rechnen — zeigt, wieso es '
                        'gebraucht wird.')
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print("\n=== Variante C: Receding Horizon, Glaube waechst mit ===")
    print(f"  Formen: {', '.join(names)}  |  {a.segments} Abschnitte, "
          f"kappa {a.kappa0} -> {a.kappa1}\n")

    def mk(seed, nxi):
        if a.checkpoint:
            return ModelPlanner(a.checkpoint, nxi=nxi)
        return GradientPlanner(nxi=nxi, metric=a.metric, steps=a.steps, seed=seed)

    variants = [True, False] if a.ablate_history else [True]
    rows = []
    for hist in variants:
        per = [run_mission(T[i], mk, n_segments=a.segments, seg_nxi=a.seg_nxi,
                           n_prior=a.n_prior, grid_res=a.grid_res,
                           n_particles=a.n_particles, kappa0=a.kappa0,
                           kappa1=a.kappa1, seed=i, use_history=hist,
                           label=f"C {names[i][:12]}")[1]
               for i in range(len(names))]
        mean = {k: (sum(r[k] for r in per) / len(per)
                    if isinstance(per[0][k], (int, float)) else per[0][k])
                for k in per[0]}
        mean['variant'] = f"C  Mittel ({'mit' if hist else 'ohne'} Gedaechtnis)"
        rows += per + [mean]

    metrics.print_table(rows)
    print("\n  plan_s summiert alle Abschnitte — das ist die Groesse,")
    print("  gegen die ein amortisierter Generator antritt.")
    if a.ablate_history:
        print("  [!] Die Gedaechtnis-Ablation ist NICHT laengenkontrolliert:")
        print("      ohne Historie faehrt jeder Abschnitt weiter aus. Erst mit")
        print("      fester Pfadlaenge wird daraus ein gueltiger Vergleich.")
    print()


if __name__ == '__main__':
    main()
