#!/usr/bin/env python3
r"""
compare.py
==========
Der faire Vergleich der Varianten A-E gegen Orakel, Maeander und Zufallsbahn.

`run_all.py` vergleicht Endwerte und ist damit unbrauchbar fuer eine Aussage:
die Varianten erzeugen unterschiedlich lange Bahnen, und bei freier Weglaenge
misst "weiter fahren" fast immer besser. Der erste Durchlauf zeigte Variante B
vorn — mit dreimal so langem Pfad wie D. Das vergleicht Budgets, keine Verfahren.

Dieses Skript behebt drei Dinge:

**1. Festes Wegbudget.** Jede Bahn wird an wachsenden Praefixen ausgewertet, und
verglichen wird bei *gleicher* gefahrener Laenge. Wer das Budget nicht
ausschoepft, erscheint als "n/a" statt mit seinem Endwert — sonst schmuggelt
sich der Budgetunterschied wieder herein.

**2. Bezugsgroessen.** Ohne Orakel ist "Abdeckung 0.044" keine Aussage. Mit ihm
wird daraus ein Anteil des Erreichbaren, und der Maeander sagt, ob der ganze
Informationsaufwand ueberhaupt etwas eingebracht hat.

**3. Gepaarte Statistik.** Mittelwerte ueber wenige Formen sind wertlos, weil
die Streuung zwischen Formen groesser ist als zwischen Verfahren. Verglichen
wird deshalb pro Form gegen die Referenzvariante und ausgezaehlt, wie oft
gewonnen wurde.

Zwei getrennte Fragen, die man nicht in einem Lauf beantworten sollte:

    # Frage 1: welches Planungsschema ist das beste? (Planer fest)
    python compare.py --shapes 12

    # Frage 2: zahlt sich Amortisierung aus? (Schema fest, Planer wechselt)
    python compare.py --shapes 12 --only C D --checkpoint ../checkpoints/x.pt
"""

import argparse
import importlib.util
import os
import statistics as st
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from common import metrics                                    # noqa: E402
from common.baselines import lawnmower_path, oracle_path, random_path  # noqa: E402
from common.data import initial_belief, load_truth            # noqa: E402
from common.planner import GradientPlanner, ModelPlanner      # noqa: E402


def _load(d, m):
    spec = importlib.util.spec_from_file_location(m, os.path.join(_here, d, m + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=8)
    p.add_argument('--grid_res', type=int, default=32)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--budgets', type=float, nargs='+', default=[0.25, 0.5, 1.0],
                   help='Wegbudgets als Anteil der Referenzlaenge (Vorgabe: die '
                        'volle Maeander-Laenge, also "einmal das Gebiet '
                        'abfahren"). Mehrere, weil ein einzelnes Budget '
                        'immer eine Methode bevorzugt.')
    p.add_argument('--anytime_points', type=int, default=10)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--lambda_cov', type=float, default=20000.0)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--target_length', type=float, default=None,
                   help='Zielaenge fuer alle Bahnen. Ohne sie produziert jede '
                        'Variante ihre eigene Laenge und der Vergleich bei '
                        'festem Budget ist nur im gemeinsamen Bereich moeglich.')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--only', nargs='+', default=list('ABCDE'))
    p.add_argument('--reference', default='A',
                   help='Variante, gegen die gepaart ausgezaehlt wird.')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    mods = {'A': _load('variant_a_combined', 'run_a'),
            'B': _load('variant_b_diffsim', 'run_b'),
            'C': _load('variant_c_receding', 'run_c'),
            'D': _load('variant_d_belief_cond', 'run_d'),
            'E': _load('variant_e_two_stage', 'run_e')}
    titles = {'A': 'A kombinierte Dichte', 'B': 'B diff. Vorausschau',
              'C': 'C receding horizon', 'D': 'D glaubenskonditioniert',
              'E': 'E zweistufig'}

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    planner_name = ('Netz ' + os.path.basename(a.checkpoint)) if a.checkpoint else 'Gradient'
    print(f"\n=== Vergleich bei festem Wegbudget ===")
    print(f"  {len(names)} Formen · Planer: {planner_name} · Metrik: {a.metric}\n")

    # Wieviel Weg eine EINZELNE geplante Bahn haben muss, damit die MISSION
    # insgesamt auf die Ziellaenge kommt. Die Varianten unterscheiden sich
    # darin erheblich, und ohne diese Umrechnung vergleicht die Tabelle wieder
    # Budgets statt Verfahren:
    #   A, B: eine Bahn, ganz gefahren
    #   C:    n_segments Abschnitte, alle gefahren
    #   D:    n_rounds Bahnen, davon je nur execute_frac gefahren
    #   E:    zwei Bahnen, beide gefahren
    D_EXEC = 0.35
    share = {'A': 1.0, 'B': 1.0, 'C': 1.0 / a.segments,
             'D': 1.0 / (a.rounds * D_EXEC), 'E': 0.5, 'ref': 1.0}

    def mk(seed, nxi=25, who='ref'):
        if a.checkpoint:
            return ModelPlanner(a.checkpoint, nxi=nxi)
        tl = None if a.target_length is None else a.target_length * share[who]
        return GradientPlanner(nxi=nxi, metric=a.metric, steps=a.steps,
                               seed=seed, target_length=tl)

    common = dict(n_prior=a.n_prior, grid_res=a.grid_res,
                  n_particles=a.n_particles)

    # ── Bahnen erzeugen ────────────────────────────────────────────────────
    paths = {k: [] for k in list(a.only) + ['Orakel', 'Maeander', 'Zufall']}
    beliefs = []
    for i, truth in enumerate(T):
        b0 = initial_belief(truth, n_prior=a.n_prior, grid_res=a.grid_res, seed=i)
        beliefs.append(b0)

        if 'A' in a.only:
            paths['A'].append(mods['A'].run_mission(
                truth, mk(i, who='A'), kappa=a.kappa, seed=i, **common)[0])
        if 'B' in a.only:
            paths['B'].append(mods['B'].run_mission(
                truth, GradientPlanner(metric=a.metric, steps=1, seed=i,
                                       target_length=(a.target_length
                                                      and a.target_length * share['B'])),
                lambda_cov=a.lambda_cov, steps=a.steps, seed=i, **common)[0])
        if 'C' in a.only:
            paths['C'].append(mods['C'].run_mission(
                truth, lambda sd, nx: mk(sd, nx, 'C'),
                n_segments=a.segments, seed=i, **common)[0])
        if 'D' in a.only:
            paths['D'].append(mods['D'].run_mission(
                truth, lambda sd: mk(sd, who='D'), rounds=a.rounds,
                execute_frac=D_EXEC, seed=i, **common)[0])
        if 'E' in a.only:
            paths['E'].append(mods['E'].run_mission(
                truth, lambda sd: mk(sd, who='E'), seed=i, **common)[0])

        pl = mk(i)
        paths['Orakel'].append(oracle_path(truth, pl, a.n_particles, seed=i))
        paths['Maeander'].append(lawnmower_path(target_length=a.target_length))
        paths['Zufall'].append(random_path(pl, seed=i))
        print(f"  [{i+1}/{len(T)}] {names[i]}")

    # ── Anytime-Kurven ─────────────────────────────────────────────────────
    curves = {k: [metrics.anytime_curve(pp, T[i], beliefs[i],
                                        n_points=a.anytime_points)
                  for i, pp in enumerate(v)]
              for k, v in paths.items() if v}

    # Referenzlaenge: einmal das Gebiet abfahren. Ein einzelnes Budget
    # bevorzugt immer jemanden — das kleinste, das alle erreichen, laesst die
    # kuerzeste Bahn den Vergleichspunkt diktieren, und dort ist der Maeander
    # noch nicht einmal durch. Deshalb mehrere Budgets nebeneinander.
    # Referenzlaenge aus der *tatsaechlich* gefahrenen Strecke, nicht aus der
    # nominellen Ziellaenge: die Laengenstrafe trifft ihr Ziel nur naeherungs-
    # weise, und ein Budget knapp oberhalb der kuerzesten Bahn macht die halbe
    # Tabelle zu "n/a". Der Faktor 0.98 haelt Abstand zur kuerzesten Bahn.
    driven = {k: st.mean(c[-1]['path_len'] for c in cs) for k, cs in curves.items()}
    ref_len = min(driven.values()) * 0.98
    budgets = [f * ref_len for f in a.budgets]
    shortest = min(driven, key=driven.get)
    print(f"\n  Referenzlaenge: {ref_len:.2f}  (98 % der kuerzesten Mission: "
          f"{shortest} mit {driven[shortest]:.2f})")
    if a.target_length:
        print(f"  Ziellaenge war {a.target_length:.2f} — Abweichungen zeigen, "
              f"wie gut die Laengenstrafe greift.")
    print("  Tatsaechlich gefahren: " + ", ".join(
        f"{k}={v:.2f}" for k, v in driven.items()))
    print(f"  Budgets: " + ", ".join(f"{f:g}x = {b:.2f}"
                                     for f, b in zip(a.budgets, budgets)))

    order = [k for k in ['Orakel'] + list(a.only) + ['Maeander', 'Zufall']
             if k in curves]

    # ── Matrix: Abdeckung je Variante und Budget ───────────────────────────
    print(f"\n  Abdeckung der wahren Dichte (kleiner ist besser), "
          f"Mittel ueber {len(names)} Formen")
    hdr = f"  {'Variante':<26}" + "".join(f"{f:g}x".rjust(12) for f in a.budgets)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    cov_at = {}
    for k in order:
        cells, line = [], f"  {titles.get(k,k):<26}"
        for b in budgets:
            vals = [metrics.at_budget(c, b, 'coverage') for c in curves[k]]
            vals = [v for v in vals if v is not None]
            cells.append(st.mean(vals) if vals else None)
            line += (f"{cells[-1]:12.4f}" if cells[-1] is not None
                     else f"{'n/a':>12}")
        cov_at[k] = cells
        print(line)

    # ── Einordnung zwischen Maeander und Orakel ────────────────────────────
    if 'Orakel' in cov_at and 'Maeander' in cov_at:
        print(f"\n  Anteil der Spanne Maeander -> Orakel "
              f"(100 % = so gut wie mit vollem Vorwissen)")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for k in order:
            if k in ('Orakel', 'Maeander'):
                continue
            line = f"  {titles.get(k,k):<26}"
            for j in range(len(budgets)):
                o, m, v = cov_at['Orakel'][j], cov_at['Maeander'][j], cov_at[k][j]
                if None in (o, m, v) or abs(m - o) < 1e-12:
                    line += f"{'n/a':>12}"
                else:
                    line += f"{(m - v) / (m - o):11.1%} "
            print(line)

    # ── Gepaarte Auszaehlung beim groessten Budget ─────────────────────────
    ref, b = a.reference, budgets[-1]
    if ref in curves:
        print(f"\n  Gepaart gegen {titles.get(ref, ref)} bei {a.budgets[-1]:g}x "
              f"(pro Form, Abdeckung)")
        rc = [metrics.at_budget(c, b, 'coverage') for c in curves[ref]]
        for k in order:
            if k == ref:
                continue
            kc = [metrics.at_budget(c, b, 'coverage') for c in curves[k]]
            pairs = [(x, y) for x, y in zip(kc, rc)
                     if x is not None and y is not None]
            if not pairs:
                print(f"    {titles.get(k,k):<26} n/a (Budget nicht erreicht)")
                continue
            wins = sum(1 for x, y in pairs if x < y)
            d = [x - y for x, y in pairs]
            print(f"    {titles.get(k,k):<26} {wins}/{len(pairs)} besser   "
                  f"Delta {st.mean(d):+.4f}"
                  + (f" ± {st.pstdev(d):.4f}" if len(d) > 1 else ""))

    print("\n  n/a heisst: die Bahn ist kuerzer als das Budget. Das ist keine")
    print("  Schwaeche des Verfahrens, sondern macht es dort unvergleichbar —")
    print("  wer eine feste Laenge fahren soll, braucht eine Laengenstrafe im")
    print("  Planer, die derzeit in keiner Variante steckt.\n")

    if a.out:
        import csv
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['variant', 'shape', 'path_len', 'coverage',
                        'info_gain', 'belief_rmse'])
            for k, cs in curves.items():
                for i, c in enumerate(cs):
                    for pt in c:
                        w.writerow([k, names[i], pt['path_len'], pt['coverage'],
                                    pt['info_gain'], pt['belief_rmse']])
        print(f"  Anytime-Kurven geschrieben nach {a.out}\n")


if __name__ == '__main__':
    main()
