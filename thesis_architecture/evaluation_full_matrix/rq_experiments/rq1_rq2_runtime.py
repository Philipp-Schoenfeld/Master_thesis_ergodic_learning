r"""
rq1_rq2_runtime.py
===================
RQ1 (Laufzeit bis zum Ziel-Niveau, spiegelt Sun et al. Fig. 4/13) und RQ2
(Qualitaet bei festem Zeitbudget, spiegelt Fig. 5) teilen sich EIN Experiment:
pro Trial (Suns Randomisierungsprotokoll -- zufaelliges trimodales GMM +
Startpunkt, siehe `exp_common.make_trial`) wird jede Methode an einer
aufsteigenden Folge von Iterationsbudgets ausgewertet und dabei sowohl die
Wall-Clock-Zeit als auch der ergodische Fehler E (K=8-Fourier, siehe
`exp_common.ergodic_error`) aufgezeichnet. Aus (iters, time_s, E) laesst sich
danach sowohl "Zeit bis E<=eps" (RQ1) als auch "E bei Zeit t" (RQ2) ablesen
-- zwei Fragen an dieselbe Kurve, kein zweiter Lauf noetig.

Methoden:
    sun_cold        Suns eigene Initialisierung (u=0), FM-Stein
    sun_heuristic   unsere Heuristik-Init (TSP+Lissajous+PID), FM-Stein
    cfm_raw         ein CFM-Vorwaertsdurchlauf, keine Verfeinerung (iters=0)
    cfm_fmstein     CFM-Bahn als Warm-Start fuer FM-Stein (dieselbe Schleife
                    wie sun_cold/sun_heuristic, nur von der CFM-Bahn aus)
    cfm_svgd        CFM-Bahn mit `SvgdRefiner` verfeinert (Standard seit
                    2026-10-06: Suns FM-Stein-Loeser auf dem Dichtegitter)

Resumable je Trial: eine Zeile pro (trial, method, iters); ein Trial gilt als
fertig, sobald alle Methoden x Checkpoints vorhanden sind, und wird beim
erneuten Start uebersprungen (`results/<out_tag>/rq1_rq2_runtime.csv`).

Usage:
    python rq1_rq2_runtime.py --n_trials 3 --checkpoints 0,10,25,50,100 \
        --out_tag smoke --no_cfm        # Sun-only Smoke-Test, Sekunden
    python rq1_rq2_runtime.py --n_trials 100 --out_tag rq1_rq2_full
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
for _p in (_here, _mat):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp_common as C                                             # noqa: E402

DEFAULT_CHECKPOINTS = [0, 10, 25, 50, 100, 200, 400, 600]
#: dt/tsteps wie bei Sun und in der Datengenerierung.
DT, TSTEPS = 0.05, 200
METHODS = ['sun_cold', 'sun_heuristic', 'cfm_raw', 'cfm_fmstein', 'cfm_svgd']
FIELDS = ['trial', 'seed', 'method', 'iters', 'time_s', 'E']


def _load_done(csv_path):
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['trial']), row['method']))
    return done


def run_trial(trial, checkpoints, planner, device, include_cfm):
    """-> Liste von Zeilen (dicts nach FIELDS, ohne 'trial'/'seed')."""
    phi = C.density_grid(trial['pdf_fn'], res=96)
    phik = C.target_coeffs(phi)
    rows = []

    x0j, u0 = C.sun_cold_x0_u0(trial['x0'], DT, TSTEPS)
    for r in C.timed_sun_solve(trial['score_fn'], x0j, u0, checkpoints):
        rows.append(dict(method='sun_cold', iters=r['iters'], time_s=r['time_s'],
                         E=C.ergodic_error(r['traj_xy'], phik)))

    x0j, u0 = C.heuristic_x0_u0(trial['x0'], trial['shape_def'], DT, TSTEPS)
    for r in C.timed_sun_solve(trial['score_fn'], x0j, u0, checkpoints):
        rows.append(dict(method='sun_heuristic', iters=r['iters'], time_s=r['time_s'],
                         E=C.ergodic_error(r['traj_xy'], phik)))

    if not include_cfm:
        return rows

    curve, t_cfm = C.cfm_raw_curve(planner, phi, start=trial['x0'], device=device)
    rows.append(dict(method='cfm_raw', iters=0, time_s=t_cfm,
                     E=C.ergodic_error(curve, phik)))

    x0j, u0 = C.cfm_x0_u0(curve, trial['x0'], DT, TSTEPS)
    for r in C.timed_sun_solve(trial['score_fn'], x0j, u0, checkpoints):
        rows.append(dict(method='cfm_fmstein', iters=r['iters'], time_s=t_cfm + r['time_s'],
                         E=C.ergodic_error(r['traj_xy'], phik)))

    for r in C.cfm_svgd_timed(curve, phi, trial['x0'], checkpoints, nxi=planner.nxi):
        rows.append(dict(method='cfm_svgd', iters=r['iters'], time_s=t_cfm + r['time_s'],
                         E=C.ergodic_error(r['traj_xy'], phik)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n_trials', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=0,
                    help='Trial i benutzt Seed seed0+i (deterministisch, resumable).')
    ap.add_argument('--checkpoints', type=str, default=','.join(map(str, DEFAULT_CHECKPOINTS)))
    ap.add_argument('--no_cfm', action='store_true',
                    help='Nur sun_cold/sun_heuristic (kein Checkpoint, kein Torch noetig).')
    ap.add_argument('--ckpt', type=str, default=None,
                    help='Start-konditionierter CFM-Checkpoint (Default: transfer/netz2d_startpunkt.pt).')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop starting new trials after this many hours; already-written '
                         'rows stay valid, re-run with the same --out_tag to resume.')
    args = ap.parse_args()

    checkpoints = sorted(set(int(c) for c in args.checkpoints.split(',') if c != ''))
    out_dir = os.path.join(_mat, 'results', args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'rq1_rq2_runtime.csv')

    planner, device = None, None
    if not args.no_cfm:
        import torch
        from run_eval_matrix import DEFAULT_CKPT
        from run_ideal_matrix import build_particle_planner
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        planner = build_particle_planner(args.ckpt or DEFAULT_CKPT, device)
        # Warmup: der erste Vorwaertsdurchlauf zahlt CUDA-/Graph-Initialisierung,
        # die nicht Teil der gemessenen CFM-Zeit sein soll (Fairness-Checkliste
        # der Thesis-Doc).
        warm_phi = C.density_grid(C.make_trial(seed=2 ** 31 - 1)['pdf_fn'], res=96)
        C.cfm_raw_curve(planner, warm_phi, start=(0.5, 0.5), device=device)
        print(f"[rq1_rq2] CFM-Planer bereit ({device}), warmgelaufen", flush=True)

    done = _load_done(out_csv)
    need_header = not os.path.isfile(out_csv)
    t0 = time.time()
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            w.writeheader()
        for i in range(args.n_trials):
            if C.time_budget_exceeded(t0, args.time_budget_h):
                print(f"[rq1_rq2] time budget reached before trial {i}, stopping.", flush=True)
                break
            seed = args.seed0 + i
            methods_here = ['sun_cold', 'sun_heuristic'] if args.no_cfm else METHODS
            if all((i, m) in done for m in methods_here):
                continue
            trial = C.make_trial(seed=seed)
            rows = run_trial(trial, checkpoints, planner, device, include_cfm=not args.no_cfm)
            for r in rows:
                r['trial'], r['seed'] = i, seed
                w.writerow(r)
            f.flush()
            print(f"[rq1_rq2] trial {i + 1}/{args.n_trials} done, "
                 f"{time.time() - t0:.0f}s seit Start", flush=True)
    print(f"[rq1_rq2] fertig -> {out_csv}")


if __name__ == '__main__':
    main()
